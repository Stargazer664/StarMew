"""Durable memory outbox, owned by the single backend process.

No database transaction spans extraction. Terminal failures block later targets
of that session until an operator explicitly retries the retained job.
"""
from datetime import datetime, timedelta
from threading import Event, Lock, Thread
import logging

from sqlalchemy import exists
from sqlalchemy.orm import aliased
from .database import SessionLocal
from .models import MemoryUpdateJob, ChatSession, ChatMessage
from .memory import memory_service
from .chat_diagnostics import run_blocking, request_metrics
from . import telemetry

_claim_lock = Lock()
_BACKOFF = (5, 30, 120, 300)
logger = logging.getLogger(__name__)


def enqueue_memory_update(db, session_ref_id, target_message_count, *, request_id=None):
    """Use the answer's transaction; never commit independently."""
    job = MemoryUpdateJob(session_ref_id=session_ref_id,
        target_message_count=target_message_count, request_id=request_id)
    db.add(job)
    db.flush()
    return job


def recover_running_jobs():
    """Only the exclusive backend owner may recover jobs at startup."""
    with _claim_lock, SessionLocal() as db:
        count = db.query(MemoryUpdateJob).filter_by(status='running').update({
            'status': 'pending', 'next_attempt_at': datetime.utcnow(),
            'last_error_code': 'PROCESS_INTERRUPTED'})
        db.commit()
        return count


def claim_next_job(*, now=None):
    now = now or datetime.utcnow()
    older = aliased(MemoryUpdateJob)
    with _claim_lock, SessionLocal() as db:
        job = db.query(MemoryUpdateJob).filter(
            MemoryUpdateJob.status == 'pending', MemoryUpdateJob.next_attempt_at <= now,
            ~exists().where(older.session_ref_id == MemoryUpdateJob.session_ref_id,
                older.target_message_count < MemoryUpdateJob.target_message_count,
                older.status != 'completed'),
        ).order_by(MemoryUpdateJob.id).with_for_update().first()
        if job is None:
            return None
        # Interrupted attempts also consume the bounded retry budget. An
        # operator can retry a terminal job without repeating committed facts.
        if job.attempts >= 5:
            job.status = 'failed'
            job.last_error_code = job.last_error_code or 'MEMORY_UPDATE_FAILED'
            db.commit()
            return None
        job.status = 'running'
        job.attempts += 1
        db.commit()
        db.expunge(job)
        return job


def retry_failed_job(job_id):
    with _claim_lock, SessionLocal() as db:
        changed = db.query(MemoryUpdateJob).filter_by(id=job_id, status='failed').update({
            'status': 'pending', 'attempts': 0, 'next_attempt_at': datetime.utcnow(),
            'last_error_code': None})
        db.commit()
        return bool(changed)


def process_job(job, service=None, *, now=None):
    service = service or memory_service
    # Clear inherited chat context even for explicit retry calls made by an
    # administrative request. A job has its own trace and model call counters.
    with telemetry.bind_context(trace=None, trace_id=None, parent_span_id=None, span_id=None, _operation=None), telemetry.trace(
            'memory', 'update', request_id=job.request_id, job_id=str(job.id),
            attempt=job.attempts, session_ref_id=job.session_ref_id,
            target_message_count=job.target_message_count) as operation, request_metrics():
        try:
            with SessionLocal() as db:
                session = db.get(ChatSession, job.session_ref_id)
                if session is None:
                    raise ValueError('MEMORY_SESSION_MISSING')
                user_id, session_id = session.user_id, session.session_id
                records = db.query(ChatMessage).filter_by(session_ref_id=session.id).order_by(
                    ChatMessage.id).limit(job.target_message_count).all()
                messages = [{'type': r.message_type, 'content': r.content, 'id': r.id,
                             'timestamp': r.timestamp.isoformat() + 'Z'} for r in records]
                if len(messages) != job.target_message_count:
                    raise ValueError('MEMORY_TARGET_MISSING')
                messages = [message for message, record in zip(messages, records)
                            if record.message_type != 'ai' or record.status == 'complete']
            service.update_from_conversation(user_id, messages, session_id=session_id, session_ref_id=job.session_ref_id)
            with SessionLocal() as db:
                row = db.get(MemoryUpdateJob, job.id)
                if row is not None:
                    row.status, row.last_error_code = 'completed', None
                    db.commit()
        except Exception:
            with SessionLocal() as db:
                row = db.get(MemoryUpdateJob, job.id)
                if row is not None:
                    row.status = 'failed' if row.attempts >= 5 else 'pending'
                    row.last_error_code = 'MEMORY_UPDATE_FAILED'
                    if row.status == 'pending':
                        row.next_attempt_at = (now or datetime.utcnow()) + timedelta(seconds=_BACKOFF[row.attempts - 1])
                    db.commit()
            operation.set_result('failed')
            operation.annotate(code='MEMORY_UPDATE_FAILED')
            telemetry.event('memory_update_failed', status='failed', code='MEMORY_UPDATE_FAILED')
            return False
        return True


class MemoryWorker:
    """One managed thread; shutdown stops claiming then joins actual extraction."""
    def __init__(self, service=None, poll_interval=0.5):
        self.service = service
        self.poll_interval = poll_interval
        self._stop = Event()
        self._thread = None

    def start(self):
        recover_running_jobs()
        self._thread = Thread(target=self._run, name='memory-outbox', daemon=False)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                job = claim_next_job()
                if job is not None:
                    process_job(job, self.service)
                    continue
            except Exception:
                logger.exception('Memory worker database operation failed')
            self._stop.wait(self.poll_interval)

    async def aclose(self):
        self._stop.set()
        if self._thread is not None:
            await run_blocking(self._thread.join)
