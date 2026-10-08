"""Short database admission transactions; no lock spans inference or network I/O."""
import hashlib
import inspect
import json
import asyncio
from contextlib import aclosing
from functools import wraps
from dataclasses import dataclass
from contextvars import copy_context
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import case
from sqlalchemy.exc import IntegrityError
from .database import SessionLocal
from .chat_diagnostics import run_blocking, _await_owned_task
from .models import User, ChatSession, ChatMessage, ChatTurn, ToolExecution
from .run_control import own_run, notify_cancel, mark_cancelled, RunCancelled, bind_control, check_cancelled


class TurnConflict(Exception):
    pass


def input_digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def locked_session(db, user_id, session_id):
    user = db.query(User).filter_by(username=user_id).one()
    query = db.query(ChatSession).filter_by(user_id=user.id, session_id=session_id)
    session = query.with_for_update().first()
    if session is None:
        try:
            db.add(ChatSession(user_id=user.id, session_id=session_id, metadata_json={}))
            db.flush()
        except IntegrityError:
            # A concurrent first request won the existing unique constraint.
            # This helper runs before any other mutations in its transaction.
            db.rollback()
        session = query.with_for_update().one()
    return session


@dataclass(frozen=True)
class TurnIdentity:
    session_ref_id: int
    turn_id: int
    request_id: str
    session_instance_id: str


def admitted_session(db, user_id, session_id, identity: TurnIdentity, *, lock=False):
    """Resolve only the admitted entities; never create or find a successor."""
    query = db.query(ChatSession).join(User).filter(User.username == user_id,
        ChatSession.session_id == session_id, ChatSession.id == identity.session_ref_id,
        ChatSession.instance_id == identity.session_instance_id)
    session = (query.with_for_update(of=ChatSession) if lock else query).first()
    turn = db.query(ChatTurn).filter_by(id=identity.turn_id, session_ref_id=identity.session_ref_id,
                                      request_id=identity.request_id).first() if session else None
    if session is None or turn is None:
        raise TurnConflict('REQUEST_ENTITY_GONE')
    return session, turn


class TurnLease:
    def __init__(self, identity: TurnIdentity, replay=None):
        self.identity, self.replay = identity, replay
        self.turn_id, self.request_id = identity.turn_id, identity.request_id
        self.closed = False

    def close(self, status='failed'):
        if self.closed or self.replay is not None:
            return
        error_code = case(
            (ChatTurn.status == 'cancel_requested', 'REQUEST_CANCELLED'),
            else_='CONNECTION_INTERRUPTED',
        ) if status == 'cancelled' else 'REQUEST_' + status.upper()
        with SessionLocal() as db:
            db.query(ChatTurn).filter_by(id=self.turn_id, session_ref_id=self.identity.session_ref_id,
                                        request_id=self.request_id).filter(ChatTurn.status.in_(['running', 'cancel_requested']),
                ChatTurn.session_ref_id.in_(db.query(ChatSession.id).filter_by(id=self.identity.session_ref_id,
                    instance_id=self.identity.session_instance_id))).update(
                {'status': status, 'error_code': error_code})
            db.commit()
        self.closed = True


def admit(user_id, session_id, request_id, user_text):
    request_id = str(UUID(str(request_id))) if request_id else str(uuid4())
    with SessionLocal() as db:
        session = locked_session(db, user_id, session_id)
        existing = db.query(ChatTurn).filter_by(session_ref_id=session.id, request_id=request_id).first()
        if existing:
            if existing.input_digest != input_digest(user_text):
                raise TurnConflict('REQUEST_INPUT_CONFLICT')
            if existing.status == 'completed':
                answer = db.get(ChatMessage, existing.assistant_message_id)
                if answer is None:
                    raise TurnConflict('REQUEST_RESULT_MISSING')
                return TurnLease(TurnIdentity(session.id, existing.id, request_id, session.instance_id), {'response': answer.content, 'rag_trace': answer.rag_trace})
            if existing.status != 'queued':
                raise TurnConflict('REQUEST_' + existing.status.upper())
        if db.query(ChatTurn).filter_by(session_ref_id=session.id).filter(
                ChatTurn.status.in_(['running', 'cancel_requested'])).first():
            raise TurnConflict('SESSION_BUSY')
        if existing is not None:
            turn = existing
            turn.status = 'running'
        else:
            if db.query(ChatTurn).filter_by(session_ref_id=session.id, status='queued').first():
                raise TurnConflict('QUEUED_REQUEST_PENDING')
            turn = _create_pending_turn(db, session, request_id, user_text, 'running')
        db.commit()
        return TurnLease(TurnIdentity(session.id, turn.id, request_id, session.instance_id))


def _create_pending_turn(db, session, request_id, user_text, status, *, parent_request_id=None):
    now = datetime.utcnow()
    message = ChatMessage(session_ref_id=session.id, message_type='human', content=user_text,
                          timestamp=now, status='complete')
    db.add(message)
    db.flush()
    turn = ChatTurn(session_ref_id=session.id, request_id=request_id,
                   input_digest=input_digest(user_text), status=status, user_message_id=message.id,
                   parent_request_id=parent_request_id)
    db.add(turn)
    session.updated_at = now
    db.flush()
    return turn


def _owned_turn(db, user_id, session_id, request_id, *, lock=False):
    query = db.query(ChatSession).join(User).filter(User.username == user_id, ChatSession.session_id == session_id)
    session = (query.with_for_update(of=ChatSession) if lock else query).first()
    turn = db.query(ChatTurn).filter_by(session_ref_id=session.id, request_id=str(request_id)).first() if session else None
    if turn is None:
        raise TurnConflict('REQUEST_NOT_FOUND')
    return session, turn


def run_status(user_id, session_id, request_id):
    with SessionLocal() as db:
        _, turn = _owned_turn(db, user_id, session_id, request_id)
        return {'run_id': turn.request_id, 'status': turn.status}


def pending_run(user_id, session_id, *, session_factory=None):
    with (session_factory or SessionLocal)() as db:
        session = db.query(ChatSession).join(User).filter(User.username == user_id, ChatSession.session_id == session_id).first()
        turn = db.query(ChatTurn).filter_by(session_ref_id=session.id, status='queued').first() if session else None
        if turn is None:
            return None
        message = db.get(ChatMessage, turn.user_message_id)
        return {'run_id': turn.request_id, 'source_run_id': turn.parent_request_id, 'message': message.content}


def request_cancel(user_id, session_id, request_id, *, supplement_message=None, supplement_request_id=None):
    with SessionLocal() as db:
        session, turn = _owned_turn(db, user_id, session_id, request_id, lock=True)
        next_id = None
        if supplement_message is not None:
            if not supplement_message.strip() or not supplement_request_id:
                raise TurnConflict('INVALID_SUPPLEMENT')
            next_id = str(UUID(str(supplement_request_id)))
            if next_id == turn.request_id:
                raise TurnConflict('REQUEST_INPUT_CONFLICT')
            queued = db.query(ChatTurn).filter_by(session_ref_id=session.id, request_id=next_id).first()
            if queued is not None:
                if queued.input_digest != input_digest(supplement_message):
                    raise TurnConflict('REQUEST_INPUT_CONFLICT')
            else:
                if db.query(ChatTurn).filter_by(session_ref_id=session.id, status='queued').first():
                    raise TurnConflict('QUEUED_REQUEST_PENDING')
                _create_pending_turn(db, session, next_id, supplement_message, 'queued', parent_request_id=turn.request_id)
        if turn.status == 'running':
            turn.status = 'cancel_requested'
        elif turn.status == 'queued':
            turn.status, turn.error_code = 'cancelled', 'REQUEST_CANCELLED'
        result = {'run_id': turn.request_id, 'status': turn.status, 'next_run_id': next_id}
        if turn.status == 'completed':
            answer = db.get(ChatMessage, turn.assistant_message_id)
            if answer is not None:
                result['response'] = answer.content
        identity = TurnIdentity(session.id, turn.id, turn.request_id, session.instance_id)
        db.commit()
        if result['status'] == 'cancel_requested':
            notify_cancel(identity)
        return result


def recover_interrupted_turns():
    """Call only after obtaining exclusive backend process ownership."""
    with SessionLocal() as db:
        db.query(ToolExecution).filter_by(status='running').update({'status': 'unknown', 'finished_at': datetime.utcnow()})
        count = db.query(ChatTurn).filter(ChatTurn.status.in_(['running', 'cancel_requested', 'queued'])).update(
            {'status': 'failed', 'error_code': 'PROCESS_INTERRUPTED'})
        db.commit()
        return count


async def admit_async(user_id, session_id, request_id, user_text):
    return await run_blocking(admit, user_id, session_id, request_id, user_text,
                              _on_cancel=lambda lease: lease.close('cancelled'))


def admitted_chat(function):
    if inspect.isasyncgenfunction(function):
        @wraps(function)
        async def stream(user_text, user_id='default_user', session_id='default_session', *, request_id=None, admission=None):
            lease = admission or await admit_async(user_id, session_id, request_id, user_text)
            status = 'cancelled'
            try:
                if lease.replay is not None:
                    yield f"data: {json.dumps({'type': 'content', 'content': lease.replay['response'], 'run_id': lease.request_id})}\n\n"
                    if lease.replay['rag_trace']:
                        from .agent import _public_rag_trace
                        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': _public_rag_trace(lease.replay['rag_trace']), 'run_id': lease.request_id})}\n\n"
                    yield 'data: [DONE]\n\n'
                else:
                    with own_run(lease.identity, session_factory=SessionLocal, bind_context=False) as control:
                        context = copy_context()
                        context.run(bind_control, control)
                        if (await run_blocking(run_status, user_id, session_id, lease.request_id))['status'] == 'cancel_requested':
                            control.request_cancel()
                        chunks = function(user_text, user_id, session_id, request_id=lease.request_id, turn_identity=lease.identity)
                        async def advance():
                            control.bind_task()
                            check_cancelled()
                            return await chunks.__anext__()
                        try:
                            while True:
                                try:
                                    chunk = await _await_owned_task(asyncio.create_task(advance(), context=context))
                                except StopAsyncIteration:
                                    break
                                if chunk.startswith('data: ') and chunk.strip() != 'data: [DONE]':
                                    payload = json.loads(chunk[6:].strip())
                                    payload['run_id'] = lease.request_id
                                    chunk = f"data: {json.dumps(payload)}\n\n"
                                yield chunk
                        except RunCancelled:
                            raise asyncio.CancelledError from None
                        finally:
                            # Tokens and generator cleanup stay in the same Context even
                            # when a caller starts/ends iteration in different tasks.
                            control.cancelled.set()
                            control.task = None
                            await _await_owned_task(asyncio.create_task(chunks.aclose(), context=context))
                status = 'failed'  # Completed rows are unchanged by conditional cleanup.
            finally:
                await run_blocking(lease.close, status)
        return stream
    @wraps(function)
    def sync(user_text, user_id='default_user', session_id='default_session', *, request_id=None, admission=None):
        lease = admission or admit(user_id, session_id, request_id, user_text)
        try:
            if lease.replay is not None:
                return lease.replay
            with own_run(lease.identity, session_factory=SessionLocal) as control:
                if run_status(user_id, session_id, lease.request_id)['status'] == 'cancel_requested':
                    control.request_cancel()
                try:
                    return function(user_text, user_id, session_id, request_id=lease.request_id, turn_identity=lease.identity)
                except RunCancelled:
                    lease.close('cancelled')
                    raise TurnConflict('REQUEST_CANCELLED') from None
        finally:
            lease.close()
    return sync
