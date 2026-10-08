"""Best-effort local telemetry: bounded queue, no content, monotonic durations."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
from queue import Queue, Full
import re
from threading import Condition, Event, Lock, Thread, get_ident
import time
from uuid import uuid4

from .diagnostics_store import DiagnosticsStore, utc_now
from .document_alerts import classify_alert


_context = ContextVar('diagnostic_context', default=None)
_runtime = None
_lifecycle_lock = Lock()
_FIELDS = set('scope operation job_id attempt request_id trace_id parent_span_id span_id stage component purpose '
              'model provider role ordinal retrieval_round retrieval_mode fallback_of cold_start streaming structured usage_source input_tokens output_tokens total_tokens '
              'first_answer_ms generation_total_ms answer_stream_ms answer_first_content_ms first_content_ms '
              'queue_wait_ms processing_ms receive_ms rollback_ms recovery_ms status code error_type '
              'blocked_by_job_id recovery_run_id cause_event_id candidate_count input_count output_count top_k mode '
              'run_id current_step recovery_required retryable total_chunks processed_chunks diagnostics_incomplete '
              'session_ref_id target_message_count user_id session_id reason result rerank_applied fallback_used cancelled_after_ms tool_status evidence_status stop_reason decision_count tool_count kb_call_count answer_status'.split())


def safe_fields(values):
    result = {}
    for key, value in values.items():
        if key not in _FIELDS or value is None:
            continue
        if isinstance(value, (bool,int)):
            result[key] = value
        elif isinstance(value,float):
            if math.isfinite(value):
                result[key] = value
        elif isinstance(value,str):
            # No arbitrary exception strings: callers only pass named fields.
            value = re.sub(r'https?://\S+|[A-Za-z]:[\\/]\S+|(?:password|api[_-]?key|token|secret)\s*[:=]\s*\S+', '[redacted]', value, flags=re.I)
            result[key] = re.sub(r'[\r\n\x00-\x1f]', ' ', value)[:160]
    return result


def get_context():
    return dict(_context.get() or {})


def current_trace():
    return get_context().get('trace')


def positive_setting(name, default):
    try:
        value=float(os.getenv(name,str(default)))
        if math.isfinite(value) and value>0:
            return value
    except ValueError:
        pass
    logging.getLogger('starmew.diagnostics').warning(json.dumps({'code':'CONFIG_DEFAULT_USED','setting':name}))
    return default


@contextmanager
def bind_context(**fields):
    token = _context.set({**get_context(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class Runtime:
    def __init__(self, path, log_dir, queue_size=4096):
        self.store=DiagnosticsStore(path)
        self.run_id=uuid4().hex
        self.queue=Queue(maxsize=queue_size)
        self.active={}
        self.waiting={}
        self.lock=Lock()
        self.drained=Condition(self.lock)
        self.stop=Event()
        self.accepting=True
        self.degraded=False
        self.failures=set()
        self.gap_pending=False
        self.gaps=0
        self.handlers={}
        self.log_dir=Path(log_dir)
        self.last_prune=0
        self.last_error_report=0
        self.last_job={}
        self.gap_records={}
        self.initialized=False
        try:
            self.store.initialize()
            self.store.mark_interrupted(self.run_id)
            self.initialized=True
        except Exception:
            self.failure('DIAGNOSTICS_STORE_UNAVAILABLE')
        self.writer=Thread(target=self._write_loop,name='diagnostics-writer',daemon=True)
        self.checker=Thread(target=self._check_loop,name='diagnostics-watch',daemon=True)
        self.writer.start()
        self.checker.start()

    def failure(self, code):
        if code not in self.failures:
            self.gap_pending=True
        self.failures.add(code)
        self.degraded=True
        self.gaps+=1
        now=time.monotonic()
        if now-self.last_error_report > 10:
            self.last_error_report=now
            # stderr must never include the original exception or business content.
            logging.getLogger('starmew.diagnostics').error(json.dumps({'code':code,'diagnostics_degraded':True}))

    def emit(self,item):
        if not self.accepting:
            return
        try:
            self.queue.put_nowait(item)
        except Full:
            self.mark_gap(item)
            self.failure('DIAGNOSTICS_QUEUE_FULL')

    def mark_gap(self,item):
        trace_id=item.get('trace_id') or (item.get('id') if item.get('kind')=='trace' else None)
        if not trace_id:
            return
        with self.lock:
            active=self.active.get(trace_id)
            if active:
                active.data['diagnostics_incomplete']=True
            if trace_id in self.gap_records or len(self.gap_records)<4096:
                self.gap_records[trace_id]={'kind':'gap','id':trace_id,'trace_id':trace_id,
                                            'scope':item.get('scope','system'),'created_at':utc_now()}

    def track_job_queue(self,job,operation,trace_id):
        ready=(job.get('status')=='pending' and job.get('current_step')!='upload') or (
            job.get('status')=='running' and job.get('current_step')=='upload' and
            any(step.get('key')=='upload' and step.get('status')=='completed' for step in job.get('steps',[])))
        if job.get('status') in ('completed','failed'):
            self.finish_wait(trace_id,'cancelled')
            return
        with self.lock:
            if not ready or trace_id in self.active or trace_id in self.waiting:
                return
            waiting=Span('queue_wait',scope='document',operation=operation,job_id=job['job_id'],
                         attempt=job.get('attempt',1),trace_id=trace_id,component='index_lock')
            waiting.started_ns=time.perf_counter_ns()
            waiting.data.update(id=waiting.id,started_at=utc_now(),status='running',run_id=self.run_id)
            self.waiting[trace_id]=waiting
        self.emit({'kind':'span',**waiting.data})

    def finish_wait(self,trace_id,status='completed'):
        with self.lock:
            waiting=self.waiting.pop(trace_id,None)
        if waiting is None:
            return None
        duration=(time.perf_counter_ns()-waiting.started_ns)/1e6
        self.emit({'kind':'span',**waiting.data,'status':status,'duration_ms':duration,'ended_at':utc_now()})
        return duration

    def _log(self,item):
        scope=item.get('scope','system')
        name='chat' if scope=='chat' else 'document'
        try:
            handler=self.handlers.get(name)
            if handler is None:
                self.log_dir.mkdir(parents=True,exist_ok=True)
                handler=RotatingFileHandler(self.log_dir/(name+'-operations.jsonl'),maxBytes=10*1024*1024,backupCount=5,encoding='utf-8')
                handler.setFormatter(logging.Formatter('%(message)s'))
                # logging.Handler normally swallows I/O errors; surface them safely.
                def handle_error(record):
                    handler.diagnostic_failed=True
                    self.failure('DIAGNOSTICS_LOG_UNAVAILABLE')
                handler.handleError=handle_error
                self.handlers[name]=handler
            public={k:v for k,v in item.items() if k not in ('user_id','session_id')}
            record=logging.LogRecord('starmew.operations',logging.INFO,'',0,json.dumps(public,ensure_ascii=False),(),None)
            handler.diagnostic_failed=False
            handler.handle(record)
            # FileHandler.handleError sets this flag on actual I/O errors.
            if not getattr(handler,'diagnostic_failed',False):
                self.failures.discard('DIAGNOSTICS_LOG_UNAVAILABLE')
        except Exception:
            self.failure('DIAGNOSTICS_LOG_UNAVAILABLE')

    def _write_loop(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:
                    return
                self._log(item)
                try:
                    # Retry initialization after transient startup failures.
                    if not self.initialized:
                        self.store.initialize()
                        self.store.mark_interrupted(self.run_id)
                        self.initialized=True
                    with self.lock:
                        gaps=dict(self.gap_records)
                    for trace_id,gap in gaps.items():
                        self.store.write(gap)
                        with self.lock:
                            if self.gap_records.get(trace_id)==gap:
                                self.gap_records.pop(trace_id,None)
                    self.store.write(item)
                    if self.gap_pending:
                        gap={'kind':'event','event_key':uuid4().hex,'created_at':utc_now(),
                             'scope':'system','name':'diagnostics_gap','code':'DIAGNOSTICS_GAP','status':'warning'}
                        self.store.write(gap)
                        self.gap_pending=False
                    self.failures.difference_update({'DIAGNOSTICS_STORE_UNAVAILABLE','DIAGNOSTICS_WRITE_FAILED','DIAGNOSTICS_QUEUE_FULL'})
                    self.degraded=bool(self.failures)
                except Exception:
                    self.mark_gap(item)
                    self.failure('DIAGNOSTICS_WRITE_FAILED')
            finally:
                self.queue.task_done()

    def _check_loop(self):
        while not self.stop.wait(10):
            self.check_slow()
            if time.monotonic()-self.last_prune > 86400:
                self.last_prune=time.monotonic()
                try:
                    result=self.store.prune(days=positive_setting('DIAGNOSTICS_RETENTION_DAYS',30))
                    if result['removed']>=100:
                        # Continue another bounded page at the next ten-second tick.
                        self.last_prune=0
                    self.failures.discard('DIAGNOSTICS_RETENTION_FAILED')
                    self.degraded=bool(self.failures)
                    if result['over_capacity']:
                        self.emit({'kind':'alert','dedupe_key':'diagnostic-capacity','rule':'DIAGNOSTICS_CAPACITY',
                                   'code':'DIAGNOSTICS_CAPACITY','severity':'warning','scope':'system','once':True})
                except Exception:
                    self.failure('DIAGNOSTICS_RETENTION_FAILED')

    def check_slow(self,now_ns=None):
        now_ns=now_ns if now_ns is not None else time.perf_counter_ns()
        with self.lock:
            operations=list(self.active.values())+list(self.waiting.values())
        for operation in operations:
            elapsed=(now_ns-operation.started_ns)/1e9
            component=operation.data.get('component')
            stage=operation.data.get('stage','')
            seconds=positive_setting('DIAGNOSTICS_STAGE_SLOW_SECONDS',300)
            if component=='model' or stage in ('generation','tool_decision'):
                seconds=positive_setting('DIAGNOSTICS_MODEL_SLOW_SECONDS',60)
            if 'wait' in stage or 'queue' in stage:
                seconds=positive_setting('DIAGNOSTICS_QUEUE_SLOW_SECONDS',120)
            elif component=='milvus':
                seconds=positive_setting('DIAGNOSTICS_SEARCH_SLOW_SECONDS',10)
            elif 'rerank' in stage:
                seconds=positive_setting('DIAGNOSTICS_RERANK_SLOW_SECONDS',15)
            if operation.kind=='trace' and operation.data.get('scope')=='chat':
                if operation.data.get('answer_first_content_ms') is not None:
                    continue
                seconds=positive_setting('DIAGNOSTICS_FIRST_ANSWER_SECONDS',60)
            if elapsed>seconds:
                self.emit({'kind':'alert',**safe_fields(operation.data),'span_id':operation.id,
                           'dedupe_key':'slow:'+operation.id,'rule':'SLOW_OPERATION','code':'SLOW_OPERATION',
                           'severity':'warning','once':True})

    def close(self):
        self.stop.set()
        self.checker.join()
        # Request cancellation does not stop a sync tool thread. It must finish
        # and enqueue its terminal events before the writer accepts its sentinel.
        with self.drained:
            while any(operation.thread_id != get_ident() for operation in self.active.values()):
                self.drained.wait(timeout=0.1)
            abandoned=list(self.active.values())
        for operation in abandoned:
            self.emit({'kind':operation.kind,**operation.data,'status':'interrupted',
                       'ended_at':utc_now(),'duration_ms':None,'diagnostics_incomplete':True})
        for trace_id in list(self.waiting):
            self.finish_wait(trace_id,'interrupted')
        self.accepting=False
        self.queue.put(None)
        self.writer.join()
        for handler in self.handlers.values():
            handler.close()


def start_diagnostics(path=None, log_dir=None, *, queue_size=4096):
    global _runtime
    with _lifecycle_lock:
        if _runtime is not None:
            return
        root=Path(__file__).resolve().parents[1]/'data'
        _runtime=Runtime(path or os.getenv('UPLOAD_JOB_DB_PATH') or root/'upload_jobs.sqlite3',
                         log_dir or os.getenv('DIAGNOSTICS_LOG_DIR') or root/'logs',queue_size)


def stop_diagnostics():
    global _runtime
    with _lifecycle_lock:
        if _runtime:
            _runtime.close()
            _runtime=None


def get_store():
    if _runtime is None:
        raise RuntimeError('诊断服务未运行')
    return _runtime.store


def diagnostics_status():
    return {'diagnostics_degraded':_runtime is None or _runtime.degraded,
            'diagnostics_gaps':_runtime.gaps if _runtime else 0}


def flush():
    if _runtime:
        _runtime.queue.join()


def check_slow_operations(now_ns=None):
    if _runtime:
        _runtime.check_slow(now_ns)


class Span:
    kind='span'

    def __init__(self,stage,**attrs):
        self.id=uuid4().hex
        self.started_ns=None
        self.data={'stage':stage,**safe_fields(attrs)}
        self.owner=None
        self._token=None
        self._runtime=None
        self.failure_reported=False
        self.thread_id=None

    def __enter__(self):
        self._runtime=_runtime
        self.started_ns=time.perf_counter_ns()
        self.thread_id=get_ident()
        context=get_context()
        self.owner=context.get('trace')
        self.data={**safe_fields(context),**self.data,'id':self.id,'started_at':utc_now(),'status':'running'}
        if self._runtime:
            self.data['run_id']=self._runtime.run_id
        if self.kind=='trace':
            self.data['trace_id']=self.id
            if self._runtime and self.data.get('scope')=='document':
                wait=self._runtime.finish_wait(self.id)
                if wait is not None:
                    self.data['queue_wait_ms']=wait
            self.owner=self
            context={**context,**self.data,'trace':self,'parent_span_id':None,'_operation':self}
        else:
            context={**context,**safe_fields(self.data),'parent_span_id':self.id,'span_id':self.id,'_operation':self}
        self._token=_context.set(context)
        if self._runtime:
            with self._runtime.lock:
                self._runtime.active[self.id]=self
            self._runtime.emit({'kind':self.kind,**self.data})
        return self

    def annotate(self,**attrs):
        self.data.update(safe_fields(attrs))

    def set_result(self,status):
        self.data['status']=status

    def mark_first_content(self):
        if self.started_ns is not None and self.data.get('first_content_ms') is None:
            self.data['first_content_ms']=(time.perf_counter_ns()-self.started_ns)/1e6

    def mark_first_answer(self):
        if self.started_ns is not None and self.data.get('answer_first_content_ms') is None:
            self.data['answer_first_content_ms']=(time.perf_counter_ns()-self.started_ns)/1e6

    def __exit__(self,exception_type,exception,traceback):
        try:
            if exception_type:
                cancelled=issubclass(exception_type,(asyncio.CancelledError,GeneratorExit))
                self.data.update(status='cancelled' if cancelled else 'failed',error_type=exception_type.__name__)
                if not cancelled:
                    frames=[]
                    while traceback is not None:
                        frame=traceback.tb_frame
                        frames.append({'module':str(frame.f_globals.get('__name__','unknown'))[:120],
                                       'function':frame.f_code.co_name[:120],'line':traceback.tb_lineno})
                        traceback=traceback.tb_next
                    self.data['error_frames']=frames[-8:]
            elif self.data['status']=='running':
                self.data['status']='completed'
            if self.kind=='span' and self.owner and self.owner.data.get('status')=='cancelled' and self.data['status']=='completed':
                self.data['status']='completed_after_cancel'
            self.data.update(ended_at=utc_now(),duration_ms=(time.perf_counter_ns()-self.started_ns)/1e6)
            if self.data.get('purpose')=='final_answer' and self.data['status']=='completed':
                self.data['generation_total_ms']=self.data['duration_ms']
                first=self.data.get('first_content_ms')
                if first is not None:
                    self.data.update(first_answer_ms=first,answer_stream_ms=max(0,self.data['duration_ms']-first))
            if self._runtime:
                self._runtime.emit({'kind':self.kind,**self.data})
                if self.data['status']=='failed' and not self.failure_reported:
                    event('operation_failed',status='failed',code=self.data.get('code') or self.data.get('stage','operation').upper()+'_FAILED',
                          span_id=self.id,error_type=self.data.get('error_type'),component=self.data.get('component'),stage=self.data.get('stage'))
                with self._runtime.drained:
                    self._runtime.active.pop(self.id,None)
                    self._runtime.drained.notify_all()
        finally:
            if self._token is not None:
                _context.reset(self._token)
        return False


class Trace(Span):
    kind='trace'

    def __init__(self,scope,operation,**attrs):
        super().__init__('request' if scope=='chat' else 'processing',scope=scope,operation=operation,**attrs)
        self.id=attrs.get('trace_id') or (f"doc:{operation}:{attrs['job_id']}:{attrs.get('attempt',1)}" if scope=='document' and attrs.get('job_id') else uuid4().hex)


def trace(scope,operation,**attrs):
    return Trace(scope,operation,**attrs)


def span(stage,**attrs):
    return Span(stage,**attrs)


def event(name,*,status='info',code=None,**attrs):
    event_id=uuid4().hex
    runtime=_runtime
    if runtime is None:
        return event_id
    active_operation=get_context().get('_operation')
    if status in ('failed','error') and active_operation:
        active_operation.failure_reported=True
    context=safe_fields(get_context())
    # Child recovery metadata takes precedence over its enclosing request.
    data={**context,**safe_fields(attrs),'name':re.sub(r'[^\w.-]','_',name)[:80],
          'status':status,'code':code,'event_key':event_id,'created_at':utc_now()}
    runtime.emit({'kind':'event',**data})
    classification=classify_alert(name,status,code,data.get('scope'))
    if classification:
        rule,severity,message=classification
        key=':'.join(str(value or '') for value in (rule,data.get('job_id') or data.get('trace_id'),
                                                 data.get('attempt'),data.get('stage')))
        runtime.emit({'kind':'alert',**data,'dedupe_key':key,'rule':rule,'severity':severity,'message':message})
    if name in ('recovery_completed','rollback_completed','status_reconciled','cleanup_completed'):
        rules={'recovery_completed':['RECOVERY_FAILED','BLOCKED_BY_RECOVERY'],
               'rollback_completed':['RECOVERY_FAILED'], 'status_reconciled':['STATUS_PERSIST_FAILED'],
               'cleanup_completed':['CLEANUP_FAILED']}[name]
        runtime.emit({'kind':'resolve',**context,**safe_fields(attrs),'rules':rules,
                      'resolve_stage':attrs.get('stage') if name=='cleanup_completed' else None})
    return event_id


def record_job(job,operation):
    if not job or not _runtime:
        return
    runtime=_runtime
    job_id=job['job_id']; attempt=job.get('attempt',1)
    trace_id=f'doc:{operation}:{job_id}:{attempt}'
    data=safe_fields(job)
    data.update(id=trace_id,trace_id=trace_id,scope='document',operation=operation,job_id=job_id,
                attempt=attempt,run_id=runtime.run_id)
    if job.get('created_at'):
        data['started_at']=job['created_at']
    data['steps']=[safe_fields({**step,'stage':step.get('key')}) for step in job.get('steps',[])]
    fingerprint=json.dumps(data,sort_keys=True)
    with runtime.lock:
        previous=runtime.last_job.get(trace_id)
        if previous==fingerprint:
            return
        runtime.last_job[trace_id]=fingerprint
        if len(runtime.last_job)>4096:
            runtime.last_job.pop(next(iter(runtime.last_job)))
    runtime.emit({'kind':'trace',**data})
    runtime.track_job_queue(job,operation,trace_id)
    if previous and json.loads(previous).get('status')==job.get('status'):
        return
    with bind_context(trace_id=trace_id,scope='document',operation=operation,job_id=job_id,attempt=attempt):
        event('task_'+job.get('status','updated'),status='warning' if job.get('status')=='failed' else 'info',
              code='TASK_FAILED' if job.get('status')=='failed' else None)


def timed(stage,*,component='',purpose=''):
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(*args,**kwargs):
                with span(stage,component=component,purpose=purpose):
                    return await function(*args,**kwargs)
            return async_wrapper
        @wraps(function)
        def wrapper(*args,**kwargs):
            with span(stage,component=component,purpose=purpose):
                return function(*args,**kwargs)
        return wrapper
    return decorate
