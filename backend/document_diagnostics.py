"""Safe document operation boundaries shared by workers and recovery."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter_ns
from uuid import uuid4

from .telemetry import bind_context, event, span, trace

_attempt = ContextVar("document_attempt", default=1)
_outcome = ContextVar("document_outcome", default=None)


def current_attempt():
    return _attempt.get()


def mark_document_blocked(blocked_by_job_id):
    _outcome.set("blocked")
    event("task_blocked", status="blocked", code="BLOCKED_BY_RECOVERY", blocked_by_job_id=blocked_by_job_id)


@contextmanager
def document_operation(stage, component, code, *, failure_event="operation_failed", **attrs):
    with span(stage, component=component, code=code, **attrs):
        try:
            yield
        except Exception as exc:
            if failure_event != "operation_failed":
                event(failure_event, status="failed", code=code, stage=stage,
                      component=component, error_type=type(exc).__name__, **attrs)
            raise


def document_execution(operation):
    """The worker enters this scope only after acquiring the existing index gate."""
    def decorate(function):
        @wraps(function)
        def run(job_id, *args, admitted_at=None, request_id=None, **kwargs):
            manager = function.__globals__[operation + "_job_manager"]
            job = manager.get_job(job_id)
            if not job or job["status"] in {"completed", "failed"}:
                return None
            attempt = int(job.get("attempt", 1))
            token = _attempt.set(attempt)
            outcome_token = _outcome.set(None)
            try:
                request_fields = {"request_id": request_id} if request_id is not None else {}
                with trace("document", operation, trace_id=f"doc:{operation}:{job_id}:{attempt}",
                           job_id=job_id, attempt=attempt, **request_fields) as execution:
                    if admitted_at is not None:
                        execution.annotate(queue_wait_ms=max(0, (perf_counter_ns() - admitted_at) / 1_000_000))
                    event("attempt_started", status="running")
                    with span("processing", component="document") as processing:
                        result = function(job_id, *args, **kwargs)
                        try:
                            final = manager.get_job(job_id)
                        except Exception:
                            final = None
                        if final is None:
                            execution.set_result("unknown")
                            processing.set_result("unknown")
                            execution.annotate(diagnostics_incomplete=True)
                        elif final["status"] != "completed":
                            execution.set_result(_outcome.get() or "failed")
                            processing.set_result(_outcome.get() or "failed")
                    return result
            finally:
                _attempt.reset(token)
                _outcome.reset(outcome_token)
        return run
    return decorate


def recovery_execution(function):
    @wraps(function)
    def run(*args, **kwargs):
        recovery_run_id = uuid4().hex
        with bind_context(job_id=None, attempt=None, trace_id=None, parent_span_id=None):
            with trace("recovery", "recovery", recovery_run_id=recovery_run_id):
                event("recovery_started", status="running")
                result = function(*args, **kwargs)
                event("recovery_completed", status="completed")
                return result
    return run
