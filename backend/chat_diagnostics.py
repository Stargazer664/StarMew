"""Content-free instrumentation around actual chat calls and stream consumption."""
from __future__ import annotations

import asyncio
import inspect
import time
from contextlib import contextmanager, nullcontext, aclosing
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import wraps
from itertools import count

from . import telemetry
from .agent_budget import AgentRunState, use_run_state, current_run_state, model_timeout_kwargs
from .llm_config import resolve_llm_settings
from .tool_contracts import ToolResult, tool_error, tool_exception
from .conversation_context import assert_model_input, model_output_kwargs


@dataclass
class RequestMetrics:
    calls: object = field(default_factory=lambda: count(1))
    retrievals: object = field(default_factory=lambda: count(1))
    retrieval_count: int = 0


_metrics: ContextVar[RequestMetrics | None] = ContextVar('chat_metrics', default=None)


@contextmanager
def request_metrics():
    token = _metrics.set(RequestMetrics())
    try:
        yield _metrics.get()
    finally:
        _metrics.reset(token)


def next_retrieval_round():
    metrics = _metrics.get()
    if metrics is None:
        return None
    metrics.retrieval_count += 1
    return next(metrics.retrievals)


def model_identity(model, role):
    # Resolve the actual bound model first; never serialize config/headers/URLs.
    actual = model
    for _ in range(4):
        bound = getattr(actual, 'bound', None)
        if bound is None:
            break
        actual = bound
    try:
        settings = resolve_llm_settings(role=role)
        provider, configured = settings.provider, settings.model
    except Exception:
        provider, configured = 'unknown', None
    name = getattr(actual, 'model_name', None) or getattr(actual, 'model', None)
    return {'model': name if isinstance(name, str) else configured, 'provider': provider, 'role': role}


def _has_text(message):
    if getattr(message, 'tool_calls', None) or getattr(message, 'tool_call_chunks', None):
        return False
    content = getattr(message, 'content', '')
    if isinstance(content, str):
        return bool(content)
    if isinstance(content, list):
        return any((isinstance(item, str) and bool(item)) or (
            isinstance(item, dict) and item.get('type') == 'text' and bool(item.get('text'))
        ) for item in content)
    return False


class ModelCall:
    def __init__(self, operation, purpose, streaming):
        self.operation, self.purpose, self.streaming = operation, purpose, streaming
        self.started = time.perf_counter_ns()
        self.first = None

    @property
    def id(self):
        return self.operation.id

    def observe(self, message):
        if self.streaming and self.first is None and _has_text(message):
            self.first = time.perf_counter_ns()
            self.operation.mark_first_content()
            if self.purpose == 'final_answer':
                self.operation.annotate(first_answer_ms=(self.first - self.started) / 1e6)
        usage = getattr(message, 'usage_metadata', None)
        if not isinstance(usage, dict):
            metadata = getattr(message, 'response_metadata', None)
            usage = metadata.get('token_usage') if isinstance(metadata, dict) else None
        if isinstance(usage, dict):
            values = {}
            for target, aliases in {'input_tokens': ('input_tokens', 'prompt_tokens'),
                                    'output_tokens': ('output_tokens', 'completion_tokens'),
                                    'total_tokens': ('total_tokens',)}.items():
                value = next((usage[k] for k in aliases if k in usage), None)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    values[target] = value
            if values:
                self.operation.annotate(usage_source='provider', **values)


@contextmanager
def model_call(model, *, purpose, role='generation', streaming=False, **attrs):
    metrics = _metrics.get()
    ordinal = next(metrics.calls) if metrics else None
    with telemetry.span('generation' if purpose == 'final_answer' else 'model_call',
                        component='model', purpose=purpose, ordinal=ordinal,
                        streaming=streaming, **model_identity(model, role), **attrs) as operation:
        call = ModelCall(operation, purpose, streaming)
        try:
            yield call
        except (asyncio.CancelledError, GeneratorExit):
            operation.set_result('cancelled')
            raise
        except Exception:
            telemetry.event('model_call_failed', status='failed', code='MODEL_CALL_FAILED', purpose=purpose)
            raise
        else:
            if purpose == 'final_answer':
                end = time.perf_counter_ns()
                operation.annotate(generation_total_ms=(end - call.started) / 1e6,
                                   answer_stream_ms=(end - call.first) / 1e6 if call.first else None)


def invoke_model(model, inputs, *, purpose, role='generation', schema=None, **attrs):
    assert_model_input(inputs, role=role, schema=schema)
    with model_call(model, purpose=purpose, role=role, **attrs) as call:
        result = model.invoke(inputs, **model_timeout_kwargs(purpose), **model_output_kwargs(model, role))
        state = current_run_state()
        if state:
            state.timeout(final=purpose == "final_answer")
        call.observe(result)
        return result


def invoke_tool(tool, inputs, *, purpose, tool_call_id=None):
    from .run_control import check_cancelled, RunCancelled
    from .tool_audit import begin_tool, finish_tool
    check_cancelled()
    with telemetry.span('tool', component='tool', purpose=purpose) as operation:
        try:
            if tool is None:
                result = tool_error('UNKNOWN_TOOL')
            elif not isinstance(inputs, dict):
                result = tool_error('INVALID_ARGUMENT')
            else:
                schema = getattr(tool, 'args_schema', None)
                fields = getattr(schema, 'model_fields', None)
                if isinstance(fields, dict) and set(inputs) - set(fields):
                    result = tool_error('INVALID_ARGUMENT')
                else:
                    # Validation/default application precedes claiming actual execution.
                    if isinstance(fields, dict):
                        inputs = schema.model_validate(inputs).model_dump()
                    state = current_run_state()
                    name = getattr(tool, 'name', None)
                    if not isinstance(name, str):
                        name = {'weather': 'get_current_weather', 'knowledge_search': 'search_knowledge_base'}.get(purpose, purpose)
                    signature, code = state.claim_tool(name, inputs) if state else (None, None)
                    if code:
                        result = tool_error(code)
                    else:
                        record_id = begin_tool(tool_call_id, name, inputs)
                        try:
                            check_cancelled()
                            result = tool.invoke(inputs)
                            if not isinstance(result, ToolResult):
                                raise TypeError('Tool must return ToolResult')
                        except RunCancelled:
                            finish_tool(record_id, status='cancelled')
                            raise
                        except Exception as exc:
                            result = tool_exception(exc)
                        except BaseException:
                            finish_tool(record_id, status='unknown')
                            raise
                        finish_tool(record_id, result=result)
                        if state:
                            state.record_tool(signature, result)
        except RunCancelled:
            raise
        except Exception as exc:
            result = tool_exception(exc)
        operation.annotate(tool_status=result.status, code=result.code,
                           retryable=result.retryable, evidence_status=result.evidence_status)
        operation.set_result({'ok': 'completed', 'empty': 'empty', 'error': 'failed', 'skipped': 'skipped'}[result.status])
        check_cancelled()
        return result


async def ainvoke_model(model, inputs, *, purpose, role='generation', schema=None, **attrs):
    assert_model_input(inputs, role=role, schema=schema)
    with model_call(model, purpose=purpose, role=role, **attrs) as call:
        timeout = model_timeout_kwargs(purpose)
        async with asyncio.timeout(timeout.get("timeout")):
            result = await model.ainvoke(inputs, **timeout, **model_output_kwargs(model, role))
        state = current_run_state()
        if state:
            state.timeout(final=purpose == "final_answer")
        call.observe(result)
        return result


def mark_public_answer():
    execution = telemetry.current_trace()
    if execution is not None:
        execution.mark_first_answer()


def mark_chat_failed():
    execution = telemetry.current_trace()
    if execution is not None:
        execution.set_result('failed')


@contextmanager
def _chat_request(user_id, session_id, request_id=None):
    existing = telemetry.current_trace()
    scope = nullcontext(existing) if existing is not None else telemetry.trace(
        'chat', 'answer', trace_id=telemetry.get_context().get('trace_id'),
        user_id=user_id, session_id=session_id, request_id=request_id)
    with telemetry.bind_context(request_id=request_id), scope, request_metrics() as metrics, use_run_state(AgentRunState()) as run:
        try:
            yield
        finally:
            with telemetry.span('agent_budget', component='agent', stop_reason=run.stop_reason,
                                decision_count=run.decision_count, tool_count=run.tool_count,
                                kb_call_count=len(run.kb_signatures), answer_status=run.answer_status):
                pass
        if not metrics.retrieval_count:
            for stage in ('retrieval', 'rerank'):
                with telemetry.span(stage, component='rag') as operation:
                    operation.set_result('skipped')


async def _await_owned_task(task):
    """Forward cancellation once, then drain actual cleanup despite re-cancels."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not cancelled:
                task.cancel()
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def run_blocking(function, *args, _on_cancel=None, **kwargs):
    """Own a synchronous operation until its real thread finishes, even on re-cancel.

    The optional result cleanup handles resources acquired during cancellation.
    Sessions must be created and closed inside function, never on the event loop.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if task.exception() is None and _on_cancel is not None:
            try:
                await run_blocking(_on_cancel, task.result())
            except asyncio.CancelledError:
                pass
        raise asyncio.CancelledError
    return task.result()


def traced_chat(function):
    if inspect.isasyncgenfunction(function):
        async def owned_stream(user_text, user_id, session_id, request_id, turn_identity):
            with _chat_request(user_id, session_id, request_id):
                async with aclosing(function(user_text, user_id, session_id, request_id=request_id, turn_identity=turn_identity)) as chunks:
                    async for chunk in chunks:
                        yield chunk

        @wraps(function)
        async def stream(user_text, user_id='default_user', session_id='default_session', *, request_id=None, turn_identity=None):
            # ASGI/direct callers may advance or close a stream from another task.
            # Always enter, resume, and exit instrumentation in one private Context.
            context = copy_context()
            chunks = owned_stream(user_text, user_id, session_id, request_id, turn_identity)
            async def advance():
                return await anext(chunks)
            async def close():
                await chunks.aclose()
            try:
                while True:
                    try:
                        chunk = await _await_owned_task(asyncio.create_task(advance(), context=context))
                    except StopAsyncIteration:
                        break
                    yield chunk
            finally:
                await _await_owned_task(asyncio.create_task(close(), context=context))
        return stream
    @wraps(function)
    def sync(user_text, user_id='default_user', session_id='default_session', *, request_id=None, turn_identity=None):
        with _chat_request(user_id, session_id, request_id):
            return function(user_text, user_id, session_id, request_id=request_id, turn_identity=turn_identity)
    return sync
