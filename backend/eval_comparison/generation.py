"""Controlled answer generation for paired RAG comparisons."""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from .contracts import ComparisonConfig, GenerationResult


COMMON_SYSTEM_PROMPT = (
    "你是一个中文问答助手。请直接回答问题，保持简洁，保留必要的数字、单位和步骤。\n"
    "如果提供了参考资料，优先使用其中与问题相关的内容。\n"
    "如果问题明确要求某份文档或某个版本的内容，不要把通用知识当作文档事实。\n"
    "信息不足且无法可靠确定时，请说明无法确定，不要编造。"
)


def build_user_prompt(question: str, contexts_text: str) -> str:
    context = contexts_text if contexts_text else "未提供参考资料"
    return f"【问题】\n{question}\n\n【参考资料】\n{context}"


def _doc_text(doc: dict[str, Any]) -> str:
    return str(doc.get("text") or doc.get("page_content") or "")


def build_context(
    docs: list[dict[str, Any]], *, top_k: int, max_chars: int
) -> tuple[str, list[dict[str, Any]]]:
    """Fit ranked document bodies and their wrappers into an exact char budget."""
    if top_k <= 0 or max_chars <= 0 or not docs:
        return "", []

    rendered = ""
    used: list[dict[str, Any]] = []
    for raw in docs[:top_k]:
        doc = dict(raw)
        original = _doc_text(doc)
        source = str(doc.get("filename") or doc.get("source") or "?")
        page = doc.get("page_number", doc.get("page", "?"))
        separator = "\n\n---\n\n" if rendered else ""
        wrapper = f"[来源: {source}, 页码: {page}]\n"
        available = max_chars - len(rendered) - len(separator) - len(wrapper)
        if available <= 0:
            break
        body = original[:available]
        if not body and original:
            break
        rendered += separator + wrapper + body
        used.append(
            {
                **doc,
                "source": source,
                "text": body,
                "original_length": len(original),
                "used_length": len(body),
                "truncated": len(body) < len(original),
            }
        )
        if len(rendered) >= max_chars:
            break
    return rendered, used


def create_generation_llm(config: ComparisonConfig):
    """Create the shared generation model lazily using existing env configuration."""
    from backend.llm_config import create_chat_model

    if not config.model:
        raise ValueError("generation model is required")
    return create_chat_model(
        role="generation",
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        timeout=config.generation_timeout,
        max_retries=0,
        **({"stream_usage": True} if config.stream else {}),
    )


def _normalise_answer_content(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content or "").strip()


def _reported_retrieval_failure(retrieved: dict[str, Any], trace: list[dict[str, Any]]) -> bool:
    if (retrieved.get("meta") or {}).get("retrieval_mode") == "failed":
        return True
    for step in trace:
        value = step.get("value", step) if isinstance(step, dict) else None
        if isinstance(value, dict) and value.get("retrieval_mode") == "failed":
            return True
    return False


async def _call_maybe_async(callable_obj: Any, *args: Any) -> Any:
    result = callable_obj(*args)
    if inspect.isawaitable(result):
        return await result
    return result


async def _invoke_llm(llm: Any, messages: list[tuple[str, str]], timeout: float) -> Any:
    if hasattr(llm, "ainvoke"):
        operation = llm.ainvoke(messages)
    elif hasattr(llm, "invoke"):
        operation = asyncio.to_thread(llm.invoke, messages)
    elif callable(llm):
        operation = _call_maybe_async(llm, messages)
    else:
        raise TypeError("llm must provide ainvoke/invoke or be callable")
    return await asyncio.wait_for(operation, timeout=timeout)


def _visible_stream_content(chunk: Any) -> str:
    """Extract answer text without exposing reasoning or stripping token spaces."""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if isinstance(block, dict):
            kind = block.get("type")
            text = block.get("text") or block.get("content")
        else:
            kind = getattr(block, "type", None)
            text = getattr(block, "text", None)
        if kind in {None, "text", "output_text"} and isinstance(text, str):
            parts.append(text)
    return "".join(parts)


_STREAM_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()


async def _close_stream_bounded(stream: Any, *, timeout: float) -> None:
    """Wait only for the supplied cleanup budget, then cancel without blocking.

    Supported SDKs cooperate with cancellation. This cannot interrupt blocking
    code or force a provider that keeps suppressing cancellation to terminate.
    Retain such a task until it ends and always retrieve its eventual exception.
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        return

    async def close_operation() -> None:
        await close()

    task = asyncio.create_task(close_operation())
    _STREAM_CLEANUP_TASKS.add(task)

    def collect_result(completed: asyncio.Task[Any]) -> None:
        _STREAM_CLEANUP_TASKS.discard(completed)
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(collect_result)
    try:
        completed, _ = await asyncio.wait({task}, timeout=timeout)
        if not completed:
            raise TimeoutError("stream cleanup exceeded its deadline")
        if task.cancelled():
            raise RuntimeError("stream cleanup cancelled itself")
        task.result()
    finally:
        if not task.done():
            task.cancel()


async def _stream_llm(
    llm: Any,
    messages: list[tuple[str, str]],
    timeout: float,
    on_visible_content: Callable[[], None],
) -> Any:
    """Consume and normally close within one deadline; bound failed cleanup.

    The supported OpenAI-compatible boundary reports cumulative usage totals,
    usually once in the final usage-only chunk. Keep the latest reported value
    for each field; adding AIMessageChunks would double-count repeated totals.
    Providers emitting usage deltas need an adapter before this boundary.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    stream = llm.astream(messages)

    async def consume() -> Any:
        parts: list[str] = []
        usage: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        async for chunk in stream:
            if chunk is None:
                continue
            text = _visible_stream_content(chunk)
            if text.strip():
                on_visible_content()
            parts.append(text)
            usage.update({
                key: value for key, value in (getattr(chunk, "usage_metadata", None) or {}).items()
                if value is not None
            })
            metadata.update({
                key: value for key, value in (getattr(chunk, "response_metadata", None) or {}).items()
                if value is not None
            })
        return SimpleNamespace(
            content="".join(parts), usage_metadata=usage, response_metadata=metadata,
        )

    failed = False
    try:
        return await asyncio.wait_for(consume(), timeout=max(0.0, deadline - loop.time()))
    except BaseException:
        failed = True
        raise
    finally:
        try:
            # A successful answer retains the original deadline for normal
            # closing; only a failed/cancelled read receives extra grace.
            cleanup_timeout = 0.1 if failed else max(0.0, deadline - loop.time())
            await _close_stream_bounded(stream, timeout=cleanup_timeout)
        except Exception:
            if not failed:
                raise


async def _invoke_retriever(
    retriever: Any, question: str, config: ComparisonConfig
) -> dict[str, Any]:
    if retriever is None:
        raise RuntimeError("rag arm requires a retriever")
    if isinstance(retriever, RetrievalProcess):
        return await asyncio.to_thread(
            retriever.retrieve,
            question,
            config.model_dump(mode="json"),
            config.retrieval_timeout,
        )
    target = getattr(retriever, "aretrieve", None) or retriever
    if inspect.iscoroutinefunction(target):
        result = await asyncio.wait_for(target(question), timeout=config.retrieval_timeout)
    else:
        result = await asyncio.wait_for(
            asyncio.to_thread(target, question), timeout=config.retrieval_timeout
        )
    if isinstance(result, list):
        return {"docs": result}
    if not isinstance(result, dict):
        raise TypeError("retriever must return a dict or document list")
    return result


async def generate_answer(
    question: str,
    *,
    arm: str,
    config: ComparisonConfig,
    llm: Any,
    retriever: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> GenerationResult:
    """Generate one answer without performing any evaluation."""
    if arm not in {"no_rag", "rag"}:
        raise ValueError("arm must be no_rag or rag")
    response_started = clock()
    retrieval_seconds = 0.0
    docs: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = []
    retrieval_usage: dict[str, Any] = {}
    retrieval_empty: bool | None = None

    if arm == "rag":
        retrieval_started = clock()
        try:
            retrieved = await _invoke_retriever(retriever, question, config)
            trace_value = retrieved.get("rag_trace") or retrieved.get("trace") or []
            trace = trace_value if isinstance(trace_value, list) else [{"value": trace_value}]
            if not retrieved.get("docs") and _reported_retrieval_failure(retrieved, trace):
                raise RuntimeError("retrieval pipeline reported failure")
        except Exception as exc:
            retrieval_seconds = clock() - retrieval_started
            response_seconds = clock() - response_started
            return GenerationResult(
                arm="rag",
                generation_status="failed",
                error=f"{type(exc).__name__}: {exc}",
                retrieved_docs=[],
                used_contexts=[],
                rag_trace=trace,
                retrieval_empty=None,
                timing={
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": None,
                    "response_seconds": response_seconds,
                    "first_content_seconds": None,
                    "generation_first_content_seconds": None,
                    "evaluation_seconds": None,
                },
                attempts=[{"stage": "retrieval", "status": "failed", "error_type": type(exc).__name__}],
            )
        retrieval_seconds = clock() - retrieval_started
        docs = [dict(doc) for doc in retrieved.get("docs", [])]
        retrieval_usage = dict(retrieved.get("usage") or {})
        retrieval_empty = not docs
        contexts_text, used = build_context(
            docs, top_k=config.top_k, max_chars=config.max_context_chars
        )
    else:
        contexts_text = ""

    prompt = build_user_prompt(question, contexts_text)
    messages = [("system", COMMON_SYSTEM_PROMPT), ("human", prompt)]
    generation_started = clock()
    first_content_at: float | None = None

    def record_first_content() -> None:
        nonlocal first_content_at
        if first_content_at is None:
            first_content_at = clock()

    try:
        if config.stream:
            response = await _stream_llm(llm, messages, config.generation_timeout, record_first_content)
        else:
            response = await _invoke_llm(llm, messages, config.generation_timeout)
        answer = _normalise_answer_content(response)
        if not answer:
            raise ValueError("model returned an empty answer")
        status = "success"
        error = None
    except Exception as exc:
        response = None
        answer = None
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    generation_seconds = clock() - generation_started
    response_seconds = clock() - response_started

    usage = {
        "generation_input_tokens": None,
        "generation_output_tokens": None,
        "retrieval_llm_tokens": retrieval_usage.get("retrieval_llm_tokens"),
        "retrieval_usage_complete": retrieval_usage.get("retrieval_usage_complete", False),
    }
    if response is not None:
        response_usage = getattr(response, "usage_metadata", None) or {}
        usage["generation_input_tokens"] = response_usage.get("input_tokens")
        usage["generation_output_tokens"] = response_usage.get("output_tokens")
        metadata = getattr(response, "response_metadata", None) or {}
        usage["returned_model"] = metadata.get("model_name") or metadata.get("model")
        usage["system_fingerprint"] = metadata.get("system_fingerprint")

    return GenerationResult(
        arm=arm,
        generation_status=status,
        answer=answer,
        error=error,
        retrieved_docs=docs,
        used_contexts=used,
        rag_trace=trace,
        retrieval_empty=retrieval_empty,
        prompt=prompt,
        timing={
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "response_seconds": response_seconds,
            "first_content_seconds": (
                first_content_at - response_started if first_content_at is not None else None
            ),
            "generation_first_content_seconds": (
                first_content_at - generation_started if first_content_at is not None else None
            ),
            "evaluation_seconds": None,
        },
        usage=usage,
        attempts=[
            {
                "stage": "generation",
                "status": status,
                "error_type": error.split(":", 1)[0] if error else None,
            }
        ],
    )


def _retrieval_worker_loop(
    target: Callable[[str, dict[str, Any]], dict[str, Any]],
    requests: Any,
    responses: Any,
) -> None:
    while True:
        request = requests.get()
        if request is None:
            return
        request_id, question, config = request
        try:
            value = target(question, config)
            if inspect.isawaitable(value):
                value = asyncio.run(value)
            responses.put((request_id, True, value))
        except BaseException as exc:  # child must serialize errors, not crash silently
            responses.put((request_id, False, (type(exc).__name__, str(exc))))


class RetrievalProcess:
    """Persistent spawn-safe retrieval boundary that can be killed on timeout."""

    def __init__(self, target: Callable[[str, dict[str, Any]], dict[str, Any]]):
        self._target = target
        self._context = mp.get_context("spawn")
        self._requests = None
        self._responses = None
        self._process = None
        self._request_id = 0

    @property
    def is_alive(self) -> bool:
        return bool(self._process and self._process.is_alive())

    def _start(self) -> None:
        self._requests = self._context.Queue()
        self._responses = self._context.Queue()
        self._process = self._context.Process(
            target=_retrieval_worker_loop,
            args=(self._target, self._requests, self._responses),
            daemon=True,
        )
        self._process.start()

    def retrieve(self, question: str, config: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not self.is_alive:
            self._start()
        self._request_id += 1
        request_id = self._request_id
        self._requests.put((request_id, question, config))
        try:
            received_id, ok, payload = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            self._terminate()
            raise TimeoutError(f"retrieval exceeded {timeout}s") from exc
        if received_id != request_id:
            self._terminate()
            raise RuntimeError("retrieval response identity mismatch")
        if not ok:
            error_type, message = payload
            raise RuntimeError(f"{error_type}: {message}")
        if not isinstance(payload, dict):
            raise TypeError("retrieval worker returned a non-object result")
        return payload

    def _terminate(self) -> None:
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._process = None
        for channel_name in ("_requests", "_responses"):
            channel = getattr(self, channel_name)
            if channel is not None:
                channel.close()
                channel.join_thread()
                setattr(self, channel_name, None)

    def close(self) -> None:
        if self.is_alive:
            self._requests.put(None)
            self._process.join(timeout=5)
        if self.is_alive:
            self._process.terminate()
            self._process.join(timeout=5)
        self._process = None
        for channel_name in ("_requests", "_responses"):
            channel = getattr(self, channel_name)
            if channel is not None:
                channel.close()
                channel.join_thread()
                setattr(self, channel_name, None)
