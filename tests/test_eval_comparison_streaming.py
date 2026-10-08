"""Offline tests for actual stream consumption and first visible content timing."""

import asyncio
import gc
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from backend.eval_comparison.contracts import ComparisonConfig
from backend.eval_comparison.generation import create_generation_llm, generate_answer


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class StreamBoundary:
    def __init__(self, events, *, clock=None, delay=0.0, close_error=None):
        self.events = iter(events)
        self.clock = clock
        self.delay = delay
        self.close_error = close_error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            elapsed, chunk = next(self.events)
        except StopIteration:
            raise StopAsyncIteration
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.clock is not None:
            self.clock.now += elapsed
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    async def aclose(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class ModelBoundary:
    def __init__(self, stream):
        self.stream = stream

    def astream(self, messages):
        return self.stream

    async def ainvoke(self, messages):
        return AIMessage(
            content="  invoke-only answer  ",
            usage_metadata={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
        )


def stream_config(**kwargs):
    # model_copy also lets RED exercise generation before the new field exists.
    return ComparisonConfig(**kwargs).model_copy(update={"stream": True})


def test_stream_first_content_excludes_retrieval_reasoning_and_empty_chunks():
    clock = Clock()
    stream = StreamBoundary(
        [
            (1, AIMessageChunk(content="", additional_kwargs={"reasoning_content": "推理"})),
            (1, AIMessageChunk(content=[{"type": "reasoning", "text": "隐藏推理"}])),
            (1, AIMessageChunk(content="  ")),
            (1, AIMessageChunk(content=[{"type": "text", "text": "alpha "}])),
            (1, AIMessageChunk(content="beta\n")),
            (1, AIMessageChunk(content="", usage_metadata={
                "input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
            })),
        ],
        clock=clock,
    )

    async def retrieve(question):
        clock.now += 2
        return {"docs": [{"filename": "fixture.md", "text": "evidence"}]}

    result = asyncio.run(generate_answer(
        "q", arm="rag", config=stream_config(), llm=ModelBoundary(stream),
        retriever=retrieve, clock=clock,
    ))

    assert result.generation_status == "success"
    assert result.answer == "alpha beta"
    assert result.timing["first_content_seconds"] == 6
    assert result.timing["generation_first_content_seconds"] == 4
    assert result.timing["generation_seconds"] == 6
    assert result.timing["response_seconds"] == 8
    assert result.usage["generation_input_tokens"] == 12
    assert result.usage["generation_output_tokens"] == 3
    assert stream.closed


@pytest.mark.parametrize("chunks", [
    [],
    [AIMessageChunk(content=""), AIMessageChunk(content=" \n")],
    [AIMessageChunk(content=[{"type": "reasoning", "text": "秘密推理"}])],
    [AIMessageChunk(content="", usage_metadata={
        "input_tokens": 10, "output_tokens": 0, "total_tokens": 10,
    })],
])
def test_stream_without_visible_answer_fails_without_inventing_first_content(chunks):
    stream = StreamBoundary([(0, chunk) for chunk in chunks])
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=stream_config(), llm=ModelBoundary(stream), retriever=None,
    ))
    assert result.generation_status == "failed"
    assert result.answer is None
    assert result.error == "ValueError: model returned an empty answer"
    assert result.timing["first_content_seconds"] is None
    assert result.timing["generation_first_content_seconds"] is None
    assert stream.closed


def test_stream_usage_is_latest_provider_total_not_sum_of_cumulative_snapshots():
    stream = StreamBoundary([
        (0, AIMessageChunk(content="a", usage_metadata={
            "input_tokens": 12, "output_tokens": 1, "total_tokens": 13,
        }, response_metadata={"model_name": "boundary-model"})),
        (0, AIMessageChunk(content="b", usage_metadata={
            "input_tokens": 12, "output_tokens": 2, "total_tokens": 14,
        })),
        (0, AIMessageChunk(content="", usage_metadata={
            "input_tokens": 12, "output_tokens": 2, "total_tokens": 14,
        }, response_metadata={"system_fingerprint": "fingerprint"})),
    ])
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=stream_config(), llm=ModelBoundary(stream), retriever=None,
    ))
    assert result.answer == "ab"
    assert result.usage["generation_input_tokens"] == 12
    assert result.usage["generation_output_tokens"] == 2
    assert result.usage["returned_model"] == "boundary-model"
    assert result.usage["system_fingerprint"] == "fingerprint"


def test_stream_failure_preserves_original_error_and_observed_first_content():
    clock = Clock()
    stream = StreamBoundary([
        (2, AIMessageChunk(content="partial")),
        (1, RuntimeError("provider disconnected")),
    ], clock=clock, close_error=ValueError("cleanup failed"))
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=stream_config(), llm=ModelBoundary(stream),
        retriever=None, clock=clock,
    ))
    assert result.generation_status == "failed"
    assert result.answer is None
    assert result.error == "RuntimeError: provider disconnected"
    assert result.timing["first_content_seconds"] == 2
    assert result.timing["generation_first_content_seconds"] == 2
    assert stream.closed


def test_stream_timeout_covers_all_chunks_and_closes_iterator():
    stream = StreamBoundary([(0, AIMessageChunk(content="x"))] * 100, delay=0.01)
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=stream_config(generation_timeout=0.08),
        llm=ModelBoundary(stream), retriever=None,
    ))
    assert result.generation_status == "failed"
    assert result.answer is None
    assert result.error.startswith("TimeoutError:")
    assert result.timing["first_content_seconds"] is not None
    assert result.timing["first_content_seconds"] < result.timing["generation_seconds"]
    assert stream.closed


def test_stream_timeout_before_answer_keeps_first_content_null():
    stream = StreamBoundary([(0, AIMessageChunk(content="late"))], delay=1)
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=stream_config(generation_timeout=0.02),
        llm=ModelBoundary(stream), retriever=None,
    ))
    assert result.generation_status == "failed"
    assert result.error.startswith("TimeoutError:")
    assert result.timing["first_content_seconds"] is None
    assert result.timing["generation_first_content_seconds"] is None
    assert stream.closed


def test_stream_timeout_does_not_wait_forever_for_aclose():
    async def exercise():
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        class BlockedClose(StreamBoundary):
            async def aclose(self):
                close_started.set()
                await close_release.wait()
                self.closed = True

        stream = BlockedClose([(0, AIMessageChunk(content="late"))], delay=1)
        task = asyncio.create_task(generate_answer(
            "q", arm="no_rag", config=ComparisonConfig(stream=True, generation_timeout=0.02),
            llm=ModelBoundary(stream), retriever=None,
        ))
        try:
            await asyncio.wait_for(close_started.wait(), timeout=1)
            completed, _ = await asyncio.wait({task}, timeout=0.5)
            assert task in completed, "generation timeout is still waiting for unbounded stream cleanup"
            result = task.result()
            assert result.generation_status == "failed"
            assert result.error.startswith("TimeoutError:")
            assert result.timing["first_content_seconds"] is None
        finally:
            # Release the fake provider even when the regression assertion fails;
            # the test itself must never leave a task blocked during loop shutdown.
            close_release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

    asyncio.run(exercise())


def test_successful_stream_allows_normal_close_within_generation_deadline():
    class SlowClose(StreamBoundary):
        async def aclose(self):
            await asyncio.sleep(0.2)
            self.closed = True

    stream = SlowClose([(0, AIMessageChunk(content="complete answer"))])
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=ComparisonConfig(stream=True, generation_timeout=1),
        llm=ModelBoundary(stream), retriever=None,
    ))
    assert result.generation_status == "success"
    assert result.answer == "complete answer"
    assert result.timing["first_content_seconds"] is not None
    assert stream.closed


def test_successful_stream_close_uses_remaining_not_restarted_generation_deadline():
    class SlowClose(StreamBoundary):
        async def aclose(self):
            await asyncio.sleep(0.15)
            self.closed = True

    stream = SlowClose([(0, AIMessageChunk(content="complete answer"))], delay=0.15)
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=ComparisonConfig(stream=True, generation_timeout=0.25),
        llm=ModelBoundary(stream), retriever=None,
    ))
    assert result.generation_status == "failed"
    assert result.error.startswith("TimeoutError:")
    assert result.answer is None
    assert result.timing["first_content_seconds"] is not None


def test_stream_cleanup_timeout_preserves_provider_error():
    async def exercise():
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        class BlockedClose(StreamBoundary):
            async def aclose(self):
                close_started.set()
                await close_release.wait()

        stream = BlockedClose([(0, RuntimeError("provider failed before deadline"))])
        task = asyncio.create_task(generate_answer(
            "q", arm="no_rag", config=ComparisonConfig(stream=True, generation_timeout=0.02),
            llm=ModelBoundary(stream), retriever=None,
        ))
        try:
            await asyncio.wait_for(close_started.wait(), timeout=1)
            completed, _ = await asyncio.wait({task}, timeout=0.5)
            assert task in completed, "provider failure was blocked by cleanup"
            result = task.result()
            assert result.error == "RuntimeError: provider failed before deadline"
        finally:
            close_release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

    asyncio.run(exercise())


def test_stream_cleanup_self_cancellation_does_not_replace_provider_error():
    class SelfCancelledClose(StreamBoundary):
        async def aclose(self):
            raise asyncio.CancelledError("provider close cancelled itself")

    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=ComparisonConfig(stream=True),
        llm=ModelBoundary(SelfCancelledClose([(0, RuntimeError("original provider error"))])),
        retriever=None,
    ))
    assert result.generation_status == "failed"
    assert result.error == "RuntimeError: original provider error"


def test_stream_caller_cancellation_propagates_with_bounded_cleanup():
    async def exercise():
        reading = asyncio.Event()
        close_started = asyncio.Event()
        close_release = asyncio.Event()

        class BlockedClose(StreamBoundary):
            async def __anext__(self):
                reading.set()
                await asyncio.Event().wait()

            async def aclose(self):
                close_started.set()
                await close_release.wait()

        task = asyncio.create_task(generate_answer(
            "q", arm="no_rag", config=ComparisonConfig(stream=True, generation_timeout=10),
            llm=ModelBoundary(BlockedClose([])), retriever=None,
        ))
        try:
            await asyncio.wait_for(reading.wait(), timeout=1)
            task.cancel()
            await asyncio.wait_for(close_started.wait(), timeout=1)
            completed, _ = await asyncio.wait({task}, timeout=0.5)
            assert task in completed, "caller cancellation was blocked by cleanup"
            with pytest.raises(asyncio.CancelledError):
                task.result()
        finally:
            close_release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

    asyncio.run(exercise())


def test_stream_cleanup_cancel_resistance_is_bounded_and_late_error_is_collected():
    async def exercise():
        loop = asyncio.get_running_loop()
        old_handler = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        close_started = asyncio.Event()
        close_release = asyncio.Event()
        close_finished = asyncio.Event()
        close_cancelled = asyncio.Event()

        class ResistantClose(StreamBoundary):
            async def aclose(self):
                close_started.set()
                try:
                    try:
                        await close_release.wait()
                    except asyncio.CancelledError:
                        close_cancelled.set()
                        await close_release.wait()
                    raise RuntimeError("late cleanup failure")
                finally:
                    close_finished.set()

        stream = ResistantClose([(0, AIMessageChunk(content="late"))], delay=1)
        task = asyncio.create_task(generate_answer(
            "q", arm="no_rag", config=ComparisonConfig(stream=True, generation_timeout=0.02),
            llm=ModelBoundary(stream), retriever=None,
        ))
        try:
            await asyncio.wait_for(close_started.wait(), timeout=1)
            completed, _ = await asyncio.wait({task}, timeout=0.5)
            assert task in completed, "cleanup ignored cancellation and blocked the caller"
            assert task.result().error.startswith("TimeoutError:")
            await asyncio.wait_for(close_cancelled.wait(), timeout=1)
            close_release.set()
            await asyncio.wait_for(close_finished.wait(), timeout=1)
            await asyncio.sleep(0)
            gc.collect()
            await asyncio.sleep(0)
            assert unhandled == []
        finally:
            close_release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)
            await asyncio.sleep(0)
            loop.set_exception_handler(old_handler)

    asyncio.run(exercise())


def test_non_stream_preserves_answer_and_never_fabricates_first_content():
    result = asyncio.run(generate_answer(
        "q", arm="no_rag", config=ComparisonConfig(),
        llm=ModelBoundary(None), retriever=None,
    ))
    assert result.generation_status == "success"
    assert result.answer == "invoke-only answer"
    assert result.usage["generation_input_tokens"] == 7
    assert result.timing["first_content_seconds"] is None
    assert result.timing["generation_first_content_seconds"] is None


def test_stream_model_requests_usage_totals_from_openai_compatible_provider(monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    import backend.llm_config as llm_config

    captured = []

    def create_chat_model(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(llm_config, "create_chat_model", create_chat_model)
    create_generation_llm(stream_config(model="offline"))
    create_generation_llm(ComparisonConfig(model="offline"))
    assert captured[0]["stream_usage"] is True
    assert "stream_usage" not in captured[1]


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_freezes_stream_switch_in_manifest_and_fingerprint(tmp_path, monkeypatch, enabled):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    import scripts.run_rag_comparison as runner

    monkeypatch.setattr(runner, "_corpus_identity", lambda: {})
    monkeypatch.setattr(runner, "_source_hashes", lambda: {})
    monkeypatch.setattr(runner, "_git_identity", lambda: {})
    run_dir = tmp_path / "stream-run"
    fixture = Path(__file__).parent / "fixtures" / "rag_comparison.json"

    class OfflineDependencies(runner.PipelineDependencies):
        def create_llm(self, config):
            class Model(ModelBoundary):
                def astream(self, messages):
                    return StreamBoundary([(0, AIMessageChunk(content="streamed answer"))])
            return Model(None)

        def create_retriever(self, config):
            async def retrieve(question):
                return {"docs": []}
            return retrieve

    args = ["run", "--dataset", str(fixture), "--limit", "1", "--run-dir", str(run_dir),
            "--phase", "generate", "--model", "offline", "--judge-model", "offline"]
    if enabled:
        args.append("--stream")
    assert runner.main(args, dependencies=OfflineDependencies()) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert manifest["config"]["stream"] is enabled
    assert manifest["fingerprint_inputs"]["config"]["stream"] is enabled
    assert ComparisonConfig.model_validate(manifest["config"]).stream is enabled
    for record in checkpoint["records"]:
        assert record["answer"] == ("streamed answer" if enabled else "invoke-only answer")
        assert (record["timing"]["first_content_seconds"] is not None) is enabled
