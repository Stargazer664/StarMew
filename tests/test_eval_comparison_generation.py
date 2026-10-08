import asyncio
import time

from backend.eval_comparison.contracts import ComparisonConfig
from backend.eval_comparison.generation import (
    COMMON_SYSTEM_PROMPT,
    RetrievalProcess,
    build_context,
    generate_answer,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {"model_name": "fake-model", "system_fingerprint": "fp"}
        self.usage_metadata = {"input_tokens": 12, "output_tokens": 3}


class FakeLLM:
    def __init__(self, content="合成回答"):
        self.content = content
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return FakeResponse(self.content)


class ExplodingRetriever:
    def __call__(self, _question):
        raise AssertionError("no_rag must not call retrieval")


def _slow_retriever(_question, _config):
    time.sleep(5)
    return {"docs": []}


def test_context_contains_only_selected_docs():
    docs = [
        {"filename": "a.md", "text": "甲证据"},
        {"filename": "b.md", "text": "乙证据"},
    ]
    text, used = build_context(docs, top_k=1, max_chars=100)
    assert "甲证据" in text
    assert "乙证据" not in text
    assert len(text) <= 100
    assert [doc["text"] for doc in used] == ["甲证据"]


def test_empty_retrieval_has_no_fabricated_context():
    assert build_context([], top_k=5, max_chars=8000) == ("", [])


def test_context_budget_includes_wrapper_and_records_chinese_truncation():
    text, used = build_context(
        [{"filename": "中文.md", "page_number": 2, "text": "甲" * 100}],
        top_k=5,
        max_chars=40,
    )
    assert len(text) == 40
    assert used[0]["truncated"] is True
    assert used[0]["original_length"] == 100
    assert used[0]["used_length"] == len(used[0]["text"])
    assert used[0]["text"] in text


def test_no_rag_never_calls_retriever_and_uses_shared_prompt():
    llm = FakeLLM()
    result = asyncio.run(
        generate_answer(
            "问题",
            arm="no_rag",
            config=ComparisonConfig(model="fake"),
            llm=llm,
            retriever=ExplodingRetriever(),
        )
    )
    assert result.generation_status == "success"
    assert result.answer == "合成回答"
    assert result.retrieved_docs == []
    assert llm.calls[0][0][1] == COMMON_SYSTEM_PROMPT
    assert "未提供参考资料" in llm.calls[0][1][1]


def test_generation_input_excludes_labels_and_unselected_docs():
    llm = FakeLLM([{"type": "text", "text": "内容块回答"}])

    async def retrieve(_question):
        return {
            "docs": [
                {"filename": "selected.md", "text": "允许证据"},
                {"filename": "excluded.md", "text": "排除证据"},
            ],
            "rag_trace": [{"step": "retrieve"}],
            "usage": {"retrieval_llm_tokens": 7},
        }

    result = asyncio.run(
        generate_answer(
            "问题",
            arm="rag",
            config=ComparisonConfig(model="fake", top_k=1),
            llm=llm,
            retriever=retrieve,
        )
    )
    prompt = llm.calls[0][1][1]
    assert result.answer == "内容块回答"
    assert "允许证据" in prompt
    assert "排除证据" not in prompt
    assert "ground_truth" not in prompt
    assert "expected_docs" not in prompt
    assert "metadata" not in prompt
    assert [item["text"] for item in result.used_contexts] == ["允许证据"]


def test_empty_retrieval_still_generates_but_retrieval_error_fails():
    async def empty(_question):
        return {"docs": []}

    async def broken(_question):
        raise RuntimeError("retrieval unavailable")

    ok = asyncio.run(
        generate_answer(
            "q", arm="rag", config=ComparisonConfig(), llm=FakeLLM(), retriever=empty
        )
    )
    failed = asyncio.run(
        generate_answer(
            "q", arm="rag", config=ComparisonConfig(), llm=FakeLLM(), retriever=broken
        )
    )
    assert ok.generation_status == "success"
    assert ok.retrieval_empty is True
    assert failed.generation_status == "failed"
    assert failed.answer is None
    assert "RuntimeError" in failed.error


def test_retrieval_failure_trace_is_not_treated_as_empty_success():
    llm = FakeLLM()

    async def failed_retrieval(_question):
        return {
            "docs": [],
            "rag_trace": [
                {"value": {"retrieval_mode": "failed", "rerank_error": "retrieve_failed"}}
            ],
        }

    result = asyncio.run(
        generate_answer(
            "q", arm="rag", config=ComparisonConfig(), llm=llm,
            retriever=failed_retrieval,
        )
    )

    assert result.generation_status == "failed"
    assert result.retrieval_empty is None
    assert result.answer is None
    assert result.rag_trace[0]["value"]["retrieval_mode"] == "failed"
    assert llm.calls == []


def test_empty_llm_answer_is_failure():
    result = asyncio.run(
        generate_answer(
            "q", arm="no_rag", config=ComparisonConfig(), llm=FakeLLM("  "), retriever=None
        )
    )
    assert result.generation_status == "failed"
    assert result.answer is None


def test_response_timing_contains_retrieval_and_generation_not_scoring():
    ticks = iter([0.0, 0.0, 2.0, 2.0, 5.0, 5.0])

    async def retrieve(_question):
        return {"docs": []}

    result = asyncio.run(
        generate_answer(
            "q",
            arm="rag",
            config=ComparisonConfig(),
            llm=FakeLLM(),
            retriever=retrieve,
            clock=lambda: next(ticks),
        )
    )
    assert result.timing == {
        "retrieval_seconds": 2.0,
        "generation_seconds": 3.0,
        "response_seconds": 5.0,
        "first_content_seconds": None,
        "generation_first_content_seconds": None,
        "evaluation_seconds": None,
    }


def test_retrieval_process_timeout_reaps_child():
    worker = RetrievalProcess(_slow_retriever)
    started = time.perf_counter()
    try:
        try:
            worker.retrieve("q", {}, timeout=0.2)
            raise AssertionError("timeout expected")
        except TimeoutError:
            pass
        assert time.perf_counter() - started < 3
        assert worker.is_alive is False
    finally:
        worker.close()
