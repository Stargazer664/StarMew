import asyncio
import math

from backend.eval_comparison.contracts import ComparisonSample
from backend.eval_comparison.scoring import (
    ABSTENTION_SYSTEM_PROMPT,
    AbstentionDecision,
    RagasScorer,
    score_answer,
    score_rag_context,
)


def _sample(answerability="answerable"):
    return ComparisonSample(
        id="q1",
        question="指定文档中的值是多少？",
        ground_truth="该值是 10。" if answerability == "answerable" else "文档未提供该值。",
        expected_docs=["doc.md"],
        metadata={
            "domain": "synthetic",
            "question_type": "fact",
            "answerability": answerability,
            "split": "dev",
            "cluster_id": "doc",
            "reference_reviewed": True,
            "reference_facts": [],
            "evidence": [],
        },
    )


class FakeScorer:
    def __init__(self, values=None, abstention=None):
        self.values = values or {}
        self.abstention = abstention
        self.calls = []

    async def score_metric(self, name, *, question, answer, reference, contexts=None):
        self.calls.append((name, question, answer, reference, contexts))
        value = self.values[name]
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            value = value()
        if asyncio.iscoroutine(value):
            value = await value
        return value

    async def judge_abstention(self, *, question, reference, answer):
        self.calls.append(("abstention", question, answer, reference, None))
        value = self.abstention
        if isinstance(value, BaseException):
            raise value
        return value


def test_metric_failure_does_not_erase_successful_metric():
    scorer = FakeScorer({
        "factual_correctness": 0.75,
        "answer_relevancy": RuntimeError("judge failed"),
    })
    results = asyncio.run(
        score_answer(
            _sample(), "回答", scorer=scorer,
            metrics=["factual_correctness", "answer_relevancy"],
        )
    )
    assert results["factual_correctness"].status == "success"
    assert results["factual_correctness"].value == 0.75
    assert results["answer_relevancy"].status == "failed"
    assert results["answer_relevancy"].value is None
    assert "RuntimeError" in results["answer_relevancy"].error


def test_nan_and_out_of_range_scores_fail_instead_of_becoming_zero():
    for invalid in (math.nan, -0.1, 1.1):
        result = asyncio.run(
            score_answer(
                _sample(), "回答",
                scorer=FakeScorer({"factual_correctness": invalid}),
                metrics=["factual_correctness"],
            )
        )["factual_correctness"]
        assert result.status == "failed"
        assert result.value is None


def test_timeout_is_failed_null():
    async def slow():
        await asyncio.sleep(1)
        return 0.5

    result = asyncio.run(
        score_answer(
            _sample(), "回答",
            scorer=FakeScorer({"factual_correctness": slow}),
            metrics=["factual_correctness"], timeout=0.01,
        )
    )["factual_correctness"]
    assert result.status == "failed"
    assert result.value is None
    assert "TimeoutError" in result.error


def test_unanswerable_does_not_run_factual_correctness():
    scorer = FakeScorer({"factual_correctness": AssertionError("must not run")})
    result = asyncio.run(
        score_answer(
            _sample("unanswerable"), "无法确定",
            scorer=scorer, metrics=["factual_correctness"],
        )
    )["factual_correctness"]
    assert result.status == "not_applicable"
    assert scorer.calls == []


def test_mixed_uncertainty_and_specific_claim_is_not_correct_abstention():
    scorer = FakeScorer(abstention={
        "abstained": True,
        "unsupported_specific_answer": True,
        "reason": "虽表示不确定，仍给出无依据数值 20",
    })
    result = asyncio.run(
        score_answer(
            _sample("unanswerable"), "不确定，但可能是 20。",
            scorer=scorer, metrics=["abstention"],
        )
    )["abstention"]
    assert result.status == "success"
    assert result.value == 0.0
    assert "20" in result.reason


def test_abstention_json_parse_failure_is_visible():
    result = asyncio.run(
        score_answer(
            _sample("unanswerable"), "不知道",
            scorer=FakeScorer(abstention="not-json"), metrics=["abstention"],
        )
    )["abstention"]
    assert result.status == "failed"
    assert result.value is None


def test_abstention_accepts_provider_chinese_alias_but_keeps_reason_required():
    scorer = FakeScorer(abstention={
        "拒答": True,
        "unsupported_specific_answer": False,
        "reason": "文档信息不足，回答明确拒答。",
    })
    result = asyncio.run(
        score_answer(
            _sample("unanswerable"), "无法确定",
            scorer=scorer, metrics=["abstention"],
        )
    )["abstention"]
    assert result.status == "success"
    assert result.value == 1.0
    assert result.reason == "文档信息不足，回答明确拒答。"


def test_abstention_schema_and_prompt_require_all_three_fields():
    schema = AbstentionDecision.model_json_schema()
    assert schema["required"] == ["abstained", "unsupported_specific_answer", "reason"]
    assert schema["properties"]["reason"]["minLength"] == 1
    assert "不可省略" in schema["properties"]["reason"]["description"]
    for field in ("abstained", "unsupported_specific_answer", "reason"):
        assert field in ABSTENTION_SYSTEM_PROMPT


def test_abstention_judge_receives_document_missing_semantics():
    class CaptureModel:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            self.messages = messages
            return AbstentionDecision(
                abstained=True,
                unsupported_specific_answer=False,
                reason="目标数值无法确定",
            )

    model = CaptureModel()
    scorer = RagasScorer(judge_model="fake")
    scorer._raw_llm = model
    scorer._ragas_llm = object()
    asyncio.run(scorer.judge_abstention(
        question="根据菜谱文档，这道菜的热量是多少？",
        reference="文档未提供热量。",
        answer="文档未提供热量，但给出了食材重量 300g。",
    ))

    system_prompt = model.messages[0][1]
    assert "文档未提供" in system_prompt
    assert "abstained 必须为 true" in system_prompt
    assert "与所问热量无关的食材重量" in system_prompt


def test_context_metrics_are_not_applicable_without_context_and_are_separate():
    scorer = FakeScorer({"faithfulness": 0.9})
    absent = asyncio.run(
        score_rag_context(_sample(), "回答", used_contexts=[], scorer=scorer)
    )
    assert absent["faithfulness"].status == "not_applicable"
    assert scorer.calls == []

    present = asyncio.run(
        score_rag_context(
            _sample(), "回答", used_contexts=["证据"], scorer=scorer,
            metrics=["faithfulness"],
        )
    )
    assert present["faithfulness"].value == 0.9
    assert scorer.calls[0][-1] == ["证据"]


def test_concurrent_scores_remain_bound_to_their_own_sample():
    async def run_one(sample_id, delay, value):
        sample = _sample()
        sample.id = sample_id

        async def delayed():
            await asyncio.sleep(delay)
            return value

        result = await score_answer(
            sample, "回答",
            scorer=FakeScorer({"factual_correctness": delayed}),
            metrics=["factual_correctness"],
        )
        return sample.id, result["factual_correctness"].value

    async def run_both():
        return await asyncio.gather(
            run_one("slow", 0.02, 0.2), run_one("fast", 0.0, 0.8)
        )

    actual = dict(asyncio.run(run_both()))
    assert actual == {"slow": 0.2, "fast": 0.8}
