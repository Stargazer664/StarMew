"""Independent per-metric scoring without shared evaluator history."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import time
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .contracts import ComparisonSample, MetricResult


ABSTENTION_SYSTEM_PROMPT = (
    "你是一个严格的中文问答审核员。仅判断待评回答是否拒答，"
    "以及它是否仍给出了参考答案无法支持的所问目标事实。"
    "若回答说文档未提供所问数据，或无法根据材料确定请求的事实，"
    "即使附带解释，abstained 必须为 true。"
    "例如询问热量时明确说文档未提供热量，就是拒答热量数值；"
    "与所问热量无关的食材重量不自动构成 unsupported_specific_answer。"
    "但若表示不确定后又给出无依据的所问热量数值，"
    "unsupported_specific_answer 必须为 true。"
    "reason 必须与两个布尔字段一致。"
    "输出必须且只能包含三个英文键：abstained、unsupported_specific_answer、reason。"
    "三个字段都不可省略，reason 必须是非空的简短中文理由。"
)


class AbstentionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    abstained: bool = Field(
        description="待评回答是否明确拒绝给出答案",
        validation_alias=AliasChoices("abstained", "拒答"),
    )
    unsupported_specific_answer: bool = Field(
        description="待评回答是否给出了参考答案无法支持的具体事实"
    )
    reason: str = Field(
        min_length=1,
        description="必须提供一条简短中文裁判理由，不可省略",
    )


def _safe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+", r"\1=<redacted>", text)
    text = re.sub(r"\b(?:sk|ak)-[A-Za-z0-9_-]{8,}\b", "<redacted>", text)
    return text[:1000]


def _validated_score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("metric score must be finite and within [0, 1]")
    return score


def _decision(value: Any) -> AbstentionDecision:
    if isinstance(value, AbstentionDecision):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return AbstentionDecision.model_validate(value)


async def _await_with_timeout(value: Any, timeout: float) -> Any:
    if not inspect.isawaitable(value):
        async def immediate():
            return value
        value = immediate()
    return await asyncio.wait_for(value, timeout=timeout)


async def _score_one(
    name: str,
    operation: Any,
    *,
    timeout: float,
    transform=_validated_score,
) -> MetricResult:
    started = time.perf_counter()
    try:
        raw = await _await_with_timeout(operation, timeout)
        value = transform(raw)
        return MetricResult(
            name=name,
            value=value,
            status="success",
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return MetricResult(
            name=name,
            value=None,
            status="failed",
            error=_safe_error(exc),
            elapsed_seconds=time.perf_counter() - started,
        )


async def score_answer(
    sample: ComparisonSample,
    answer: str,
    *,
    scorer: Any,
    metrics: list[str],
    timeout: float = 180.0,
) -> dict[str, MetricResult]:
    """Score answer metrics independently; no arm or trace is accepted."""
    results: dict[str, MetricResult] = {}
    is_answerable = sample.metadata.answerability == "answerable"
    for name in metrics:
        if name in {"factual_correctness", "answer_relevancy"}:
            if not is_answerable:
                results[name] = MetricResult(
                    name=name, status="not_applicable", reason="sample is unanswerable"
                )
                continue
            operation = scorer.score_metric(
                name,
                question=sample.question,
                answer=answer,
                reference=sample.ground_truth,
                contexts=None,
            )
            results[name] = await _score_one(name, operation, timeout=timeout)
            continue

        if name == "abstention":
            operation = scorer.judge_abstention(
                question=sample.question,
                reference=sample.ground_truth,
                answer=answer,
            )
            started = time.perf_counter()
            try:
                decision = _decision(await _await_with_timeout(operation, timeout))
                if is_answerable:
                    value = 1.0 if decision.abstained else 0.0
                else:
                    value = 1.0 if decision.abstained and not decision.unsupported_specific_answer else 0.0
                result = MetricResult(
                    name=name,
                    value=value,
                    status="success",
                    reason=decision.reason,
                    elapsed_seconds=time.perf_counter() - started,
                )
            except Exception as exc:
                result = MetricResult(
                    name=name,
                    value=None,
                    status="failed",
                    error=_safe_error(exc),
                    elapsed_seconds=time.perf_counter() - started,
                )
            results[name] = result
            continue

        results[name] = MetricResult(
            name=name, status="not_applicable", reason="unknown answer metric"
        )
    return results


async def score_rag_context(
    sample: ComparisonSample,
    answer: str,
    *,
    used_contexts: list[str],
    scorer: Any,
    metrics: list[str] | None = None,
    timeout: float = 180.0,
) -> dict[str, MetricResult]:
    """Run RAG-only diagnostics against exactly the contexts shown to the model."""
    requested = metrics or ["faithfulness", "context_precision", "context_recall"]
    results: dict[str, MetricResult] = {}
    for name in requested:
        if not used_contexts:
            results[name] = MetricResult(
                name=name, status="not_applicable", reason="no used context"
            )
            continue
        if name in {"context_precision", "context_recall"} and sample.metadata.answerability != "answerable":
            results[name] = MetricResult(
                name=name, status="not_applicable", reason="reference is not an answerable fact set"
            )
            continue
        operation = scorer.score_metric(
            name,
            question=sample.question,
            answer=answer,
            reference=sample.ground_truth,
            contexts=used_contexts,
        )
        results[name] = await _score_one(name, operation, timeout=timeout)
    return results


class RagasScorer:
    """Lazy Ragas 0.4.x adapter with one metric per SDK call."""

    def __init__(self, *, judge_model: str, timeout: float = 180.0):
        self.judge_model = judge_model
        self.timeout = timeout
        self._raw_llm = None
        self._ragas_llm = None
        self._embeddings = None

    def _ensure_llm(self):
        if self._ragas_llm is not None:
            return self._ragas_llm
        from ragas.llms import LangchainLLMWrapper
        from backend.llm_config import create_chat_model

        self._raw_llm = create_chat_model(
            role="evaluation",
            model=self.judge_model,
            temperature=0,
            timeout=self.timeout,
            max_retries=0,
        )
        self._ragas_llm = LangchainLLMWrapper(self._raw_llm, bypass_n=True)
        return self._ragas_llm

    def _ensure_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    async def score_metric(
        self,
        name: str,
        *,
        question: str,
        answer: str,
        reference: str,
        contexts: list[str] | None = None,
    ) -> float:
        from ragas import SingleTurnSample
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
        )

        llm = self._ensure_llm()
        factories = {
            "factual_correctness": lambda: FactualCorrectness(
                llm=llm, mode="f1", atomicity="low", coverage="low"
            ),
            "answer_relevancy": lambda: AnswerRelevancy(
                llm=llm, embeddings=self._ensure_embeddings()
            ),
            "faithfulness": lambda: Faithfulness(llm=llm),
            "context_precision": lambda: ContextPrecision(llm=llm),
            "context_recall": lambda: ContextRecall(llm=llm),
        }
        if name not in factories:
            raise ValueError(f"unsupported metric: {name}")
        metric = factories[name]()
        ragas_sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=reference,
            retrieved_contexts=contexts or [],
        )
        return float(await metric.single_turn_ascore(ragas_sample, timeout=self.timeout))

    async def judge_abstention(
        self, *, question: str, reference: str, answer: str
    ) -> AbstentionDecision:
        self._ensure_llm()
        structured = self._raw_llm.with_structured_output(AbstentionDecision)
        prompt = (
            f"【问题】\n{question}\n\n【参考答案】\n{reference}"
            f"\n\n【待评回答】\n{answer}"
        )
        return await structured.ainvoke(
            [("system", ABSTENTION_SYSTEM_PROMPT), ("human", prompt)]
        )
