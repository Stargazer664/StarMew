"""
RAG 评估模块 - 基于官方 Ragas 框架

使用 Ragas (RAG Assessment) 官方 SDK v0.4.x 进行评估。
https://docs.ragas.io/

Ragas 使用 LLM-as-Judge 模式，通过精心设计的标准化提示词对 RAG 系统进行评估。

核心指标：
- Faithfulness (忠实度):        回答是否基于检索上下文（检测幻觉）
- AnswerRelevancy (答案相关性):  回答是否切中问题核心
- ContextPrecision (上下文精确率): 检索结果的信噪比 + 排名质量
- ContextRecall (上下文召回率):   检索结果是否完整覆盖问题所需信息
- FactualCorrectness (事实正确性): 回答与标准答案的事实符合度

依赖：
    pip install ragas langchain-openai
"""

import os
import json
import math
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from numbers import Real
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from .llm_config import create_chat_model, resolve_llm_settings
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# 加载 .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

EVAL_MODEL = resolve_llm_settings(role="evaluation").model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

console = Console()

# ─── Ragas SDK 可用性检查 ───────────────────────────────────────────
_RAGAS_AVAILABLE = False
_RAGAS_VERSION = "not_installed"

try:
    import ragas
    from ragas import evaluate as ragas_evaluate
    from ragas import SingleTurnSample, EvaluationDataset

    # Ragas 0.4.x: 使用 legacy metrics（兼容 evaluate() / aevaluate() API）
    # collections metrics 改用 BaseMetric 基类，不兼容 evaluate()，
    # 需配合 @experiment 装饰器。为保持简洁，使用 legacy metrics。
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
    )

    _RAGAS_AVAILABLE = True
    _RAGAS_VERSION = ragas.__version__
except ImportError as e:
    console.print(f"[yellow]⚠ Ragas SDK 未完整安装 ({e})，将退回 LLM 模拟模式[/yellow]")


def _utcnow():
    """获取当前 UTC 时间"""
    return datetime.now(timezone.utc).isoformat()


# ─── 数据类 ──────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """单条评测样本"""
    question: str
    ground_truth: Optional[str] = None
    expected_docs: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalSample":
        return cls(
            question=data.get("question", data.get("user_input", "")),
            ground_truth=data.get("ground_truth", data.get("reference")),
            expected_docs=data.get("expected_docs", data.get("retrieved_contexts")),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "expected_docs": self.expected_docs,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """单条评测结果"""
    sample: EvalSample
    answer: str = ""
    retrieved_docs: List[dict] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    overall_score: Optional[float] = None
    rag_trace: Optional[dict] = None
    latency: float = 0.0
    error: Optional[str] = None
    ragas_version: str = ""
    ragas_native: Optional[bool] = None
    evaluation_status: str = "unknown"
    metric_status: Dict[str, str] = field(default_factory=dict)
    metric_errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.sample.question,
            "answer": self.answer,
            "retrieved_docs_count": len(self.retrieved_docs),
            "metrics": self.metrics,
            "overall_score": self.overall_score,
            "rag_trace": self.rag_trace,
            "latency": self.latency,
            "error": self.error,
            "ground_truth": self.sample.ground_truth,
            "ragas_version": self.ragas_version,
            "ragas_native": self.ragas_native,
            "evaluation_status": self.evaluation_status,
            "metric_status": self.metric_status,
            "metric_errors": self.metric_errors,
        }


# ─── Ragas 指标定义 ──────────────────────────────────────────────────

# 各指标类别和权重
_METRICS_DEFINITION = {
    "faithfulness": {
        "weight": 0.25,
        "description": "回答中有多少内容是基于检索上下文的（不是编造的幻觉）",
        "category": "generation",
        "needs_ground_truth": False,
        "needs_retrieved_contexts": True,
    },
    "answer_relevancy": {
        "weight": 0.25,
        "description": "回答与问题的相关程度",
        "category": "generation",
        "needs_ground_truth": False,
        "needs_retrieved_contexts": False,
    },
    "context_precision": {
        "weight": 0.25,
        "description": "检索到的上下文块与问题的相关程度（含排名质量）",
        "category": "retrieval",
        "needs_ground_truth": True,
        "needs_retrieved_contexts": True,
    },
    "context_recall": {
        "weight": 0.25,
        "description": "检索到的上下文覆盖问题答案的程度",
        "category": "retrieval",
        "needs_ground_truth": True,
        "needs_retrieved_contexts": True,
    },
    "factual_correctness": {
        "weight": 0.20,
        "description": "回答与标准答案的事实符合度",
        "category": "generation",
        "needs_ground_truth": True,
        "needs_retrieved_contexts": False,
    },
}


def is_valid_score(value) -> bool:
    """A missing/invalid observation must never become a plausible score."""
    return isinstance(value, Real) and not isinstance(value, bool) and 0 <= value <= 1 and math.isfinite(value)


def _applicable_metrics(question, answer, contexts, ground_truth):
    return {name for name, config in _METRICS_DEFINITION.items()
            if (name == 'factual_correctness' or bool(question.strip()))
            and (config['category'] != 'generation' or bool(answer.strip()))
            and (not config['needs_ground_truth'] or bool((ground_truth or '').strip()))
            and (not config['needs_retrieved_contexts'] or bool(contexts))}


def _successful_metrics(record):
    return {name: float(value) for name, value in record.metrics.items()
            if name in _METRICS_DEFINITION and is_valid_score(value)
            and not record.metric_errors.get(name)
            and record.metric_status.get(name, 'failed' if record.error else 'success') == 'success'}


def calculate_overall_score(metrics: Dict[str, Optional[float]]) -> Optional[float]:
    """按当前指标权重计算有效指标的归一化综合分。"""
    valid = {
        key: value
        for key, value in metrics.items()
        if key in _METRICS_DEFINITION
        and is_valid_score(value)
    }
    if not valid:
        return None
    total_weight = sum(_METRICS_DEFINITION[key]["weight"] for key in valid)
    if total_weight <= 0:
        return round(sum(valid.values()) / len(valid), 4)
    weighted = sum(
        valid[key] * _METRICS_DEFINITION[key]["weight"]
        for key in valid
    )
    return round(weighted / total_weight, 4)


# ─── Ragas 官方评估器 ───────────────────────────────────────────────

class RAGASEvaluator:
    """
    基于官方 Ragas SDK 的 RAG 评估器

    特性：
    - 使用 Ragas 官方标准化指标（非自定义 prompt）
    - 支持 LLM-as-Judge 模式（通过 LangchainLLMWrapper）
    - 支持单条和批量评测
    - 支持黑盒（直接测 Pipeline）和白盒（传入 results）模式
    - 退路机制：如果 Ragas 未安装，使用内置 LLM 模拟

    用法：
        evaluator = RAGASEvaluator()

        # 评估 RAG Pipeline
        result = evaluator.evaluate_rag_pipeline(sample, run_rag_graph)
        evaluator.print_summary()
    """

    def __init__(self):
        self._llm = None          # LangchainLLMWrapper
        self._embeddings = None   # HuggingFaceEmbeddings
        self._rag_llm = None      # 原始 LangChain LLM（退路模式）
        self.history: List[EvalResult] = []

    # ── LLM & Embedding 初始化 ─────────────────────────────────────

    def _get_ragas_llm(self):
        """
        获取 Ragas LangChain LLM 包装器。

        当前兼容 OpenAI 协议的自定义端点无法稳定处理 llm_factory 创建的
        InstructorLLM 结构化请求，因此使用 Ragas 的 LangChain 包装器。
        """
        if self._llm is None and _RAGAS_AVAILABLE and EVAL_MODEL:
            try:
                from ragas.llms import LangchainLLMWrapper

                raw_llm = create_chat_model(
                    role="evaluation",
                    model=EVAL_MODEL,
                    temperature=0,
                )
                self._llm = LangchainLLMWrapper(raw_llm, bypass_n=True)
                console.print(f"[dim]✓ Ragas LLM 初始化完成: {EVAL_MODEL} (LangChain wrapper)[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ 初始化 Ragas LLM 失败: {e}[/yellow]")
        return self._llm

    def _get_ragas_embeddings(self):
        """
        获取 Ragas 评估用 Embeddings

        使用本地 HuggingFace BGE-m3（与 RAG 系统共用同一模型），
        具备 embed_query 方法，兼容 Ragas legacy metrics
        """
        if self._embeddings is None and _RAGAS_AVAILABLE:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                console.print(f"[dim]✓ Ragas Embeddings 初始化完成 (本地 {EMBEDDING_MODEL})[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠ 初始化 Ragas Embeddings 失败: {e}[/yellow]")
        return self._embeddings

    def _get_fallback_llm(self):
        """获取退路 LLM（LangChain init_chat_model，用于非 Ragas 模式）"""
        if self._rag_llm is None and EVAL_MODEL:
            try:
                self._rag_llm = create_chat_model(
                    role="evaluation",
                    model=EVAL_MODEL,
                    temperature=0,
                )
            except Exception as e:
                console.print(f"[yellow]⚠ 初始化退路 LLM 失败: {e}[/yellow]")
        return self._rag_llm

    # ── 核心评估方法 ──────────────────────────────────────────────

    def _evaluate_with_ragas(
        self,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """
        使用官方 Ragas SDK 评估单条数据

        Returns:
            {metric_name: score} 字典，分数范围 [0, 1]
        """
        llm = self._get_ragas_llm()
        applicable = _applicable_metrics(question, answer, retrieved_contexts, ground_truth)
        embeddings = self._get_ragas_embeddings() if 'answer_relevancy' in applicable else None

        if not llm:
            raise RuntimeError("Ragas LLM 初始化失败")
        if 'answer_relevancy' in applicable and embeddings is None:
            raise RuntimeError("Ragas Embeddings 初始化失败")

        # 构建 Ragas Sample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=ground_truth or "",
            retrieved_contexts=retrieved_contexts if retrieved_contexts else [],
        )

        dataset = EvaluationDataset(samples=[sample])

        # 挑选适用指标（根据是否有 ground_truth / contexts）
        metrics = []

        # Ragas 0.4.x 指标需要 llm 参数
        if 'faithfulness' in applicable:
            metrics.append(Faithfulness(llm=llm))
        if 'answer_relevancy' in applicable:
            metrics.append(AnswerRelevancy(llm=llm, embeddings=embeddings))

        # 需要检索上下文的指标
        if 'context_precision' in applicable:
            metrics.append(ContextPrecision(llm=llm))

        # 需要 ground_truth 的指标
        if 'context_recall' in applicable:
            metrics.append(ContextRecall(llm=llm))
        if 'factual_correctness' in applicable:
            metrics.append(FactualCorrectness(llm=llm))
        if not metrics:
            return {}

        try:
            result = ragas_evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                show_progress=False,
            )

            # 从 Ragas EvaluationResult 中提取分数
            scores = {}
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for col in df.columns:
                    val = df[col].iloc[0]
                    if col not in ('user_input', 'response', 'reference', 'retrieved_contexts'):
                        scores[col] = val
            elif isinstance(result, dict):
                for k, v in result.items():
                    scores[k] = v

            return scores

        except Exception as e:
            raise RuntimeError(f"Ragas evaluate() 调用失败: {e}")

    def _evaluate_with_fallback_llm(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[dict],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """
        退路模式：用 LLM 模拟 Ragas 指标（当 Ragas SDK 不可用时）
        """
        import re

        llm = self._get_fallback_llm()
        if not llm:
            raise RuntimeError("Fallback evaluation model unavailable")

        contexts = [doc.get("text", "")[:500] for doc in (retrieved_docs or [])[:3]]
        docs_text = "\n".join([
            f"[文档 {i+1}] {c}..."
            for i, c in enumerate(contexts)
        ]) if contexts else "（无检索结果）"

        ground_truth_section = f"\n【标准答案】\n{ground_truth}" if ground_truth else ""

        prompt = f"""你是一个专业的 RAG 系统评估员。请评估以下系统的表现。

【问题】
{question}

【回答】
{answer}

【检索到的上下文】
{docs_text}
{ground_truth_section}

请按以下标准给出 0-1 的分数：
1. faithfulness: 回答是否严格基于上下文（无幻觉）
2. answer_relevancy: 回答是否切中问题核心
3. context_precision: 上下文是否与问题相关
4. context_recall: 上下文是否完整覆盖答案所需信息
5. factual_correctness: 回答与标准答案的符合度（如提供了标准答案）

请输出JSON格式：
{{"faithfulness": 0.x, "answer_relevancy": 0.x, "context_precision": 0.x, "context_recall": 0.x, "factual_correctness": 0.x}}

只输出JSON："""

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        if not match:
            raise ValueError("Invalid fallback evaluation output")
        scores = json.loads(match.group())
        # Preserve missing/invalid values for the common validator; never clamp or fill.
        return {key: scores.get(key) for key in _METRICS_DEFINITION}

    def evaluate(
        self, question: str, answer: str, retrieved_docs: List[dict],
        ground_truth: Optional[str] = None, rag_trace: Optional[dict] = None,
    ) -> dict:
        return self._evaluate(question, answer, retrieved_docs, ground_truth, rag_trace)[0]

    def _evaluate(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[dict],
        ground_truth: Optional[str] = None,
        rag_trace: Optional[dict] = None,
    ) -> tuple[dict, EvalResult]:
        """
        评估单条 RAG 结果（返回 dict，兼容旧 API）

        Args:
            question:       用户问题
            answer:         生成的回答
            retrieved_docs: 检索到的文档列表 [{"text": "...", "filename": "...", ...}, ...]
            ground_truth:   标准答案
            rag_trace:      RAG 追踪元数据

        Returns:
            包含 metrics、overall_score 等字段的字典
        """
        start_time = time.time()

        contexts = [doc['text'] for doc in (retrieved_docs or []) if isinstance(doc.get('text'), str) and doc['text'].strip()]
        applicable = _applicable_metrics(question, answer, contexts, ground_truth)
        result = {
            "timestamp": _utcnow(),
            "question": question[:100],
            "answer_length": len(answer),
            "contexts_count": len(contexts),
            "metrics": {},
            "overall_score": None,
            "ragas_style": True,
            "ragas_version": _RAGAS_VERSION,
            "ragas_native": _RAGAS_AVAILABLE,
            "retrieval_mode": rag_trace.get("retrieval_mode") if rag_trace else None,
            "rerank_applied": rag_trace.get("rerank_applied") if rag_trace else False,
            "auto_merge_applied": rag_trace.get("auto_merge_applied") if rag_trace else False,
        }

        try:
            if not applicable:
                metrics = {}
            elif _RAGAS_AVAILABLE:
                # ─── 使用官方 Ragas SDK ────────────────────────────
                metrics = self._evaluate_with_ragas(
                    question=question,
                    answer=answer,
                    retrieved_contexts=contexts,
                    ground_truth=ground_truth,
                )
            else:
                # ─── 退路模式 ───────────────────────────────────────
                metrics = self._evaluate_with_fallback_llm(
                    question=question,
                    answer=answer,
                    retrieved_docs=retrieved_docs,
                    ground_truth=ground_truth,
                )
            if not isinstance(metrics, dict):
                raise ValueError("Invalid evaluation result")
        except Exception as e:
            result["error"] = f"evaluation_failed: {type(e).__name__}"
            metrics = {}

        # 规范化指标名（Ragas 返回的 key 可能略有差异）
        normalized = {}
        metric_name_map = {
            "faithfulness": ["faithfulness"],
            "answer_relevancy": ["answer_relevancy", "answerrelevancy", "response_relevancy"],
            "context_precision": ["context_precision", "contextprecision", "llm_context_precision_with_reference"],
            "context_recall": ["context_recall", "contextrecall", "llm_context_recall"],
            # Ragas 0.4.x 将 FactualCorrectness 的模式写入列名，例如
            # ``factual_correctness(mode=f1)``，需要兼容该返回键。
            "factual_correctness": [
                "factual_correctness",
                "factualcorrectness",
                "factual_correctness(mode=f1)",
            ],
        }
        statuses, errors = {}, {}
        for std_name, aliases in metric_name_map.items():
            normalized[std_name] = None
            if std_name not in applicable:
                statuses[std_name] = 'not_applicable'
                continue
            alias = next((name for name in aliases if name in metrics), None)
            value = metrics.get(alias) if alias else None
            if is_valid_score(value):
                normalized[std_name] = float(value)
                statuses[std_name] = 'success'
            else:
                statuses[std_name] = 'failed'
                errors[std_name] = result.get('error') or ('missing_score' if value is None else 'invalid_score')

        result["metrics"] = normalized
        result['metric_status'], result['metric_errors'] = statuses, errors
        succeeded = sum(status == 'success' for status in statuses.values())
        result['evaluation_status'] = ('not_applicable' if not applicable else
                                       'success' if not errors else 'partial' if succeeded else 'failed')
        if errors and not result.get('error'):
            result['error'] = 'one_or_more_metrics_failed'

        result["overall_score"] = calculate_overall_score(normalized)

        result["latency"] = round(time.time() - start_time, 2)

        # 写入历史
        sample = EvalSample(question=question, ground_truth=ground_truth)
        eval_result = EvalResult(
            sample=sample,
            answer=answer,
            retrieved_docs=retrieved_docs or [],
            metrics=normalized,
            overall_score=result["overall_score"],
            rag_trace=rag_trace,
            latency=result["latency"],
            error=result.get("error"),
            ragas_version=_RAGAS_VERSION,
            ragas_native=_RAGAS_AVAILABLE,
            evaluation_status=result['evaluation_status'],
            metric_status=statuses,
            metric_errors=errors,
        )
        self.history.append(eval_result)

        return result, eval_result

    def evaluate_sample(
        self,
        sample: EvalSample,
        answer: str,
        retrieved_docs: List[dict],
        rag_trace: Optional[dict] = None,
    ) -> EvalResult:
        """
        评估单条样本

        直接使用本次返回的记录，不能从并发共享的 history[-1] 取结果。
        """
        start_time = time.time()
        _, eval_result = self._evaluate(
            question=sample.question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            ground_truth=sample.ground_truth,
            rag_trace=rag_trace,
        )

        eval_result.sample = sample
        eval_result.answer = answer
        eval_result.retrieved_docs = retrieved_docs
        eval_result.rag_trace = rag_trace
        eval_result.latency = time.time() - start_time
        return eval_result

    def evaluate_rag_pipeline(
        self,
        sample: EvalSample,
        rag_pipeline_fn: Callable[[str], dict],
    ) -> EvalResult:
        """
        黑盒模式：通过 RAG Pipeline 自动评估

        Args:
            sample:           评测样本
            rag_pipeline_fn:  RAG Pipeline 函数，
                              接收 question: str，
                              返回 {answer, docs, rag_trace}

        Returns:
            EvalResult
        """
        start_time = time.time()
        try:
            rag_result = rag_pipeline_fn(sample.question)
            answer = rag_result.get("answer", "")
            docs = rag_result.get("docs", [])
            rag_trace = rag_result.get("rag_trace")

            return self.evaluate_sample(
                sample=sample,
                answer=answer,
                retrieved_docs=docs,
                rag_trace=rag_trace,
            )
        except Exception as e:
            return self._record_pipeline_failure(sample, e, time.time() - start_time)

    def _record_pipeline_failure(self, sample, error, latency=0.0):
        reason = f"pipeline_failed: {type(error).__name__}"
        record = EvalResult(sample=sample, metrics=dict.fromkeys(_METRICS_DEFINITION),
                            error=reason, latency=latency, evaluation_status='failed',
                            metric_status={key: 'failed' for key in _METRICS_DEFINITION},
                            metric_errors={key: reason for key in _METRICS_DEFINITION},
                            ragas_version=_RAGAS_VERSION)
        self.history.append(record)
        return record

    # ── 批量评估 ──────────────────────────────────────────────────

    async def evaluate_batch(
        self,
        samples: List[EvalSample],
        rag_pipeline_fn: Optional[Callable[[str], dict]] = None,
        agent_fn: Optional[Callable[[str], dict]] = None,
        max_concurrency: int = 3,
    ) -> List[EvalResult]:
        """
        异步批量评估

        Args:
            samples:           评测样本列表
            rag_pipeline_fn:   RAG Pipeline 函数
            agent_fn:          Agent 函数（优先于 rag_pipeline_fn）
            max_concurrency:   最大并发数

        Returns:
            EvalResult 列表
        """
        pipeline_fn = agent_fn or rag_pipeline_fn
        if not pipeline_fn:
            raise ValueError("必须提供 rag_pipeline_fn 或 agent_fn")

        async def evaluate_one(sample: EvalSample) -> EvalResult:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.evaluate_rag_pipeline(sample, pipeline_fn),
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]正在评测 {len(samples)} 条样本 (Ragas v{_RAGAS_VERSION})...",
                total=len(samples),
            )

            semaphore = asyncio.Semaphore(max_concurrency)

            async def sem_evaluate_one(sample: EvalSample) -> EvalResult:
                async with semaphore:
                    result = await evaluate_one(sample)
                    progress.advance(task)
                    return result

            tasks = [sem_evaluate_one(sample) for sample in samples]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                final_results.append(self._record_pipeline_failure(samples[i], r))
            else:
                final_results.append(r)

        return final_results

    def evaluate_batch_sync(
        self,
        samples: List[EvalSample],
        rag_pipeline_fn: Optional[Callable[[str], dict]] = None,
        agent_fn: Optional[Callable[[str], dict]] = None,
    ) -> List[EvalResult]:
        """同步批量评估（线程池）"""
        pipeline_fn = agent_fn or rag_pipeline_fn
        if not pipeline_fn:
            raise ValueError("必须提供 rag_pipeline_fn 或 agent_fn")

        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]正在评测 {len(samples)} 条样本 (Ragas v{_RAGAS_VERSION})...",
                total=len(samples),
            )

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self.evaluate_rag_pipeline, sample, pipeline_fn)
                    for sample in samples
                ]
                for future in futures:
                    results.append(future.result())
                    progress.advance(task)

        return results

    # ── 历史与统计 ────────────────────────────────────────────────

    def get_history(self, limit: int = 100) -> List[dict]:
        """获取评估历史"""
        return [r.to_dict() for r in self.history[-limit:]]

    def get_average_scores(self) -> dict:
        """只汇总有效观测，同时公开每项指标的分母和失败样本数。"""
        sums = {k: 0.0 for k in _METRICS_DEFINITION}
        counts = {k: 0 for k in _METRICS_DEFINITION}
        overall_scores = []
        status_counts = {}

        for r in self.history:
            valid = _successful_metrics(r)
            for k, v in valid.items():
                sums[k] += v
                counts[k] += 1
            overall = calculate_overall_score(valid)
            if overall is not None:
                overall_scores.append(overall)
            status_counts[r.evaluation_status] = status_counts.get(r.evaluation_status, 0) + 1

        avg = {
            f"avg_{k}": round(sums[k] / counts[k], 4) if counts[k] > 0 else None
            for k in _METRICS_DEFINITION
        }
        avg["avg_overall_score"] = round(
            sum(overall_scores) / len(overall_scores), 4
        ) if overall_scores else None
        avg["valid_counts"] = counts
        avg["valid_overall_count"] = len(overall_scores)
        avg["evaluation_status_counts"] = status_counts
        avg["total_evaluations"] = len(self.history)
        return avg

    def get_metrics_summary(self) -> dict:
        return {
            "ragas_native": _RAGAS_AVAILABLE,
            "ragas_version": _RAGAS_VERSION,
            "metrics": {
                name: {
                    "category": cfg["category"],
                    "description": cfg["description"],
                    "weight": cfg["weight"],
                    "needs_ground_truth": cfg["needs_ground_truth"],
                }
                for name, cfg in _METRICS_DEFINITION.items()
            },
            "total_evaluations": len(self.history),
        }

    def print_summary(self):
        """打印评测摘要表格"""
        table = Table(title=f"Ragas 评测摘要 (v{_RAGAS_VERSION})" if _RAGAS_AVAILABLE else "Ragas 评测摘要 (退路模式)")

        table.add_column("指标", style="cyan")
        table.add_column("类别", style="magenta")
        table.add_column("平均分", style="green")
        table.add_column("有效样本", style="green")
        table.add_column("权重", style="yellow")

        scores = self.get_average_scores()
        for name, cfg in _METRICS_DEFINITION.items():
            avg_key = f"avg_{name}"
            score = scores.get(avg_key)
            table.add_row(name, cfg["category"], f"{score:.4f}" if score is not None else "N/A",
                          str(scores['valid_counts'][name]), f"{cfg['weight']:.2f}")

        console.print(table)
        overall = scores['avg_overall_score']
        overall_text = f"{overall:.4f}" if overall is not None else "N/A"
        console.print(f"\n[bold]综合评分:[/bold] [green]{overall_text}[/green]")
        console.print(f"[bold]评测总数:[/bold] {scores['total_evaluations']}")
        if _RAGAS_AVAILABLE:
            console.print(f"[bold]Ragas 版本:[/bold] {_RAGAS_VERSION}")

    # ── 导出 ──────────────────────────────────────────────────────

    def export_results(
        self,
        filepath: str = "data/ragas_evaluation_results.json",
        include_history: bool = True,
    ) -> str:
        """导出评估结果为 JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "export_time": _utcnow(),
            "ragas_version": _RAGAS_VERSION,
            "ragas_native": _RAGAS_AVAILABLE,
            "eval_model": EVAL_MODEL,
            "summary": self.get_average_scores(),
            "metrics_summary": self.get_metrics_summary(),
            "evaluations": [r.to_dict() for r in self.history] if include_history else [],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)

        return filepath

    def export_csv(self, filepath: str = "data/ragas_evaluation_results.csv") -> str:
        """导出评估结果为 CSV"""
        import csv

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        headers = ["question", "answer_length", "contexts_count", "overall_score"]
        headers += list(_METRICS_DEFINITION.keys())
        headers += ["error", "latency", "evaluation_status", "metric_status", "metric_errors",
                    "ragas_native", "ragas_version"]

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in self.history:
                row = {
                    "question": r.sample.question[:100],
                    "answer_length": len(r.answer),
                    "contexts_count": len(r.retrieved_docs),
                    "overall_score": r.overall_score,
                    "error": r.error or "",
                    "latency": r.latency,
                    "evaluation_status": r.evaluation_status,
                    "metric_status": json.dumps(r.metric_status, ensure_ascii=False),
                    "metric_errors": json.dumps(r.metric_errors, ensure_ascii=False),
                    "ragas_native": r.ragas_native,
                    "ragas_version": r.ragas_version,
                }
                for metric in _METRICS_DEFINITION:
                    row[metric] = r.metrics.get(metric, "") if r.metrics.get(metric) is not None else ""
                writer.writerow(row)

        return filepath

    def generate_visualization_data(self) -> dict:
        """生成可视化数据"""
        if not self.history:
            return {"error": "No evaluation history"}

        scores = self.get_average_scores()

        metric_distributions = {}
        valid_rows = [_successful_metrics(r) for r in self.history]
        for metric in _METRICS_DEFINITION:
            values = [
                row[metric] for row in valid_rows if metric in row
            ]
            if values:
                metric_distributions[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values),
                }

        return {
            "ragas_version": _RAGAS_VERSION,
            "ragas_native": _RAGAS_AVAILABLE,
            "summary": scores,
            "metric_distributions": metric_distributions,
            "total_samples": len(self.history),
        }

    def clear_history(self):
        """清空评估历史"""
        self.history.clear()


# ─── 数据集管理器 ──────────────────────────────────────────────────

class EvalDatasetManager:
    """评测数据集管理器（兼容 JSON/CSV）"""

    @staticmethod
    def load_from_json(filepath: str) -> List[EvalSample]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [EvalSample.from_dict(item) for item in data]
        elif isinstance(data, dict) and "samples" in data:
            return [EvalSample.from_dict(item) for item in data["samples"]]
        else:
            raise ValueError(f"不支持的数据格式: {filepath}")

    @staticmethod
    def load_from_csv(filepath: str) -> List[EvalSample]:
        import csv
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                samples.append(EvalSample(
                    question=row.get("question", row.get("user_input", "")),
                    ground_truth=row.get("ground_truth", row.get("reference")),
                    metadata={"source": "csv"},
                ))
        return samples

    @staticmethod
    def save_to_json(samples: List[EvalSample], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "export_time": _utcnow(),
            "total_samples": len(samples),
            "samples": [s.to_dict() for s in samples],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def validate(samples: List[EvalSample]) -> List[str]:
        errors = []
        for i, sample in enumerate(samples):
            if not sample.question:
                errors.append(f"样本 {i}: 问题不能为空")
            if len(sample.question) < 3:
                errors.append(f"样本 {i}: 问题太短（少于3个字符）")
        return errors

    @staticmethod
    def split(
        samples: List[EvalSample],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple:
        import random
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]


# ─── 全局单例 ──────────────────────────────────────────────────────

evaluator = RAGASEvaluator()


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="StarMew RAG 评测工具 (Ragas SDK)")
    parser.add_argument("--dataset", "-d", type=str, help="评测数据集路径（JSON/CSV）")
    parser.add_argument("--output", "-o", type=str, default="data/ragas_results.json", help="输出路径")
    parser.add_argument("--export-csv", action="store_true", help="同时导出 CSV")
    parser.add_argument("--clear", action="store_true", help="清空历史后开始")
    parser.add_argument("--max-concurrency", "-c", type=int, default=3, help="最大并发数")

    args = parser.parse_args()

    if args.clear:
        evaluator.clear_history()
        console.print("[yellow]已清空历史记录[/yellow]")

    if not args.dataset:
        console.print("[yellow]请通过 --dataset 指定评测数据集[/yellow]")
        console.print("示例: uv run python -m backend.evaluation --dataset data/eval_samples.json")
        return

    samples = []
    if args.dataset.endswith(".json"):
        samples = EvalDatasetManager.load_from_json(args.dataset)
    elif args.dataset.endswith(".csv"):
        samples = EvalDatasetManager.load_from_csv(args.dataset)
    else:
        console.print(f"[red]不支持的文件格式: {args.dataset}[/red]")
        return

    console.print(f"[green]已加载 {len(samples)} 条评测样本[/green]")
    console.print(f"[dim]Ragas 版本: {_RAGAS_VERSION}, 原生模式: {_RAGAS_AVAILABLE}[/dim]")

    errors = EvalDatasetManager.validate(samples)
    if errors:
        console.print("[yellow]数据集验证警告:[/yellow]")
        for err in errors[:5]:
            console.print(f"  - {err}")

    # 导入 RAG Pipeline
    try:
        from .rag_pipeline import run_rag_graph
    except ImportError as e:
        console.print(f"[red]导入 RAG Pipeline 失败: {e}[/red]")
        return

    console.print(f"\n[cyan]开始评测（并发数: {args.max_concurrency}）...[/cyan]\n")

    asyncio.run(evaluator.evaluate_batch(
        samples=samples,
        rag_pipeline_fn=run_rag_graph,
        max_concurrency=args.max_concurrency,
    ))

    console.print("\n")
    evaluator.print_summary()

    output_path = evaluator.export_results(args.output)
    console.print(f"\n[green]✅ 评测结果已保存到: {output_path}[/green]")

    if args.export_csv:
        csv_path = evaluator.export_csv(args.output.replace(".json", ".csv"))
        console.print(f"[green]✅ CSV 格式已保存到: {csv_path}[/green]")


if __name__ == "__main__":
    main()
