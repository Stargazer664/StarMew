"""为已有 dishes Ragas 结果补算 factual_correctness，支持断点续跑。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 批处理不需要 LangSmith；关闭追踪可避免不可达服务拖慢退出。
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from backend.evaluation import (  # noqa: E402
    EVAL_MODEL,
    RAGASEvaluator,
    _METRICS_DEFINITION,
    calculate_overall_score,
    is_valid_score,
)
from ragas import SingleTurnSample  # noqa: E402
from ragas.metrics import FactualCorrectness  # noqa: E402


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "eval_dishes_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval_dishes_results_resumed.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_payload(source: Path, output: Path) -> dict:
    path = output if output.is_file() else source
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("evaluations"), list):
        raise ValueError(f"结果文件缺少 evaluations 数组: {path}")
    return payload


def valid_metrics(item: dict) -> dict:
    """Accept legacy observations only when no recorded failure contradicts them."""
    statuses = item.get("metric_status", {})
    errors = item.get("metric_errors", {})
    return {
        name: float(value)
        for name, value in item.get("metrics", {}).items()
        if name in _METRICS_DEFINITION and is_valid_score(value)
        and statuses.get(name, "failed" if item.get("error") else "success") == "success"
        and not errors.get(name)
    }


def update_summary(payload: dict) -> None:
    evaluations = payload.get("evaluations", [])
    valid_rows = [valid_metrics(item) for item in evaluations]
    summary = {"valid_counts": {}}
    for metric_name in _METRICS_DEFINITION:
        values = [metrics[metric_name] for metrics in valid_rows if metric_name in metrics]
        summary[f"avg_{metric_name}"] = (
            round(sum(values) / len(values), 4) if values else None
        )
        summary["valid_counts"][metric_name] = len(values)
    overall = [score for metrics in valid_rows
               if (score := calculate_overall_score(metrics)) is not None]
    summary["avg_overall_score"] = (
        round(sum(overall) / len(overall), 4) if overall else None
    )
    summary["valid_overall_count"] = len(overall)
    summary["evaluation_status_counts"] = {}
    for item in evaluations:
        status = item.get("evaluation_status", "unknown")
        counts = summary["evaluation_status_counts"]
        counts[status] = counts.get(status, 0) + 1
    summary["total_evaluations"] = len(evaluations)
    summary["factual_completed"] = summary["valid_counts"]["factual_correctness"]
    summary["factual_errors"] = sum(
        bool(item.get("metric_errors", {}).get("factual_correctness"))
        for item in evaluations
    )
    payload["summary"] = summary
    payload["export_time"] = utc_now()
    payload["eval_model"] = EVAL_MODEL
    payload["resume_mode"] = "factual_correctness_only"


def save_checkpoint(payload: dict, output: Path) -> None:
    update_summary(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(output)


async def score_one(
    index: int,
    item: dict,
    metric: FactualCorrectness,
    semaphore: asyncio.Semaphore,
) -> tuple[int, float | None, str | None, float]:
    sample = SingleTurnSample(
        user_input=item.get("question", ""),
        response=item.get("answer", ""),
        reference=item.get("ground_truth", "") or "",
    )
    started = time.time()
    try:
        async with semaphore:
            score = await asyncio.wait_for(
                metric.single_turn_ascore(sample),
                timeout=180,
            )
        if not is_valid_score(score):
            raise ValueError("FactualCorrectness returned an invalid score")
        return index, float(score), None, time.time() - started
    except Exception as exc:
        return index, None, type(exc).__name__, time.time() - started


async def run(source: Path, output: Path, concurrency: int) -> None:
    payload = load_payload(source, output)
    evaluations = payload["evaluations"]
    pending = [
        (index, item)
        for index, item in enumerate(evaluations)
        if "factual_correctness" not in valid_metrics(item)
        and item.get("metric_status", {}).get("factual_correctness") != "not_applicable"
    ]
    completed_before = sum("factual_correctness" in valid_metrics(item) for item in evaluations)
    not_applicable = sum(item.get("metric_status", {}).get("factual_correctness") == "not_applicable"
                         for item in evaluations)
    print(
        f"断点状态: 已完成 {completed_before}/{len(evaluations)}，"
        f"不适用 {not_applicable}，待处理 {len(pending)}",
        flush=True,
    )
    if not pending:
        save_checkpoint(payload, output)
        print(f"无需续跑，结果已完整: {output}", flush=True)
        return

    llm = RAGASEvaluator()._get_ragas_llm()
    if llm is None:
        raise RuntimeError("Ragas LLM 初始化失败")
    metric = FactualCorrectness(llm=llm)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    tasks = [
        asyncio.create_task(score_one(index, item, metric, semaphore))
        for index, item in pending
    ]

    finished = completed_before
    successful = completed_before
    errors = 0
    for future in asyncio.as_completed(tasks):
        index, score, error, elapsed = await future
        item = evaluations[index]
        item.setdefault("metrics", {})["factual_correctness"] = score
        metric_status = item.setdefault("metric_status", {})
        metric_status["factual_correctness"] = "failed" if error else "success"
        metric_errors = item.setdefault("metric_errors", {})
        if error:
            metric_errors["factual_correctness"] = error
            errors += 1
        else:
            metric_errors.pop("factual_correctness", None)
            successful += 1
        previous_error = item.get("error")
        known_aggregate_error = isinstance(previous_error, str) and (
            previous_error == "one_or_more_metrics_failed"
            or previous_error.startswith("evaluation_failed:")
        )
        fully_recovered = not metric_errors and all(
            metric_status.get(name) == "not_applicable"
            or (metric_status.get(name) == "success" and is_valid_score(item["metrics"].get(name)))
            for name in _METRICS_DEFINITION
        )
        if known_aggregate_error and fully_recovered:
            item["error"] = None
        item["overall_score"] = calculate_overall_score(valid_metrics(item))
        has_failure = bool(item.get("error") or metric_errors) or "failed" in metric_status.values()
        item["evaluation_status"] = (
            "failed" if item["overall_score"] is None else "partial" if has_failure else "success"
        )
        item["factual_correctness_latency"] = round(elapsed, 2)
        save_checkpoint(payload, output)
        finished += 1
        state = f"score={score:.4f}" if score is not None else f"error={error}"
        print(
            f"[{finished}/{len(evaluations)}] index={index + 1} {state} "
            f"耗时={elapsed:.1f}s 成功={successful} 错误={errors}",
            flush=True,
        )

    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)
    print(f"结果已保存: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="续跑 dishes factual_correctness")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run(args.source.resolve(), args.output.resolve(), args.concurrency))


if __name__ == "__main__":
    main()
