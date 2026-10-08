"""Paired aggregation, cluster bootstrap, and offline report export."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contracts import ComparisonSample


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(record: dict[str, Any] | None, metric: str) -> float | None:
    if not record or record.get("generation_status") != "success":
        return None
    if record.get("metric_status", {}).get(metric) != "success":
        return None
    value = _finite_number(record.get("metrics", {}).get(metric))
    return value if value is not None and 0.0 <= value <= 1.0 else None


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
    }


def _cluster_bootstrap(
    details: list[dict[str, Any]], *, seed: int, samples: int
) -> list[float] | None:
    if not details or samples <= 0:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for detail in details:
        grouped[detail["cluster_id"]].append(detail["delta"])
    clusters = sorted(grouped)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(clusters) for _ in clusters]
        values = [delta for cluster in selected for delta in grouped[cluster]]
        draws.append(statistics.fmean(values))
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _paired_metric(
    metric: str,
    samples: list[ComparisonSample],
    indexed: dict[tuple[str, int, str], dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_by_id = {sample.id: sample for sample in samples}
    rounds: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample in samples:
        repeat_indices = sorted(
            {
                repeat
                for sample_id, repeat, _arm in indexed
                if sample_id == sample.id
            }
        )
        for repeat in repeat_indices:
            baseline = _metric_value(indexed.get((sample.id, repeat, "no_rag")), metric)
            rag = _metric_value(indexed.get((sample.id, repeat, "rag")), metric)
            if baseline is not None and rag is not None:
                rounds[sample.id].append(
                    {"repeat_index": repeat, "no_rag": baseline, "rag": rag}
                )

    details: list[dict[str, Any]] = []
    for sample_id, paired_rounds in sorted(rounds.items()):
        sample = sample_by_id[sample_id]
        baseline = statistics.fmean(item["no_rag"] for item in paired_rounds)
        rag = statistics.fmean(item["rag"] for item in paired_rounds)
        details.append(
            {
                "sample_id": sample_id,
                "question": sample.question,
                "domain": sample.metadata.domain,
                "question_type": sample.metadata.question_type,
                "cluster_id": sample.metadata.cluster_id or sample_id,
                "no_rag": baseline,
                "rag": rag,
                "delta": rag - baseline,
                "paired_rounds": len(paired_rounds),
            }
        )

    no_rag_values = [item["no_rag"] for item in details]
    rag_values = [item["rag"] for item in details]
    deltas = [item["delta"] for item in details]
    no_rag_mean = statistics.fmean(no_rag_values) if no_rag_values else None
    rag_mean = statistics.fmean(rag_values) if rag_values else None
    gain = statistics.fmean(deltas) if deltas else None
    relative = None if no_rag_mean in (None, 0.0) or gain is None else gain / no_rag_mean * 100
    cluster_count = len({item["cluster_id"] for item in details})
    result = {
        "no_rag_mean": no_rag_mean,
        "rag_mean": rag_mean,
        "absolute_gain": gain,
        "relative_gain_pct": relative,
        "ci95": _cluster_bootstrap(
            details, seed=seed, samples=bootstrap_samples
        ),
        "paired_question_count": len(details),
        "paired_round_count": sum(item["paired_rounds"] for item in details),
        "cluster_count": cluster_count,
        "ci_exploratory": bool(details) and cluster_count < 10,
        "wins": sum(item["delta"] > 0.02 for item in details),
        "ties": sum(abs(item["delta"]) <= 0.02 for item in details),
        "losses": sum(item["delta"] < -0.02 for item in details),
    }
    return result, details


def _paired_rate(
    samples: list[ComparisonSample],
    indexed: dict[tuple[str, int, str], dict[str, Any]],
    *,
    answerability: str,
) -> dict[str, Any]:
    values = {"no_rag": [], "rag": []}
    for sample in samples:
        if sample.metadata.answerability != answerability:
            continue
        repeats = {
            repeat for sample_id, repeat, _arm in indexed if sample_id == sample.id
        }
        for repeat in repeats:
            baseline = _metric_value(indexed.get((sample.id, repeat, "no_rag")), "abstention")
            rag = _metric_value(indexed.get((sample.id, repeat, "rag")), "abstention")
            if baseline is not None and rag is not None:
                values["no_rag"].append(baseline)
                values["rag"].append(rag)
    result: dict[str, Any] = {}
    for arm in ("no_rag", "rag"):
        arm_values = values[arm]
        result[arm] = statistics.fmean(arm_values) if arm_values else None
        result[f"{arm}_count"] = (
            f"{sum(value == 1.0 for value in arm_values)}/{len(arm_values)}"
            if arm_values
            else "0/0"
        )
    result["paired_round_count"] = len(values["no_rag"])
    return result


def summarize_pairs(
    records: list[dict[str, Any]],
    samples: list[ComparisonSample],
    *,
    seed: int = 42,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    indexed = {
        (str(record.get("sample_id")), int(record.get("repeat_index", 0)), str(record.get("arm"))): record
        for record in records
    }
    answerable_samples = [
        sample for sample in samples if sample.metadata.answerability == "answerable"
    ]
    factual, details = _paired_metric(
        "factual_correctness",
        answerable_samples,
        indexed,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )

    planned_repeat_count = max(
        (int(record.get("repeat_index", 0)) for record in records), default=0
    ) + 1
    invalid = 0
    for record in records:
        status = record.get("metric_status", {}).get("factual_correctness")
        if status == "failed" or (
            status == "success"
            and _metric_value(record, "factual_correctness") is None
        ):
            invalid += 1
    coverage_value = (
        factual["paired_question_count"] / len(answerable_samples)
        if answerable_samples
        else None
    )
    generation_failures = {
        arm: sum(
            record.get("arm") == arm and record.get("generation_status") == "failed"
            for record in records
        )
        for arm in ("no_rag", "rag")
    }

    latency_values = {"no_rag": [], "rag": []}
    for sample in samples:
        repeats = {
            repeat for sample_id, repeat, _arm in indexed if sample_id == sample.id
        }
        for repeat in repeats:
            pair = [indexed.get((sample.id, repeat, arm)) for arm in ("no_rag", "rag")]
            if all(record and record.get("generation_status") == "success" for record in pair):
                for arm, record in zip(("no_rag", "rag"), pair):
                    value = _finite_number(record.get("timing", {}).get("response_seconds"))
                    if value is not None:
                        latency_values[arm].append(value)

    token_totals: dict[str, dict[str, int]] = {
        "no_rag": defaultdict(int),
        "rag": defaultdict(int),
    }
    retrieval_complete = True
    for record in records:
        arm = record.get("arm")
        if arm not in token_totals:
            continue
        usage = record.get("usage", {})
        for field in (
            "generation_input_tokens",
            "generation_output_tokens",
            "retrieval_llm_tokens",
        ):
            value = usage.get(field)
            if isinstance(value, int):
                token_totals[arm][field] += value
        if arm == "rag" and not usage.get("retrieval_usage_complete", False):
            retrieval_complete = False

    groups: dict[str, dict[str, dict[str, Any]]] = {
        "domain": {},
        "question_type": {},
    }
    for field in ("domain", "question_type"):
        all_values = sorted({getattr(sample.metadata, field) for sample in samples})
        for value in all_values:
            selected = [item for item in details if item[field] == value]
            groups[field][value] = {
                "paired_question_count": len(selected),
                "absolute_gain": statistics.fmean(item["delta"] for item in selected)
                if selected
                else None,
            }

    warnings: list[str] = []
    if any(not sample.metadata.reference_reviewed for sample in samples):
        warnings.append("参考答案未完成人工审核，结论仅为探索性")
    if factual["no_rag_mean"] == 0 and factual["absolute_gain"] is not None:
        warnings.append("基线为 0，不计算相对提升")
    if coverage_value is not None and coverage_value < 0.95:
        warnings.append("主指标有效配对覆盖率低于 95%，结论仅为探索性")
    if factual["ci_exploratory"]:
        warnings.append("有效来源组少于 10，置信区间仅作探索")
    if not retrieval_complete:
        warnings.append("检索内 LLM usage 不完整，不计算完整费用提升比例")

    return {
        "coverage": {
            "planned_question_count": len(samples),
            "planned_repeat_count": planned_repeat_count,
            "planned_execution_count": len(samples) * planned_repeat_count * 2,
            "generation_failures": generation_failures,
            "metric_failed_or_invalid": invalid,
            "factual_correctness_paired_question_coverage": coverage_value,
        },
        "answerable": {
            "factual_correctness": factual,
            "over_refusal_rate": _paired_rate(
                samples, indexed, answerability="answerable"
            ),
        },
        "unanswerable": {
            "correct_abstention_rate": _paired_rate(
                samples, indexed, answerability="unanswerable"
            )
        },
        "performance": {
            "response_latency_seconds": {
                arm: _stats(values) for arm, values in latency_values.items()
            }
        },
        "cost": {
            "tokens": {arm: dict(values) for arm, values in token_totals.items()},
            "retrieval_usage_complete": retrieval_complete,
            "full_cost_gain_pct": None,
        },
        "groups": groups,
        "warnings": warnings,
        "pair_details": sorted(details, key=lambda item: item["delta"], reverse=True),
    }


def _display(value: Any) -> str:
    if value is None:
        return "失败/不适用"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def export_report(
    run_dir: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    fields = [
        "sample_id",
        "repeat_index",
        "arm",
        "generation_status",
        "answer",
        "metric_name",
        "metric_status",
        "metric_value",
        "response_seconds",
    ]
    with (directory / "pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metric_names = sorted(
                set(record.get("metrics", {})) | set(record.get("metric_status", {}))
            ) or [""]
            for metric in metric_names:
                value = _finite_number(record.get("metrics", {}).get(metric))
                writer.writerow(
                    {
                        "sample_id": record.get("sample_id", ""),
                        "repeat_index": record.get("repeat_index", 0),
                        "arm": record.get("arm", ""),
                        "generation_status": record.get("generation_status", ""),
                        "answer": record.get("answer", ""),
                        "metric_name": metric,
                        "metric_status": record.get("metric_status", {}).get(metric, ""),
                        "metric_value": "" if value is None else value,
                        "response_seconds": record.get("timing", {}).get("response_seconds", ""),
                    }
                )

    factual = summary["answerable"]["factual_correctness"]
    lines = [
        "# RAG 与无 RAG 配对测评报告",
        "",
        "## 主指标",
        "",
        f"- 无 RAG factual correctness F1：{_display(factual['no_rag_mean'])}",
        f"- RAG factual correctness F1：{_display(factual['rag_mean'])}",
        f"- 绝对增益：{_display(factual['absolute_gain'])}",
        f"- 相对增益：{_display(factual['relative_gain_pct'])}",
        f"- 95% 区间：{_display(factual['ci95'])}",
        f"- 有效配对题/轮：{factual['paired_question_count']}/{factual['paired_round_count']}",
        "",
        "## 警告与限制",
        "",
    ]
    lines.extend(f"- {warning}" for warning in summary.get("warnings", []))
    lines.extend(["", "## 改善与退步案例", ""])
    details = summary.get("pair_details", [])
    improvements = [item for item in details if item["delta"] > 0]
    regressions = [item for item in details if item["delta"] < 0]
    for item in improvements[:10]:
        lines.append(f"- 改善 {item['delta']:.4f}：{item['sample_id']} {item['question']}")
    for item in sorted(regressions, key=lambda value: value["delta"])[:10]:
        lines.append(f"- 退步 {item['delta']:.4f}：{item['sample_id']} {item['question']}")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rng = random.Random(42)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record.get("sample_id")), int(record.get("repeat_index", 0)))].append(record)
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for number, ((sample_id, repeat), pair) in enumerate(sorted(grouped.items()), start=1):
        shuffled = list(pair)
        rng.shuffle(shuffled)
        review_id = f"review-{number:04d}"
        for label, record in zip(("A", "B"), shuffled):
            review_rows.append(
                {"review_id": review_id, "answer_label": label, "answer": record.get("answer", "")}
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "answer_label": label,
                    "sample_id": sample_id,
                    "repeat_index": repeat,
                    "arm": record.get("arm"),
                }
            )
    with (directory / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "answer_label", "answer"])
        writer.writeheader()
        writer.writerows(review_rows)
    (directory / "review_key.json").write_text(
        json.dumps(key_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
