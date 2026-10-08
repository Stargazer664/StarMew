"""Read an existing comparison run and export auditable, offline resume metrics.

No model, retrieval service, database, .env, or network is used. Recall requires
metadata.relevance_annotations_complete=true; otherwise expected_docs supplies
expected-source coverage only. Output always goes to a separate directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import posixpath
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.eval_comparison.contracts import ComparisonSample, load_samples
from backend.eval_comparison.reporting import summarize_pairs


METRICS = (
    "expected_source_coverage_at_5", "expected_source_coverage_at_10",
    "recall_at_5", "recall_at_10", "hit_at_5", "hit_at_10", "mrr_at_10",
)
INPUT_NAMES = ("manifest.json", "dataset.json", "checkpoint.json")


def _normalise_source(value: Any) -> str:
    """Keep full path and case: basename/suffix/substring matching is unsafe."""
    if not isinstance(value, str) or not value.strip():
        return ""
    text = unicodedata.normalize("NFKC", value.strip()).replace("\\", "/")
    return posixpath.normpath(text)


def _question_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _stats(values: list[float]) -> dict[str, float | None]:
    def percentile(probability: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = (len(ordered) - 1) * probability
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {"mean": _mean(values), "p50": percentile(0.5), "p95": percentile(0.95)}


def _source_aliases(doc: Any) -> set[str]:
    if not isinstance(doc, dict):
        return set()
    metadata = doc.get("metadata") or {}
    values = [doc.get("source"), doc.get("doc_id"), doc.get("filename")]
    if isinstance(metadata, dict):
        values.append(metadata.get("source"))
    return {normal for value in values if (normal := _normalise_source(value))}


def evaluate_retrieval(expected_docs: list[str], retrieved_docs: list[dict],
                       ks: tuple[int, ...] = (5, 10)) -> dict[str, Any]:
    """Compute exact source coverage on returned candidate positions, without I/O.

    Coverage is not named recall here: completeness of relevance labels belongs
    to the caller. Missing labels or unidentifiable candidates produce nulls.
    A successfully retrieved empty list has zero coverage, hit, and MRR.
    """
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("ks 必须包含正整数")
    expected = {_normalise_source(value) for value in expected_docs} - {""}
    result = {"status": "success", "candidate_count": len(retrieved_docs), "mrr_at_10": None}
    for k in ks:
        result.update({f"coverage_at_{k}": None, f"hit_at_{k}": None})
    if not expected:
        result["status"] = "missing_annotations"
        return result
    aliases = [_source_aliases(doc) for doc in retrieved_docs[:max(10, max(ks))]]
    if any(not alias for alias in aliases):
        result["status"] = "unidentifiable_candidates"
        return result
    if any("/" in identifier for identifier in expected):
        for doc, alias in zip(retrieved_docs, aliases):
            filename = _normalise_source(doc.get("filename"))
            if (filename and "/" not in filename and not expected.intersection(alias)
                    and not any("/" in identifier for identifier in alias)):
                result["status"] = "identity_mapping_required"
                return result
    for k in ks:
        found = set().union(*aliases[:k]) if aliases else set()
        result[f"coverage_at_{k}"] = len(expected & found) / len(expected)
        result[f"hit_at_{k}"] = float(bool(expected & found))
    result["mrr_at_10"] = next(
        (1.0 / rank for rank, alias in enumerate(aliases[:10], start=1) if expected & alias), 0.0
    )
    return result


def _retrieval_failure(record: dict[str, Any]) -> bool:
    if record.get("retrieval_status") == "failed":
        return True
    if any(isinstance(item, dict) and item.get("stage") == "retrieval"
           and item.get("status") == "failed" for item in record.get("attempts", [])):
        return True
    for step in record.get("rag_trace", []):
        value = step.get("value", step) if isinstance(step, dict) else None
        if isinstance(value, dict) and value.get("retrieval_mode") == "failed":
            return True
    return False


def _retrieval_round(sample: ComparisonSample, record: dict[str, Any] | None) -> dict[str, Any]:
    status = "success"
    docs = record.get("retrieved_docs") if record else None
    if sample.metadata.answerability == "unanswerable":
        status = "unanswerable_not_applicable"
    elif record and _retrieval_failure(record):
        status = "retrieval_failed"
    elif not record or record.get("generation_status") in {"pending", "running"}:
        status = "retrieval_unavailable"
    elif not isinstance(docs, list):
        status = "retrieval_unavailable"
    elif record.get("generation_status") != "success" and record.get("retrieval_empty") is None:
        status = "retrieval_unavailable"
    result = {"status": status, "candidate_count": len(docs) if isinstance(docs, list) else None}
    result.update({name: None for name in METRICS})
    if status != "success":
        return result

    ranking = evaluate_retrieval(sample.expected_docs, docs)
    result["status"] = ranking["status"]
    if ranking["status"] != "success":
        return result
    complete = getattr(sample.metadata, "relevance_annotations_complete", False) is True
    for cutoff in (5, 10):
        covered = ranking[f"coverage_at_{cutoff}"]
        result[f"expected_source_coverage_at_{cutoff}"] = covered
        result[f"recall_at_{cutoff}"] = covered if complete else None
        result[f"hit_at_{cutoff}"] = ranking[f"hit_at_{cutoff}"]
    result["mrr_at_10"] = ranking["mrr_at_10"]
    return result


def _read_run(run_dir: Path) -> tuple[dict, list[ComparisonSample], list[dict], dict[str, str]]:
    raw = {name: (run_dir / name).read_bytes() for name in INPUT_NAMES}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    manifest = json.loads(raw["manifest.json"].decode("utf-8-sig"))
    checkpoint = json.loads(raw["checkpoint.json"].decode("utf-8-sig"))
    samples = load_samples(run_dir / "dataset.json")
    records = checkpoint.get("records")
    if not isinstance(records, list) or not samples:
        raise ValueError("dataset 必须非空，checkpoint.records 必须为数组")
    sample_ids = [sample.id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("dataset 存在重复 sample id")
    if any(not _question_key(sample.question) for sample in samples):
        raise ValueError("dataset 存在空问题")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("checkpoint record 必须为对象")
        key = (record.get("sample_id"), record.get("repeat_index", 0), record.get("arm"))
        if key[0] not in sample_ids or key[2] not in {"rag", "no_rag"}:
            raise ValueError("checkpoint 存在未知 sample_id 或 arm")
        if type(key[1]) is not int or key[1] < 0:
            raise ValueError("repeat_index 必须为非负整数")
        if key in seen:
            raise ValueError(f"checkpoint 存在重复执行键: {key}")
        seen.add(key)
    declared_hash = manifest.get("dataset_sha256")
    if declared_hash and declared_hash != hashes["dataset.json"]:
        raise ValueError("dataset SHA256 与 manifest 不一致")
    if manifest.get("fingerprint") and checkpoint.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("checkpoint 与 manifest fingerprint 不一致")
    return manifest, samples, records, hashes


def _groups(samples: list[ComparisonSample]) -> tuple[dict[str, list[ComparisonSample]], dict[str, str]]:
    groups: dict[str, list[ComparisonSample]] = defaultdict(list)
    for sample in samples:
        groups[_question_key(sample.question)].append(sample)
    canonical = {sample.id: group[0].id for group in groups.values() for sample in group}
    return dict(groups), canonical


def _answer_metrics(samples: list[ComparisonSample], records: list[dict], groups: dict,
                    canonical: dict[str, str], seed: int) -> dict:
    """Reuse the existing scorer outputs and paired reporting, never score anew."""
    collapsed_samples = []
    for group in groups.values():
        sample = group[0].model_copy(deep=True)
        sample.metadata.reference_reviewed = all(item.metadata.reference_reviewed for item in group)
        collapsed_samples.append(sample)
    round_map: dict[tuple[str, int], int] = {}
    next_round = Counter()
    collapsed_records = []
    for record in records:
        key = (record["sample_id"], record.get("repeat_index", 0))
        identifier = canonical[record["sample_id"]]
        if key not in round_map:
            round_map[key] = next_round[identifier]
            next_round[identifier] += 1
        collapsed_records.append({**record, "sample_id": identifier, "repeat_index": round_map[key]})
    summary = summarize_pairs(collapsed_records, collapsed_samples, seed=seed)
    return {
        "method": {
            "implementation": "backend.eval_comparison.reporting.summarize_pairs",
            "factual_weighting": "paired_rounds_then_unique_questions",
            "refusal_weighting": "paired_rounds",
            "note": "事实 F1 仅使用生成和评分均成功的可回答题配对，先题内轮次平均；"
                    "拒答率沿用既有按有效配对轮计数口径，不能把轮数当独立题数；未重跑裁判。",
        },
        "answerable": summary["answerable"], "unanswerable": summary["unanswerable"],
        "coverage": summary["coverage"], "warnings": summary["warnings"],
        "pair_details": summary["pair_details"],
    }


def _latency(records: list[dict], canonical: dict[str, str], plan: list[dict]) -> dict:
    fields = {"response_seconds", "retrieval_seconds", "generation_seconds", "evaluation_seconds",
              "first_content_seconds", "generation_first_content_seconds"}
    fields.update(key for record in records for key in (record.get("timing") or {}))
    def summarize(selected: list[dict]) -> dict:
        by_arm = {}
        for arm in ("no_rag", "rag"):
            arm_records = [record for record in selected if record["arm"] == arm]
            by_arm[arm] = {}
            for field in sorted(fields):
                values: dict[str, list[float]] = defaultdict(list)
                for record in arm_records:
                    value = _number((record.get("timing") or {}).get(field))
                    if value is not None and record.get("generation_status") == "success":
                        values[canonical[record["sample_id"]]].append(value)
                count = sum(map(len, values.values()))
                by_arm[arm][field] = {
                    **_stats([statistics.fmean(rounds) for rounds in values.values()]),
                    "question_count": len(values), "observed_round_count": count,
                    "missing_or_failed_round_count": len(arm_records) - count,
                }
        return by_arm

    def key(record: dict) -> tuple:
        return record.get("sample_id"), record.get("repeat_index", 0), record.get("arm")

    indexed = {key(record): record for record in records}
    excluded_keys = set()
    excluded_arms = set()
    exclusions = []
    for request in plan:
        record = indexed.get(key(request))
        if not record or record.get("generation_status") != "success" or record["arm"] in excluded_arms:
            continue
        excluded_keys.add(key(record))
        excluded_arms.add(record["arm"])
        exclusions.append({"sample_id": record["sample_id"], "repeat_index": record.get("repeat_index", 0),
                           "arm": record["arm"]})
    by_arm = summarize(records)
    subsequent = summarize([record for record in records if key(record) not in excluded_keys]) if plan else {}
    first_count = sum(by_arm[arm]["first_content_seconds"]["observed_round_count"] for arm in by_arm)
    generation_count = sum(by_arm[arm]["generation_first_content_seconds"]["observed_round_count"] for arm in by_arm)
    return {
        "unit": "seconds", "weighting": "successful_rounds_then_unique_questions",
        "note": "仅成功生成且有有限非负 timing 数值的轮次；题内平均后报告均值/分位数，"
                "分位数是逐题平均延迟的分布，不是逐请求延迟 SLO；未强制双臂配对。",
        "by_arm": by_arm,
        "subsequent_requests_by_arm": subsequent,
        "startup_exclusions": exclusions,
        "startup_reporting_status": "plan_order_applied" if plan else "plan_unavailable",
        "startup_reporting_note": "全量与后续进程样本并列：按 manifest.plan 顺序剔除每臂首个生成成功请求，"
                                  "不按耗时排序，缺少 timing 也剔除该请求；仅称后续进程样本，非严格热机/冷机实验。"
                                  "恢复运行或执行顺序改变时，plan 顺序不能证明实际启动顺序；缺少 plan 时不生成第二组。",
        "first_content": {
            "status": "collected" if first_count else "not_collected",
            "observed_round_count": first_count,
            "generation_only_observed_round_count": generation_count,
            "note": "first_content_seconds 为请求开始到首个可见正文（RAG 含检索）；"
                    "generation_first_content_seconds 为模型生成开始到首正文；缺失不以总耗时替代。",
        },
    }


def summarize_run(run_dir: Path) -> dict:
    manifest, samples, records, hashes = _read_run(run_dir)
    groups, canonical = _groups(samples)
    indexed = {(r["sample_id"], r.get("repeat_index", 0), r["arm"]): r for r in records}
    repeats: dict[str, set[int]] = defaultdict(set)
    for item in manifest.get("plan", []):
        if item.get("arm") == "rag" and item.get("sample_id") in canonical:
            repeats[item["sample_id"]].add(int(item.get("repeat_index", 0)))
    for record in records:
        repeats[record["sample_id"]].add(record.get("repeat_index", 0))
    fallback_repeat = int(manifest.get("config", {}).get("repeat", 1))
    if fallback_repeat < 1:
        raise ValueError("manifest.config.repeat 必须大于零")
    rows = []
    exclusions = Counter()
    depths = []
    observed_depths = []
    all_rounds = []
    annotation_conflicts = []
    for group in groups.values():
        rounds = []
        for sample in group:
            for repeat in sorted(repeats[sample.id] or range(fallback_repeat)):
                record = indexed.get((sample.id, repeat, "rag"))
                item = _retrieval_round(sample, record)
                item.update({"sample_id": sample.id, "repeat_index": repeat})
                rounds.append(item)
                if item["status"] != "success":
                    exclusions[item["status"]] += 1
                if item["status"] == "success":
                    depths.append(item["candidate_count"])
                if item["status"] not in {"retrieval_failed", "retrieval_unavailable", "unanswerable_not_applicable"}:
                    observed_depths.append(item["candidate_count"])
        valid = [item for item in rounds if item["status"] == "success"]
        annotations = {tuple(sorted({_normalise_source(x) for x in s.expected_docs})) for s in group}
        complete = all(getattr(s.metadata, "relevance_annotations_complete", False) is True for s in group)
        if len(annotations) > 1 or len({s.metadata.answerability for s in group}) > 1:
            annotation_conflicts.append(group[0].id)
            complete = False
        row = {
            "sample_id": group[0].id, "sample_ids": [s.id for s in group], "question": group[0].question,
            "reference_reviewed": all(s.metadata.reference_reviewed for s in group),
            "relevance_annotations_complete": complete,
            "expected_docs": sorted({x for s in group for x in s.expected_docs}),
            "split": sorted({s.metadata.split for s in group}),
            "valid_round_count": len(valid), "planned_round_count": len(rounds),
            "excluded_rounds": dict(Counter(item["status"] for item in rounds if item["status"] != "success")),
        }
        for metric in METRICS:
            values = [item[metric] for item in valid if item[metric] is not None]
            row[metric] = _mean(values) if complete or not metric.startswith("recall") else None
        row["observed_candidate_min"] = min((item["candidate_count"] for item in valid), default=None)
        row["observed_candidate_max"] = max((item["candidate_count"] for item in valid), default=None)
        rows.append(row)
        all_rounds.extend(rounds)

    answers = _answer_metrics(samples, records, groups, canonical, int(manifest.get("config", {}).get("seed", 42)))
    warnings = list(answers["warnings"])
    if any(not row["relevance_annotations_complete"] for row in rows):
        warnings.append("相关文档未全部显式标为完整标注：主指标为预期来源覆盖率；Recall 仅在完整标注子集报告。")
    if observed_depths and min(observed_depths) < 10:
        warnings.append("部分返回列表少于 10 个候选，@10 仅截取实际返回列表，不能声称完成十候选检索实验。")
    if any(sample.metadata.split != "test" for sample in samples):
        warnings.append("包含非 test 划分，不能作为独立测试集成绩。")
    if len(groups) < len(samples):
        warnings.append("相同规范化问题已合并为一题；重复 ID 的执行作为题内轮次，不能扩大独立样本数。")
    if annotation_conflicts:
        warnings.append("重复问题存在预期来源或可回答性冲突：相关组不报告 Recall；回答指标沿用该组首条题目口径，仅供探索。")
    if exclusions:
        warnings.append("检索缺失、失败、无标注、来源不可识别及不适用轮次已单列排除，未作为零分。")
    if exclusions["unanswerable_not_applicable"]:
        warnings.append("不可回答题的 expected_docs 仅表示应查阅来源，不代表包含答案的相关文档；"
                        "负例不进入主检索质量分母，只参与拒答质量指标，同时保留延迟统计。")
    if exclusions["identity_mapping_required"]:
        warnings.append("标注使用路径而候选只有文件名，缺少可验证的来源标识映射；相关指标置空，不能据此解释为检索零分。")
    if not depths:
        warnings.append("没有可用的检索标注配对，检索指标为空。")
    latency = _latency(records, canonical, manifest.get("plan") or [])
    if latency["first_content"]["status"] == "not_collected":
        warnings.append("首内容延迟未采集，不能用 response_seconds 推断。")

    factual = answers["answerable"]["factual_correctness"]
    paired_details = {item["sample_id"]: item for item in answers["pair_details"]}
    for row in rows:
        pair = paired_details.get(row["sample_id"], {})
        row.update({"factual_no_rag": pair.get("no_rag"), "factual_rag": pair.get("rag"),
                    "factual_delta": pair.get("delta"), "factual_paired_rounds": pair.get("paired_rounds", 0)})
    result = {
        "schema_version": 1, "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir.resolve()), "input_sha256": hashes,
        "audit": {
            "dataset_rows": len(samples), "unique_question_count": len(groups),
            "source_cluster_count": len({s.metadata.cluster_id.strip() for s in samples if s.metadata.cluster_id.strip()}),
            "missing_cluster_rows": sum(not s.metadata.cluster_id.strip() for s in samples),
            "duplicate_question_rows": len(samples) - len(groups), "checkpoint_record_count": len(records),
            "reference_reviewed_rows": sum(s.metadata.reference_reviewed for s in samples),
            "reference_unreviewed_rows": sum(not s.metadata.reference_reviewed for s in samples),
            "split_counts": dict(Counter(s.metadata.split for s in samples)),
            "complete_annotation_question_count": sum(row["relevance_annotations_complete"] for row in rows),
            "annotation_conflicts": annotation_conflicts,
            "exploratory": bool(warnings) or factual["paired_question_count"] == 0,
            "question_identity": "Unicode NFKC + strip/collapse whitespace; similar paraphrases are not automatically deduplicated",
        },
        "retrieval": {
            "scope": "最终返回候选排序的来源/文档级指标；不是向量召回阶段的 Recall，也不是 used_contexts 的覆盖率。",
            "matching": "source/doc_id/filename/metadata.source；NFKC、路径分隔符及 ./ 规范化；完整标识精确且区分大小写，不取 basename，不作子串匹配。",
            "weighting": "valid_rounds_then_unique_questions",
            "definition": "前 K 个返回候选的位置保持不变；期望文档去重、命中文档只计一次；"
                          "覆盖率=命中期望来源数/期望来源数，Hit=至少命中一个，MRR=首个命中候选位置倒数；不足 K 不补候选。",
            "recall_eligibility": "metadata.relevance_annotations_complete 必须显式为 true；不能由 reference_reviewed 或 evidence 推断。",
            "metrics": {metric: _mean([row[metric] for row in rows if row[metric] is not None]) for metric in METRICS},
            "metric_question_counts": {metric: sum(row[metric] is not None for row in rows) for metric in METRICS},
            "valid_question_count": sum(row["valid_round_count"] > 0 for row in rows),
            "valid_round_count": len(depths), "planned_round_count": len(all_rounds),
            "unanswerable_question_count": sum(any(s.metadata.answerability == "unanswerable" for s in group)
                                               for group in groups.values()),
            "excluded_not_applicable": exclusions["unanswerable_not_applicable"],
            "excluded_rounds": dict(exclusions),
            "missing_annotation_question_count": sum(not row["expected_docs"] for row in rows),
            "observed_depth": {"min_candidates": min(observed_depths, default=None), "max_candidates": max(observed_depths, default=None),
                               "rounds_below_5": sum(x < 5 for x in observed_depths), "rounds_below_10": sum(x < 10 for x in observed_depths)},
        },
        "answer_metrics": answers, "latency": latency, "warnings": warnings,
        "questions": rows, "retrieval_rounds": all_rounds,
    }
    for name, digest in hashes.items():
        if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError("汇总期间输入发生变化，请在评估停止写入后重试")
    return result


def _display(value: Any) -> str:
    return "未采集/无有效数据" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def export(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resume_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = result["questions"]
    with (output_dir / "resume_metrics_questions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    audit, retrieval, answers = result["audit"], result["retrieval"], result["answer_metrics"]
    factual = answers["answerable"]["factual_correctness"]
    lines = ["# 简历量化指标：离线证据报告", "",
             f"- Run：{result['run_id']}",
             f"- 结论性质：{'探索性，不能直接当作正式独立测试结论' if audit['exploratory'] else '具备已声明的审核与测试条件，仍需人工复核'}",
             f"- 数据集 {audit['dataset_rows']} 行；去重问题 {audit['unique_question_count']} 题；重复问题行 {audit['duplicate_question_rows']}。",
             f"- 来源簇 {audit['source_cluster_count']} 个；缺少簇标注 {audit['missing_cluster_rows']} 行；去重题数不代表样本统计独立。",
             f"- 参考答案审核：已审 {audit['reference_reviewed_rows']}，未审 {audit['reference_unreviewed_rows']}；划分 {audit['split_counts']}。",
             "", "## 返回候选的来源级指标", "", retrieval["scope"], "", retrieval["matching"], "", retrieval["definition"], "",
             f"- 有效题/轮：{retrieval['valid_question_count']}/{retrieval['valid_round_count']}；计划 RAG 轮：{retrieval['planned_round_count']}。",
             f"- 排除轮次：{retrieval['excluded_rounds']}；缺少来源标注的题数：{retrieval['missing_annotation_question_count']}。",
             f"- 不可回答题 {retrieval['unanswerable_question_count']} 题，检索不适用 {retrieval['excluded_not_applicable']} 轮；只参与拒答质量指标，并保留延迟统计。",
             f"- 观测候选深度：{retrieval['observed_depth']}。不足 K 的结果只反映实际返回列表。", "",
             "| 指标 | 逐题平均值 | 有效题数 |", "|---|---:|---:|"]
    for metric in METRICS:
        label = metric.replace("expected_source_coverage_at_", "预期来源覆盖率@").replace("recall_at_", "完整标注子集 Recall@").replace("hit_at_", "Hit@").replace("mrr_at_", "MRR@")
        lines.append(f"| {label} | {_display(retrieval['metrics'][metric])} | {retrieval['metric_question_counts'][metric]} |")
    lines.extend(["", "## 配对回答指标", "", answers["method"]["note"], "",
                  f"- F1 无 RAG / RAG：{_display(factual['no_rag_mean'])} / {_display(factual['rag_mean'])}。",
                  f"- F1 绝对增益：{_display(factual['absolute_gain'])}；相对增益 %：{_display(factual['relative_gain_pct'])}。",
                  f"- 有效配对题/轮：{factual['paired_question_count']}/{factual['paired_round_count']}；来源簇：{factual['cluster_count']}；95% 区间：{factual['ci95']}。",
                  f"- 可回答题过度拒答率：{answers['answerable']['over_refusal_rate']}。",
                  f"- 不可回答题正确拒答率：{answers['unanswerable']['correct_abstention_rate']}。",
                  "", "## 已记录延迟", "", result["latency"]["note"], "", result["latency"]["first_content"]["note"], "",
                  result["latency"]["startup_reporting_note"], "",
                  f"- 剔除请求：{result['latency']['startup_exclusions']}；状态：{result['latency']['startup_reporting_status']}。", "",
                  "| 样本集 | 分组 / timing 字段 | 均值 秒 | P50 秒 | P95 秒 | 有效题/轮 | 缺失或失败轮 |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for label, group in (("全量", result["latency"]["by_arm"]),
                         ("后续进程样本", result["latency"]["subsequent_requests_by_arm"])):
        for arm, fields in group.items():
            for field, stats in fields.items():
                if stats["observed_round_count"]:
                    lines.append(f"| {label} | {arm} / {field} | {_display(stats['mean'])} | {_display(stats['p50'])} | {_display(stats['p95'])} | {stats['question_count']}/{stats['observed_round_count']} | {stats['missing_or_failed_round_count']} |")
    lines.extend(["", "## 限制与缺失", ""] + [f"- {warning}" for warning in result["warnings"]])
    lines.extend(["", "## 输入完整性", "", "仅读取以下快照；不运行模型，不改写输入。", ""])
    lines.extend(f"- `{name}`：`{digest}`" for name, digest in result["input_sha256"].items())
    (output_dir / "resume_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线汇总现有 RAG comparison run 的简历量化证据")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    try:
        if output_dir == run_dir or output_dir.is_relative_to(run_dir):
            raise ValueError("output-dir 必须在 run-dir 之外，避免修改输入目录")
        print("昕，[1/3] 读取并校验评估快照", flush=True)
        result = summarize_run(run_dir)
        print(f"昕，[2/3] 已汇总 {result['audit']['unique_question_count']} 个去重问题、{result['audit']['source_cluster_count']} 个来源簇", flush=True)
        export(result, output_dir)
        print(f"昕，[3/3] 报告已保存：{output_dir}", flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"昕，错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
