"""Offline evidence Recall/Precision and binary answer Accuracy on frozen inputs.

Only Python's standard library is used: no .env, model, database, or network is
loaded. Qrels cover every indexed leaf chunk in each question's named sources;
other sources are irrelevant under that declared question scope. This is not a
measurement of unrestricted corpus-wide semantic relevance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import posixpath
import statistics
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES = ("dense", "hybrid")
ARMS = ("rag", "no_rag")
KS = (5, 10)
METRICS = tuple(f"{name}_at_{k}" for k in KS for name in ("recall", "precision", "hit")) + ("mrr_at_10",)
INPUTS = ("dataset", "corpus", "qrels", "retrieval", "answers", "accuracy")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{context}: 必须是非空字符串")
    return value


def normalize_text(value: str) -> str:
    """Exact identity after NFKC and whitespace removal, without fuzzy matching."""
    return "".join(unicodedata.normalize("NFKC", value).split())


def _source(value: Any, context: str) -> str:
    value = _text(value, context)
    return posixpath.normpath(unicodedata.normalize("NFKC", value.strip()).replace("\\", "/"))


def _sources(value: Any, context: str) -> set[str]:
    _require(isinstance(value, list) and bool(value), f"{context}: expected_docs 不能为空")
    result = {_source(item, context) for item in value}
    _require(len(result) == len(value), f"{context}: expected_docs 重复")
    return result


def _rows(value: Any, context: str) -> list[dict]:
    _require(isinstance(value, list), f"{context}: 必须是列表")
    _require(all(isinstance(row, dict) for row in value), f"{context}: 列表项必须是对象")
    return value


def _unique(rows: list[dict], fields: tuple[str, ...], context: str) -> dict[Any, dict]:
    result = {}
    for row in rows:
        parts = tuple(_text(row.get(field), f"{context}.{field}") for field in fields)
        key = parts[0] if len(parts) == 1 else parts
        _require(key not in result, f"{context}: 重复记录 {key}")
        result[key] = row
    return result


def _json_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"JSON 重复键: {key}")
        result[key] = value
    return result


def _read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    def invalid_constant(value: str) -> None:
        raise ValueError(f"JSON 非有限值: {value}")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_json_object,
                       parse_constant=invalid_constant)
    _require(isinstance(value, dict), f"{path.name}: JSON 根必须是对象")
    return value, hashlib.sha256(raw).hexdigest()


def _hash_matches(value: Any, expected: str, context: str) -> None:
    _require(value == expected, f"{context}: sha256 不匹配")


def _validate_dataset(dataset: dict) -> dict[str, dict]:
    samples = _unique(_rows(dataset.get("samples"), "dataset.samples"), ("id",), "dataset")
    _require(bool(samples), "dataset: 空题集")
    questions = set()
    for identifier, sample in samples.items():
        question = normalize_text(_text(sample.get("question"), f"dataset.{identifier}.question"))
        _require(question not in questions, "dataset: 重复问题不能作为独立样本")
        questions.add(question)
        _sources(sample.get("expected_docs"), f"dataset.{identifier}")
        metadata = sample.get("metadata")
        _require(isinstance(metadata, dict), f"dataset.{identifier}: 缺少 metadata")
        _require(metadata.get("answerability") in ("answerable", "unanswerable"),
                 f"dataset.{identifier}: answerability 未定义")
    return samples


def _validate_corpus(corpus: dict) -> dict[str, dict]:
    _require(corpus.get("schema_version") == 1, "corpus: schema_version 必须为 1")
    _text(corpus.get("collection"), "corpus.collection")
    index_hash = _text(corpus.get("index_identity_sha256"), "corpus.index_identity_sha256")
    _require(len(index_hash) == 64 and all(c in "0123456789abcdef" for c in index_hash),
             "corpus: index_identity_sha256 格式错误")
    chunks = _unique(_rows(corpus.get("chunks"), "corpus.chunks"), ("chunk_id",), "corpus")
    _require(bool(chunks), "corpus: 空语料")
    for identifier, chunk in chunks.items():
        _source(chunk.get("filename"), f"corpus.{identifier}.filename")
        _text(chunk.get("text"), f"corpus.{identifier}.text")
        _require(type(chunk.get("chunk_level")) is int and chunk["chunk_level"] == 3,
                 f"corpus.{identifier}: 仅支持 chunk_level=3 的叶块")
    return chunks


def _validate_qrels(qrels: dict, samples: dict, chunks: dict) -> dict[str, dict]:
    _require(qrels.get("schema_version") == 1, "qrels: schema_version 必须为 1")
    _require(qrels.get("human_reviewed") is False, "qrels: 本协议必须明确非人工复核")
    _require(qrels.get("review_method") == "ai_source_checked", "qrels: review_method 不匹配")
    items = _unique(_rows(qrels.get("samples"), "qrels.samples"), ("sample_id",), "qrels")
    positive_ids = {key for key, sample in samples.items()
                    if sample["metadata"]["answerability"] == "answerable"}
    _require(set(items) == positive_ids, "qrels: 必须恰好覆盖全部可答题，缺失/多余样本不可计零")
    for identifier, item in items.items():
        context = f"qrels.{identifier}"
        expected = _sources(item.get("expected_docs"), context)
        _require(expected == _sources(samples[identifier]["expected_docs"], context),
                 f"{context}: expected_docs 与 dataset 不一致")
        for flag in ("scope_complete", "reference_facts_verified"):
            _require(type(item.get(flag)) is bool, f"{context}: {flag} 必须为 bool")
        scope = {key for key, chunk in chunks.items() if _source(chunk["filename"], context) in expected}
        covered_sources = {_source(chunks[key]["filename"], context) for key in scope}
        _require(covered_sources == expected, f"{context}: 目标来源在 corpus 中缺失")
        judgments = _unique(_rows(item.get("judgments"), f"{context}.judgments"), ("chunk_id",), context)
        _require(set(judgments) == scope, f"{context}: judgments 必须完整覆盖目标来源的全部 chunk_id")
        relevant = set()
        for chunk_id, judgment in judgments.items():
            _require(type(judgment.get("relevant")) is bool, f"{context}: relevant 必须为 bool")
            _text(judgment.get("reason"), f"{context}.reason")
            quote = judgment.get("quote")
            _require(isinstance(quote, str), f"{context}: quote 必须为字符串")
            if judgment["relevant"]:
                _text(quote, f"{context}.quote")
                relevant.add(chunk_id)
            if quote.strip():
                _require(normalize_text(quote) in normalize_text(chunks[chunk_id]["text"]),
                         f"{context}: quote 不在对应 chunk 文本中")
        declared = item.get("relevant_chunk_ids")
        _require(isinstance(declared, list) and all(isinstance(x, str) for x in declared),
                 f"{context}: relevant_chunk_ids 必须为字符串列表")
        _require(len(declared) == len(set(declared)) and set(declared) == relevant,
                 f"{context}: relevant_chunk_ids 与逐项 judgments 不一致")
        if item["scope_complete"] and item["reference_facts_verified"]:
            _require(bool(relevant), f"{context}: 已验证可答题不能没有相关证据 chunk")
    return items


def _validate_candidate(candidate: dict, chunks: dict, context: str) -> str:
    identifier = _text(candidate.get("chunk_id"), f"{context}.chunk_id")
    _require(identifier in chunks, f"{context}: candidate 未知 chunk_id {identifier}")
    original = chunks[identifier]
    _require(_source(candidate.get("filename"), f"{context}.filename") ==
             _source(original["filename"], context), f"{context}: candidate filename 不匹配")
    text = _text(candidate.get("text"), f"{context}.text")
    _require(normalize_text(text) == normalize_text(original["text"]), f"{context}: candidate text 不匹配")
    _require(type(candidate.get("chunk_level")) is int and candidate["chunk_level"] == 3,
             f"{context}: candidate chunk_level 不匹配")
    if "id" in candidate and "id" in original:
        _require(type(candidate["id"]) is type(original["id"]) and candidate["id"] == original["id"],
                 f"{context}: candidate 数据库 id 不匹配")
    return identifier


def _validate_retrieval(retrieval: dict, corpus: dict, samples: dict, chunks: dict) -> dict:
    _require(retrieval.get("collection") == corpus["collection"], "retrieval: collection 不匹配")
    top_k = retrieval.get("top_k")
    _require(type(top_k) is int and top_k >= max(KS), "retrieval: 配置深度不足 10，不能报告 Recall@10")
    if "index_identity_sha256" in retrieval:
        _hash_matches(retrieval["index_identity_sha256"], corpus["index_identity_sha256"], "retrieval.index_identity_sha256")
    records = _unique(_rows(retrieval.get("records"), "retrieval.records"), ("sample_id", "mode"), "retrieval")
    for (identifier, mode), row in records.items():
        _require(identifier in samples and samples[identifier]["metadata"]["answerability"] == "answerable",
                 f"retrieval: {identifier} 不属于可答题集")
        _require(mode in MODES, f"retrieval: 未定义 mode {mode}")
        _require(row.get("status") in ("success", "failed", "pending", "running"), "retrieval: 未定义 status")
        docs = _rows(row.get("docs"), "retrieval.docs")
        _require(len(docs) <= top_k, "retrieval: 返回候选数超出声明 top_k")
        for candidate in docs:
            _validate_candidate(candidate, chunks, f"retrieval.{identifier}.{mode}.candidate")
    return records


def _validate_accuracy(answers: dict, accuracy: dict, samples: dict) -> tuple[dict, dict]:
    expected = {(identifier, arm) for identifier in samples for arm in ARMS}
    generations = _unique(_rows(answers.get("records"), "answers.records"), ("sample_id", "arm"), "answers")
    _require(set(generations) == expected, "answers: 必须恰好覆盖全部题目的 rag/no_rag 双组，禁止缺失/重复")
    _require(accuracy.get("human_reviewed") is False, "accuracy: 本协议必须明确非人工复核")
    _require(accuracy.get("review_method") == "two_ai_reviews_with_adjudication", "accuracy: review_method 不匹配")
    labels = _unique(_rows(accuracy.get("records"), "accuracy.records"), ("sample_id", "arm"), "accuracy")
    _require(set(labels) == expected, "accuracy: 必须恰好覆盖全部题目的双组，缺失/多余记录拒绝")
    for key, generation in generations.items():
        context = f"accuracy.{key[0]}.{key[1]}"
        _require(type(generation.get("repeat_index")) is int and generation["repeat_index"] == 0,
                 "answers: 本协议只支持每题每组一份原始生成，重复轮次不能算额外题")
        _require(generation.get("generation_status") in ("success", "failed", "pending", "running"),
                 "answers: generation_status 未定义")
        answer = generation.get("answer")
        _require(isinstance(answer, str), "answers: answer 必须是原字符串")
        label = labels[key]
        _hash_matches(label.get("answer_sha256"), hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                      f"{context}.answer_sha256")
        _text(label.get("reason"), f"{context}.reason")
        _require(type(label.get("reviewer_agreement")) is bool, f"{context}: reviewer_agreement 必须为 bool")
        positive = samples[key[0]]["metadata"]["answerability"] == "answerable"
        applicable = "correct" if positive else "correct_abstention"
        other = "correct_abstention" if positive else "correct"
        _require(applicable in label and other in label, f"{context}: 缺少判定字段")
        _require(label[other] is None, f"{context}: 非本题类型的判定字段必须为 null")
        _require(label[applicable] is None or type(label[applicable]) is bool,
                 f"{context}: {applicable} 必须为 bool 或 null")
        if generation["generation_status"] != "success":
            _require(label[applicable] is None, f"{context}: 失败/未完成生成不能拥有成功判定")
    return generations, labels


def evaluate_evidence(relevant_chunk_ids: list[str], candidate_chunk_ids: list[str]) -> dict:
    """Score already validated identities; duplicates keep rank but cannot add hits."""
    relevant = set(relevant_chunk_ids)
    _require(bool(relevant), "evidence: relevant_chunk_ids 不能为空")
    result = {"status": "success", "candidate_count": len(candidate_chunk_ids),
              "observed_depth": min(len(candidate_chunk_ids), 10),
              "duplicate_candidate_count": len(candidate_chunk_ids[:10]) - len(set(candidate_chunk_ids[:10])),
              "relevant_chunk_count": len(relevant)}
    for k in KS:
        found = len(set(candidate_chunk_ids[:k]) & relevant)
        result[f"recall_at_{k}"] = found / len(relevant)
        result[f"precision_at_{k}"] = found / k
        result[f"hit_at_{k}"] = float(found > 0)
    result["mrr_at_10"] = next((1 / rank for rank, key in enumerate(candidate_chunk_ids[:10], 1)
                                      if key in relevant), 0.0)
    return result


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sensitivity(accuracy: dict, labels: dict, samples: dict, by_arm: dict) -> dict | None:
    """Derive the optional RAG boundary-label interpretation from bound records."""
    field = "lenient_sensitivity_correct"
    alternatives = {key: row for key, row in labels.items() if field in row}
    if not alternatives:
        _require("sensitivity" not in accuracy, "accuracy.sensitivity: 顶层汇总缺少逐条替代标签")
        return None
    main = by_arm["rag"]["answerable"]
    _require(main["complete"], "accuracy.sensitivity: 需要完整的 RAG 可答题主判定")
    boundaries = []
    for (identifier, arm), row in alternatives.items():
        _require(arm == "rag" and samples[identifier]["metadata"]["answerability"] == "answerable",
                 "accuracy.sensitivity: 替代字段仅适用于 RAG 可答题")
        _require(type(row[field]) is bool and type(row["correct"]) is bool,
                 "accuracy.sensitivity: 替代标签与主标签必须为 bool")
        _require(row[field] >= row["correct"], "accuracy.sensitivity: 宽松标签不能比严格标签更低")
        if row[field] != row["correct"]:
            adjudication = row.get("adjudication") or {}
            _require(isinstance(adjudication, dict), "accuracy.sensitivity: adjudication 必须是对象")
            reason = adjudication.get("sensitivity_reason") or row.get("sensitivity_reason") or row["reason"]
            boundaries.append({"sample_id": identifier, "arm": arm, "strict_correct": row["correct"],
                               "lenient_correct": row[field], "strict_reason": row["reason"],
                               "lenient_reason": _text(reason, "accuracy.sensitivity.reason")})
    lenient = sum(row.get(field, row["correct"]) for (identifier, arm), row in labels.items()
                  if arm == "rag" and samples[identifier]["metadata"]["answerability"] == "answerable")
    result = {"scope": "仅已知标签解释变化，非置信区间", "arm": "rag", "answerability": "answerable",
              "strict_correct": main["correct"], "lenient_correct": lenient, "denominator": main["planned"],
              "strict_accuracy": main["accuracy"], "lenient_accuracy": lenient / main["planned"],
              "sample_ids": sorted(row["sample_id"] for row in boundaries),
              "boundary_records": sorted(boundaries, key=lambda row: row["sample_id"])}
    if "sensitivity" in accuracy:
        declared = accuracy["sensitivity"]
        _require(isinstance(declared, dict), "accuracy.sensitivity: 顶层汇总必须为对象")
        for key in ("strict_correct", "lenient_correct", "denominator"):
            _require(type(declared.get(key)) is int and declared[key] == result[key],
                     f"accuracy.sensitivity.{key}: 与逐条标签计数不符")
        for key in ("strict_accuracy", "lenient_accuracy"):
            _require(type(declared.get(key)) in (int, float) and abs(declared[key] - result[key]) <= 1e-12,
                     f"accuracy.sensitivity.{key}: 与逐条标签计算不符")
        identifiers = declared.get("sample_ids")
        _require(isinstance(identifiers, list) and all(isinstance(x, str) for x in identifiers)
                 and sorted(identifiers) == result["sample_ids"],
                 "accuracy.sensitivity.sample_ids: 与逐条边界标签不符")
    return result


def summarize(*, dataset_path: Path, corpus_path: Path, qrels_path: Path,
              retrieval_path: Path, answers_path: Path, accuracy_path: Path) -> dict:
    """Read each input exactly once, validate the frozen chain, return a report."""
    paths = dict(dataset=dataset_path, corpus=corpus_path, qrels=qrels_path,
                 retrieval=retrieval_path, answers=answers_path, accuracy=accuracy_path)
    values, hashes = {}, {}
    for name, path in paths.items():
        values[name], hashes[name] = _read(Path(path))
    dataset, corpus, qrels, retrieval, answers, accuracy = (values[name] for name in INPUTS)
    for name, field, target in (("qrels", "dataset_sha256", "dataset"), ("qrels", "corpus_sha256", "corpus"),
                                ("retrieval", "dataset_sha256", "dataset"), ("accuracy", "dataset_sha256", "dataset"),
                                ("accuracy", "answers_sha256", "answers")):
        _hash_matches(values[name].get(field), hashes[target], f"{name}.{field}")
    samples = _validate_dataset(dataset)
    chunks = _validate_corpus(corpus)
    relations = _validate_qrels(qrels, samples, chunks)
    retrievals = _validate_retrieval(retrieval, corpus, samples, chunks)
    generations, labels = _validate_accuracy(answers, accuracy, samples)
    questions = []
    by_mode = {}
    for mode in MODES:
        included, excluded = [], Counter()
        for identifier, relation in relations.items():
            row = retrievals.get((identifier, mode))
            if not relation["scope_complete"] or not relation["reference_facts_verified"]:
                status = "unverified_annotations"
            elif row is None:
                status = "missing_retrieval"
            elif row["status"] != "success":
                status = "retrieval_" + row["status"]
            else:
                status = "success"
            if status == "success":
                metrics = evaluate_evidence(relation["relevant_chunk_ids"], [c["chunk_id"] for c in row["docs"]])
                included.append(metrics)
            else:
                metrics = {"status": status, **dict.fromkeys(METRICS)}
                excluded[status] += 1
            questions.append({"sample_id": identifier, "question": samples[identifier]["question"],
                              "answerability": "answerable", "mode": mode, "arm": None, "retrieval": metrics})
        by_mode[mode] = {"planned_questions": len(relations), "eligible_questions": len(included),
                         "excluded": dict(excluded), "complete": bool(relations) and not excluded,
                         "metrics": {metric: _mean([row[metric] for row in included]) for metric in METRICS},
                         "observed_depth_counts": dict(Counter(row["observed_depth"] for row in included)),
                         "questions_with_duplicate_candidates": sum(row["duplicate_candidate_count"] > 0 for row in included)}
    by_arm = {}
    for arm in ARMS:
        by_arm[arm] = {}
        for answerability, field, metric in (("answerable", "correct", "accuracy"),
                                             ("unanswerable", "correct_abstention", "correct_abstention_rate")):
            selected = [key for key in labels if key[1] == arm and samples[key[0]]["metadata"]["answerability"] == answerability]
            scores = []
            for key in selected:
                label, generation = labels[key], generations[key]
                score = label[field]
                status = ("generation_" + generation["generation_status"] if generation["generation_status"] != "success"
                          else "unreviewed" if score is None else "reviewed")
                if status == "reviewed":
                    scores.append(score)
                questions.append({"sample_id": key[0], "question": samples[key[0]]["question"],
                                  "answerability": answerability, "mode": None, "arm": arm,
                                  "accuracy": {"status": status, "field": field, "value": score,
                                               "reason": label["reason"], "reviewer_agreement": label["reviewer_agreement"],
                                               "answer_sha256": label["answer_sha256"]}})
            by_arm[arm][answerability] = {"planned": len(selected), "reviewed": len(scores),
                                          "correct": sum(scores), "missing": len(selected) - len(scores),
                                          metric: _mean(scores), "complete": bool(selected) and len(scores) == len(selected)}
    clusters = {s["metadata"].get("cluster_id") for s in samples.values()} - {None, ""}
    positive_count = len(relations)
    audit = {"sample_count": len(samples), "deduplicated_question_count": len(samples),
             "answerable_count": positive_count, "unanswerable_count": len(samples) - positive_count,
             "source_cluster_count": len(clusters),
             "missing_cluster_count": sum(not s["metadata"].get("cluster_id") for s in samples.values()),
             "human_reviewed": False, "qrels_review_method": qrels["review_method"],
             "accuracy_review_method": accuracy["review_method"],
             "reviewer_disagreement_count": sum(not row["reviewer_agreement"] for row in labels.values()),
             "split": dataset.get("split", "unspecified"), "formal_holdout": dataset.get("formal_holdout") is True,
             "corpus_chunk_count": len(chunks), "qrels_judgment_count": sum(len(r["judgments"]) for r in relations.values()),
             "scope": "source-scoped corpus qrels", "collection": corpus["collection"],
             "index_identity_sha256": corpus["index_identity_sha256"]}
    complete = all(row["complete"] for row in by_mode.values()) and all(row["complete"] for arm in by_arm.values() for row in arm.values())
    answer_metrics = {"by_arm": by_arm}
    sensitivity = _sensitivity(accuracy, labels, samples, by_arm)
    if sensitivity is not None:
        answer_metrics["sensitivity"] = sensitivity
        if "policy_clarification" in accuracy:
            answer_metrics["policy_clarification"] = _text(accuracy["policy_clarification"], "accuracy.policy_clarification")
    return {"schema_version": 1, "collected_at": datetime.now(timezone.utc).isoformat(),
            "complete": complete, "input_sha256": hashes,
            "input_paths": {name: str(Path(path).resolve()) for name, path in paths.items()},
            "audit": audit, "retrieval": {"unit": "indexed_leaf_chunk", "by_mode": by_mode},
            "accuracy": answer_metrics, "questions": questions,
            "limitations": [
                "AI 复核，非人工标注；仍可能有标签误差。qrels 为逐来源审阅，Accuracy 为两路 AI 复核及分歧裁决。",
                "source-scoped corpus qrels：逐题完整审阅指定来源的全部索引 L3 chunk；题意限定指定来源，其他来源视为不相关。",
                "Recall 分母为本协议标记相关的唯一 chunk 数，不是文件数或 evidence span 数；不宣称无来源限制的全库语义召回。",
                "Precision@K 使用固定 K 分母；重复 chunk 仅命中一次且不压缩原始排名，返回不足 K 的空位也占分母。",
                "检索指标逐去重问题宏平均。失败/未完成/未验证标注排除且不计零；非完整时均值仅代表已观测子集。",
                "Accuracy 为可答题整题正确判定比例，负例只计正确拒答率；两者分开，不是事实 F1 的改名。",
                "题目共享来源簇，不假设统计独立；本次探索性题集不构成独立盲测或泛化性能证明。",
                "输入哈希验证数据身份，不能替代对 AI 标签正确性的人工审核。",
            ]}


def _format(value: float | None) -> str:
    return "未采集/不可用" if value is None else f"{value:.4f}"


def _markdown(report: dict) -> str:
    audit = report["audit"]
    lines = ["# 证据检索与整题准确率", "", "AI 复核，非人工标注；source-scoped corpus qrels。",
             f"样本：{audit['answerable_count']} 道可答题、{audit['unanswerable_count']} 道负例；"
             f"{audit['deduplicated_question_count']} 道去重问题、{audit['source_cluster_count']} 个来源簇。",
             f"完整性：{'全部计划记录可用' if report['complete'] else '部分记录不可用，仅报告已观测子集'}；"
             f"split={audit['split']}，formal_holdout={str(audit['formal_holdout']).lower()}。", "",
             "| 检索方式 | 有效/计划题数 | Recall@5 | Recall@10 | Precision@5 | Precision@10 | MRR@10 |", 
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, row in report["retrieval"]["by_mode"].items():
        scores = " | ".join(_format(row["metrics"][key]) for key in
                            ("recall_at_5", "recall_at_10", "precision_at_5", "precision_at_10", "mrr_at_10"))
        lines.append(f"| {mode} | {row['eligible_questions']}/{row['planned_questions']} | {scores} |")
    lines.extend(["", "Precision 使用固定 K 分母；Recall 以标注相关的唯一叶块为分母。", "",
                  "| 回答组 | 可答题整题 Accuracy | 已审/计划 | 正确拒答率 | 已审/计划 |",
                  "|---|---:|---:|---:|---:|"])
    for arm, groups in report["accuracy"]["by_arm"].items():
        yes, no = groups["answerable"], groups["unanswerable"]
        lines.append(f"| {arm} | {_format(yes['accuracy'])} | {yes['reviewed']}/{yes['planned']} | "
                     f"{_format(no['correct_abstention_rate'])} | {no['reviewed']}/{no['planned']} |")
    sensitivity = report["accuracy"].get("sensitivity")
    if sensitivity is not None:
        lines.extend(["", "## 标签解释敏感性", "",
                      f"RAG 可答题严格主结果：{sensitivity['strict_correct']}/{sensitivity['denominator']}"
                      f" = {_format(sensitivity['strict_accuracy'])}；"
                      f"宽松解释：{sensitivity['lenient_correct']}/{sensitivity['denominator']}"
                      f" = {_format(sensitivity['lenient_accuracy'])}。",
                      "仅已知标签解释变化，非置信区间；严格主结果保持不变。"])
        if "policy_clarification" in report["accuracy"]:
            lines.append(report["accuracy"]["policy_clarification"])
        lines.append("")
        for boundary in sensitivity["boundary_records"]:
            lines.append(f"- {boundary['sample_id']}：严格口径——{boundary['strict_reason']} "
                         f"宽松解释——{boundary['lenient_reason']}")
        lines.extend(["", f"判定输入：`{report['input_paths']['accuracy']}`；SHA256：`{report['input_sha256']['accuracy']}`。"])
    lines.extend(["", "## 口径与缺失数据", ""])
    lines.extend(f"- {text}" for text in report["limitations"])
    for mode, row in report["retrieval"]["by_mode"].items():
        lines.append(f"- {mode} 排除：{json.dumps(row['excluded'], ensure_ascii=False)}；"
                     f"实际候选深度：{json.dumps(row['observed_depth_counts'], ensure_ascii=False)}。")
    lines.extend(["", "## 输入 SHA256", ""])
    lines.extend(f"- {name}: `{value}`" for name, value in report["input_sha256"].items())
    return "\n".join(lines) + "\n"


def write_report(report: dict, output_dir: Path) -> None:
    """Create a new directory exclusively; never replace an existing artifact."""
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), f"输出目录已存在，拒绝覆盖: {output_dir}")
    fields = ("sample_id", "question", "answerability", "mode", "arm", "status", *METRICS,
              "candidate_count", "observed_depth", "duplicate_candidate_count", "relevant_chunk_count",
              "accuracy_field", "correct", "reviewer_agreement", "answer_sha256", "reason")
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    for question in report["questions"]:
        row = {key: question[key] for key in ("sample_id", "question", "answerability", "mode", "arm")}
        if "retrieval" in question:
            row.update(question["retrieval"])
        else:
            accuracy = question["accuracy"]
            row.update({"status": accuracy["status"], "accuracy_field": accuracy["field"],
                        "correct": accuracy["value"], "reviewer_agreement": accuracy["reviewer_agreement"],
                        "answer_sha256": accuracy["answer_sha256"], "reason": accuracy["reason"]})
        writer.writerow(row)
    artifacts = {"evidence_accuracy.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                 "evidence_accuracy.md": _markdown(report), "evidence_accuracy_questions.csv": table.getvalue()}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in artifacts.items():
        with (output_dir / name).open("x", encoding="utf-8-sig" if name.endswith(".csv") else "utf-8", newline="") as handle:
            handle.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        _require(not args.output_dir.exists(), f"输出目录已存在，拒绝覆盖: {args.output_dir}")
        report = summarize(**{f"{name}_path": getattr(args, name) for name in INPUTS})
        write_report(report, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"证据汇总失败: {exc}\n")
    print(f"离线汇总完成：{args.output_dir}；完整性={report['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
