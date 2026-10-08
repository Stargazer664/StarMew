"""Behavior checks for offline, source-scoped evidence and binary-answer metrics."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_evidence_accuracy.py"


def test_offline_command_exposes_the_six_inputs():
    result = subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), "--help"],
                            capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr
    assert all(f"--{name}" in result.stdout for name in
               ("dataset", "corpus", "qrels", "retrieval", "answers", "accuracy", "output-dir"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


@pytest.fixture
def module():
    assert SCRIPT.exists(), "offline evidence/accuracy summarizer has not been implemented"
    spec = importlib.util.spec_from_file_location("evidence_accuracy", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def inputs(tmp_path):
    def chunk(identifier, filename, text=None):
        return {"chunk_id": identifier, "filename": filename, "text": text or identifier,
                "chunk_level": 3}

    chunks = [chunk("a1", "a.md", "第 一事实，10 克。"), chunk("a2", "a.md"),
              chunk("a3", "a.md"), chunk("b1", "b.md")]
    chunks += [chunk(f"x{i}", "other.md") for i in range(1, 9)]
    by_id = {c["chunk_id"]: c for c in chunks}
    samples = [
        {"id": "q1", "question": "按《a》回答第一题", "expected_docs": ["a.md"],
         "metadata": {"answerability": "answerable", "cluster_id": "source-a"}},
        {"id": "q2", "question": "按《b》回答第二题", "expected_docs": ["b.md"],
         "metadata": {"answerability": "answerable", "cluster_id": "source-b"}},
        {"id": "q3", "question": "《a》没有给出的内容是什么", "expected_docs": ["a.md"],
         "metadata": {"answerability": "unanswerable", "cluster_id": "source-a"}},
    ]
    dataset = {"schema_version": 1, "split": "exploratory", "formal_holdout": False,
               "samples": samples}
    corpus = {"schema_version": 1, "collection": "fixture", "index_identity_sha256": "a" * 64,
              "chunks": chunks}
    qrels = {"schema_version": 1, "review_method": "ai_source_checked", "human_reviewed": False,
             "dataset_sha256": digest(encode(dataset)), "corpus_sha256": digest(encode(corpus)),
             "samples": []}
    for sample, relevant in [(samples[0], {"a1", "a2"}), (samples[1], {"b1"})]:
        qrels["samples"].append({
            "sample_id": sample["id"], "expected_docs": sample["expected_docs"],
            "scope_complete": True, "reference_facts_verified": True,
            "relevant_chunk_ids": sorted(relevant),
            "judgments": [{"chunk_id": c["chunk_id"], "relevant": c["chunk_id"] in relevant,
                           "reason": "逐项检查原文", "quote": c["text"] if c["chunk_id"] in relevant else ""}
                          for c in chunks if c["filename"] in sample["expected_docs"]],
        })
    retrieval = {"measurement_kind": "dense_vs_hybrid_retrieval_only", "status": "complete",
                 "dataset_sha256": digest(encode(dataset)), "collection": "fixture", "top_k": 10,
                 "records": []}
    for sample_id, mode, identifiers in [
        ("q1", "dense", ["x1", "a1", "a1", "x2", "x3", "a2"]),
        ("q1", "hybrid", ["a2", "a1"]), ("q2", "dense", ["x4"]),
        ("q2", "hybrid", ["b1"]),
    ]:
        retrieval["records"].append({"sample_id": sample_id, "mode": mode, "status": "success",
                                     "docs": [copy.deepcopy(by_id[i]) for i in identifiers]})
    answers = {"schema_version": 1, "records": []}
    accuracy = {"dataset_sha256": digest(encode(dataset)), "review_method": "two_ai_reviews_with_adjudication",
                "human_reviewed": False, "records": []}
    for sample in samples:
        for arm in ("rag", "no_rag"):
            answer = f"原回答 {sample['id']} {arm}，10 克。"
            positive = sample["metadata"]["answerability"] == "answerable"
            answers["records"].append({"sample_id": sample["id"], "arm": arm, "repeat_index": 0,
                                       "generation_status": "success", "answer": answer})
            accuracy["records"].append({
                "sample_id": sample["id"], "arm": arm,
                "answer_sha256": digest(answer.encode("utf-8")),
                "correct": (arm == "rag" and sample["id"] == "q1") if positive else None,
                "correct_abstention": (arm == "rag") if not positive else None,
                "reason": "核对参考事实及回答是否含实质错误", "reviewer_agreement": True,
            })
    accuracy["answers_sha256"] = digest(encode(answers))
    values = dict(dataset=dataset, corpus=corpus, qrels=qrels, retrieval=retrieval,
                  answers=answers, accuracy=accuracy)
    paths = {key: tmp_path / f"{key}.json" for key in values}
    for key, value in values.items():
        paths[key].write_bytes(encode(value))
    return values, paths


def save(inputs, *names, rehash=False):
    values, paths = inputs
    if rehash:
        values["qrels"]["dataset_sha256"] = digest(encode(values["dataset"]))
        values["qrels"]["corpus_sha256"] = digest(encode(values["corpus"]))
        values["retrieval"]["dataset_sha256"] = digest(encode(values["dataset"]))
        values["accuracy"]["dataset_sha256"] = digest(encode(values["dataset"]))
        values["accuracy"]["answers_sha256"] = digest(encode(values["answers"]))
        names = tuple(values)
    for name in names:
        paths[name].write_bytes(encode(values[name]))


def build(module, inputs):
    return module.summarize(**{f"{key}_path": path for key, path in inputs[1].items()})


def test_metrics_are_chunk_macro_averages_and_fixed_k_precision(module, inputs):
    value = build(module, inputs)
    dense = value["retrieval"]["by_mode"]["dense"]
    assert dense["metrics"]["recall_at_5"] == pytest.approx(0.25)
    assert dense["metrics"]["recall_at_10"] == pytest.approx(0.5)
    assert dense["metrics"]["precision_at_5"] == pytest.approx(0.1)
    assert dense["metrics"]["precision_at_10"] == pytest.approx(0.1)
    assert dense["metrics"]["mrr_at_10"] == pytest.approx(0.25)
    hybrid = value["retrieval"]["by_mode"]["hybrid"]
    assert hybrid["metrics"]["recall_at_5"] == 1
    assert hybrid["metrics"]["precision_at_5"] == pytest.approx(0.3)
    assert dense["eligible_questions"] == 2
    assert dense["complete"] is True


def test_duplicate_chunk_keeps_original_rank_and_never_adds_a_second_hit(module, inputs):
    row = next(r for r in build(module, inputs)["questions"]
               if r["sample_id"] == "q1" and r["mode"] == "dense")
    assert row["retrieval"]["duplicate_candidate_count"] == 1
    assert row["retrieval"]["recall_at_5"] == 0.5  # a2 stays at rank 6
    assert row["retrieval"]["precision_at_10"] == 0.2
    assert row["retrieval"]["mrr_at_10"] == 0.5


def test_accuracy_uses_answerability_and_does_not_call_f1_accuracy(module, inputs):
    value = build(module, inputs)
    rag = value["accuracy"]["by_arm"]["rag"]
    assert rag["answerable"] == {"planned": 2, "reviewed": 2, "correct": 1,
                                  "missing": 0, "accuracy": 0.5, "complete": True}
    assert rag["unanswerable"]["correct_abstention_rate"] == 1
    assert rag["unanswerable"]["planned"] == 1
    assert value["accuracy"]["by_arm"]["no_rag"]["answerable"]["accuracy"] == 0
    assert value["audit"]["source_cluster_count"] == 2
    assert value["audit"]["human_reviewed"] is False
    assert value["audit"]["formal_holdout"] is False


@pytest.mark.parametrize("name,field", [("qrels", "dataset_sha256"), ("qrels", "corpus_sha256"),
                                       ("retrieval", "dataset_sha256"), ("accuracy", "dataset_sha256"),
                                       ("accuracy", "answers_sha256")])
def test_input_hash_mismatch_rejected(module, inputs, name, field):
    inputs[0][name][field] = "0" * 64
    save(inputs, name)
    with pytest.raises(ValueError, match="sha256"):
        build(module, inputs)


@pytest.mark.parametrize("case", ["duplicate_id", "missing_text", "wrong_level"])
def test_corpus_must_have_unique_leaf_ids_and_text(module, inputs, case):
    chunks = inputs[0]["corpus"]["chunks"]
    if case == "duplicate_id":
        chunks.append(copy.deepcopy(chunks[0]))
    elif case == "missing_text":
        del chunks[0]["text"]
    else:
        chunks[0]["chunk_level"] = 2
    save(inputs, rehash=True)
    with pytest.raises(ValueError, match="corpus"):
        build(module, inputs)


@pytest.mark.parametrize("case", ["missing_sample", "duplicate_sample", "missing_judgment",
                                  "duplicate_judgment", "foreign_judgment", "wrong_relevant_set",
                                  "wrong_expected_docs", "bad_quote", "empty_reason", "not_bool"])
def test_qrels_require_full_source_scope_and_consistent_labels(module, inputs, case):
    qrels = inputs[0]["qrels"]
    sample = qrels["samples"][0]
    if case == "missing_sample":
        qrels["samples"].pop()
    elif case == "duplicate_sample":
        qrels["samples"].append(copy.deepcopy(sample))
    elif case == "missing_judgment":
        sample["judgments"].pop()
    elif case == "duplicate_judgment":
        sample["judgments"].append(copy.deepcopy(sample["judgments"][0]))
    elif case == "foreign_judgment":
        sample["judgments"][-1]["chunk_id"] = "x1"
    elif case == "wrong_relevant_set":
        sample["relevant_chunk_ids"] = ["a1"]
    elif case == "wrong_expected_docs":
        sample["expected_docs"] = ["other.md"]
    elif case == "bad_quote":
        sample["judgments"][0]["quote"] = "第 一事实，999 克。"
    elif case == "empty_reason":
        sample["judgments"][0]["reason"] = " "
    else:
        sample["judgments"][0]["relevant"] = 1
    save(inputs, "qrels")
    with pytest.raises(ValueError, match="qrels"):
        build(module, inputs)


def test_normalization_allows_formatting_only_and_never_substring_candidate_identity(module, inputs):
    inputs[0]["qrels"]["samples"][0]["judgments"][0]["quote"] = "第一事实,10克。"
    inputs[0]["retrieval"]["records"][0]["docs"][1]["text"] = "第一事实,10克。"
    save(inputs, "qrels", "retrieval")
    assert build(module, inputs)["retrieval"]["by_mode"]["dense"]["complete"]
    inputs[0]["retrieval"]["records"][0]["docs"][1]["text"] = "第一事实"
    save(inputs, "retrieval")
    with pytest.raises(ValueError, match="candidate.*text"):
        build(module, inputs)


@pytest.mark.parametrize("field,value", [("chunk_id", "missing"), ("filename", "wrong.md"),
                                        ("text", "unverified"), ("chunk_level", 2)])
def test_unknown_or_changed_candidate_is_not_assumed_irrelevant(module, inputs, field, value):
    inputs[0]["retrieval"]["records"][0]["docs"][0][field] = value
    save(inputs, "retrieval")
    with pytest.raises(ValueError, match="candidate"):
        build(module, inputs)


@pytest.mark.parametrize("flag", ["scope_complete", "reference_facts_verified"])
def test_unverified_qrels_are_excluded_not_zero(module, inputs, flag):
    inputs[0]["qrels"]["samples"][0][flag] = False
    save(inputs, "qrels")
    value = build(module, inputs)["retrieval"]["by_mode"]["hybrid"]
    assert value["complete"] is False
    assert value["eligible_questions"] == 1
    assert value["excluded"]["unverified_annotations"] == 1
    assert value["metrics"]["recall_at_5"] == 1


@pytest.mark.parametrize("status", ["failed", "missing"])
def test_failed_or_absent_retrieval_excluded_but_successful_empty_is_zero(module, inputs, status):
    records = inputs[0]["retrieval"]["records"]
    if status == "failed":
        records[0]["status"] = "failed"
        records[0]["docs"] = []
    else:
        records.pop(0)
    save(inputs, "retrieval")
    result = build(module, inputs)["retrieval"]["by_mode"]["dense"]
    assert result["eligible_questions"] == 1
    assert result["metrics"]["recall_at_5"] == 0
    assert result["complete"] is False
    records = inputs[0]["retrieval"]["records"]
    if status == "missing":
        records.insert(0, {"sample_id": "q1", "mode": "dense", "status": "success", "docs": []})
    else:
        records[0]["status"] = "success"
    save(inputs, "retrieval")
    result = build(module, inputs)["retrieval"]["by_mode"]["dense"]
    assert result["eligible_questions"] == 2
    assert result["metrics"]["recall_at_5"] == 0


@pytest.mark.parametrize("case", ["duplicate", "wrong_collection", "shallow_configuration", "negative"])
def test_retrieval_design_must_match_defined_corpus_and_positive_questions(module, inputs, case):
    retrieval = inputs[0]["retrieval"]
    if case == "duplicate":
        retrieval["records"].append(copy.deepcopy(retrieval["records"][0]))
    elif case == "wrong_collection":
        retrieval["collection"] = "another"
    elif case == "shallow_configuration":
        retrieval["top_k"] = 5
    else:
        retrieval["records"][0]["sample_id"] = "q3"
    save(inputs, "retrieval")
    with pytest.raises(ValueError, match="retrieval"):
        build(module, inputs)


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong_hash", "wrong_label", "wrong_arm",
                                  "non_bool", "missing_reason", "fake_human_review", "wrong_method"])
def test_accuracy_requires_complete_unique_answer_bound_adjudications(module, inputs, case):
    accuracy = inputs[0]["accuracy"]
    row = accuracy["records"][0]
    if case == "missing":
        accuracy["records"].pop()
    elif case == "duplicate":
        accuracy["records"].append(copy.deepcopy(row))
    elif case == "wrong_hash":
        row["answer_sha256"] = "0" * 64
    elif case == "wrong_label":
        row["correct_abstention"] = True
    elif case == "wrong_arm":
        row["arm"] = "third_arm"
    elif case == "non_bool":
        row["correct"] = 1
    elif case == "missing_reason":
        row["reason"] = ""
    elif case == "fake_human_review":
        accuracy["human_reviewed"] = True
    else:
        accuracy["review_method"] = "single_ai_review"
    save(inputs, "accuracy")
    with pytest.raises(ValueError, match="accuracy"):
        build(module, inputs)


def test_missing_correctness_does_not_create_error_or_full_denominator(module, inputs):
    inputs[0]["accuracy"]["records"][0]["correct"] = None
    save(inputs, "accuracy")
    value = build(module, inputs)["accuracy"]["by_arm"]["rag"]["answerable"]
    assert value == {"planned": 2, "reviewed": 1, "correct": 0, "missing": 1,
                     "accuracy": 0.0, "complete": False}


@pytest.mark.parametrize("case", ["missing", "duplicate", "repeated", "failed_with_label", "changed_answer"])
def test_generation_records_cannot_be_replaced_or_counted_as_extra_questions(module, inputs, case):
    answers = inputs[0]["answers"]["records"]
    if case == "missing":
        answers.pop()
    elif case == "duplicate":
        answers.append(copy.deepcopy(answers[0]))
    elif case == "repeated":
        answers[0]["repeat_index"] = 1
    elif case == "failed_with_label":
        answers[0]["generation_status"] = "failed"
    else:
        answers[0]["answer"] += "后续编辑"
    save(inputs, rehash=True)
    with pytest.raises(ValueError, match="answers|answer_sha256|accuracy"):
        build(module, inputs)


def test_all_unreviewed_values_remain_null(module, inputs):
    for row in inputs[0]["accuracy"]["records"]:
        row["correct"] = row["correct_abstention"] = None
    for row in inputs[0]["qrels"]["samples"]:
        row["scope_complete"] = False
    save(inputs, "qrels", "accuracy")
    value = build(module, inputs)
    assert value["accuracy"]["by_arm"]["rag"]["answerable"]["accuracy"] is None
    assert value["retrieval"]["by_mode"]["dense"]["metrics"]["recall_at_5"] is None
    assert value["complete"] is False


def test_cli_emits_three_artifacts_with_hashes_and_rejects_overwrite(module, inputs, tmp_path):
    paths = inputs[1]
    before = {key: path.read_bytes() for key, path in paths.items()}
    output = tmp_path / "new-output"
    args = [sys.executable, "-X", "utf8", str(SCRIPT)]
    for name, path in paths.items():
        args += [f"--{name}", str(path)]
    args += ["--output-dir", str(output)]
    first = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert first.returncode == 0, first.stderr
    assert {p.name for p in output.iterdir()} == {
        "evidence_accuracy.json", "evidence_accuracy.md", "evidence_accuracy_questions.csv"}
    report = json.loads((output / "evidence_accuracy.json").read_text(encoding="utf-8"))
    assert report["input_sha256"] == {key: digest(data) for key, data in before.items()}
    prose = (output / "evidence_accuracy.md").read_text(encoding="utf-8")
    assert "AI" in prose and "非人工" in prose and "source-scoped" in prose
    assert "固定" in prose and "探索" in prose
    assert "66.76%" not in prose
    artifacts_before = {p.name: p.read_bytes() for p in output.iterdir()}
    second = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert second.returncode != 0
    assert {p.name: p.read_bytes() for p in output.iterdir()} == artifacts_before
    assert {key: path.read_bytes() for key, path in paths.items()} == before


def test_existing_empty_output_directory_also_refused(module, inputs, tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises((ValueError, FileExistsError), match="存在|exist"):
        module.write_report(build(module, inputs), output)


def add_sensitivity(inputs):
    accuracy = inputs[0]["accuracy"]
    boundary = next(row for row in accuracy["records"] if row["sample_id"] == "q2" and row["arm"] == "rag")
    boundary["lenient_sensitivity_correct"] = True
    boundary["adjudication"] = {"sensitivity_reason": "若接受最后明确自我修正，可判正确。"}
    accuracy["policy_clarification"] = "保守要求整条回答一致，同时保留自我修正的宽松解释。"
    accuracy["sensitivity"] = {"strict_correct": 1, "lenient_correct": 2, "denominator": 2,
                               "strict_accuracy": 0.5, "lenient_accuracy": 1.0, "sample_ids": ["q2"]}
    save(inputs, "accuracy")
    return boundary


def test_sensitivity_is_derived_from_boundary_labels_without_changing_main_accuracy(module, inputs, tmp_path):
    add_sensitivity(inputs)
    result = build(module, inputs)
    assert result["accuracy"]["by_arm"]["rag"]["answerable"]["accuracy"] == 0.5
    sensitivity = result["accuracy"]["sensitivity"]
    assert sensitivity["strict_correct"] == 1 and sensitivity["lenient_correct"] == 2
    assert sensitivity["denominator"] == 2
    assert sensitivity["strict_accuracy"] == 0.5 and sensitivity["lenient_accuracy"] == 1.0
    assert sensitivity["sample_ids"] == ["q2"]
    assert sensitivity["boundary_records"][0]["lenient_reason"] == "若接受最后明确自我修正，可判正确。"
    assert result["accuracy"]["policy_clarification"] == inputs[0]["accuracy"]["policy_clarification"]
    output = tmp_path / "sensitivity-output"
    module.write_report(result, output)
    prose = (output / "evidence_accuracy.md").read_text(encoding="utf-8")
    for expected in ("1/2", "2/2", "q2", "仅已知标签解释变化，非置信区间", "自我修正",
                     str(inputs[1]["accuracy"].resolve())):
        assert expected in prose


def test_without_alternative_labels_report_has_no_sensitivity(module, inputs):
    result = build(module, inputs)
    assert "sensitivity" not in result["accuracy"]
    assert "标签解释敏感性" not in module._markdown(result)


@pytest.mark.parametrize("invalid", ["label_type", "aggregate_count", "aggregate_without_labels"])
def test_sensitivity_rejects_wrong_labels_or_unverified_top_level_counts(module, inputs, invalid):
    boundary = add_sensitivity(inputs)
    if invalid == "label_type":
        boundary["lenient_sensitivity_correct"] = 1
    elif invalid == "aggregate_count":
        inputs[0]["accuracy"]["sensitivity"]["lenient_correct"] = 999
    else:
        del boundary["lenient_sensitivity_correct"]
    save(inputs, "accuracy")
    with pytest.raises(ValueError, match="sensitivity"):
        build(module, inputs)
