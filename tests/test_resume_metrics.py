"""Offline resume reports must not turn missing evidence into impressive scores."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_resume_metrics.py"


def sample(identifier, *, question=None, expected=None, complete=False, reviewed=True):
    return {
        "id": identifier,
        "question": question or f"问题 {identifier}",
        "ground_truth": "参考事实",
        "expected_docs": ["a.md"] if expected is None else expected,
        "metadata": {
            "answerability": "answerable", "split": "test", "domain": "fixture",
            "question_type": "fact", "cluster_id": identifier,
            "reference_reviewed": reviewed,
            "relevance_annotations_complete": complete,
            "reference_facts": ["参考事实"],
            "evidence": [{"source": "a.md", "text": "参考事实"}],
        },
    }


def record(identifier, *, docs=None, repeat=0, arm="rag", score=0.8, **changes):
    value = {
        "sample_id": identifier, "repeat_index": repeat, "arm": arm,
        "generation_status": "success", "answer": "回答", "error": None,
        "retrieved_docs": [{"source": "a.md"}] if docs is None else docs,
        "used_contexts": [], "rag_trace": [], "retrieval_empty": False,
        "timing": {"response_seconds": 2.0, "retrieval_seconds": 0.5},
        "usage": {}, "attempts": [],
        "metrics": {"factual_correctness": score, "abstention": 0.0},
        "metric_status": {"factual_correctness": "success", "abstention": "success"},
    }
    value.update(changes)
    return value


def invoke(tmp_path, samples, records, *, repeat=1, same_output=False):
    run = tmp_path / "run"
    run.mkdir()
    dataset = json.dumps({"samples": samples}, ensure_ascii=False).encode("utf-8")
    (run / "dataset.json").write_bytes(dataset)
    manifest = {
        "run_id": "test-run", "config": {"repeat": repeat, "seed": 42},
        "dataset_sha256": hashlib.sha256(dataset).hexdigest(), "fingerprint": "test-fingerprint",
        "plan": [{"sample_id": s["id"], "repeat_index": i, "arm": arm}
                 for s in samples for i in range(repeat) for arm in ("rag", "no_rag")],
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "checkpoint.json").write_text(
        json.dumps({"records": records, "fingerprint": "test-fingerprint"}), encoding="utf-8"
    )
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    output = run if same_output else tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPT), "--run-dir", str(run), "--output-dir", str(output)],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    return result, output, before


def report(result, output):
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((output / "resume_metrics.json").read_text(encoding="utf-8"))


def test_exact_sources_preserve_candidate_ranks_and_document_denominator(tmp_path):
    docs = [
        {"source": "manual/a.md.backup"},
        {"metadata": {"source": r"manual\a.md"}},
        {"source": "manual/a.md"}, {"source": "other.md"}, {"doc_id": "c-id"},
        {"source": "b.md"},
    ]
    result, output, before = invoke(
        tmp_path, [sample("q", expected=["manual/a.md", "b.md", "c-id"], complete=True)],
        [record("q", docs=docs)],
    )
    value = report(result, output)
    metrics = value["retrieval"]["metrics"]
    assert metrics["recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["recall_at_10"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 0.5
    assert value["input_sha256"] == {name: hashlib.sha256(data).hexdigest() for name, data in before.items()}
    assert (output / "resume_metrics.md").is_file()
    with (output / "resume_metrics_questions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert float(rows[0]["recall_at_5"]) == pytest.approx(2 / 3)


def test_incomplete_annotations_are_coverage_and_substrings_never_hit(tmp_path):
    result, output, _ = invoke(tmp_path, [sample("q")], [record("q", docs=[{"source": "a.md.backup"}])])
    value = report(result, output)
    assert value["retrieval"]["metrics"]["expected_source_coverage_at_5"] == 0
    assert value["retrieval"]["metrics"]["recall_at_5"] is None
    assert value["retrieval"]["metrics"]["mrr_at_10"] == 0
    assert value["audit"]["exploratory"] is True
    assert "预期来源覆盖率" in (output / "resume_metrics.md").read_text(encoding="utf-8")


def test_repeats_and_duplicate_question_ids_are_one_independent_question(tmp_path):
    samples = [sample("a", question="重复问题"), sample("a-copy", question=" 重复问题 "), sample("b")]
    records = [record("a"), record("a", repeat=1), record("a-copy", docs=[]), record("b", docs=[])]
    result, output, _ = invoke(tmp_path, samples, records, repeat=2)
    value = report(result, output)
    assert value["audit"]["dataset_rows"] == 3
    assert value["audit"]["unique_question_count"] == 2
    assert value["audit"]["duplicate_question_rows"] == 1
    assert value["retrieval"]["metrics"]["hit_at_5"] == pytest.approx(1 / 3)
    assert value["retrieval"]["valid_question_count"] == 2
    assert value["retrieval"]["valid_round_count"] == 4


def test_failures_missing_labels_and_unknown_sources_are_excluded_not_zero(tmp_path):
    samples = [sample(name, expected=[] if name == "unlabeled" else None)
               for name in ("good", "unlabeled", "failed", "pending", "unknown", "llm_failed")]
    records = [
        record("good"), record("unlabeled"),
        record("failed", docs=[], generation_status="failed", retrieval_empty=None,
               attempts=[{"stage": "retrieval", "status": "failed"}]),
        record("pending", docs=[], generation_status="pending", retrieval_empty=None),
        record("unknown", docs=[{"text": "a.md"}]),
        record("llm_failed", generation_status="failed", attempts=[{"stage": "generation", "status": "failed"}]),
    ]
    result, output, _ = invoke(tmp_path, samples, records)
    value = report(result, output)
    assert value["retrieval"]["metrics"]["hit_at_5"] == 1.0
    assert value["retrieval"]["valid_round_count"] == 2
    assert value["retrieval"]["excluded_rounds"] == {
        "missing_annotations": 1, "retrieval_failed": 1,
        "retrieval_unavailable": 1, "unidentifiable_candidates": 1,
    }


def test_unanswerable_sources_are_not_retrieval_ground_truth_but_keep_refusal_and_latency(tmp_path):
    negative = sample("negative")
    negative["metadata"]["answerability"] = "unanswerable"
    records = [record("positive"), record("positive", arm="no_rag", score=0.2),
               record("negative", docs=[{"filename": "unrelated.md"}],
                      metrics={"abstention": 1.0}, metric_status={"abstention": "success"},
                      timing={"response_seconds": 8}),
               record("negative", arm="no_rag", docs=[], metrics={"abstention": 0.0},
                      metric_status={"abstention": "success"})]
    result, output, _ = invoke(tmp_path, [sample("positive"), negative], records)
    value = report(result, output)
    retrieval = value["retrieval"]
    assert retrieval["metrics"]["expected_source_coverage_at_5"] == 1
    assert retrieval["valid_question_count"] == 1
    assert retrieval["unanswerable_question_count"] == 1
    assert retrieval["excluded_not_applicable"] == 1
    assert retrieval["excluded_rounds"] == {"unanswerable_not_applicable": 1}
    negative_row = next(row for row in value["questions"] if row["sample_id"] == "negative")
    assert negative_row["hit_at_5"] is None
    assert value["answer_metrics"]["unanswerable"]["correct_abstention_rate"]["rag"] == 1
    assert value["answer_metrics"]["answerable"]["factual_correctness"]["paired_question_count"] == 1
    assert value["latency"]["by_arm"]["rag"]["response_seconds"]["mean"] == 5


def test_pairing_reuses_f1_and_marks_original_refusal_weighting(tmp_path):
    samples = [sample("q"), sample("missing")]
    records = [record("q", arm="no_rag", score=0.2), record("q", score=0.8), record("missing")]
    result, output, _ = invoke(tmp_path, samples, records)
    value = report(result, output)
    factual = value["answer_metrics"]["answerable"]["factual_correctness"]
    assert factual["paired_question_count"] == 1
    assert factual["absolute_gain"] == pytest.approx(0.6)
    assert value["answer_metrics"]["method"]["refusal_weighting"] == "paired_rounds"
    assert value["audit"]["exploratory"] is True


def test_latency_is_observed_only_and_ttfc_is_not_fabricated(tmp_path):
    result, output, _ = invoke(tmp_path, [sample("q"), sample("b")], [
        record("q", timing={"response_seconds": 2, "retrieval_seconds": 0.2}),
        record("q", repeat=1, timing={"response_seconds": 4, "retrieval_seconds": 0.4}),
        record("b", timing={"response_seconds": 9, "retrieval_seconds": None}),
    ], repeat=2)
    value = report(result, output)
    latency = value["latency"]["by_arm"]["rag"]["response_seconds"]
    assert latency["mean"] == 6.0
    assert latency["observed_round_count"] == 3
    assert latency["question_count"] == 2
    assert value["latency"]["first_content"]["status"] == "not_collected"
    assert "首内容延迟未采集" in (output / "resume_metrics.md").read_text(encoding="utf-8")


def test_observed_depth_and_real_first_content_are_disclosed(tmp_path):
    result, output, _ = invoke(tmp_path, [sample("q")], [record("q", timing={
        "response_seconds": 9, "first_content_seconds": 2.5,
        "generation_first_content_seconds": 1.0,
    })])
    value = report(result, output)
    assert value["retrieval"]["observed_depth"]["max_candidates"] == 1
    assert value["retrieval"]["observed_depth"]["rounds_below_10"] == 1
    assert value["latency"]["first_content"]["status"] == "collected"
    assert value["latency"]["by_arm"]["rag"]["first_content_seconds"]["mean"] == 2.5
    assert value["latency"]["by_arm"]["rag"]["generation_first_content_seconds"]["mean"] == 1.0


def test_subsequent_latency_excludes_plan_first_success_not_fastest_or_record_first(tmp_path):
    samples = [sample("first"), sample("fast"), sample("missing")]
    records = [
        record("fast", arm="no_rag", timing={"response_seconds": 1}),
        record("fast", timing={"response_seconds": 2}),
        record("missing", timing={}), record("missing", arm="no_rag", timing={}),
        record("first", timing={"response_seconds": 30}),
        record("first", arm="no_rag", timing={"response_seconds": 20}),
    ]
    result, output, _ = invoke(tmp_path, samples, records)
    value = report(result, output)
    latency = value["latency"]
    assert latency["by_arm"]["rag"]["response_seconds"]["mean"] == 16
    assert latency["subsequent_requests_by_arm"]["rag"]["response_seconds"]["mean"] == 2
    assert latency["subsequent_requests_by_arm"]["no_rag"]["response_seconds"]["mean"] == 1
    assert latency["subsequent_requests_by_arm"]["rag"]["response_seconds"]["observed_round_count"] == 1
    assert latency["subsequent_requests_by_arm"]["rag"]["response_seconds"]["missing_or_failed_round_count"] == 1
    assert {(item["sample_id"], item["repeat_index"], item["arm"])
            for item in latency["startup_exclusions"]} == {("first", 0, "rag"), ("first", 0, "no_rag")}


def test_first_success_is_excluded_even_without_timing_and_failures_do_not_count(tmp_path):
    result, output, _ = invoke(tmp_path, [sample("failed"), sample("untimed"), sample("later")], [
        record("failed", generation_status="failed", timing={"response_seconds": 999}),
        record("untimed", timing={}), record("later", timing={"response_seconds": 4}),
    ])
    latency = report(result, output)["latency"]
    assert latency["startup_exclusions"] == [{"sample_id": "untimed", "repeat_index": 0, "arm": "rag"}]
    later = latency["subsequent_requests_by_arm"]["rag"]["response_seconds"]
    assert later["mean"] == 4
    assert later["observed_round_count"] == 1
    assert later["missing_or_failed_round_count"] == 1
    assert latency["subsequent_requests_by_arm"]["no_rag"]["response_seconds"]["mean"] is None


def test_source_cluster_count_is_distinct_from_unique_question_count(tmp_path):
    samples = [sample("a"), sample("b"), sample("c")]
    samples[1]["metadata"]["cluster_id"] = "a"
    samples[2]["metadata"]["cluster_id"] = ""
    result, output, _ = invoke(tmp_path, samples, [record("a"), record("b"), record("c")])
    audit = report(result, output)["audit"]
    assert audit["unique_question_count"] == 3
    assert audit["source_cluster_count"] == 1
    assert audit["missing_cluster_rows"] == 1


def test_public_ranking_api_supports_real_filename_without_fuzzy_matching():
    from scripts import summarize_resume_metrics as metrics

    value = metrics.evaluate_retrieval(
        ["菜谱.pdf", "id-b"],
        [{"filename": "菜谱.pdf.backup"}, {"filename": "菜谱.pdf"}, {"doc_id": "id-b"}],
    )
    assert value["status"] == "success"
    assert value["coverage_at_5"] == 1.0
    assert value["hit_at_10"] == 1.0
    assert value["mrr_at_10"] == 0.5
    assert value["candidate_count"] == 3
    empty = metrics.evaluate_retrieval(["a.md"], [])
    assert empty["coverage_at_10"] == 0.0
    assert metrics.evaluate_retrieval([], [])["coverage_at_5"] is None
    assert metrics.evaluate_retrieval(["a.md"], [{"text": "a.md"}])["coverage_at_5"] is None


def test_path_labels_and_filename_only_candidates_require_identity_mapping(tmp_path):
    result, output, _ = invoke(tmp_path, [sample("q", expected=["dishes/a.md"])],
                                [record("q", docs=[{"filename": "a.md"}])])
    value = report(result, output)
    assert value["retrieval"]["metrics"]["expected_source_coverage_at_5"] is None
    assert value["retrieval"]["excluded_rounds"] == {"identity_mapping_required": 1}
    assert value["retrieval"]["observed_depth"]["max_candidates"] == 1
    assert "来源标识映射" in (output / "resume_metrics.md").read_text(encoding="utf-8")
    from scripts.summarize_resume_metrics import evaluate_retrieval

    assert evaluate_retrieval(["dishes/a.md"], [{"filename": "another.md"}])["status"] == "identity_mapping_required"
    known = evaluate_retrieval(["dishes/a.md"], [{"filename": "a.md", "source": "dishes/a.md"}])
    assert known["coverage_at_5"] == 1.0


@pytest.mark.parametrize("violation", ["duplicate_record", "foreign_sample", "same_output"])
def test_invalid_input_cannot_silently_overwrite_or_reweight(tmp_path, violation):
    records = [record("q")]
    if violation == "duplicate_record":
        records.append(record("q"))
    if violation == "foreign_sample":
        records.append(record("foreign"))
    result, output, _ = invoke(tmp_path, [sample("q")], records, same_output=violation == "same_output")
    assert result.returncode == 2
    assert "错误" in result.stderr
    assert not (output / "resume_metrics.json").exists()
