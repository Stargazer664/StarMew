import csv
import json
import math

import pytest

from backend.eval_comparison.contracts import ComparisonSample
from backend.eval_comparison.reporting import export_report, summarize_pairs


def _sample(index, *, answerability="answerable", cluster=None, question_type="fact"):
    return ComparisonSample(
        id=f"q{index}",
        question=f"合成问题{index}",
        ground_truth="合成答案",
        expected_docs=[],
        metadata={
            "domain": "synthetic",
            "question_type": question_type,
            "answerability": answerability,
            "cluster_id": cluster or f"c{index}",
            "split": "dev",
            "reference_reviewed": False,
        },
    )


def _record(sample_id, arm, value, *, repeat=0, metric="factual_correctness", latency=1.0):
    return {
        "sample_id": sample_id,
        "repeat_index": repeat,
        "arm": arm,
        "generation_status": "success",
        "answer": f"{arm}-{sample_id}-{repeat}",
        "metrics": {metric: value},
        "metric_status": {metric: "success"},
        "metric_errors": {},
        "timing": {"response_seconds": latency},
        "usage": {
            "generation_input_tokens": 10,
            "generation_output_tokens": 2,
            "retrieval_llm_tokens": 3 if arm == "rag" else None,
            "retrieval_usage_complete": arm == "no_rag",
        },
    }


def test_paired_gain_matches_hand_calculation():
    samples = [_sample(0), _sample(1)]
    records = []
    for index, (baseline, rag) in enumerate([(0.4, 0.6), (0.8, 1.0)]):
        records.extend([
            _record(f"q{index}", "no_rag", baseline),
            _record(f"q{index}", "rag", rag),
        ])
    result = summarize_pairs(records, samples, seed=42, bootstrap_samples=100)[
        "answerable"
    ]["factual_correctness"]
    assert result["paired_question_count"] == 2
    assert result["no_rag_mean"] == pytest.approx(0.6)
    assert result["rag_mean"] == pytest.approx(0.8)
    assert result["absolute_gain"] == pytest.approx(0.2)
    assert result["relative_gain_pct"] == pytest.approx(100 / 3)


def test_zero_baseline_has_null_relative_gain():
    summary = summarize_pairs(
        [_record("q0", "no_rag", 0.0), _record("q0", "rag", 0.5)],
        [_sample(0)], bootstrap_samples=50,
    )
    assert summary["answerable"]["factual_correctness"]["relative_gain_pct"] is None
    assert any("基线为 0" in warning for warning in summary["warnings"])


def test_incomplete_pair_and_nan_are_excluded_and_reported():
    bad = _record("q1", "rag", math.nan)
    summary = summarize_pairs(
        [
            _record("q0", "no_rag", 0.4),
            _record("q0", "rag", 0.6),
            _record("q1", "no_rag", 0.8),
            bad,
        ],
        [_sample(0), _sample(1)], bootstrap_samples=50,
    )
    metric = summary["answerable"]["factual_correctness"]
    assert metric["paired_question_count"] == 1
    assert summary["coverage"]["factual_correctness_paired_question_coverage"] == 0.5
    assert summary["coverage"]["metric_failed_or_invalid"] == 1


def test_repeated_rounds_are_averaged_within_question_first():
    records = [
        _record("q0", "no_rag", 0.0, repeat=0),
        _record("q0", "rag", 1.0, repeat=0),
        _record("q0", "no_rag", 1.0, repeat=1),
        _record("q0", "rag", 1.0, repeat=1),
        _record("q1", "no_rag", 1.0, repeat=0),
        _record("q1", "rag", 0.0, repeat=0),
    ]
    metric = summarize_pairs(records, [_sample(0), _sample(1)], bootstrap_samples=50)[
        "answerable"
    ]["factual_correctness"]
    assert metric["paired_round_count"] == 3
    assert metric["absolute_gain"] == pytest.approx(-0.25)


def test_cluster_bootstrap_is_reproducible_and_flags_small_cluster_count():
    samples = [_sample(0, cluster="same"), _sample(1, cluster="same"), _sample(2, cluster="other")]
    records = []
    for sample, delta in zip(samples, [0.2, 0.4, -0.1]):
        records.extend([
            _record(sample.id, "no_rag", 0.5),
            _record(sample.id, "rag", 0.5 + delta),
        ])
    first = summarize_pairs(records, samples, seed=9, bootstrap_samples=200)
    second = summarize_pairs(records, samples, seed=9, bootstrap_samples=200)
    metric = first["answerable"]["factual_correctness"]
    assert metric["ci95"] == second["answerable"]["factual_correctness"]["ci95"]
    assert metric["cluster_count"] == 2
    assert metric["ci_exploratory"] is True


def test_unreviewed_references_make_report_explicitly_exploratory(tmp_path):
    samples = [_sample(index) for index in range(10)]
    records = [
        _record(sample.id, arm, 0.5)
        for sample in samples
        for arm in ("no_rag", "rag")
    ]
    summary = summarize_pairs(records, samples, bootstrap_samples=10)
    export_report(tmp_path, summary, records)

    assert any("参考答案未完成人工审核" in warning for warning in summary["warnings"])
    assert "结论仅为探索性" in (tmp_path / "report.md").read_text(encoding="utf-8")

    for sample in samples:
        sample.metadata.reference_reviewed = True
    reviewed = summarize_pairs(records, samples, bootstrap_samples=10)
    assert not any("参考答案未完成人工审核" in warning for warning in reviewed["warnings"])


def test_unanswerable_abstention_and_answerable_over_refusal_are_separate():
    samples = [_sample(0), _sample(1, answerability="unanswerable")]
    records = [
        _record("q0", "no_rag", 1.0, metric="abstention"),
        _record("q0", "rag", 0.0, metric="abstention"),
        _record("q1", "no_rag", 0.0, metric="abstention"),
        _record("q1", "rag", 1.0, metric="abstention"),
    ]
    summary = summarize_pairs(records, samples, bootstrap_samples=50)
    assert summary["answerable"]["over_refusal_rate"]["no_rag"] == 1.0
    assert summary["answerable"]["over_refusal_rate"]["rag"] == 0.0
    assert summary["unanswerable"]["correct_abstention_rate"]["no_rag"] == 0.0
    assert summary["unanswerable"]["correct_abstention_rate"]["rag"] == 1.0
    assert summary["unanswerable"]["correct_abstention_rate"]["rag_count"] == "1/1"


def test_all_failed_and_empty_group_produce_null_not_zero():
    failed = _record("q0", "no_rag", None)
    failed["generation_status"] = "failed"
    failed["metric_status"]["factual_correctness"] = "failed"
    summary = summarize_pairs([failed], [_sample(0)], bootstrap_samples=50)
    metric = summary["answerable"]["factual_correctness"]
    assert metric["no_rag_mean"] is None
    assert metric["absolute_gain"] is None
    assert summary["groups"]["question_type"] == {"fact": {"paired_question_count": 0, "absolute_gain": None}}


def test_export_uses_null_json_blank_csv_and_chinese_status(tmp_path):
    records = [_record("q0", "no_rag", None)]
    records[0]["metric_status"]["factual_correctness"] = "failed"
    summary = summarize_pairs(records, [_sample(0)], bootstrap_samples=10)
    export_report(tmp_path, summary, records)

    parsed = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert parsed["answerable"]["factual_correctness"]["absolute_gain"] is None
    with (tmp_path / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["metric_value"] == ""
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "失败/不适用" in markdown
    assert (tmp_path / "review.csv").is_file()
    assert (tmp_path / "review_key.json").is_file()


def test_markdown_regressions_only_include_negative_deltas(tmp_path):
    samples = [_sample(0), _sample(1), _sample(2)]
    records = [
        _record("q0", "no_rag", 0.2),
        _record("q0", "rag", 0.8),
        _record("q1", "no_rag", 0.7),
        _record("q1", "rag", 0.4),
        _record("q2", "no_rag", 0.5),
        _record("q2", "rag", 0.5),
    ]
    summary = summarize_pairs(records, samples, bootstrap_samples=10)
    export_report(tmp_path, summary, records)

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "- 改善 0.6000：q0" in markdown
    assert "- 退步 -0.3000：q1" in markdown
    assert "- 退步 0.6000" not in markdown
    assert "q2" not in markdown
