"""Offline safeguards for reusing unscored generation after reference correction."""

import hashlib
import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool():
    assert importlib.util.find_spec("scripts.prepare_resume_rescore") is not None, (
        "reference-only scoring preparation script is missing"
    )
    return importlib.import_module("scripts.prepare_resume_rescore")


@pytest.fixture
def case(tmp_path, monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    import scripts.run_rag_comparison as runner
    from backend.eval_comparison.contracts import ComparisonConfig
    from backend.eval_comparison.storage import canonical_fingerprint

    monkeypatch.setattr(runner, "_corpus_identity", lambda: {"documents": []})
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"generation.py": "original-generation-hash"})
    monkeypatch.setattr(runner, "_git_identity", lambda: {"head": "offline-head"})
    monkeypatch.setattr(runner, "_ragas_version", lambda: "offline")
    monkeypatch.setattr(runner, "_endpoint_identifier", lambda: "http://offline")
    monkeypatch.setattr(runner, "_llm_identity", lambda: {
        "provider": "offline", "generation_base_url": "http://offline",
        "judge_provider": "offline", "judge_base_url": "http://offline",
    })
    source = tmp_path / "original"
    source.mkdir()
    sample = {
        "id": "q1", "question": "How long?", "ground_truth": "20 minutes. Extra step.",
        "expected_docs": ["source.md"],
        "metadata": {
            "domain": "fixture", "question_type": "fact", "answerability": "answerable",
            "split": "exploratory", "cluster_id": "source", "reference_reviewed": False,
            "reference_facts": ["20 minutes", "Extra step"],
            "evidence": [{"source": "source.md", "locator": "L1", "text": "20 minutes. Extra step."}],
        },
    }
    original_dataset = {"schema_version": 1, "dataset_version": "v1", "samples": [sample]}
    _json(source / "dataset.json", original_dataset)
    revised = deepcopy(original_dataset)
    revised["dataset_version"] = "v2"
    revised["samples"][0]["ground_truth"] = "20 minutes."
    revised["samples"][0]["metadata"]["reference_facts"] = ["20 minutes"]
    revised["reference_revision"] = {"review_type": "independent_ai", "generation_outputs_consulted": False}
    dataset = tmp_path / "revised.json"
    _json(dataset, revised)
    config = ComparisonConfig(model="offline", judge_model="offline", stream=True,
                              metrics=["factual_correctness"])
    inputs = runner._fingerprint_inputs(source / "dataset.json", config)
    fingerprint = canonical_fingerprint(inputs)
    plan = [{"sample_id": "q1", "repeat_index": 0, "arm": arm} for arm in ["no_rag", "rag"]]
    manifest = {
        "schema_version": 1, "run_id": "original", "created_at": "2026-10-04T00:00:00Z",
        "dataset_source": str(source / "dataset.json"), "dataset_sha256": _sha(source / "dataset.json"),
        "config": config.model_dump(mode="json"), "selected_sample_ids": ["q1"], "plan": plan,
        "source_hashes": inputs["source_hashes"], "corpus_identity": inputs["corpus_identity"],
        "actual_generation_model": "offline", "actual_judge_model": "offline",
        "fingerprint_inputs": inputs, "fingerprint": fingerprint,
    }
    records = []
    for item in plan:
        record = runner._new_record(item, config.metrics)
        record.update({
            "generation_status": "success", "answer": "20 minutes.", "prompt": "original prompt",
            "retrieved_docs": [{"text": "unchanged retrieval"}],
            "used_contexts": [{"text": "unchanged context"}], "rag_trace": [{"step": "retrieve"}],
            "retrieval_empty": False, "usage": {"generation_input_tokens": 12, "generation_output_tokens": 3},
            "attempts": [{"stage": "generation", "status": "success"}],
            "generation_attempts": [{"status": "success", "elapsed_seconds": 3.0}],
            "timing": {"retrieval_seconds": 1.0, "generation_seconds": 2.0, "response_seconds": 3.0,
                       "first_content_seconds": 1.5, "generation_first_content_seconds": 0.5,
                       "evaluation_seconds": None},
        })
        records.append(record)
    _json(source / "manifest.json", manifest)
    _json(source / "checkpoint.json", {"schema_version": 1, "run_id": "original",
          "fingerprint": fingerprint, "updated_at": "2026-10-04T00:01:00Z", "records": records})
    return source, dataset, tmp_path / "corrected-score", runner


def _prepare(case):
    source, dataset, output, _ = case
    return _tool().prepare_rescore(source, dataset, output)


def test_fork_preserves_generation_and_supports_score_resume_without_generation(case):
    source, dataset, output, runner = case
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    original_records = _read(source / "checkpoint.json")["records"]
    assert _tool().main(["--source-run", str(source), "--dataset", str(dataset),
                         "--output-run", str(output)]) == 0
    manifest, samples, checkpoint = runner._load_run(output)
    assert checkpoint["records"] == original_records
    assert samples[0].ground_truth == "20 minutes."
    assert _read(output / "dataset.json")["dataset_version"] == "v2"
    assert manifest["generation_reused"] is True
    provenance = manifest["provenance"]
    assert provenance["source_manifest_sha256"] == _sha(source / "manifest.json")
    assert provenance["source_checkpoint_sha256"] == _sha(source / "checkpoint.json")
    assert provenance["source_dataset_sha256"] == _sha(source / "dataset.json")
    assert provenance["generation_source_hashes"] == {"generation.py": "original-generation-hash"}
    assert provenance["changed_reference_sample_ids"] == ["q1"]
    assert manifest["fingerprint"] != _read(source / "manifest.json")["fingerprint"]

    from backend.eval_comparison.contracts import MetricResult

    class ScoreOnly(runner.PipelineDependencies):
        def create_llm(self, config):
            raise AssertionError("must not regenerate")
        def create_retriever(self, config):
            raise AssertionError("must not retrieve")
        def create_scorer(self, config):
            return object()
        async def score(self, sample, answer, scorer, metrics, timeout):
            assert sample.ground_truth == "20 minutes."
            assert answer == "20 minutes."
            return {"factual_correctness": MetricResult(name="factual_correctness", value=1.0, status="success")}

    assert runner.main(["resume", "--run-dir", str(output), "--phase", "score"], dependencies=ScoreOnly()) == 0
    assert all(r["metrics"]["factual_correctness"] == 1.0 for r in _read(output / "checkpoint.json")["records"])
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


@pytest.mark.parametrize("status", ["pending", "running", "unknown"])
def test_incomplete_generation_cannot_be_reused(case, status):
    source, _, output, _ = case
    payload = _read(source / "checkpoint.json")
    payload["records"][0]["generation_status"] = status
    _json(source / "checkpoint.json", payload)
    with pytest.raises(ValueError, match="generation"):
        _prepare(case)
    assert not output.exists()


def test_failed_generation_is_retained_as_failed(case):
    source, _, output, _ = case
    payload = _read(source / "checkpoint.json")
    payload["records"][0].update(generation_status="failed", answer=None, error="TimeoutError: fixture")
    _json(source / "checkpoint.json", payload)
    _prepare(case)
    assert _read(output / "checkpoint.json")["records"] == payload["records"]


@pytest.mark.parametrize("field,value", [
    ("metrics", {"factual_correctness": 0.0}),
    ("metric_status", {"factual_correctness": "running"}),
    ("metric_status", {"factual_correctness": "failed"}),
    ("metric_status", {"factual_correctness": "success"}),
    ("metric_status", {"factual_correctness": "not_applicable"}),
    ("metric_errors", {"factual_correctness": "earlier error"}),
    ("metric_reasons", {"factual_correctness": "earlier reason"}),
    ("scoring_attempts", [{"status": "failed"}]),
    ("timing", {"evaluation_seconds": 0.0}),
])
def test_any_prior_scoring_evidence_is_rejected(case, field, value):
    source, _, output, _ = case
    payload = _read(source / "checkpoint.json")
    payload["records"][0][field] = value
    _json(source / "checkpoint.json", payload)
    with pytest.raises(ValueError, match="scor"):
        _prepare(case)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("id", "different-id"), ("question", "Different question?"),
    ("expected_docs", ["different.md"]), ("extra_prompt_hint", "new hint"),
    ("metadata.answerability", "unanswerable"), ("metadata.evidence", []),
    ("metadata.cluster_id", "different-cluster"), ("metadata.split", "test"),
])
def test_changes_outside_reference_and_review_metadata_are_rejected(case, field, value):
    _, dataset, output, _ = case
    revised = _read(dataset)
    target = revised["samples"][0]
    if field.startswith("metadata."):
        target = target["metadata"]
        field = field.split(".")[1]
    target[field] = value
    _json(dataset, revised)
    with pytest.raises(ValueError):
        _prepare(case)
    assert not output.exists()


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "extra_repeat", "missing_plan_arm"])
def test_checkpoint_must_match_complete_paired_plan(case, alteration):
    source, _, output, _ = case
    payload = _read(source / "checkpoint.json")
    manifest = _read(source / "manifest.json")
    if alteration == "missing":
        payload["records"].pop()
    elif alteration == "duplicate":
        payload["records"].append(deepcopy(payload["records"][0]))
    elif alteration == "extra_repeat":
        payload["records"][0]["repeat_index"] = 3
    else:
        manifest["plan"].pop()
        payload["records"].pop()
        _json(source / "manifest.json", manifest)
    _json(source / "checkpoint.json", payload)
    with pytest.raises(ValueError, match="plan|record|execution"):
        _prepare(case)
    assert not output.exists()


def test_rejects_locked_source_and_existing_output(case):
    source, _, output, _ = case
    lock = source / ".run.lock"
    lock.write_text("active", encoding="utf-8")
    with pytest.raises(ValueError, match="lock"):
        _prepare(case)
    assert lock.read_text(encoding="utf-8") == "active"
    lock.unlink()
    output.mkdir()
    with pytest.raises(FileExistsError):
        _prepare(case)


def test_output_inside_source_is_rejected_without_source_writes(case):
    source, dataset, _, _ = case
    with pytest.raises(ValueError, match="source"):
        _tool().prepare_rescore(source, dataset, source / "nested")
    assert not (source / "nested").exists()


def test_source_dataset_and_checkpoint_fingerprints_are_checked(case):
    source, _, output, _ = case
    payload = _read(source / "checkpoint.json")
    payload["fingerprint"] = "not-original"
    _json(source / "checkpoint.json", payload)
    with pytest.raises(ValueError, match="fingerprint"):
        _prepare(case)
    assert not output.exists()


def test_duplicate_new_sample_ids_are_rejected(case):
    _, dataset, output, _ = case
    revised = _read(dataset)
    revised["samples"].append(deepcopy(revised["samples"][0]))
    _json(dataset, revised)
    with pytest.raises(ValueError, match="duplicate|IDs"):
        _prepare(case)
    assert not output.exists()


def test_review_version_metadata_is_allowed_and_original_generation_hashes_survive(case, monkeypatch):
    _, dataset, output, runner = case
    revised = _read(dataset)
    revised["samples"][0]["metadata"]["reference_revision"] = {"method": "independent_ai"}
    _json(dataset, revised)
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"generation.py": "later-code-hash"})
    _prepare(case)
    manifest, _, _ = runner._load_run(output)
    assert manifest["source_hashes"] == {"generation.py": "later-code-hash"}
    assert manifest["provenance"]["generation_source_hashes"] == {"generation.py": "original-generation-hash"}


def test_source_changed_during_preparation_is_not_copied(case, monkeypatch):
    source, _, output, runner = case
    original = runner._fingerprint_inputs
    def race(path, config):
        result = original(path, config)
        payload = _read(source / "checkpoint.json")
        payload["updated_at"] = "changed after snapshot"
        _json(source / "checkpoint.json", payload)
        return result
    monkeypatch.setattr(runner, "_fingerprint_inputs", race)
    with pytest.raises(ValueError, match="changed"):
        _prepare(case)
    assert not (output / "checkpoint.json").exists()
