import json
from pathlib import Path

import pytest

from backend.eval_comparison.contracts import (
    ComparisonSample,
    load_samples,
    validate_samples,
)


FIXTURE = Path(__file__).parent / "fixtures" / "rag_comparison.json"


def _sample(sample_id: str = "q1", **metadata_overrides) -> ComparisonSample:
    metadata = {
        "domain": "synthetic",
        "question_type": "fact",
        "answerability": "answerable",
        "split": "dev",
        "cluster_id": "cluster-a",
        "reference_reviewed": True,
        "reference_facts": ["答案是甲"],
        "evidence": [{"source": "fixture.md", "locator": "L1", "text": "甲"}],
    }
    metadata.update(metadata_overrides)
    return ComparisonSample(
        id=sample_id,
        question="问题",
        ground_truth="答案",
        expected_docs=["fixture.md"],
        metadata=metadata,
    )


def test_loads_formal_fixture_without_errors():
    samples = load_samples(FIXTURE)
    assert [sample.id for sample in samples] == ["synthetic-0001", "synthetic-0002"]
    assert validate_samples(samples, formal=True) == []


def test_legacy_reference_is_converted_without_claiming_review(tmp_path):
    path = tmp_path / "eval_samples_legacy.json"
    path.write_text(
        json.dumps(
            [{
                "question": "旧问题",
                "reference": "旧答案",
                "metadata": {"unanswerable": False},
            }],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sample = load_samples(path)[0]

    assert sample.id == "legacy-0001"
    assert sample.ground_truth == "旧答案"
    assert sample.metadata.answerability == "answerable"
    assert sample.metadata.reference_reviewed is False
    assert sample.metadata.split == "exploratory"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda samples: setattr(samples[0], "question", "  "), "q1: question"),
        (lambda samples: setattr(samples[0], "ground_truth", ""), "q1: ground_truth"),
        (lambda samples: setattr(samples[0].metadata, "answerability", "maybe"), "q1: answerability"),
    ],
)
def test_validation_rejects_invalid_core_fields(mutate, expected):
    samples = [_sample()]
    mutate(samples)
    assert any(expected in error for error in validate_samples(samples, formal=False))


def test_validation_rejects_duplicate_ids():
    errors = validate_samples([_sample("same"), _sample("same")], formal=False)
    assert any("same: duplicate id" in error for error in errors)


def test_validation_rejects_cluster_split_leakage():
    samples = [_sample("dev"), _sample("test", split="test")]
    errors = validate_samples(samples, formal=False)
    assert any("cluster-a" in error and "dev/test" in error for error in errors)


def test_formal_validation_requires_review_cluster_and_evidence():
    sample = _sample(
        reference_reviewed=False,
        cluster_id="",
        split="exploratory",
        evidence=[],
    )
    errors = validate_samples([sample], formal=True)
    assert any("q1: reference_reviewed" in error for error in errors)
    assert any("q1: cluster_id" in error for error in errors)
    assert any("q1: split" in error for error in errors)
    assert any("q1: evidence" in error for error in errors)


def test_errors_identify_row_without_echoing_secret(tmp_path):
    path = tmp_path / "unsafe.json"
    path.write_text(
        json.dumps([{"id": "bad-row", "question": "", "ground_truth": "sk-secret-value"}]),
        encoding="utf-8",
    )
    sample = load_samples(path)[0]
    errors = validate_samples([sample], formal=False)
    rendered = "\n".join(errors)
    assert "bad-row" in rendered
    assert "row 1" in rendered
    assert "sk-secret-value" not in rendered


def test_existing_datasets_load_without_model_initialization():
    root = Path(__file__).parents[1] / "data"
    counts = {
        name: len(load_samples(root / name))
        for name in (
            "eval_samples_dishes.json",
            "eval_samples_hello_agents.json",
            "eval_samples_pico.json",
        )
    }
    assert counts == {
        "eval_samples_dishes.json": 120,
        "eval_samples_hello_agents.json": 27,
        "eval_samples_pico.json": 20,
    }
