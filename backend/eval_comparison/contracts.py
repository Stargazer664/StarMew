"""Data contracts for the isolated RAG comparison workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


Answerability = Literal["answerable", "unanswerable"]
Split = Literal["dev", "test", "exploratory"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = ""
    locator: str = ""
    text: str = ""


class SampleMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    domain: str = ""
    question_type: str = "unclassified"
    answerability: str = "answerable"
    split: str = "exploratory"
    cluster_id: str = ""
    reference_reviewed: bool = False
    reference_facts: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class ComparisonSample(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    question: str
    ground_truth: str
    expected_docs: list[str] = Field(default_factory=list)
    metadata: SampleMetadata
    _source_row: int | None = PrivateAttr(default=None)


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = ""
    judge_model: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 1024
    top_k: int = 5
    max_context_chars: int = 8000
    seed: int = 42
    repeat: int = 1
    score_concurrency: int = 2
    generation_timeout: float = 180.0
    stream: bool = False
    retrieval_timeout: float = 300.0
    scoring_timeout: float = 180.0
    max_attempts: int = 3
    metrics: list[str] = Field(
        default_factory=lambda: ["factual_correctness", "abstention"]
    )
    diagnostics: list[str] = Field(default_factory=list)
    prompt_version: str = "rag-comparison-v1"


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: str = ""
    repeat_index: int = 0
    arm: Literal["no_rag", "rag"]
    generation_status: Literal["pending", "running", "success", "failed"]
    answer: str | None = None
    error: str | None = None
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    used_contexts: list[dict[str, Any]] = Field(default_factory=list)
    rag_trace: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_empty: bool | None = None
    timing: dict[str, float | None] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    value: float | None = None
    status: Literal["pending", "running", "success", "failed", "not_applicable"]
    error: str | None = None
    reason: str | None = None
    elapsed_seconds: float | None = None


def _domain_from_path(path: Path) -> str:
    stem = re.sub(r"^eval_samples_?", "", path.stem).strip("_-")
    return stem or "dataset"


def _normalise_legacy_sample(raw: dict[str, Any], *, domain: str, row: int) -> dict[str, Any]:
    metadata = dict(raw.get("metadata") or {})
    ground_truth = raw.get("ground_truth", raw.get("reference", ""))
    answerability = metadata.get("answerability")
    if answerability is None:
        answerability = "unanswerable" if metadata.get("unanswerable") else "answerable"

    metadata.setdefault("domain", domain)
    metadata.setdefault("question_type", "unclassified")
    metadata.setdefault("answerability", answerability)
    metadata.setdefault("split", "exploratory")
    metadata.setdefault("cluster_id", "")
    metadata.setdefault("reference_reviewed", False)
    metadata.setdefault("reference_facts", [])
    metadata.setdefault("evidence", [])

    return {
        **raw,
        "id": str(raw.get("id") or f"{domain}-{row:04d}"),
        "question": str(raw.get("question") or ""),
        "ground_truth": str(ground_truth or ""),
        "expected_docs": list(raw.get("expected_docs") or []),
        "metadata": metadata,
    }


def load_samples(path: Path) -> list[ComparisonSample]:
    """Load current or legacy JSON without importing expensive project modules."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    raw_samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(raw_samples, list):
        raise ValueError(f"dataset must be an array or contain a samples array: {source}")

    domain = _domain_from_path(source)
    samples: list[ComparisonSample] = []
    for row, raw in enumerate(raw_samples, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"row {row}: sample must be an object")
        sample = ComparisonSample.model_validate(
            _normalise_legacy_sample(raw, domain=domain, row=row)
        )
        sample._source_row = row
        samples.append(sample)
    return samples


def validate_samples(samples: list[ComparisonSample], *, formal: bool) -> list[str]:
    """Return deterministic, non-sensitive validation messages."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    cluster_splits: dict[str, set[str]] = {}

    for index, sample in enumerate(samples, start=1):
        row = sample._source_row or index
        prefix = f"{sample.id or '<missing-id>'} (row {row})"
        if not sample.id.strip():
            errors.append(f"{prefix}: id is empty")
        elif sample.id in seen_ids:
            errors.append(f"{sample.id}: duplicate id (row {row})")
        seen_ids.add(sample.id)

        if not sample.question.strip():
            errors.append(f"{sample.id}: question is empty (row {row})")
        if not sample.ground_truth.strip():
            errors.append(f"{sample.id}: ground_truth is empty (row {row})")
        if sample.metadata.answerability not in {"answerable", "unanswerable"}:
            errors.append(f"{sample.id}: answerability must be answerable/unanswerable (row {row})")
        if sample.metadata.split not in {"dev", "test", "exploratory"}:
            errors.append(f"{sample.id}: split must be dev/test/exploratory (row {row})")

        cluster_id = sample.metadata.cluster_id.strip()
        if cluster_id and sample.metadata.split in {"dev", "test"}:
            cluster_splits.setdefault(cluster_id, set()).add(sample.metadata.split)

        if formal:
            if not sample.metadata.reference_reviewed:
                errors.append(f"{sample.id}: reference_reviewed must be true for formal data (row {row})")
            if not cluster_id:
                errors.append(f"{sample.id}: cluster_id is required for formal data (row {row})")
            if sample.metadata.split not in {"dev", "test"}:
                errors.append(f"{sample.id}: split must be dev or test for formal data (row {row})")
            if not sample.metadata.evidence:
                errors.append(f"{sample.id}: evidence is required for formal data (row {row})")
            if sample.metadata.answerability == "answerable" and not sample.metadata.reference_facts:
                errors.append(f"{sample.id}: reference_facts are required for answerable formal data (row {row})")

    for cluster_id, splits in sorted(cluster_splits.items()):
        if splits == {"dev", "test"}:
            errors.append(f"cluster {cluster_id}: cluster crosses dev/test split")
    return errors


def save_frozen_samples(
    path: Path,
    samples: list[ComparisonSample],
    *,
    source: str | None = None,
) -> None:
    """Write a separate stable-ID dataset snapshot; never modifies its source."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": source,
        "total_samples": len(samples),
        "samples": [sample.model_dump(mode="json") for sample in samples],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
