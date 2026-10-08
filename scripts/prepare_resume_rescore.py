"""Fork completed, never-scored generation for reference-only first scoring.

This command never initializes a model and never writes to the source run.
It does not support changing questions, evidence, retrieval labels, or splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.eval_comparison.contracts import ComparisonConfig, load_samples, validate_samples  # noqa: E402
from backend.eval_comparison.storage import canonical_fingerprint, save_checkpoint, utc_now  # noqa: E402
import scripts.run_rag_comparison as runner  # noqa: E402


# Everything else at sample level, including all evidence fields, is immutable.
REVIEW_METADATA = frozenset({
    "reference_facts", "reference_reviewed", "review_method", "review_status", "review",
    "human_reviewer", "human_reviewed_at", "reference_revision", "dataset_version",
})
EVALUATION_FIELDS = frozenset({
    "metrics", "metric_status", "metric_errors", "metric_reasons", "scoring_attempts",
})


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(value: bytes, label: str) -> dict[str, Any]:
    parsed = json.loads(value.decode("utf-8-sig"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _sample_map(path: Path) -> dict[str, dict[str, Any]]:
    samples = load_samples(path)
    errors = validate_samples(samples, formal=False)
    if errors:
        raise ValueError("invalid dataset: " + "; ".join(errors[:10]))
    if not samples:
        raise ValueError("dataset sample IDs must not be empty")
    return {sample.id: sample.model_dump(mode="json") for sample in samples}


def _reference_changes(source: Path, revised: Path) -> list[str]:
    original, updated = _sample_map(source), _sample_map(revised)
    if set(original) != set(updated):
        raise ValueError("dataset sample IDs differ")
    changed: list[str] = []
    for sample_id, old in original.items():
        new = updated[sample_id]
        old_fixed, new_fixed = deepcopy(old), deepcopy(new)
        old_fixed.pop("ground_truth")
        new_fixed.pop("ground_truth")
        for item in (old_fixed, new_fixed):
            for field in REVIEW_METADATA:
                item["metadata"].pop(field, None)
        if old_fixed != new_fixed:
            raise ValueError(f"{sample_id}: change outside reference/review metadata")
        if (old["ground_truth"] != new["ground_truth"] or
                old["metadata"]["reference_facts"] != new["metadata"]["reference_facts"]):
            changed.append(sample_id)
    if not changed:
        raise ValueError("no reference correction was supplied")
    return changed


def _key(item: dict[str, Any]) -> tuple[str, int, str]:
    sample_id, repeat, arm = item.get("sample_id"), item.get("repeat_index"), item.get("arm")
    if (not isinstance(sample_id, str) or not sample_id or type(repeat) is not int
            or repeat < 0 or arm not in {"no_rag", "rag"}):
        raise ValueError("invalid execution key in plan/record")
    return sample_id, repeat, arm


def _validate_generation(manifest: dict[str, Any], payload: dict[str, Any],
                         sample_ids: set[str], config: ComparisonConfig) -> None:
    plan, records = manifest.get("plan"), payload.get("records")
    if not isinstance(plan, list) or not isinstance(records, list):
        raise ValueError("plan and checkpoint records must be arrays")
    expected = {(sample_id, repeat, arm) for sample_id in sample_ids
                for repeat in range(config.repeat) for arm in ("no_rag", "rag")}
    plan_keys, record_keys = [_key(item) for item in plan], [_key(item) for item in records]
    if not expected or len(plan_keys) != len(set(plan_keys)) or set(plan_keys) != expected:
        raise ValueError("plan is not a complete paired execution plan")
    if len(record_keys) != len(set(record_keys)) or set(record_keys) != expected:
        raise ValueError("checkpoint records do not match the complete execution plan")
    selected = manifest.get("selected_sample_ids")
    if not isinstance(selected, list) or len(selected) != len(set(selected)) or set(selected) != sample_ids:
        raise ValueError("manifest selected sample IDs differ from dataset")
    for record in records:
        if record.get("generation_status") not in {"success", "failed"}:
            raise ValueError(f"{_key(record)}: generation is not complete")
        if (any(value is not None for value in record.get("metrics", {}).values())
                or any(status != "pending" for status in record.get("metric_status", {}).values())
                or record.get("metric_errors") or record.get("metric_reasons")
                or record.get("scoring_attempts")
                or record.get("timing", {}).get("evaluation_seconds") is not None):
            raise ValueError(f"{_key(record)}: prior scoring evidence; source must never have been scored")


def _generation_projection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = deepcopy(records)
    for record in projected:
        for field in EVALUATION_FIELDS:
            record.pop(field, None)
        record.get("timing", {}).pop("evaluation_seconds", None)
    return projected


def prepare_rescore(source_run: Path, dataset: Path, output_run: Path) -> dict[str, Any]:
    """Create a new score-ready run; reject active, scored or changed source data."""
    source_run, dataset, output_run = (Path(value).resolve() for value in (source_run, dataset, output_run))
    if output_run == source_run or output_run.is_relative_to(source_run):
        raise ValueError("output run must be separate from the source run")
    if output_run.exists():
        raise FileExistsError(f"output run already exists: {output_run}")
    if (source_run / ".run.lock").exists():
        raise ValueError("source run is locked; wait for generation to finish")
    paths = {name: source_run / name for name in ("manifest.json", "dataset.json", "checkpoint.json")}
    snapshots = {name: path.read_bytes() for name, path in paths.items()}
    revised_bytes = dataset.read_bytes()
    manifest = _object(snapshots["manifest.json"], "source manifest")
    payload = _object(snapshots["checkpoint.json"], "source checkpoint")
    revised_payload = _object(revised_bytes, "revised dataset")
    inputs = manifest.get("fingerprint_inputs")
    if not isinstance(inputs, dict) or canonical_fingerprint(inputs) != manifest.get("fingerprint"):
        raise ValueError("source manifest fingerprint does not match its inputs")
    if payload.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("source checkpoint fingerprint does not match manifest")
    dataset_hash = _digest(snapshots["dataset.json"])
    if manifest.get("dataset_sha256") != dataset_hash or inputs.get("dataset_sha256") != dataset_hash:
        raise ValueError("source dataset fingerprint does not match manifest")
    if manifest.get("config") != inputs.get("config"):
        raise ValueError("source config does not match fingerprint inputs")
    if manifest.get("source_hashes") != inputs.get("source_hashes"):
        raise ValueError("source code hashes do not match fingerprint inputs")
    if payload.get("run_id") != manifest.get("run_id"):
        raise ValueError("source checkpoint run ID differs from manifest")

    config = ComparisonConfig.model_validate(manifest["config"])
    changed = _reference_changes(paths["dataset.json"], dataset)
    sample_ids = set(_sample_map(paths["dataset.json"]))
    _validate_generation(manifest, payload, sample_ids, config)

    # Fingerprinting is read-only; it uses the revised file bytes which will be
    # copied unchanged into output_run/dataset.json. No temporary source edits.
    new_inputs = runner._fingerprint_inputs(dataset, config)
    new_fingerprint = canonical_fingerprint(new_inputs)
    if (source_run / ".run.lock").exists() or any(
            path.read_bytes() != snapshots[name] for name, path in paths.items()):
        raise ValueError("source run changed during preparation")
    if dataset.read_bytes() != revised_bytes or new_inputs["dataset_sha256"] != _digest(revised_bytes):
        raise ValueError("revised dataset changed during preparation")

    records = deepcopy(payload["records"])
    for record in records:
        record["metrics"] = {name: None for name in config.metrics}
        record["metric_status"] = {name: "pending" for name in config.metrics}
        record["metric_errors"] = {}
        record["metric_reasons"] = {}
        record["scoring_attempts"] = []
        record.setdefault("timing", {})["evaluation_seconds"] = None
    generation_hash = canonical_fingerprint(_generation_projection(payload["records"]))
    if canonical_fingerprint(_generation_projection(records)) != generation_hash:
        raise ValueError("generation content changed while resetting evaluation")

    created = utc_now()
    new_manifest = {
        **deepcopy(manifest),
        "run_id": output_run.name,
        "created_at": created,
        "dataset_source": str(dataset),
        "dataset_sha256": new_inputs["dataset_sha256"],
        "source_hashes": new_inputs["source_hashes"],
        "corpus_identity": new_inputs["corpus_identity"],
        "ragas_version": new_inputs["ragas_version"],
        "prompt_sha256": new_inputs["prompt_sha256"],
        "config": config.model_dump(mode="json"),
        "fingerprint_inputs": new_inputs,
        "fingerprint": new_fingerprint,
        "git": runner._git_identity(),
        "generation_reused": True,
        "purpose": "reference-only correction before first scoring",
        "reference_dataset_version": revised_payload.get("dataset_version"),
        "provenance": {
            "source_run": str(source_run),
            "source_run_id": manifest["run_id"],
            "source_manifest": str(paths["manifest.json"]),
            "source_manifest_sha256": _digest(snapshots["manifest.json"]),
            "source_checkpoint": str(paths["checkpoint.json"]),
            "source_checkpoint_sha256": _digest(snapshots["checkpoint.json"]),
            "source_dataset": str(paths["dataset.json"]),
            "source_dataset_sha256": dataset_hash,
            "revised_dataset": str(dataset),
            "revised_dataset_sha256": _digest(revised_bytes),
            "source_fingerprint": manifest["fingerprint"],
            "generation_source_hashes": deepcopy(manifest["source_hashes"]),
            "generation_fingerprint_inputs": deepcopy(inputs),
            "generation_records_sha256": generation_hash,
            "generation_reused": True,
            "generation_rerun": False,
            "source_was_unscored": True,
            "changed_reference_sample_ids": changed,
            "reference_revision": revised_payload.get("reference_revision"),
            "preparation_script_sha256": _digest(Path(__file__).read_bytes()),
        },
    }
    new_payload = {
        **deepcopy(payload), "run_id": output_run.name, "fingerprint": new_fingerprint,
        "updated_at": created, "records": records,
    }
    output_run.mkdir(parents=True, exist_ok=False)
    with (output_run / "dataset.json").open("xb") as handle:
        handle.write(revised_bytes)
    runner._write_manifest(output_run / "manifest.json", new_manifest)
    save_checkpoint(output_run, new_payload)
    return new_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为首次评分前的参考答案修订复制已完成生成，不重新调用模型")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_rescore(args.source_run, args.dataset, args.output_run)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"准备失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"已准备 {len(manifest['plan'])} 条原始生成记录供首次评分：{args.output_run}")
    print("未重新生成；源运行未改写。使用 run_rag_comparison.py resume --phase score 继续。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
