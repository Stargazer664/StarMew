"""Freeze a dishes dataset only after every row has an explicit human review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from backend.eval_comparison.contracts import ComparisonSample, validate_samples


def finalize_review(
    draft_path: Path,
    review_sheet_path: Path,
    output_path: Path,
    *,
    dataset_version: str,
) -> int:
    if not dataset_version.strip():
        raise ValueError("dataset_version is required")
    payload = json.loads(Path(draft_path).read_text(encoding="utf-8-sig"))
    raw_samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(raw_samples, list):
        raise ValueError("draft must contain a samples array")

    sheet_path = Path(review_sheet_path)
    with sheet_path.open(encoding="utf-8-sig", newline="") as handle:
        decisions = list(csv.DictReader(handle))
    by_id: dict[str, dict[str, str]] = {}
    for row in decisions:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id or sample_id in by_id:
            raise ValueError(f"duplicate or empty review sample_id: {sample_id!r}")
        by_id[sample_id] = row

    expected_ids = {str(item.get("id") or "") for item in raw_samples}
    if set(by_id) != expected_ids or len(raw_samples) != len(expected_ids):
        raise ValueError("review sheet sample IDs do not match the draft")

    frozen: list[dict] = []
    for source in raw_samples:
        item = deepcopy(source)
        sample_id = item["id"]
        row = by_id[sample_id]
        decision = (row.get("review_decision") or "").strip().lower()
        if decision not in {"approve", "revise"}:
            raise ValueError(f"{sample_id}: review_decision must be approve or revise")
        reviewer = (row.get("reviewer") or "").strip()
        reviewed_at = (row.get("reviewed_at") or "").strip()
        if not reviewer or not reviewed_at:
            raise ValueError(f"{sample_id}: reviewer and reviewed_at are required")
        try:
            date.fromisoformat(reviewed_at)
        except ValueError as exc:
            raise ValueError(f"{sample_id}: reviewed_at must be YYYY-MM-DD") from exc

        corrected_question = (row.get("corrected_question") or "").strip()
        corrected_ground_truth = (row.get("corrected_ground_truth") or "").strip()
        if decision == "approve" and (corrected_question or corrected_ground_truth):
            raise ValueError(f"{sample_id}: use revise when entering corrections")
        if decision == "revise":
            if not (corrected_question or corrected_ground_truth):
                raise ValueError(f"{sample_id}: revise requires a corrected field")
            item["question"] = corrected_question or item["question"]
            item["ground_truth"] = corrected_ground_truth or item["ground_truth"]

        metadata = item["metadata"]
        metadata["reference_reviewed"] = True
        metadata["reference_facts"] = [item["ground_truth"]]
        metadata["review"] = {
            "status": "human_reviewed",
            "decision": decision,
            "reviewer": reviewer,
            "date": reviewed_at,
            "notes": (row.get("review_notes") or "").strip(),
        }
        frozen.append(item)

    errors = validate_samples(
        [ComparisonSample.model_validate(item) for item in frozen], formal=True
    )
    if errors:
        raise ValueError("formal validation failed: " + "; ".join(errors[:10]))

    output = {
        **payload,
        "dataset_version": dataset_version,
        "review_sheet_sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(),
        "source_draft": str(Path(draft_path).resolve()),
        "samples": frozen,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return len(frozen)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据逐题人工复核表冻结正式菜谱测评数据")
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    count = finalize_review(
        args.draft,
        args.review_sheet,
        args.output,
        dataset_version=args.dataset_version,
    )
    print(f"已冻结 {count} 道人工复核题：{args.output}")


if __name__ == "__main__":
    main()
