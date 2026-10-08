"""Export an evidence-linked, still-unreviewed dishes checklist for a person."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "sample_id",
    "split",
    "question_type",
    "question",
    "ground_truth",
    "source",
    "locator",
    "evidence",
    "review_hint",
    "review_decision",
    "corrected_question",
    "corrected_ground_truth",
    "reviewer",
    "reviewed_at",
    "review_notes",
)


def _review_hint(sample: dict) -> str:
    metadata = sample.get("metadata") or {}
    question_type = metadata.get("question_type")
    if metadata.get("answerability") == "unanswerable":
        return "核查指定文档全文确实缺失所问事实，并确认题面限定该文档。"
    if question_type == "detail":
        return "题面允许任选操作细节；确认参考答案是否覆盖其他同样正确的细节，必要时收窄题面。"
    if question_type == "notes":
        return "确认参考答案是可回答的注意事项，不只是教程链接或其他字段。"
    return "逐项核对题面、答案、单位与来源证据；确认没有遗漏或额外事实。"


def export_review_sheet(dataset_path: Path, output_path: Path) -> int:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8-sig"))
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError("dataset must contain a samples array")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample in samples:
            metadata = sample.get("metadata") or {}
            evidence = metadata.get("evidence") or []
            writer.writerow(
                {
                    "sample_id": sample.get("id", ""),
                    "split": metadata.get("split", ""),
                    "question_type": metadata.get("question_type", ""),
                    "question": sample.get("question", ""),
                    "ground_truth": sample.get("ground_truth", ""),
                    "source": " | ".join(item.get("source", "") for item in evidence),
                    "locator": " | ".join(item.get("locator", "") for item in evidence),
                    "evidence": "\n---\n".join(item.get("text", "") for item in evidence),
                    "review_hint": _review_hint(sample),
                    "review_decision": "",
                    "corrected_question": "",
                    "corrected_ground_truth": "",
                    "reviewer": "",
                    "reviewed_at": "",
                    "review_notes": "",
                }
            )
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出菜谱测评数据人工复核表")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_review_sheet(args.dataset, args.output)
    print(f"已导出 {count} 道待复核题：{args.output}")


if __name__ == "__main__":
    main()
