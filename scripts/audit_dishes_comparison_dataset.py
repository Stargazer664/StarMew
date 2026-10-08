"""Create an evidence-linked dishes draft from the legacy generated set.

This command never modifies the legacy input. It validates every referenced source,
attaches the relevant source section, checks every negative question against the
full document, and keeps each source cluster wholly in dev or test. Automatic
checks do not constitute human reference review.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.eval_comparison.contracts import load_samples, validate_samples  # noqa: E402


SECTION_BY_TYPE = {
    "ingredients": "必备原料和工具",
    "calculation": "计算",
    "process": "操作",
    "detail": "操作",
    "notes": "附加内容",
}

MISSING_FACT_PATTERNS = {
    "热量": r"热量|卡路里|千卡|大卡",
    "保存期限": r"保存期限|保质期|保存时间|冷藏|冷冻保存",
    "作者": r"作者|创作者|发布者",
    "起源": r"起源|发源地|历史",
    "成本": r"成本|价格|售价|花费",
    "营养成分": r"营养成分|蛋白质|脂肪|碳水",
}


def _extract_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^##\s*{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _difficulty_evidence(text: str) -> str:
    intro_match = re.search(r"^#\s+.+?$([\s\S]*?)(?=^##\s+|\Z)", text, re.MULTILINE)
    intro = intro_match.group(1).strip() if intro_match else ""
    return intro


def _clean_reference(value: str) -> str:
    parts = []
    for part in (item.strip() for item in value.split("；")):
        if not part or part.startswith("###") or part.lower().startswith("how to "):
            continue
        parts.append(part)
    return "；".join(parts)


def _choose_dev_clusters(
    groups: dict[str, list[dict[str, Any]]], *, target: int, seed: int
) -> set[str]:
    question_types = sorted(
        {item["metadata"]["question_type"] for values in groups.values() for item in values}
    )
    type_bits = {name: 1 << index for index, name in enumerate(question_types)}
    entries = list(groups.items())
    random.Random(seed).shuffle(entries)
    # State: (sample count, type mask, negative count capped at 2) -> cluster list.
    states: dict[tuple[int, int, int], list[str]] = {(0, 0, 0): []}
    for cluster, values in entries:
        size = len(values)
        mask = 0
        negatives = 0
        for item in values:
            mask |= type_bits[item["metadata"]["question_type"]]
            negatives += bool(item["metadata"].get("unanswerable"))
        for (count, current_mask, current_negatives), selected in list(states.items()):
            if count + size > target:
                continue
            key = (
                count + size,
                current_mask | mask,
                min(2, current_negatives + negatives),
            )
            states.setdefault(key, selected + [cluster])
    candidates = [
        (key, selected) for key, selected in states.items() if key[0] == target
    ]
    if not candidates:
        raise ValueError(f"cannot form a cluster-complete dev split of {target} samples")
    _, best = max(
        candidates,
        key=lambda item: (item[0][1].bit_count(), item[0][2], -len(item[1])),
    )
    return set(best)


def audit_dishes_dataset(
    input_path: Path,
    source_root: Path,
    output_path: Path,
    *,
    dev_target: int = 20,
    seed: int = 42,
    audit_date: str,
) -> dict[str, Any]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8-sig"))
    raw_samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(raw_samples, list):
        raise ValueError("input dataset must contain a samples array")

    reviewed: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_samples, start=1):
        item = dict(raw)
        expected_docs = list(item.get("expected_docs") or [])
        if len(expected_docs) != 1:
            raise ValueError(f"row {index}: exactly one expected document is required")
        source_name = expected_docs[0].replace("\\", "/")
        source_path = Path(source_root) / Path(source_name)
        if not source_path.is_file():
            raise ValueError(f"row {index}: source does not exist: {source_name}")
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
        metadata = dict(item.get("metadata") or {})
        unanswerable = bool(metadata.get("unanswerable"))
        question_type = str(metadata.get("question_type") or "unclassified")
        question = str(item.get("question") or "").strip()
        ground_truth = _clean_reference(str(item.get("ground_truth") or ""))
        if not ground_truth:
            raise ValueError(f"row {index}: ground truth is empty after review cleanup")

        if unanswerable:
            topic = next((name for name in MISSING_FACT_PATTERNS if name in question), None)
            if topic is None:
                raise ValueError(f"row {index}: unsupported negative question topic")
            if re.search(MISSING_FACT_PATTERNS[topic], source_text, re.IGNORECASE):
                raise ValueError(
                    f"row {index}: negative evidence check failed; source contains {topic}"
                )
            locator = "全文核查"
            evidence_text = f"核查范围：全文；缺失事实：{topic}"
            if not re.match(r"^(按|根据).*(菜谱|文档)", question):
                question = f"根据《{Path(source_name).stem}》菜谱文档，{question}"
        else:
            heading = SECTION_BY_TYPE.get(question_type)
            if heading:
                evidence_text = _extract_section(source_text, heading)
                locator = f"## {heading}"
            elif question_type == "difficulty":
                evidence_text = _difficulty_evidence(source_text)
                locator = "导语与预估烹饪难度"
            else:
                raise ValueError(f"row {index}: unsupported question type {question_type}")
            if not evidence_text:
                raise ValueError(f"row {index}: source evidence section is empty")

        reviewed.append(
            {
                "id": f"dishes-{index:04d}",
                "question": question,
                "ground_truth": ground_truth,
                "expected_docs": [source_name],
                "metadata": {
                    **metadata,
                    "domain": "dishes",
                    "question_type": question_type,
                    "answerability": "unanswerable" if unanswerable else "answerable",
                    "cluster_id": source_name,
                    "reference_reviewed": False,
                    "reference_facts": [ground_truth],
                    "evidence": [
                        {
                            "source": source_name,
                            "locator": locator,
                            "text": evidence_text,
                        }
                    ],
                    "review": {
                        "date": audit_date,
                        "method": "automated source-section and negative full-document check",
                        "status": "pending_human_review",
                    },
                },
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reviewed:
        groups[item["metadata"]["cluster_id"]].append(item)
    dev_clusters = _choose_dev_clusters(groups, target=dev_target, seed=seed)
    for item in reviewed:
        item["metadata"]["split"] = (
            "dev" if item["metadata"]["cluster_id"] in dev_clusters else "test"
        )

    output = {
        "schema_version": 1,
        "dataset_version": f"dishes-evidence-draft-{audit_date}-v2",
        "audit_date": audit_date,
        "source": str(Path(input_path).resolve()),
        "source_root": str(Path(source_root).resolve()),
        "total_samples": len(reviewed),
        "samples": reviewed,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    loaded = load_samples(destination)
    errors = validate_samples(loaded, formal=False)
    if errors:
        raise ValueError("draft validation failed: " + "; ".join(errors[:10]))

    splits = Counter(item["metadata"]["split"] for item in reviewed)
    types = Counter(item["metadata"]["question_type"] for item in reviewed)
    return {
        "output": str(destination),
        "total_samples": len(reviewed),
        "dev_samples": splits["dev"],
        "test_samples": splits["test"],
        "unanswerable_samples": sum(
            item["metadata"]["answerability"] == "unanswerable" for item in reviewed
        ),
        "question_types": dict(sorted(types.items())),
        "source_clusters": len(groups),
        "reviewed_samples": 0,
        "human_review_required": len(reviewed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成待人工复核的 dishes 配对测评草稿")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-target", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-date", default="2026-09-22")
    args = parser.parse_args()
    summary = audit_dishes_dataset(
        args.input,
        args.source_root,
        args.output,
        dev_target=args.dev_target,
        seed=args.seed,
        audit_date=args.audit_date,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
