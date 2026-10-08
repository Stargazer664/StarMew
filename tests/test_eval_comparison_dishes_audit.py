import json

import pytest

from scripts.audit_dishes_comparison_dataset import audit_dishes_dataset
from backend.eval_comparison.contracts import load_samples, validate_samples


def _write_source(root, name, *, include_heat=False):
    path = root / "dishes" / "demo" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = "\n每份热量 100 千卡。" if include_heat else ""
    path.write_text(
        f"# {name}\n预估烹饪难度：★\n\n## 操作\n- 小火煮 3 分钟{heat}\n\n## 附加内容\n- 保持简洁\n",
        encoding="utf-8",
    )
    return path


def test_audit_writes_evidence_without_claiming_human_review(tmp_path):
    source_root = tmp_path / "source"
    _write_source(source_root, "a")
    _write_source(source_root, "b")
    payload = {
        "samples": [
            {
                "question": "a 怎么做？",
                "ground_truth": "小火煮 3 分钟；### 参考资料",
                "expected_docs": ["dishes/demo/a.md"],
                "metadata": {"question_type": "process", "unanswerable": False},
            },
            {
                "question": "a 的热量是多少？",
                "ground_truth": "文档未提供热量。",
                "expected_docs": ["dishes/demo/a.md"],
                "metadata": {"question_type": "unanswerable", "unanswerable": True},
            },
            {
                "question": "b 怎么做？",
                "ground_truth": "小火煮 3 分钟",
                "expected_docs": ["dishes/demo/b.md"],
                "metadata": {"question_type": "process", "unanswerable": False},
            },
        ]
    }
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = audit_dishes_dataset(
        input_path, source_root, output_path, dev_target=2, seed=42,
        audit_date="2026-09-22",
    )
    frozen = json.loads(output_path.read_text(encoding="utf-8"))["samples"]

    assert summary["total_samples"] == 3
    assert all(item["metadata"]["reference_reviewed"] is False for item in frozen)
    assert all(item["metadata"]["evidence"] for item in frozen)
    assert "### 参考资料" not in frozen[0]["ground_truth"]
    assert frozen[1]["question"].startswith("根据《a》菜谱文档，")
    assert validate_samples(load_samples(output_path), formal=True)
    a_splits = {item["metadata"]["split"] for item in frozen if item["metadata"]["cluster_id"] == "dishes/demo/a.md"}
    assert len(a_splits) == 1
    assert sum(item["metadata"]["split"] == "dev" for item in frozen) == 2


def test_negative_audit_fails_if_requested_fact_exists(tmp_path):
    source_root = tmp_path / "source"
    _write_source(source_root, "a", include_heat=True)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"samples": [{
            "question": "a 的热量是多少？",
            "ground_truth": "文档未提供热量。",
            "expected_docs": ["dishes/demo/a.md"],
            "metadata": {"question_type": "unanswerable", "unanswerable": True},
        }]}, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="negative evidence check failed"):
        audit_dishes_dataset(
            input_path, source_root, tmp_path / "output.json", dev_target=1,
            seed=42, audit_date="2026-09-22",
        )
