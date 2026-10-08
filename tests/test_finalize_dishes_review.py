import csv
import json

import pytest

from backend.eval_comparison.contracts import load_samples, validate_samples
from scripts.export_dishes_review_sheet import export_review_sheet
from scripts.finalize_dishes_review import finalize_review


def _draft_and_sheet(tmp_path):
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "draft-v2",
                "samples": [
                    {
                        "id": "q1",
                        "question": "文档中的关键操作是什么？",
                        "ground_truth": "小火煮三分钟",
                        "expected_docs": ["dishes/a.md"],
                        "metadata": {
                            "domain": "dishes",
                            "question_type": "detail",
                            "answerability": "answerable",
                            "split": "test",
                            "cluster_id": "dishes/a.md",
                            "reference_reviewed": False,
                            "reference_facts": ["小火煮三分钟"],
                            "evidence": [
                                {"source": "dishes/a.md", "locator": "## 操作", "text": "小火煮三分钟"}
                            ],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sheet = tmp_path / "review.csv"
    export_review_sheet(draft, sheet)
    return draft, sheet


def _edit_sheet(sheet, **changes):
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    rows[0].update(changes)
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_finalize_requires_explicit_human_decision(tmp_path):
    draft, sheet = _draft_and_sheet(tmp_path)
    output = tmp_path / "formal.json"

    with pytest.raises(ValueError, match="q1.*review_decision"):
        finalize_review(draft, sheet, output, dataset_version="formal-v3")

    assert not output.exists()


def test_finalize_applies_reviewed_question_and_validates_formal_data(tmp_path):
    draft, sheet = _draft_and_sheet(tmp_path)
    _edit_sheet(
        sheet,
        review_decision="revise",
        corrected_question="根据文档，小火应煮几分钟？",
        corrected_ground_truth="小火煮三分钟",
        reviewer="reviewer-a",
        reviewed_at="2026-09-25",
    )
    output = tmp_path / "formal.json"

    assert finalize_review(draft, sheet, output, dataset_version="formal-v3") == 1

    sample = load_samples(output)[0]
    assert sample.question == "根据文档，小火应煮几分钟？"
    assert sample.metadata.reference_reviewed is True
    assert validate_samples([sample], formal=True) == []
