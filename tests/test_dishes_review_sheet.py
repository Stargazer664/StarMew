import csv

from scripts.export_dishes_review_sheet import export_review_sheet


def test_review_sheet_lists_evidence_and_leaves_decisions_blank(tmp_path):
    dataset = tmp_path / "draft.json"
    dataset.write_text(
        '{"samples": [{"id": "q1", "question": "根据菜谱，热量是多少？", '
        '"ground_truth": "文档未提供", "expected_docs": ["dishes/a.md"], '
        '"metadata": {"split": "test", "question_type": "unanswerable", '
        '"answerability": "unanswerable", "reference_reviewed": false, '
        '"evidence": [{"source": "dishes/a.md", "locator": "全文核查", '
        '"text": "缺失热量"}]}}]}',
        encoding="utf-8",
    )
    output = tmp_path / "review.csv"

    assert export_review_sheet(dataset, output) == 1

    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == "q1"
    assert rows[0]["source"] == "dishes/a.md"
    assert rows[0]["evidence"] == "缺失热量"
    assert rows[0]["review_decision"] == ""
    assert "全文" in rows[0]["review_hint"]
