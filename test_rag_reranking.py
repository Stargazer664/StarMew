"""Rerank behavior with real pipeline code and isolated model/API responses."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from backend import rag_utils as rag


@pytest.fixture(autouse=True)
def disable_relevance_thresholds(monkeypatch):
    """Keep legacy ranking behavior independent of local threshold settings."""
    monkeypatch.setenv("LLM_RERANK_MIN_SCORE", "")
    monkeypatch.setenv("RERANK_MIN_SCORE", "")


@pytest.fixture
def external_response(monkeypatch):
    monkeypatch.setattr(rag, "RERANK_MODEL", "test-reranker")
    monkeypatch.setattr(rag, "RERANK_BINDING_HOST", "https://example.invalid")
    monkeypatch.setattr(rag, "RERANK_API_KEY", "test-key")

    def provide(body=None, status=200, error=None):
        def post(*args, **kwargs):
            if error is not None:
                raise error
            return SimpleNamespace(status_code=status, text="test response", json=lambda: body)
        monkeypatch.setattr(rag.requests, "post", post)
    return provide


@pytest.mark.parametrize("items", [[], [{"doc_id": 1}], [{"doc_id": 99, "score": 8}]])
def test_llm_without_valid_scores_falls_back_without_success(docs, llm_scores, items):
    llm_scores(items)
    result, meta = rag._rerank_documents("question", docs, 2)
    assert [doc["chunk_id"] for doc in result] == ["A", "B"]
    assert meta["rerank_applied"] is False
    assert meta["rerank_error"]


@pytest.mark.parametrize("body", [None, [], {}, {"results": []}, {"results": None},
    {"results": [{"index": 1}]}, {"results": [{"index": True, "relevance_score": 9}]},
    {"results": [{"index": 0, "relevance_score": float("nan")}]},
    {"results": [None, "invalid"]}])
def test_invalid_external_results_do_not_claim_success(docs, external_response, body):
    external_response(body)
    result, meta = rag._rerank_documents("question", docs, 2)
    assert [doc["chunk_id"] for doc in result] == ["A", "B"]
    assert meta["rerank_applied"] is False
    assert meta["rerank_error"]


@pytest.mark.parametrize("failure", ["http", "timeout"])
def test_external_failures_keep_recall_and_false_success(docs, external_response, failure):
    external_response(status=503, error=rag.requests.Timeout("timeout") if failure == "timeout" else None)
    result, meta = rag._rerank_documents("question", docs, 2)
    assert [doc["chunk_id"] for doc in result] == ["A", "B"]
    assert meta["rerank_applied"] is False
    assert meta["rerank_error"]


def test_external_scores_are_validated_deduplicated_and_sorted(docs, external_response):
    external_response({"results": [
        {"index": 0, "relevance_score": 0.1},
        {"index": 0, "relevance_score": 0.2},
        {"index": 2, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.9},
        {"index": 3, "relevance_score": "99"}, None,
    ]})
    result, meta = rag._rerank_documents("question", docs, 3)
    assert [doc["chunk_id"] for doc in result] == ["B", "C", "A"]
    assert [doc["rerank_score"] for doc in result] == [0.9, 0.9, 0.2]
    assert meta["rerank_applied"] is True
    assert meta["rerank_error"] is None


def test_valid_llm_scores_claim_success(docs, llm_scores):
    llm_scores([{"doc_id": 2, "score": 0}])
    result, meta = rag._rerank_documents("question", docs, 1)
    assert result[0]["chunk_id"] == "B"
    assert meta["rerank_applied"] is True
    assert meta["rerank_error"] is None


def test_llm_initialization_failure_preserves_recall(docs, monkeypatch):
    monkeypatch.setattr(rag, "RERANK_MODEL", None)
    def unavailable():
        raise RuntimeError("model configuration unavailable")
    monkeypatch.setattr(rag, "_get_stepback_model", unavailable)
    result, meta = rag._rerank_documents("question", docs, 2)
    assert [doc["chunk_id"] for doc in result] == ["A", "B"]
    assert meta["rerank_applied"] is False
    assert "model configuration unavailable" in meta["rerank_error"]


@pytest.fixture
def docs():
    return [
        {"chunk_id": "A", "text": "alpha", "score": 0.9, "rrf_rank": 1},
        {"chunk_id": "B", "text": "beta", "score": 0.6, "rrf_rank": 2},
        {"chunk_id": "C", "text": "gamma", "score": 0.3, "rrf_rank": 3},
        {"chunk_id": "D", "text": "delta", "score": 0.1, "rrf_rank": 4},
    ]


@pytest.fixture
def llm_scores(monkeypatch):
    monkeypatch.setattr(rag, "RERANK_MODEL", None)

    def provide(items):
        model = SimpleNamespace(invoke=lambda prompt, **kwargs: SimpleNamespace(content=json.dumps(items)))
        monkeypatch.setattr(rag, "_get_stepback_model", lambda: model)

    return provide


def test_llm_scores_are_sorted_before_top_k(docs, llm_scores):
    llm_scores([{"doc_id": 1, "score": 1}, {"doc_id": 2, "score": 10}])
    result, _ = rag._rerank_documents("question", docs, 1)
    assert [doc["chunk_id"] for doc in result] == ["B"]


def test_equal_scores_keep_original_recall_order(docs, llm_scores):
    llm_scores([{"doc_id": 3, "score": 8}, {"doc_id": 1, "score": 8}])
    result, _ = rag._rerank_documents("question", docs, 2)
    assert [doc["chunk_id"] for doc in result] == ["A", "C"]


def test_duplicate_ids_use_best_score_and_do_not_occupy_multiple_slots(docs, llm_scores):
    llm_scores([
        {"doc_id": 1, "score": 2}, {"doc_id": 1, "score": 9},
        {"doc_id": 2, "score": 8},
    ])
    result, _ = rag._rerank_documents("question", docs, 2)
    assert [doc["chunk_id"] for doc in result] == ["A", "B"]
    assert result[0]["llm_rerank_score"] == 9


@pytest.mark.parametrize("invalid", [
    {"doc_id": 1}, {"score": 10}, {"doc_id": True, "score": 10},
    {"doc_id": "1", "score": 10}, {"doc_id": 1.0, "score": 10},
    {"doc_id": 0, "score": 10}, {"doc_id": 99, "score": 10},
    {"doc_id": 1, "score": float("nan")}, {"doc_id": 1, "score": float("inf")},
    {"doc_id": 1, "score": True}, {"doc_id": 1, "score": "10"},
    {"doc_id": 1, "score": -1}, {"doc_id": 1, "score": 11}, None, "invalid",
])
def test_invalid_entry_does_not_discard_valid_scores(docs, llm_scores, invalid):
    llm_scores([invalid, {"doc_id": 2, "score": 3}])
    result, _ = rag._rerank_documents("question", docs, 1)
    assert [doc["chunk_id"] for doc in result] == ["B"]


def test_zero_score_is_valid_and_unscored_candidates_fill_in_recall_order(docs, llm_scores):
    llm_scores([{"doc_id": 3, "score": 0}])
    before = deepcopy(docs)
    result, _ = rag._rerank_documents("question", docs, 3)
    assert [doc["chunk_id"] for doc in result] == ["C", "A", "B"]
    assert result[0]["llm_rerank_score"] == 0
    assert docs == before


@pytest.fixture
def parent_map(monkeypatch):
    parents = {
        "P": {"chunk_id": "P", "text": "parent P", "parent_chunk_id": "R"},
        "Q": {"chunk_id": "Q", "text": "parent Q", "parent_chunk_id": "R"},
        "R": {"chunk_id": "R", "text": "root R", "parent_chunk_id": ""},
    }
    monkeypatch.setattr(rag, "AUTO_MERGE_ENABLED", True)
    monkeypatch.setattr(rag, "AUTO_MERGE_THRESHOLD", 2)
    monkeypatch.setattr(rag, "_parent_chunk_store", SimpleNamespace(
        get_documents_by_ids=lambda ids: [deepcopy(parents[key]) for key in ids if key in parents],
    ))
    return parents


@pytest.mark.parametrize("field", ["rerank_score", "llm_rerank_score"])
def test_merge_enabled_preserves_rerank_order_without_matching_parents(docs, parent_map, field):
    ranked = [dict(docs[1], **{field: 9}), dict(docs[0], **{field: 1})]
    before = deepcopy(ranked)
    result, meta = rag._auto_merge_documents(ranked, 2)
    assert [doc["chunk_id"] for doc in result] == ["B", "A"]
    assert meta["auto_merge_applied"] is False
    assert ranked == before


@pytest.mark.parametrize("field", ["rerank_score", "llm_rerank_score"])
@pytest.mark.parametrize("reverse_children", [False, True])
def test_parent_aggregates_child_scores_and_keeps_rank(docs, parent_map, field, reverse_children):
    children = [dict(docs[1], parent_chunk_id="P", **{field: 9}),
                dict(docs[0], parent_chunk_id="P", **{field: 1})]
    if reverse_children:
        children.reverse()
    independent = dict(docs[2], **{field: 5})
    before = deepcopy(children)
    result, _ = rag._auto_merge_documents(children + [independent], 3)
    assert [doc["chunk_id"] for doc in result] == ["P", "C"]
    assert result[0][field] == 9
    assert result[0]["score"] == 0.9
    assert result[0]["rrf_rank"] == 1
    assert result[0]["merged_child_count"] == 2
    assert children == before


def test_two_level_merge_preserves_best_leaf_scores(docs, parent_map):
    children = [dict(doc, parent_chunk_id="P" if i < 2 else "Q", rerank_score=score)
                for i, (doc, score) in enumerate(zip(docs, [1, 9, 2, 8]))]
    result, meta = rag._auto_merge_documents(children, 4)
    assert [doc["chunk_id"] for doc in result] == ["R"]
    assert result[0]["rerank_score"] == 9
    assert result[0]["score"] == 0.9
    assert result[0]["rrf_rank"] == 1
    assert meta["auto_merge_steps"] == 2


@pytest.mark.parametrize("score", [0, -0.5])
def test_valid_rerank_score_precedes_unscored_recall_result(docs, parent_map, score):
    candidates = [docs[0], dict(docs[1], rerank_score=score)]
    result, _ = rag._auto_merge_documents(candidates, 2)
    assert [doc["chunk_id"] for doc in result] == ["B", "A"]


def test_invalid_rerank_score_falls_back_to_recall_ranking(docs, parent_map):
    candidates = [dict(docs[1], rerank_score=float("nan")), docs[0]]
    result, _ = rag._auto_merge_documents(candidates, 2)
    assert [doc["chunk_id"] for doc in result] == ["A", "B"]


def test_equal_merged_scores_use_best_original_rank(docs, parent_map):
    children = [dict(docs[1], parent_chunk_id="P", rerank_score=8),
                dict(docs[0], parent_chunk_id="P", rerank_score=8)]
    result, _ = rag._auto_merge_documents([dict(docs[2], rerank_score=8)] + children, 3)
    assert [doc["chunk_id"] for doc in result] == ["P", "C"]
