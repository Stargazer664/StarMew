"""Exercise real retrieval and index gating against isolated in-memory storage."""
import json
import threading
from types import SimpleNamespace

import pytest

from backend import rag_utils as rag
from backend.document_transactions import RecoveryRequiredError, assert_index_available


@pytest.fixture(autouse=True)
def disable_relevance_thresholds(monkeypatch):
    """Keep legacy snapshot behavior independent of local threshold settings."""
    monkeypatch.setenv("LLM_RERANK_MIN_SCORE", "")
    monkeypatch.setenv("RERANK_MIN_SCORE", "")


@pytest.fixture
def storage(monkeypatch, tmp_path):
    leaves = [
        {"chunk_id": key, "text": "old leaf " + key, "score": 1 - i / 10,
         "parent_chunk_id": "P" if i < 2 else "Q"}
        for i, key in enumerate("ABCD")
    ]
    parents = {
        "P": {"chunk_id": "P", "text": "old parent P", "parent_chunk_id": "R"},
        "Q": {"chunk_id": "Q", "text": "old parent Q", "parent_chunk_id": "R"},
        "R": {"chunk_id": "R", "text": "old root", "parent_chunk_id": ""},
    }
    calls = []
    monkeypatch.setattr(rag, "assert_index_available", lambda: assert_index_available(tmp_path))
    monkeypatch.setattr(rag, "AUTO_MERGE_ENABLED", True)
    monkeypatch.setattr(rag, "AUTO_MERGE_THRESHOLD", 2)
    monkeypatch.setattr(rag, "_embedding_service", SimpleNamespace(
        get_embeddings=lambda texts: [[1.0]], get_sparse_embedding=lambda text: {0: 1.0},
    ))
    def recall(**kwargs):
        calls.append("recall")
        return leaves
    monkeypatch.setattr(rag, "_milvus_manager", SimpleNamespace(hybrid_retrieve=recall, dense_retrieve=recall))
    def fetch(ids):
        calls.append("parents")
        # Return shared objects so the pipeline must detach them while holding the gate.
        return [parents[key] for key in ids if key in parents]
    monkeypatch.setattr(rag, "_parent_chunk_store", SimpleNamespace(get_documents_by_ids=fetch))
    monkeypatch.setattr(rag, "RERANK_MODEL", "test")
    monkeypatch.setattr(rag, "RERANK_BINDING_HOST", "https://example.invalid")
    monkeypatch.setattr(rag, "RERANK_API_KEY", "test")
    def response(*args, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: {"results": [
            {"index": i, "relevance_score": score} for i, score in enumerate([1, 9, 2, 8])
        ]})
    monkeypatch.setattr(rag.requests, "post", response)
    return SimpleNamespace(leaves=leaves, parents=parents, calls=calls, journal=tmp_path, response=response)


@pytest.mark.parametrize("change", ["replace", "delete"])
def test_slow_rerank_releases_gate_and_keeps_two_level_snapshot(storage, monkeypatch, change):
    entered, release, reader_done, writer_done = [threading.Event() for _ in range(4)]
    results, errors = {}, []
    def post(*args, **kwargs):
        if kwargs["json"]["query"] == "first":
            entered.set()
            if not release.wait(5):
                raise AssertionError("test did not release reranker")
        return storage.response(*args, **kwargs)
    monkeypatch.setattr(rag.requests, "post", post)
    def read(name, done=None):
        try:
            results[name] = rag.retrieve_documents(name, top_k=4)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if done:
                done.set()
    def mutate():
        try:
            def update():
                for leaf in storage.leaves:
                    leaf["text"] = "new leaf"
                for parent in storage.parents.values():
                    parent["text"] = "new root"
                if change == "delete":
                    storage.leaves.clear()
                    storage.parents.clear()
            rag.run_serialized_index_mutation(update)
        except BaseException as exc:
            errors.append(exc)
        finally:
            writer_done.set()
    first = threading.Thread(target=read, args=("first",))
    second = threading.Thread(target=read, args=("second", reader_done))
    writer = threading.Thread(target=mutate)
    first.start()
    try:
        assert entered.wait(3)
        second.start()
        writer.start()
        assert reader_done.wait(2), "another retrieval is blocked by the slow reranker"
        assert writer_done.wait(2), "index mutation is blocked by the slow reranker"
    finally:
        release.set()
        for thread in (first, second, writer):
            if thread.ident is not None:
                thread.join(5)
                assert not thread.is_alive()
    assert not errors
    first_docs = results["first"]["docs"]
    assert [(doc["chunk_id"], doc["text"], doc["rerank_score"]) for doc in first_docs] == [("R", "old root", 9)]
    assert results["first"]["meta"]["auto_merge_steps"] == 2
    newest = rag.retrieve_documents("newest", top_k=4)
    assert [doc["text"] for doc in newest["docs"]] == (["new root"] if change == "replace" else [])


def test_merge_reads_only_snapshot_even_when_parent_is_missing(storage, monkeypatch):
    storage.parents.clear()
    def post(*args, **kwargs):
        storage.parents["P"] = {"chunk_id": "P", "text": "newly created parent"}
        return storage.response(*args, **kwargs)
    monkeypatch.setattr(rag.requests, "post", post)
    result = rag.retrieve_documents("question", top_k=4)
    assert [doc["chunk_id"] for doc in result["docs"]] == ["B", "D", "C", "A"]


def test_recovery_required_never_recalls_or_reranks(storage, monkeypatch):
    (storage.journal / ("a" * 32 + ".json")).write_text(json.dumps({"phase": "recovery_required"}))
    monkeypatch.setattr(rag.requests, "post", lambda *args, **kwargs: pytest.fail("must not rerank"))
    with pytest.raises(RecoveryRequiredError):
        rag.retrieve_documents("question")
    assert storage.calls == []


def test_dense_fallback_also_snapshots_parents_before_rerank(storage, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("hybrid unavailable")
    monkeypatch.setattr(rag._milvus_manager, "hybrid_retrieve", fail)
    def post(*args, **kwargs):
        for parent in storage.parents.values():
            parent["text"] = "new root"
        return storage.response(*args, **kwargs)
    monkeypatch.setattr(rag.requests, "post", post)
    result = rag.retrieve_documents("question", top_k=4)
    assert result["meta"]["retrieval_mode"] == "dense_fallback"
    assert result["docs"][0]["text"] == "old root"


@pytest.mark.parametrize("top_k", [1, 4])
def test_no_possible_merge_does_not_require_parent_database(storage, monkeypatch, top_k):
    if top_k == 4:
        for leaf in storage.leaves:
            leaf["parent_chunk_id"] = "parent-" + leaf["chunk_id"]
    def unavailable(ids):
        raise RuntimeError("parent database unavailable")
    monkeypatch.setattr(rag._parent_chunk_store, "get_documents_by_ids", unavailable)
    result = rag.retrieve_documents("question", top_k=top_k)
    assert result["meta"]["retrieval_mode"] == "hybrid"
    assert [doc["chunk_id"] for doc in result["docs"]] == ["B", "D", "C", "A"][:top_k]


def test_unmerged_leaf_content_is_detached_before_rerank(storage, monkeypatch):
    monkeypatch.setattr(rag, "AUTO_MERGE_ENABLED", False)
    def post(*args, **kwargs):
        for leaf in storage.leaves:
            leaf["text"] = "new leaf"
        return storage.response(*args, **kwargs)
    monkeypatch.setattr(rag.requests, "post", post)
    result = rag.retrieve_documents("question", top_k=4)
    assert [doc["text"] for doc in result["docs"]] == ["old leaf B", "old leaf D", "old leaf C", "old leaf A"]
