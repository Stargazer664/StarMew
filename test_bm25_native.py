from types import SimpleNamespace

import pytest
from pymilvus import MilvusClient

from backend.milvus_client import MilvusManager
from backend.bm25 import BM25MigrationRequired


def test_lexical_text_preserves_versions_identifiers_and_chinese_order():
    from backend.bm25 import tokenize
    assert tokenize("GPT-4") != tokenize("GPT-3.5")
    assert tokenize("C++") == ["c++"]
    assert tokenize("C#") == ["c#"]
    assert tokenize("2024 2025 v1.2.3") == ["2024", "2025", "v1.2.3"]
    assert "向量" in tokenize("向量数据库")
    assert "数据库" in tokenize("向量数据库")
    assert set(tokenize("北京")) != set(tokenize("京北"))


def test_native_schema_uses_dynamic_bm25_scoring():
    captured = {}
    class Indexes:
        def add_index(self, **kwargs):
            captured.setdefault("indexes", []).append(kwargs)
    manager = MilvusManager()
    manager.client = SimpleNamespace(
        has_collection=lambda name: False,
        create_schema=MilvusClient.create_schema,
        prepare_index_params=Indexes,
        create_collection=lambda **kwargs: captured.update(kwargs),
    )
    manager.init_collection(dense_dim=2)
    schema = captured["schema"].to_dict()
    functions = schema.get("functions", [])
    assert len(functions) == 1
    assert functions[0]["input_field_names"] == ["lexical_text"]
    index = next(i for i in captured["indexes"] if i["field_name"] == "sparse_embedding")
    assert index["metric_type"] == "BM25"
    assert index["params"]["inverted_index_algo"] == "TAAT_NAIVE"
    # A fresh process must accept the schema returned by the actual SDK.
    restarted = MilvusManager()
    restarted.client = SimpleNamespace(has_collection=lambda name: True,
        describe_collection=lambda name: schema,
        describe_index=lambda *args: {"metric_type": "BM25", **index["params"]})
    restarted.init_collection(dense_dim=2)
    restarted._bm25_validated = False
    restarted.client.describe_index = lambda *args: {"metric_type": "IP"}
    with pytest.raises(BM25MigrationRequired):
        restarted.assert_bm25_compatible()


def test_existing_legacy_collection_is_not_silently_accepted():
    manager = MilvusManager()
    manager.client = SimpleNamespace(has_collection=lambda name: True,
        describe_collection=lambda name: {"description": "", "fields": [], "functions": []})
    with pytest.raises(RuntimeError, match="BM25"):
        manager.init_collection()


def test_native_query_sends_text_instead_of_weighted_sparse_vector(monkeypatch):
    manager = MilvusManager()
    # Schema validation is exercised separately. Inspect real AnnSearchRequest objects.
    monkeypatch.setattr(manager, "assert_bm25_compatible", lambda: None, raising=False)
    captured = {}
    manager.client = SimpleNamespace(hybrid_search=lambda **kwargs: captured.update(kwargs) or [])
    manager.hybrid_retrieve([1.0, 0.0], "GPT-4 2025")
    request = captured["reqs"][1]
    assert request.param["metric_type"] == "BM25"
    assert isinstance(request.data[0], str)
    assert "gpt-4" in request.data[0] and "2025" in request.data[0]


def test_sparse_audit_uses_same_native_query_without_loading_embedding():
    import ast
    from pathlib import Path
    from backend.bm25 import lexical_text
    tree = ast.parse(Path("scripts/run_dishes_sparse_audit.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_retrieve")
    namespace = {"MilvusManager": MilvusManager, "lexical_text": lexical_text}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "sparse-audit", "exec"), namespace)
    calls = {}
    client = SimpleNamespace(search=lambda **kw: calls.update(kw) or [])
    manager = SimpleNamespace(assert_bm25_compatible=lambda: None, _get_client=lambda: client,
                              collection_name="new")
    assert namespace["sparse_retrieve"](manager, "GPT-4 2025") == []
    assert calls["data"] == ["gpt-4 2025"]
    assert calls["search_params"]["metric_type"] == "BM25"


def test_retired_query_encoder_cannot_mutate_legacy_state(tmp_path):
    from backend.embedding import EmbeddingService
    service = EmbeddingService(tmp_path / "state.json")
    service.increment_add_documents(["old text"])
    before = (tmp_path / "state.json").read_bytes()
    with pytest.raises(RuntimeError, match="BM25"):
        service.get_sparse_embedding("neverindexed")
    assert (tmp_path / "state.json").read_bytes() == before


def test_missing_local_state_does_not_affect_native_query(tmp_path, monkeypatch):
    from backend.embedding import EmbeddingService
    from backend import rag_utils as rag
    service = EmbeddingService(tmp_path / "missing.json")
    service._embedder_ready = True
    service._embedder = SimpleNamespace(embed_documents=lambda texts: [[1.0]])
    monkeypatch.setattr(rag, "_embedding_service", service)
    monkeypatch.setattr(rag, "assert_index_available", lambda: None)
    observed = []
    monkeypatch.setattr(rag, "_milvus_manager", SimpleNamespace(
        hybrid_retrieve=lambda **kw: observed.append(kw) or []))
    result = rag.retrieve_documents("GPT-4 2025")
    assert result["meta"]["retrieval_mode"] == "hybrid"
    assert observed[0]["query_text"] == "GPT-4 2025"
    assert not (tmp_path / "missing.json").exists()


def test_native_transaction_and_rollback_never_write_local_bm25_state(tmp_path, monkeypatch):
    from test_document_transactions import make_service
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    vectors.native_bm25 = service.native_bm25 = True
    vectors.collection_name = "native_collection"
    vectors.rows[0].pop("sparse_embedding")
    vectors.rows[0]["lexical_text"] = "old leaf"
    before = embedding._state_path.read_bytes()
    def fail(*args):
        pytest.fail("native BM25 must not write local statistics")
    monkeypatch.setattr(embedding, "increment_add_documents", fail)
    monkeypatch.setattr(embedding, "increment_remove_documents", fail)
    monkeypatch.setattr(embedding, "restore_state", fail)
    vectors.fail_insert = True
    with pytest.raises(RuntimeError):
        service.replace("a" * 32, staging, "guide.md")
    assert vectors.rows[0]["text"] == "old leaf"
    assert embedding._state_path.read_bytes() == before
    service.replace("a" * 32, staging, "guide.md")
    assert vectors.rows[0]["lexical_text"] == "new leaf"
    assert "sparse_embedding" not in vectors.rows[0]
    service.delete("c" * 32, "guide.md")
    assert vectors.rows == []
    assert embedding._state_path.read_bytes() == before


def test_format_mismatch_does_not_silently_fallback_to_dense(monkeypatch):
    from backend import rag_utils as rag
    monkeypatch.setattr(rag, "assert_index_available", lambda: None)
    monkeypatch.setattr(rag, "_embedding_service", SimpleNamespace(get_embeddings=lambda _: [[1.0]]))
    def reject(**kwargs):
        raise BM25MigrationRequired("migrate BM25 first")
    monkeypatch.setattr(rag, "_milvus_manager", SimpleNamespace(hybrid_retrieve=reject,
        dense_retrieve=lambda **kwargs: pytest.fail("schema mismatch must not be hidden")))
    with pytest.raises(BM25MigrationRequired):
        rag.retrieve_documents("query")


def test_native_crash_recovery_reinserts_text_and_rejects_wrong_collection(tmp_path):
    import copy
    import json
    from test_document_transactions import make_service, ProcessCrash
    from backend.document_transactions import RecoveryRequiredError
    service, staging, original, vectors, parents, embedding = make_service(tmp_path)
    vectors.native_bm25 = service.native_bm25 = True
    vectors.collection_name = "native_collection"
    vectors.rows[0].pop("sparse_embedding")
    vectors.rows[0]["lexical_text"] = "old leaf"
    old = copy.deepcopy(vectors.rows)
    vectors.crash_insert = True
    with pytest.raises(ProcessCrash):
        service.replace("a" * 32, staging, "guide.md")
    dirty = copy.deepcopy(vectors.rows)
    restarted = type(service)(service.root, service.upload_root, service.loader, service.writer, parents, embedding)
    vectors.collection_name = "other_collection"
    with pytest.raises(RecoveryRequiredError):
        restarted.recover()
    assert vectors.rows == dirty
    vectors.collection_name = "native_collection"
    assert restarted.recover()["a" * 32]["status"] == "failed"
    assert vectors.rows == old
    assert json.loads((service.root / ("a" * 32 + ".json")).read_text())["phase"] == "rolled_back"
