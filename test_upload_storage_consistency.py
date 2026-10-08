"""Storage-boundary regressions; only in-memory stores and RPC fakes are used."""
from types import SimpleNamespace

import grpc
import pytest
from pymilvus import MilvusClient
from pymilvus.decorators import retry_on_rpc_failure
from pymilvus.exceptions import MilvusException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import milvus_client, parent_chunk_store
from backend.cache import RedisCache
from backend.milvus_client import MilvusManager
from backend.models import ParentChunk


def test_lost_insert_response_is_not_replayed_by_the_pymilvus_sdk():
    class LostResponse(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "response lost after storage commit"

    class Connection:
        def __init__(self):
            self.rows = []

        # MilvusClient.insert delegates to insert_rows, which has this SDK
        # decorator. Keep the real retry policy; replace only the remote write.
        @retry_on_rpc_failure(initial_back_off=0)
        def insert_rows(self, collection_name, data, **kwargs):
            self.rows.extend(data)
            if len(self.rows) == 1:
                raise LostResponse()
            return SimpleNamespace(insert_count=1, primary_keys=[2], cost=0)

    connection = Connection()
    client = MilvusClient.__new__(MilvusClient)
    client._get_connection = lambda: connection
    client._generate_call_context = lambda **kwargs: None
    manager = MilvusManager()
    manager.client = client

    with pytest.raises(MilvusException, match="response lost after storage commit"):
        manager.insert([{"text": "one committed row"}])
    assert connection.rows == [{"text": "one committed row"}]


@pytest.fixture
def parent_store(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    ParentChunk.__table__.create(engine)
    monkeypatch.setattr(parent_chunk_store, "SessionLocal", sessionmaker(bind=engine))

    class RedisWithFailedInvalidation:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def setex(self, key, ttl, value):
            self.values[key] = value

        def delete(self, key):
            raise ConnectionError("cache invalidation connection lost")

    cache = RedisCache()
    cache._client = RedisWithFailedInvalidation()
    monkeypatch.setattr(parent_chunk_store, "cache", cache)
    try:
        yield parent_chunk_store.ParentChunkStore(), cache
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("before", "after"),
    [("old parent", "new parent"), ("new parent", "old parent"), ("old parent", None)],
    ids=["replacement", "rollback", "deletion"],
)
def test_parent_reads_use_committed_data_despite_stale_redis(parent_store, before, after):
    store, cache = parent_store
    chunk_id = "guide.md::p1::l2::0"
    old_document = {"chunk_id": chunk_id, "filename": "guide.md", "text": before}
    store.replace_documents("guide.md", [old_document])
    cache.set_json("parent_chunk:" + chunk_id, old_document)

    new_documents = [] if after is None else [{**old_document, "text": after}]
    store.replace_documents("guide.md", new_documents)

    assert cache.get_json("parent_chunk:" + chunk_id)["text"] == before
    actual = store.get_documents_by_ids([chunk_id])
    assert [document["text"] for document in actual] == ([] if after is None else [after])


def test_parent_reads_preserve_requested_order_duplicates_and_missing_ids(parent_store):
    store, _ = parent_store
    store.replace_documents("guide.md", [
        {"chunk_id": "first", "filename": "guide.md", "text": "first parent"},
        {"chunk_id": "second", "filename": "guide.md", "text": "second parent"},
    ])
    actual = store.get_documents_by_ids(["second", "missing", "first", "second", ""])
    assert [document["text"] for document in actual] == ["second parent", "first parent", "second parent"]
    assert store.get_documents_by_ids([]) == []


@pytest.mark.parametrize("operation", ["query", "query_all", "hybrid_retrieve", "dense_retrieve"])
def test_reads_request_strong_consistency_after_index_mutation(monkeypatch, operation):
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        return [{"filename": "guide.md"}] if kwargs["offset"] == 0 else []

    def search(**kwargs):
        calls.append(kwargs)
        return []

    manager = MilvusManager()
    manager.client = SimpleNamespace(query=query, search=search, hybrid_search=search)
    monkeypatch.setattr(manager, "assert_bm25_compatible", lambda: None)
    # Force query_all through a second page, so every page is checked.
    monkeypatch.setattr(milvus_client, "QUERY_MAX_LIMIT", 1)
    if operation == "hybrid_retrieve":
        manager.hybrid_retrieve([1.0, 0.0], "example")
    elif operation == "dense_retrieve":
        manager.dense_retrieve([1.0, 0.0])
    else:
        getattr(manager, operation)()

    assert len(calls) == (2 if operation == "query_all" else 1)
    assert all(call.get("consistency_level") == "Strong" for call in calls)
