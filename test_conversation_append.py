import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
"""Append turns against an isolated database; model and Redis never participate."""
from copy import deepcopy
from types import SimpleNamespace
import asyncio

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend import agent
from backend.database import Base
from backend.models import User, ChatSession, ChatMessage
from langchain_core.messages import AIMessage, HumanMessage
from test_agent_streaming import CompleteGraph, FakeMemoryService, patched_stream_dependencies, collect_sse, parse_sse


class LocalCache:
    def __init__(self):
        self.values = {}
        self.invalidations = []
        self.observe_delete = lambda key: None

    def get_json(self, key):
        return deepcopy(self.values.get(key))

    def set_json(self, key, value, **kwargs):
        self.values[key] = deepcopy(value)

    def delete(self, key):
        self.observe_delete(key)
        self.invalidations.append(key)
        self.values.pop(key, None)


@pytest.fixture
def store(monkeypatch, tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "append.sqlite"), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with sessions() as db:
        db.add_all([User(username="xin", password_hash="unused"), User(username="other", password_hash="unused")])
        db.commit()
    local_cache = LocalCache()
    monkeypatch.setattr(agent, "SessionLocal", sessions)
    monkeypatch.setattr(agent, "cache", local_cache)
    yield SimpleNamespace(storage=agent.ConversationStorage(), sessions=sessions, cache=local_cache)
    engine.dispose()


def rows(store):
    with store.sessions() as db:
        return [(r.id, r.session_ref_id, r.message_type, r.content, r.timestamp, deepcopy(r.rag_trace))
                for r in db.query(ChatMessage).order_by(ChatMessage.id).all()]


def metadata(store):
    with store.sessions() as db:
        return [(r.session_id, deepcopy(r.metadata_json), r.updated_at)
                for r in db.query(ChatSession).order_by(ChatSession.id).all()]


def append(store, text="first", answer="answer", **kwargs):
    return store.storage.append_turn("xin", "one", text, answer,
        context_state=kwargs.get("context_state", {"summary": "", "count": 0}),
        rag_trace=kwargs.get("rag_trace", {"source": "first.pdf"}))


@pytest.mark.parametrize("hot", [False, True])
def test_append_keeps_history_ids_timestamps_text_trace_and_cache_consistent(store, hot):
    first = append(store)
    before = rows(store)
    old_records = store.storage.get_session_messages("xin", "one")
    assert first == {"user_message_id": before[0][0], "assistant_message_id": before[1][0], "message_count": 2}
    if hot:
        assert [m.content for m in store.storage.load("xin", "one")] == ["first", "answer"]
        assert store.storage.list_session_infos("xin")[0]["message_count"] == 2
    else:
        store.cache.values.clear()
    observations = []
    store.cache.observe_delete = lambda key: observations.append(len(rows(store)))
    second = append(store, "second", "next", rag_trace={"source": "second.pdf"})
    after = rows(store)
    assert len(after) == len(before) + 2
    assert after[:len(before)] == before
    assert second == {"user_message_id": after[2][0], "assistant_message_id": after[3][0], "message_count": 4}
    assert observations == [4, 4]  # Both invalidations happen after the commit.
    assert store.storage._messages_cache_key("xin", "one") not in store.cache.values
    assert store.storage._sessions_cache_key("xin") not in store.cache.values
    records = store.storage.get_session_messages("xin", "one")
    assert records[:2] == old_records
    assert records[1]["rag_trace"] == {"source": "first.pdf"}
    assert records[3]["rag_trace"] == {"source": "second.pdf"}
    assert [m.content for m in store.storage.load("xin", "one")] == ["first", "answer", "second", "next"]
    assert store.storage.list_session_infos("xin")[0]["message_count"] == 4
    store.cache.values.clear()
    assert store.storage.get_session_messages("xin", "one") == records


def test_append_merges_context_state_without_overwriting_memory_checkpoint_or_metadata(store):
    append(store)
    with store.sessions() as db:
        session = db.query(ChatSession).one()
        session.metadata_json = {"memory_checkpoint": {"count": 2, "digest": "keep"},
            "custom": {"untouched": True}, "context_state": {"summary": "old", "count": 0}}
        db.commit()
    append(store, "second", context_state={"summary": "saved", "count": 2})
    assert metadata(store)[0][1] == {"memory_checkpoint": {"count": 2, "digest": "keep"},
        "custom": {"untouched": True}, "context_state": {"summary": "saved", "count": 2}}
    assert agent.ConversationStorage().load_context_state("xin", "one") == {"summary": "saved", "count": 2}


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["second_insert", "commit"])
def test_append_failure_rolls_back_both_messages_metadata_and_preserves_cache(store, monkeypatch, failure, existing):
    if existing:
        append(store)
        store.storage.get_session_messages("xin", "one")
        store.storage.list_session_infos("xin")
    before_rows, before_meta = rows(store), metadata(store)
    before_cache, before_invalidations = deepcopy(store.cache.values), list(store.cache.invalidations)
    if failure == "second_insert":
        inserts = []
        def reject_second_insert(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith("INSERT INTO chat_messages"):
                inserts.append(parameters)
            if statement.startswith("INSERT INTO chat_messages") and len(inserts) == 2:
                raise RuntimeError("second insert failed")
        engine = store.sessions.kw["bind"]
        event.listen(engine, "before_cursor_execute", reject_second_insert)
        try:
            with pytest.raises(RuntimeError, match="second insert failed"):
                append(store, "do not keep", "do not keep", context_state={"count": 99})
        finally:
            event.remove(engine, "before_cursor_execute", reject_second_insert)
        assert len(inserts) == 2
    else:
        with monkeypatch.context() as scoped:
            def reject_commit(self):
                self.flush()  # Both INSERTs ran; their transaction must still roll back.
                raise RuntimeError("commit failed")
            scoped.setattr(Session, "commit", reject_commit)
            with pytest.raises(RuntimeError, match="commit failed"):
                append(store, "do not keep", "do not keep", context_state={"count": 99})
    assert rows(store) == before_rows
    assert metadata(store) == before_meta
    assert store.cache.values == before_cache
    assert store.cache.invalidations == before_invalidations
    append(store, "retry", "success")
    assert len(rows(store)) == len(before_rows) + 2


def test_append_scopes_same_session_name_to_user(store):
    append(store)
    result = store.storage.append_turn("other", "one", "other text", "other answer", context_state={}, rag_trace=None)
    assert result["message_count"] == 2
    assert [m.content for m in store.storage.load("xin", "one")] == ["first", "answer"]
    assert [m.content for m in store.storage.load("other", "one")] == ["other text", "other answer"]


@pytest.mark.parametrize("streaming", [False, True])
def test_normal_chat_appends_and_preserves_first_turn_trace(store, streaming):
    append(store)
    before = rows(store)
    graph = CompleteGraph()
    graph.invoke = lambda state, **kwargs: {"messages": state["messages"] + [AIMessage(content="sync answer")]}
    with patched_stream_dependencies(graph, store.storage, FakeMemoryService(), rag_trace={"source": "next.pdf"}):
        if streaming:
            events = parse_sse(asyncio.run(collect_sse(agent.chat_with_agent_stream("next question", "xin", "one"))))
            assert events[-1] == "[DONE]"
            assert not any(isinstance(e, dict) and e.get("type") == "error" for e in events)
        else:
            assert agent.chat_with_agent("next question", "xin", "one")["response"] == "sync answer"
    after = rows(store)
    assert after[:2] == before
    assert len(after) == 4
    assert after[-1][-1] == {"source": "next.pdf"}


def test_explicit_full_replacement_preserves_metadata_and_reloads_canonical_records(store):
    append(store)
    with store.sessions() as db:
        session = db.query(ChatSession).one()
        session.metadata_json = {**session.metadata_json, "memory_checkpoint": {"count": 2}}
        db.commit()
    store.storage.replace_all_messages("xin", "one", [HumanMessage(content="repaired"), AIMessage(content="fixed")],
        metadata={"repair": True}, extra_message_data=[None, {"rag_trace": {"source": "repair.pdf"}}])
    assert [r[3] for r in rows(store)] == ["repaired", "fixed"]
    assert metadata(store)[0][1]["memory_checkpoint"] == {"count": 2}
    records = store.storage.get_session_messages("xin", "one")
    assert records[-1]["rag_trace"] == {"source": "repair.pdf"}
