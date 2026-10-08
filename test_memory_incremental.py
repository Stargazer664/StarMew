import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
"""Incremental memory against an isolated real SQLAlchemy database."""
from types import SimpleNamespace
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import memory
from backend.database import Base
from backend.models import User, UserMemory, ChatSession


@pytest.fixture
def service(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        db.add(User(id=1, username="xin", password_hash="unused"))
        db.add(ChatSession(user_id=1, session_id="one", metadata_json={"unrelated": True}))
        db.add(ChatSession(user_id=1, session_id="two", metadata_json={}))
        db.commit()
    monkeypatch.setattr(memory, "SessionLocal", sessions)
    monkeypatch.setattr(memory, "cache", SimpleNamespace(set_json=lambda *a, **k: None))
    result = memory.UserMemoryService()
    calls = []
    monkeypatch.setattr(result, "_extract_facts_by_llm", lambda msgs: calls.append(list(msgs)) or {})
    yield result, sessions, calls
    engine.dispose()


def pair(text):
    return [{"type": "human", "content": text}, {"type": "ai", "content": "收到"}]


def test_memory_updates_are_really_persisted(service):
    svc, sessions, _ = service
    svc.update_from_conversation(1, pair("我叫昕。"))
    with sessions() as db:
        row = db.query(UserMemory).one()
        assert row.total_messages == 1
        assert row.facts["user_name"] == "昕"


def test_session_checkpoint_skips_old_messages_and_survives_service_restart(service):
    svc, sessions, calls = service
    old, new = pair("我叫昕。"), pair("我在做检索项目。")
    svc.update_from_conversation(1, old, session_id="one")
    svc.update_from_conversation(1, old, session_id="one")
    restarted = memory.UserMemoryService()
    restarted._extract_facts_by_llm = lambda msgs: calls.append(list(msgs)) or {}
    restarted.update_from_conversation(1, old + new, session_id="one")
    assert calls == [old, new]
    with sessions() as db:
        assert db.query(UserMemory).one().total_messages == 2
        assert db.query(ChatSession).filter_by(session_id="one").one().metadata_json["unrelated"] is True


def test_checkpoints_are_session_scoped_and_identical_new_turns_are_not_deduplicated(service):
    svc, sessions, calls = service
    turn = pair("Python")
    svc.update_from_conversation(1, turn, session_id="one")
    svc.update_from_conversation(1, turn + turn, session_id="one")
    svc.update_from_conversation(1, turn, session_id="two")
    assert calls == [turn, turn, turn]
    with sessions() as db:
        assert db.query(UserMemory).one().total_messages == 3


def test_stale_shorter_history_cannot_rewind_memory_checkpoint(service):
    svc, sessions, calls = service
    old, new = pair("old"), pair("new")
    svc.update_from_conversation(1, old + new, session_id="one")
    svc.update_from_conversation(1, old, session_id="one")
    assert calls == [old + new]
    with sessions() as db:
        assert db.query(UserMemory).one().total_messages == 2


def test_memory_and_checkpoint_rollback_together_on_commit_failure(service, monkeypatch):
    svc, sessions, _ = service
    old, new = pair("old"), pair("new")
    svc.update_from_conversation(1, old, session_id="one")
    from sqlalchemy.orm import Session
    def fail(self):
        raise RuntimeError("commit failed")
    with monkeypatch.context() as scoped:
        scoped.setattr(Session, "commit", fail)
        with pytest.raises(RuntimeError, match="commit failed"):
            svc.update_from_conversation(1, old + new, session_id="one")
    svc.update_from_conversation(1, old + new, session_id="one")
    with sessions() as db:
        assert db.query(UserMemory).one().total_messages == 2


def test_summary_state_and_memory_cursor_survive_conversation_save_and_reload(service, monkeypatch):
    from backend import agent
    from langchain_core.messages import HumanMessage, AIMessage
    svc, sessions, _ = service
    monkeypatch.setattr(agent, "SessionLocal", sessions)
    monkeypatch.setattr(agent, "cache", SimpleNamespace(set_json=lambda *a, **k: None, delete=lambda *a: None))
    messages = [HumanMessage(content="hello"), AIMessage(content="answer")]
    storage = agent.ConversationStorage()
    storage.replace_all_messages("xin", "one", messages, metadata={"context_state": {"summary": "saved", "count": 0}})
    svc.update_from_conversation(1, pair("hello"), session_id="one")
    storage.replace_all_messages("xin", "one", messages)
    restarted = agent.ConversationStorage()
    assert restarted.load_context_state("xin", "one")["summary"] == "saved"
    assert restarted.load_context_state("other-user", "one") == {}
    with sessions() as db:
        state = db.query(ChatSession).filter_by(session_id="one").one().metadata_json
        assert state["memory_checkpoint"]["count"] == 2
        assert state["unrelated"] is True


@pytest.mark.parametrize("failure", [RuntimeError("model offline"),
    SimpleNamespace(content="invalid response"), SimpleNamespace(content='{"location":123}')])
def test_failed_extraction_does_not_advance_checkpoint_and_can_retry(service, monkeypatch, failure):
    svc, sessions, _ = service
    monkeypatch.setattr(svc, "_extract_facts_by_llm", memory.UserMemoryService._extract_facts_by_llm.__get__(svc))
    valid = {'operations': [dict(action='upsert', key='location', category='fact', value='苏州',
        scope='global', project='', source_index=0, quote='我住在苏州', expires_at=None)]}
    responses = iter([failure, SimpleNamespace(content=json.dumps(valid, ensure_ascii=False))])
    class Model:
        def invoke(self, prompt, **kwargs):
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            return result
    monkeypatch.setattr(svc, "_get_extract_llm", lambda: Model())
    with pytest.raises((RuntimeError, ValueError)):
        svc.update_from_conversation(1, pair("我住在苏州"), session_id="one")
    with sessions() as db:
        assert "memory_checkpoint" not in db.query(ChatSession).filter_by(session_id="one").one().metadata_json
    svc.update_from_conversation(1, pair("我住在苏州"), session_id="one")
    with sessions() as db:
        row = db.query(UserMemory).one()
        assert row.facts["location"] == "苏州"
        assert row.total_messages == 1


def test_backlog_extraction_reads_every_new_message_in_bounded_batches(service, monkeypatch):
    from backend.conversation_context import ContextBudget
    svc, sessions, _ = service
    monkeypatch.setattr(svc, "_extract_facts_by_llm", memory.UserMemoryService._extract_facts_by_llm.__get__(svc))
    prompts = []
    class Model:
        def invoke(self, prompt, **kwargs):
            prompts.append(prompt)
            operations = [dict(action='upsert', key='location', category='fact', value='苏州',
                scope='global', project='', source_index=0, quote='FIRST_LOCATION 苏州', expires_at=None)] if 'FIRST_LOCATION' in prompt else []
            return SimpleNamespace(content=json.dumps({'operations': operations}, ensure_ascii=False))
    monkeypatch.setattr(svc, "_get_extract_llm", lambda: Model())
    history = pair("FIRST_LOCATION 苏州") + sum((pair(f"MESSAGE_{i}_" + "长消息" * 400) for i in range(5)), [])
    svc.update_from_conversation(1, history, session_id="one")
    assert len(prompts) > 1
    assert all(ContextBudget.from_env().count_text(p) + 24 <= ContextBudget.from_env().input_limit for p in prompts)
    for i in range(5):
        assert any(f"MESSAGE_{i}_" in p for p in prompts)
    with sessions() as db:
        assert db.query(UserMemory).one().facts["location"] == "苏州"

def test_explicit_session_reference_rejects_a_different_named_session(service):
    svc, sessions, calls = service
    with sessions() as db:
        one = db.query(ChatSession).filter_by(session_id='one').one().id
    with pytest.raises(ValueError, match='会话不存在'):
        svc.update_from_conversation(1, pair('wrong target'), session_id='two', session_ref_id=one)
    assert calls == []
    with sessions() as db:
        assert db.query(UserMemory).count() == 0
    svc.update_from_conversation(1, pair('我叫昕。'), session_ref_id=one)
    assert calls == [pair('我叫昕。')]
    with sessions() as db:
        assert db.query(ChatSession).filter_by(session_id='one').one().metadata_json['memory_checkpoint']['count'] == 2
        assert 'memory_checkpoint' not in db.query(ChatSession).filter_by(session_id='two').one().metadata_json
