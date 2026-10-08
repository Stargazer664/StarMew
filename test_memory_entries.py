"""Memory lifecycle contracts against a real, isolated SQLAlchemy database."""
import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend import memory, models
from backend.database import Base


def pair(text):
    return [{'type': 'human', 'content': text}, {'type': 'ai', 'content': '收到'}]


def operation(value, *, key='answer_style', category='preference', scope='global', project='', action='upsert', quote=None, expires_at=None):
    return dict(key=key, category=category, value=value, scope=scope, project=project,
                action=action, source_index=0, quote=quote or value, expires_at=expires_at)


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine('sqlite:///' + str(tmp_path / 'entries.sqlite'), connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(memory, 'SessionLocal', sessions)
    monkeypatch.setattr(memory, 'cache', SimpleNamespace(set_json=lambda *a, **k: None, delete=lambda *a: None))
    with sessions() as db:
        db.add_all([models.User(id=1, username='xin', password_hash='x'), models.User(id=2, username='other', password_hash='x')])
        db.add_all([models.ChatSession(id=1, user_id=1, session_id='one'), models.ChatSession(id=2, user_id=1, session_id='two')])
        db.commit()
    svc = memory.UserMemoryService()
    yield svc, sessions
    engine.dispose()


def extract(svc, ops):
    svc._extract_facts_by_llm = lambda messages: ops


def test_shorter_new_fact_replaces_long_old_fact(env):
    svc, sessions = env
    extract(svc, {'company': '很长的旧公司名称'})
    svc.update_from_conversation(1, pair('我在很长的旧公司名称工作'), session_id='one')
    extract(svc, {'company': '新公司'})
    svc.update_from_conversation(1, pair('我在很长的旧公司名称工作') + pair('我现在在新公司工作'), session_id='one')
    assert svc.get_or_create(1).facts['company'] == '新公司'


def test_freeform_preference_correction_revocation_and_provenance(env):
    svc, sessions = env
    assert hasattr(models, 'MemoryEntry'), 'Versioned memory entries are missing'
    history = pair('回答先给结论再解释')
    extract(svc, [operation('先给结论再解释')])
    svc.update_from_conversation(1, history, session_id='one')
    assert '先给结论再解释' in svc.get_memory_for_context(1)
    history += pair('以后只给结论')
    extract(svc, [operation('只给结论')])
    svc.update_from_conversation(1, history, session_id='one')
    context = svc.get_memory_for_context(1)
    assert '只给结论' in context and '先给结论再解释' not in context
    with sessions() as db:
        rows = db.query(models.MemoryEntry).order_by(models.MemoryEntry.id).all()
        assert [r.status for r in rows] == ['superseded', 'active']
        assert rows[-1].source_quote == '只给结论'
        assert rows[-1].source_session_id == 1
    history += pair('忘记我的回答风格')
    extract(svc, [operation('', action='revoke', quote='忘记我的回答风格')])
    svc.update_from_conversation(1, history, session_id='one')
    assert '只给结论' not in svc.get_memory_for_context(1)


def test_scope_expiry_and_user_isolation(env):
    svc, sessions = env
    assert hasattr(models, 'MemoryEntry'), 'Scoped memory entries are missing'
    extract(svc, [operation('只用本地模型', key='model_policy', category='constraint', scope='project', project='SuperMew', quote='SuperMew只用本地模型')])
    svc.update_from_conversation(1, pair('SuperMew只用本地模型'), session_id='one')
    assert '只用本地模型' in svc.get_memory_for_context(1, query='SuperMew怎么部署', session_id='two')
    assert '只用本地模型' in svc.get_memory_for_context(1, session_id='one')
    assert '只用本地模型' not in svc.get_memory_for_context(1, query='另一个项目', session_id='two')
    assert '只用本地模型' not in svc.get_memory_for_context(2, query='SuperMew')
    extract(svc, [operation('先测登录', key='next_task', category='plan', scope='session')])
    svc.update_from_conversation(1, pair('先测登录'), session_id='two')
    assert '先测登录' in svc.get_memory_for_context(1, session_id='two')
    assert '先测登录' not in svc.get_memory_for_context(1, session_id='one')
    past = (datetime.now() - timedelta(days=1)).isoformat()
    extract(svc, [operation('临时约定', key='temporary', expires_at=past)])
    svc.update_from_conversation(1, pair('临时约定'))
    assert '临时约定' not in svc.get_memory_for_context(1)


def test_legacy_facts_and_preferences_migrate_once(env):
    svc, sessions = env
    assert hasattr(models, 'MemoryEntry'), 'Legacy migration is missing'
    with sessions() as db:
        db.add(models.UserMemory(user_id=1, facts={'location': '苏州'}, preferences={'style': '简短'}))
        db.commit()
    assert '苏州' in svc.get_memory_for_context(1) and '简短' in svc.get_memory_for_context(1)
    entries = svc.list_entries(1)
    assert len(entries) == 2
    location = next(e for e in entries if e['key'] == 'location')
    assert location['source_kind'] == 'legacy'
    assert svc.forget_entry(1, location['id'])
    assert '苏州' not in svc.get_memory_for_context(1)
    assert len(svc.list_entries(1)) == 2
    assert not svc.forget_entry(2, location['id'])


@pytest.mark.parametrize('ops', [
    [operation('虚构', quote='用户没有说的话')],
    [dict(operation('收到'), source_index=1)],
    [operation('只用本地模型', scope='project', project='虚构项目')],
])
def test_invalid_sources_never_advance_checkpoint(env, ops):
    svc, sessions = env
    assert hasattr(models, 'MemoryEntry'), 'Grounded extraction is missing'
    extract(svc, ops)
    with pytest.raises(ValueError):
        svc.update_from_conversation(1, pair('只用本地模型'), session_id='one')
    with sessions() as db:
        assert 'memory_checkpoint' not in db.get(models.ChatSession, 1).metadata_json
        assert db.query(models.MemoryEntry).count() == 0


def test_reset_drains_existing_targets_and_new_messages_can_be_remembered(env, monkeypatch):
    from backend import memory_jobs
    svc, sessions = env
    assert hasattr(models, 'MemoryEntry'), 'Reset barrier is missing'
    monkeypatch.setattr(memory_jobs, 'SessionLocal', sessions)
    with sessions() as db:
        for message in pair('我叫旧名字'):
            db.add(models.ChatMessage(session_ref_id=1, message_type=message['type'], content=message['content']))
        db.add(models.MemoryUpdateJob(session_ref_id=1, target_message_count=2))
        db.commit()
    claimed = memory_jobs.claim_next_job()
    svc.reset_memory(1)
    extract(svc, {'user_name': '旧名字'})
    assert memory_jobs.process_job(claimed, svc)
    assert not svc.get_or_create(1).facts
    extract(svc, {'user_name': '昕'})
    svc.update_from_conversation(1, pair('我叫旧名字') + pair('我叫昕'), session_id='one')
    assert svc.get_or_create(1).facts['user_name'] == '昕'


def test_context_reads_do_not_wait_for_background_model(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    svc, _ = env
    entered, release = Event(), Event()
    extract(svc, {'user_name': '昕'})
    svc.update_from_conversation(1, pair('我叫昕'), session_id='one')
    def blocking(messages):
        entered.set()
        assert release.wait(5)
        return []
    svc._extract_facts_by_llm = blocking
    with ThreadPoolExecutor(2) as pool:
        worker = pool.submit(svc.update_from_conversation, 1, pair('new'), session_id='two')
        try:
            assert entered.wait(2)
            assert '昕' in pool.submit(svc.get_memory_for_context, 1).result(timeout=1)
        finally:
            release.set()
        worker.result(timeout=3)


def test_forgetting_blocks_queued_old_sources_but_not_new_declarations(env, monkeypatch):
    from backend import memory_jobs
    svc, sessions = env
    monkeypatch.setattr(memory_jobs, 'SessionLocal', sessions)
    with sessions() as db:
        for message in pair('以后回答简短'):
            db.add(models.ChatMessage(session_ref_id=1, message_type=message['type'], content=message['content']))
        db.add(models.MemoryUpdateJob(session_ref_id=1, target_message_count=2))
        db.commit()
    extract(svc, [operation('简短')])
    svc.update_from_conversation(1, pair('回答简短'))
    row = svc.list_entries(1)[0]
    assert svc.forget_entry(1, row['id'])
    assert memory_jobs.process_job(memory_jobs.claim_next_job(), svc)
    assert '简短' not in svc.get_memory_for_context(1)
    with sessions() as db:
        human = models.ChatMessage(session_ref_id=1, message_type='human', content='以后回答简短')
        db.add(human)
        db.add(models.ChatMessage(session_ref_id=1, message_type='ai', content='收到'))
        db.add(models.MemoryUpdateJob(session_ref_id=1, target_message_count=4))
        db.commit()
    assert memory_jobs.process_job(memory_jobs.claim_next_job(), svc)
    assert '简短' in svc.get_memory_for_context(1)


def test_real_extractor_does_not_receive_assistant_claims_and_respects_4k_budget(env, monkeypatch):
    import json
    from backend.conversation_context import ContextBudget
    svc, _ = env
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '4096')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '1024')
    prompts = []
    class Model:
        def invoke(self, prompt, **kwargs):
            prompts.append(prompt)
            return SimpleNamespace(content=json.dumps({'operations': []}))
    monkeypatch.setattr(svc, '_get_extract_llm', lambda: Model())
    history = [{'type': 'human', 'content': '假设有人叫小明，怎么称呼他？' + '长内容\\\"' * 2000},
               {'type': 'ai', 'content': 'ASSISTANT_INVENTED_FACT'}]
    svc.update_from_conversation(1, history, session_id='one')
    assert len(prompts) > 1
    assert all('ASSISTANT_INVENTED_FACT' not in p for p in prompts)
    assert all(ContextBudget.from_env().count_text(p) + 24 <= 3072 for p in prompts)
    assert not svc.get_memory_for_context(1)


def test_http_lists_and_forgets_only_authenticated_users_entries(env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import api
    svc, _ = env
    extract(svc, [operation('简短')])
    svc.update_from_conversation(1, pair('回答简短'), session_id='one')
    monkeypatch.setattr(api, 'memory_service', svc)
    app = FastAPI()
    app.include_router(api.router, prefix='/api')
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(id=1, username='xin')
    with TestClient(app) as client:
        response = client.get('/api/memory')
        assert response.status_code == 200
        assert response.json().get('entries'), 'HTTP memory entries missing'
        row = response.json()['entries'][0]
        assert row['value'] == '简短' and row['source_quote'] == '简短'
        app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(id=2, username='other')
        assert client.get('/api/memory').json()['entries'] == []
        assert client.delete(f"/api/memory/entries/{row['id']}").status_code == 404
        app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(id=1, username='xin')
        assert client.delete(f"/api/memory/entries/{row['id']}").status_code == 200
        assert client.get('/api/memory').json()['entries'][0]['status'] == 'revoked'


@pytest.mark.parametrize('stream', [False, True])
def test_chat_uses_scoped_memory_selection(env, monkeypatch, stream):
    import asyncio
    from unittest.mock import patch
    from backend import agent
    from test_agent_memory_context import RecordingModel, HistoryStorage
    from test_agent_streaming import patched_stream_dependencies, collect_sse, parse_sse
    svc, sessions = env
    extract(svc, [operation('只用本地模型', key='model_policy', category='constraint', scope='project', project='SuperMew', quote='SuperMew只用本地模型')])
    svc.update_from_conversation(1, pair('SuperMew只用本地模型'), session_id='one')
    class StreamingModel(RecordingModel):
        async def astream(self, messages, **kwargs):
            for chunk in self.stream(messages, **kwargs):
                yield chunk
    model, storage = StreamingModel(), HistoryStorage([])
    with patch.object(agent, 'init_chat_model', return_value=model):
        graph, _ = agent._create_react_agent()
    from backend import memory_jobs, conversation_turns
    with patched_stream_dependencies(graph, storage, svc), patch.object(memory, 'SessionLocal', sessions), \
            patch.object(agent, 'SessionLocal', sessions), patch.object(memory_jobs, 'SessionLocal', sessions), \
            patch.object(conversation_turns, 'SessionLocal', sessions):
        if stream:
            events = parse_sse(asyncio.run(collect_sse(agent.chat_with_agent_stream('SuperMew如何部署', 'xin', 'two'))))
            assert events[-1] == '[DONE]'
            assert not any(isinstance(e, dict) and e.get('type') == 'error' for e in events)
        else:
            agent.chat_with_agent('SuperMew如何部署', 'xin', 'two')
    assert model.inputs
    assert '只用本地模型' in str(model.inputs[0][0].content)


def test_reset_waits_for_inflight_extraction_then_removes_result(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    svc, sessions = env
    entered, release, reset_started = Event(), Event(), Event()
    def blocking(messages):
        entered.set()
        assert release.wait(5)
        return [operation('简短')]
    svc._extract_facts_by_llm = blocking
    def reset():
        reset_started.set()
        svc.reset_memory(1)
    with ThreadPoolExecutor(2) as pool:
        worker = pool.submit(svc.update_from_conversation, 1, pair('回答简短'), session_id='one')
        try:
            assert entered.wait(2)
            resetting = pool.submit(reset)
            assert reset_started.wait(2)
        finally:
            release.set()
        worker.result(timeout=3)
        resetting.result(timeout=3)
    assert svc.list_entries(1) == []


def test_freeform_categories_never_reimport_as_immortal_legacy_facts(env):
    svc, sessions = env
    extract(svc, [operation('实验期间只用CPU', key='compute_rule', category='experiment_constraint',
                            expires_at=(datetime.now() + timedelta(days=1)).isoformat())])
    svc.update_from_conversation(1, pair('实验期间只用CPU'), session_id='one')
    assert len(svc.list_entries(1)) == 1
    with sessions() as db:
        db.query(models.MemoryEntry).one().expires_at = datetime.now() - timedelta(days=1)
        db.commit()
    assert '实验期间只用CPU' not in svc.get_memory_for_context(1)
    assert len(svc.list_entries(1)) == 1
    assert svc.list_entries(1)[0]['status'] == 'expired'


def test_manual_update_cannot_resurrect_forgotten_queued_sources(env, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import api
    svc, sessions = env
    extract(svc, [operation('简短')])
    svc.update_from_conversation(1, pair('回答简短'))
    with sessions() as db:
        db.add_all([models.ChatMessage(session_ref_id=2, message_type=m['type'], content=m['content']) for m in pair('回答简短')])
        db.commit()
    row = svc.list_entries(1)[0]
    assert svc.forget_entry(1, row['id'])
    monkeypatch.setattr(api, 'memory_service', svc)
    app = FastAPI()
    app.include_router(api.router, prefix='/api')
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(id=1, username='xin')
    with TestClient(app) as client:
        assert client.post('/api/memory/update?session_id=two').status_code == 200
    assert '简短' not in svc.get_memory_for_context(1)


def test_revoke_dated_global_plan_keeps_target_scope(env):
    svc, _ = env
    history = pair('明天去上海出差')
    extract(svc, [operation('去上海出差', key='travel', category='plan',
                            expires_at=(datetime.now() + timedelta(days=2)).isoformat())])
    svc.update_from_conversation(1, history, session_id='one')
    assert '去上海出差' in svc.get_memory_for_context(1)
    extract(svc, [operation('', key='travel', category='plan', action='revoke', quote='忘记出差计划')])
    svc.update_from_conversation(1, history + pair('忘记出差计划'), session_id='one')
    assert '去上海出差' not in svc.get_memory_for_context(1)


def test_missing_summary_uses_verified_user_quote(env):
    svc, _ = env
    raw = operation('先给结论再解释')
    del raw['value']  # Observed qwen3.5:4b output: valid operation without value.
    extract(svc, [raw])
    svc.update_from_conversation(1, pair('请记住先给结论再解释'), session_id='one')
    assert '先给结论再解释' in svc.get_memory_for_context(1)


def test_project_constraint_mislabeled_global_cannot_leak(env):
    svc, _ = env
    extract(svc, [operation('只使用本地模型', category='constraint', quote='SuperMew项目只使用本地模型')])
    svc.update_from_conversation(1, pair('SuperMew项目只使用本地模型'), session_id='one')
    assert '只使用本地模型' not in svc.get_memory_for_context(1, session_id='two', query='另一个项目')
    assert '只使用本地模型' in svc.get_memory_for_context(1, session_id='one')


@pytest.mark.parametrize('quote', ['以后我的所有项目都允许GPU', 'All codebase work may use GPU'])
def test_explicit_global_project_rule_can_correct_global_memory(env, quote):
    svc, _ = env
    history = pair('只用CPU')
    extract(svc, [operation('只用CPU', key='compute_rule', category='constraint')])
    svc.update_from_conversation(1, history, session_id='one')
    extract(svc, [operation('允许GPU', key='compute_rule', category='constraint', quote=quote)])
    svc.update_from_conversation(1, history + pair(quote), session_id='one')
    context = svc.get_memory_for_context(1, session_id='two')
    assert '允许GPU' in context and '只用CPU' not in context
