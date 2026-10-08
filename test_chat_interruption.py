"""Cancellation must preserve inputs without treating partial answers as facts."""
import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
from uuid import uuid4

import pytest
from backend import agent, models
from test_conversation_turns import database


def test_cancelled_question_is_saved_and_available_to_next_run(database):
    sessions, turns = database
    first = turns.admit('xin', 'one', str(uuid4()), '推荐蛋糕配方')
    with sessions() as db:
        assert [r.content for r in db.query(models.ChatMessage).all()] == ['推荐蛋糕配方']
    first.close('cancelled')
    second = turns.admit('xin', 'one', str(uuid4()), '不能吃鸡蛋')
    history = agent.ConversationStorage().load('xin', 'one', turn_identity=second.identity)
    assert [m.content for m in history] == ['推荐蛋糕配方']
    second.close('cancelled')


def test_deleted_session_cannot_reuse_a_stale_run_identity(database):
    sessions, turns = database
    rid = str(uuid4())
    old = turns.admit('xin', 'one', rid, 'question')
    store = agent.ConversationStorage()
    assert store.delete_session('xin', 'one')
    new = turns.admit('xin', 'one', rid, 'question')
    assert old.identity != new.identity
    with pytest.raises(turns.TurnConflict, match='REQUEST_ENTITY_GONE'):
        store.append_turn('xin', 'one', 'question', 'stale', context_state={}, rag_trace=None,
                          request_id=rid, turn_identity=old.identity)
    old.close('cancelled')
    assert turns.run_status('xin', 'one', rid)['status'] == 'running'
    new.close('cancelled')


def test_session_deletion_cleans_queued_runs_even_without_sqlite_fk_cascade(database):
    from sqlalchemy import text
    sessions, turns = database
    with sessions() as db:
        db.execute(text('PRAGMA foreign_keys=OFF'))
    rid, next_id = str(uuid4()), str(uuid4())
    old = turns.admit('xin', 'one', rid, 'question')
    turns.request_cancel('xin', 'one', rid, supplement_message='supplement', supplement_request_id=next_id)
    store = agent.ConversationStorage()
    assert store.delete_session('xin', 'one')
    new = turns.admit('xin', 'one', rid, 'question')
    old.close('cancelled')
    assert turns.run_status('xin', 'one', rid)['status'] == 'running'
    new.close('cancelled')


def test_supplement_queue_is_durable_idempotent_and_waits_for_worker(database):
    sessions, turns = database
    rid, next_id = str(uuid4()), str(uuid4())
    first = turns.admit('xin', 'one', rid, '推荐蛋糕配方')
    for _ in range(2):
        state = turns.request_cancel('xin', 'one', rid,
            supplement_message='不能吃鸡蛋', supplement_request_id=next_id)
        assert state['status'] == 'cancel_requested'
        assert state['next_run_id'] == next_id
    with sessions() as db:
        assert [r.content for r in db.query(models.ChatMessage).order_by(models.ChatMessage.id)] == ['推荐蛋糕配方', '不能吃鸡蛋']
    with pytest.raises(turns.TurnConflict, match='SESSION_BUSY'):
        turns.admit('xin', 'one', next_id, '不能吃鸡蛋')
    first.close('cancelled')
    second = turns.admit('xin', 'one', next_id, '不能吃鸡蛋')
    assert [m.content for m in agent.ConversationStorage().load('xin', 'one', turn_identity=second.identity)] == ['推荐蛋糕配方']
    second.close('cancelled')


def test_cancel_wins_before_completion_without_deleting_user_message(database):
    sessions, turns = database
    rid = str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'question')
    turns.request_cancel('xin', 'one', rid)
    with pytest.raises(turns.TurnConflict, match='REQUEST_CANCEL'):
        agent.ConversationStorage().append_turn('xin', 'one', 'question', 'late answer',
            context_state={}, rag_trace=None, request_id=rid, turn_identity=lease.identity)
    lease.close('cancelled')
    with sessions() as db:
        assert [r.content for r in db.query(models.ChatMessage)] == ['question']
        assert db.query(models.MemoryUpdateJob).count() == 0


def test_transport_disconnect_and_explicit_stop_keep_distinct_reasons(database):
    sessions, turns = database
    disconnected = turns.admit('xin', 'one', str(uuid4()), 'question')
    disconnected.close('cancelled')
    explicit = turns.admit('xin', 'one', str(uuid4()), 'supplement')
    turns.request_cancel('xin', 'one', explicit.request_id)
    explicit.close('cancelled')
    with sessions() as db:
        assert db.get(models.ChatTurn, disconnected.turn_id).error_code == 'CONNECTION_INTERRUPTED'
        assert db.get(models.ChatTurn, explicit.turn_id).error_code == 'REQUEST_CANCELLED'


def test_completion_wins_late_cancel_and_replays(database):
    sessions, turns = database
    rid = str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'question')
    agent.ConversationStorage().append_turn('xin', 'one', 'question', 'answer',
        context_state={}, rag_trace=None, request_id=rid, turn_identity=lease.identity)
    stopped = turns.request_cancel('xin', 'one', rid)
    assert stopped['status'] == 'completed'
    assert stopped['response'] == 'answer'
    lease.close('cancelled')
    assert turns.admit('xin', 'one', rid, 'question').replay['response'] == 'answer'


def test_partial_answer_is_visible_but_excluded_from_model_history(database):
    sessions, turns = database
    rid = str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'question')
    store = agent.ConversationStorage()
    store.save_interrupted('xin', 'one', '半个答案', turn_identity=lease.identity)
    lease.close('cancelled')
    assert [(r['content'], r['status']) for r in store.get_session_messages('xin', 'one')] == [('question', 'complete'), ('半个答案', 'interrupted')]
    assert [m.content for m in store.load('xin', 'one')] == ['question']
    with sessions() as db:
        assert db.query(models.MemoryUpdateJob).count() == 0


def test_cancelled_tool_node_drains_first_tool_and_skips_second(database, monkeypatch):
    import asyncio
    import json
    from threading import Event
    from types import SimpleNamespace
    from langchain_core.messages import AIMessageChunk
    from test_agent_streaming import KnowledgeBoundaryModel, FakeMemoryService
    sessions, turns = database
    entered, release = Event(), Event()
    invoked = []
    class TwoTools(KnowledgeBoundaryModel):
        def decision(self, messages):
            return AIMessageChunk(content='', tool_call_chunks=[
                {'name': 'search_knowledge_base', 'args': json.dumps({'query': 'cake'}), 'id': 'first', 'index': 0},
                {'name': 'get_current_weather', 'args': json.dumps({'location': '北京'}), 'id': 'second', 'index': 1},
            ])
    def first_tool(inputs):
        from backend.tool_contracts import ToolResult
        invoked.append('first')
        entered.set()
        assert release.wait(5)
        return ToolResult('ok', 'retrieved')
    def second_tool(inputs):
        invoked.append('second')
        return 'weather'
    monkeypatch.setattr(agent, 'init_chat_model', lambda **kw: TwoTools())
    graph, _ = agent._create_react_agent()
    monkeypatch.setattr(agent, 'react_graph', graph)
    monkeypatch.setattr(agent, 'storage', agent.ConversationStorage())
    monkeypatch.setattr(agent, 'search_knowledge_base', SimpleNamespace(invoke=first_tool))
    monkeypatch.setattr(agent, 'get_current_weather', SimpleNamespace(invoke=second_tool))
    monkeypatch.setattr(agent, 'memory_service', FakeMemoryService())
    rid = str(uuid4())
    async def scenario():
        async def consume():
            async for _ in agent.chat_with_agent_stream('推荐蛋糕配方', 'xin', 'one', request_id=rid):
                pass
        task = asyncio.create_task(consume())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            turns.request_cancel('xin', 'one', rid)
            await asyncio.sleep(0.02)
            assert not task.done()
            assert turns.run_status('xin', 'one', rid)['status'] == 'cancel_requested'
            release.set()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert invoked == ['first']
            assert turns.run_status('xin', 'one', rid)['status'] == 'cancelled'
            with sessions() as db:
                record = db.query(models.ToolExecution).one()
                assert record.status == 'completed'
                assert record.cancel_requested is True
                assert record.tool_call_id == 'first'
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


def test_existing_database_migrates_message_status_without_losing_rows(tmp_path):
    from sqlalchemy import create_engine, text
    from backend.database import migrate_chat_schema
    engine = create_engine('sqlite:///' + str(tmp_path / 'legacy.sqlite'))
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, content TEXT)'))
        connection.execute(text("INSERT INTO chat_messages VALUES (1, 'existing')"))
        connection.execute(text('CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY)'))
        connection.execute(text('INSERT INTO chat_sessions VALUES (1), (2)'))
        connection.execute(text('CREATE TABLE chat_turns (id INTEGER PRIMARY KEY)'))
    migrate_chat_schema(engine)
    migrate_chat_schema(engine)
    with engine.connect() as connection:
        assert connection.execute(text('SELECT content, status FROM chat_messages')).one() == ('existing', 'complete')
        instances = connection.execute(text('SELECT instance_id FROM chat_sessions')).scalars().all()
        assert len(set(instances)) == 2 and all(instances)
        assert 'parent_request_id' in connection.execute(text('SELECT * FROM chat_turns')).keys()
    engine.dispose()


def test_cancel_api_is_owned_idempotent_and_queues_supplement(database):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from backend import api
    sessions, turns = database
    rid, next_id = str(uuid4()), str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'original')
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    with TestClient(app) as client:
        body = {'session_id': 'one', 'supplement_message': 'supplement', 'supplement_request_id': next_id}
        for _ in range(2):
            response = client.post(f'/chat/runs/{rid}/cancel', json=body)
            assert response.status_code == 200, response.text
            assert response.json()['next_run_id'] == next_id
        assert client.get(f'/chat/runs/{rid}', params={'session_id': 'one'}).json()['status'] == 'cancel_requested'
        pending = client.get('/sessions/one').json()['pending_run']
        assert pending == {'run_id': next_id, 'source_run_id': rid, 'message': 'supplement'}
        app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='other')
        assert client.post(f'/chat/runs/{rid}/cancel', json={'session_id': 'one'}).status_code == 404
        assert client.get(f'/chat/runs/{rid}', params={'session_id': 'one'}).status_code == 404
    lease.close('cancelled')


def test_stream_interrupt_persists_partial_and_supplement_prompt_once(database, monkeypatch):
    import asyncio
    import json
    from langchain_core.messages import AIMessage, AIMessageChunk
    from test_agent_streaming import FakeMemoryService
    sessions, turns = database
    rid, next_id = str(uuid4()), str(uuid4())
    inputs = []
    class Graph:
        async def astream(self, state, **kwargs):
            inputs.append(state['messages'])
            if len(inputs) == 1:
                yield ('messages', (AIMessageChunk(content='半个答案'), {'langgraph_node': 'final'}))
                await asyncio.Event().wait()
            else:
                yield ('messages', (AIMessageChunk(content='无蛋配方'), {'langgraph_node': 'final'}))
                yield ('values', {'messages': [AIMessage(content='无蛋配方')]})
    monkeypatch.setattr(agent, 'react_graph', Graph())
    monkeypatch.setattr(agent, 'storage', agent.ConversationStorage())
    monkeypatch.setattr(agent, 'memory_service', FakeMemoryService())
    async def scenario():
        stream = agent.chat_with_agent_stream('推荐蛋糕配方', 'xin', 'one', request_id=rid)
        while True:
            event = await stream.__anext__()
            if json.loads(event[6:]).get('content') == '半个答案':
                assert rid in event
                break
        turns.request_cancel('xin', 'one', rid, supplement_message='不能吃鸡蛋', supplement_request_id=next_id)
        await stream.aclose()
        second = [event async for event in agent.chat_with_agent_stream('不能吃鸡蛋', 'xin', 'one', request_id=next_id)]
        assert any('无蛋配方' == json.loads(event[6:]).get('content') for event in second if event.strip() != 'data: [DONE]')
    asyncio.run(scenario())
    assert [m.content for m in inputs[1] if m.type == 'human'] == ['推荐蛋糕配方', '不能吃鸡蛋']
    assert not [m for m in inputs[1] if m.type == 'ai']
    history = agent.ConversationStorage().get_session_messages('xin', 'one')
    assert sum(r['content'] == '不能吃鸡蛋' for r in history) == 1
    assert any(r['content'] == '半个答案' and r['status'] == 'interrupted' for r in history)
