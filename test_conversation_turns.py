"""Turn admission and atomic idempotency against isolated databases."""
import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
from uuid import uuid4
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from backend import agent, models
from backend.database import Base

@pytest.fixture
def database(monkeypatch):
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    @event.listens_for(engine, 'connect')
    def foreign_keys(connection, record):
        connection.execute('PRAGMA foreign_keys=ON')
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        db.add(models.User(username='xin', password_hash='unused'))
        db.commit()
    monkeypatch.setattr(agent, 'SessionLocal', sessions)
    from backend import conversation_turns as turns
    monkeypatch.setattr(turns, 'SessionLocal', sessions)
    from backend import memory, memory_jobs
    monkeypatch.setattr(memory, 'SessionLocal', sessions)
    monkeypatch.setattr(memory_jobs, 'SessionLocal', sessions)
    monkeypatch.setattr(agent.cache, 'delete', lambda *a: None)
    yield sessions, turns
    engine.dispose()

def test_chat_request_has_uuid():
    from backend.schemas import ChatRequest
    request_id = uuid4()
    assert getattr(ChatRequest(message='hi', request_id=request_id), 'request_id', None) == request_id


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('kind', ['web', 'retained', 'empty', 'legacy'])
def test_public_evidence_http_new_replay_and_history(database, monkeypatch, streaming, kind):
    from copy import deepcopy
    import json
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage, AIMessageChunk
    from backend import api, tools
    selected = [{'citation_id': '2', 'source': 'https://example.test/fact' if kind == 'web' else 'kept.pdf',
                 'page': '7', 'text': 'The retained fact.', 'rank': 2, 'score': None}]
    trace = {} if kind == 'web' else {'tool_used': True, 'tool_name': 'search_knowledge_base',
        'retrieved_chunks': [{'filename': 'omitted.pdf', 'page_number': 9, 'text': 'RAW CANDIDATE'}],
        'initial_retrieved_chunks': [{'filename': 'omitted.pdf'}],
        'grading_records': [{'internal_query': 'SECRET_QUERY', 'context_chunks': [{'filename': 'omitted.pdf'}]}]}
    if kind != 'legacy':
        trace.update(answer_evidence=[] if kind == 'empty' else selected, evidence_omitted_count=1)
    trace['retrieval_records'] = [{'internal_query': 'SECRET_QUERY', 'nested': {'internal_query': 'SECRET_QUERY'}}]
    calls = []
    class Graph:
        def invoke(self, state, **kwargs):
            calls.append(state)
            tools._request_context().last_rag_context = {'rag_trace': deepcopy(trace)}
            return {'messages': [AIMessage(content='answer')]}
        async def astream(self, state, **kwargs):
            result = self.invoke(state)
            yield ('messages', (AIMessageChunk(content='answer'), {'langgraph_node': 'final'}))
            yield ('values', result)
    monkeypatch.setattr(agent, 'storage', agent.ConversationStorage())
    monkeypatch.setattr(api, 'storage', agent.storage)
    monkeypatch.setattr(agent, 'react_graph', Graph())
    monkeypatch.setattr(agent.memory_service, 'get_memory_for_context', lambda user, **kwargs: '')
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    body = {'message': 'facts', 'session_id': 'one', 'request_id': str(uuid4())}
    public = []
    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(2):
            response = client.post('/chat/stream' if streaming else '/chat', json=body)
            assert response.status_code == 200, response.text
            if streaming:
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
                assert not [event for event in events if event.get('type') == 'error'], events
                public.append(next(event['rag_trace'] for event in events if event.get('type') == 'trace'))
            else:
                public.append(response.json()['rag_trace'])
        history = client.get('/sessions/one')
        assert history.status_code == 200, history.text
        public.append(history.json()['messages'][-1]['rag_trace'])
    assert len(calls) == 1
    for value in public:
        rendered = json.dumps(value)
        assert 'SECRET_QUERY' not in rendered and 'internal_query' not in rendered
        if kind == 'legacy':
            assert value['retrieved_chunks'][0]['filename'] == 'omitted.pdf'
        else:
            assert 'omitted.pdf' not in rendered and 'RAW CANDIDATE' not in rendered
            assert value['evidence_omitted_count'] == 1
            assert value['answer_evidence'] == ([] if kind == 'empty' else selected)
            assert [chunk['filename'] for chunk in value['retrieved_chunks']] == ([] if kind == 'empty' else [selected[0]['source']])
            if kind != 'empty':
                assert value['retrieved_chunks'][0]['citation_id'] == '2'
                assert value['retrieved_chunks'][0]['page_number'] == 7
    with database[0]() as db:
        persisted = db.query(models.ChatMessage).filter_by(message_type='ai').one().rag_trace
        assert persisted == trace  # Public projection must not destroy diagnostics.


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('old_first', [False, True])
def test_deleted_admitted_http_turn_cannot_complete_successor(database, monkeypatch, streaming, old_first):
    import asyncio
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage, AIMessageChunk
    from backend import api
    sessions, turns = database
    storage = agent.ConversationStorage()
    storage.append_turn('xin', 'one', 'DELETED HISTORY', 'old response', context_state={}, rag_trace=None)
    monkeypatch.setattr(agent, 'storage', storage)
    monkeypatch.setattr(api, 'storage', storage)
    monkeypatch.setattr(agent.memory_service, 'get_memory_for_context', lambda user, **kwargs: '')
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    states, guard = [], threading.Lock()
    class Graph:
        def begin(self, state):
            with guard:
                index = len(states)
                states.append(state)
            entered[index].set()
            return index
        def invoke(self, state, **kwargs):
            index = self.begin(state)
            assert release[index].wait(8)
            return {'messages': [AIMessage(content='STALE ANSWER' if index == 0 else 'NEW ANSWER')]}
        async def astream(self, state, **kwargs):
            index = self.begin(state)
            assert await asyncio.to_thread(release[index].wait, 8)
            answer = 'STALE ANSWER' if index == 0 else 'NEW ANSWER'
            yield ('messages', (AIMessageChunk(content=answer), {'langgraph_node': 'final'}))
            yield ('values', {'messages': [AIMessage(content=answer)]})
    monkeypatch.setattr(agent, 'react_graph', Graph())
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    body = {'message': 'same question', 'session_id': 'one', 'request_id': str(uuid4())}
    endpoint = '/chat/stream' if streaming else '/chat'
    with TestClient(app, raise_server_exceptions=False) as client, ThreadPoolExecutor(2) as pool:
        old = pool.submit(client.post, endpoint, json=body)
        try:
            assert entered[0].wait(5)
            with sessions() as db:
                old_turn = db.query(models.ChatTurn).filter_by(request_id=body['request_id']).one()
                old_ids = old_turn.session_ref_id, old_turn.id
            anchor = turns.admit('xin', 'anchor', str(uuid4()), 'anchor')
            assert client.delete('/sessions/one').status_code == 200
            new = pool.submit(client.post, endpoint, json=body)
            assert entered[1].wait(5)
            with sessions() as db:
                new_turn = db.query(models.ChatTurn).filter_by(request_id=body['request_id']).one()
                new_ids = new_turn.session_ref_id, new_turn.id
            assert old_ids != new_ids
            first = 0 if old_first else 1
            release[first].set()
            (old if old_first else new).result(timeout=5)
            with sessions() as db:
                interim = db.get(models.ChatTurn, new_ids[1]).status
            release[1-first].set()
            old_response, new_response = old.result(timeout=5), new.result(timeout=5)
        finally:
            for signal in release:
                signal.set()
        anchor.close()
    assert interim == ('running' if old_first else 'completed')
    if streaming:
        events = [json.loads(line[6:]) for line in old_response.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
        assert any(event.get('type') == 'error' and 'REQUEST_ENTITY_GONE' in event['content'] for event in events), events
    else:
        assert old_response.status_code == 409, old_response.text
        assert old_response.json()['detail']['code'] == 'REQUEST_ENTITY_GONE'
    assert new_response.status_code == 200 and 'NEW ANSWER' in new_response.text
    assert 'DELETED HISTORY' in str(states[0]['messages'])
    assert 'DELETED HISTORY' not in str(states[1]['messages'])
    with sessions() as db:
        session = db.get(models.ChatSession, new_ids[0])
        assert 'DELETED HISTORY' not in str(session.metadata_json)
        assert [row.content for row in db.query(models.ChatMessage).filter_by(session_ref_id=new_ids[0]).order_by(models.ChatMessage.id)] == ['same question', 'NEW ANSWER']
        assert db.get(models.ChatTurn, new_ids[1]).status == 'completed'
        jobs = db.query(models.MemoryUpdateJob).filter_by(session_ref_id=new_ids[0]).all()
        assert len(jobs) == 1 and jobs[0].target_message_count == 2 and jobs[0].request_id == body['request_id']

def test_atomic_completion_replay_and_conflict(database):
    sessions, turns = database
    rid = str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'question')
    with pytest.raises(turns.TurnConflict, match='SESSION_BUSY'):
        turns.admit('xin', 'one', str(uuid4()), 'other')
    other = turns.admit('xin', 'two', str(uuid4()), 'question')
    other.close()
    result = agent.ConversationStorage().append_turn('xin', 'one', 'question', 'answer', context_state={}, rag_trace=None, request_id=rid, turn_identity=lease.identity)
    repeated = agent.ConversationStorage().append_turn('xin', 'one', 'question', 'ignored', context_state={}, rag_trace=None, request_id=rid, turn_identity=lease.identity)
    assert result == repeated
    lease.close()
    replay = turns.admit('xin', 'one', rid, 'question')
    assert replay.replay['response'] == 'answer'
    with pytest.raises(turns.TurnConflict, match='REQUEST_INPUT_CONFLICT'):
        turns.admit('xin', 'one', rid, 'changed')
    with sessions() as db:
        assert db.query(models.ChatMessage).filter_by(session_ref_id=lease.identity.session_ref_id).count() == 2
        assert db.query(models.ChatTurn).filter_by(request_id=rid).one().status == 'completed'

def test_rollback_and_recovery(database):
    sessions, turns = database
    rid = str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'question')
    def fail(db):
        raise RuntimeError('commit failure')
    event.listen(sessions, 'before_commit', fail)
    try:
        with pytest.raises(RuntimeError, match='commit failure'):
            agent.ConversationStorage().append_turn('xin', 'one', 'question', 'answer', context_state={}, rag_trace=None, request_id=rid, turn_identity=lease.identity)
    finally:
        event.remove(sessions, 'before_commit', fail)
    with sessions() as db:
        assert [r.content for r in db.query(models.ChatMessage)] == ['question']
        assert db.query(models.ChatTurn).one().status == 'running'
    lease.close('cancelled')
    with pytest.raises(turns.TurnConflict, match='REQUEST_CANCELLED'):
        turns.admit('xin', 'one', rid, 'question')
    next_lease = turns.admit('xin', 'one', str(uuid4()), 'next')
    assert turns.recover_interrupted_turns() == 1
    next_lease.close()
    with sessions() as db:
        assert db.query(models.ChatTurn).filter_by(error_code='PROCESS_INTERRUPTED').count() == 1

def test_http_conflict_before_headers_and_unconsumed_disconnect(database):
    import asyncio
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import api
    from backend.schemas import ChatRequest
    from types import SimpleNamespace
    sessions, turns = database
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    rid = str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'question')
    with TestClient(app) as client:
        response = client.post('/chat/stream', json={'message': 'other', 'session_id': 'one', 'request_id': str(uuid4())})
        assert response.status_code == 409, response.text
        assert response.json()['detail']['code'] == 'SESSION_BUSY'
    lease.close()
    async def disconnect():
        response = await api.chat_stream_endpoint(ChatRequest(message='new', session_id='one'), SimpleNamespace(username='xin'))
        async def send(message):
            raise OSError('disconnected before body')
        async def receive():
            return {'type': 'http.disconnect'}
        from starlette.requests import ClientDisconnect
        with pytest.raises(ClientDisconnect):
            await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)
    asyncio.run(disconnect())
    next_lease = turns.admit('xin', 'one', str(uuid4()), 'new')
    next_lease.close()


@pytest.fixture
def postgres(monkeypatch):
    if os.getenv('T02_POSTGRES_TEST') != '1':
        pytest.skip('Set T02_POSTGRES_TEST=1 for isolated real PostgreSQL checks')
    from sqlalchemy import text
    schema = 't02_' + uuid4().hex
    url = 'postgresql+psycopg2://agent_test:agent-test-only@127.0.0.1:55432/agent_optimization_test'
    raw = create_engine(url, connect_args={'connect_timeout': 3, 'options': '-c statement_timeout=5000 -c lock_timeout=3000'})
    with raw.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={'connect_timeout': 3, 'options': f'-c search_path={schema} -c statement_timeout=5000 -c lock_timeout=3000'})
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    from backend import conversation_turns as turns, memory
    try:
        Base.metadata.create_all(engine)
        with sessions() as db:
            db.add(models.User(username='xin', password_hash='unused'))
            db.commit()
        monkeypatch.setattr(agent, 'SessionLocal', sessions)
        monkeypatch.setattr(turns, 'SessionLocal', sessions)
        from backend import memory_jobs
        monkeypatch.setattr(memory_jobs, 'SessionLocal', sessions)
        monkeypatch.setattr(memory, 'SessionLocal', sessions)
        monkeypatch.setattr(agent.cache, 'delete', lambda *args: None)
        yield sessions, turns
    finally:
        engine.dispose()
        with raw.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        raw.dispose()


def test_postgres_first_session_race_and_duplicate_commit(postgres):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    sessions, turns = postgres
    barrier = Barrier(2)
    rid = str(uuid4())
    def enter():
        barrier.wait(timeout=3)
        try:
            return turns.admit('xin', 'first', rid, 'question')
        except turns.TurnConflict as exc:
            return str(exc)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(enter) for _ in range(2)]
        results = [f.result(timeout=8) for f in futures]
    assert sum(isinstance(r, turns.TurnLease) for r in results) == 1
    assert 'REQUEST_RUNNING' in results
    identity = next(r.identity for r in results if isinstance(r, turns.TurnLease))
    def commit():
        barrier.wait(timeout=3)
        return agent.ConversationStorage().append_turn('xin', 'first', 'question', 'answer', context_state={}, rag_trace=None, request_id=rid, turn_identity=identity)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(commit) for _ in range(2)]
        results = [f.result(timeout=8) for f in futures]
    assert results[0] == results[1]
    with sessions() as db:
        assert db.query(models.ChatSession).count() == 1
        assert db.query(models.ChatMessage).count() == 2
        assert db.query(models.ChatTurn).count() == 1


def test_postgres_rollback(postgres):
    test_rollback_and_recovery(postgres)


def test_postgres_create_new_tables_preserves_legacy_message_identity(postgres):
    from datetime import datetime
    from sqlalchemy import inspect
    sessions, _ = postgres
    engine = sessions.kw['bind']
    # Only this fixture's random t02_ schema is used. Recreate the legacy-only
    # shape before exercising the same create_all deployment operation.
    models.MemoryUpdateJob.__table__.drop(engine)
    models.ChatTurn.__table__.drop(engine)
    when = datetime(2024, 2, 3, 4, 5, 6)
    with sessions() as db:
        user = db.query(models.User).one()
        session = models.ChatSession(user_id=user.id, session_id='legacy',
            metadata_json={'context_state': {'summary': 'keep', 'count': 2}}, updated_at=when)
        db.add(session)
        db.flush()
        db.add_all([models.ChatMessage(session_ref_id=session.id, message_type='human', content='old question', timestamp=when),
            models.ChatMessage(session_ref_id=session.id, message_type='ai', content='old answer', timestamp=when,
                rag_trace={'legacy_reference': 'old.pdf'})])
        db.commit()
    def snapshot():
        with sessions() as db:
            return [(m.id, m.session_ref_id, m.timestamp, m.content, m.rag_trace)
                for m in db.query(models.ChatMessage).order_by(models.ChatMessage.id)]
    before = snapshot()
    Base.metadata.create_all(engine)
    Base.metadata.create_all(engine)  # Existing deployments can start again.
    assert {'chat_turns', 'memory_update_jobs'} <= set(inspect(engine).get_table_names())
    assert snapshot() == before
    # Legacy callers omit request_id and still append without rewriting history.
    result = agent.ConversationStorage().append_turn('xin', 'legacy', 'next', 'new', context_state={}, rag_trace=None)
    assert result['message_count'] == 4
    assert snapshot()[:2] == before


def test_postgres_concurrent_first_user_memory(postgres, monkeypatch):
    from backend import memory
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from sqlalchemy.orm import Session
    sessions, turns = postgres
    barrier = Barrier(2)
    def before_flush(db, context, instances):
        if any(isinstance(row, models.UserMemory) for row in db.new):
            barrier.wait(timeout=3)
    event.listen(sessions, 'before_flush', before_flush)
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(memory.memory_service.get_memory_for_context, 'xin') for _ in range(2)]
            assert [f.result(timeout=8) for f in futures] == ['', '']
    finally:
        event.remove(sessions, 'before_flush', before_flush)
    with sessions() as db:
        assert db.query(models.UserMemory).count() == 1


def test_http_replay_has_single_done_and_no_model_call(database, monkeypatch):
    from backend import api
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    sessions, turns = database
    rid = str(uuid4())
    lease = turns.admit('xin', 'one', rid, 'question')
    agent.ConversationStorage().append_turn('xin', 'one', 'question', 'answer', context_state={}, rag_trace=None, request_id=rid, turn_identity=lease.identity)
    monkeypatch.setattr(agent, 'react_graph', None)
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    with TestClient(app) as client:
        body = {'message': 'question', 'session_id': 'one', 'request_id': rid}
        replay = client.post('/chat/stream', json=body)
        assert replay.status_code == 200
        assert replay.text.count('data: [DONE]') == 1
        assert 'answer' in replay.text
        assert client.post('/chat', json=body).json()['response'] == 'answer'
        body['message'] = 'different'
        assert client.post('/chat/stream', json=body).status_code == 409
    with sessions() as db:
        assert db.query(models.ChatMessage).count() == 2


@pytest.mark.parametrize('streaming', [False, True])
def test_http_legacy_request_without_uuid_completes_original_admission(database, monkeypatch, streaming):
    from backend import api
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage
    from test_agent_streaming import CompleteGraph
    graph = CompleteGraph()
    graph.invoke = lambda state, **kwargs: {'messages': [AIMessage(content='完整回答')]}
    monkeypatch.setattr(agent, 'react_graph', graph)
    monkeypatch.setattr(agent, 'storage', agent.ConversationStorage())
    monkeypatch.setattr(agent.memory_service, 'get_memory_for_context', lambda user, **kwargs: '')
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    with TestClient(app) as client:
        response = client.post('/chat/stream' if streaming else '/chat', json={'message': 'hello', 'session_id': 'legacy'})
        assert response.status_code == 200 and 'error' not in response.text
    with database[0]() as db:
        turn = db.query(models.ChatTurn).one()
        assert turn.status == 'completed' and len(turn.request_id) == 36
        assert db.query(models.ChatMessage).filter_by(session_ref_id=turn.session_ref_id).count() == 2
        assert db.query(models.MemoryUpdateJob).one().request_id == turn.request_id


def test_committed_answer_error_then_real_frontend_retry_replays_without_appending(database, monkeypatch):
    """Node Vue methods -> loopback HTTP -> real FastAPI/Agent/SQLite -> retry."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import api
    from test_agent_streaming import CompleteGraph
    from test_chat_request_frontend import run_frontend
    sessions, turns = database
    graph_calls = []
    class CountedGraph(CompleteGraph):
        async def astream(self, *args, **kwargs):
            graph_calls.append('generation')
            async for event in super().astream(*args, **kwargs):
                yield event
    original_append = agent.storage.append_turn
    def post_commit_failure(*args, **kwargs):
        original_append(*args, **kwargs)
        raise RuntimeError('post-commit transport boundary failure')
    monkeypatch.setattr(agent.storage, 'append_turn', post_commit_failure)
    monkeypatch.setattr(agent, 'react_graph', CountedGraph())
    monkeypatch.setattr(agent, 'memory_service', SimpleNamespace(
        get_memory_for_context=lambda *a, **kw: ''))
    monkeypatch.setattr(agent.cache, 'get_json', lambda *a: None)
    monkeypatch.setattr(agent.cache, 'set_json', lambda *a, **kw: None)
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    observed = []
    with TestClient(app) as client:
        class Bridge(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                response = client.post('/chat/stream', json=body)
                with sessions() as db:
                    observed.append({'request': body, 'response': response.text,
                        'message_count': db.query(models.ChatMessage).count(),
                        'turn_status': db.query(models.ChatTurn).filter_by(request_id=body['request_id']).one().status})
                self.send_response(response.status_code)
                self.send_header('Content-Type', response.headers['content-type'])
                self.send_header('Content-Length', str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)

            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Bridge)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            runtime = run_frontend('''
state.isAuthenticated = true;
state.$nextTick = fn => fn();
state.scrollToBottom = () => {};
state.loadSessions = () => {};
state.userInput = 'question';
query = async (url, opts) => fetch('http://127.0.0.1:PORT/chat/stream', opts);
await state.handleSend();
const firstError = state.messages.at(-1).streamError;
await state.retryMessage(state.messages.at(-1));
emit({firstError, retriedText:state.messages.at(-1).text, retryError:state.messages.at(-1).streamError});
'''.replace('PORT', str(server.server_port)))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
    assert len(observed) == 2
    assert 'post-commit transport boundary failure' in runtime['firstError']
    assert observed[0]['message_count'] == 2
    assert observed[0]['turn_status'] == 'completed'
    assert observed[1]['request']['request_id'] == observed[0]['request']['request_id']
    assert observed[1]['message_count'] == 2
    assert observed[1]['response'].count('data: [DONE]') == 1
    assert runtime['retryError'] == ''
    with sessions() as db:
        answer = db.query(models.ChatMessage).filter_by(message_type='ai').one()
        assert runtime['retriedText'] == answer.content
        assert db.query(models.ChatTurn).count() == 1
    assert graph_calls == ['generation']


def test_sync_http_cancel_keeps_admission_until_real_thread_finishes(database, monkeypatch):
    import asyncio
    from threading import Event
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage
    from backend import api
    from backend.schemas import ChatRequest
    sessions, turns = database
    entered, release = Event(), Event()
    def invoke(state, **kwargs):
        entered.set()
        assert release.wait(5)
        return {'messages': [AIMessage(content='answer')]}
    monkeypatch.setattr(agent, 'react_graph', SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(agent, 'memory_service', SimpleNamespace(get_memory_for_context=lambda *a, **kw: '', update_from_conversation=lambda *a, **kw: None))
    monkeypatch.setattr(agent.cache, 'get_json', lambda *a: None)
    monkeypatch.setattr(agent.cache, 'set_json', lambda *a, **kw: None)
    async def scenario():
        task = asyncio.create_task(api.chat_endpoint(ChatRequest(message='question', session_id='one'), SimpleNamespace(username='xin')))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        with pytest.raises(turns.TurnConflict, match='SESSION_BUSY'):
            turns.admit('xin', 'one', str(uuid4()), 'second')
        other = turns.admit('xin', 'other', str(uuid4()), 'second')
        other.close()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        lease = turns.admit('xin', 'one', str(uuid4()), 'second')
        lease.close()
    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_cancelled_stream_waits_for_actual_sync_tool_thread(database, monkeypatch):
    import asyncio
    from threading import Event
    from types import SimpleNamespace
    from test_agent_streaming import KnowledgeBoundaryModel, FakeMemoryService
    sessions, turns = database
    entered, release = Event(), Event()
    def tool(inputs):
        entered.set()
        assert release.wait(5)
        return 'tool result'
    model = KnowledgeBoundaryModel()
    monkeypatch.setattr(agent, 'init_chat_model', lambda **kw: model)
    graph, _ = agent._create_react_agent()
    monkeypatch.setattr(agent, 'react_graph', graph)
    monkeypatch.setattr(agent, 'search_knowledge_base', SimpleNamespace(invoke=tool))
    monkeypatch.setattr(agent, 'memory_service', FakeMemoryService())
    monkeypatch.setattr(agent.cache, 'get_json', lambda *a: None)
    monkeypatch.setattr(agent.cache, 'set_json', lambda *a, **kw: None)
    async def scenario():
        async def consume():
            async for _ in agent.chat_with_agent_stream('question', 'xin', 'one'):
                pass
        task = asyncio.create_task(consume())
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.03)
        try:
            assert not task.done(), 'Cancelled wrapper must still own the running tool thread'
            with pytest.raises(turns.TurnConflict, match='SESSION_BUSY'):
                turns.admit('xin', 'one', str(uuid4()), 'next')
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        lease = turns.admit('xin', 'one', str(uuid4()), 'next')
        lease.close()
        with sessions() as db:
            assert [r.content for r in db.query(models.ChatMessage)] == ['question', 'next']
    try:
        asyncio.run(scenario())
    finally:
        release.set()
