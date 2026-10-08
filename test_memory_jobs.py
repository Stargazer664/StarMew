import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
import asyncio
import importlib
import threading
from datetime import datetime, timedelta
from uuid import uuid4
from types import SimpleNamespace
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from backend import agent, memory, conversation_turns, models
from backend.database import Base

def test_persistent_outbox_schema_exists():
    assert hasattr(models, 'MemoryUpdateJob'), 'Persistent memory outbox is missing'

@pytest.fixture
def env(tmp_path, monkeypatch, request):
    assert hasattr(models, 'MemoryUpdateJob'), 'Persistent memory outbox is missing'
    jobs = importlib.import_module('backend.memory_jobs')
    admin = schema = None
    if getattr(request, 'param', 'sqlite') == 'postgres':
        from sqlalchemy import text
        schema = 't03_' + uuid4().hex
        url = 'postgresql+psycopg2://agent_test:agent-test-only@127.0.0.1:55432/agent_optimization_test'
        admin = create_engine(url, connect_args={'connect_timeout': 3, 'options': '-c statement_timeout=5000 -c lock_timeout=3000'})
        with admin.begin() as db:
            db.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_engine(url, connect_args={'connect_timeout': 3, 'options': f'-c search_path={schema} -c statement_timeout=5000 -c lock_timeout=3000'})
    else:
        engine = create_engine('sqlite:///' + str(tmp_path / 'memory.sqlite'), connect_args={'check_same_thread': False})
        @event.listens_for(engine, 'connect')
        def enable_foreign_keys(connection, record):
            connection.execute('PRAGMA foreign_keys=ON')
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    for module in (agent, memory, conversation_turns, jobs):
        monkeypatch.setattr(module, 'SessionLocal', sessions)
    monkeypatch.setattr(agent.cache, 'delete', lambda *a: None)
    monkeypatch.setattr(agent.cache, 'get_json', lambda *a: None)
    monkeypatch.setattr(agent.cache, 'set_json', lambda *a, **k: None)
    with sessions() as db:
        db.add(models.User(username='xin', password_hash='test'))
        db.commit()
    svc = memory.UserMemoryService()
    calls = []
    monkeypatch.setattr(svc, '_extract_facts_by_llm', lambda messages: calls.append(messages) or {})
    try:
        yield jobs, sessions, svc, calls
    finally:
        engine.dispose()
        if admin is not None:
            assert schema.startswith('t03_') and len(schema) == 36
            with admin.begin() as db:
                db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            admin.dispose()

def append(session='one', text='hello'):
    rid = str(uuid4())
    lease = conversation_turns.admit('xin', session, rid, text)
    return rid, agent.ConversationStorage().append_turn('xin', session, text, 'answer', context_state={}, rag_trace=None, request_id=rid, turn_identity=lease.identity)

def test_outbox_atomic_commit_and_enqueue_rollback(env):
    jobs, sessions, _, _ = env
    rid, result = append()
    with sessions() as db:
        row = db.query(models.MemoryUpdateJob).one()
        assert (row.target_message_count, row.request_id, row.status) == (2, rid, 'pending')
    def fail(mapper, connection, target):
        raise RuntimeError('outbox insert failed')
    event.listen(models.MemoryUpdateJob, 'before_insert', fail)
    try:
        with pytest.raises(RuntimeError, match='outbox insert failed'):
            append(text='second')
    finally:
        event.remove(models.MemoryUpdateJob, 'before_insert', fail)
    with sessions() as db:
        assert [row.content for row in db.query(models.ChatMessage).order_by(models.ChatMessage.id)] == ['hello', 'answer', 'second']
        assert db.query(models.ChatTurn).filter_by(status='completed').count() == 1
        assert db.query(models.MemoryUpdateJob).count() == 1


def test_interrupted_answer_never_enters_memory_extraction(env):
    jobs, sessions, svc, calls = env
    first_id = str(uuid4())
    lease = conversation_turns.admit('xin', 'one', first_id, 'first question')
    agent.ConversationStorage().save_interrupted('xin', 'one', 'PARTIAL_UNTRUSTED', turn_identity=lease.identity)
    lease.close('cancelled')
    append(text='supplement')
    assert jobs.process_job(jobs.claim_next_job(), svc)
    assert [[message['content'] for message in group] for group in calls] == [['first question', 'supplement', 'answer']]

def test_order_boundaries_restart_and_checkpoint_no_duplicate(env):
    jobs, sessions, svc, calls = env
    append(text='first'); append(text='second')
    first = jobs.claim_next_job()
    assert first.target_message_count == 2
    assert jobs.claim_next_job() is None
    jobs.process_job(first, svc)
    second = jobs.claim_next_job()
    jobs.process_job(second, svc)
    # Crash after checkpoint commit but before job completion.
    with sessions() as db:
        db.get(models.MemoryUpdateJob, second.id).status = 'running'
        db.commit()
    assert jobs.recover_running_jobs() == 1
    jobs.process_job(jobs.claim_next_job(), svc)
    assert [[m['content'] for m in c] for c in calls] == [['first', 'answer'], ['second', 'answer']]
    with sessions() as db:
        assert db.query(models.UserMemory).one().total_messages == 2
        assert db.query(models.MemoryUpdateJob).filter_by(status='completed').count() == 2

def test_retry_schedule_terminal_failure_blocks_newer_until_explicit_retry(env):
    jobs, sessions, svc, calls = env
    append(); append(text='newer')
    def failure(messages):
        raise RuntimeError('sensitive model details')
    svc._extract_facts_by_llm = failure
    now = datetime(2030, 1, 1)
    for attempt, delay in enumerate([5, 30, 120, 300, None], 1):
        row = jobs.claim_next_job(now=now)
        assert row.attempts == attempt
        jobs.process_job(row, svc, now=now)
        with sessions() as db:
            saved = db.get(models.MemoryUpdateJob, row.id)
            assert saved.last_error_code == 'MEMORY_UPDATE_FAILED'
            assert saved.status == ('failed' if delay is None else 'pending')
            if delay:
                assert saved.next_attempt_at == now + timedelta(seconds=delay)
        assert jobs.claim_next_job(now=now) is None
        now += timedelta(seconds=delay or 1)
    assert jobs.claim_next_job(now=now) is None
    assert jobs.retry_failed_job(row.id)
    svc._extract_facts_by_llm = lambda messages: {}
    jobs.process_job(jobs.claim_next_job(now=now), svc)
    assert jobs.claim_next_job(now=now).target_message_count == 4


def test_background_model_does_not_block_other_stream_and_shutdown_drains(env, monkeypatch):
    jobs, sessions, svc, _ = env
    append()
    entered, release = threading.Event(), threading.Event()
    def block(messages):
        entered.set()
        assert release.wait(5)
        return {}
    svc._extract_facts_by_llm = block
    from test_agent_streaming import CompleteGraph, FakeMemoryService, collect_sse, parse_sse
    monkeypatch.setattr(agent, 'react_graph', CompleteGraph())
    monkeypatch.setattr(agent, 'memory_service', FakeMemoryService())
    worker = jobs.MemoryWorker(service=svc, poll_interval=0.01)
    async def scenario():
        worker.start()
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream('other', 'xin', 'two')), 2))
            assert events[-1] == '[DONE]' and not any(isinstance(e, dict) and e['type'] == 'error' for e in events)
            closing = asyncio.create_task(worker.aclose())
            await asyncio.sleep(.02)
            closing.cancel(); await asyncio.sleep(.01); closing.cancel()
            assert not closing.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            with sessions() as db:
                assert db.query(models.MemoryUpdateJob).filter_by(status='pending').count() == 1
        finally:
            release.set()
            await worker.aclose()
    asyncio.run(scenario())

def test_job_diagnostics_are_persisted_and_do_not_fail_chat(env, tmp_path):
    import json
    from backend import telemetry
    jobs, sessions, svc, _ = env
    rid, _ = append()
    job = jobs.claim_next_job()
    svc._extract_facts_by_llm = lambda messages: (_ for _ in ()).throw(RuntimeError('private secret'))
    telemetry.start_diagnostics(tmp_path / 'diagnostics.sqlite', tmp_path / 'logs')
    try:
        with telemetry.trace('chat', 'answer', request_id=rid) as chat:
            chat.set_result('completed')
            assert jobs.process_job(job, svc) is False
            assert chat.data['status'] == 'completed'
        telemetry.flush()
        with telemetry.get_store().connection() as db:
            traces = [json.loads(r['payload']) for r in db.execute('SELECT payload FROM diagnostic_traces')]
        mem = next(t for t in traces if t['scope'] == 'memory')
        request = next(t for t in traces if t['scope'] == 'chat')
        assert request['status'] == 'completed'
        assert mem['status'] == 'failed' and mem['request_id'] == rid
        assert mem['target_message_count'] == 2 and mem['session_ref_id'] == job.session_ref_id
        assert mem['job_id'] == str(job.id) and mem['attempt'] == 1
        assert mem['trace_id'] != request['trace_id']
        assert 'private secret' not in str(traces)
    finally:
        telemetry.stop_diagnostics()


def test_same_user_sessions_serialize_and_inference_has_no_open_transactions(env):
    from concurrent.futures import ThreadPoolExecutor
    jobs, sessions, svc, _ = env
    append('one', 'first'); append('two', 'second')
    first, second = jobs.claim_next_job(), jobs.claim_next_job()
    entered, release = threading.Event(), threading.Event()
    active, maximum, calls = 0, 0, []
    guard = threading.Lock()
    checked_out = {}
    def checkout(connection, record, proxy):
        checked_out[threading.get_ident()] = checked_out.get(threading.get_ident(), 0) + 1
    def checkin(connection, record):
        checked_out[threading.get_ident()] -= 1
    event.listen(sessions.kw['bind'], 'checkout', checkout)
    event.listen(sessions.kw['bind'], 'checkin', checkin)
    def extract(messages):
        nonlocal active, maximum
        # The target read and checkpoint read must return their connections.
        assert checked_out.get(threading.get_ident(), 0) == 0
        with guard:
            active += 1
            maximum = max(maximum, active)
            calls.append(messages[0]['content'])
        if messages[0]['content'] == 'first':
            entered.set()
            assert release.wait(3)
        with guard:
            active -= 1
        return {}
    svc._extract_facts_by_llm = extract
    with ThreadPoolExecutor(2) as executor:
        task1 = executor.submit(jobs.process_job, first, svc)
        assert entered.wait(2)
        task2 = executor.submit(jobs.process_job, second, svc)
        try:
            assert not task2.done()
        finally:
            release.set()
        assert task1.result() and task2.result()
    assert maximum == 1 and calls == ['first', 'second']
    with sessions() as db:
        assert db.query(models.UserMemory).one().total_messages == 2


def test_cancelled_admission_drains_thread_and_releases_new_lease(env, monkeypatch):
    jobs, sessions, _, _ = env
    entered, release = threading.Event(), threading.Event()
    original = conversation_turns.admit
    def blocked(*args):
        lease = original(*args)
        entered.set()
        assert release.wait(3)
        return lease
    monkeypatch.setattr(conversation_turns, 'admit', blocked)
    async def scenario():
        task = asyncio.create_task(conversation_turns.admit_async('xin', 'one', str(uuid4()), 'question'))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel(); await asyncio.sleep(.01); task.cancel()
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with sessions() as db:
            assert db.query(models.ChatTurn).one().status == 'cancelled'
    asyncio.run(scenario())


def test_cancelled_commit_retains_lease_until_message_and_job_committed(env, monkeypatch):
    jobs, sessions, _, _ = env
    from test_agent_streaming import CompleteGraph, FakeMemoryService, collect_sse
    monkeypatch.setattr(agent, 'react_graph', CompleteGraph())
    monkeypatch.setattr(agent, 'memory_service', FakeMemoryService())
    entered, release = threading.Event(), threading.Event()
    original = agent.storage.append_turn
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(agent.storage, 'append_turn', blocked)
    async def scenario():
        task = asyncio.create_task(collect_sse(agent.chat_with_agent_stream('question', 'xin', 'one')))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel(); await asyncio.sleep(.01); task.cancel()
            assert not task.done()
            with pytest.raises(conversation_turns.TurnConflict, match='SESSION_BUSY'):
                await asyncio.to_thread(conversation_turns.admit, 'xin', 'one', str(uuid4()), 'next')
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with sessions() as db:
            assert db.query(models.ChatTurn).one().status == 'completed'
            assert db.query(models.ChatMessage).count() == 2
            assert db.query(models.MemoryUpdateJob).one().target_message_count == 2
    asyncio.run(scenario())

def test_lifespan_keeps_process_lock_until_real_memory_thread_finishes(env, tmp_path, monkeypatch):
    from test_process_guard import startup_namespace, guard_module
    from backend import upload_jobs
    # The real shutdown mutates this module singleton; restore its prior value
    # when this isolated lifespan ends, including on a failed assertion.
    monkeypatch.setattr(upload_jobs, '_index_stopping', threading.Event())
    monkeypatch.setattr(upload_jobs, '_index_generation', upload_jobs._index_generation)
    jobs, sessions, svc, _ = env
    append()
    entered, release = threading.Event(), threading.Event()
    def extract(messages):
        entered.set()
        assert release.wait(5)
        return {}
    svc._extract_facts_by_llm = extract
    path, calls = tmp_path / 'backend.lock', []
    ns = startup_namespace(path, calls)
    worker = jobs.MemoryWorker(service=svc, poll_interval=.01)
    ns['MemoryWorker'] = lambda: worker
    guard = guard_module()
    async def scenario():
        serving, leave = asyncio.Event(), asyncio.Event()
        async def application():
            async with ns['lifespan'](SimpleNamespace(state=SimpleNamespace())):
                serving.set()
                await leave.wait()
        task = asyncio.create_task(application())
        try:
            await asyncio.wait_for(serving.wait(), 2)
            assert await asyncio.to_thread(entered.wait, 2)
            leave.set()
            await asyncio.sleep(.02)
            task.cancel(); await asyncio.sleep(.01); task.cancel()
            assert not task.done()
            with pytest.raises(guard.BackendAlreadyRunning):
                with guard.BackendProcessLock(path):
                    pytest.fail('Running memory thread lost backend ownership')
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with guard.BackendProcessLock(path):
            pass
    asyncio.run(scenario())
    assert worker._thread is not None and not worker._thread.is_alive()


def test_earliest_failed_job_does_not_block_other_sessions(env):
    jobs, sessions, _, _ = env
    append('one'); append('one', 'newer'); append('two')
    with sessions() as db:
        db.query(models.MemoryUpdateJob).order_by(models.MemoryUpdateJob.id).first().status = 'failed'
        db.commit()
    claimed = jobs.claim_next_job()
    with sessions() as db:
        assert db.get(models.ChatSession, claimed.session_ref_id).session_id == 'two'
    assert jobs.claim_next_job() is None


def test_enqueue_uses_caller_transaction_and_unique_target(env):
    from sqlalchemy.exc import IntegrityError
    jobs, sessions, _, _ = env
    conversation_turns.admit('xin', 'one', str(uuid4()), 'question')
    with sessions() as db:
        sid = db.query(models.ChatSession).one().id
        jobs.enqueue_memory_update(db, sid, 2)
        db.rollback()
    with sessions() as db:
        assert db.query(models.MemoryUpdateJob).count() == 0
        jobs.enqueue_memory_update(db, sid, 2)
        db.commit()
    with sessions() as db:
        with pytest.raises(IntegrityError):
            jobs.enqueue_memory_update(db, sid, 2)

@pytest.mark.skipif(os.getenv('T03_POSTGRES_TEST') != '1', reason='requires isolated task PostgreSQL')
def test_postgres_atomic_job_rollback_and_order(tmp_path, monkeypatch):
    from sqlalchemy import text
    jobs = importlib.import_module('backend.memory_jobs')
    schema = 't03_' + uuid4().hex
    url = 'postgresql+psycopg2://agent_test:agent-test-only@127.0.0.1:55432/agent_optimization_test'
    admin = create_engine(url, connect_args={'connect_timeout': 3, 'options': '-c statement_timeout=5000 -c lock_timeout=3000'})
    with admin.begin() as db:
        db.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={'connect_timeout': 3, 'options': f'-c search_path={schema} -c statement_timeout=5000 -c lock_timeout=3000'})
    try:
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        for module in (agent, memory, conversation_turns, jobs):
            monkeypatch.setattr(module, 'SessionLocal', sessions)
        monkeypatch.setattr(agent.cache, 'delete', lambda *a: None)
        monkeypatch.setattr(agent.cache, 'set_json', lambda *a, **kw: None)
        with sessions() as db:
            db.add(models.User(username='xin', password_hash='test'))
            db.commit()
        svc, calls = memory.UserMemoryService(), []
        monkeypatch.setattr(svc, '_extract_facts_by_llm', lambda messages: calls.append(messages) or {})
        isolated = (jobs, sessions, svc, calls)
        test_outbox_atomic_commit_and_enqueue_rollback(isolated)
        with sessions() as db:
            db.query(models.ChatTurn).filter_by(status='running').update({'status':'failed'})
            db.commit()
        job = jobs.claim_next_job()
        assert jobs.claim_next_job() is None
        assert jobs.process_job(job, svc)
        append(text='second')
        assert jobs.process_job(jobs.claim_next_job(), svc)
        with sessions() as db:
            assert db.query(models.UserMemory).one().total_messages == 2
            assert db.query(models.MemoryUpdateJob).filter_by(status='completed').count() == 2
    finally:
        engine.dispose()
        assert schema.startswith('t03_') and len(schema) == 36
        with admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()

@pytest.mark.parametrize('operation', ['get_session_messages', 'list_sessions', 'delete_session', 'get_memory', 'update_memory', 'reset_memory'])
def test_session_http_database_work_stays_off_event_loop(env, operation):
    from backend import api
    _, sessions, _, _ = env
    append()
    loop_thread = threading.get_ident()
    def no_loop_sql(connection, cursor, statement, parameters, context, executemany):
        assert threading.get_ident() != loop_thread, 'Session SQL blocked the event loop'
    engine = sessions.kw['bind']
    event.listen(engine, 'before_cursor_execute', no_loop_sql)
    async def scenario():
        user = SimpleNamespace(id=1, username='xin')
        if operation == 'list_sessions':
            result = await api.list_sessions(user)
            assert len(result.sessions) == 1
        elif operation == 'get_session_messages':
            result = await api.get_session_messages('one', user)
            assert len(result.messages) == 2
        elif operation == 'delete_session':
            result = await api.delete_session('one', user)
            assert result.session_id == 'one'
        else:
            result = await getattr(api, operation)(user)
            assert result is not None
    try:
        asyncio.run(scenario())
    finally:
        event.remove(engine, 'before_cursor_execute', no_loop_sql)

def test_completion_marker_failure_retries_without_reextracting_committed_checkpoint(env):
    jobs, sessions, svc, calls = env
    append()
    job = jobs.claim_next_job()
    armed = True
    def fail_completion(db):
        nonlocal armed
        if armed and any(isinstance(row, models.MemoryUpdateJob) and row.status == 'completed' for row in db.dirty):
            armed = False
            raise RuntimeError('ack commit failed')
    event.listen(sessions, 'before_commit', fail_completion)
    try:
        assert jobs.process_job(job, svc, now=datetime(2030, 1, 1)) is False
    finally:
        event.remove(sessions, 'before_commit', fail_completion)
    with sessions() as db:
        assert db.get(models.MemoryUpdateJob, job.id).status == 'pending'
    assert jobs.process_job(jobs.claim_next_job(now=datetime(2030, 1, 2)), svc)
    assert len(calls) == 1
    with sessions() as db:
        assert db.query(models.UserMemory).one().total_messages == 1

@pytest.mark.parametrize('env', ['sqlite', pytest.param('postgres', marks=pytest.mark.skipif(
    os.getenv('T03_POSTGRES_TEST') != '1', reason='requires isolated task PostgreSQL'))], indirect=True)
def test_deleted_session_job_cannot_write_to_recreated_same_name(env):
    from concurrent.futures import ThreadPoolExecutor
    jobs, sessions, svc, calls = env
    append('one', 'old-first'); append('one', 'old-second')
    # Retain a real second session/job so SQLite's ROWID allocation, which may
    # reuse a deleted maximum, cannot alias either old identity in this test.
    append('anchor', 'anchor')
    old_job = jobs.claim_next_job()
    assert jobs.process_job(old_job, svc)
    old_job = jobs.claim_next_job()
    assert old_job.target_message_count == 4
    with sessions() as db:
        original_memory = db.query(models.UserMemory).one().total_messages
    entered, release = threading.Event(), threading.Event()
    extracted = []
    def extract(messages):
        texts = [m['content'] for m in messages if m['type'] == 'human']
        extracted.append(texts)
        if texts == ['old-second']:
            entered.set()
            assert release.wait(5)
            return {'company': 'old-company-should-never-commit'}
        return {'location': 'new-location'}
    svc._extract_facts_by_llm = extract
    with ThreadPoolExecutor(1) as executor:
        running = executor.submit(jobs.process_job, old_job, svc)
        try:
            assert entered.wait(2)
            assert agent.ConversationStorage().delete_session('xin', 'one')
            with sessions() as db:
                assert db.get(models.ChatSession, old_job.session_ref_id) is None
                assert db.get(models.MemoryUpdateJob, old_job.id) is None
            rid, _ = append('one', 'new-content')
            with sessions() as db:
                replacement = db.query(models.ChatSession).filter_by(session_id='one').one()
                new_job = db.query(models.MemoryUpdateJob).filter_by(request_id=rid).one()
                assert replacement.id != old_job.session_ref_id
                assert new_job.id != old_job.id
                new_job_id = new_job.id
        finally:
            release.set()
        assert running.result(timeout=3) is False
    with sessions() as db:
        replacement = db.query(models.ChatSession).filter_by(session_id='one').one()
        assert 'memory_checkpoint' not in replacement.metadata_json
        mem = db.query(models.UserMemory).one()
        assert mem.total_messages == original_memory
        assert 'company' not in mem.facts
        assert db.get(models.MemoryUpdateJob, new_job_id).status == 'pending'
    # Drain the independent anchor, then the replacement's actual task.
    assert jobs.process_job(jobs.claim_next_job(), svc)
    target = jobs.claim_next_job()
    assert target.id == new_job_id
    assert jobs.process_job(target, svc)
    assert extracted == [['old-second'], ['anchor'], ['new-content']]
    with sessions() as db:
        replacement = db.query(models.ChatSession).filter_by(session_id='one').one()
        assert replacement.metadata_json['memory_checkpoint']['count'] == 2
        assert db.query(models.UserMemory).one().facts['location'] == 'new-location'
        assert db.get(models.MemoryUpdateJob, new_job_id).status == 'completed'
