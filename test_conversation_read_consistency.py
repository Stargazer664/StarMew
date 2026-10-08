"""Canonical history reads across late cache fills, outages and process restarts."""
import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.orm import Query
from backend import agent
from backend.models import ChatMessage
from test_conversation_append import store, append, rows


def http_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend import api
    monkeypatch.setattr(api, 'storage', agent.ConversationStorage())
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username='xin')
    return TestClient(app)


@pytest.mark.parametrize('read_method', ['http', 'load'])
def test_inflight_old_read_cannot_poison_next_http_or_other_storage_read(store, monkeypatch, read_method):
    append(store, rag_trace=None)
    entered, release = threading.Event(), threading.Event()
    original = Query.all
    def pause_materialized_history(query):
        result = original(query)
        if (not entered.is_set() and
                query.column_descriptions[0].get('entity') is ChatMessage):
            entered.set()
            assert release.wait(5)
        return result
    monkeypatch.setattr(Query, 'all', pause_materialized_history)
    with http_client(monkeypatch) as client, ThreadPoolExecutor(1, thread_name_prefix='old-history') as pool:
        old = (pool.submit(client.get, '/sessions/one') if read_method == 'http' else
               pool.submit(store.storage.load, 'xin', 'one'))
        try:
            assert entered.wait(3)
            append(store, 'second', 'next', rag_trace=None)
        finally:
            release.set()
        snapshot = old.result(timeout=3)
        if read_method == 'http':
            assert snapshot.status_code == 200, snapshot.text
            snapshot = snapshot.json()['messages']
        assert len(snapshot) == 2  # The already-started snapshot is legal.
    assert len(rows(store)) == 4
    with http_client(monkeypatch) as client:
        response = client.get('/sessions/one')
        assert response.status_code == 200
        assert len(response.json()['messages']) == 4
    assert len(agent.ConversationStorage().load('xin', 'one')) == 4


@pytest.mark.parametrize('operation', ['append', 'replace', 'delete'])
def test_stale_legacy_cache_is_not_authoritative_after_mutation(store, monkeypatch, operation):
    from langchain_core.messages import HumanMessage, AIMessage
    append(store)
    old_records = store.storage.get_session_messages('xin', 'one')
    old_list = store.storage.list_session_infos('xin')
    # Explicitly retain the legacy values as happens on a failed DELETE.
    store.cache.set_json(store.storage._messages_cache_key('xin', 'one'), old_records)
    store.cache.set_json(store.storage._sessions_cache_key('xin'), old_list)
    monkeypatch.setattr(store.cache, 'delete', lambda key: None)
    if operation == 'append':
        append(store, 'second', 'next')
        expected = ['first', 'answer', 'second', 'next']
    elif operation == 'replace':
        store.storage.replace_all_messages('xin', 'one', [HumanMessage(content='repair'), AIMessage(content='fixed')])
        expected = ['repair', 'fixed']
    else:
        assert store.storage.delete_session('xin', 'one')
        expected = []
    other = agent.ConversationStorage()
    assert [m.content for m in other.load('xin', 'one')] == expected
    infos = other.list_session_infos('xin')
    assert ([s['message_count'] for s in infos]) == ([len(expected)] if expected else [])


@pytest.mark.skipif(os.getenv('T09_REDIS_TEST') != '1', reason='dedicated Redis 56379 opt-in')
def test_real_redis_failed_delete_recovery_http_and_process_restart(store, monkeypatch, record_property):
    import redis
    from backend.cache import RedisCache
    cache = RedisCache()
    cache.redis_url = 'redis://127.0.0.1:56379/0'
    cache.key_prefix = 't09_read_' + uuid4().hex
    record_property('redis_prefix', cache.key_prefix)
    client = redis.Redis.from_url(cache.redis_url, decode_responses=True,
        socket_connect_timeout=2, socket_timeout=2)
    assert client.ping()
    cache._client = client
    monkeypatch.setattr(agent, 'cache', cache)
    keys = [cache._key(store.storage._messages_cache_key('xin', 'one')),
            cache._key(store.storage._sessions_cache_key('xin'))]
    try:
        append(store, rag_trace=None)
        records = store.storage.get_session_messages('xin', 'one')
        infos = store.storage.list_session_infos('xin')
        cache.set_json(store.storage._messages_cache_key('xin', 'one'), records)
        cache.set_json(store.storage._sessions_cache_key('xin'), infos)
        class FailedDelete:
            def __getattr__(self, name): return getattr(client, name)
            def delete(self, *args): raise redis.ConnectionError('isolated injected transport outage')
        cache._client = FailedDelete()
        append(store, 'second', 'next', rag_trace=None)
        assert len(rows(store)) == 4
        assert len(json.loads(client.get(keys[0]))) == 2  # Actual server still holds old history.
        cache._client = client
        with http_client(monkeypatch) as http:
            history = http.get('/sessions/one')
            listing = http.get('/sessions')
        assert history.status_code == listing.status_code == 200
        assert len(history.json()['messages']) == 4
        assert listing.json()['sessions'][0]['message_count'] == 4
        child_env = dict(os.environ, DATABASE_URL=str(store.sessions.kw['bind'].url),
            REDIS_URL=cache.redis_url, REDIS_KEY_PREFIX=cache.key_prefix)
        child = subprocess.run([sys.executable, '-X', 'utf8', '-c',
            "from backend.agent import ConversationStorage; s=ConversationStorage(); print('READ_COUNTS', len(s.load('xin','one')), s.list_session_infos('xin')[0]['message_count'])"],
            env=child_env, capture_output=True, text=True, encoding='utf-8', timeout=45)
        assert child.returncode == 0, child.stderr
        assert 'READ_COUNTS 4 4' in child.stdout
    finally:
        client.delete(*keys)  # Exact two owned keys only; never flush or pattern-delete.
        client.close()
