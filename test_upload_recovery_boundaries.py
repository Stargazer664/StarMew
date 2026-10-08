"""Read-side and startup boundaries use real guards with isolated storage."""
import ast
import asyncio
from contextlib import asynccontextmanager, nullcontext
import json
import logging
import sys
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from backend.document_transactions import RecoveryRequiredError, assert_index_available
from backend.upload_jobs import run_serialized_index_mutation
from backend.telemetry import span, bind_context
from backend.chat_diagnostics import next_retrieval_round
from backend.agent_budget import remaining_timeout


def load_functions(filename, names, namespace):
    tree = ast.parse((Path(__file__).parent / 'backend' / filename).read_text(encoding='utf-8'))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def retrieval_namespace(tmp_path, calls):
    return load_functions('rag_utils.py', {'retrieve_documents'}, {
        '__name__': 'backend.rag_utils', 'remaining_timeout': remaining_timeout,
        'Dict': dict, 'Any': object,
        'span': span, 'bind_context': bind_context, 'next_retrieval_round': next_retrieval_round, 'sys': sys,
        'assert_index_available': lambda: assert_index_available(tmp_path),
        'run_serialized_index_mutation': run_serialized_index_mutation,
        '_retrieve_snapshot': lambda query, top_k: calls.append(query) or {'docs': ['old version']},
        '_finish_retrieval': lambda query, snapshot, top_k: snapshot,
    })


def test_recovery_guard_blocks_retrieval_instead_of_falling_back(tmp_path):
    (tmp_path / ('a' * 32 + '.json')).write_text(json.dumps({'phase': 'recovery_required'}), encoding='utf-8')
    calls = []
    ns = retrieval_namespace(tmp_path, calls)
    with pytest.raises(RecoveryRequiredError):
        ns['retrieve_documents']('hello')
    assert calls == []


def test_reader_waits_until_upload_commit_releases_index_gate(tmp_path):
    entered, release, reader_started, reader_finished = [threading.Event() for _ in range(4)]
    calls = []
    ns = retrieval_namespace(tmp_path, calls)
    def mutate():
        entered.set()
        assert release.wait(3)
    writer = threading.Thread(target=run_serialized_index_mutation, args=(mutate,))
    def read():
        reader_started.set()
        ns['retrieve_documents']('hello')
        reader_finished.set()
    reader = threading.Thread(target=read)
    writer.start()
    assert entered.wait(3)
    reader.start()
    try:
        assert reader_started.wait(3)
        assert not reader_finished.wait(0.05)
        assert calls == []
    finally:
        release.set()
        writer.join(3)
        reader.join(3)
    assert calls == ['hello']
    assert reader_finished.is_set()


@pytest.mark.parametrize('dependency_offline', [False, True])
def test_startup_runs_recovery_before_serving_and_keeps_admin_ui_available(dependency_offline):
    from backend.chat_diagnostics import run_blocking
    calls = []
    lifecycle = []
    class Worker:
        def start(self): lifecycle.append('worker_start')
        async def aclose(self): lifecycle.append('worker_stop')
    def recover(**kwargs):
        assert lifecycle == ['init', 'turns', 'worker_start']
        calls.append(kwargs)
        if dependency_offline:
            raise RecoveryRequiredError('storage offline')
    ns = load_functions('app.py', {'lifespan'}, {
        'asynccontextmanager': asynccontextmanager, 'asyncio': asyncio,
        'run_blocking': run_blocking, 'MemoryWorker': Worker,
        'init_db': lambda: lifecycle.append('init'),
        'recover_interrupted_turns': lambda: lifecycle.append('turns'),
        'api_module': SimpleNamespace(recover_upload_operations=recover, embedding_service=SimpleNamespace(reload_state=lambda: None)),
        'backend_process_lock': lambda: nullcontext(SimpleNamespace(is_held=True)),
        'resume_index_operations': lambda: None, 'stop_index_operations': lambda: None,
        'start_diagnostics': lambda: None, 'stop_diagnostics': lambda: None,
        'logger': logging.getLogger('test.startup'),
    })
    async def scenario():
        async with ns['lifespan'](SimpleNamespace(state=SimpleNamespace())):
            assert calls == [{'mark_interrupted': True}]
    asyncio.run(scenario())
    assert lifecycle == ['init', 'turns', 'worker_start', 'worker_stop']
