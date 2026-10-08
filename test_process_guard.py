"""Use actual OS processes and temporary lock/state files; no live stores."""
import asyncio
from contextlib import asynccontextmanager
import importlib
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from test_upload_recovery_boundaries import load_functions


@pytest.fixture(autouse=True)
def restore_index_gate_after_test():
    # Unit tests call index services directly, outside an ASGI lifecycle.
    from backend import upload_jobs
    yield
    resume = getattr(upload_jobs, 'resume_index_operations', None)
    if resume:
        resume()


def guard_module():
    assert (Path(__file__).parent / 'backend/process_guard.py').exists(), 'Missing backend process ownership guard'
    return importlib.import_module('backend.process_guard')


ATTEMPT = '''
import sys
from backend.process_guard import BackendProcessLock, BackendAlreadyRunning
try:
    with BackendProcessLock(sys.argv[1]):
        print('ACQUIRED')
except BackendAlreadyRunning:
    print('BLOCKED')
'''


def attempt(path):
    result = subprocess.run([sys.executable, '-c', ATTEMPT, str(path)], cwd=Path(__file__).parent,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_second_process_is_rejected_and_normal_shutdown_releases_lock(tmp_path):
    guard = guard_module()
    path = tmp_path / 'backend.lock'
    with guard.BackendProcessLock(path):
        assert attempt(path) == 'BLOCKED'
    assert path.exists(), 'Do not unlink the lock file: that can create two lock identities'
    assert attempt(path) == 'ACQUIRED'


def test_crashed_owner_does_not_leave_a_permanent_lock(tmp_path):
    guard = guard_module()
    path, ready = tmp_path / 'backend.lock', tmp_path / 'ready'
    script = '''
import sys, time
from pathlib import Path
from backend.process_guard import BackendProcessLock
with BackendProcessLock(sys.argv[1]):
    Path(sys.argv[2]).write_text('ready')
    time.sleep(60)
'''
    process = subprocess.Popen([sys.executable, '-c', script, str(path), str(ready)],
                               cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), 'Owner process failed to acquire lock'
        with pytest.raises(guard.BackendAlreadyRunning):
            with guard.BackendProcessLock(path):
                pytest.fail('Second owner admitted')
    finally:
        process.kill()
        process.communicate(timeout=5)
    with guard.BackendProcessLock(path):
        pass


def test_lock_error_fails_closed_and_exception_releases_ownership(tmp_path):
    guard = guard_module()
    invalid_parent = tmp_path / 'file'
    invalid_parent.write_text('not a directory')
    with pytest.raises(OSError):
        with guard.BackendProcessLock(invalid_parent / 'backend.lock'):
            pytest.fail('Must not serve without a usable lock')
    path = tmp_path / 'backend.lock'
    with pytest.raises(ValueError):
        with guard.BackendProcessLock(path):
            raise ValueError('startup failed')
    assert attempt(path) == 'ACQUIRED'


def startup_namespace(path, calls, recover=None):
    guard = guard_module()
    from backend import upload_jobs
    from backend.chat_diagnostics import run_blocking
    class Worker:
        def start(self):
            calls.append('memory_start')
        async def aclose(self):
            calls.append('memory_stop')
    return load_functions('app.py', {'lifespan'}, {
        'asynccontextmanager': asynccontextmanager, 'asyncio': asyncio,
        'run_blocking': run_blocking, 'MemoryWorker': Worker,
        'backend_process_lock': lambda: guard.BackendProcessLock(path),
        'resume_index_operations': getattr(upload_jobs, 'resume_index_operations', lambda: None),
        'stop_index_operations': getattr(upload_jobs, 'stop_index_operations', lambda: None),
        'start_diagnostics': lambda: None, 'stop_diagnostics': lambda: None,
        'init_db': lambda: calls.append('init'),
        'recover_interrupted_turns': lambda: calls.append('turns'),
        'api_module': SimpleNamespace(
            embedding_service=SimpleNamespace(reload_state=lambda: calls.append('reload')),
            recover_upload_operations=recover or (lambda **kwargs: calls.append('recover')),
        ),
        'logger': logging.getLogger('test.process.guard'),
    })


def test_lifespan_rejects_second_owner_before_reloading_or_recovering(tmp_path):
    guard = guard_module()
    path, calls = tmp_path / 'backend.lock', []
    ns = startup_namespace(path, calls)
    async def scenario():
        with guard.BackendProcessLock(path):
            with pytest.raises(guard.BackendAlreadyRunning):
                async with ns['lifespan'](SimpleNamespace(state=SimpleNamespace())):
                    pytest.fail('Blocked worker was allowed to serve')
        assert calls == []
    asyncio.run(scenario())


def test_native_owner_recovers_without_local_bm25_reload_and_holds_lock(tmp_path):
    guard = guard_module()
    path, calls = tmp_path / 'backend.lock', []
    ns = startup_namespace(path, calls)
    async def scenario():
        async with ns['lifespan'](SimpleNamespace(state=SimpleNamespace())):
            assert calls == ['init', 'turns', 'memory_start', 'recover']
            with pytest.raises(guard.BackendAlreadyRunning):
                with guard.BackendProcessLock(path):
                    pytest.fail('Ownership released before shutdown')
        with guard.BackendProcessLock(path):
            pass
    asyncio.run(scenario())


def test_bm25_reload_replaces_stale_memory_and_clears_removed_state(tmp_path):
    from backend.embedding import EmbeddingService
    path = tmp_path / 'bm25.json'
    stale, writer = EmbeddingService(path), EmbeddingService(path)
    writer.increment_add_documents(['new document'])
    assert stale.snapshot_state()['total_docs'] == 0
    assert hasattr(stale, 'reload_state'), 'Startup must refresh pre-lock BM25 memory'
    stale.reload_state()
    assert stale.snapshot_state() == writer.snapshot_state()
    path.unlink()
    stale.reload_state()
    assert stale.snapshot_state()['total_docs'] == 0
    assert stale.snapshot_state()['vocab'] == {}


def test_corrupt_bm25_is_not_silently_served_as_empty_on_startup(tmp_path):
    from backend.embedding import EmbeddingService
    path = tmp_path / 'bm25.json'
    state = EmbeddingService(path)
    path.write_text('{corrupt', encoding='utf-8')
    assert hasattr(state, 'reload_state'), 'Missing startup reload'
    with pytest.raises((ValueError, RuntimeError)):
        state.reload_state()


def test_cancelled_startup_retains_lock_until_recovery_thread_finishes(tmp_path):
    guard = guard_module()
    path, calls = tmp_path / 'backend.lock', []
    entered, release = threading.Event(), threading.Event()
    def recover(**kwargs):
        entered.set()
        assert release.wait(5)
    ns = startup_namespace(path, calls, recover)
    async def scenario():
        async def start():
            async with ns['lifespan'](SimpleNamespace(state=SimpleNamespace())):
                pytest.fail('Cancelled startup must not serve')
        task = asyncio.create_task(start())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            with pytest.raises(guard.BackendAlreadyRunning):
                with guard.BackendProcessLock(path):
                    pytest.fail('Recovery is still mutating under cancellation')
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with guard.BackendProcessLock(path):
            pass
    asyncio.run(scenario())


def test_requests_without_lifespan_ownership_are_rejected_before_database_init():
    from fastapi.responses import JSONResponse
    next_calls = []
    ns = load_functions('app.py', {'_init_db_on_first_request'}, {
        'app': SimpleNamespace(middleware=lambda _: lambda fn: fn),
        'JSONResponse': JSONResponse, '_db_initialized': False,
    })
    async def call_next(request):
        next_calls.append(request)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    response = asyncio.run(ns['_init_db_on_first_request'](request, call_next))
    assert response.status_code == 503
    assert next_calls == [] and not ns['_db_initialized']


def test_default_lock_identity_follows_shared_job_database(tmp_path, monkeypatch):
    guard = guard_module()
    database = tmp_path / 'shared.sqlite3'
    monkeypatch.setenv('UPLOAD_JOB_DB_PATH', str(database))
    first, second = guard.backend_process_lock(), guard.backend_process_lock()
    with first:
        with pytest.raises(guard.BackendAlreadyRunning):
            with second:
                pytest.fail('Two owners of the same task database')


def test_shutdown_drains_actual_index_worker_before_releasing_process_lock(tmp_path):
    from backend.upload_jobs import run_serialized_index_mutation
    path, calls = tmp_path / 'backend.lock', []
    ns = startup_namespace(path, calls)
    entered, inspect_shutdown = threading.Event(), threading.Event()
    observed = []
    worker_errors = []
    def mutate():
        entered.set()
        assert inspect_shutdown.wait(5)
        # The old thread is still inside an index write while lifespan exits.
        observed.append(attempt(path))
        # Active transactions must still be allowed to finish their nested work.
        run_serialized_index_mutation(lambda: observed.append('NESTED_COMPLETED'))
    def worker():
        try:
            run_serialized_index_mutation(mutate)
        except BaseException as exc:
            worker_errors.append(exc)
    async def scenario():
        async with ns['lifespan'](SimpleNamespace(state=SimpleNamespace())):
            thread = threading.Thread(target=worker)
            thread.start()
            assert entered.wait(2)
            # Let the worker inspect ownership only after shutdown has started.
            timer = threading.Timer(0.1, inspect_shutdown.set)
            timer.start()
        thread.join(5)
        timer.join(2)
        assert not thread.is_alive() and not worker_errors
    asyncio.run(scenario())
    assert observed == ['BLOCKED', 'NESTED_COMPLETED']
    with pytest.raises(RuntimeError, match='关闭'):
        run_serialized_index_mutation(lambda: pytest.fail('Late request mutated after shutdown'))
    assert attempt(path) == 'ACQUIRED'
