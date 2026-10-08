import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.eval_comparison.storage import (
    CheckpointWriter,
    RunLock,
    canonical_fingerprint,
    load_resume,
    save_checkpoint,
)
from backend.eval_comparison import storage


def _payload(fingerprint="fp", records=None):
    return {
        "schema_version": 1,
        "run_id": "synthetic-test",
        "fingerprint": fingerprint,
        "updated_at": "2026-09-22T00:00:00Z",
        "records": records or [],
    }


def _record(sample="q1", arm="no_rag", generation="success", metric="failed"):
    return {
        "sample_id": sample,
        "repeat_index": 0,
        "arm": arm,
        "generation_status": generation,
        "answer": "合成回答" if generation == "success" else None,
        "metrics": {"factual_correctness": None},
        "metric_status": {"factual_correctness": metric},
        "metric_errors": {"factual_correctness": "TimeoutError"} if metric == "failed" else {},
        "generation_attempts": [],
        "scoring_attempts": [],
    }


def test_checkpoint_preserves_null_score_and_error(tmp_path):
    payload = _payload(records=[_record()])
    save_checkpoint(tmp_path, payload)
    actual = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert actual == payload
    assert actual["records"][0]["metrics"]["factual_correctness"] is None


def test_generation_success_is_preserved_when_scoring_failed(tmp_path):
    save_checkpoint(tmp_path, _payload(records=[_record()]))
    actual = load_resume(tmp_path, "fp")
    record = actual["records"][0]
    assert record["generation_status"] == "success"
    assert record["answer"] == "合成回答"
    assert record["metric_status"]["factual_correctness"] == "failed"


def test_resume_rejects_changed_fingerprint(tmp_path):
    save_checkpoint(tmp_path, _payload(fingerprint="old"))
    with pytest.raises(ValueError, match="fingerprint"):
        load_resume(tmp_path, "new")


def test_duplicate_execution_key_is_rejected(tmp_path):
    record = _record()
    save_checkpoint(tmp_path, _payload(records=[record, dict(record)]))
    with pytest.raises(ValueError, match="duplicate execution key"):
        load_resume(tmp_path, "fp")


def test_running_states_become_interrupted_pending_on_resume(tmp_path):
    record = _record(generation="running", metric="running")
    save_checkpoint(tmp_path, _payload(records=[record]))
    actual = load_resume(tmp_path, "fp")
    resumed = actual["records"][0]
    assert resumed["generation_status"] == "pending"
    assert resumed["metric_status"]["factual_correctness"] == "pending"
    assert resumed["generation_attempts"][-1]["status"] == "interrupted"
    assert resumed["scoring_attempts"][-1]["status"] == "interrupted"


def test_atomic_write_failure_keeps_old_checkpoint(tmp_path, monkeypatch):
    old = _payload(records=[_record(sample="old")])
    save_checkpoint(tmp_path, old)

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("backend.eval_comparison.storage.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(tmp_path, _payload(records=[_record(sample="new")]))
    actual = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert actual == old


def test_checkpoint_writer_merges_concurrent_completion_events(tmp_path):
    writer = CheckpointWriter(tmp_path, _payload())

    def complete(index):
        writer.upsert(_record(sample=f"q{index}"))

    threads = [threading.Thread(target=complete, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    actual = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert len(actual["records"]) == 20
    assert {record["sample_id"] for record in actual["records"]} == {
        f"q{index}" for index in range(20)
    }


def test_second_process_lock_is_rejected(tmp_path):
    first = RunLock(tmp_path)
    second = RunLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="locked"):
            second.acquire()
    finally:
        first.release()


def test_canonical_fingerprint_is_order_independent_and_sensitive():
    left = canonical_fingerprint({"model": "m", "prompt": "p", "data": {"b": 2, "a": 1}})
    reordered = canonical_fingerprint({"data": {"a": 1, "b": 2}, "prompt": "p", "model": "m"})
    changed = canonical_fingerprint({"model": "m2", "prompt": "p", "data": {"a": 1, "b": 2}})
    assert left == reordered
    assert left != changed


@pytest.fixture
def windows_pid_guard(monkeypatch):
    if os.name != "nt":
        pytest.skip("Windows process handle semantics")

    def forbidden_kill(*_args):
        raise AssertionError("Windows PID checks must never send a console event")

    monkeypatch.setattr(storage.os, "kill", forbidden_kill)


def test_windows_pid_check_current_process_never_sends_signal(windows_pid_guard):
    assert storage._pid_is_alive(os.getpid()) is True


def test_windows_out_of_range_pid_does_not_alias_live_owner(
    windows_pid_guard, monkeypatch, tmp_path
):
    import ctypes

    monkeypatch.setattr(
        ctypes, "WinDLL", Mock(side_effect=AssertionError("invalid PID must not reach Win32"))
    )
    invalid_pid = 0x100000000 + os.getpid()
    assert storage._pid_is_alive(invalid_pid) is False
    lock_path = tmp_path / ".run.lock"
    lock_path.write_text(json.dumps({"pid": invalid_pid, "token": "stale"}), encoding="utf-8")
    with RunLock(tmp_path):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        assert owner["token"] != "stale"
    assert not lock_path.exists()


def test_windows_pid_check_child_alive_and_natural_exit(windows_pid_guard):
    with subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    ) as child:
        try:
            assert storage._pid_is_alive(child.pid) is True
        finally:
            child.communicate(b"x", timeout=5)
        assert child.returncode == 0
        assert storage._pid_is_alive(child.pid) is False


@pytest.mark.parametrize("error, expected", [(5, True), (87, False), (6, None)])
def test_windows_pid_open_errors_preserve_uncertain_owner(
    windows_pid_guard, monkeypatch, error, expected
):
    import ctypes

    api = SimpleNamespace(
        OpenProcess=Mock(return_value=None),
        WaitForSingleObject=Mock(),
        CloseHandle=Mock(),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: api)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error)
    if expected is None:
        with pytest.raises(OSError):
            storage._pid_is_alive(1234)
    else:
        assert storage._pid_is_alive(1234) is expected
    api.OpenProcess.assert_called_once_with(0x00100000, False, 1234)
    api.WaitForSingleObject.assert_not_called()
    api.CloseHandle.assert_not_called()


@pytest.mark.parametrize("wait_result, expected", [(0x102, True), (0, False), (0xFFFFFFFF, None)])
def test_windows_pid_check_always_closes_handle(
    windows_pid_guard, monkeypatch, wait_result, expected
):
    import ctypes

    api = SimpleNamespace(
        OpenProcess=Mock(return_value=123),
        WaitForSingleObject=Mock(return_value=wait_result),
        CloseHandle=Mock(return_value=True),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: api)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 6)
    if expected is None:
        with pytest.raises(OSError):
            storage._pid_is_alive(1234)
    else:
        assert storage._pid_is_alive(1234) is expected
    api.WaitForSingleObject.assert_called_once_with(123, 0)
    api.CloseHandle.assert_called_once_with(123)


@pytest.mark.parametrize(
    "error, expected",
    [(None, True), (ProcessLookupError(), False), (PermissionError(), True), (OSError(), None)],
)
def test_posix_pid_check_distinguishes_missing_and_inaccessible_owner(monkeypatch, error, expected):
    kill = Mock(side_effect=error)
    monkeypatch.setattr(storage, "os", SimpleNamespace(name="posix", kill=kill))
    if expected is None:
        with pytest.raises(OSError):
            storage._pid_is_alive(1234)
    else:
        assert storage._pid_is_alive(1234) is expected
    kill.assert_called_once_with(1234, 0)
