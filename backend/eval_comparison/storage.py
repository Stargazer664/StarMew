"""Atomic persistence, fingerprints, and exclusive run locking."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_NAME = "checkpoint.json"
LOCK_NAME = ".run.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def execution_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("sample_id", "")),
        int(record.get("repeat_index", 0)),
        str(record.get("arm", "")),
    )


def save_checkpoint(run_dir: Path, payload: dict[str, Any]) -> None:
    """Durably replace checkpoint.json without exposing a partial document."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / CHECKPOINT_NAME
    temporary = directory / (
        f".{CHECKPOINT_NAME}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_unique_records(records: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        key = execution_key(record)
        if key in seen:
            raise ValueError(f"duplicate execution key: {key}")
        seen.add(key)


def _recover_running(record: dict[str, Any]) -> None:
    if record.get("generation_status") == "running":
        record["generation_status"] = "pending"
        record.setdefault("generation_attempts", []).append(
            {"status": "interrupted", "recovered_at": utc_now()}
        )
    statuses = record.setdefault("metric_status", {})
    for metric, status in list(statuses.items()):
        if status == "running":
            statuses[metric] = "pending"
            record.setdefault("scoring_attempts", []).append(
                {"metric": metric, "status": "interrupted", "recovered_at": utc_now()}
            )


def load_resume(run_dir: Path, expected_fingerprint: str) -> dict[str, Any]:
    path = Path(run_dir) / CHECKPOINT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = payload.get("fingerprint")
    if actual != expected_fingerprint:
        raise ValueError(
            f"fingerprint mismatch: checkpoint={actual!r}, expected={expected_fingerprint!r}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("checkpoint records must be an array")
    _validate_unique_records(records)
    for record in records:
        _recover_running(record)
    return payload


class CheckpointWriter:
    """Single in-process writer that merges completion events by execution key."""

    def __init__(self, run_dir: Path, payload: dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.payload = deepcopy(payload)
        self._mutex = threading.Lock()
        _validate_unique_records(self.payload.setdefault("records", []))
        save_checkpoint(self.run_dir, self.payload)

    def upsert(self, record: dict[str, Any]) -> None:
        key = execution_key(record)
        with self._mutex:
            records = self.payload["records"]
            matches = [index for index, current in enumerate(records) if execution_key(current) == key]
            if len(matches) > 1:
                raise ValueError(f"duplicate execution key: {key}")
            if matches:
                records[matches[0]] = deepcopy(record)
            else:
                records.append(deepcopy(record))
            self.payload["updated_at"] = utc_now()
            save_checkpoint(self.run_dir, self.payload)

    def snapshot(self) -> dict[str, Any]:
        with self._mutex:
            return deepcopy(self.payload)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        if pid > 0xFFFFFFFF:
            return False  # Do not let ctypes truncate an invalid PID to a live DWORD PID.
        # Windows maps signal 0 to CTRL_C_EVENT: os.kill(pid, 0) is not a probe.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not handle:
            error = ctypes.get_last_error()
            if error == 5:  # ERROR_ACCESS_DENIED: never steal an inaccessible owner's lock.
                return True
            if error == 87:  # ERROR_INVALID_PARAMETER: PID does not exist.
                return False
            raise ctypes.WinError(error)
        try:
            status = kernel32.WaitForSingleObject(handle, 0)
            if status == 0x102:  # WAIT_TIMEOUT: process is still running.
                return True
            if status == 0:  # WAIT_OBJECT_0: process has exited.
                return False
            raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """Cross-process lock based on exclusive file creation with stale PID recovery."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / LOCK_NAME
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
                owner_pid = int(owner.get("pid", -1))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                owner_pid = -1
            if _pid_is_alive(owner_pid):
                raise RuntimeError(f"run directory is locked by pid {owner_pid}")
            self.path.unlink(missing_ok=True)

        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.path, flags)
        except FileExistsError as exc:
            raise RuntimeError("run directory is locked") from exc
        payload = canonical_json(
            {"pid": os.getpid(), "token": self.token, "created_at": utc_now()}
        ).encode("utf-8")
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            if owner.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        finally:
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
