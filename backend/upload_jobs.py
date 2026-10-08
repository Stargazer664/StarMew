"""任务进度与显式重试状态，支持本机 SQLite 持久化和独立内存模式。

SQLite 事务保证多个 manager 更新同一任务时不会覆盖彼此的新状态。
索引变更锁仍只覆盖当前进程，应用应以 single-worker 运行。
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import Event, Lock, RLock, local
from typing import Callable, Literal, TypeVar
from uuid import uuid4

from .telemetry import event, record_job


StepStatus = Literal["pending", "running", "completed", "failed"]
JobStatus = Literal["pending", "running", "completed", "failed"]
ResultT = TypeVar("ResultT")


_index_mutation_lock = RLock()
_index_stopping = Event()
_index_local = local()
_index_generation = 0


def run_serialized_index_mutation(processor: Callable[..., ResultT], *args, **kwargs) -> ResultT:
    """在当前服务进程内串行执行会更新共享检索状态的任务。"""
    generation = _index_generation
    with _index_mutation_lock:
        depth = getattr(_index_local, "depth", 0)
        if depth == 0 and (_index_stopping.is_set() or generation != _index_generation):
            raise RuntimeError("后端正在关闭，不能开始新的索引操作")
        _index_local.depth = depth + 1
        try:
            return processor(*args, **kwargs)
        finally:
            _index_local.depth = depth


def stop_index_operations() -> None:
    """拒绝新操作，并等当前线程的完整事务（含嵌套补偿）退出。"""
    _index_stopping.set()
    with _index_mutation_lock:
        pass


def resume_index_operations() -> None:
    """仅在取得后端进程独占权限后，开启新的服务生命周期。"""
    global _index_generation
    with _index_mutation_lock:
        _index_generation += 1
        _index_stopping.clear()


def create_upload_staging_path(upload_root: Path, job_id: str, filename: str) -> Path:
    """为单次上传创建与同名任务互不覆盖的临时落盘路径。"""
    staging_dir = upload_root / ".staging" / job_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    return staging_dir / filename


DEFAULT_STEPS = [
    ("upload", "文档上传"),
    ("parse", "解析与分块"),
    ("cleanup", "清理旧版本"),
    ("parent_store", "父级分块入库"),
    ("vector_store", "向量化入库"),
]

DELETE_STEPS = [
    ("prepare", "准备删除"),
    ("bm25", "同步 BM25 统计"),
    ("milvus", "删除向量数据"),
    ("parent_store", "删除父级分块"),
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class UploadJobManager:
    """线程安全的任务容器；store_path=None 时仅保存到实例内存。

    构造及模块导入不打开数据库，也不会自动修改未完成任务。
    恢复数据事务之后，由应用启动流程显式调用 mark_interrupted。
    """

    def __init__(self, store_path: str | Path | None = None, namespace: str = "upload"):
        self._jobs: dict[str, dict] = {}
        self._lock = Lock()
        self._store_path = Path(store_path) if store_path is not None else None
        self._namespace = namespace

    @contextmanager
    def _transaction(self, *, write: bool = False):
        with self._lock:
            if self._store_path is None:
                previous = deepcopy(self._jobs) if write else None
                try:
                    yield None
                except BaseException:
                    if previous is not None:
                        self._jobs = previous
                    raise
                return

            if not write and not self._store_path.exists():
                yield None
                return
            if write:
                self._store_path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self._store_path, timeout=30, isolation_level=None)
            else:
                # 状态查询不会意外创建数据库或改写数据库结构。
                uri = self._store_path.resolve().as_uri() + "?mode=ro"
                connection = sqlite3.connect(uri, uri=True, timeout=30, isolation_level=None)
            try:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                if write:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS document_jobs ("
                        "namespace TEXT NOT NULL, job_id TEXT NOT NULL, payload TEXT NOT NULL, "
                        "PRIMARY KEY (namespace, job_id))"
                    )
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def _has_table(connection: sqlite3.Connection) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'document_jobs'"
        ).fetchone() is not None

    def _read_job(self, connection: sqlite3.Connection | None, job_id: str) -> dict | None:
        if connection is None:
            return deepcopy(self._jobs.get(job_id))
        if not self._has_table(connection):
            return None
        row = connection.execute(
            "SELECT payload FROM document_jobs WHERE namespace = ? AND job_id = ?",
            (self._namespace, job_id),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _read_jobs(self, connection: sqlite3.Connection | None) -> list[dict]:
        if connection is None:
            return deepcopy(list(self._jobs.values()))
        if not self._has_table(connection):
            return []
        rows = connection.execute(
            "SELECT payload FROM document_jobs WHERE namespace = ? ORDER BY rowid",
            (self._namespace,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _save_job(self, connection: sqlite3.Connection | None, job: dict) -> dict:
        if connection is None:
            self._jobs[job["job_id"]] = deepcopy(job)
        else:
            payload = json.dumps(job, ensure_ascii=False, allow_nan=False)
            connection.execute(
                "INSERT INTO document_jobs (namespace, job_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(namespace, job_id) DO UPDATE SET payload = excluded.payload",
                (self._namespace, job["job_id"], payload),
            )
        return deepcopy(job)

    def _mutate_job(self, job_id: str, mutate: Callable[[dict], dict | None]) -> dict | None:
        with self._transaction(write=True) as connection:
            job = self._read_job(connection, job_id)
            if job is None:
                return None
            updated = mutate(job)
            if updated is None:
                return None
            updated["updated_at"] = _now_iso()
            saved = self._save_job(connection, updated)
        self._record_snapshot(saved)
        return saved

    def _record_snapshot(self, job: dict) -> None:
        # Never let optional diagnostics alter a committed business transaction.
        try:
            record_job(job, self._namespace)
        except Exception:
            pass

    def create_job(
        self,
        filename: str,
        *,
        steps: list[tuple[str, str]] | None = None,
        current_step: str = "upload",
        message: str = "等待上传",
        completion_step: str = "vector_store",
        job_id: str | None = None,
    ) -> dict:
        steps = steps or DEFAULT_STEPS
        if job_id is None:
            job_id = uuid4().hex
        if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-fA-F]{32}", job_id) is None:
            raise ValueError("job_id 必须为 32 位十六进制字符串")
        now = _now_iso()
        job = {
            "job_id": job_id,
            "filename": filename,
            "status": "pending",
            "current_step": current_step,
            "message": message,
            # 完成节点用于区分上传和删除，避免 complete_job 写死最后一步。
            "completion_step": completion_step,
            "total_chunks": 0,
            "processed_chunks": 0,
            "error": None,
            "retryable": False,
            "recovery_required": False,
            "attempt": 1,
            "staging_path": None,
            "created_at": now,
            "updated_at": now,
            "steps": [
                {
                    "key": key,
                    "label": label,
                    "percent": 0,
                    "status": "pending",
                    "message": "",
                }
                for key, label in steps
            ],
        }
        with self._transaction(write=True) as connection:
            existing = self._read_job(connection, job_id)
            if existing is not None:
                if existing["filename"] != filename:
                    raise ValueError("该 job_id 已用于其他文件")
                return existing
            saved = self._save_job(connection, job)
        self._record_snapshot(saved)
        return saved

    def get_job(self, job_id: str) -> dict | None:
        with self._transaction() as connection:
            return self._read_job(connection, job_id)

    def update_job(self, job_id: str, **fields) -> dict | None:
        """原子更新任务元数据；任务标识、文件名和创建时间不可更改。"""
        protected = {"job_id", "filename", "created_at", "updated_at"}
        if protected.intersection(fields):
            raise ValueError("不能修改任务标识、文件名或时间戳")

        def update(job: dict) -> dict:
            unknown = fields.keys() - job.keys()
            if unknown:
                raise ValueError(f"未知任务字段: {', '.join(sorted(unknown))}")
            job.update(deepcopy(fields))
            if job.get("staging_path") is not None:
                job["staging_path"] = str(job["staging_path"])
            return job

        return self._mutate_job(job_id, update)

    def update_step(
        self,
        job_id: str,
        step_key: str,
        percent: int,
        status: StepStatus = "running",
        message: str = "",
        *,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
    ) -> dict | None:
        percent = max(0, min(100, int(percent)))
        def update(job: dict) -> dict | None:
            step = self._find_step(job, step_key)
            if not step:
                return None

            step["percent"] = percent
            step["status"] = status
            step["message"] = message
            job["status"] = "failed" if status == "failed" else "running"
            job["current_step"] = step_key
            job["message"] = message

            if total_chunks is not None:
                job["total_chunks"] = int(total_chunks)
            if processed_chunks is not None:
                job["processed_chunks"] = int(processed_chunks)

            return job

        return self._mutate_job(job_id, update)

    def complete_step(self, job_id: str, step_key: str, message: str = "") -> dict | None:
        return self.update_step(job_id, step_key, 100, "completed", message)

    def complete_job(self, job_id: str, message: str = "文档入库完成") -> dict | None:
        def complete(job: dict) -> dict:
            for step in job["steps"]:
                # A durable commit may recover metadata previously marked failed.
                if step["status"] == "failed":
                    step["message"] = message
                step["percent"] = 100
                step["status"] = "completed"
            job["status"] = "completed"
            job["current_step"] = job.get("completion_step") or job["current_step"]
            job["message"] = message
            job["error"] = None
            job["retryable"] = False
            job["recovery_required"] = False
            return job

        return self._mutate_job(job_id, complete)

    def fail_job(self, job_id: str, step_key: str, error: str) -> dict | None:
        def fail(job: dict) -> dict:
            step = self._find_step(job, step_key)
            if step:
                step["status"] = "failed"
                step["message"] = error
            job["status"] = "failed"
            job["current_step"] = step_key
            job["message"] = error
            job["error"] = error
            return job

        return self._mutate_job(job_id, fail)

    def list_jobs(self) -> list[dict]:
        with self._transaction() as connection:
            return self._read_jobs(connection)

    def retry_job(self, job_id: str) -> dict | None:
        """领取一次显式重试；检查与状态重置在同一写事务内完成。"""
        def retry(job: dict) -> dict:
            if job["status"] != "failed" or not job.get("retryable"):
                raise ValueError("任务当前不可重试")
            if job.get("recovery_required"):
                raise ValueError("任务需先完成数据恢复")
            for step in job["steps"]:
                if step["key"] == "upload" and step["status"] == "completed":
                    continue
                step.update(percent=0, status="pending", message="")
            next_step = next(
                (step for step in job["steps"] if step["status"] != "completed"), None
            )
            job.update(
                status="pending",
                current_step=next_step["key"] if next_step else job["current_step"],
                message="等待重试",
                error=None,
                retryable=False,
                total_chunks=0,
                processed_chunks=0,
                attempt=int(job.get("attempt", 1)) + 1,
            )
            return job

        return self._mutate_job(job_id, retry)

    def mark_interrupted(self, *, exclude_job_ids=()) -> list[dict]:
        """启动恢复完成后，将仍在排队或运行的任务明确标记为中断。"""
        interrupted = []
        with self._transaction(write=True) as connection:
            for job in self._read_jobs(connection):
                if job["job_id"] in exclude_job_ids:
                    continue
                if job["status"] not in {"pending", "running"}:
                    continue
                uploaded = self._find_step(job, "upload")
                staging_path = job.get("staging_path")
                retryable = bool(
                    uploaded and uploaded["status"] == "completed"
                    and staging_path and Path(staging_path).is_file()
                    and not job.get("recovery_required")
                )
                error = "服务重启导致任务中断，请重试" if retryable else "服务重启导致任务中断"
                step = self._find_step(job, job["current_step"])
                if step and step["status"] != "completed":
                    step.update(status="failed", message=error)
                job.update(
                    status="failed", message=error, error=error,
                    retryable=retryable, updated_at=_now_iso(),
                )
                interrupted.append(self._save_job(connection, job))
        for job in interrupted:
            self._record_snapshot(job)
            event("task_interrupted", status="interrupted", job_id=job["job_id"],
                  operation=self._namespace, attempt=job.get("attempt", 1), scope="document")
        return interrupted

    @staticmethod
    def _find_step(job: dict, step_key: str) -> dict | None:
        for step in job["steps"]:
            if step["key"] == step_key:
                return step
        return None


_default_store_path = Path(__file__).resolve().parents[1] / "data" / "upload_jobs.sqlite3"
_store_path = os.environ.get("UPLOAD_JOB_DB_PATH") or _default_store_path
upload_job_manager = UploadJobManager(_store_path, namespace="upload")
delete_job_manager = UploadJobManager(_store_path, namespace="delete")
