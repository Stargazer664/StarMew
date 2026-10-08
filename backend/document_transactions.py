"""Single-host document replacement/deletion with durable compensation.

The journal is written before the first mutation. A crash is recovered by
restoring the snapshot, unless the committed marker was already persisted.
This is recovery, not a distributed ACID transaction across the three stores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil

from .document_loader import DocumentLoader
from .upload_jobs import run_serialized_index_mutation
from .document_diagnostics import current_attempt, document_operation
from .telemetry import bind_context, event, span


class RecoveryRequiredError(RuntimeError):
    def __init__(self, message, *, failed_job_id=None, outcomes=None, blocked_job_ids=None, metadata_job_ids=None):
        super().__init__(message)
        self.failed_job_id = failed_job_id
        self.outcomes = outcomes or {}
        self.blocked_job_ids = blocked_job_ids or []
        self.metadata_job_ids = metadata_job_ids or []


def transaction_root() -> Path:
    return Path(os.getenv("DOCUMENT_TRANSACTION_DIR") or Path(__file__).resolve().parent.parent / "data" / "document_transactions")


def assert_index_available(root: Path | None = None) -> None:
    """Used under the index gate so retrieval cannot observe a half-commit."""
    for path in (root or transaction_root()).glob("*.json"):
        try:
            phase = json.loads(path.read_text(encoding="utf-8"))["phase"]
        except (ValueError, KeyError, OSError) as exc:
            raise RecoveryRequiredError("索引恢复日志不可读，请先恢复数据") from exc
        if phase not in {"committed", "rolled_back"}:
            raise RecoveryRequiredError("索引存在未完成事务，请先恢复数据")


def _copy_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".restore.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class DocumentTransactionService:
    def __init__(self, root, upload_root, loader, writer, parent_store, embedding):
        self.root, self.upload_root = Path(root).resolve(), Path(upload_root).resolve()
        self.loader, self.writer = loader, writer
        self.parents, self.embedding = parent_store, embedding
        self.vectors = writer.milvus_manager
        self.native_bm25 = getattr(self.vectors, "native_bm25", False)

    def _journal_path(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("任务 ID 无效")
        return self.root / (job_id + ".json")

    def _canonical(self, filename):
        if DocumentLoader.validate_filename(filename) != filename:
            raise ValueError("文件名必须是安全的文件基名")
        path = (self.upload_root / filename).resolve()
        if path.parent != self.upload_root:
            raise ValueError("文件路径不安全")
        return path

    def _save_manifest(self, manifest):
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._journal_path(manifest["job_id"])
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, default=lambda value: value.tolist())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)

    def _publish_file(self, source, destination):
        _copy_atomic(source, destination)

    def replace(self, job_id, staging_path, filename, progress=None):
        return run_serialized_index_mutation(self._replace, job_id, Path(staging_path), filename, progress)

    def _replace(self, job_id, staging_path, filename, progress):
        assert_index_available(self.root)
        self._journal_path(job_id)
        canonical = self._canonical(filename)
        expected = self.upload_root / ".staging" / job_id / filename
        if staging_path.resolve() != expected.resolve() or not staging_path.is_file():
            raise ValueError("暂存文件不存在或路径不安全，请重新上传")
        def notify(step, percent, message):
            if progress:
                progress(step, percent, message)
        notify("parse", 5, "正在解析文档")
        with document_operation("parse", "loader", "DOCUMENT_PARSE_FAILED"):
            docs = self.loader.load_document(str(staging_path), filename)
            if not docs:
                raise ValueError("文档处理失败，未能提取内容")
            parents = [doc for doc in docs if int(doc.get("chunk_level", 0) or 0) in (1, 2)]
            leaves = [doc for doc in docs if int(doc.get("chunk_level", 0) or 0) == 3]
            if not leaves:
                raise ValueError("文档处理失败，未生成可检索叶子分块")
            for doc in docs:
                doc["file_path"] = str(canonical)
        notify("parse", 100, f"解析完成：父块 {len(parents)}，叶子分块 {len(leaves)}")
        notify("vector_store", 0, "正在准备新向量，旧版本尚未修改")
        with document_operation("dense_prepare", "embedding", "DENSE_PREPARE_FAILED"):
            prepared = self.writer.prepare_dense(leaves)
        with document_operation("collection_init", "milvus", "COLLECTION_INIT_FAILED"):
            self.vectors.init_collection()
        with self.embedding.state_lock():
            with document_operation("vector_snapshot", "milvus", "VECTOR_SNAPSHOT_FAILED"):
                old_vectors = self.vectors.snapshot_documents(filename)
            with document_operation("parent_snapshot", "postgresql", "PARENT_SNAPSHOT_FAILED"):
                old_parents = self.parents.snapshot_documents(filename)
            self.root.mkdir(parents=True, exist_ok=True)
            backup = self.root / (job_id + ".original")
            if canonical.exists():
                with document_operation("file_snapshot", "filesystem", "FILE_SNAPSHOT_FAILED"):
                    _copy_atomic(canonical, backup)
            manifest = dict(job_id=job_id, filename=filename, operation="upload", phase="prepared",
                            attempt=current_attempt(),
                            vectors=old_vectors, parents=old_parents, bm25=self.embedding.snapshot_state(),
                            file_existed=canonical.exists(), result={"parent_count": len(parents), "leaf_count": len(leaves)})
            manifest["bm25_backend"] = "native" if self.native_bm25 else "legacy"
            manifest["collection_name"] = getattr(self.vectors, "collection_name", None)
            with document_operation("snapshot_save", "journal", "SNAPSHOT_SAVE_FAILED"):
                self._save_manifest(manifest)
            try:
                manifest["phase"] = "applying"
                self._save_manifest(manifest)
                if not self.native_bm25:  # Compatibility with legacy recovery adapters only.
                    with document_operation("bm25_remove", "bm25", "BM25_UPDATE_FAILED"):
                        self.embedding.increment_remove_documents([row["text"] for row in old_vectors])
                    with document_operation("bm25_add", "bm25", "BM25_UPDATE_FAILED"):
                        self.embedding.increment_add_documents([row["text"] for row in leaves])
                    with document_operation("sparse_prepare", "bm25", "SPARSE_PREPARE_FAILED"):
                        sparse = self.embedding.get_sparse_embeddings([row["text"] for row in leaves])
                        if len(sparse) != len(prepared):
                            raise ValueError("稀疏向量数量与分块数量不一致")
                        for row, vector in zip(prepared, sparse):
                            row["sparse_embedding"] = vector
                notify("cleanup", 10, "恢复快照已保存，正在替换旧索引")
                with document_operation("vector_delete", "milvus", "VECTOR_DELETE_FAILED"):
                    self.vectors.delete(f"filename == {json.dumps(filename, ensure_ascii=False)}")
                notify("cleanup", 100, "旧向量已清理，失败时将恢复旧版本")
                notify("parent_store", 20, "正在提交父级分块")
                with document_operation("parent_replace", "postgresql", "PARENT_WRITE_FAILED"):
                    self.parents.replace_documents(filename, parents)
                notify("parent_store", 100, "父级分块已提交")
                with document_operation("vector_insert", "milvus", "VECTOR_WRITE_FAILED"):
                    self.writer.write_prepared(prepared, progress_callback=lambda n, total: notify(
                        "vector_store", round(n * 100 / total), f"正在提交向量：{n} / {total}"))
                with document_operation("file_publish", "filesystem", "FILE_PUBLISH_FAILED"):
                    self._publish_file(staging_path, canonical)
                manifest["phase"] = "committed"
                with document_operation("manifest_commit", "journal", "MANIFEST_COMMIT_FAILED"):
                    self._save_manifest(manifest)
            except Exception as exc:
                manifest["error"] = str(exc)
                manifest["cause_event_id"] = event("primary_error", status="failed", error_type=type(exc).__name__)
                self._rollback(manifest)
                raise
        return manifest["result"]

    def delete(self, job_id, filename, progress=None):
        return run_serialized_index_mutation(self._delete, job_id, filename, progress)

    def _delete(self, job_id, filename, progress):
        assert_index_available(self.root)
        self._journal_path(job_id)
        self._canonical(filename)
        def notify(step, percent, message):
            if progress:
                progress(step, percent, message)
        notify("prepare", 10, "正在保存删除前快照")
        with document_operation("collection_init", "milvus", "COLLECTION_INIT_FAILED"):
            self.vectors.init_collection()
        with self.embedding.state_lock():
            with document_operation("vector_snapshot", "milvus", "VECTOR_SNAPSHOT_FAILED"):
                old_vectors = self.vectors.snapshot_documents(filename)
            with document_operation("parent_snapshot", "postgresql", "PARENT_SNAPSHOT_FAILED"):
                old_parents = self.parents.snapshot_documents(filename)
            manifest = dict(
                job_id=job_id, filename=filename, operation="delete", phase="prepared",
                attempt=current_attempt(), vectors=old_vectors, parents=old_parents,
                bm25=self.embedding.snapshot_state(), result={"deleted_count": len(old_vectors)},
            )
            manifest["bm25_backend"] = "native" if self.native_bm25 else "legacy"
            manifest["collection_name"] = getattr(self.vectors, "collection_name", None)
            with document_operation("snapshot_save", "journal", "SNAPSHOT_SAVE_FAILED"):
                self._save_manifest(manifest)
            try:
                manifest["phase"] = "applying"
                self._save_manifest(manifest)
                notify("prepare", 100, "删除前快照已保存")
                notify("bm25", 20, "正在同步 BM25 统计")
                if not self.native_bm25:
                    with document_operation("bm25_remove", "bm25", "BM25_UPDATE_FAILED"):
                        self.embedding.increment_remove_documents([row["text"] for row in old_vectors])
                notify("bm25", 100, "BM25 统计由 Milvus 随向量删除同步" if self.native_bm25 else "BM25 统计已同步")
                notify("milvus", 20, "正在删除向量数据")
                with document_operation("vector_delete", "milvus", "VECTOR_DELETE_FAILED"):
                    self.vectors.delete(f"filename == {json.dumps(filename, ensure_ascii=False)}")
                notify("milvus", 100, "向量数据已删除")
                notify("parent_store", 20, "正在删除父级分块")
                with document_operation("parent_replace", "postgresql", "PARENT_WRITE_FAILED"):
                    self.parents.replace_documents(filename, [])
                notify("parent_store", 100, "父级分块已删除，本地原文件保留")
                manifest["phase"] = "committed"
                with document_operation("manifest_commit", "journal", "MANIFEST_COMMIT_FAILED"):
                    self._save_manifest(manifest)
            except Exception as exc:
                manifest["error"] = str(exc)
                manifest["cause_event_id"] = event("primary_error", status="failed", error_type=type(exc).__name__)
                self._rollback(manifest)
                raise
        return manifest["result"]

    def _rollback(self, manifest):
        with bind_context(cause_event_id=manifest.get("cause_event_id")):
            event("rollback_started", status="running")
            with document_operation("rollback", "document", "ROLLBACK_FAILED", failure_event="rollback_failed"):
                self._rollback_stores(manifest)
            event("rollback_completed", status="completed")

    def _rollback_stores(self, manifest):
        recorded_collection = manifest.get("collection_name")
        if recorded_collection and recorded_collection != getattr(self.vectors, "collection_name", None):
            raise RecoveryRequiredError("恢复日志属于其他集合，请先切回原集合完成恢复")
        errors = []
        filename, job_id = manifest["filename"], manifest["job_id"]
        canonical = self._canonical(filename)
        def restore_vectors():
            self.vectors.delete(f"filename == {json.dumps(filename, ensure_ascii=False)}")
            rows = manifest["vectors"]
            for offset in range(0, len(rows), 50):
                batch = []
                for source in rows[offset:offset + 50]:
                    row = dict(source)
                    row.pop("id", None)
                    if manifest.get("bm25_backend") != "native":
                        row["sparse_embedding"] = {int(key): value for key, value in row["sparse_embedding"].items()}
                    batch.append(row)
                self.vectors.insert(batch)
        def restore_file():
            # Deletion only changes the index. It must not overwrite or remove
            # the original file, including during crash recovery.
            if manifest.get("operation", "upload") == "delete":
                return
            if manifest["file_existed"]:
                _copy_atomic(self.root / (job_id + ".original"), canonical)
            else:
                canonical.unlink(missing_ok=True)
        for name, code, action in (
            ("Milvus", "VECTOR_RESTORE_FAILED", restore_vectors),
            ("PostgreSQL", "PARENT_RESTORE_FAILED", lambda: self.parents.replace_documents(filename, manifest["parents"])),
            ("BM25", "BM25_RESTORE_FAILED", lambda: None if manifest.get("bm25_backend") == "native" else self.embedding.restore_state(manifest["bm25"])),
            ("file", "FILE_RESTORE_FAILED", restore_file),
        ):
            try:
                with document_operation("restore_" + name.lower(), name.lower(), code):
                    action()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        manifest["phase"] = "recovery_required" if errors else "rolled_back"
        manifest["recovery_errors"] = errors
        with document_operation("rollback_manifest", "journal", "ROLLBACK_MANIFEST_FAILED"):
            self._save_manifest(manifest)
        if errors:
            raise RecoveryRequiredError("旧版本恢复未完成，请恢复依赖后重试恢复：" + "; ".join(errors))

    def recover(self, on_recovered=None):
        return run_serialized_index_mutation(self._recover, on_recovered)

    def _recover(self, on_recovered=None):
        results = {}
        metadata_errors = {}
        with self.embedding.state_lock():
            paths = sorted(self.root.glob("*.json"))
            for index, path in enumerate(paths):
                try:
                    with bind_context(job_id=path.stem):
                        with document_operation("journal_read", "journal", "JOURNAL_UNREADABLE"):
                            manifest = json.loads(path.read_text(encoding="utf-8"))
                            if self._journal_path(manifest["job_id"]) != path:
                                raise ValueError("恢复日志 ID 不匹配")
                            operation = manifest.get("operation", "upload")
                            if operation not in {"upload", "delete"}:
                                raise ValueError("恢复日志操作类型无效")
                        with bind_context(operation=operation, attempt=manifest.get("attempt", 1)):
                            with document_operation("recover_job", "document", "RECOVERY_FAILED", failure_event="recovery_failed"):
                                if manifest["phase"] not in {"committed", "rolled_back"}:
                                    self._rollback(manifest)
                                outcome = {
                                    "operation": operation,
                                    "attempt": manifest.get("attempt", 1),
                                    "status": "completed" if manifest["phase"] == "committed" else "failed",
                                    "result": manifest.get("result", {}),
                                    "error": manifest.get("error", "服务中断，已恢复原版本，可重试"),
                                }
                                event("recovery_completed", status="completed")
                            if on_recovered:
                                try:
                                    on_recovered(manifest["job_id"], outcome)
                                except Exception as exc:
                                    # Data is terminal. Retain the journal and continue
                                    # other jobs; task metadata/cleanup cannot require rollback.
                                    metadata_errors[manifest["job_id"]] = exc
                            results[manifest["job_id"]] = outcome
                except Exception as exc:
                    blocked = [item.stem for item in paths[index + 1:]]
                    for job_id in blocked:
                        event("task_blocked", status="blocked", code="BLOCKED_BY_RECOVERY",
                              job_id=job_id, blocked_by_job_id=path.stem)
                    raise RecoveryRequiredError(f"任务 {path.stem} 的数据恢复未完成：{exc}",
                                                failed_job_id=path.stem, outcomes=results,
                                                blocked_job_ids=blocked,
                                                metadata_job_ids=list(metadata_errors)) from exc
        if metadata_errors:
            first_error = next(iter(metadata_errors.values()))
            raise RecoveryRequiredError("数据恢复已完成，部分任务状态或恢复文件清理仍待重试",
                                        outcomes=results, metadata_job_ids=list(metadata_errors)) from first_error
        return results

    def acknowledge(self, job_id):
        """Only after durable task status is saved may its recovery artifacts go."""
        with document_operation("journal_cleanup", "filesystem", "CLEANUP_FAILED", failure_event="cleanup_failed"):
            self._acknowledge(job_id)
        event("cleanup_completed", status="completed", stage="journal_cleanup")

    def _acknowledge(self, job_id):
        path = self._journal_path(job_id)
        if path.exists():
            phase = json.loads(path.read_text(encoding="utf-8"))["phase"]
            if phase not in {"committed", "rolled_back"}:
                raise RecoveryRequiredError("不能删除仍需恢复的事务日志")
        # Keep the retry marker until every recovery artifact has been removed.
        (self.root / (job_id + ".original")).unlink(missing_ok=True)
        path.unlink(missing_ok=True)
