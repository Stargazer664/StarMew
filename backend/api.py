import asyncio
import json
import os
import re
import logging
from pathlib import Path
from typing import Optional as OptionalType
from uuid import UUID, uuid4
from types import SimpleNamespace

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from contextlib import aclosing
from sqlalchemy.orm import Session

import sys
print("[api] importing agent...", file=sys.stderr, flush=True)
from .agent import chat_with_agent, chat_with_agent_stream, storage
from .conversation_turns import admit_async, TurnConflict, request_cancel, run_status
from .chat_diagnostics import run_blocking
print("[api] importing auth...", file=sys.stderr, flush=True)
from .auth import authenticate_user, create_access_token, get_current_user, get_db, get_password_hash, require_admin, resolve_role
print("[api] importing document_loader...", file=sys.stderr, flush=True)
from .document_loader import DocumentLoader
print("[api] importing embedding...", file=sys.stderr, flush=True)
from .embedding import embedding_service
print("[api] importing models...", file=sys.stderr, flush=True)
from .models import User
from .schemas import (
    AuthResponse,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    ChatRequest,
    ChatCancelRequest,
    ChatResponse,
    CurrentUserResponse,
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
    EvalDatasetResponse,
    EvalSampleSchema,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationHistoryResponse,
    EvaluationMetrics,
    EvaluationResult,
    LoginRequest,
    MemoryResetResponse,
    MemoryResponse,
    MemoryUpdateResponse,
    MessageInfo,
    RegisterRequest,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
)
from .memory import memory_service
from .evaluation import evaluator
from .document_transactions import DocumentTransactionService, RecoveryRequiredError, assert_index_available, transaction_root
from .document_diagnostics import document_execution, document_operation, recovery_execution, mark_document_blocked
from .telemetry import bind_context, event
from time import perf_counter_ns
from .upload_jobs import (
    DELETE_STEPS,
    create_upload_staging_path,
    delete_job_manager,
    run_serialized_index_mutation,
    upload_job_manager,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "documents"
logger = logging.getLogger(__name__)

# 延迟导入重型模块
_loader = None
_parent_chunk_store = None
_milvus_manager = None
_milvus_writer = None
_document_transactions = None


def _get_document_transactions():
    global _document_transactions
    if _document_transactions is None:
        _document_transactions = DocumentTransactionService(
            transaction_root(), UPLOAD_DIR,
            SimpleNamespace(load_document=lambda *args: _get_loader().load_document(*args)), _get_milvus_writer(),
            _get_parent_chunk_store(), embedding_service,
        )
    return _document_transactions


def _get_loader():
    global _loader
    if _loader is None:
        _loader = DocumentLoader()
    return _loader


def _get_parent_chunk_store():
    global _parent_chunk_store
    if _parent_chunk_store is None:
        from .parent_chunk_store import ParentChunkStore
        _parent_chunk_store = ParentChunkStore()
    return _parent_chunk_store


def _get_milvus_manager():
    global _milvus_manager
    if _milvus_manager is None:
        from .milvus_client import MilvusManager
        _milvus_manager = MilvusManager()
    return _milvus_manager


def _get_milvus_writer():
    global _milvus_writer
    if _milvus_writer is None:
        from .milvus_writer import MilvusWriter
        _milvus_writer = MilvusWriter(
            embedding_service=embedding_service,
            milvus_manager=_get_milvus_manager()
        )
    return _milvus_writer


router = APIRouter()


@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    username = (request.username or "").strip()
    password = (request.password or "").strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=409, detail="用户名已存在")

    role = resolve_role(request.role, request.admin_code)
    user = User(username=username, password_hash=get_password_hash(password), role=role)
    db.add(user)
    db.commit()

    token = create_access_token(username=username, role=role)
    return AuthResponse(access_token=token, username=username, role=role)


@router.post("/auth/login", response_model=AuthResponse)
async def login(
    request: Request,  # 支持多种 body 格式
    username: OptionalType[str] = Form(None),
    password: OptionalType[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    支持两种调用方式：
    1. JSON body: {"username": "...", "password": "..."}  (前端)
    2. Form data: username=...&password=...  (OpenAPI docs 测试用)
    """
    # 如果 Form 参数为空，尝试从 JSON body 读取
    if username is None or password is None:
        try:
            body = await request.json()
            username = body.get("username", "").strip()
            password = body.get("password", "").strip()
        except Exception:
            pass

    username = (username or "").strip()
    password = (password or "").strip()

    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")

    user = authenticate_user(db, username, password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(username=user.username, role=user.role)
    return AuthResponse(access_token=token, username=user.username, role=user.role)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return CurrentUserResponse(username=current_user.username, role=current_user.role)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    """获取指定会话的所有消息"""
    try:
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                status=msg.get('status', 'complete'),
                rag_trace=msg.get("rag_trace"),
            )
            for msg in await run_blocking(storage.get_session_messages, current_user.username, session_id)
        ]
        return SessionMessagesResponse(messages=messages,
            pending_run=await run_blocking(storage.get_pending_run, current_user.username, session_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    """获取当前用户的所有会话列表"""
    try:
        sessions = [SessionInfo(**item) for item in await run_blocking(storage.list_session_infos, current_user.username)]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    """删除当前用户的指定会话"""
    try:
        deleted = await run_blocking(storage.delete_session, current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在")
        return SessionDeleteResponse(session_id=session_id, message="成功删除会话")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        # Cancellation cannot stop a Python worker. Drain it before returning so
        # its admission stays owned until its real commit/cleanup has completed.
        worker = asyncio.create_task(asyncio.to_thread(chat_with_agent,
            request.message, current_user.username, session_id, request_id=request.request_id))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        if cancelled:
            worker.exception()
            raise asyncio.CancelledError
        resp = worker.result()
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except TurnConflict as e:
        raise HTTPException(status_code=409, detail={"code": str(e)})
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post('/chat/runs/{request_id}/cancel')
async def cancel_chat_run(request_id: UUID, request: ChatCancelRequest, current_user: User = Depends(get_current_user)):
    try:
        return await run_blocking(request_cancel, current_user.username, request.session_id, str(request_id),
            supplement_message=request.supplement_message, supplement_request_id=request.supplement_request_id)
    except TurnConflict as exc:
        raise HTTPException(status_code=404 if str(exc) == 'REQUEST_NOT_FOUND' else 409, detail={'code': str(exc)}) from None


@router.get('/chat/runs/{request_id}')
async def get_chat_run(request_id: UUID, session_id: str = 'default_session', current_user: User = Depends(get_current_user)):
    try:
        return await run_blocking(run_status, current_user.username, session_id, str(request_id))
    except TurnConflict as exc:
        raise HTTPException(status_code=404 if str(exc) == 'REQUEST_NOT_FOUND' else 409, detail={'code': str(exc)}) from None


class AdmittedStreamingResponse(StreamingResponse):
    def __init__(self, *args, admission, **kwargs):
        super().__init__(*args, **kwargs)
        self.admission = admission

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                # Also runs if send(headers) failed before the generator started.
                await run_blocking(self.admission.close, 'cancelled')


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""
    session_id = request.session_id or "default_session"
    try:
        admission = await admit_async(current_user.username, session_id, request.request_id, request.message)
    except TurnConflict as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)})
    stream_function = chat_with_agent_stream

    async def event_generator():
        try:
            async with aclosing(stream_function(request.message, current_user.username, session_id,
                    request_id=admission.request_id, admission=admission)) as stream:
                async for chunk in stream:
                    yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return AdmittedStreamingResponse(
        event_generator(),
        admission=admission,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _is_supported_document(filename: str) -> bool:
    return DocumentLoader.is_supported(filename)


def _filename_filter_expr(filename: str) -> str:
    """使用 JSON 字符串转义生成安全的 Milvus 字面量。"""
    return f"filename == {json.dumps(filename, ensure_ascii=False)}"


def _validated_upload_path(filename: str) -> Path:
    file_path = (UPLOAD_DIR / filename).resolve()
    if file_path.parent != UPLOAD_DIR.resolve():
        raise HTTPException(status_code=400, detail="文件路径不安全")
    return file_path


async def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    """按块写入上传文件，避免大文件一次性读入内存。"""
    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())


def _cleanup_upload_stage(job: dict) -> None:
    """Only remove a verified, completed task's staging file."""
    expected = (UPLOAD_DIR / ".staging" / job["job_id"] / job["filename"]).resolve()
    supplied = Path(job.get("staging_path") or expected).resolve()
    if supplied != expected or UPLOAD_DIR.resolve() not in supplied.parents:
        logger.error("Refusing unsafe staging cleanup for job %s", job["job_id"])
        return
    try:
        with document_operation("staging_cleanup", "filesystem", "CLEANUP_FAILED", failure_event="cleanup_failed",
                                job_id=job["job_id"], operation="upload", attempt=job.get("attempt", 1)):
            supplied.unlink(missing_ok=True)
            try:
                supplied.parent.rmdir()
            except FileNotFoundError:
                pass
        event("cleanup_completed", status="completed", job_id=job["job_id"], operation="upload", stage="staging_cleanup")
    except OSError:
        logger.warning("Staging cleanup pending for job %s", job["job_id"], exc_info=True)


def _transaction_job_manager(operation: str):
    if operation == "upload":
        return upload_job_manager
    if operation == "delete":
        return delete_job_manager
    raise RecoveryRequiredError("未知的文档事务类型")


@recovery_execution
def recover_upload_operations(*, mark_interrupted: bool = False) -> int:
    """Reconcile successful journals even when a later recovery stops the batch."""
    terminal_job_ids = set()
    def recover():
        service = _get_document_transactions()
        def reconcile(job_id, outcome):
            operation = outcome.get("operation", "upload")
            manager = _transaction_job_manager(operation)
            job = manager.get_job(job_id)
            if not job:
                event("recovery_failed", status="failed", code="JOB_RECORD_MISSING", job_id=job_id, operation=operation)
                return
            with bind_context(job_id=job_id, operation=operation, attempt=outcome.get("attempt", job.get("attempt", 1))):
                with document_operation("status_reconcile", "sqlite", "STATUS_PERSIST_FAILED"):
                    if outcome["status"] == "completed":
                        result = outcome["result"]
                        count = result.get("deleted_count" if operation == "delete" else "leaf_count", 0)
                        manager.update_job(job_id, total_chunks=count, processed_chunks=count)
                        manager.complete_job(job_id, "删除已完成，任务状态已恢复" if operation == "delete" else "入库已完成，任务状态已恢复")
                    else:
                        manager.fail_job(job_id, job["current_step"], outcome["error"] or "已恢复旧版本，可重试")
                        manager.update_job(
                            job_id, recovery_required=False,
                            retryable=operation == "delete" or bool(job.get("staging_path") and Path(job["staging_path"]).is_file()),
                        )
                event("status_reconciled", status="completed")
                service.acknowledge(job_id)
                if outcome["status"] == "completed" and operation == "upload":
                    _cleanup_upload_stage(job)
        recovery_error = None
        try:
            outcomes = service.recover(on_recovered=reconcile)
        except Exception as exc:
            recovery_error = exc
            outcomes = getattr(exc, "outcomes", {})
            terminal_job_ids.update(getattr(exc, "metadata_job_ids", []))
        if recovery_error is not None:
            failed_id = getattr(recovery_error, "failed_job_id", None)
            blocked_ids = getattr(recovery_error, "blocked_job_ids", [])
            for job_id in ([failed_id] if failed_id else []) + blocked_ids:
                for manager in (upload_job_manager, delete_job_manager):
                    job = manager.get_job(job_id)
                    if job and job["status"] != "completed":
                        operation = "upload" if manager is upload_job_manager else "delete"
                        if job_id != failed_id:
                            event("task_blocked", status="blocked", code="BLOCKED_BY_RECOVERY",
                                  job_id=job_id, operation=operation, attempt=job.get("attempt", 1),
                                  blocked_by_job_id=failed_id)
                        try:
                            phase = json.loads((service.root / (job_id + ".json")).read_text(encoding="utf-8"))["phase"]
                        except (OSError, ValueError, KeyError):
                            phase = None
                        if phase in {"committed", "rolled_back"}:
                            # The journal remains for status/cleanup retry; data is already final.
                            terminal_job_ids.add(job_id)
                            continue
                        # An unprocessed journal has no new failure of its own.
                        message = str(recovery_error) if job_id == failed_id else "尚未恢复：被其他任务的恢复阻塞"
                        manager.fail_job(job_id, job["current_step"], message)
                        manager.update_job(job_id, recovery_required=True, retryable=False)
            raise recovery_error
        return len(outcomes)
    try:
        return run_serialized_index_mutation(recover)
    finally:
        if mark_interrupted:
            upload_job_manager.mark_interrupted(exclude_job_ids=terminal_job_ids)
            delete_job_manager.mark_interrupted(exclude_job_ids=terminal_job_ids)


@document_execution("upload")
def _process_upload_job(job_id: str, file_path: str, filename: str) -> None:
    """Persist progress, compensate on failure and keep the source for retries."""
    job = upload_job_manager.get_job(job_id)
    if not job or job["status"] in {"completed", "failed"}:
        return
    failed_step = "parse"
    service = _get_document_transactions()
    try:
        recover_upload_operations()
        recovered_job = upload_job_manager.get_job(job_id)
        if not recovered_job or recovered_job["status"] in {"completed", "failed"}:
            return
        def progress(step, percent, message):
            nonlocal failed_step
            failed_step = step
            upload_job_manager.update_step(job_id, step, percent,
                                           "completed" if percent == 100 else "running", message)
        result = service.replace(job_id, file_path, filename, progress)
        with document_operation("status_persist", "sqlite", "STATUS_PERSIST_FAILED"):
            upload_job_manager.update_job(job_id, total_chunks=result["leaf_count"], processed_chunks=result["leaf_count"])
            upload_job_manager.complete_job(job_id, f"成功上传并处理 {filename}")
    except Exception as exc:
        # A completed data commit must not be turned into a failed/rolled-back job
        # merely because persisting task progress was interrupted.
        journal = service.root / (job_id + ".json")
        try:
            phase = json.loads(journal.read_text(encoding="utf-8"))["phase"] if journal.exists() else None
        except (OSError, ValueError, KeyError):
            phase = "recovery_required"
        if phase == "committed":
            logger.exception("Data committed; task %s awaits status recovery", job_id)
            return
        # A different job's recovery may block admission before this job has a
        # journal. Only this job's journal can require its own compensation.
        needs_recovery = phase not in {None, "rolled_back"}
        blocked_by = getattr(exc, "failed_job_id", None) or next(iter(getattr(exc, "metadata_job_ids", [])), None)
        if blocked_by and blocked_by != job_id:
            mark_document_blocked(blocked_by)
        upload_job_manager.fail_job(job_id, failed_step, "尚未执行：被其他任务的恢复阻塞" if blocked_by and blocked_by != job_id else str(exc))
        upload_job_manager.update_job(job_id, retryable=Path(file_path).is_file() and not needs_recovery,
                                      recovery_required=needs_recovery)
        if not needs_recovery:
            service.acknowledge(job_id)
        return
    try:
        service.acknowledge(job_id)
        _cleanup_upload_stage(upload_job_manager.get_job(job_id))
    except Exception:
        logger.exception("Completed task %s still has recovery artifacts", job_id)


@document_execution("delete")
def _process_delete_job(job_id: str, filename: str) -> None:
    """Delete with the same journal, compensation and status recovery as uploads."""
    job = delete_job_manager.get_job(job_id)
    if not job or job["status"] in {"completed", "failed"}:
        return
    service = _get_document_transactions()
    failed_step = "prepare"
    try:
        recover_upload_operations()
        job = delete_job_manager.get_job(job_id)
        if not job or job["status"] in {"completed", "failed"}:
            return
        def progress(step, percent, message):
            nonlocal failed_step
            failed_step = step
            delete_job_manager.update_step(job_id, step, percent,
                                           "completed" if percent == 100 else "running", message)
        result = service.delete(job_id, filename, progress)
        count = result["deleted_count"]
        with document_operation("status_persist", "sqlite", "STATUS_PERSIST_FAILED"):
            delete_job_manager.update_job(job_id, total_chunks=count, processed_chunks=count)
            delete_job_manager.complete_job(job_id, f"已删除 {filename} 的检索索引，共 {count} 条；本地原文件保留")
    except Exception as exc:
        journal = service.root / (job_id + ".json")
        try:
            phase = json.loads(journal.read_text(encoding="utf-8"))["phase"] if journal.exists() else None
        except (OSError, ValueError, KeyError):
            phase = "recovery_required"
        if phase == "committed":
            logger.exception("Deletion committed; task %s awaits status recovery", job_id)
            return
        needs_recovery = phase not in {None, "rolled_back"}
        blocked_by = getattr(exc, "failed_job_id", None) or next(iter(getattr(exc, "metadata_job_ids", [])), None)
        if blocked_by and blocked_by != job_id:
            mark_document_blocked(blocked_by)
        delete_job_manager.fail_job(job_id, failed_step, "尚未执行：被其他任务的恢复阻塞" if blocked_by and blocked_by != job_id else str(exc))
        delete_job_manager.update_job(job_id, retryable=not needs_recovery, recovery_required=needs_recovery)
        if not needs_recovery:
            service.acknowledge(job_id)
        return
    try:
        service.acknowledge(job_id)
    except Exception:
        logger.exception("Completed delete task %s still has recovery artifacts", job_id)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_: User = Depends(require_admin)):
    """获取已上传的文档列表（管理员）"""
    try:
        def read():
            assert_index_available()
            _get_milvus_manager().init_collection()
            return _get_milvus_manager().query_all(output_fields=["filename", "file_type"])
        results = await asyncio.to_thread(run_serialized_index_mutation, read)

        file_stats = {}
        for item in results:
            filename = item.get("filename", "")
            file_type = item.get("file_type", "")
            if filename not in file_stats:
                file_stats[filename] = {
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_count": 0,
                }
            file_stats[filename]["chunk_count"] += 1

        documents = [DocumentInfo(**stats) for stats in file_stats.values()]
        return DocumentListResponse(documents=documents)
    except RecoveryRequiredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")

@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    upload_id: OptionalType[str] = Form(None),
    _: User = Depends(require_admin),
):
    """Receive once under a stable client ID, then run a recoverable background job."""
    try:
        filename = DocumentLoader.validate_filename(file.filename or "")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    job_id = upload_id or uuid4().hex
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=400, detail="upload_id 必须是 32 位小写十六进制字符串")
    if delete_job_manager.get_job(job_id):
        raise HTTPException(status_code=409, detail="该任务 ID 已用于删除操作")
    # No await between check/create: atomic admission on this single-worker event loop.
    existing = upload_job_manager.get_job(job_id)
    if existing:
        if existing["filename"] != filename:
            raise HTTPException(status_code=409, detail="该上传 ID 已用于其他文件")
        return DocumentUploadStartResponse(job_id=job_id, filename=filename, message=existing["message"])
    upload_job_manager.create_job(filename, job_id=job_id)
    file_path = None
    try:
        file_path = create_upload_staging_path(UPLOAD_DIR, job_id, filename)
        upload_job_manager.update_job(job_id, staging_path=str(file_path))
        upload_job_manager.update_step(job_id, "upload", 1, "running", "正在保存文件到服务器")
        with bind_context(scope="document", trace_id=f"doc:upload:{job_id}:1", job_id=job_id, operation="upload", attempt=1):
            with document_operation("receive", "filesystem", "RECEIVE_FAILED"):
                await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(job_id, "upload", "文件已上传，等待后台处理")
        upload_job_manager.update_job(job_id, retryable=True)
    except Exception as exc:
        upload_job_manager.fail_job(job_id, "upload", f"文件保存失败: {exc}")
        upload_job_manager.update_job(job_id, retryable=False)
        _cleanup_upload_stage(upload_job_manager.get_job(job_id))
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc
    background_tasks.add_task(run_serialized_index_mutation, _process_upload_job, job_id, str(file_path), filename, admitted_at=perf_counter_ns())
    return DocumentUploadStartResponse(job_id=job_id, filename=filename, message="文件已接收，正在后台处理")


@router.post("/documents/recover")
@router.post("/documents/upload/recover")
async def recover_upload_jobs(_: User = Depends(require_admin)):
    try:
        recovered = await asyncio.to_thread(recover_upload_operations)
        return {"recovered": recovered}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"数据恢复未完成：{exc}") from exc


@router.post("/documents/upload/jobs/{job_id}/retry", response_model=DocumentUploadStartResponse)
async def retry_upload_job(job_id: str, background_tasks: BackgroundTasks, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在")
    if job.get("recovery_required"):
        raise HTTPException(status_code=409, detail="请先恢复数据，再重试任务")
    expected = (UPLOAD_DIR / ".staging" / job_id / job["filename"]).resolve()
    if not job.get("staging_path") or Path(job["staging_path"]).resolve() != expected or not expected.is_file():
        upload_job_manager.update_job(job_id, retryable=False)
        raise HTTPException(status_code=409, detail="完整暂存文件不存在，请重新上传")
    try:
        job = upload_job_manager.retry_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(run_serialized_index_mutation, _process_upload_job, job_id, str(expected), job["filename"], admitted_at=perf_counter_ns())
    return DocumentUploadStartResponse(job_id=job_id, filename=job["filename"], message="正在使用已接收文件重新处理")


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(job_id: str, _: User = Depends(require_admin)):
    job = upload_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(_: User = Depends(require_admin)):
    jobs = upload_job_manager.list_jobs()
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return [DocumentUploadJobResponse(**job) for job in jobs]


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    delete_id: OptionalType[str] = None,
    _: User = Depends(require_admin),
):
    """Stable request IDs prevent replay from deleting a subsequently uploaded version."""
    try:
        if DocumentLoader.validate_filename(filename) != filename:
            raise ValueError("文件名必须是安全的文件基名")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = delete_id or uuid4().hex
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=400, detail="delete_id 必须是 32 位小写十六进制字符串")
    if upload_job_manager.get_job(job_id):
        raise HTTPException(status_code=409, detail="该任务 ID 已用于上传操作")
    existing = delete_job_manager.get_job(job_id)
    if existing:
        if existing["filename"] != filename:
            raise HTTPException(status_code=409, detail="该删除 ID 已用于其他文件")
        return DocumentDeleteStartResponse(job_id=job_id, filename=filename, message=existing["message"])
    job = delete_job_manager.create_job(
        filename,
        job_id=job_id,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(job["job_id"], "prepare", 1, "running", "删除任务已提交")
    background_tasks.add_task(
        run_serialized_index_mutation,
        _process_delete_job,
        job["job_id"],
        filename,
        admitted_at=perf_counter_ns(),
    )
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(job_id: str, _: User = Depends(require_admin)):
    job = delete_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


@router.get("/documents/delete/jobs", response_model=list[DocumentDeleteJobResponse])
async def list_delete_jobs(_: User = Depends(require_admin)):
    jobs = sorted(delete_job_manager.list_jobs(), key=lambda job: job["created_at"], reverse=True)
    return [DocumentDeleteJobResponse(**job) for job in jobs]


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    """Compatibility endpoint shares the exact same durable upload workflow."""
    background = BackgroundTasks()
    start = await upload_document_async(background, file, None, _)
    await background()
    job = upload_job_manager.get_job(start.job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=500, detail=f"文档处理失败（任务 {start.job_id}）：{job['message']}")
    return DocumentUploadResponse(filename=start.filename, chunks_processed=job["total_chunks"], message=job["message"])


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str, _: User = Depends(require_admin)):
    """Compatibility endpoint shares the recoverable deletion workflow."""
    background = BackgroundTasks()
    start = await delete_document_async(filename, background, None, _)
    await background()
    job = delete_job_manager.get_job(start.job_id)
    if job["status"] != "completed":
        code = 503 if job["recovery_required"] else 500
        try:
            assert_index_available()
        except RecoveryRequiredError:
            code = 503
        raise HTTPException(status_code=code, detail=f"删除未完成（任务 {start.job_id}）：{job['message']}")
    return DocumentDeleteResponse(filename=filename, chunks_deleted=job["total_chunks"], message=job["message"])


@router.get("/memory", response_model=MemoryResponse)
async def get_memory(current_user: User = Depends(get_current_user)):
    """获取当前用户的长期记忆"""
    try:
        entries = await run_blocking(memory_service.list_entries, current_user.id)
        memory = await run_blocking(memory_service.get_or_create, current_user.id)
        stats = await run_blocking(memory_service.get_memory_stats, current_user.id)
        return MemoryResponse(
            username=current_user.username,
            preferences=memory.preferences or {},
            facts=memory.facts or {},
            session_summaries=memory.session_summaries or {},
            detected_topics=memory.detected_topics or [],
            stats=stats,
            entries=entries,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取记忆失败: {str(e)}")


@router.post("/memory/update", response_model=MemoryUpdateResponse)
async def update_memory(current_user: User = Depends(get_current_user), session_id: str = "default_session"):
    """手动触发记忆更新"""
    try:
        await run_blocking(memory_service.update_from_session, current_user.id, session_id)
        memory = await run_blocking(memory_service.get_or_create, current_user.id)
        return MemoryUpdateResponse(
            message="记忆更新成功",
            updated_facts=memory.facts or {},
            updated_preferences=memory.preferences or {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新记忆失败: {str(e)}")


@router.delete("/memory/entries/{entry_id}")
async def forget_memory_entry(entry_id: int, current_user: User = Depends(get_current_user)):
    forgotten = await run_blocking(memory_service.forget_entry, current_user.id, entry_id)
    if not forgotten:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"message": "已遗忘该记忆", "entry_id": entry_id}


@router.delete("/memory/reset", response_model=MemoryResetResponse)
async def reset_memory(current_user: User = Depends(get_current_user)):
    """重置用户记忆"""
    try:
        await run_blocking(memory_service.reset_memory, current_user.id)
        return MemoryResetResponse(
            message="记忆已重置",
            username=current_user.username,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置记忆失败: {str(e)}")


# ============ RAG 评估 API ============

@router.post("/evaluation/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest, current_user: User = Depends(get_current_user)):
    """运行单条 RAG 评估"""
    try:
        result = evaluator.evaluate(
            question=request.question,
            answer=request.answer,
            retrieved_docs=request.retrieved_docs,
            ground_truth=request.ground_truth,
            rag_trace=request.rag_trace,
        )

        return EvaluationResponse(
            evaluation=EvaluationResult(
                timestamp=result.get("timestamp", ""),
                question=result.get("question", ""),
                answer=request.answer,
                answer_length=result.get("answer_length", 0),
                contexts_count=result.get("contexts_count", 0),
                metrics=EvaluationMetrics(
                    faithfulness=result.get("metrics", {}).get("faithfulness"),
                    answer_relevancy=result.get("metrics", {}).get("answer_relevancy"),
                    context_precision=result.get("metrics", {}).get("context_precision"),
                    context_recall=result.get("metrics", {}).get("context_recall"),
                    factual_correctness=result.get("metrics", {}).get("factual_correctness"),
                    answer_correctness=result.get("metrics", {}).get("answer_correctness"),
                ),
                overall_score=result.get("overall_score"),
                evaluation_status=result.get("evaluation_status", "unknown"),
                metric_status=result.get("metric_status", {}),
                metric_errors=result.get("metric_errors", {}),
                ragas_native=result.get("ragas_native"),
                ragas_version=result.get("ragas_version", ""),
                ragas_style=result.get("ragas_style", True),
                detailed=result.get("detailed", False),
                retrieval_mode=result.get("retrieval_mode"),
                rerank_applied=result.get("rerank_applied", False),
                auto_merge_applied=result.get("auto_merge_applied", False),
                latency=result.get("latency"),
                error=result.get("error"),
            ),
            average_scores=evaluator.get_average_scores(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@router.post("/evaluation/batch", response_model=BatchEvaluationResponse)
async def run_batch_evaluation(
    request: BatchEvaluationRequest,
    current_user: User = Depends(require_admin),
):
    """批量评测 RAG Pipeline（管理员）"""
    try:
        from .evaluation import EvalDatasetManager

        # 加载数据集
        if request.dataset_path:
            if request.dataset_path.endswith(".json"):
                samples = EvalDatasetManager.load_from_json(request.dataset_path)
            elif request.dataset_path.endswith(".csv"):
                samples = EvalDatasetManager.load_from_csv(request.dataset_path)
            else:
                raise HTTPException(status_code=400, detail="不支持的数据集格式，请使用 JSON 或 CSV")
        elif request.questions:
            from .evaluation import EvalSample
            samples = [EvalSample(question=q) for q in request.questions]
        else:
            # 默认使用内置评测数据集
            default_path = BASE_DIR.parent / "data" / "eval_samples.json"
            if default_path.exists():
                samples = EvalDatasetManager.load_from_json(str(default_path))
            else:
                raise HTTPException(status_code=400, detail="未找到评测数据集")

        # 导入 RAG Pipeline
        from .rag_pipeline import run_rag_graph

        # 运行批量评测
        results = evaluator.evaluate_batch_sync(
            samples=samples,
            rag_pipeline_fn=run_rag_graph,
        )

        # 导出结果
        export_path = f"data/eval_batch_{_utcnow().replace(':', '-')[:19]}.json"
        evaluator.export_results(BASE_DIR.parent / export_path)

        eval_results = [
            EvaluationResult(
                timestamp="",
                question=r.sample.question,
                answer=r.answer,
                answer_length=len(r.answer),
                contexts_count=len(r.retrieved_docs),
                metrics=EvaluationMetrics(
                    faithfulness=r.metrics.get("faithfulness"),
                    answer_relevancy=r.metrics.get("answer_relevancy"),
                    context_precision=r.metrics.get("context_precision"),
                    context_recall=r.metrics.get("context_recall"),
                    factual_correctness=r.metrics.get("factual_correctness"),
                    answer_correctness=r.metrics.get("answer_correctness"),
                ),
                overall_score=r.overall_score,
                evaluation_status=getattr(r, "evaluation_status", "unknown"),
                metric_status=getattr(r, "metric_status", {}),
                metric_errors=getattr(r, "metric_errors", {}),
                ragas_native=getattr(r, "ragas_native", None),
                ragas_version=getattr(r, "ragas_version", ""),
                error=r.error,
                latency=r.latency,
            )
            for r in results
        ]

        return BatchEvaluationResponse(
            result_count=len(eval_results),
            results=eval_results,
            average_scores=evaluator.get_average_scores(),
            export_path=export_path,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量评估失败: {str(e)}")


@router.get("/evaluation/history", response_model=EvaluationHistoryResponse)
async def get_evaluation_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
):
    """获取评估历史"""
    try:
        history_raw = evaluator.get_history(limit=limit)
        evaluations = [
            EvaluationResult(
                timestamp="",
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                answer_length=item.get("answer_length", 0),
                contexts_count=item.get("retrieved_docs_count", 0),
                metrics=EvaluationMetrics(
                    faithfulness=item.get("metrics", {}).get("faithfulness"),
                    answer_relevancy=item.get("metrics", {}).get("answer_relevancy"),
                    context_precision=item.get("metrics", {}).get("context_precision"),
                    context_recall=item.get("metrics", {}).get("context_recall"),
                    factual_correctness=item.get("metrics", {}).get("factual_correctness"),
                    answer_correctness=item.get("metrics", {}).get("answer_correctness"),
                ),
                overall_score=item.get("overall_score"),
                evaluation_status=item.get("evaluation_status", "unknown"),
                metric_status=item.get("metric_status", {}),
                metric_errors=item.get("metric_errors", {}),
                ragas_native=item.get("ragas_native"),
                ragas_version=item.get("ragas_version", ""),
                error=item.get("error"),
            )
            for item in history_raw
        ]
        return EvaluationHistoryResponse(
            evaluations=evaluations,
            average_scores=evaluator.get_average_scores(),
            total=len(evaluations),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估历史失败: {str(e)}")


@router.get("/evaluation/scores")
async def get_evaluation_scores(current_user: User = Depends(get_current_user)):
    """获取评估平均分数"""
    return evaluator.get_average_scores()


@router.get("/evaluation/visualization")
async def get_evaluation_visualization(current_user: User = Depends(get_current_user)):
    """获取评估可视化数据"""
    return evaluator.generate_visualization_data()


@router.post("/evaluation/export")
async def export_evaluation_results(
    filepath: str = "data/evaluation_results.json",
    current_user: User = Depends(get_current_user),
):
    """导出评估结果到文件"""
    try:
        path = evaluator.export_results(filepath)
        return {"message": "导出成功", "filepath": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@router.delete("/evaluation/history")
async def clear_evaluation_history(current_user: User = Depends(get_current_user)):
    """清空评估历史"""
    evaluator.clear_history()
    return {"message": "评估历史已清空"}


@router.get("/evaluation/dataset", response_model=EvalDatasetResponse)
async def get_evaluation_dataset(current_user: User = Depends(get_current_user)):
    """获取内置评测数据集"""
    try:
        from .evaluation import EvalDatasetManager

        default_path = BASE_DIR.parent / "data" / "eval_samples.json"
        if default_path.exists():
            samples = EvalDatasetManager.load_from_json(str(default_path))
        else:
            samples = []

        return EvalDatasetResponse(
            total_samples=len(samples),
            samples=[
                EvalSampleSchema(
                    question=s.question,
                    ground_truth=s.ground_truth,
                    expected_docs=s.expected_docs,
                    metadata=s.metadata,
                )
                for s in samples
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集失败: {str(e)}")


def _utcnow():
    """获取 UTC 时间字符串"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
