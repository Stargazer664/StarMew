from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
import logging
import os
import sys

# 必须在导入其他模块前禁用 LangSmith tracing，避免网络阻塞
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

print("[app] loading...", file=sys.stderr, flush=True)

# 逐步导入以定位阻塞点
print("[app] importing api...", file=sys.stderr, flush=True)
from . import api as api_module
from .process_guard import backend_process_lock
from .database import init_db
from .memory_jobs import MemoryWorker
from .chat_diagnostics import run_blocking
from .conversation_turns import recover_interrupted_turns
from .upload_jobs import resume_index_operations, stop_index_operations
from .telemetry import start_diagnostics, stop_diagnostics
from .diagnostics_api import router as diagnostics_router, RequestCorrelationMiddleware
print("[app] api loaded", file=sys.stderr, flush=True)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

print("[app] creating FastAPI", file=sys.stderr)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    # Acquisition failure must escape startup, before touching recovery or jobs.
    with backend_process_lock() as owner:
        # Initialize the new table before recovery. Failure must fail startup,
        # and another process must never mark an active owner's turn failed.
        await run_blocking(init_db)
        await run_blocking(recover_interrupted_turns)
        resume_index_operations()
        memory_worker = MemoryWorker()
        try:
            await run_blocking(start_diagnostics)
            await run_blocking(memory_worker.start)
            recovery = asyncio.create_task(asyncio.to_thread(api_module.recover_upload_operations, mark_interrupted=True))
            cancelled = False
            while not recovery.done():
                try:
                    await asyncio.shield(recovery)
                except asyncio.CancelledError:
                    # Cancelling to_thread cannot stop its mutations. Retain ownership
                    # until it finishes, even if startup is cancelled repeatedly.
                    cancelled = True
                except Exception:
                    break
            if cancelled:
                recovery.exception()
                raise asyncio.CancelledError
            try:
                recovery.result()
            except Exception:
                # Keep the admin recovery UI reachable. Index access remains guarded.
                logger.exception("Document recovery incomplete; index access remains guarded")
            app.state.backend_owner = owner
            yield
        finally:
            app.state.backend_owner = None
            # Drain the actual sync worker, not its cancellable asyncio wrapper.
            # Each cleanup must finish even if shutdown is cancelled repeatedly.
            cancelled_cleanup = False
            for cleanup in (memory_worker.aclose,
                            lambda: run_blocking(stop_index_operations),
                            lambda: run_blocking(stop_diagnostics)):
                try:
                    await cleanup()
                except asyncio.CancelledError:
                    cancelled_cleanup = True
            if cancelled_cleanup:
                raise asyncio.CancelledError


app = FastAPI(title="Cute Cat Bot API", lifespan=lifespan)

print("[app] adding middleware", file=sys.stderr)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Trace-ID"],
)
app.add_middleware(RequestCorrelationMiddleware)

app.include_router(api_module.router, prefix="/api")
app.include_router(diagnostics_router, prefix="/api")
print("[app] router included (prefix=/api)", file=sys.stderr)

# Frontend static files — mounted at root level AFTER the API routes
# API requests under /api/* will be matched by the router first
try:
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    if FRONTEND_DIR.exists():

        @app.exception_handler(StarletteHTTPException)
        async def _spa_fallback(request, exc):
            """非 /api 路径的 404 返回前端 index.html（HTML 入口）"""
            if not request.url.path.startswith("/api") and exc.status_code == 404:
                index_path = FRONTEND_DIR / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

        # 挂载静态资源（HTML/CSS/JS）
        # 由于 router 已优先处理 /api/*，根路径 fallback 不会影响 API
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
        print("[app] frontend mounted (with /api router priority)", file=sys.stderr)
except Exception as e:
    print(f"[app] frontend mount skipped: {e}", file=sys.stderr)

# Requests require the lifecycle to have initialized the database under ownership.
_db_initialized = False

@app.middleware("http")
async def _init_db_on_first_request(request, call_next):
    global _db_initialized
    owner = getattr(request.app.state, "backend_owner", None)
    if owner is None or not owner.is_held:
        return JSONResponse(status_code=503, content={"detail": "后端未取得独占运行权限，请启用 lifespan 并使用 --workers 1"})
    return await call_next(request)

print("[app] ready", file=sys.stderr)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
