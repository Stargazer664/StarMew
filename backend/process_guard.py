"""One backend owner per local task database, enforced by the operating system.

The persistent file is only a lock identity. Its existence is NOT ownership;
never unlink it on shutdown, because another process could then lock a new file.
"""
import errno
import os
from pathlib import Path


class BackendAlreadyRunning(RuntimeError):
    pass


class BackendProcessLock:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self._handle = None
        self._pid = None

    @property
    def is_held(self):
        return self._handle is not None and self._pid == os.getpid()

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("后端进程锁不能重复获取")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            os.set_inheritable(handle.fileno(), False)
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                raise BackendAlreadyRunning(
                    "已有后端进程持有该任务库的操作权限。请使用 --workers 1，"
                    "并关闭重复启动的后端；不要删除锁文件。锁文件：" + str(self.path)
                ) from exc
        except BaseException:
            handle.close()
            raise
        self._handle, self._pid = handle, os.getpid()
        return self

    def __exit__(self, exc_type, exc, traceback):
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            # A forked child must not explicitly unlock its parent's ownership.
            if self._pid == os.getpid():
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._pid = None


def backend_process_lock():
    database = Path(os.getenv("UPLOAD_JOB_DB_PATH") or Path(__file__).resolve().parents[1] / "data" / "upload_jobs.sqlite3").resolve()
    return BackendProcessLock(database.with_name(database.name + ".backend.lock"))
