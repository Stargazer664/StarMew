"""Cancellation ownership for one admitted run; thread work remains owned until exit."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, Lock


class RunCancelled(Exception):
    """A deliberate stop, never a retryable tool/provider failure."""


class RunControl:
    def __init__(self, identity, session_factory):
        self.identity = identity
        self.session_factory = session_factory
        self.cancelled = Event()
        self.task = self.loop = None

    def request_cancel(self):
        with _lock:
            first = not self.cancelled.is_set()
            self.cancelled.set()
            task, loop = self.task, self.loop
        if first and task is not None and loop is not None:
            def cancel_once():
                if not task.done():
                    task.cancel()
            loop.call_soon_threadsafe(cancel_once)

    def bind_task(self):
        with _lock:
            self.task, self.loop = asyncio.current_task(), asyncio.get_running_loop()


_lock = Lock()
_controls = {}
_current = ContextVar('chat_run_control', default=None)


def _key(identity):
    return (identity.session_instance_id, identity.session_ref_id, identity.turn_id, identity.request_id)


@contextmanager
def own_run(identity, *, session_factory, bind_context=True):
    control = RunControl(identity, session_factory)
    with _lock:
        _controls[_key(identity)] = control
    token = _current.set(control) if bind_context else None
    try:
        yield control
    finally:
        if token is not None:
            _current.reset(token)
        with _lock:
            if _controls.get(_key(identity)) is control:
                _controls.pop(_key(identity), None)


def notify_cancel(identity):
    with _lock:
        control = _controls.get(_key(identity))
    if control is not None:
        control.request_cancel()


def current_control():
    return _current.get()


def bind_control(control):
    """Bind only in an isolated async-generator context, never across caller yields."""
    _current.set(control)


def check_cancelled():
    control = current_control()
    if control is not None and control.cancelled.is_set():
        raise RunCancelled('REQUEST_CANCELLED')


def mark_cancelled():
    """Transport cancellation also stops subsequent work in the owned thread."""
    control = current_control()
    if control is not None:
        control.cancelled.set()
