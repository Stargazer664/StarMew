import threading

import pytest

from backend.milvus_client import MilvusManager


class FakeClient:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def manager_with_replacement_client():
    manager = MilvusManager.__new__(MilvusManager)
    manager._client_lock = threading.RLock()
    first = FakeClient("first")
    replacement = FakeClient("replacement")
    manager.client = first

    def get_client():
        if manager.client is None:
            manager.client = replacement
        return manager.client

    manager._get_client = get_client
    return manager, first, replacement


def test_milvus_exception_closed_channel_resets_client_and_retries_once():
    manager, first, replacement = manager_with_replacement_client()
    calls = []

    def operation(client):
        calls.append(client.name)
        if client is first:
            raise RuntimeError(
                "<MilvusException: (code=1, message=Unexpected error, "
                "message=<Cannot invoke RPC on closed channel!>)>"
            )
        return "reconnected"

    assert manager._run_with_reconnect(operation) == "reconnected"
    assert calls == ["first", "replacement"]
    assert first.closed is True
    assert manager.client is replacement


def test_non_closed_channel_error_is_not_retried():
    manager, first, _ = manager_with_replacement_client()
    calls = []

    def operation(client):
        calls.append(client.name)
        raise RuntimeError("Milvus collection does not exist")

    with pytest.raises(RuntimeError, match="collection does not exist"):
        manager._run_with_reconnect(operation)

    assert calls == ["first"]
    assert first.closed is False
