"""Upload recovery tests isolate every persistent store in a temporary directory."""
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parent / "backend" / "upload_jobs.py"


@pytest.fixture
def upload_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_JOB_DB_PATH", str(tmp_path / "global-jobs.sqlite3"))
    spec = importlib.util.spec_from_file_location("backend.isolated_upload_jobs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_defers_database_creation(upload_jobs, tmp_path):
    assert not (tmp_path / "global-jobs.sqlite3").exists()
    assert upload_jobs.upload_job_manager.list_jobs() == []
    assert upload_jobs.delete_job_manager.list_jobs() == []
    assert not (tmp_path / "global-jobs.sqlite3").exists()


def test_jobs_and_staging_metadata_survive_manager_recreation(upload_jobs, tmp_path):
    database = tmp_path / "nested" / "jobs.sqlite3"
    manager = upload_jobs.UploadJobManager(store_path=database)
    job = manager.create_job("文档.md")
    source = tmp_path / "文档.md"
    source.write_text("complete document", encoding="utf-8")
    manager.update_job(job["job_id"], staging_path=str(source), retryable=True)
    manager.complete_step(job["job_id"], "upload", "saved")
    manager.update_step(job["job_id"], "parse", 25, total_chunks=8)

    restored = upload_jobs.UploadJobManager(store_path=database)
    actual = restored.get_job(job["job_id"])
    assert actual["status"] == "running"
    assert actual["current_step"] == "parse"
    assert actual["total_chunks"] == 8
    assert actual["staging_path"] == str(source)
    assert actual["retryable"] is True
    assert actual["attempt"] == 1
    assert actual["recovery_required"] is False
    assert restored.list_jobs() == [actual]


def test_global_managers_share_configured_database_with_isolated_namespaces(upload_jobs, tmp_path):
    job_id = "a" * 32
    uploaded = upload_jobs.upload_job_manager.create_job("upload.md", job_id=job_id)
    deleted = upload_jobs.delete_job_manager.create_job(
        "delete.md", job_id=job_id, steps=upload_jobs.DELETE_STEPS,
        current_step="prepare", completion_step="parent_store",
    )
    assert (tmp_path / "global-jobs.sqlite3").is_file()
    assert upload_jobs.upload_job_manager.get_job(job_id) == uploaded
    assert upload_jobs.delete_job_manager.get_job(job_id) == deleted


def test_existing_id_is_idempotent_and_rejects_a_different_filename(upload_jobs, tmp_path):
    database = tmp_path / "jobs.sqlite3"
    first = upload_jobs.UploadJobManager(database)
    second = upload_jobs.UploadJobManager(database)
    job_id = "b" * 32
    first.create_job("guide.md", job_id=job_id)
    expected = first.complete_step(job_id, "upload", "already saved")

    assert second.create_job("guide.md", job_id=job_id) == expected
    assert len(second.list_jobs()) == 1
    with pytest.raises(ValueError):
        second.create_job("another.md", job_id=job_id)
    assert first.get_job(job_id) == expected


@pytest.mark.parametrize("job_id", ["", "../job", "a" * 31, "a" * 33, "g" * 32, 42])
def test_invalid_explicit_job_ids_are_rejected(upload_jobs, job_id):
    manager = upload_jobs.UploadJobManager()
    with pytest.raises(ValueError):
        manager.create_job("guide.md", job_id=job_id)
    assert manager.list_jobs() == []


@pytest.mark.parametrize("persistent", [False, True])
def test_update_cannot_reassign_job_identity(upload_jobs, tmp_path, persistent):
    manager = upload_jobs.UploadJobManager(tmp_path / "jobs.sqlite3" if persistent else None)
    job = manager.create_job("guide.md")
    for fields in ({"filename": "other.md"}, {"created_at": "yesterday"}):
        with pytest.raises(ValueError):
            manager.update_job(job["job_id"], **fields)
        assert manager.get_job(job["job_id"]) == job


def test_distinct_manager_updates_preserve_other_latest_fields(upload_jobs, tmp_path):
    database = tmp_path / "jobs.sqlite3"
    first = upload_jobs.UploadJobManager(database)
    second = upload_jobs.UploadJobManager(database)
    job = first.create_job("guide.md")
    second.get_job(job["job_id"])
    first.update_step(job["job_id"], "parse", 50, total_chunks=12)
    second.update_job(job["job_id"], retryable=True, staging_path="retained.md")
    first.update_step(job["job_id"], "vector_store", 25, processed_chunks=3)

    actual = second.get_job(job["job_id"])
    assert actual["total_chunks"] == 12
    assert actual["processed_chunks"] == 3
    assert actual["retryable"] is True
    assert actual["staging_path"] == "retained.md"
    assert next(step for step in actual["steps"] if step["key"] == "parse")["percent"] == 50


@pytest.mark.parametrize("persistent", [False, True])
def test_invalid_step_update_rolls_back_all_mutations(upload_jobs, tmp_path, persistent):
    manager = upload_jobs.UploadJobManager(tmp_path / "jobs.sqlite3" if persistent else None)
    before = manager.create_job("guide.md")
    with pytest.raises(ValueError):
        manager.update_step(before["job_id"], "parse", 75, total_chunks="invalid")
    assert manager.get_job(before["job_id"]) == before
    updated = manager.complete_step(before["job_id"], "upload")
    assert updated["status"] == "running"


def test_failed_serialization_does_not_leave_partial_data_or_locked_store(upload_jobs, tmp_path):
    manager = upload_jobs.UploadJobManager(tmp_path / "jobs.sqlite3")
    before = manager.create_job("guide.md")
    with pytest.raises(TypeError):
        manager.update_job(before["job_id"], retryable=True, message=object())
    assert manager.get_job(before["job_id"]) == before
    assert manager.update_job(before["job_id"], retryable=True)["retryable"] is True


def test_only_one_concurrent_retry_can_claim_a_failed_job(upload_jobs, tmp_path):
    database = tmp_path / "jobs.sqlite3"
    first = upload_jobs.UploadJobManager(database)
    second = upload_jobs.UploadJobManager(database)
    job_id = first.create_job("guide.md")["job_id"]
    first.complete_step(job_id, "upload", "saved")
    first.update_step(job_id, "parse", 50, total_chunks=7, processed_chunks=2)
    first.fail_job(job_id, "parse", "interrupted")
    first.update_job(job_id, retryable=True, staging_path="retained.md")
    barrier = threading.Barrier(2)

    def retry(manager):
        barrier.wait()
        try:
            return manager.retry_job(job_id)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(retry, [first, second]))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    retried = first.get_job(job_id)
    assert retried == winners[0]
    assert retried["attempt"] == 2
    assert retried["status"] == "pending"
    assert retried["current_step"] == "parse"
    assert retried["error"] is None
    assert retried["retryable"] is False
    assert retried["total_chunks"] == retried["processed_chunks"] == 0
    assert retried["staging_path"] == "retained.md"
    assert retried["steps"][0]["status"] == "completed"
    assert all(step["status"] == "pending" and step["percent"] == 0 for step in retried["steps"][1:])


@pytest.mark.parametrize("status,retryable,recovery_required", [
    ("pending", True, False), ("running", True, False),
    ("completed", True, False), ("failed", False, False), ("failed", True, True),
])
def test_retry_rejects_ineligible_states_without_changing_them(
    upload_jobs, tmp_path, status, retryable, recovery_required,
):
    manager = upload_jobs.UploadJobManager(tmp_path / "jobs.sqlite3")
    job_id = manager.create_job("guide.md")["job_id"]
    before = manager.update_job(
        job_id, status=status, retryable=retryable, recovery_required=recovery_required,
    )
    with pytest.raises(ValueError):
        manager.retry_job(job_id)
    assert manager.get_job(job_id) == before
    assert manager.retry_job("f" * 32) is None


def test_startup_marks_only_unfinished_jobs_and_requires_complete_staging_file(upload_jobs, tmp_path):
    database = tmp_path / "jobs.sqlite3"
    manager = upload_jobs.UploadJobManager(database)
    source = tmp_path / "received.md"
    source.write_text("full document", encoding="utf-8")
    ready = manager.create_job("ready.md")["job_id"]
    partial = manager.create_job("partial.md")["job_id"]
    missing = manager.create_job("missing.md")["job_id"]
    blocked = manager.create_job("blocked.md")["job_id"]
    directory = manager.create_job("directory.md")["job_id"]
    completed = manager.create_job("done.md")["job_id"]
    failed = manager.create_job("failed.md")["job_id"]
    for job_id in [ready, partial, blocked]:
        manager.update_job(job_id, staging_path=str(source))
    manager.update_job(missing, staging_path=str(tmp_path / "missing.md"))
    manager.update_job(directory, staging_path=str(tmp_path))
    for job_id in [ready, missing, blocked, directory]:
        manager.complete_step(job_id, "upload")
    manager.update_job(blocked, recovery_required=True)
    completed_before = manager.complete_job(completed)
    failed_before = manager.fail_job(failed, "parse", "existing failure")

    restarted = upload_jobs.UploadJobManager(database)
    assert restarted.get_job(ready)["status"] == "running"
    interrupted = restarted.mark_interrupted()
    assert {job["job_id"] for job in interrupted} == {ready, partial, missing, blocked, directory}
    assert restarted.get_job(ready)["retryable"] is True
    for job_id in [partial, missing, blocked, directory]:
        assert restarted.get_job(job_id)["retryable"] is False
    for job in interrupted:
        assert job["status"] == "failed"
        assert job["error"]
    assert restarted.get_job(blocked)["recovery_required"] is True
    assert restarted.get_job(completed) == completed_before
    assert restarted.get_job(failed) == failed_before
    assert restarted.mark_interrupted() == []
    assert manager.get_job(ready)["status"] == "failed"


def test_memory_managers_are_independent_and_return_defensive_copies(upload_jobs):
    manager = upload_jobs.UploadJobManager()
    job = manager.create_job("guide.md")
    job["steps"][0]["status"] = "failed"
    listed = manager.list_jobs()
    listed[0]["filename"] = "corrupted.md"
    assert manager.get_job(job["job_id"])["steps"][0]["status"] == "pending"
    assert manager.get_job(job["job_id"])["filename"] == "guide.md"
    assert upload_jobs.UploadJobManager().list_jobs() == []


def test_index_mutation_gate_allows_nested_transactions(upload_jobs):
    finished = threading.Event()

    def nested():
        upload_jobs.run_serialized_index_mutation(
            lambda: upload_jobs.run_serialized_index_mutation(finished.set)
        )

    worker = threading.Thread(target=nested, daemon=True)
    worker.start()
    assert finished.wait(2), "nested mutation gate deadlocked"
    worker.join(timeout=2)
