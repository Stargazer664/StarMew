"""Document recovery attribution and telemetry boundaries, without live services."""
import json
import sqlite3

import pytest

from backend import upload_jobs
from backend.document_transactions import RecoveryRequiredError
from test_document_transactions import make_service
from test_upload_recovery_api import upload_api


@pytest.fixture
def diagnostics(tmp_path):
    from backend import telemetry
    telemetry.start_diagnostics(tmp_path / 'diagnostics.sqlite3', tmp_path / 'logs')
    try:
        yield telemetry
    finally:
        telemetry.stop_diagnostics()


def test_recovery_failure_keeps_completed_results_and_identifies_unprocessed_jobs(tmp_path):
    service, staging, _, vectors, _, _ = make_service(tmp_path)
    service.replace('a' * 32, staging, 'guide.md')
    manifest = json.loads((service.root / ('a' * 32 + '.json')).read_text(encoding='utf-8'))
    for job_id in ('b' * 32, 'c' * 32):
        pending = dict(manifest, job_id=job_id, phase='applying')
        service._save_manifest(pending)
    vectors.offline = True
    with pytest.raises(RecoveryRequiredError) as failure:
        service.recover()
    assert getattr(failure.value, 'failed_job_id', None) == 'b' * 32
    assert failure.value.outcomes['a' * 32]['status'] == 'completed'
    assert failure.value.blocked_job_ids == ['c' * 32]
    last = json.loads((service.root / ('c' * 32 + '.json')).read_text(encoding='utf-8'))
    assert last['phase'] == 'applying'


def test_attempt_snapshots_publish_after_sqlite_write_transaction_releases(tmp_path, monkeypatch):
    path = tmp_path / 'jobs.sqlite3'
    recorded = []
    def capture(job, operation):
        with sqlite3.connect(path, timeout=0) as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute('SELECT payload FROM document_jobs WHERE job_id=?', (job['job_id'],)).fetchone()
        assert json.loads(row[0])['attempt'] == job['attempt']
        recorded.append((job, operation))
    monkeypatch.setattr(upload_jobs, 'record_job', capture, raising=False)
    manager = upload_jobs.UploadJobManager(path)
    job = manager.create_job('private.md', job_id='a' * 32)
    manager.fail_job(job['job_id'], 'parse', 'private error')
    manager.update_job(job['job_id'], retryable=True)
    manager.retry_job(job['job_id'])
    assert [item[0]['attempt'] for item in recorded] == [1, 1, 1, 2]
    assert all(operation == 'upload' for _, operation in recorded)


def test_diagnostic_snapshot_failure_never_breaks_committed_job(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OSError('diagnostics unavailable')
    monkeypatch.setattr(upload_jobs, 'record_job', unavailable, raising=False)
    manager = upload_jobs.UploadJobManager(tmp_path / 'jobs.sqlite3')
    job = manager.create_job('guide.md')
    manager.complete_job(job['job_id'])
    assert manager.get_job(job['job_id'])['status'] == 'completed'


def test_recovery_batch_reconciles_prior_commit_and_does_not_copy_failure_to_blocked(upload_api):
    _, jobs, service, vectors, _, namespace = upload_api
    for job_id in ('a' * 32, 'b' * 32, 'c' * 32):
        jobs.create_job('guide.md', job_id=job_id)
    staging = service.upload_root / '.staging' / ('a' * 32) / 'guide.md'
    service.replace('a' * 32, staging, 'guide.md')
    manifest = json.loads((service.root / ('a' * 32 + '.json')).read_text(encoding='utf-8'))
    for job_id in ('b' * 32, 'c' * 32):
        service._save_manifest(dict(manifest, job_id=job_id, phase='applying', file_existed=False))
    vectors.offline = True
    with pytest.raises(RecoveryRequiredError):
        namespace['recover_upload_operations']()
    assert jobs.get_job('a' * 32)['status'] == 'completed'
    assert not (service.root / ('a' * 32 + '.json')).exists()
    failed, blocked = jobs.get_job('b' * 32), jobs.get_job('c' * 32)
    assert failed['recovery_required'] and blocked['recovery_required']
    assert 'storage offline' in failed['error']
    assert 'storage offline' not in blocked['error']
    assert '阻塞' in blocked['error']


def test_primary_failure_and_compensation_error_are_separate_and_safe(tmp_path, monkeypatch, diagnostics):
    service, staging, _, vectors, parents, _ = make_service(tmp_path)
    vectors.fail_insert = True
    original_replace = parents.replace_documents
    def parent_restore_fails(filename, rows):
        if rows[0]['text'] == 'old parent':
            raise ConnectionError('SECRET PASSWORD file:///private')
        return original_replace(filename, rows)
    monkeypatch.setattr(parents, 'replace_documents', parent_restore_fails)
    with diagnostics.trace('document', 'upload', job_id='a' * 32, attempt=1):
        with pytest.raises(RecoveryRequiredError):
            service.replace('a' * 32, staging, 'guide.md')
    diagnostics.flush()
    events = diagnostics.get_store().events(job_id='a' * 32, limit=100)['items']
    failure_codes = {item.get('code') for item in events}
    assert 'VECTOR_WRITE_FAILED' in failure_codes
    assert 'PARENT_RESTORE_FAILED' in failure_codes
    assert 'ROLLBACK_FAILED' in failure_codes
    assert 'SECRET' not in json.dumps(events)
    manifest = json.loads((service.root / ('a' * 32 + '.json')).read_text(encoding='utf-8'))
    assert manifest['error'] == 'insert acknowledgement lost'
    assert manifest['cause_event_id']


def test_attempt_history_retains_failed_and_successful_executions(upload_api, diagnostics):
    from test_upload_recovery_api import submit
    client, jobs, _, vectors, _, _ = upload_api
    vectors.fail_insert = True
    assert submit(client).status_code == 200
    assert client.post('/api/documents/upload/jobs/' + 'a' * 32 + '/retry').status_code == 200
    diagnostics.flush()
    history = diagnostics.get_store().document_attempts('upload', 'a' * 32)
    by_attempt = {item['attempt']: item for item in history}
    assert by_attempt[1]['status'] == 'failed'
    assert by_attempt[2]['status'] == 'completed'
    assert by_attempt[1]['duration_ms'] >= 0
    assert by_attempt[2]['queue_wait_ms'] >= 0
    assert any(item['stage'] == 'vector_insert' and item['status'] == 'failed' for item in by_attempt[1]['spans'])
    assert all('private' not in json.dumps(item) for item in history)
    assert jobs.get_job('a' * 32)['status'] == 'completed'


def test_recovery_saves_each_success_before_attempting_next_journal(upload_api, monkeypatch):
    _, jobs, service, _, _, namespace = upload_api
    for job_id in ('a' * 32, 'b' * 32):
        jobs.create_job('guide.md', job_id=job_id)
    staging = service.upload_root / '.staging' / ('a' * 32) / 'guide.md'
    service.replace('a' * 32, staging, 'guide.md')
    manifest = json.loads((service.root / ('a' * 32 + '.json')).read_text(encoding='utf-8'))
    service._save_manifest(dict(manifest, job_id='b' * 32, phase='applying', file_existed=False))
    real_rollback = service._rollback
    def check_prior_commit(manifest):
        assert jobs.get_job('a' * 32)['status'] == 'completed'
        return real_rollback(manifest)
    monkeypatch.setattr(service, '_rollback', check_prior_commit)
    assert namespace['recover_upload_operations']() == 2


def test_committed_recovery_status_failure_does_not_mark_data_failed(upload_api, monkeypatch):
    _, jobs, service, vectors, _, namespace = upload_api
    job_id = 'a' * 32
    jobs.create_job('guide.md', job_id=job_id)
    staging = service.upload_root / '.staging' / job_id / 'guide.md'
    service.replace(job_id, staging, 'guide.md')
    def unavailable(*args):
        raise OSError('metadata storage unavailable')
    monkeypatch.setattr(jobs, 'complete_job', unavailable)
    with pytest.raises(RecoveryRequiredError):
        namespace['recover_upload_operations']()
    assert jobs.get_job(job_id)['status'] == 'pending'
    assert not jobs.get_job(job_id)['recovery_required']
    assert vectors.rows[0]['text'] == 'new leaf'
    assert json.loads((service.root / (job_id + '.json')).read_text(encoding='utf-8'))['phase'] == 'committed'


def test_job_blocked_before_own_mutation_has_blocked_trace(upload_api, diagnostics):
    from test_upload_recovery_api import submit
    client, jobs, service, vectors, _, _ = upload_api
    staging = service.upload_root / '.staging' / ('a' * 32) / 'guide.md'
    jobs.create_job('guide.md', job_id='a' * 32)
    service.replace('a' * 32, staging, 'guide.md')
    manifest = json.loads((service.root / ('a' * 32 + '.json')).read_text(encoding='utf-8'))
    service._save_manifest(dict(manifest, phase='applying'))
    vectors.offline = True
    assert submit(client, upload_id='b' * 32).status_code == 200
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('doc:upload:' + 'b' * 32 + ':1')
    assert detail['trace']['status'] == 'blocked'
    assert any(item['stage'] == 'processing' and item['status'] == 'blocked' for item in detail['spans'])
    assert any(item.get('blocked_by_job_id') == 'a' * 32 for item in detail['events'])


@pytest.mark.parametrize('documents', [[], [{'chunk_level': 2, 'text': 'parent only'}]])
def test_invalid_parser_output_marks_parse_failed(tmp_path, diagnostics, documents):
    service, staging, _, _, _, _ = make_service(tmp_path)
    service.loader.load_document = lambda *args: documents
    with diagnostics.trace('document', 'upload', job_id='a' * 32, attempt=1) as execution:
        with pytest.raises(ValueError):
            service.replace('a' * 32, staging, 'guide.md')
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail(execution.id)
    assert next(item for item in detail['spans'] if item['stage'] == 'parse')['status'] == 'failed'


def test_final_diagnostic_status_read_cannot_break_worker_result():
    from backend.document_diagnostics import document_execution
    class Manager:
        calls = 0
        def get_job(self, job_id):
            self.calls += 1
            if self.calls > 1:
                raise OSError('optional final status read failed')
            return {'status': 'pending', 'attempt': 1}
    namespace = {'upload_job_manager': Manager()}
    exec('def worker(job_id):\n    return 42\n', namespace)
    worker = document_execution('upload')(namespace['worker'])
    assert worker('a' * 32) == 42


@pytest.mark.parametrize('startup', [False, True])
def test_recovery_status_failure_continues_other_committed_jobs(upload_api, monkeypatch, startup):
    _, jobs, service, _, _, namespace = upload_api
    for job_id in ('a' * 32, 'b' * 32):
        jobs.create_job('guide.md', job_id=job_id)
    staging = service.upload_root / '.staging' / ('a' * 32) / 'guide.md'
    service.replace('a' * 32, staging, 'guide.md')
    manifest = json.loads((service.root / ('a' * 32 + '.json')).read_text(encoding='utf-8'))
    service._save_manifest(dict(manifest, job_id='b' * 32))
    complete = jobs.complete_job
    def fail_one(job_id, *args):
        if job_id == 'a' * 32:
            raise OSError('one metadata failure')
        return complete(job_id, *args)
    monkeypatch.setattr(jobs, 'complete_job', fail_one)
    with pytest.raises(RecoveryRequiredError) as failure:
        namespace['recover_upload_operations'](mark_interrupted=startup)
    assert jobs.get_job('b' * 32)['status'] == 'completed'
    assert not jobs.get_job('b' * 32)['recovery_required']
    assert jobs.get_job('a' * 32)['status'] == 'pending'
    assert not jobs.get_job('a' * 32)['recovery_required']
    assert failure.value.failed_job_id is None
    assert failure.value.blocked_job_ids == []
    assert failure.value.metadata_job_ids == ['a' * 32]
    assert (service.root / ('a' * 32 + '.json')).exists()
    assert not (service.root / ('b' * 32 + '.json')).exists()


def test_backup_cleanup_failure_preserves_journal_for_retry(tmp_path, monkeypatch):
    from pathlib import Path
    service, staging, _, _, _, _ = make_service(tmp_path)
    service.replace('a' * 32, staging, 'guide.md')
    unlink = Path.unlink
    def fail_backup(path, *args, **kwargs):
        if path.suffix == '.original':
            raise OSError('backup cleanup temporarily unavailable')
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', fail_backup)
        with pytest.raises(OSError):
            service.acknowledge('a' * 32)
    assert (service.root / ('a' * 32 + '.json')).exists()
    service.acknowledge('a' * 32)
    assert not list(service.root.glob('*'))


def test_staging_cleanup_is_successful_when_already_removed(upload_api, diagnostics):
    from test_upload_recovery_api import submit
    client, jobs, _, _, _, namespace = upload_api
    assert submit(client).status_code == 200
    with diagnostics.bind_context(scope='document', job_id='a' * 32, operation='upload', attempt=1):
        namespace['_cleanup_upload_stage'](jobs.get_job('a' * 32))
    diagnostics.flush()
    events = diagnostics.get_store().events(job_id='a' * 32, limit=100)['items']
    assert not any(item['name'] == 'cleanup_failed' for item in events)
