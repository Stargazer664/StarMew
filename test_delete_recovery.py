import copy
import json

import pytest

from backend.document_transactions import RecoveryRequiredError, assert_index_available
from test_document_transactions import make_service, ProcessCrash
from test_upload_recovery_api import upload_api


@pytest.mark.parametrize('failure', ['bm25', 'vectors', 'parents', 'commit'])
def test_delete_failure_restores_all_stores(tmp_path, monkeypatch, failure):
    service, _, original, vectors, parents, embedding = make_service(tmp_path)
    before = copy.deepcopy((vectors.rows, parents.rows, embedding.snapshot_state()))
    if failure == 'bm25':
        remove = embedding.increment_remove_documents
        def fail(texts):
            remove(texts)
            raise OSError('BM25 write acknowledgement lost')
        monkeypatch.setattr(embedding, 'increment_remove_documents', fail)
    elif failure == 'vectors':
        remove = vectors.delete
        first = True
        def fail(expr):
            nonlocal first
            remove(expr)
            if first:
                first = False
                raise OSError('delete acknowledgement lost')
        monkeypatch.setattr(vectors, 'delete', fail)
    elif failure == 'parents':
        parents.fail_replace = True
    else:
        save = service._save_manifest
        def fail(manifest):
            if manifest['phase'] == 'committed':
                raise OSError('commit marker disk full')
            save(manifest)
        monkeypatch.setattr(service, '_save_manifest', fail)
    with pytest.raises((OSError, RuntimeError)):
        service.delete('b' * 32, 'guide.md')
    assert (vectors.rows, parents.rows, embedding.snapshot_state()) == before
    assert original.read_text(encoding='utf-8') == 'old file'
    manifest = json.loads((service.root / ('b' * 32 + '.json')).read_text(encoding='utf-8'))
    assert manifest['operation'] == 'delete' and manifest['phase'] == 'rolled_back'


def test_delete_success_and_committed_recovery_preserve_original_file(tmp_path):
    service, _, original, vectors, parents, embedding = make_service(tmp_path)
    result = service.delete('b' * 32, 'guide.md')
    assert result['deleted_count'] == 1
    assert vectors.rows == parents.rows == []
    assert embedding.snapshot_state()['total_docs'] == 0
    assert original.read_text(encoding='utf-8') == 'old file'
    outcome = service.recover()['b' * 32]
    assert outcome['operation'] == 'delete' and outcome['status'] == 'completed'
    assert vectors.rows == []


def test_delete_crash_recovery_is_idempotent_and_does_not_overwrite_local_file(tmp_path, monkeypatch):
    service, _, original, vectors, parents, embedding = make_service(tmp_path)
    before = copy.deepcopy((vectors.rows, parents.rows, embedding.snapshot_state()))
    remove = vectors.delete
    def crash(expr):
        remove(expr)
        raise ProcessCrash()
    monkeypatch.setattr(vectors, 'delete', crash)
    with pytest.raises(ProcessCrash):
        service.delete('b' * 32, 'guide.md')
    monkeypatch.setattr(vectors, 'delete', remove)
    original.write_text('externally updated local file', encoding='utf-8')
    restarted = type(service)(service.root, service.upload_root, service.loader, service.writer, parents, embedding)
    assert restarted.recover()['b' * 32]['status'] == 'failed'
    restarted.recover()
    assert (vectors.rows, parents.rows, embedding.snapshot_state()) == before
    assert original.read_text(encoding='utf-8') == 'externally updated local file'


def test_delete_failed_compensation_blocks_upload_until_dependencies_recover(tmp_path, monkeypatch):
    service, staging, _, vectors, parents, embedding = make_service(tmp_path)
    before = copy.deepcopy((vectors.rows, parents.rows, embedding.snapshot_state()))
    remove = vectors.delete
    def lose_storage(expr):
        remove(expr)
        vectors.offline = True
        raise OSError('lost storage')
    monkeypatch.setattr(vectors, 'delete', lose_storage)
    with pytest.raises(RecoveryRequiredError):
        service.delete('b' * 32, 'guide.md')
    with pytest.raises(RecoveryRequiredError):
        assert_index_available(service.root)
    with pytest.raises(RecoveryRequiredError):
        service.replace('a' * 32, staging, 'guide.md')
    vectors.offline = False
    monkeypatch.setattr(vectors, 'delete', remove)
    service.recover()
    assert (vectors.rows, parents.rows, embedding.snapshot_state()) == before


def test_async_delete_failure_reports_recovery_flags_and_keeps_old_data(upload_api):
    client, _, service, vectors, original, ns = upload_api
    service.parents.fail_replace = True
    result = client.delete('/api/documents/delete/async/guide.md?delete_id=' + 'b' * 32)
    assert result.status_code == 200
    job = ns['delete_job_manager'].get_job(result.json()['job_id'])
    assert job['status'] == 'failed' and job['retryable']
    assert not job['recovery_required']
    assert [row['text'] for row in vectors.rows] == ['old leaf']
    assert original.read_text(encoding='utf-8') == 'old file'


def test_delete_id_prevents_repeated_deletion_after_new_upload(upload_api):
    client, _, service, vectors, _, ns = upload_api
    url = '/api/documents/delete/async/guide.md?delete_id=' + 'b' * 32
    first = client.delete(url)
    assert first.json()['job_id'] == 'b' * 32
    assert ns['delete_job_manager'].get_job('b' * 32)['status'] == 'completed'
    vectors.rows.append({'filename': 'guide.md', 'text': 'newer version'})
    assert client.delete(url).json()['job_id'] == first.json()['job_id']
    assert vectors.rows[0]['text'] == 'newer version'
    assert client.delete('/api/documents/delete/async/other.md?delete_id=' + 'b' * 32).status_code == 409


def test_recovery_updates_delete_namespace_and_exposes_jobs_when_index_unavailable(upload_api, monkeypatch):
    client, uploads, service, vectors, _, ns = upload_api
    remove = vectors.delete
    def lose_storage(expr):
        remove(expr)
        vectors.offline = True
        raise OSError('lost storage')
    monkeypatch.setattr(vectors, 'delete', lose_storage)
    start = client.delete('/api/documents/delete/async/guide.md')
    job_id = start.json()['job_id']
    status = client.get('/api/documents/delete/jobs/' + job_id)
    assert status.status_code == 200
    assert status.json()['recovery_required']
    assert client.get('/api/documents/delete/jobs').json()[0]['job_id'] == job_id
    assert client.get('/api/documents').status_code == 503
    vectors.offline = False
    monkeypatch.setattr(vectors, 'delete', remove)
    assert client.post('/api/documents/recover').status_code == 200
    job = ns['delete_job_manager'].get_job(job_id)
    assert not job['recovery_required'] and job['retryable']
    assert uploads.list_jobs() == []
    assert [row['text'] for row in vectors.rows] == ['old leaf']


def test_sync_delete_uses_same_compensation(upload_api):
    client, _, service, vectors, original, _ = upload_api
    service.parents.fail_replace = True
    assert client.delete('/api/documents/guide.md').status_code == 500
    assert [row['text'] for row in vectors.rows] == ['old leaf']
    assert original.read_text(encoding='utf-8') == 'old file'


def test_restart_finalizes_committed_delete_in_correct_task_store(upload_api):
    _, uploads, service, vectors, _, ns = upload_api
    jobs = ns['delete_job_manager']
    from backend.upload_jobs import DELETE_STEPS
    job_id = jobs.create_job('guide.md', steps=DELETE_STEPS, current_step='prepare')['job_id']
    service.delete(job_id, 'guide.md')
    ns['recover_upload_operations'](mark_interrupted=True)
    assert jobs.get_job(job_id)['status'] == 'completed'
    assert jobs.get_job(job_id)['total_chunks'] == 1
    assert uploads.list_jobs() == [] and vectors.rows == []


def test_committed_delete_status_write_failure_can_be_manually_recovered(upload_api, monkeypatch):
    client, _, service, vectors, _, ns = upload_api
    jobs = ns['delete_job_manager']
    complete = jobs.complete_job
    monkeypatch.setattr(jobs, 'complete_job', lambda *a: (_ for _ in ()).throw(OSError('task DB transient failure')))
    response = client.delete('/api/documents/delete/async/guide.md')
    job_id = response.json()['job_id']
    assert vectors.rows == []
    assert jobs.get_job(job_id)['status'] == 'running'
    monkeypatch.setattr(jobs, 'complete_job', complete)
    assert client.post('/api/documents/recover').status_code == 200
    assert jobs.get_job(job_id)['status'] == 'completed'
    assert not list(service.root.glob('*.json'))
