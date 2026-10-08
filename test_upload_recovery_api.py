import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import json
import logging
import re
import os
from time import perf_counter_ns
from backend.document_diagnostics import document_execution, document_operation, recovery_execution, mark_document_blocked
from backend.telemetry import bind_context, event

import pytest
from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.testclient import TestClient
from typing import Optional

from backend.document_loader import DocumentLoader
from backend.schemas import (DocumentUploadStartResponse, DocumentUploadJobResponse, DocumentUploadResponse,
                             DocumentInfo, DocumentListResponse, DocumentDeleteResponse, DocumentDeleteStartResponse,
                             DocumentDeleteJobResponse)
from backend.upload_jobs import UploadJobManager, create_upload_staging_path, run_serialized_index_mutation, DELETE_STEPS
from test_document_transactions import make_service, ProcessCrash


@pytest.fixture
def upload_api(tmp_path):
    # Execute the real upload route bodies without importing unrelated live LLM/evaluation services.
    service, _, original, vectors, parents, embedding = make_service(tmp_path)
    jobs = UploadJobManager(tmp_path / 'jobs.sqlite3')
    from backend.document_transactions import RecoveryRequiredError, assert_index_available
    namespace = dict(asyncio=asyncio, Path=Path, uuid4=uuid4, json=json, logging=logging, re=re, os=os,
                     document_execution=document_execution, document_operation=document_operation,
                     mark_document_blocked=mark_document_blocked,
                     recovery_execution=recovery_execution, bind_context=bind_context, event=event, perf_counter_ns=perf_counter_ns,
                     APIRouter=APIRouter, BackgroundTasks=BackgroundTasks, Depends=Depends, File=File,
                     Form=Form, HTTPException=HTTPException, UploadFile=UploadFile, OptionalType=Optional,
                     User=object, require_admin=lambda: object(), router=APIRouter(),
                     DocumentLoader=DocumentLoader, UPLOAD_DIR=service.upload_root,
                     DocumentUploadStartResponse=DocumentUploadStartResponse, DocumentUploadJobResponse=DocumentUploadJobResponse,
                     DocumentUploadResponse=DocumentUploadResponse, upload_job_manager=jobs,
                     DocumentInfo=DocumentInfo, DocumentListResponse=DocumentListResponse,
                     DocumentDeleteResponse=DocumentDeleteResponse, DocumentDeleteStartResponse=DocumentDeleteStartResponse,
                     DocumentDeleteJobResponse=DocumentDeleteJobResponse,
                     DELETE_STEPS=DELETE_STEPS, assert_index_available=lambda: assert_index_available(service.root),
                     delete_job_manager=UploadJobManager(tmp_path / 'jobs.sqlite3', namespace='delete'), create_upload_staging_path=create_upload_staging_path,
                     run_serialized_index_mutation=run_serialized_index_mutation,
                     _get_document_transactions=lambda: service, RecoveryRequiredError=RecoveryRequiredError,
                     logger=logging.getLogger('test.upload'), _get_loader=lambda: service.loader,
                     _get_milvus_manager=lambda: vectors, _get_milvus_writer=lambda: service.writer,
                     _get_parent_chunk_store=lambda: parents, embedding_service=embedding)
    names = {'_save_upload_file', '_cleanup_upload_stage', '_process_upload_job', 'recover_upload_operations',
             'upload_document_async', 'get_upload_job', 'list_upload_jobs', 'retry_upload_job',
             'recover_upload_jobs', 'upload_document', '_filename_filter_expr', '_remove_bm25_stats_for_filename',
             'list_documents', 'delete_document', 'delete_document_async', '_process_delete_job',
             '_transaction_job_manager', 'get_delete_job', 'list_delete_jobs'}
    tree = ast.parse((Path(__file__).parent / 'backend/api.py').read_text(encoding='utf-8'))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), 'backend/api.py', 'exec'), namespace)
    app = FastAPI()
    app.include_router(namespace['router'], prefix='/api')
    with TestClient(app) as client:
        yield client, jobs, service, vectors, original, namespace


def submit(client, upload_id='a' * 32, filename='guide.md'):
    return client.post('/api/documents/upload/async', data={'upload_id': upload_id}, files={'file': (filename, b'new file', 'text/markdown')})


def test_duplicate_submission_uses_one_durable_job(upload_api):
    client, jobs, service, vectors, original, _ = upload_api
    response = submit(client)
    assert response.status_code == 200, response.text
    assert response.json()['job_id'] == 'a' * 32
    assert jobs.get_job('a' * 32)['status'] == 'completed'
    assert submit(client).json()['job_id'] == 'a' * 32
    assert len(vectors.rows) == 1
    assert len(jobs.list_jobs()) == 1
    assert submit(client, filename='different.md').status_code == 409


def test_failed_upload_keeps_staging_and_explicit_retry_succeeds(upload_api):
    client, jobs, service, vectors, original, _ = upload_api
    vectors.fail_insert = True
    assert submit(client).status_code == 200
    failure = client.get('/api/documents/upload/jobs/' + 'a' * 32).json()
    assert failure['status'] == 'failed'
    assert failure['retryable'] is True
    assert 'staging_path' not in failure
    assert original.read_text(encoding='utf-8') == 'old file'
    response = client.post('/api/documents/upload/jobs/' + 'a' * 32 + '/retry')
    assert response.status_code == 200, response.text
    assert jobs.get_job('a' * 32)['status'] == 'completed'
    assert jobs.get_job('a' * 32)['attempt'] == 2
    assert original.read_text(encoding='utf-8') == 'new file'
    assert client.post('/api/documents/upload/jobs/' + 'a' * 32 + '/retry').status_code == 409


def test_bad_upload_id_rejected_before_job_creation(upload_api):
    client, jobs, *_ = upload_api
    assert submit(client, upload_id='../not-a-job').status_code == 400
    assert jobs.list_jobs() == []


def test_startup_recovery_finalizes_committed_job_without_replacing_again(upload_api):
    client, jobs, service, vectors, original, namespace = upload_api
    job_id = 'a' * 32
    staging = service.upload_root / '.staging' / job_id / 'guide.md'
    jobs.create_job('guide.md', job_id=job_id)
    jobs.update_job(job_id, staging_path=str(staging))
    jobs.complete_step(job_id, 'upload')
    service.replace(job_id, staging, 'guide.md')
    assert jobs.get_job(job_id)['status'] == 'running'
    jobs.fail_job(job_id, 'vector_store', 'stale task status')
    namespace['recover_upload_operations'](mark_interrupted=True)
    assert jobs.get_job(job_id)['status'] == 'completed'
    assert all(step['status'] == 'completed' for step in jobs.get_job(job_id)['steps'])
    assert len(vectors.rows) == 1
    assert not staging.exists()


def test_retry_of_stale_status_finalizes_existing_commit_without_reentering_replace(upload_api):
    client, jobs, service, vectors, original, _ = upload_api
    job_id = 'a' * 32
    staging = service.upload_root / '.staging' / job_id / 'guide.md'
    jobs.create_job('guide.md', job_id=job_id)
    jobs.update_job(job_id, staging_path=str(staging))
    jobs.complete_step(job_id, 'upload')
    service.replace(job_id, staging, 'guide.md')
    # The data commit persisted, but job metadata was restored from an older state.
    jobs.fail_job(job_id, 'vector_store', 'interrupted status update')
    jobs.update_job(job_id, retryable=True)
    response = client.post('/api/documents/upload/jobs/' + job_id + '/retry')
    assert response.status_code == 200
    assert jobs.get_job(job_id)['status'] == 'completed'
    assert len(vectors.rows) == 1
    assert original.read_text(encoding='utf-8') == 'new file'


def test_staging_creation_failure_is_recorded(upload_api, monkeypatch):
    client, jobs, *_, namespace = upload_api
    monkeypatch.setitem(namespace, 'create_upload_staging_path', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    response = submit(client)
    assert response.status_code == 500
    assert jobs.get_job('a' * 32)['status'] == 'failed'
    assert jobs.get_job('a' * 32)['retryable'] is False


def test_sync_upload_uses_compensating_workflow(upload_api):
    client, jobs, service, vectors, original, _ = upload_api
    vectors.fail_insert = True
    response = client.post('/api/documents/upload', files={'file': ('guide.md', b'new file')})
    assert response.status_code == 500
    job = jobs.list_jobs()[0]
    assert job['status'] == 'failed' and job['retryable']
    assert original.read_text(encoding='utf-8') == 'old file'
    assert [row['text'] for row in vectors.rows] == ['old leaf']


def test_receive_failure_never_retries_partial_file(upload_api, monkeypatch):
    client, jobs, service, vectors, original, namespace = upload_api
    async def receive_partial(file, path):
        path.write_bytes(b'partial')
        raise OSError('receiver disconnected')
    monkeypatch.setitem(namespace, '_save_upload_file', receive_partial)
    assert submit(client).status_code == 500
    job = jobs.get_job('a' * 32)
    assert job['retryable'] is False
    assert not Path(job['staging_path']).exists()
    assert client.post('/api/documents/upload/jobs/' + 'a' * 32 + '/retry').status_code == 409
    assert original.read_text(encoding='utf-8') == 'old file'


def test_failed_compensation_blocks_reads_deletes_and_retry_until_recovered(upload_api, monkeypatch):
    client, jobs, service, vectors, original, namespace = upload_api
    real_insert = vectors.insert
    def lose_storage(rows):
        real_insert(rows)
        vectors.offline = True
        raise RuntimeError('storage offline during insert')
    monkeypatch.setattr(vectors, 'insert', lose_storage)
    assert submit(client).status_code == 200
    job = jobs.get_job('a' * 32)
    assert job['recovery_required'] and not job['retryable']
    assert client.get('/api/documents').status_code == 503
    assert client.delete('/api/documents/guide.md').status_code == 503
    deleted = client.delete('/api/documents/delete/async/guide.md')
    assert deleted.status_code == 200
    assert namespace['delete_job_manager'].get_job(deleted.json()['job_id'])['status'] == 'failed'
    assert client.post('/api/documents/upload/jobs/' + 'a' * 32 + '/retry').status_code == 409
    assert client.post('/api/documents/upload/recover').status_code == 503
    # A later upload blocked by this journal has not mutated any data itself.
    # It must not become a permanently recovery-required orphan without a journal.
    assert submit(client, upload_id='b' * 32).status_code == 200
    blocked = jobs.get_job('b' * 32)
    assert blocked['status'] == 'failed'
    assert not blocked['recovery_required'] and blocked['retryable']
    vectors.offline = False
    monkeypatch.setattr(vectors, 'insert', real_insert)
    assert client.post('/api/documents/upload/recover').status_code == 200
    assert [row['text'] for row in vectors.rows] == ['old leaf']
    job = jobs.get_job('a' * 32)
    assert not job['recovery_required'] and job['retryable']
    assert client.post('/api/documents/upload/jobs/' + 'a' * 32 + '/retry').status_code == 200
    assert jobs.get_job('a' * 32)['status'] == 'completed'
    assert client.post('/api/documents/upload/jobs/' + 'b' * 32 + '/retry').status_code == 200
    assert jobs.get_job('b' * 32)['status'] == 'completed'


def test_startup_compensates_crash_and_requires_explicit_retry(upload_api):
    _, jobs, service, vectors, original, namespace = upload_api
    job_id = 'a' * 32
    staging = service.upload_root / '.staging' / job_id / 'guide.md'
    jobs.create_job('guide.md', job_id=job_id)
    jobs.update_job(job_id, staging_path=str(staging))
    jobs.complete_step(job_id, 'upload')
    vectors.crash_insert = True
    with pytest.raises(ProcessCrash):
        namespace['_process_upload_job'](job_id, str(staging), 'guide.md')
    namespace['recover_upload_operations'](mark_interrupted=True)
    job = jobs.get_job(job_id)
    assert job['status'] == 'failed' and job['retryable']
    assert original.read_text(encoding='utf-8') == 'old file'
    assert [row['text'] for row in vectors.rows] == ['old leaf']
    assert staging.exists()
