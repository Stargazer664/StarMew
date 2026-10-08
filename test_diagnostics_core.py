import asyncio
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def telemetry(tmp_path):
    assert Path('backend/telemetry.py').exists(), 'Missing shared diagnostics runtime'
    module = importlib.import_module('backend.telemetry')
    module.start_diagnostics(tmp_path / 'jobs.sqlite3', tmp_path / 'logs')
    yield module
    module.stop_diagnostics()


def test_nested_spans_first_answer_and_safe_errors(telemetry):
    with telemetry.trace('chat', 'answer') as execution:
        with telemetry.span('retrieval'):
            with pytest.raises(ValueError):
                with telemetry.span('milvus', component='milvus'):
                    raise ValueError('password=secret document body https://user:pass@example.test')
        execution.mark_first_answer()
    telemetry.flush()
    detail = telemetry.get_store().trace_detail(execution.id)
    assert detail['trace']['status'] == 'completed'
    assert detail['trace']['answer_first_content_ms'] >= 0
    outer, inner = sorted(detail['spans'], key=lambda item: item['started_at'])
    assert inner['parent_span_id'] == outer['id']
    assert inner['status'] == 'failed' and inner['error_type'] == 'ValueError'
    assert 'secret' not in json.dumps(detail)
    assert 'document body' not in json.dumps(detail)


def test_concurrent_contexts_do_not_mix(telemetry):
    async def request(label):
        with telemetry.trace('chat', 'answer') as execution:
            await asyncio.sleep(0)
            with telemetry.span(label):
                assert telemetry.current_trace() is execution
            return execution.id
    async def run():
        return await asyncio.gather(request('first'), request('second'))
    first, second = asyncio.run(run())
    telemetry.flush()
    assert [x['stage'] for x in telemetry.get_store().trace_detail(first)['spans']] == ['first']
    assert [x['stage'] for x in telemetry.get_store().trace_detail(second)['spans']] == ['second']
    assert telemetry.current_trace() is None


def test_document_attempt_history_survives_retry(telemetry):
    for attempt, status in [(1, 'failed'), (2, 'completed')]:
        telemetry.record_job({'job_id': 'a'*32, 'attempt': attempt, 'status': status,
                              'filename': 'private.txt', 'error': 'secret', 'steps': [],
                              'created_at': '2026-09-30T00:00:00+00:00'}, 'upload')
    telemetry.flush()
    rows = telemetry.get_store().document_attempts('upload', 'a'*32)
    assert {x['attempt']: x['status'] for x in rows} == {1:'failed', 2:'completed'}
    assert 'private' not in json.dumps(rows) and 'secret' not in json.dumps(rows)


def test_alert_acknowledgement_is_not_resolution(telemetry):
    with telemetry.trace('document', 'upload', job_id='b'*32) as execution:
        for _ in range(2):
            telemetry.event('recovery_failed', status='failed', code='PARENT_RESTORE_FAILED')
    telemetry.flush()
    store = telemetry.get_store()
    alerts = store.alerts()['items']
    assert len(alerts) == 1 and alerts[0]['occurrences'] == 2
    store.acknowledge(alerts[0]['id'], 'admin')
    assert store.alerts()['items'][0]['resolved_at'] is None
    with telemetry.bind_context(trace_id=execution.id, scope='document', operation='upload', job_id='b'*32):
        telemetry.event('recovery_completed', status='completed')
    telemetry.flush()
    assert store.alerts(state='open')['items'] == []


def test_diagnostics_failure_does_not_change_business_result(telemetry, monkeypatch):
    def broken(_):
        raise OSError('disk failed')
    monkeypatch.setattr(telemetry.get_store(), 'write', broken)
    with telemetry.trace('chat', 'answer'):
        with telemetry.span('generation'):
            result = 42
    telemetry.flush()
    assert result == 42 and telemetry.diagnostics_status()['diagnostics_degraded']


def test_summary_excludes_incomplete_duration(telemetry):
    with telemetry.trace('chat', 'answer'):
        with telemetry.span('generation', model='demo'):
            pass
    telemetry.flush()
    summary = telemetry.get_store().summary(scope='chat')
    group = next(x for x in summary['groups'] if x['stage'] == 'generation')
    assert group['count'] == 1 and group['avg_ms'] >= 0
    assert group['insufficient_samples'] is True


def test_slow_scan_deduplicates_until_span_ends(telemetry):
    with telemetry.trace('chat', 'answer'):
        with telemetry.span('generation', component='model') as operation:
            telemetry.check_slow_operations(now_ns=operation.started_ns + 400_000_000_000)
            telemetry.check_slow_operations(now_ns=operation.started_ns + 500_000_000_000)
            telemetry.flush()
            alerts = telemetry.get_store().alerts()['items']
            # Both public-first-answer wait and the individual model call are slow.
            assert len(alerts) == 2 and all(item['occurrences'] == 1 for item in alerts)
    telemetry.flush()
    assert telemetry.get_store().alerts()['items'] == []


def test_restart_marks_unfinished_spans_interrupted(telemetry):
    store = telemetry.get_store()
    store.write({'kind':'trace', 'id':'old', 'scope':'chat', 'operation':'answer', 'status':'running',
                 'run_id':'previous', 'started_at':'2026-09-29T00:00:00+00:00'})
    store.write({'kind':'span', 'id':'unfinished', 'trace_id':'old', 'scope':'chat', 'stage':'generation',
                 'status':'running', 'started_at':'2026-09-29T00:00:00+00:00'})
    store.mark_interrupted('new-run')
    detail = store.trace_detail('old')
    assert detail['trace']['status'] == 'interrupted'
    assert detail['spans'][0]['duration_ms'] is None


def test_recovery_does_not_resolve_unpersisted_status_or_unrelated_cleanup(telemetry):
    with telemetry.trace('document','upload',job_id='d'*32):
        telemetry.event('status_failed',status='failed',code='STATUS_PERSIST_FAILED',stage='status_persist')
        telemetry.event('cleanup_failed',status='warning',code='CLEANUP_FAILED',stage='staging_cleanup')
        telemetry.event('recovery_completed',status='completed')
        telemetry.event('cleanup_completed',status='completed',stage='journal_cleanup')
    telemetry.flush()
    assert {row['rule'] for row in telemetry.get_store().alerts()['items']}=={'STATUS_PERSIST_FAILED','CLEANUP_FAILED'}


def test_prune_protects_running_child_after_request_cancelled(telemetry):
    store=telemetry.get_store()
    store.write({'kind':'trace','id':'cancelled','scope':'chat','status':'cancelled','started_at':'2000-01-01T00:00:00+00:00'})
    store.write({'kind':'span','id':'still-running','trace_id':'cancelled','status':'running','started_at':'2000-01-01T00:00:00+00:00'})
    store.prune()
    assert store.trace_detail('cancelled') is not None


def test_restart_resolves_slow_interrupted_operation(telemetry):
    store=telemetry.get_store()
    store.write({'kind':'trace','id':'interrupted','scope':'chat','status':'running','started_at':'2000-01-01T00:00:00+00:00'})
    store.write({'kind':'alert','dedupe_key':'oldslow','rule':'SLOW_OPERATION','span_id':'interrupted','trace_id':'interrupted'})
    store.mark_interrupted('new-run')
    assert store.alerts()['items']==[]


def test_actionable_alert_severity_and_blocked_resolution(telemetry):
    with telemetry.trace('document','upload',job_id='e'*32):
        telemetry.event('operation_failed',status='failed',code='VECTOR_WRITE_FAILED',stage='vector_write')
        telemetry.event('rollback_completed',status='completed')
        telemetry.event('operation_failed',status='failed',code='JOURNAL_UNREADABLE',stage='journal_read')
        telemetry.event('operation_failed',status='failed',code='CLEANUP_FAILED',stage='staging_cleanup')
        telemetry.event('task_blocked',status='blocked',code='BLOCKED_BY_RECOVERY',job_id='f'*32,blocked_by_job_id='e'*32)
    telemetry.flush()
    alerts=telemetry.get_store().alerts()['items']
    assert next(item for item in alerts if item['code']=='JOURNAL_UNREADABLE')['severity']=='critical'
    assert next(item for item in alerts if item['code']=='CLEANUP_FAILED')['severity']=='warning'
    assert next(item for item in alerts if item['code']=='BLOCKED_BY_RECOVERY')['severity']=='critical'
    assert not any(item['code']=='VECTOR_WRITE_FAILED' and item['severity']=='error' for item in alerts)
    with telemetry.bind_context(job_id='e'*32,operation='upload',scope='recovery'):
        telemetry.event('recovery_completed',status='completed')
    telemetry.flush()
    assert not any(item['code']=='BLOCKED_BY_RECOVERY' for item in telemetry.get_store().alerts()['items'])


def test_late_slow_scan_cannot_reopen_completed_span_alert(telemetry):
    with telemetry.trace('chat','answer'):
        with telemetry.span('generation') as operation:
            pass
    telemetry.flush()
    telemetry.get_store().write({'kind':'alert','dedupe_key':'late','rule':'SLOW_OPERATION','span_id':operation.id})
    assert telemetry.get_store().alerts()['items']==[]


def test_transient_dropped_spans_leave_persistent_trace_gap(telemetry,monkeypatch):
    store=telemetry.get_store()
    original=store.write
    with telemetry.trace('chat','answer') as execution:
        telemetry.flush()
        with monkeypatch.context() as patch:
            patch.setattr(store,'write',lambda item: (_ for _ in ()).throw(OSError()) if item['kind']=='span' else original(item))
            with telemetry.span('generation'):
                pass
            telemetry.flush()
        telemetry.event('resumed')
    telemetry.flush()
    assert store.trace_detail(execution.id)['trace']['diagnostics_incomplete'] is True


def test_alerts_do_not_expose_user_or_session_identifiers(telemetry):
    with telemetry.trace('chat','answer',user_id='private-owner',session_id='private-session'):
        telemetry.event('model_call_failed',status='failed',code='MODEL_CALL_FAILED')
    telemetry.flush()
    assert 'private-owner' not in json.dumps(telemetry.get_store().alerts())
    assert 'private-session' not in json.dumps(telemetry.get_store().alerts())


def test_delayed_startup_retries_interrupted_reconciliation(tmp_path,monkeypatch):
    from backend import telemetry
    from backend.diagnostics_store import DiagnosticsStore
    path=tmp_path/'jobs.sqlite3'
    store=DiagnosticsStore(path);store.initialize()
    store.write({'kind':'trace','id':'old-run','scope':'chat','status':'running','run_id':'old'})
    initialize=DiagnosticsStore.initialize
    calls=[]
    def transient(self):
        calls.append(1)
        if len(calls)==1:
            raise OSError()
        initialize(self)
    monkeypatch.setattr(DiagnosticsStore,'initialize',transient)
    telemetry.start_diagnostics(path,tmp_path/'logs')
    try:
        with telemetry.trace('chat','answer'):
            pass
        telemetry.flush()
        assert store.trace_detail('old-run')['trace']['status']=='interrupted'
    finally:
        telemetry.stop_diagnostics()


def test_capacity_counts_traces_and_spans_without_events(telemetry):
    store=telemetry.get_store()
    for number in range(4):
        store.write({'kind':'trace','id':str(number),'scope':'chat','status':'completed','started_at':'2099-01-01T00:00:00+00:00'})
        store.write({'kind':'span','id':'s'+str(number),'trace_id':str(number),'status':'completed'})
    assert store.prune(max_records=1)['removed']>0


def test_shutdown_waits_actual_thread_terminal_event(telemetry):
    import threading
    from contextvars import copy_context
    store=telemetry.get_store()
    entered=threading.Event();release=threading.Event()
    errors=[]
    with telemetry.trace('chat','answer') as execution:
        context=copy_context()
        def work():
            try:
                with telemetry.span('model_call',component='model'):
                    entered.set()
                    assert release.wait(3)
            except BaseException as exc:
                errors.append(exc)
        worker=threading.Thread(target=context.run,args=(work,))
        worker.start();assert entered.wait(2)
        execution.set_result('cancelled')
    timer=threading.Timer(.1,release.set);timer.start()
    telemetry.stop_diagnostics()
    worker.join(2);timer.join(2)
    assert not worker.is_alive() and not errors
    assert store.trace_detail(execution.id)['spans'][0]['status']=='completed_after_cancel'


def test_queued_document_is_observed_before_worker_obtains_lock(telemetry):
    import time
    telemetry.record_job({'job_id':'1'*32,'attempt':1,'status':'pending','current_step':'prepare'},'delete')
    telemetry.check_slow_operations(now_ns=time.perf_counter_ns()+200_000_000_000)
    telemetry.flush()
    assert any(row['rule']=='SLOW_OPERATION' for row in telemetry.get_store().alerts()['items'])
    with telemetry.trace('document','delete',job_id='1'*32):
        pass
    telemetry.flush()
    assert telemetry.get_store().alerts()['items']==[]
