"""Run the real Vue diagnostics methods with controlled request and timer boundaries."""
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent
HARNESS = r"""
const vm = require('node:vm');
const fs = require('node:fs');
let options, sequence = 0;
const intervals = new Map();
const documentDouble = {hidden: false, documentElement: {dataset: {}}, addEventListener() {}, removeEventListener() {}};
const windowDouble = {matchMedia() {return {matches: false};}, addEventListener() {}, removeEventListener() {}};
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
 Vue: {createApp(value) {options = value; return {mount() {}};}},
 window: windowDouble, document: documentDouble,
 localStorage: {getItem() {}, setItem() {}, removeItem() {}},
 AbortController, Date, console,
 setInterval(fn, ms) {const id = ++sequence; intervals.set(id, {fn, ms}); return id;},
 clearInterval(id) {intervals.delete(id);}, setTimeout, clearTimeout,
});
const state = options.data();
for (const [name, method] of Object.entries(options.methods)) state[name] = method.bind(state);
for (const [name, getter] of Object.entries(options.computed)) Object.defineProperty(state, name, {get: () => getter.call(state)});
state.token = 'admin-token'; state.currentUser = {role: 'admin'}; state.activeWorkspace = 'documents';
state.showToast = () => {}; state.$nextTick = fn => fn(); state.closeMobilePanels = () => {};
const requests = [];
const response = (status, body) => ({ok: status >= 200 && status < 300, status, json: async () => body});
let query = async () => response(200, {items: [], counts: {}, groups: [], next_cursor: null});
state.authFetch = async (url, opts = {}) => {requests.push({url, opts}); return query(url, opts);};
const deferred = () => {let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve};};
const emit = value => process.stdout.write(JSON.stringify(value));
"""


def run(scenario):
    completed = subprocess.run(
        ['node', '-e', HARNESS + '\n(async () => {\n' + scenario
         + '\n})().catch(error => {console.error(error); process.exitCode = 1;});'],
        input=(ROOT / 'frontend/script.js').read_text(encoding='utf-8'),
        capture_output=True, text=True, encoding='utf-8', timeout=12,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_refresh_failure_preserves_previous_data_and_marks_it_stale():
    result = run("""
query = async url => response(200, url.includes('summary')
  ? {counts: {failed: 3}, groups: []} : {items: [{id: 1}], next_cursor: null});
await state.refreshDiagnostics();
query = async () => response(503, {detail: 'diagnostics offline'});
await state.refreshDiagnostics();
emit({summary: state.diagnosticsSummary, alerts: state.diagnosticsAlerts});
""")
    assert result['summary']['data']['counts']['failed'] == 3
    assert result['summary']['stale'] and result['summary']['error']
    assert result['alerts']['data']['items'] == [{'id': 1}]


def test_polling_is_visible_admin_only_nonoverlapping_and_discards_late_results():
    result = run("""
const waiting = deferred(); query = async () => waiting.promise;
state.syncDiagnosticsPolling();
const tick = [...intervals.values()][0];
await tick.fn(); await tick.fn();
const count = requests.length;
documentDouble.hidden = true; state.syncDiagnosticsPolling();
waiting.resolve(response(200, {counts: {failed: 99}, items: [], groups: []}));
await new Promise(resolve => setImmediate(resolve));
const hidden = {timers: intervals.size, summary: state.diagnosticsSummary.data};
documentDouble.hidden = false; state.currentUser = {role: 'user'}; state.syncDiagnosticsPolling();
emit({count, interval: tick.ms, hidden, timers: intervals.size});
""")
    assert result['count'] == 4 and result['interval'] == 10000
    assert result['hidden'] == {'timers': 0, 'summary': None}
    assert result['timers'] == 0


def test_switching_trace_cannot_be_overwritten_by_older_response():
    result = run("""
const old = deferred();
query = async url => url.endsWith('/first') ? old.promise : response(200, {trace: {id: 'second'}, spans: [], events: [], alerts: []});
const first = state.selectDiagnosticTrace('first');
await state.selectDiagnosticTrace('second');
old.resolve(response(200, {trace: {id: 'first'}, spans: [], events: [], alerts: []}));
await first;
emit(state.diagnosticsTraceDetail.data);
""")
    assert result['trace']['id'] == 'second'


def test_acknowledgement_never_resolves_alert_or_replays_business_action():
    result = run("""
state.diagnosticsAlerts.data = {items: [{id: 8, resolved_at: null, acknowledged_at: null}]};
query = async () => response(200, {id: 8, acknowledged_at: '2026-09-30', acknowledged_by: 'admin', resolved_at: null});
await state.acknowledgeDiagnosticAlert(8);
emit({requests: requests.map(r => ({url: r.url, method: r.opts.method})), alert: state.diagnosticsAlerts.data.items[0]});
""")
    assert result['requests'] == [{'url': '/api/documents/diagnostics/alerts/8/acknowledge', 'method': 'POST'}]
    assert result['alert']['acknowledged_at'] and result['alert']['resolved_at'] is None


def test_attempt_switch_discards_old_events_and_cursor_pages_are_deduplicated():
    result = run("""
const old = deferred();
state.diagnosticsJobOperation = 'upload'; state.diagnosticsJobId = 'a'.repeat(32);
query = async url => url.includes('attempt=1') ? old.promise
  : response(200, {items: [{id: 2, attempt: 2}], next_cursor: 2});
const first = state.selectDiagnosticAttempt(1);
await state.selectDiagnosticAttempt(2);
old.resolve(response(200, {items: [{id: 1, attempt: 1}], next_cursor: 1})); await first;
await state.loadDiagnosticEvents(true);
emit({events: state.diagnosticsEvents.data.items, cursor: state.diagnosticsEvents.data.next_cursor, urls: requests.map(r => r.url)});
""")
    assert result['events'] == [{'id': 2, 'attempt': 2}] and result['cursor'] == 2
    assert 'after_id=2' in result['urls'][-1]


def test_logout_erases_diagnostic_data_and_cancels_pending_work():
    result = run("""
const waiting = deferred(); query = async () => waiting.promise;
state.diagnosticsAlerts.data = {items: [{id: 1}]}; state.syncDiagnosticsPolling();
state.handleLogout();
waiting.resolve(response(200, {items: [{id: 2}], counts: {}, groups: []}));
await new Promise(resolve => setImmediate(resolve));
emit({alerts: state.diagnosticsAlerts.data, timers: intervals.size, token: state.token});
""")
    assert result == {'alerts': None, 'timers': 0, 'token': ''}


def test_missing_durations_and_cancelled_states_are_not_reported_as_zero_or_success():
    result = run("""
emit({unknown: state.formatDiagnosticDuration(null), zero: state.formatDiagnosticDuration(0),
 cancelled: state.diagnosticStatusLabel('cancelled'), skipped: state.diagnosticStatusLabel('not_executed')});
""")
    assert result['unknown'] == '未知' and result['zero'] == '0 ms'
    assert result['cancelled'] == '已取消' and result['skipped'] == '未执行'


def test_document_index_failure_does_not_block_diagnostics_loading():
    result = run("""
query = async url => {
 if (url === '/api/documents') return response(503, {detail: 'index recovery pending'});
 if (url.includes('/diagnostics/')) return response(200, {items: [], counts: {failed: 2}, groups: []});
 return response(200, []);
};
await state.loadDocuments();
state.stopDiagnosticsPolling();
emit({error: state.documentsError, summary: state.diagnosticsSummary.data, urls: requests.map(r => r.url)});
""")
    assert 'index recovery pending' in result['error']
    assert result['summary']['counts']['failed'] == 2
    assert '/api/documents/diagnostics/alerts?state=open&limit=50' in result['urls']


def test_editing_trace_input_clears_previous_detail_when_submitting_new_selection():
    result = run("""
query = async () => response(200, {trace: {id: 'first'}, spans: []});
await state.selectDiagnosticTrace('first');
state.diagnosticsTraceId = 'second';
const pending = deferred(); query = async () => pending.promise;
const request = state.selectDiagnosticTrace(state.diagnosticsTraceId);
const during = state.diagnosticsTraceDetail.data;
pending.resolve(response(200, {trace: {id: 'second'}, spans: []})); await request;
emit({during, after: state.diagnosticsTraceDetail.data.trace.id});
""")
    assert result == {'during': None, 'after': 'second'}


def test_leaving_workspace_and_unmount_both_stop_polling():
    result = run("""
state.syncDiagnosticsPolling(); state.activeWorkspace = 'chat';
options.watch.activeWorkspace.call(state);
const afterLeaving = intervals.size;
state.activeWorkspace = 'documents'; options.watch.activeWorkspace.call(state);
options.beforeUnmount.call(state);
emit({afterLeaving, afterUnmount: intervals.size, requests: Object.keys(state.diagnosticsRequests)});
""")
    assert result == {'afterLeaving': 0, 'afterUnmount': 0, 'requests': []}


def test_task_history_defaults_to_latest_attempt_regardless_of_api_sort_order():
    result = run("""
query = async url => response(200, url.includes('/events') ? {items: [], next_cursor: null}
 : {attempts: [{attempt: 3, spans: []}, {attempt: 1, spans: []}], diagnostics_available: true});
await state.selectDiagnosticJob('upload', 'a'.repeat(32));
emit({attempt: state.diagnosticsAttempt, lastUrl: requests[requests.length - 1].url});
""")
    assert result['attempt'] == 3 and 'attempt=3' in result['lastUrl']


def test_malformed_success_response_does_not_erase_alerts_or_claim_healthy():
    result = run("""
state.diagnosticsAlerts.data = {items: [{id: 9, severity: 'critical'}]};
query = async () => response(200, {});
await state.loadDiagnosticAlerts();
emit(state.diagnosticsAlerts);
""")
    assert result['data']['items'][0]['id'] == 9
    assert result['stale'] and result['error']


def test_persistent_trace_gap_remains_visible_after_current_degradation_clears():
    result = run("""
state.diagnosticsTab = 'chat';
state.diagnosticsTraceDetail.data = {trace: {status: 'completed', diagnostics_incomplete: true}, spans: [], diagnostics_degraded: false, diagnostics_gaps: 0};
emit({warning: state.diagnosticIntegrityMessage(state.diagnosticsTraceDetail), empty: state.diagnosticMissingStagesMessage});
""")
    assert '记录不完整，部分阶段未知' in result['warning']
    assert '记录不完整，部分阶段未知' in result['empty']
    assert '可能未执行' not in result['empty']


def test_attempt_and_runtime_gap_evidence_are_both_visible():
    result = run("""
state.diagnosticsAttempt = 2;
state.diagnosticsJobDetail.data = {attempts: [{attempt: 1, spans: []}, {attempt: 2, diagnostics_incomplete: true, spans: []}], diagnostics_degraded: false};
state.diagnosticsSummary.data = {counts: {completed: 1}, groups: [], diagnostics_degraded: false, diagnostics_gaps: 3};
emit({attempt: state.diagnosticIntegrityMessage(state.diagnosticsJobDetail), summary: state.diagnosticIntegrityMessage(state.diagnosticsSummary), empty: state.diagnosticMissingStagesMessage});
""")
    assert '记录不完整，部分阶段未知' in result['attempt']
    assert '3' in result['summary'] and '缺口' in result['summary']
    assert '可能未执行' not in result['empty']
