"""Exercise the real Vue upload methods with controlled HTTP boundaries."""

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parent
HARNESS = r"""
const vm = require('node:vm');
const fs = require('node:fs');
const crypto = require('node:crypto').webcrypto;
let options;
let now = 0;
let uploadReplies = [];
let requests = [];
let fetchRequests = [];
let query = async () => response(200, []);
let rawFetch = async () => response(200, {});
const response = (status, body) => ({status, ok: status >= 200 && status < 300, json: async () => body});
const job = (jobId, status, extra = {}) => ({
  job_id: jobId, filename: 'first.md', status, message: status, steps: [],
  retryable: status === 'failed', recovery_required: false, attempt: 1, ...extra,
});
class FormDataDouble {
  constructor() { this.values = {}; }
  append(name, value) { this.values[name] = value; }
}
class XMLHttpRequestDouble {
  constructor() { this.upload = {}; }
  open(method, url) { this.method = method; this.url = url; }
  setRequestHeader() {}
  send(form) {
    requests.push({url: this.url, method: this.method, values: form.values});
    const plannedReply = uploadReplies.shift();
    const reply = typeof plannedReply === 'function' ? plannedReply(this, form) : plannedReply;
    if (!reply) throw new Error('unexpected second upload');
    queueMicrotask(() => {
      if (reply.event) this[reply.event]();
      else {
        this.status = reply.status;
        this.responseText = reply.body;
        this.onload();
      }
    });
  }
}
const mediaQuery = {matches: false, addEventListener() {}, removeEventListener() {}};
const browserWindow = {matchMedia() {return mediaQuery;}, addEventListener() {}, removeEventListener() {}, setTimeout};
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
  Vue: {createApp(value) { options = value; return {mount() {}}; }},
  window: browserWindow,
  document: {documentElement: {dataset: {}}, createElement() {return {}; }},
  localStorage: {getItem() {return null;}, setItem() {}, removeItem() {}},
  marked: {setOptions() {}, parse(value) {return value;}},
  hljs: {getLanguage() {return false;}, highlight(value) {return {value};}},
  fetch: async (url, opts) => {fetchRequests.push({url, opts}); return rawFetch(url, opts);},
  XMLHttpRequest: XMLHttpRequestDouble, FormData: FormDataDouble, crypto, AbortController,
  Date: class extends Date {static now() {return now;}},
  setTimeout, clearTimeout, setInterval, clearInterval,
});
function createState() {
  const state = options.data();
  for (const [name, method] of Object.entries(options.methods)) state[name] = method.bind(state);
  state.$refs = {fileInput: {value: ''}, folderInput: {value: ''}};
  state.showToast = (message, tone) => state.toasts.push({message, tone});
  state.apiPathCandidates = path => [path, path.replace('/api/', '/')];
  state.authFetch = async (url, opts) => query(url, opts);
  state.waitBeforeUploadPoll = async delay => {now += delay;};
  state.loadDocuments = async () => {};
  return state;
}
const state = createState();
const emit = value => process.stdout.write(JSON.stringify(value));
"""


def run_frontend(scenario):
    result = subprocess.run(
        ["node", "-e", HARNESS + "\n(async () => {\n" + scenario
         + "\n})().catch(error => {console.error(error); process.exitCode = 1;});"],
        input=(ROOT / "frontend" / "script.js").read_text(encoding="utf-8"),
        encoding="utf-8", text=True, capture_output=True, timeout=12,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("reply", [
    {"event": "onerror"}, {"event": "ontimeout"},
    {"status": 200, "body": "<invalid json>"},
    {"status": 200, "body": "{}"},
    {"status": 503, "body": '{"detail":"upstream timeout"}'},
    {"status": 501, "body": '{"detail":"gateway error"}'},
])
def test_lost_upload_response_checks_stable_id_without_resending(reply):
    runtime = run_frontend("""
uploadReplies = [REPLY];
const queriedIds = [];
query = async url => {
  const id = url.split('/').pop();
  queriedIds.push(id);
  return response(200, job(id, 'completed'));
};
state.selectedFiles = [{name: 'first.md', size: 4}];
await state.uploadDocument();
emit({requests, queriedIds, progress: state.uploadProgress, failures: state.uploadFailures,
  unknown: state.uploadStatusUnknowns, busy: state.isUploading});
""".replace("REPLY", json.dumps(reply)))
    assert len(runtime["requests"]) == 1
    upload_id = runtime["requests"][0]["values"].get("upload_id")
    assert upload_id and len(upload_id) == 32
    assert all(char in "0123456789abcdef" for char in upload_id)
    assert runtime["queriedIds"] and set(runtime["queriedIds"]) == {upload_id}
    assert "成功 1 个" in runtime["progress"]
    assert runtime["failures"] == runtime["unknown"] == []
    assert runtime["busy"] is False


@pytest.mark.parametrize("unavailable", ["missing", "offline"])
def test_unconfirmed_submission_pauses_batch_and_keeps_remaining_files(unavailable):
    runtime = run_frontend("""
uploadReplies = [{event: 'onerror'}];
query = async () => {
  if ('MODE' === 'offline') throw new Error('offline');
  return response(404, {detail: 'not found'});
};
state.selectedFiles = [{name: 'first.md', size: 4}, {name: 'second.md', size: 5}];
await state.uploadDocument();
emit({requests, remaining: state.selectedFiles, failures: state.uploadFailures,
  unknown: state.uploadStatusUnknowns, progress: state.uploadProgress});
""".replace("MODE", unavailable))
    assert len(runtime["requests"]) == 1
    assert [file["name"] for file in runtime["remaining"]] == ["second.md"]
    assert runtime["failures"] == []
    assert runtime["unknown"][0]["jobId"] == runtime["requests"][0]["values"]["upload_id"]
    assert "暂停" in runtime["progress"]


@pytest.mark.parametrize("status", [400, 401, 413])
def test_explicit_upload_rejection_does_not_probe_or_resend(status):
    runtime = run_frontend("""
uploadReplies = [{status: STATUS, body: 'rejected'}];
let queries = 0;
let loggedOut = false;
query = async () => {queries += 1; throw new Error('must not query');};
state.handleLogout = () => {loggedOut = true;};
let error;
try {await state.uploadFileWithProgress({name: 'first.md', size: 1});}
catch (caught) {error = {message: caught.message, unknown: !!caught.isUploadStatusUnknown};}
emit({queries, requests, loggedOut, error});
""".replace("STATUS", str(status)))
    assert runtime["queries"] == 0
    assert len(runtime["requests"]) == 1
    assert runtime["error"]["unknown"] is False
    assert runtime["loggedOut"] is (status == 401)
    assert str(status) in runtime["error"]["message"] or status == 401


def test_polling_retains_terminal_failure_metadata_and_stops_on_deadline():
    runtime = run_frontend("""
let count = 0;
query = async url => response(200, job(url.split('/').pop(), ++count === 1 ? 'running' : 'completed'));
const completed = await state.waitForUploadJob('success', 3000);
query = async () => response(200, job('failed', 'failed', {attempt: 2, recovery_required: true}));
let failed;
try {await state.waitForUploadJob('failed', 3000);}
catch (error) {failed = {id: error.jobId, retryable: error.retryable,
  recoveryRequired: error.recoveryRequired, attempt: error.attempt, terminal: error.isTerminalUploadFailure};}
query = async () => response(200, job('slow', 'running'));
let timedOut;
try {await state.waitForUploadJob('slow', 1500);}
catch (error) {timedOut = {id: error.jobId, unknown: error.isUploadStatusUnknown};}
emit({completed: completed.status, failed, timedOut, now});
""")
    assert runtime["completed"] == "completed"
    assert runtime["failed"] == {
        "id": "failed", "retryable": True, "recoveryRequired": True,
        "attempt": 2, "terminal": True,
    }
    assert runtime["timedOut"] == {"id": "slow", "unknown": True}
    assert runtime["now"] < 5000


def test_failed_upload_stores_job_metadata_for_explicit_retry():
    runtime = run_frontend("""
uploadReplies = [{status: 200, body: JSON.stringify({job_id: 'failed-id'})}];
query = async () => response(200, job('failed-id', 'failed', {attempt: 3}));
state.selectedFiles = [{name: 'first.md', size: 4}];
await state.uploadDocument();
emit(state.uploadFailures);
""")
    assert len(runtime) == 1
    assert runtime[0]["jobId"] == "failed-id"
    assert runtime[0]["retryable"] is True
    assert runtime[0]["recoveryRequired"] is False
    assert runtime[0]["attempt"] == 3


def test_history_restores_failures_and_recheck_removes_only_completed_job():
    runtime = run_frontend("""
state.loadDocuments = options.methods.loadDocuments.bind(state);
query = async url => {
  if (url === '/api/documents') return response(200, {documents: []});
  if (url.endsWith('/jobs')) return response(200, [
    job('history-fail', 'failed', {staging_path: '/private/staging/file', attempt: 2}),
    job('history-active', 'running'), job('history-done', 'completed')]);
  return response(200, job('history-fail', 'completed'));
};
await state.loadDocuments();
const before = JSON.parse(JSON.stringify(state.uploadFailures));
await state.recheckUploadJob(state.uploadFailures[0]);
emit({before, failures: state.uploadFailures, unknown: state.uploadStatusUnknowns,
  stepsVisible: state.uploadSteps.length > 0});
""")
    assert runtime["before"][0]["jobId"] == "history-fail"
    assert runtime["before"][0]["attempt"] == 2
    assert "staging_path" not in runtime["before"][0]
    assert "history-done" not in json.dumps(runtime["before"])
    # A refreshed list in this fixture is stale: the recheck must still clear its completed row.
    assert runtime["failures"] == []
    assert runtime["unknown"][0]["jobId"] == "history-active"
    assert runtime["stepsVisible"] is True


def test_explicit_retry_reuses_server_file_and_blocks_duplicate_clicks():
    runtime = run_frontend("""
let release;
rawFetch = async () => new Promise(resolve => {release = () => resolve(response(200, {job_id: 'retry-id'}));});
query = async () => response(200, job('retry-id', 'completed', {attempt: 2}));
const failure = {filename: 'first.md', jobId: 'retry-id', message: 'failed', retryable: true, recoveryRequired: false};
state.uploadFailures = [failure];
const first = state.retryUploadJob(failure);
await state.retryUploadJob(failure);
release();
await first;
emit({fetchRequests, uploads: requests.length, failures: state.uploadFailures, busy: state.isUploading});
""")
    assert len(runtime["fetchRequests"]) == 1
    request = runtime["fetchRequests"][0]
    assert request["url"] == "/api/documents/upload/jobs/retry-id/retry"
    assert request["opts"]["method"] == "POST"
    assert not request["opts"].get("body")
    assert runtime["uploads"] == 0
    assert runtime["failures"] == []
    assert runtime["busy"] is False


def test_retry_lost_response_is_checked_without_reposting():
    runtime = run_frontend("""
rawFetch = async () => {throw new Error('response lost');};
query = async () => response(200, job('retry-id', 'completed', {attempt: 2}));
const failure = {filename: 'first.md', jobId: 'retry-id', retryable: true, recoveryRequired: false};
state.uploadFailures = [failure];
await state.retryUploadJob(failure);
emit({fetchRequests, failures: state.uploadFailures, unknown: state.uploadStatusUnknowns});
""")
    assert len(runtime["fetchRequests"]) == 1
    assert runtime["failures"] == runtime["unknown"] == []


def test_retry_requires_retryable_failure_and_no_pending_recovery():
    runtime = run_frontend("""
await state.retryUploadJob({jobId: 'unsafe', retryable: true, recoveryRequired: true});
await state.retryUploadJob({jobId: 'disabled', retryable: false, recoveryRequired: false});
emit({fetchRequests, busy: state.isUploading});
""")
    assert runtime["fetchRequests"] == []
    assert runtime["busy"] is False


def test_upload_document_double_click_starts_one_batch():
    runtime = run_frontend("""
uploadReplies = [{status: 200, body: JSON.stringify({job_id: 'one'})}];
query = async () => response(200, job('one', 'completed'));
state.selectedFiles = [{name: 'first.md', size: 1}];
await Promise.all([state.uploadDocument(), state.uploadDocument()]);
emit({count: requests.length, failures: state.uploadFailures});
""")
    assert runtime["count"] == 1
    assert runtime["failures"] == []


def test_recovery_is_explicit_single_post_then_refreshes_retry_eligibility():
    runtime = run_frontend("""
let release;
rawFetch = async () => new Promise(resolve => {release = () => resolve(response(200, {recovered: 1}));});
query = async () => response(200, job('blocked-id', 'failed', {attempt: 2}));
const failure = {filename: 'first.md', jobId: 'blocked-id', retryable: false, recoveryRequired: true};
state.uploadFailures = [failure];
const first = state.recoverUploadData(failure);
await state.recoverUploadData(failure);
release();
await first;
emit({fetchRequests, failures: state.uploadFailures, uploads: requests.length, busy: state.isUploading});
""")
    assert len(runtime["fetchRequests"]) == 1
    assert runtime["fetchRequests"][0]["url"] == "/api/documents/upload/recover"
    assert runtime["fetchRequests"][0]["opts"]["method"] == "POST"
    assert runtime["failures"][0]["retryable"] is True
    assert runtime["failures"][0]["recoveryRequired"] is False
    assert runtime["uploads"] == 0
    assert runtime["busy"] is False


@pytest.mark.parametrize("origin_mode", ["configured", "inferred"])
@pytest.mark.parametrize("operation", ["upload", "retry", "recovery"])
def test_upload_mutations_use_known_backend_before_static_server_501(origin_mode, operation):
    runtime = run_frontend("""
browserWindow.location = {protocol: 'http:', hostname: '127.0.0.1', port: '8765'};
if ('ORIGIN_MODE' === 'configured') browserWindow.STARMEW_API_ORIGIN = 'https://api.example.test';
state.apiPathCandidates = options.methods.apiPathCandidates.bind(state);
const backend = 'ORIGIN_MODE' === 'configured' ? 'https://api.example.test' : 'http://127.0.0.1:8000';
const operation = 'OPERATION';
let accepted = false;
uploadReplies = [(xhr, form) => {
  if (!xhr.url.startsWith(backend + '/')) return {status: 501, body: '{}'};
  accepted = true;
  return {status: 202, body: JSON.stringify({job_id: form.values.upload_id})};
}];
rawFetch = async url => {
  if (!url.startsWith(backend + '/')) return response(501, {detail: 'static server cannot POST'});
  accepted = true;
  return response(200, operation === 'recovery' ? {recovered: 1} : {job_id: 'existing-id'});
};
query = async url => {
  const id = url.split('/').pop();
  if (!accepted && operation === 'upload') return response(404, {detail: 'not received'});
  return response(200, job(id, operation === 'recovery' || !accepted ? 'failed' : 'completed', {
    retryable: accepted, recovery_required: operation === 'recovery' && !accepted,
  }));
};
const failure = {filename: 'first.md', jobId: 'existing-id', retryable: true, recoveryRequired: operation === 'recovery'};
if (operation === 'upload') {
  state.selectedFiles = [{name: 'first.md', size: 1}];
  await state.uploadDocument();
} else {
  state.uploadFailures = [failure];
  await (operation === 'retry' ? state.retryUploadJob(failure) : state.recoverUploadData(failure));
}
emit({urls: requests.concat(fetchRequests).map(item => item.url), accepted,
  failures: state.uploadFailures, unknown: state.uploadStatusUnknowns});
""".replace("ORIGIN_MODE", origin_mode).replace("OPERATION", operation))
    backend = "https://api.example.test" if origin_mode == "configured" else "http://127.0.0.1:8000"
    path = {
        "upload": "/api/documents/upload/async",
        "retry": "/api/documents/upload/jobs/existing-id/retry",
        "recovery": "/api/documents/upload/recover",
    }[operation]
    assert runtime["urls"] == [backend + path]
    assert runtime["accepted"] is True
    assert runtime["unknown"] == []
    if operation == "recovery":
        assert runtime["failures"][0]["retryable"] is True
        assert runtime["failures"][0]["recoveryRequired"] is False
    else:
        assert runtime["failures"] == []


@pytest.mark.parametrize("operation", ["upload", "retry", "recovery"])
def test_backend_501_does_not_replay_mutation_at_another_origin(operation):
    runtime = run_frontend("""
browserWindow.STARMEW_API_ORIGIN = 'https://api.example.test';
state.apiPathCandidates = options.methods.apiPathCandidates.bind(state);
uploadReplies = [{status: 501, body: '{}'}];
rawFetch = async () => response(501, {detail: 'uncertain gateway response'});
query = async () => response(404, {detail: 'status not yet available'});
const operation = 'OPERATION';
const failure = {filename: 'first.md', jobId: 'existing-id', retryable: true, recoveryRequired: operation === 'recovery'};
if (operation === 'upload') {
  state.selectedFiles = [{name: 'first.md', size: 1}];
  await state.uploadDocument();
} else {
  state.uploadFailures = [failure];
  await (operation === 'retry' ? state.retryUploadJob(failure) : state.recoverUploadData(failure));
}
emit({urls: requests.concat(fetchRequests).map(item => item.url), failures: state.uploadFailures,
  unknown: state.uploadStatusUnknowns});
""".replace("OPERATION", operation))
    assert len(runtime["urls"]) == 1
    assert runtime["urls"][0].startswith("https://api.example.test/api/documents/upload/")
    assert runtime["failures"] == []
    assert len(runtime["unknown"]) == 1
