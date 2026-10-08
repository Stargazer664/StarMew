from test_upload_recovery_frontend import run_frontend


def test_delete_response_loss_checks_stable_id_without_replaying():
    result = run_frontend("""
rawFetch = async () => {throw new Error('response lost');};
const queried = [];
query = async url => {queried.push(url); return response(200, job(url.split('/').pop(), 'completed'));};
state.scheduleDeletedDocumentRemoval = () => {};
await state.performDeleteDocument('first.md');
state.stopAllDeleteJobPolling();
emit({requests: fetchRequests, queried, stored: state.deleteJobs['first.md']});
""")
    assert len(result['requests']) == 1
    request = result['requests'][0]
    assert request['opts']['method'] == 'DELETE'
    delete_id = request['url'].split('delete_id=')[-1]
    assert len(delete_id) == 32
    assert result['queried'] == ['/api/documents/delete/jobs/' + delete_id]
    assert result['stored']['status'] == 'completed'


def test_failed_delete_poll_means_unknown_and_blocks_resubmission():
    result = run_frontend("""
query = async () => {throw new Error('offline');};
state.setDeleteJob('first.md', {jobId: 'abc', status: 'running'});
await state.recheckDeleteJob('first.md');
emit({job: state.deleteJobs['first.md'], locked: state.isDeleteActionLocked('first.md')});
""")
    assert result['job']['status'] == 'unknown'
    assert result['locked'] is True


def test_delete_history_loads_even_when_document_index_is_unavailable():
    result = run_frontend("""
query = async url => {
  if (url === '/api/documents') return response(503, {detail: 'pending recovery'});
  if (url === '/api/documents/delete/jobs') return response(200, [job('delete-1', 'failed', {recovery_required: true})]);
  return response(200, []);
};
await options.methods.loadDocuments.call(state);
emit({error: state.documentsError, job: state.deleteJobs['first.md']});
""")
    assert 'pending recovery' in result['error']
    assert result['job']['jobId'] == 'delete-1' and result['job']['recoveryRequired']


def test_recovery_never_reissues_delete_and_checks_result_after_lost_response():
    result = run_frontend("""
state.syncDeleteJob('first.md', job('delete-1', 'failed', {recovery_required: true}));
rawFetch = async () => {throw new Error('lost recovery response');};
query = async () => response(200, job('delete-1', 'failed', {recovery_required: false}));
await state.recoverDeleteData('first.md');
emit({requests: fetchRequests, job: state.deleteJobs['first.md']});
""")
    assert len(result['requests']) == 1
    assert result['requests'][0]['url'] == '/api/documents/recover'
    assert result['requests'][0]['opts']['method'] == 'POST'
    assert result['job']['recoveryRequired'] is False


def test_latest_completed_delete_does_not_resurrect_older_failure():
    result = run_frontend("""
query = async () => response(200, [job('new', 'completed'), job('old', 'failed', {recovery_required: false})]);
await state.loadDeleteJobs();
emit({jobs: state.deleteJobs});
""")
    assert result['jobs'] == {}


def test_older_unfinished_recovery_is_not_hidden_by_newer_blocked_task():
    result = run_frontend("""
query = async () => response(200, [job('new', 'failed'), job('old', 'failed', {recovery_required: true})]);
state.setDeleteJob('first.md', {jobId: 'new', status: 'unknown'});
state.deletePollTimers['first.md'] = 123;
await state.loadDeleteJobs();
emit({job: state.deleteJobs['first.md'], timers: Object.keys(state.deletePollTimers)});
""")
    assert result['job']['jobId'] == 'old' and result['job']['recoveryRequired']
    assert result['timers'] == []


def test_committed_delete_with_stale_running_status_can_be_recovered_from_ui():
    result = run_frontend("""
state.syncDeleteJob('first.md', job('committed', 'running', {recovery_required: false}));
rawFetch = async () => response(200, {recovered: 1});
query = async () => response(200, job('committed', 'completed'));
state.scheduleDeletedDocumentRemoval = () => {};
await state.recoverDeleteData('first.md');
emit({requests: fetchRequests, job: state.deleteJobs['first.md']});
""")
    assert len(result['requests']) == 1 and result['requests'][0]['url'] == '/api/documents/recover'
    assert result['job']['status'] == 'completed'


def test_delete_confirmation_explains_original_file_retention():
    result = run_frontend("""
state.requestConfirmation = value => {state.pendingConfirmation = value;};
state.deleteDocument('first.md');
emit({message: state.pendingConfirmation.message});
""")
    assert '保留' in result['message'] and '永久删除' not in result['message']
