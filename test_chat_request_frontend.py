import json
import subprocess
import pytest
from test_upload_recovery_frontend import HARNESS, ROOT


def run_frontend(scenario):
    # Chat streams use the platform decoder; retain the existing real Vue harness.
    harness = HARNESS.replace('crypto, AbortController,', 'crypto, AbortController, TextDecoder,')
    result = subprocess.run(
        ['node', '-e', harness + '\n(async () => {\n' + scenario
         + '\n})().catch(error => {console.error(error); process.exitCode = 1;});'],
        input=(ROOT / 'frontend/script.js').read_text(encoding='utf-8'),
        encoding='utf-8', text=True, capture_output=True, timeout=12)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_network_retry_reuses_uuid_but_new_question_gets_new_uuid():
    result = run_frontend('''
state.isAuthenticated = true;
state.$nextTick = fn => fn();
state.scrollToBottom = () => {};
state.loadSessions = () => {};
state.userInput = 'first';
const bodies = [];
query = async (url, opts) => {bodies.push(JSON.parse(opts.body)); throw new TypeError('network lost');};
await state.handleSend();
await state.retryMessage(state.messages.at(-1));
state.userInput = 'second';
await state.handleSend();
emit(bodies);
''')
    from uuid import UUID
    assert len(result) == 3
    assert UUID(result[0]['request_id']).version == 4
    assert result[0]['request_id'] == result[1]['request_id']
    assert result[2]['request_id'] != result[0]['request_id']


@pytest.mark.parametrize('terminal_code', ['REQUEST_FAILED', 'REQUEST_CANCELLED'])
def test_busy_and_terminal_retry_feedback(terminal_code):
    result = run_frontend('''
state.isAuthenticated = true;
state.$nextTick = fn => fn();
state.scrollToBottom = () => {};
state.loadSessions = () => {};
state.userInput = 'first';
const bodies = [];
query = async (url, opts) => {bodies.push(JSON.parse(opts.body)); return response(409, {detail:{code:'SESSION_BUSY'}});};
await state.handleSend();
const busy = state.messages.at(-1).streamError;
query = async (url, opts) => {bodies.push(JSON.parse(opts.body)); return response(409, {detail:{code:'TERMINAL_CODE'}});};
await state.retryMessage(state.messages.at(-1));
await state.retryMessage(state.messages.at(-1));
emit({busy, bodies});
'''.replace('TERMINAL_CODE', terminal_code))
    assert '正在' in result['busy']
    assert result['bodies'][0]['request_id'] == result['bodies'][1]['request_id']
    assert result['bodies'][2]['request_id'] != result['bodies'][1]['request_id']


@pytest.mark.parametrize('failure', ['sse_error', 'abort'])
def test_unconfirmed_error_or_abort_keeps_request_identity(failure):
    result = run_frontend('''
state.isAuthenticated = true;
state.$nextTick = fn => fn();
state.scrollToBottom = () => {};
state.loadSessions = () => {};
state.userInput = 'question';
const bodies = [];
query = async (url, opts) => {
    bodies.push(JSON.parse(opts.body));
    if ('FAILURE' === 'abort') {
        const error = new Error('client aborted'); error.name = 'AbortError'; throw error;
    }
    let read = false;
    return {ok: true, body: {getReader: () => ({read: async () => {
        if (read) return {done: true};
        read = true;
        return {done: false, value: new TextEncoder().encode('data: {"type":"content","content":"answer"}\\n\\ndata: {"type":"error","content":"post-commit memory failure"}\\n\\ndata: [DONE]\\n\\n')};
    }})}};
};
await state.handleSend();
await state.retryMessage(state.messages.at(-1));
emit(bodies);
'''.replace('FAILURE', failure))
    assert result[0]['request_id'] == result[1]['request_id']


def test_supplement_is_saved_before_abort_then_starts_queued_run():
    result = run_frontend('''
state.isAuthenticated = true;
state.$nextTick = fn => fn(); state.scrollToBottom = () => {}; state.loadSessions = () => {};
state.userInput = 'original';
const calls = []; let releaseRead; let ready;
const began = new Promise(resolve => ready = resolve);
query = async (url, opts = {}) => {
    calls.push({url, body: opts.body ? JSON.parse(opts.body) : null});
    if (url.endsWith('/cancel')) {
        if (state.abortController.signal.aborted) throw new Error('abort before durable supplement');
        return response(200, {status:'cancel_requested', next_run_id:JSON.parse(opts.body).supplement_request_id});
    }
    if (url.includes('/chat/runs/')) return response(200, {status:'cancelled'});
    const body = JSON.parse(opts.body);
    if (body.message === 'original') return {ok:true, body:{getReader:()=>({read:()=>new Promise((resolve,reject)=>{
        opts.signal.addEventListener('abort',()=>{const error=new Error('cancelled');error.name='AbortError';reject(error);}); ready();
    })})}};
    let read=false;
    return {ok:true, body:{getReader:()=>({read:async()=>{
        if(read)return {done:true};read=true;
        return {done:false,value:new TextEncoder().encode('data: {"type":"content","content":"new answer"}\\n\\ndata: [DONE]\\n\\n')};
    }})}};
};
const original = state.handleSend(); await began;
state.userInput = 'supplement'; await state.handleSend(); await original;
emit({calls, messages:state.messages.map(m=>m.text), loading:state.isLoading});
''')
    calls = result['calls']
    assert calls[1]['url'].endswith('/cancel')
    next_id = calls[1]['body']['supplement_request_id']
    assert calls[-1]['body']['request_id'] == next_id
    assert calls[-1]['body']['message'] == 'supplement'
    assert result['messages'].count('supplement') == 1
    assert result['messages'][-1] == 'new answer'
    assert result['loading'] is False


def test_stop_explicitly_cancels_run_without_using_composer_text():
    result = run_frontend('''
state.isAuthenticated=true; state.isLoading=true;
state.sessionId='one'; state.activeRunId='d6f5d798-26ee-40c1-9655-abc44b2fb884';
state.abortController=new AbortController(); state.userInput='unsent supplement';
const calls=[];
query=async(url,opts={})=>{calls.push({url,body:opts.body?JSON.parse(opts.body):null});return response(200,{status:url.endsWith('/cancel')?'cancel_requested':'cancelled'});};
await state.handleStop();
emit({calls, aborted:state.abortController?.signal.aborted, input:state.userInput});
''')
    assert len(result['calls']) == 2
    assert result['calls'][0]['url'].endswith('/cancel')
    assert result['calls'][0]['body'] == {'session_id': 'one'}
    assert result['input'] == 'unsent supplement'


def test_stream_ignores_events_from_another_run():
    result = run_frontend('''
state.isAuthenticated=true;state.$nextTick=fn=>fn();state.scrollToBottom=()=>{};state.loadSessions=()=>{};
state.userInput='question';
query=async(url,opts)=>{
    const rid=JSON.parse(opts.body).request_id;let read=false;
    const events=[{type:'content',content:'stale',run_id:'old-run'},{type:'content',content:'current',run_id:rid}];
    return {ok:true,body:{getReader:()=>({read:async()=>{if(read)return {done:true};read=true;
        return {done:false,value:new TextEncoder().encode(events.map(e=>'data: '+JSON.stringify(e)+'\\n\\n').join('')+'data: [DONE]\\n\\n')};
    }})}};
};
await state.handleSend();emit(state.messages.at(-1).text);
''')
    assert result == 'current'


def test_failed_supplement_handoff_retries_saved_uuid_without_duplicate_input():
    result = run_frontend('''
state.isAuthenticated=true;state.isLoading=true;state.$nextTick=fn=>fn();state.scrollToBottom=()=>{};state.loadSessions=()=>{};
state.activeRunId='d6f5d798-26ee-40c1-9655-abc44b2fb884';state.abortController=new AbortController();
state.messages=[{text:'original',isUser:true},{text:'partial',isUser:false,requestId:state.activeRunId}];
state.userInput='supplement';let fail=true;const bodies=[];
query=async(url,opts={})=>{
    if(url.endsWith('/cancel')){bodies.push(JSON.parse(opts.body));return response(200,{status:'cancel_requested'});}
    if(url.includes('/chat/runs/')){if(fail)throw new Error('temporary network loss');return response(200,{status:'cancelled'});}
    bodies.push(JSON.parse(opts.body));let read=false;
    return {ok:true,body:{getReader:()=>({read:async()=>{if(read)return {done:true};read=true;
        return {done:false,value:new TextEncoder().encode('data: {"type":"content","content":"answer"}\\n\\ndata: [DONE]\\n\\n')};
    }})}};
};
await state.handleSupplement();const saved=state.messages.at(-1);fail=false;await state.retryMessage(saved);
emit({bodies,inputs:state.messages.filter(m=>m.isUser).map(m=>m.text),answer:saved.text});
''')
    assert result['inputs'] == ['original', 'supplement']
    assert result['bodies'][0]['supplement_request_id'] == result['bodies'][1]['request_id']
    assert result['answer'] == 'answer'


def test_unconfirmed_supplement_post_retries_same_transaction_identity():
    result = run_frontend('''
state.isAuthenticated=true;state.isLoading=true;state.$nextTick=fn=>fn();state.scrollToBottom=()=>{};state.loadSessions=()=>{};
state.activeRunId='d6f5d798-26ee-40c1-9655-abc44b2fb884';state.abortController=new AbortController();
state.messages=[{text:'original',isUser:true},{text:'partial',isUser:false,requestId:state.activeRunId}];
state.userInput='supplement';let first=true;const bodies=[];
query=async(url,opts={})=>{
    if(url.endsWith('/cancel')){bodies.push(JSON.parse(opts.body));if(first){first=false;throw new TypeError('response lost after commit');}return response(200,{status:'cancel_requested'});}
    if(url.includes('/chat/runs/'))return response(200,{status:'cancelled'});
    bodies.push(JSON.parse(opts.body));let read=false;
    return {ok:true,body:{getReader:()=>({read:async()=>{if(read)return {done:true};read=true;
        return {done:false,value:new TextEncoder().encode('data: {"type":"content","content":"answer"}\\n\\ndata: [DONE]\\n\\n')};
    }})}};
};
await state.handleSupplement();const saved=state.messages.at(-1);await state.retryMessage(saved);
emit({bodies,inputs:state.messages.filter(m=>m.isUser).map(m=>m.text),answer:saved.text});
''')
    assert len(result['bodies']) == 3
    assert result['bodies'][0] == result['bodies'][1]
    assert result['bodies'][0]['supplement_request_id'] == result['bodies'][2]['request_id']
    assert result['inputs'] == ['original', 'supplement']
    assert result['answer'] == 'answer'


def test_session_switch_while_waiting_never_sends_old_question_to_new_session():
    result = run_frontend('''
state.isAuthenticated=true;state.$nextTick=fn=>fn();state.scrollToBottom=()=>{};state.loadSessions=()=>{};
state.sessionId='old';state.stoppingRunId='old-run';state.userInput='old question';
let release,ready;const began=new Promise(resolve=>ready=resolve);const posts=[];
query=async(url,opts={})=>{
    if(opts.method==='POST'){posts.push(JSON.parse(opts.body));throw new Error('unexpected POST');}
    ready();return await new Promise(resolve=>release=resolve);
};
const old=state.handleSend();await began;state.cancelActiveStream();state.sessionId='new';
release(response(200,{status:'cancelled'}));await old;emit(posts);
''')
    assert result == []


def test_reload_restores_saved_supplement_and_resumes_its_uuid():
    result = run_frontend('''
state.isAuthenticated=true;state.$nextTick=fn=>fn();state.scrollToBottom=()=>{};state.loadSessions=()=>{};
state.closeInspector=()=>{};state.closeMobilePanels=()=>{};
const posts=[];const rid='4a6042c2-904a-4ce4-965d-9f9f7110c9c8';
query=async(url,opts={})=>{
    if(url.includes('/sessions/'))return response(200,{messages:[{type:'human',content:'original'},{type:'human',content:'supplement'}],pending_run:{run_id:rid,source_run_id:'old-run',message:'supplement'}});
    if(url.includes('/chat/runs/'))return response(200,{status:'cancelled'});
    posts.push(JSON.parse(opts.body));let read=false;
    return {ok:true,body:{getReader:()=>({read:async()=>{if(read)return {done:true};read=true;
        return {done:false,value:new TextEncoder().encode('data: {"type":"content","content":"answer"}\\n\\ndata: [DONE]\\n\\n')};
    }})}};
};
await state.loadSession('one');await state.retryMessage(state.messages.at(-1));
emit({posts,inputs:state.messages.filter(m=>m.isUser).map(m=>m.text),answer:state.messages.at(-1).text});
''')
    assert result['posts'][0]['request_id'] == '4a6042c2-904a-4ce4-965d-9f9f7110c9c8'
    assert result['inputs'] == ['original', 'supplement']
    assert result['answer'] == 'answer'


def test_reactive_pending_wrapper_does_not_break_supplement_resume():
    result = run_frontend('''
state.isAuthenticated=true;state.isLoading=true;state.$nextTick=fn=>fn();state.scrollToBottom=()=>{};state.loadSessions=()=>{};
state.activeRunId='d6f5d798-26ee-40c1-9655-abc44b2fb884';state.abortController=new AbortController();
state.messages=[{text:'original',isUser:true},{text:'partial',isUser:false,requestId:state.activeRunId}];
let rawPending=null;
Object.defineProperty(state,'pendingSupplement',{get:()=>rawPending?new Proxy(rawPending,{}):null,set:value=>rawPending=value});
state.userInput='supplement';const posts=[];
query=async(url,opts={})=>{
    if(url.endsWith('/cancel'))return response(200,{status:'cancel_requested'});
    if(url.includes('/chat/runs/'))return response(200,{status:'cancelled'});
    posts.push(JSON.parse(opts.body));let read=false;
    return {ok:true,body:{getReader:()=>({read:async()=>{if(read)return {done:true};read=true;
        return {done:false,value:new TextEncoder().encode('data: {"type":"content","content":"answer"}\\n\\ndata: [DONE]\\n\\n')};
    }})}};
};
await state.handleSupplement();emit({posts,answer:state.messages.at(-1).text});
''')
    assert len(result['posts']) == 1
    assert result['answer'] == 'answer'


def test_late_history_failure_cannot_clear_new_session_messages():
    result = run_frontend('''
state.isAuthenticated=true;state.$nextTick=fn=>fn();state.scrollToBottom=()=>{};
state.closeInspector=()=>{};state.closeMobilePanels=()=>{};
let rejectOld,ready;const began=new Promise(resolve=>ready=resolve);
query=async()=>{ready();return await new Promise((resolve,reject)=>rejectOld=reject);};
const old=state.loadSession('old');await began;
state.cancelActiveStream();state.sessionId='new';state.messages=[{text:'new answer',isUser:false}];state.isLoading=true;
rejectOld(new Error('late failure'));await old;emit({messages:state.messages,loading:state.isLoading});
''')
    assert result['messages'][0]['text'] == 'new answer'
    assert result['loading'] is True
