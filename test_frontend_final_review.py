from pathlib import Path
import json
import subprocess

from test_frontend_workspace import css_media_block, css_rules, frontend_source, parse_frontend


ROOT = Path(__file__).parent


def test_static_css_and_js_assets_are_cache_busted_for_browser_updates():
    """HTML must not let a browser pair a stale stylesheet with the new shell."""
    parser = parse_frontend()
    stylesheet = [attrs.get("href", "") for tag, attrs in parser.attrs if tag == "link" and attrs.get("rel") == "stylesheet" and "style.css" in attrs.get("href", "")]
    scripts = [attrs.get("src", "") for tag, attrs in parser.attrs if tag == "script" and "script.js" in attrs.get("src", "")]

    assert stylesheet and "?v=" in stylesheet[0]
    assert scripts and "?v=" in scripts[0]


def run_runtime_contract():
    """Run the real Vue option methods against deterministic browser boundary fakes."""
    harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
const crypto = require('node:crypto').webcrypto;
let options;
let mobile = false;
let overlay = false;
let fetchHandler = async () => ({ ok: true, status: 200, json: async () => ({}) });
const fetchCalls = [];
const focusEvents = [];
const makeFocusable = name => ({
    name,
    focus() { focusEvents.push(name); document.activeElement = this; },
});
const mql = matches => ({ matches, addEventListener() {}, removeEventListener() {} });
const storage = new Map();
const document = {
    documentElement: { dataset: {} },
    activeElement: null,
    createElement() { return { textContent: '', innerHTML: '' }; },
};
class FakeFormData {
    constructor() { this.entries = []; }
    append(field, file, filename) { this.entries.push({ field, file, filename }); }
}
const xhrPlans = [];
const xhrCalls = [];
class FakeXHR {
    constructor() { this.upload = {}; this.headers = {}; }
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(key, value) { this.headers[key] = value; }
    send(body) {
        xhrCalls.push({
            url: this.url,
            authorization: this.headers.Authorization,
            filename: body.entries[0]?.filename || null,
        });
        const plan = xhrPlans.shift();
        this.status = plan.status;
        this.responseText = JSON.stringify(plan.body || {});
        queueMicrotask(() => this.onload());
    }
}
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
    Vue: { createApp(value) { options = value; return { mount() {} }; } },
    window: {
        location: { protocol: 'http:', hostname: '127.0.0.1', port: '8765' },
        matchMedia(query) {
            if (query === '(max-width: 680px)') return mql(mobile);
            if (query === '(max-width: 1100px)') return mql(overlay);
            return mql(false);
        },
        addEventListener() {}, removeEventListener() {}, setTimeout() { return 1; },
    },
    document,
    localStorage: {
        getItem(key) { return storage.get(key) || null; },
        setItem(key, value) { storage.set(key, value); },
        removeItem(key) { storage.delete(key); },
    },
    marked: { setOptions() {}, parse(value) { return value; } },
    hljs: { getLanguage() { return false; }, highlight(value) { return { value }; } },
    fetch: async (url, requestOptions = {}) => {
        fetchCalls.push({ url, authorization: requestOptions.headers?.Authorization || null });
        return fetchHandler(url, requestOptions);
    },
    XMLHttpRequest: FakeXHR,
    FormData: FakeFormData,
    crypto,
    AbortController,
    TextDecoder,
    setInterval() { return 1; }, clearInterval() {}, clearTimeout() {}, setTimeout,
});
const bind = () => {
    const state = options.data();
    for (const [name, method] of Object.entries(options.methods)) state[name] = method.bind(state);
    for (const [name, getter] of Object.entries(options.computed)) {
        Object.defineProperty(state, name, { get: () => getter.call(state), configurable: true });
    }
    state.$nextTick = callback => callback();
    state.$refs = {};
    return state;
};
const response = (status, body = {}) => ({
    status, ok: status >= 200 && status < 300,
    json: async () => body,
});

(async () => {
    const result = {};

    const api = bind();
    api.token = 'secret'; api.currentUser = { username: 'tester', role: 'user' };
    fetchHandler = async url => url.startsWith('/api/')
        ? response(404, { detail: 'missing prefix' })
        : response(200, { sessions: [] });
    await api.authFetch('/api/sessions');
    result.authFallback = fetchCalls.splice(0);

    const login = bind();
    login.authForm.username = 'tester'; login.authForm.password = 'password';
    fetchHandler = async url => url.startsWith('/api/')
        ? response(404, { detail: 'missing prefix' })
        : response(200, { access_token: 'new-token', username: 'tester', role: 'user' });
    await login.handleAuthSubmit();
    result.loginFallback = { calls: fetchCalls.splice(0).map(call => call.url), token: login.token };

    const staticOrigin = bind();
    staticOrigin.token = '';
    staticOrigin.authForm.username = 'tester'; staticOrigin.authForm.password = 'password';
    fetchHandler = async url => url.startsWith('http://127.0.0.1:8000/api/')
        ? response(200, { access_token: 'backend-token', username: 'tester', role: 'user' })
        : response(501, { detail: '静态预览不支持 POST' });
    await staticOrigin.handleAuthSubmit();
    result.staticOriginFallback = {
        calls: fetchCalls.splice(0).map(call => call.url), token: staticOrigin.token,
    };

    const expired = bind();
    expired.token = 'expired'; expired.currentUser = { username: 'tester', role: 'user' };
    fetchHandler = async () => response(401, { detail: 'expired' });
    let expiredMessage = '';
    try { await expired.authFetch('/api/sessions'); } catch (error) { expiredMessage = error.message; }
    result.expired = {
        calls: fetchCalls.splice(0).map(call => call.url),
        token: expired.token, authError: expired.authError, message: expiredMessage,
    };

    const upload = bind();
    upload.token = 'upload-token'; upload.currentUser = { username: 'admin', role: 'admin' };
    xhrPlans.push({ status: 404 }, { status: 202, body: { job_id: 'job-1' } });
    let uploadResult = null; let uploadError = null;
    try { uploadResult = await upload.uploadFileWithProgress({ name: 'guide.pdf', webkitRelativePath: '资料/guide.pdf' }); }
    catch (error) { uploadError = error.message; }
    result.uploadFallback = { calls: xhrCalls, uploadResult, uploadError };

    const sessions = bind();
    sessions.token = 'secret'; sessions.currentUser = { username: 'tester', role: 'user' };
    sessions.authFetch = async () => response(503, { detail: '服务暂不可用' });
    await sessions.loadSessions();
    sessions.currentUser = { username: 'admin', role: 'admin' };
    await sessions.loadDocuments();
    result.loadingErrors = { sessionsError: sessions.sessionsError, documentsError: sessions.documentsError };

    const sources = bind();
    sources.messages = [{ isUser: false, ragTrace: { expanded_retrieved_chunks: [
        { filename: 'a.pdf', page_number: 1, text: 'one' },
        { filename: 'b.pdf', page_number: 2, text: 'two' },
        { filename: 'a.pdf', page_number: 3, text: 'three' },
    ] } }];
    sources.inspectorMessageIndex = 0;
    result.groupedSources = sources.groupedInspectorSources === undefined
        ? null
        : JSON.parse(JSON.stringify(sources.groupedInspectorSources));

    mobile = true; overlay = true;
    const layers = bind();
    layers.isMobileViewport = true; layers.isOverlayViewport = true;
    const sidebarFirst = makeFocusable('sidebar-first');
    const inspectorTab = makeFocusable('inspector-tab');
    layers.$refs = {
        mobileNavTrigger: makeFocusable('mobile-trigger'),
        mobileSidebarFirstAction: sidebarFirst,
        mobileSidebarPanel: { querySelectorAll() { return [sidebarFirst]; } },
        inspectorProcessTab: inspectorTab,
        inspectorSourcesTab: inspectorTab,
        inspectorPanel: { querySelectorAll() { return [inspectorTab]; } },
    };
    layers.openMobileSidebar();
    const afterSidebarOpen = [layers.mobileSidebarOpen, layers.mainContentInert, document.activeElement?.name];
    document.activeElement = makeFocusable('answer-trigger');
    layers.messages = [{ text: 'answer', isUser: false }];
    layers.openInspector(0, 'process');
    const afterInspectorOpen = [layers.inspectorOpen, layers.mainContentInert, layers.sidebarInert, document.activeElement?.name];
    layers.handleGlobalKeydown({ key: 'Escape' });
    const afterFirstEscape = [layers.inspectorOpen, layers.mobileSidebarOpen];
    layers.handleGlobalKeydown({ key: 'Escape' });
    const afterSecondEscape = [layers.inspectorOpen, layers.mobileSidebarOpen];
    result.layers = { afterSidebarOpen, afterInspectorOpen, afterFirstEscape, afterSecondEscape, focusEvents };

    let readResolve;
    const stream = bind();
    stream.token = 'secret'; stream.currentUser = { username: 'tester', role: 'user' };
    stream.userInput = 'first question';
    stream.authFetch = async () => ({ ok: true, status: 200, body: { getReader() { return {
        read() { return new Promise(resolve => { readResolve = resolve; }); }
    }; } } });
    const sendPromise = stream.handleSend();
    await new Promise(resolve => setImmediate(resolve));
    const oldController = stream.abortController;
    stream.handleNewChat();
    if (readResolve) {
        readResolve({ done: false, value: new TextEncoder().encode('data: {"type":"content","content":"stale"}\n\n') });
        await new Promise(resolve => setImmediate(resolve));
        if (readResolve) readResolve({ done: true });
    }
    let streamError = null;
    try { await sendPromise; } catch (error) { streamError = error.message; }
    result.streamIsolation = {
        messages: JSON.parse(JSON.stringify(stream.messages)), isLoading: stream.isLoading,
        aborted: oldController.signal.aborted, streamError,
    };

    let readCount = 0;
    const interrupted = bind();
    interrupted.token = 'secret'; interrupted.currentUser = { username: 'tester', role: 'user' };
    interrupted.userInput = 'retry me';
    interrupted.authFetch = async () => ({ ok: true, status: 200, body: { getReader() { return {
        async read() {
            readCount += 1;
            if (readCount === 1) return { done: false, value: new TextEncoder().encode('data: {"type":"content","content":"partial answer"}\n\n') };
            throw new Error('offline');
        }
    }; } } });
    await interrupted.handleSend();
    const bot = interrupted.messages.find(message => !message.isUser);
    result.interrupted = {
        text: bot?.text || null,
        streamError: bot?.streamError || null,
        retryText: bot?.retryText || null,
    };

    process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness], input=frontend_source("script.js"), text=True,
        encoding="utf-8", capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def test_api_paths_fallback_once_for_authenticated_public_and_xhr_requests():
    """A backend mounted with or without /api must receive the same request contract."""
    runtime = run_runtime_contract()
    assert runtime["authFallback"] == [
        {"url": "/api/sessions", "authorization": "Bearer secret"},
        {"url": "/sessions", "authorization": "Bearer secret"},
    ]
    assert runtime["loginFallback"] == {
        "calls": ["/api/auth/login", "/auth/login"], "token": "new-token",
    }
    assert runtime["uploadFallback"] == {
        "calls": [
            {"url": "http://127.0.0.1:8000/api/documents/upload/async", "authorization": "Bearer upload-token", "filename": "guide.pdf"},
            {"url": "http://127.0.0.1:8000/documents/upload/async", "authorization": "Bearer upload-token", "filename": "guide.pdf"},
        ],
        "uploadResult": {"job_id": "job-1"}, "uploadError": None,
    }


def test_static_preview_auth_falls_back_to_local_fastapi_origin():
    """A static preview on port 8765 must still reach the FastAPI auth API on 8000."""
    runtime = run_runtime_contract()
    assert runtime["staticOriginFallback"] == {
        "calls": [
            "/api/auth/login", "/auth/login",
            "http://127.0.0.1:8000/api/auth/login",
        ],
        "token": "backend-token",
    }


def test_unauthorized_request_stops_fallback_and_preserves_relogin_error():
    """A real 401 must not be retried against another route or erased during logout."""
    runtime = run_runtime_contract()["expired"]
    assert runtime == {
        "calls": ["/api/sessions"],
        "token": "",
        "authError": "登录已过期，请重新登录",
        "message": "登录已过期，请重新登录",
    }


def test_session_boundary_aborts_and_isolates_the_old_stream():
    """Late SSE tokens must never enter a newly-created conversation."""
    isolation = run_runtime_contract()["streamIsolation"]
    assert isolation == {"messages": [], "isLoading": False, "aborted": True, "streamError": None}


def test_interrupted_stream_preserves_partial_text_and_exposes_retry_payload():
    """A network failure must retain generated text and enough state for a retry action."""
    interrupted = run_runtime_contract()["interrupted"]
    assert interrupted["text"].startswith("partial answer")
    assert "offline" in interrupted["streamError"]
    assert interrupted["retryText"] == "retry me"


def test_loading_failures_and_source_groups_are_user_visible_state():
    """Failed lists differ from empty lists and source chunks are grouped by file."""
    runtime = run_runtime_contract()
    assert runtime["loadingErrors"] == {
        "sessionsError": "加载会话失败：服务暂不可用",
        "documentsError": "加载文档列表失败：服务暂不可用",
    }
    assert runtime["groupedSources"] == [
        {"filename": "a.pdf", "chunks": [
            {"filename": "a.pdf", "page_number": 1, "text": "one"},
            {"filename": "a.pdf", "page_number": 3, "text": "three"},
        ]},
        {"filename": "b.pdf", "chunks": [
            {"filename": "b.pdf", "page_number": 2, "text": "two"},
        ]},
    ]


def test_overlay_layers_focus_and_escape_one_level_at_a_time():
    """Mobile drawers trap attention and Escape dismisses only the topmost layer."""
    layers = run_runtime_contract()["layers"]
    assert layers["afterSidebarOpen"] == [True, True, "sidebar-first"]
    assert layers["afterInspectorOpen"] == [True, True, True, "inspector-tab"]
    assert layers["afterFirstEscape"] == [False, True]
    assert layers["afterSecondEscape"] == [False, False]


def test_templates_expose_persistent_errors_retry_actions_and_grouped_sources():
    parser = parse_frontend()
    html = frontend_source("index.html")
    assert 'v-if="authError"' in html and '@click="authError = \'\'"' in html
    assert 'v-else-if="sessionsError"' in html and '@click="loadSessions"' in html
    assert 'v-else-if="documentsError"' in html and '@click="loadDocuments"' in html
    assert 'v-for="group in groupedInspectorSources"' in html
    assert 'v-for="(chunk, index) in group.chunks"' in html
    assert '@click="retryMessage(msg)"' in html

    main = next(attrs for tag, attrs in parser.attrs
                if tag == "main" and "workspace-main" in attrs.get("class", "").split())
    assert main[":inert"] == "mainContentInert"
    assert main[":aria-hidden"] == "mainContentInert"


def test_1024_overlay_contrast_tokens_and_icon_targets_are_explicit():
    css = frontend_source("style.css")
    root = css_rules(css.split("@media (max-width: 1100px)", 1)[0])
    medium = css_rules(css_media_block(css, "max-width: 1100px"))
    assert medium[".workspace-shell.inspector-is-open"]["grid-template-columns"] == "260px minmax(0, 1fr) 0"
    assert medium[".inspector-panel"]["position"] == "fixed"
    assert root[":root"]["--text-muted"] == "var(--color-text-muted)"
    assert root[":root"]["--accent-text"] == "var(--color-accent-text)"
    assert root[".big-avatar"]["background"] == "var(--avatar-background)"
    assert root[".big-avatar"]["border"] == "1px solid var(--avatar-border)"
    for selector in (
        ".icon-btn", ".sidebar-toggle-btn", ".conversation-delete", ".attach-btn",
        ".send-btn", ".inspector-header > button",
    ):
        assert int(root[selector]["min-width"].removesuffix("px")) >= 44
        assert int(root[selector]["min-height"].removesuffix("px")) >= 44
