from html.parser import HTMLParser
from pathlib import Path
import json
import re
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"
SVG = "{http://www.w3.org/2000/svg}"


class MarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attrs = []
        self.title = []
        self.in_title = False
        self.brand_paths = []
        self.in_brand = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        self.attrs.append((tag, attrs))
        if tag == "title":
            self.in_title = True
        if tag == "symbol" and attrs.get("id") == "icon-brand":
            self.in_brand = True
        if tag == "path" and self.in_brand:
            self.brand_paths.append(attrs)

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag == "symbol":
            self.in_brand = False

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)


def frontend_source(name):
    return (FRONTEND / name).read_text(encoding="utf-8")


def parse_frontend():
    parser = MarkupParser()
    parser.feed(frontend_source("index.html"))
    return parser


def css_media_block(css, query):
    """Return a complete media block body without relying on nested-brace regexes."""
    marker = f"@media ({query})"
    marker_start = css.index(marker)
    block_start = css.index("{", marker_start)
    depth = 0
    for index in range(block_start, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[block_start + 1:index]
    raise AssertionError(f"unterminated media block: {query}")


def css_rules(css):
    """Parse flat CSS rules into selector/declaration mappings for contract tests."""
    clean_css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    rules = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", clean_css):
        declarations = {
            key: value.strip()
            for key, value in re.findall(r"([\w-]+)\s*:\s*([^;]+)", body)
        }
        for item in selector.split(","):
            rules.setdefault(item.strip(), {}).update(declarations)
    return rules


def test_starmew_brand_assets_are_valid_svg_and_linked():
    for name in ("starmew-mark.svg", "starmew-wordmark.svg", "favicon.svg"):
        root = ET.parse(FRONTEND / "assets" / name).getroot()
        assert root.tag == SVG + "svg"
        view_box = [float(value) for value in root.attrib["viewBox"].split()]
        assert len(view_box) == 4 and all(value > 0 for value in view_box[2:])
        assert root.findall(".//" + SVG + "path"), name

    parser = parse_frontend()
    icons = [attrs for tag, attrs in parser.attrs if tag == "link" and attrs.get("rel") == "icon"]
    assert len(icons) == 1
    assert icons[0]["type"] == "image/svg+xml"
    assert icons[0]["href"] == "assets/favicon.svg"
    assert "".join(parser.title).strip() == "StarMew · 星喵助手"


def test_inline_and_external_brand_share_geometry_and_theme_colors():
    mark = ET.parse(FRONTEND / "assets/starmew-mark.svg").getroot()
    wordmark = ET.parse(FRONTEND / "assets/starmew-wordmark.svg").getroot()
    mark_paths = mark.findall(".//" + SVG + "path")
    wordmark_paths = wordmark.findall(".//" + SVG + "path")
    inline_paths = parse_frontend().brand_paths
    assert len(mark_paths) == len(wordmark_paths) == len(inline_paths) == 3
    assert [path.attrib["d"] for path in mark_paths] == [path.attrib["d"] for path in wordmark_paths]
    assert [path["d"] for path in inline_paths] == [path.attrib["d"] for path in mark_paths]
    assert all(path["stroke"] == "currentColor" and path["fill"] == "none" for path in inline_paths[:2])
    assert inline_paths[2]["fill"] == "var(--brand-accent)"


def test_external_wordmark_has_both_names_and_self_contained_dark_theme():
    for name in ("starmew-mark.svg", "starmew-wordmark.svg"):
        root = ET.parse(FRONTEND / "assets" / name).getroot()
        style = root.find(SVG + "style")
        assert style is not None
        assert "prefers-color-scheme: dark" in style.text
        assert "#F5F1E8" in style.text
        paths = root.findall(".//" + SVG + "path")
        assert all(path.attrib.get("class") == "brand-ink" for path in paths[:2])
        assert all(path.attrib["stroke"] == "#1D1D1B" for path in paths[:2])

    wordmark = ET.parse(FRONTEND / "assets/starmew-wordmark.svg").getroot()
    labels = wordmark.findall(SVG + "text")
    assert [label.text for label in labels] == ["StarMew", "星喵助手"]
    assert [label.attrib["class"] for label in labels] == ["brand-label", "brand-subtitle"]


def frontend_runtime_contract():
    """Execute actual Vue option methods without mounting or calling live APIs."""
    harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
let options;
const storage = new Map();
let mobile = false;
const focusEvents = [];
const makeFocusable = name => ({
    name,
    focus() {
        focusEvents.push(name);
        document.activeElement = this;
    },
});
const mediaQuery = {
    matches: false,
    listener: null,
    removedListener: null,
    addEventListener(type, listener) {
        if (type === 'change') this.listener = listener;
    },
    removeEventListener(type, listener) {
        if (type === 'change') this.removedListener = listener;
    },
};
const mobileMediaQuery = {
    get matches() { return mobile; },
    listener: null,
    removedListener: null,
    addEventListener(type, listener) {
        if (type === 'change') this.listener = listener;
    },
    removeEventListener(type, listener) {
        if (type === 'change') this.removedListener = listener;
    },
};
const document = {
    documentElement: { dataset: {} },
    activeElement: null,
    createElement() { return { textContent: '', innerHTML: '' }; },
};
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
    Vue: { createApp(value) { options = value; return { mount() {} }; } },
    window: {
        matchMedia(query) {
            if (query === '(max-width: 680px)') return mobileMediaQuery;
            if (query === '(max-width: 1100px)') return mediaQuery;
            if (query !== '(prefers-color-scheme: dark)') throw new Error(`unexpected media query: ${query}`);
            return mediaQuery;
        },
        addEventListener() {},
        removeEventListener() {},
        setTimeout() { return 1; },
    },
    document,
    localStorage: {
        getItem(key) { return storage.has(key) ? storage.get(key) : null; },
        setItem(key, value) { storage.set(key, value); },
        removeItem(key) { storage.delete(key); },
    },
    marked: { setOptions() {}, parse(value) { return value; } },
    hljs: { getLanguage() { return false; }, highlight(value) { return { value }; } },
    fetch: async () => ({ ok: true, json: async () => ({ access_token: 'token', username: 'tester', role: 'user' }) }),
});
(async () => {
    const state = options.data();
    for (const [name, method] of Object.entries(options.methods)) {
        state[name] = method.bind(state);
    }
    for (const [name, getter] of Object.entries(options.computed)) {
        Object.defineProperty(state, name, { get: () => getter.call(state), configurable: true });
    }
    state.$nextTick = callback => callback();
    state.$refs = {
        inspectorProcessTab: makeFocusable('process-tab'),
        inspectorSourcesTab: makeFocusable('sources-tab'),
    };
    const result = { methods: Object.keys(options.methods), data: Object.keys(state) };
    if (state.toggleSidebar) {
        result.collapsed = [state.sidebarCollapsed];
        state.toggleSidebar(); result.collapsed.push(state.sidebarCollapsed);
        state.toggleSidebar(); result.collapsed.push(state.sidebarCollapsed);
    }
    if (state.formatSessionTitle) result.title = state.formatSessionTitle({ session_id: 'untitled' });
    if (state.loadSessions) {
        state.token = 'token';
        state.currentUser = { username: 'tester', role: 'user' };
        state.authFetch = async (url) => {
            result.sessionUrl = url;
            return { ok: true, json: async () => ({ sessions: [{ session_id: 'saved' }] }) };
        };
        await state.loadSessions(); result.sessions = state.sessions;
    }
    result.files = ['PDF', 'Word', 'Excel', 'unknown'].map(type => state.getFileIcon(type));
    result.deletion = ['running', 'completed', 'failed', undefined].map(status => {
        state.deleteJobs = status ? { example: { status } } : {};
        return state.getDeleteButtonIcon('example');
    });
    if (state.getThinkingIcon) result.thinking = ['🔍', '📊', '✏️', '🧠', '📝', '🔄', '⚠️', '🔧', '✅', '🤔', '▶', 'unknown'].map(icon => state.getThinkingIcon(icon));
    state.messages = [{ content: 'example' }];
    state.sessionId = 'saved-session';
    const originalSession = state.sessionId;
    state.handleClearChat();
    result.clearRequested = [state.messages.length, state.confirmation?.title];
    await state.confirmPendingAction();
    result.cleared = state.messages.length === 0 && state.sessionId !== originalSession;
    if (state.openInspector && state.closeInspector) {
        state.messages = [{
            text: 'answer',
            isUser: false,
            ragSteps: [{ label: '检索完成', detail: '命中 3 个片段' }],
            ragTrace: {
                expanded_retrieved_chunks: [{ filename: 'expanded.pdf', text: 'expanded' }],
                initial_retrieved_chunks: [{ filename: 'initial.pdf', text: 'initial' }],
                retrieved_chunks: [{ filename: 'legacy.pdf', text: 'legacy' }],
            },
        }];
        document.activeElement = makeFocusable('message-trigger');
        state.openInspector(0, 'sources');
        const expandedSources = state.inspectorSources;
        const opened = [state.inspectorOpen, state.inspectorTab, state.inspectorMessageIndex,
            state.inspectorMessage.text];
        state.messages[0].ragTrace.expanded_retrieved_chunks = [];
        const initialSources = state.inspectorSources;
        state.messages[0].ragTrace.initial_retrieved_chunks = [];
        const legacySources = state.inspectorSources;
        state.messages[0].ragTrace = null;
        const emptySources = state.inspectorSources;
        state.openInspector(0, 'unexpected');
        const normalizedTab = state.inspectorTab;
        state.focusInspectorTab('process');
        state.closeInspector();
        result.inspector = {
            opened,
            expandedSources,
            initialSources,
            legacySources,
            emptySources,
            normalizedTab,
            closed: [state.inspectorOpen, state.inspectorMessageIndex],
            focusEvents: [...focusEvents],
        };
    }
    if (state.switchWorkspace && state.closeMobilePanels) {
        const snapshots = [];
        const requests = [];
        state.$refs = {};
        state.$nextTick = callback => callback();
        state.authFetch = async url => {
            requests.push(url);
            return { ok: true, json: async () => url === '/api/documents'
                ? { documents: [{ filename: 'guide.pdf' }] }
                : url === '/api/documents/upload/jobs'
                    ? []
                : { messages: [{ content: 'saved answer', type: 'ai', rag_trace: null }] } };
        };
        const capture = () => snapshots.push([state.activeWorkspace, state.mobileSidebarOpen]);
        capture();
        state.mobileSidebarOpen = true;
        state.messages = [{ text: 'answer', isUser: false, ragTrace: { retrieved_chunks: [] } }];
        state.openInspector(0, 'sources');
        state.switchWorkspace('settings'); capture();
        const inspectorAfterWorkspaceSwitch = [state.inspectorOpen, state.inspectorMessageIndex,
            state.mobileOverlayVisible];
        state.switchWorkspace('invalid'); capture();
        const toastCountBeforeDeny = state.toasts.length;
        state.switchWorkspace('documents'); capture();
        const deniedRequests = requests.length;
        const deniedToasts = state.toasts.slice(toastCountBeforeDeny);
        state.currentUser = { username: 'admin', role: 'admin' };
        state.switchWorkspace('documents'); capture();
        await new Promise(resolve => setTimeout(resolve, 0));
        const loadedDocuments = state.documents;
        state.mobileSidebarOpen = true;
        state.messages = [{ text: 'answer', isUser: false, ragTrace: { retrieved_chunks: [] } }];
        state.openInspector(0, 'sources');
        state.handleNewChat(); capture();
        const inspectorAfterNew = [state.inspectorOpen, state.inspectorMessageIndex];
        state.switchWorkspace('settings');
        state.openInspector(0, 'process');
        await state.loadSession('saved/session'); capture();
        const inspectorAfterLoad = [state.inspectorOpen, state.inspectorMessageIndex];
        const loadedMessages = state.messages;
        state.activeWorkspace = 'settings';
        state.authForm.username = 'tester'; state.authForm.password = 'password';
        await state.handleAuthSubmit(); capture();
        mobile = true;
        state.toggleSidebar();
        const mobileOpened = [state.mobileSidebarOpen, state.sidebarCollapsed, state.mobileOverlayVisible];
        state.inspectorOpen = true;
        state.closeMobilePanels();
        const mobileClosed = [state.mobileSidebarOpen, state.inspectorOpen, state.mobileOverlayVisible];
        state.openInspector(0, 'sources');
        state.handleLogout();
        const inspectorAfterLogout = [state.inspectorOpen, state.inspectorMessageIndex];
        mobile = false;
        result.workspace = { snapshots, deniedRequests, deniedToasts, requests, loadedDocuments,
            loadedMessages, mobileOpened, mobileClosed, inspectorAfterNew, inspectorAfterLoad,
            inspectorAfterLogout, inspectorAfterWorkspaceSwitch };
    }
    if (state.applyTheme) {
        state.token = '';
        state.currentUser = null;
        result.theme = {
            initialPreference: state.themePreference,
            initialResolved: state.resolvedTheme,
        };
        await options.mounted.call(state);
        result.theme.applied = document.documentElement.dataset.theme;
        result.theme.listenerRegistered = typeof mediaQuery.listener === 'function';
        mediaQuery.listener({ matches: true });
        result.theme.systemChange = document.documentElement.dataset.theme;
        state.setThemePreference('light');
        result.theme.explicitLight = document.documentElement.dataset.theme;
        result.theme.storedLight = storage.get('starmew-theme');
        state.setThemePreference('invalid');
        result.theme.invalidPreference = state.themePreference;
        result.theme.invalidResolved = document.documentElement.dataset.theme;
        options.beforeUnmount.call(state);
        result.theme.listenerRemoved = mediaQuery.removedListener === mediaQuery.listener;

        storage.set('starmew-theme', 'dark');
        mediaQuery.matches = false;
        const restored = options.data();
        for (const [name, method] of Object.entries(options.methods)) {
            restored[name] = method.bind(restored);
        }
        for (const [name, getter] of Object.entries(options.computed)) {
            Object.defineProperty(restored, name, { get: () => getter.call(restored), configurable: true });
        }
        result.theme.restoredPreference = restored.themePreference;
        result.theme.restoredResolved = restored.resolvedTheme;

        storage.set('starmew-theme', 'sepia');
        const invalidStored = options.data();
        result.theme.invalidStoredPreference = invalidStored.themePreference;
    }
    process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness], input=frontend_source("script.js"),
        text=True, encoding="utf-8", capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def feedback_runtime_contract():
    """Execute feedback and destructive-action methods against deterministic fakes."""
    harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
let options;
const storage = new Map();
const toastTimers = [];
const requests = [];
const focusEvents = [];
const makeFocusable = name => ({
    name,
    focus() {
        focusEvents.push(name);
        document.activeElement = this;
    },
});
const window = {
    matchMedia() { return { matches: false, addEventListener() {}, removeEventListener() {} }; },
    setTimeout(callback, delay) { toastTimers.push({ callback, delay }); return toastTimers.length; },
};
const document = {
    documentElement: { dataset: {} }, activeElement: null,
    createElement() { return { textContent: '', innerHTML: '' }; },
};
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
    Vue: { createApp(value) { options = value; return { mount() {} }; } },
    window,
    document,
    localStorage: {
        getItem(key) { return storage.has(key) ? storage.get(key) : null; },
        setItem(key, value) { storage.set(key, value); },
        removeItem(key) { storage.delete(key); },
    },
    marked: { setOptions() {}, parse(value) { return value; } },
    hljs: { getLanguage() { return false; }, highlight(value) { return { value }; } },
    fetch: async () => ({ ok: true, json: async () => ({}) }),
    setTimeout() { throw new Error('unexpected global timeout'); },
    crypto: { randomUUID() { return 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'; } },
    clearTimeout() {}, setInterval() { return 7; }, clearInterval() {},
});

const bindState = () => {
    const state = options.data();
    for (const [name, method] of Object.entries(options.methods)) state[name] = method.bind(state);
    for (const [name, getter] of Object.entries(options.computed)) {
        Object.defineProperty(state, name, { get: () => getter.call(state), configurable: true });
    }
    state.$nextTick = callback => callback();
    state.$refs = {
        confirmCancelButton: makeFocusable('confirm-cancel'),
        confirmActionButton: makeFocusable('confirm-action'),
    };
    state.withUploadRequestTimeout = operation => operation(undefined);
    state.closeMobilePanels = () => {};
    return state;
};

(async () => {
    const state = bindState();
    const result = { data: Object.keys(state), methods: Object.keys(options.methods) };

    state.showToast('上传完成', 'success');
    result.toastAdded = JSON.parse(JSON.stringify(state.toasts));
    result.toastDelay = toastTimers[0].delay;
    toastTimers[0].callback();
    result.toastAfterTimeout = JSON.parse(JSON.stringify(state.toasts));

    let asyncActionState = 'idle';
    document.activeElement = makeFocusable('confirmation-trigger');
    state.requestConfirmation({
        title: '执行操作？', message: '需要确认',
        onConfirm: async () => {
            asyncActionState = state.confirmation === null ? 'started-after-close' : 'started-before-close';
            await Promise.resolve();
            asyncActionState = 'completed';
        },
    });
    result.confirmationDefaults = {
        title: state.confirmation.title,
        message: state.confirmation.message,
        confirmLabel: state.confirmation.confirmLabel,
        tone: state.confirmation.tone,
    };
    result.initialConfirmationFocus = document.activeElement.name;
    const focusPrevented = [];
    state.trapConfirmationFocus({ shiftKey: false, preventDefault() { focusPrevented.push('forward'); } });
    state.trapConfirmationFocus({ shiftKey: false, preventDefault() { focusPrevented.push('wrap'); } });
    state.trapConfirmationFocus({ shiftKey: true, preventDefault() { focusPrevented.push('backward'); } });
    result.confirmationTabCycle = {
        events: focusEvents.slice(-4),
        prevented: focusPrevented,
        active: document.activeElement.name,
    };
    const confirmPromise = state.confirmPendingAction();
    result.confirmationClearedSynchronously = state.confirmation === null;
    result.focusAfterConfirm = document.activeElement.name;
    result.actionAtConfirm = asyncActionState;
    await confirmPromise;
    result.actionAfterAwait = asyncActionState;

    let cancelledActionRan = false;
    document.activeElement = makeFocusable('cancel-trigger');
    state.requestConfirmation({ title: '取消？', message: '不会执行', onConfirm: () => { cancelledActionRan = true; } });
    state.cancelPendingAction();
    result.cancelled = [state.confirmation, cancelledActionRan, document.activeElement.name];

    state.messages = [{ text: '保留到确认', isUser: true }];
    state.token = 'token'; state.currentUser = { username: 'tester', role: 'user' };
    state.handleClearChat();
    result.clearBeforeConfirm = [state.messages.length, state.confirmation.title, state.confirmation.confirmLabel];
    await state.confirmPendingAction();
    result.clearAfterConfirm = state.messages.length;

    state.sessions = [{ session_id: 'saved' }, { session_id: 'other' }];
    state.sessionId = 'saved';
    state.messages = [{ text: 'saved answer', isUser: false }];
    state.authFetch = async (url, init = {}) => {
        requests.push([url, init.method || 'GET']);
        return { ok: true, json: async () => ({ message: '会话已删除' }) };
    };
    state.deleteSession('saved');
    result.sessionBeforeConfirm = [requests.length, state.sessions.length, state.confirmation.title];
    await state.confirmPendingAction();
    result.sessionAfterConfirm = {
        requests: JSON.parse(JSON.stringify(requests)),
        sessions: JSON.parse(JSON.stringify(state.sessions)),
        messages: JSON.parse(JSON.stringify(state.messages)),
        resetSession: state.sessionId !== 'saved',
        toast: state.toasts.at(-1),
    };

    const failedSession = bindState();
    failedSession.sessions = [{ session_id: 'blocked' }];
    failedSession.authFetch = async () => ({
        ok: false,
        json: async () => ({ detail: '无权删除会话' }),
    });
    failedSession.deleteSession('blocked');
    await failedSession.confirmPendingAction();
    result.sessionFailure = {
        sessions: JSON.parse(JSON.stringify(failedSession.sessions)),
        toast: failedSession.toasts.at(-1),
    };

    requests.length = 0;
    state.documents = [{ filename: 'guide.pdf' }];
    state.deleteJobs = {};
    state.startDeleteJobPolling = (filename, jobId) => { result.pollStarted = [filename, jobId]; };
    state.postUploadAction = async (url, method) => {
        requests.push([url, method]);
        return { job_id: 'a'.repeat(32) };
    };
    state.authFetch = async (url, init = {}) => {
        requests.push([url, init.method || 'GET']);
        return { ok: true, json: async () => ({ job_id: 'a'.repeat(32), status: 'running', message: '删除任务已提交' }) };
    };
    state.deleteDocument('guide.pdf');
    result.documentBeforeConfirm = [requests.length, state.confirmation.title, state.deleteJobs['guide.pdf'] || null];
    await state.confirmPendingAction();
    result.documentAfterConfirm = {
        requests: JSON.parse(JSON.stringify(requests)),
        job: JSON.parse(JSON.stringify(state.deleteJobs['guide.pdf'])),
        pollStarted: result.pollStarted,
    };

    const failed = bindState();
    failed.postUploadAction = async () => {
        const error = new Error('服务拒绝删除');
        error.isUploadRequestRejected = true;
        throw error;
    };
    failed.deleteDocument('broken.pdf');
    await failed.confirmPendingAction();
    result.documentFailure = {
        job: JSON.parse(JSON.stringify(failed.deleteJobs['broken.pdf'])),
        toast: failed.toasts.at(-1),
    };

    const completedDelete = bindState();
    completedDelete.deleteJobs = { 'done.pdf': { jobId: 'delete-complete', status: 'running', steps: completedDelete.createDeleteSteps() } };
    completedDelete.authFetch = async () => ({
        ok: true,
        json: async () => ({ job_id: 'delete-complete', status: 'completed', message: '文档删除完成', steps: [] }),
    });
    completedDelete.stopDeleteJobPolling = () => {};
    completedDelete.scheduleDeletedDocumentRemoval = filename => { result.removalScheduled = filename; };
    completedDelete.startDeleteJobPolling('done.pdf', 'delete-complete');
    await new Promise(resolve => setTimeout(resolve, 0));
    result.documentCompletion = {
        removalScheduled: result.removalScheduled,
        toast: completedDelete.toasts.at(-1),
    };

    const completedUpload = bindState();
    completedUpload.isUploading = true;
    completedUpload.selectedFiles = [{ name: 'done.pdf' }];
    completedUpload.authFetch = async () => ({
        ok: true,
        json: async () => ({ job_id: 'upload-complete', status: 'completed', message: '文档上传完成', steps: [] }),
    });
    completedUpload.stopUploadJobPolling = () => {};
    completedUpload.loadDocuments = async () => { result.uploadReloaded = true; };
    completedUpload.startUploadJobPolling('upload-complete');
    await new Promise(resolve => setTimeout(resolve, 0));
    result.uploadCompletion = {
        idle: !completedUpload.isUploading,
        selectedFiles: completedUpload.selectedFiles,
        reloaded: result.uploadReloaded,
        toast: completedUpload.toasts.at(-1),
    };

    const validation = bindState();
    validation.uploadDocument();
    validation.switchWorkspace('documents');
    result.validationToasts = JSON.parse(JSON.stringify(validation.toasts));
    result.deleteTaskId = state.getDeleteTaskStepsId('guide v2.pdf');

    process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness], input=frontend_source("script.js"),
        text=True, encoding="utf-8", capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def keyboard_runtime_contract():
    """Exercise the real global-key method and component lifecycle with DOM fakes."""
    harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
let options;
const listeners = new Map();
const removedListeners = [];
const focusEvents = [];
const makeFocusable = name => ({
    name,
    focus() {
        focusEvents.push(name);
        document.activeElement = this;
    },
});
const themeMediaQuery = {
    matches: false,
    addEventListener() {},
    removeEventListener() {},
};
const mobileMediaQuery = {
    matches: true,
    listener: null,
    removedListener: null,
    addEventListener(type, listener) {
        if (type === 'change') this.listener = listener;
    },
    removeEventListener(type, listener) {
        if (type === 'change') this.removedListener = listener;
    },
};
const window = {
    matchMedia(query) {
        if (query === '(prefers-color-scheme: dark)') return themeMediaQuery;
        if (query === '(max-width: 680px)') return mobileMediaQuery;
        if (query === '(max-width: 1100px)') return themeMediaQuery;
        throw new Error(`unexpected media query: ${query}`);
    },
    addEventListener(type, listener) { listeners.set(type, listener); },
    removeEventListener(type, listener) {
        removedListeners.push([type, listener]);
        if (listeners.get(type) === listener) listeners.delete(type);
    },
    setTimeout() { return 1; },
};
const document = {
    documentElement: { dataset: {} },
    activeElement: null,
    createElement() { return { textContent: '', innerHTML: '' }; },
};
vm.runInNewContext(fs.readFileSync(0, 'utf8'), {
    Vue: { createApp(value) { options = value; return { mount() {} }; } },
    window,
    document,
    localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
    marked: { setOptions() {}, parse(value) { return value; } },
    hljs: { getLanguage() { return false; }, highlight(value) { return { value }; } },
    fetch: async () => ({ ok: true, json: async () => ({}) }),
    clearTimeout() {},
    clearInterval() {},
});

(async () => {
    const state = options.data();
    for (const [name, method] of Object.entries(options.methods)) state[name] = method.bind(state);
    for (const [name, getter] of Object.entries(options.computed)) {
        Object.defineProperty(state, name, { get: () => getter.call(state), configurable: true });
    }
    state.$nextTick = callback => callback();
    state.$refs = { mobileNavTrigger: makeFocusable('mobile-nav-trigger') };

    await options.mounted.call(state);
    const registered = listeners.get('keydown');
    const result = {
        methodPresent: typeof state.handleGlobalKeydown === 'function',
        listenerRegistered: registered === state.handleGlobalKeydown,
    };

    state.sidebarCollapsed = true;
    state.openMobileSidebar();
    result.mobileSidebarOpened = [state.mobileSidebarOpen, state.sidebarCollapsed,
        state.mobileSidebarHidden];
    state.closeMobilePanels();
    result.mobileSidebarClosed = [state.mobileSidebarOpen, state.mobileSidebarHidden,
        document.activeElement?.name];
    state.openMobileSidebar();
    state.switchWorkspace('settings');
    result.mobileSidebarWorkspaceClose = [state.mobileSidebarOpen, state.mobileSidebarHidden,
        document.activeElement?.name];
    result.mobileListenerRegistered = typeof mobileMediaQuery.listener === 'function';
    if (mobileMediaQuery.listener) mobileMediaQuery.listener({ matches: false });
    result.desktopSidebarHidden = state.mobileSidebarHidden;

    state.confirmation = { title: '确认', onConfirm() {} };
    state.confirmationTrigger = makeFocusable('confirmation-trigger');
    state.mobileSidebarOpen = true;
    state.inspectorOpen = true;
    registered({ key: 'Enter' });
    result.nonEscape = [state.confirmation !== null, state.mobileSidebarOpen, state.inspectorOpen];
    registered({ key: 'Escape' });
    result.confirmationEscape = [state.confirmation, state.mobileSidebarOpen, state.inspectorOpen,
        document.activeElement?.name];

    state.mobileSidebarOpen = false;
    state.inspectorOpen = true;
    state.inspectorTrigger = makeFocusable('inspector-trigger');
    registered({ key: 'Escape' });
    result.inspectorEscape = [state.inspectorOpen, state.mobileSidebarOpen,
        document.activeElement?.name];

    state.mobileSidebarOpen = true;
    registered({ key: 'Escape' });
    result.sidebarEscape = [state.mobileSidebarOpen, state.inspectorOpen];

    options.beforeUnmount.call(state);
    result.listenerRemoved = !listeners.has('keydown')
        && removedListeners.some(([type, listener]) => type === 'keydown' && listener === registered);
    result.mobileListenerRemoved = mobileMediaQuery.removedListener === mobileMediaQuery.listener;
    process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", harness], input=frontend_source("script.js"),
        text=True, encoding="utf-8", capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def test_theme_contract_supports_system_light_and_dark():
    parser = parse_frontend()
    css = re.sub(
        r"/\*.*?\*/", "",
        frontend_source("style.css").split("@media (max-width: 1100px)", 1)[0],
        flags=re.S,
    )
    rules = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        declarations = dict(re.findall(r"(--[\w-]+|[\w-]+)\s*:\s*([^;]+)", body))
        for item in selector.split(","):
            rules[item.strip()] = {key: value.strip() for key, value in declarations.items()}

    html_attrs = next(attrs for tag, attrs in parser.attrs if tag == "html")
    assert html_attrs.get("data-theme") == "light"
    for selector in (':root', '[data-theme="light"]', '[data-theme="dark"]'):
        assert selector in rules
    for variable in (
        "--color-bg", "--color-panel", "--color-text", "--color-accent-hover",
        "--color-on-accent", "--color-on-danger",
    ):
        assert variable in rules['[data-theme="light"]']
        assert variable in rules['[data-theme="dark"]']

    def relative_luminance(hex_color):
        channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    for selector in ('[data-theme="light"]', '[data-theme="dark"]'):
        for background_variable, foreground_variable in (
            ("--color-accent", "--color-on-accent"),
            ("--color-accent-hover", "--color-on-accent"),
            ("--color-danger", "--color-on-danger"),
        ):
            background = relative_luminance(rules[selector][background_variable])
            foreground = relative_luminance(rules[selector][foreground_variable])
            contrast = (max(background, foreground) + 0.05) / (min(background, foreground) + 0.05)
            assert contrast >= 4.5

    assert rules[".workspace-header"]["background"] == "var(--color-surface)"
    assert rules[".source-card"]["background"] == "var(--color-surface)"
    assert rules[".upload-step-bar"]["background"] == "var(--color-surface)"
    assert rules[".message-user .message-body"]["color"] == "var(--color-text)"
    assert rules[".welcome-eyebrow"]["color"] == "var(--color-text-secondary)"
    for selector in (".prompt-suggestions button:hover", ".message-actions button:hover"):
        assert rules[selector]["color"] == "var(--color-text)"
    inline_code = rules[".message-user .message-body :not(pre) > code"]
    assert inline_code["color"] == "var(--color-text)"

    for selector in ('[data-theme="light"]', '[data-theme="dark"]'):
        theme = rules[selector]
        for background_variable, foreground_variable in (
            ("--color-surface", "--color-text-secondary"),
            ("--color-elevated", "--color-text"),
            ("--color-accent-soft", "--color-text"),
        ):
            background = relative_luminance(theme[background_variable])
            foreground = relative_luminance(theme[foreground_variable])
            contrast = (max(background, foreground) + 0.05) / (min(background, foreground) + 0.05)
            assert contrast >= 4.5
    for selector in (".send-btn", ".upload-btn", ".btn-primary"):
        assert rules[selector]["color"] == "var(--color-on-accent)"
    assert rules[".stop-btn"]["color"] == "var(--color-on-danger)"

    theme_selects = [
        attrs for tag, attrs in parser.attrs
        if tag == "select" and attrs.get("aria-label") == "界面主题"
    ]
    assert len(theme_selects) == 1
    assert theme_selects[0].get(":value") == "themePreference"
    assert theme_selects[0].get("@change") == "setThemePreference($event.target.value)"
    theme_values = {
        attrs.get("value") for tag, attrs in parser.attrs
        if tag == "option" and attrs.get("value") in {"system", "light", "dark"}
    }
    assert theme_values == {"system", "light", "dark"}

    runtime = frontend_runtime_contract()
    assert {"setThemePreference", "syncSystemTheme", "applyTheme"} <= set(runtime["methods"])
    assert {"themePreference", "systemThemeDark", "themeMediaQuery"} <= set(runtime["data"])
    assert runtime["theme"] == {
        "initialPreference": "system",
        "initialResolved": "light",
        "applied": "light",
        "listenerRegistered": True,
        "systemChange": "dark",
        "explicitLight": "light",
        "storedLight": "light",
        "invalidPreference": "system",
        "invalidResolved": "dark",
        "listenerRemoved": True,
        "restoredPreference": "dark",
        "restoredResolved": "dark",
        "invalidStoredPreference": "system",
    }


def test_frontend_contract_sidebar_and_sessions_bind_to_working_methods():
    runtime = frontend_runtime_contract()
    parser = parse_frontend()
    handlers = {attrs.get("@click") for _, attrs in parser.attrs}
    assert {"toggleSidebar", "loadSessions"} <= handlers
    assert {"toggleSidebar", "loadSessions", "formatSessionTitle", "getThinkingIcon"} <= set(runtime["methods"])
    assert "sidebarCollapsed" in runtime["data"]
    assert runtime["collapsed"] == [False, True, False]
    assert runtime["title"] == "untitled"
    assert runtime["sessionUrl"] == "/api/sessions"
    assert runtime["sessions"] == [{"session_id": "saved"}]


def test_frontend_contract_dynamic_icons_resolve_to_existing_symbols():
    runtime = frontend_runtime_contract()
    symbols = {attrs["id"] for tag, attrs in parse_frontend().attrs if tag == "symbol"}
    assert runtime["files"] == ["file-pdf", "file-word", "file-excel", "file"]
    assert runtime["deletion"] == ["spinner", "check", "trash", "trash"]
    assert runtime.get("thinking")
    for icon in runtime["files"] + runtime["deletion"] + runtime["thinking"]:
        assert "icon-" + icon in symbols


def test_frontend_contract_sprite_and_display_icons_have_layout_rules():
    css = re.sub(r"/\*.*?\*/", "", frontend_source("style.css"), flags=re.S)
    rules = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        declarations = dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", body))
        for item in selector.split(","):
            rules[item.strip()] = {key: value.strip() for key, value in declarations.items()}
    assert rules.get(".icon-sprite", {}).get("width") == "0"
    assert rules[".icon-sprite"]["height"] == "0"
    assert rules[".icon-sprite"]["overflow"] == "hidden"
    assert rules.get(".ui-icon", {}).get("width") not in (None, "0")
    assert rules[".ui-icon"]["height"] not in (None, "0")
    assert any(".brand-mark" in selector and "width" in rule and "height" in rule for selector, rule in rules.items())


def test_frontend_contract_clear_chat_remains_reachable():
    buttons = [attrs for tag, attrs in parse_frontend().attrs if tag == "button" and attrs.get("@click") == "handleClearChat"]
    assert len(buttons) == 1
    assert buttons[0].get("aria-label") == "开始新对话"
    runtime = frontend_runtime_contract()
    assert runtime["clearRequested"] == [1, "开始新对话？"]
    assert runtime["cleared"]


def test_frontend_contract_collapsed_logout_has_accessible_name():
    buttons = [attrs for tag, attrs in parse_frontend().attrs if tag == "button" and attrs.get("@click") == "handleLogout"]
    assert len(buttons) == 1
    assert buttons[0].get("aria-label") == "退出登录"
    assert buttons[0].get("title") == "退出登录"


def test_workspace_has_sidebar_main_and_inspector_regions():
    """Catch missing semantic regions, navigation wiring, or conditional view gates."""
    parser = parse_frontend()
    def region(class_name):
        matches = [(tag, attrs) for tag, attrs in parser.attrs
                   if class_name in attrs.get("class", "").split()]
        assert len(matches) == 1, f"missing or duplicate {class_name}"
        return matches[0]

    shell_tag, shell = region("workspace-shell")
    assert shell_tag == "div" and "id" not in shell
    assert region("workspace-sidebar")[0] == "aside"
    assert region("workspace-sidebar")[1]["aria-label"] == "主导航"
    assert region("workspace-main")[0] == "main"
    assert region("inspector-panel")[0] == "aside"
    assert region("inspector-panel")[1][":aria-hidden"] == "!inspectorOpen"
    for name, workspace, directive in (("chat-workspace", "chat", "v-show"),
                                       ("document-library", "documents", "v-if"),
                                       ("settings-view", "settings", "v-if")):
        tag, attrs = region(name)
        assert tag == "section"
        assert "isAuthenticated" in attrs[directive]
        assert f"activeWorkspace === '{workspace}'" in attrs[directive]
    nav_buttons = {attrs.get("@click"): attrs for tag, attrs in parser.attrs if tag == "button"}
    for workspace in ("chat", "documents", "settings"):
        assert f"switchWorkspace('{workspace}')" in nav_buttons
    assert nav_buttons["switchWorkspace('documents')"]["v-if"] == "isAdmin"
    assert "v-if" not in nav_buttons["switchWorkspace('settings')"]
    assert region("conversation-list")[1]["aria-label"] == "历史会话"
    assert region("mobile-scrim")[1]["@click"] == "closeMobilePanels"


def test_mobile_navigation_trigger_is_global_across_authenticated_workspaces():
    """Catch mobile users becoming trapped after leaving the chat workspace."""
    parser = parse_frontend()
    triggers = [attrs for tag, attrs in parser.attrs
                if tag == "button" and "mobile-nav-toggle" in attrs.get("class", "").split()]
    assert triggers == [{
        "v-if": "isAuthenticated",
        "ref": "mobileNavTrigger",
        "class": "mobile-nav-toggle icon-btn",
        "@click": "openMobileSidebar",
        "aria-label": "打开主导航",
        ":aria-expanded": "mobileSidebarOpen",
    }]

    mobile_css = css_media_block(frontend_source("style.css"), "max-width: 680px")
    match = re.search(r"\.workspace-main\s*>\s*\.mobile-nav-toggle\s*\{([^}]*)\}", mobile_css)
    assert match
    declarations = dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", match.group(1)))
    assert declarations["display"].strip() == "flex"
    assert declarations["position"].strip() == "absolute"


def test_chat_is_document_flow_with_external_rag_inspector():
    """Catch regressions back to chat bubbles or inline RAG details."""
    parser = parse_frontend()

    def region(class_name):
        matches = [(tag, attrs) for tag, attrs in parser.attrs
                   if class_name in attrs.get("class", "").split()]
        assert len(matches) == 1, f"missing or duplicate {class_name}"
        return matches[0]

    assert region("conversation-document")[0] == "div"
    assert region("composer-shell")[0] == "div"
    assert region("run-status") == (
        "div", {"v-if": "!msg.isUser && msg.isThinking && !msg.text",
                "class": "run-status", "role": "status", "aria-live": "polite"},
    )

    actions = {attrs.get("@click") for tag, attrs in parser.attrs if tag == "button"}
    assert "openInspector(index, 'process')" in actions
    assert "openInspector(index, 'sources')" in actions
    assert '<details class="reasoning-details">' not in frontend_source("index.html")


def test_rag_inspector_has_accessible_tabs_and_external_content_regions():
    """Catch inaccessible inspector tabs or loss of process/source presentation."""
    parser = parse_frontend()
    tablists = [attrs for tag, attrs in parser.attrs
                if tag == "div" and attrs.get("role") == "tablist"]
    assert tablists == [{"role": "tablist", "aria-label": "回答详情"}]

    tabs = {attrs["id"]: attrs for tag, attrs in parser.attrs
            if tag == "button" and attrs.get("role") == "tab"}
    assert set(tabs) == {"inspector-process-tab", "inspector-sources-tab"}
    assert tabs["inspector-process-tab"] == {
        "id": "inspector-process-tab", "ref": "inspectorProcessTab", "role": "tab",
        ":aria-selected": "inspectorTab === 'process'", "aria-controls": "inspector-process-panel",
        ":tabindex": "inspectorTab === 'process' ? 0 : -1",
        "@click": "inspectorTab = 'process'", "@keydown.right.prevent": "focusInspectorTab('sources')",
    }
    assert tabs["inspector-sources-tab"] == {
        "id": "inspector-sources-tab", "ref": "inspectorSourcesTab", "role": "tab",
        ":aria-selected": "inspectorTab === 'sources'", "aria-controls": "inspector-sources-panel",
        ":tabindex": "inspectorTab === 'sources' ? 0 : -1",
        "@click": "inspectorTab = 'sources'", "@keydown.left.prevent": "focusInspectorTab('process')",
    }
    inspector = next(attrs for tag, attrs in parser.attrs
                     if tag == "aside" and "inspector-panel" in attrs.get("class", "").split())
    assert inspector[":inert"] == "!inspectorOpen"
    panels = {attrs.get("id"): attrs for _, attrs in parser.attrs if attrs.get("role") == "tabpanel"}
    assert panels["inspector-process-panel"]["aria-labelledby"] == "inspector-process-tab"
    assert panels["inspector-sources-panel"]["aria-labelledby"] == "inspector-sources-tab"
    assert any("process-timeline" in attrs.get("class", "").split()
               for _, attrs in parser.attrs)
    assert any("source-list" in attrs.get("class", "").split()
               for _, attrs in parser.attrs)
    assert any(tag == "details" and "advanced-trace" in attrs.get("class", "").split()
               for tag, attrs in parser.attrs)
    close_buttons = [attrs for tag, attrs in parser.attrs
                     if tag == "button" and attrs.get("@click") == "closeInspector"]
    assert close_buttons == [{"@click": "closeInspector", "aria-label": "关闭回答详情"}]


def test_rag_inspector_normalizes_sources_and_open_close_state():
    """Catch wrong source priority, tab normalization, or inspector state transitions."""
    runtime = frontend_runtime_contract()
    assert {"inspectorMessageIndex", "inspectorTab", "inspectorOpen", "inspectorTrigger"} <= set(runtime["data"])
    assert {"openInspector", "closeInspector", "focusInspectorTab"} <= set(runtime["methods"])
    inspector = runtime["inspector"]
    assert inspector["opened"] == [True, "sources", 0, "answer"]
    assert inspector["expandedSources"] == [{"filename": "expanded.pdf", "text": "expanded"}]
    assert inspector["initialSources"] == [{"filename": "initial.pdf", "text": "initial"}]
    assert inspector["legacySources"] == [{"filename": "legacy.pdf", "text": "legacy"}]
    assert inspector["emptySources"] == []
    assert inspector["normalizedTab"] == "process"
    assert inspector["closed"] == [False, 0]
    assert inspector["focusEvents"] == ["sources-tab", "process-tab", "process-tab", "message-trigger"]


def test_rag_inspector_selection_resets_for_new_loaded_and_logged_out_sessions():
    """Catch stale inspector content leaking across conversation boundaries."""
    workspace = frontend_runtime_contract()["workspace"]
    assert workspace["inspectorAfterNew"] == [False, None]
    assert workspace["inspectorAfterLoad"] == [False, None]
    assert workspace["inspectorAfterLogout"] == [False, None]
    assert workspace["inspectorAfterWorkspaceSwitch"] == [False, None, False]


def test_workspace_grid_reserves_main_space_for_collapsed_and_inspector_states():
    css = re.sub(
        r"/\*.*?\*/", "",
        frontend_source("style.css").split("@media (max-width: 1100px)", 1)[0],
        flags=re.S,
    )
    rules = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        declarations = dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", body))
        for item in selector.split(","):
            rules.setdefault(item.strip(), {}).update({key: value.strip() for key, value in declarations.items()})
    assert rules.get(".workspace-shell", {}).get("display") == "grid"
    for selector, columns in (
        (".workspace-shell", "260px minmax(0, 1fr) 0"),
        (".workspace-shell.sidebar-is-collapsed", "56px minmax(0, 1fr) 0"),
        (".workspace-shell.inspector-is-open", "260px minmax(0, 1fr) 360px"),
        (".workspace-shell.sidebar-is-collapsed.inspector-is-open", "56px minmax(0, 1fr) 360px"),
    ):
        assert rules[selector]["grid-template-columns"] == columns
    for selector in (".workspace-sidebar", ".workspace-main", ".inspector-panel"):
        assert rules[selector]["min-width"] == "0"
    assert rules[".workspace-shell"]["height"] == "100dvh"


def test_workspace_shell_binding_is_inside_the_vue_mount_host():
    """Vue must compile the shell class binding so collapse changes the grid track."""
    parser = parse_frontend()
    mount = next(attrs for tag, attrs in parser.attrs if tag == "div" and attrs.get("id") == "app")
    shell = next(attrs for tag, attrs in parser.attrs
                 if tag == "div" and "workspace-shell" in attrs.get("class", "").split())

    assert "workspace-shell" not in mount.get("class", "").split()
    assert shell[":class"] == "{ 'sidebar-is-collapsed': sidebarCollapsed, 'inspector-is-open': inspectorOpen, 'mobile-sidebar-is-open': mobileSidebarOpen }"


def test_workspace_navigation_enforces_permissions_and_returns_sessions_to_chat():
    runtime = frontend_runtime_contract()
    assert {"activeWorkspace", "mobileSidebarOpen", "inspectorOpen"} <= set(runtime["data"])
    assert {"switchWorkspace", "closeMobilePanels"} <= set(runtime["methods"])
    workspace = runtime["workspace"]
    assert workspace["snapshots"] == [
        ["chat", False], ["settings", False], ["settings", False], ["settings", False],
        ["documents", False], ["chat", False], ["chat", False], ["chat", False],
    ]
    assert workspace["deniedRequests"] == 0
    assert len(workspace["deniedToasts"]) == 1
    assert workspace["deniedToasts"][0]["tone"] == "warning"
    assert "管理员" in workspace["deniedToasts"][0]["message"]
    assert workspace["requests"] == [
        "/api/documents", "/api/documents/upload/jobs", "/api/documents/delete/jobs", "/api/sessions/saved%2Fsession",
    ]
    assert workspace["loadedDocuments"] == [{"filename": "guide.pdf"}]
    assert workspace["loadedMessages"] == [{"text": "saved answer", "isUser": False, "ragTrace": None}]
    assert workspace["mobileOpened"] == [True, False, True]
    assert workspace["mobileClosed"] == [False, False, False]


def test_documents_settings_and_feedback_are_separate_views():
    """Catch merged settings/documents UI or regressions to ad-hoc progress and native dialogs."""
    parser = parse_frontend()
    source = frontend_source("script.js")

    def region(class_name):
        matches = [(tag, attrs) for tag, attrs in parser.attrs
                   if class_name in attrs.get("class", "").split()]
        assert matches, f"missing {class_name}"
        return matches

    document_view = region("document-library")
    settings_view = region("settings-view")
    assert document_view == [("section", {
        "v-if": "isAuthenticated && activeWorkspace === 'documents'",
        "class": "content-view document-library",
    })]
    assert settings_view == [("section", {
        "v-if": "isAuthenticated && activeWorkspace === 'settings'",
        "class": "content-view settings-view",
    })]
    assert region("content-view-header")
    assert region("upload-zone")
    assert region("document-collection")
    assert region("settings-group")

    task_cards = region("task-card")
    task_lists = region("task-step-list")
    task_steps = region("task-step")
    assert len(task_cards) == len(task_lists) == len(task_steps) == 2
    assert {attrs.get("v-if") for _, attrs in task_cards} == {
        "uploadSteps.length", "deleteJobs[doc.filename]",
    }
    assert all(attrs.get(":key") == "step.key" for _, attrs in task_steps)

    toast_region = region("toast-region")
    assert toast_region == [("div", {
        "class": "toast-region", "aria-live": "polite", "aria-atomic": "false",
    })]
    dialog = [(tag, attrs) for tag, attrs in parser.attrs if attrs.get("role") == "dialog"]
    assert dialog == [("section", {
        "class": "confirm-dialog", "role": "dialog", "aria-modal": "true",
        "aria-labelledby": "confirm-title", "@keydown.tab": "trapConfirmationFocus",
    })]
    handlers = {attrs.get("@click") for _, attrs in parser.attrs}
    assert {"dismissToast(toast.id)", "cancelPendingAction", "confirmPendingAction"} <= handlers

    assert "alert(" not in source
    assert "confirm(" not in source


def test_settings_theme_control_survives_sidebar_collapse_and_task_cards_expose_state():
    """Catch sidebar-only CSS leaking onto settings or task toggles losing ARIA state."""
    html = frontend_source("index.html")
    css = re.sub(r"/\*.*?\*/", "", frontend_source("style.css"), flags=re.S)
    settings_markup = html.split("activeWorkspace === 'settings'", 1)[1].split(
        "activeWorkspace === 'chat'", 1
    )[0]
    assert 'aria-label="界面主题"' in settings_markup
    assert not re.search(r"\.sidebar-is-collapsed\s+\.theme-control\s*\{", css)

    task_buttons = [attrs for tag, attrs in parse_frontend().attrs
                    if tag == "button" and "task-card-header" in attrs.get("class", "").split()]
    assert task_buttons == [
        {
            "type": "button", "class": "task-card-header",
            "@click": "toggleUploadProgressCollapsed",
            ":aria-expanded": "!uploadProgressCollapsed",
            "aria-controls": "upload-task-steps",
        },
        {
            "type": "button", "class": "task-card-header",
            "@click": "toggleDeleteJobCollapsed(doc.filename)",
            ":aria-expanded": "!deleteJobs[doc.filename].collapsed",
            ":aria-controls": "getDeleteTaskStepsId(doc.filename)",
        },
    ]
    task_lists = [attrs for tag, attrs in parse_frontend().attrs
                  if tag == "div" and "task-step-list" in attrs.get("class", "").split()]
    assert task_lists[0]["id"] == "upload-task-steps"
    assert task_lists[1][":id"] == "getDeleteTaskStepsId(doc.filename)"
    assert feedback_runtime_contract()["deleteTaskId"] == "delete-task-steps-guide%20v2.pdf"


def test_confirmation_focus_is_trapped_and_restored():
    """Catch modal focus remaining on background controls or escaping its action buttons."""
    runtime = feedback_runtime_contract()
    assert "confirmationTrigger" in runtime["data"]
    assert "trapConfirmationFocus" in runtime["methods"]
    assert runtime["initialConfirmationFocus"] == "confirm-cancel"
    assert runtime["confirmationTabCycle"] == {
        "events": ["confirm-cancel", "confirm-action", "confirm-cancel", "confirm-action"],
        "prevented": ["forward", "wrap", "backward"],
        "active": "confirm-action",
    }
    assert runtime["focusAfterConfirm"] == "confirmation-trigger"
    assert runtime["cancelled"] == [None, False, "cancel-trigger"]


def test_feedback_methods_run_confirmed_actions_and_report_outcomes():
    """Catch eager destructive requests, lost async callbacks, or missing in-page feedback."""
    runtime = feedback_runtime_contract()
    assert {"toasts", "toastSequence", "confirmation"} <= set(runtime["data"])
    assert {
        "showToast", "dismissToast", "requestConfirmation", "confirmPendingAction",
        "cancelPendingAction", "performDeleteSession", "performDeleteDocument",
    } <= set(runtime["methods"])

    assert runtime["toastAdded"] == [{"id": 1, "message": "上传完成", "tone": "success"}]
    assert runtime["toastDelay"] == 4200
    assert runtime["toastAfterTimeout"] == []
    assert runtime["confirmationDefaults"] == {
        "title": "执行操作？", "message": "需要确认",
        "confirmLabel": "确认", "tone": "danger",
    }
    assert runtime["confirmationClearedSynchronously"]
    assert runtime["actionAtConfirm"] == "started-after-close"
    assert runtime["actionAfterAwait"] == "completed"
    assert runtime["cancelled"] == [None, False, "cancel-trigger"]

    assert runtime["clearBeforeConfirm"] == [1, "开始新对话？", "开始新对话"]
    assert runtime["clearAfterConfirm"] == 0
    assert runtime["sessionBeforeConfirm"] == [0, 2, "删除会话？"]
    assert runtime["sessionAfterConfirm"]["requests"] == [["/api/sessions/saved", "DELETE"]]
    assert runtime["sessionAfterConfirm"]["sessions"] == [{"session_id": "other"}]
    assert runtime["sessionAfterConfirm"]["messages"] == []
    assert runtime["sessionAfterConfirm"]["resetSession"]
    assert runtime["sessionAfterConfirm"]["toast"]["tone"] == "success"
    assert runtime["sessionFailure"]["sessions"] == [{"session_id": "blocked"}]
    assert runtime["sessionFailure"]["toast"]["tone"] == "danger"

    assert runtime["documentBeforeConfirm"] == [0, "删除文档索引？", None]
    assert runtime["documentAfterConfirm"]["requests"] == [
        ["/api/documents/delete/async/guide.pdf?delete_id=" + "a" * 32, "DELETE"],
        ["/api/documents/delete/jobs/" + "a" * 32, "GET"],
    ]
    assert runtime["documentAfterConfirm"]["job"]["jobId"] == "a" * 32
    assert runtime["documentAfterConfirm"]["job"]["status"] == "running"
    assert runtime["documentAfterConfirm"]["pollStarted"] == ["guide.pdf", "a" * 32]
    assert runtime["documentFailure"]["job"]["status"] == "failed"
    assert runtime["documentFailure"]["toast"]["tone"] == "danger"
    assert runtime["documentCompletion"]["removalScheduled"] == "done.pdf"
    assert runtime["documentCompletion"]["toast"]["tone"] == "success"
    assert runtime["uploadCompletion"]["idle"]
    assert runtime["uploadCompletion"]["selectedFiles"] == []
    assert runtime["uploadCompletion"]["reloaded"]
    assert runtime["uploadCompletion"]["toast"]["tone"] == "success"
    assert [toast["tone"] for toast in runtime["validationToasts"]] == ["warning", "warning"]


def test_workspace_has_responsive_and_accessible_contracts():
    """Catch regressions in both drawer breakpoints, focus visibility, and motion safety."""
    html = frontend_source("index.html")
    css = frontend_source("style.css")
    parser = parse_frontend()

    medium = css_rules(css_media_block(css, "max-width: 980px"))
    for selector in (
        ".workspace-shell",
        ".workspace-shell.sidebar-is-collapsed",
        ".workspace-shell.inspector-is-open",
    ):
        assert medium[selector]["grid-template-columns"] == "56px minmax(0, 1fr)"
    for property_name, expected in {
        "position": "fixed",
        "inset": "0 0 0 auto",
        "width": "min(380px, calc(100vw - 56px))",
        "z-index": "40",
        "transform": "translateX(100%)",
        "background": "var(--color-panel)",
    }.items():
        assert medium[".inspector-panel"][property_name] == expected
    assert medium[".inspector-panel.open"]["transform"] == "translateX(0)"

    mobile = css_rules(css_media_block(css, "max-width: 680px"))
    assert mobile[".workspace-shell"]["display"] == "block"
    assert mobile[".workspace-main"]["height"] == "100dvh"
    assert mobile[".workspace-sidebar"]["position"] == "fixed"
    assert mobile[".workspace-sidebar"]["inset"] == "0 auto 0 0"
    assert mobile[".workspace-sidebar"]["width"] == "min(300px, 88vw)"
    assert mobile[".workspace-sidebar"]["z-index"] == "50"
    assert mobile[".workspace-sidebar"]["transform"] == "translateX(-100%)"
    assert mobile[".workspace-sidebar.mobile-open"]["transform"] == "translateX(0)"
    assert mobile[".inspector-panel"]["width"] == "min(100vw, 420px)"
    assert mobile[".conversation-scroll"]["padding"] == "16px 16px 160px"
    assert mobile[".composer-dock"]["padding"] == "16px 12px"
    assert mobile[".message-user .message-body"]["max-width"] == "90%"

    global_rules = css_rules(css.split("@media (max-width: 1100px)", 1)[0])
    assert global_rules[":focus-visible"] == {
        "outline": "2px solid var(--color-accent)",
        "outline-offset": "2px",
    }
    for selector in ("button", "select", ".icon-btn"):
        assert global_rules[selector]["min-height"] == "44px"

    reduced = css_rules(css_media_block(css, "prefers-reduced-motion: reduce"))
    for selector in ("*", "*::before", "*::after"):
        assert reduced[selector] == {
            "scroll-behavior": "auto !important",
            "animation-duration": ".01ms !important",
            "animation-iteration-count": "1 !important",
            "transition-duration": ".01ms !important",
        }

    assert 'aria-live="polite"' in html
    assert 'aria-modal="true"' in html
    sidebar = next(attrs for tag, attrs in parser.attrs
                   if tag == "aside" and "workspace-sidebar" in attrs.get("class", "").split())
    assert sidebar[":class"] == "{ 'mobile-open': mobileSidebarOpen }"
    scrim = next(attrs for tag, attrs in parser.attrs
                 if tag == "button" and "mobile-scrim" in attrs.get("class", "").split())
    assert scrim["v-if"] == "mobileOverlayVisible"
    assert scrim["@click"] == "closeMobilePanels"

    icon_delete = next(attrs for tag, attrs in parser.attrs
                       if tag == "button" and attrs.get("@click") == "deleteDocument(doc.filename)")
    assert icon_delete[":aria-label"] == "'删除文档：' + doc.filename"


def test_global_escape_closes_topmost_layer_and_listener_is_cleaned_up():
    """Escape cancels the modal first, then closes inspector/sidebar drawers."""
    runtime = keyboard_runtime_contract()
    assert runtime["methodPresent"]
    assert runtime["listenerRegistered"]
    assert runtime["nonEscape"] == [True, True, True]
    assert runtime["confirmationEscape"] == [None, True, True, "confirmation-trigger"]
    assert runtime["inspectorEscape"] == [False, False, "inspector-trigger"]
    assert runtime["sidebarEscape"] == [False, False]
    assert runtime["listenerRemoved"]


def test_closed_mobile_sidebar_is_inert_and_restores_trigger_focus():
    """A translated-offscreen drawer must leave both the tab order and accessibility tree."""
    parser = parse_frontend()
    sidebar = next(attrs for tag, attrs in parser.attrs
                   if tag == "aside" and "workspace-sidebar" in attrs.get("class", "").split())
    assert sidebar[":aria-hidden"] == "sidebarInert"
    assert sidebar[":inert"] == "sidebarInert"

    mobile_trigger = next(attrs for tag, attrs in parser.attrs
                          if tag == "button" and "mobile-nav-toggle" in attrs.get("class", "").split())
    assert mobile_trigger["ref"] == "mobileNavTrigger"

    mobile = css_rules(css_media_block(frontend_source("style.css"), "max-width: 680px"))
    assert mobile[".workspace-sidebar"]["visibility"] == "hidden"
    assert mobile[".workspace-sidebar.mobile-open"]["visibility"] == "visible"

    runtime = keyboard_runtime_contract()
    assert runtime["mobileSidebarOpened"] == [True, False, False]
    assert runtime["mobileSidebarClosed"] == [False, True, "mobile-nav-trigger"]
    assert runtime["mobileSidebarWorkspaceClose"] == [False, True, "mobile-nav-trigger"]
    assert runtime["mobileListenerRegistered"]
    assert runtime["desktopSidebarHidden"] is False
    assert runtime["mobileListenerRemoved"]


def test_form_controls_preserve_visible_keyboard_focus_outlines():
    """Local form resets must not suppress the global keyboard focus indicator."""
    global_rules = css_rules(frontend_source("style.css").split("@media (max-width: 1100px)", 1)[0])
    assert global_rules[":focus-visible"]["outline"] == "2px solid var(--color-accent)"
    for selector in (".auth-form input", ".auth-form select", "textarea"):
        assert global_rules[selector].get("outline") != "none"
