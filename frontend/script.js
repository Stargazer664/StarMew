const { createApp } = Vue;

const emptyDiagnosticResource = () => ({ data: null, loading: false, stale: false, error: '', updatedAt: null });

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeWorkspace: 'chat',
            abortController: null,
            streamGeneration: 0,
            activeRunId: null,
            isInterrupting: false,
            interruptionOwner: null,
            pendingSupplement: null,
            stoppingRunId: null,
            apiPrefix: null,
            sessionId: 'session_' + Date.now(),
            sessions: [],
            sessionsLoading: false,
            sessionsError: '',
            sidebarCollapsed: false,
            mobileSidebarOpen: false,
            isMobileViewport: window.matchMedia('(max-width: 680px)').matches,
            mobileMediaQuery: null,
            isOverlayViewport: window.matchMedia('(max-width: 1100px)').matches,
            overlayMediaQuery: null,
            inspectorOpen: false,
            inspectorTab: 'process',
            inspectorMessageIndex: null,
            inspectorTrigger: null,
            isComposing: false,
            documents: [],
            documentsLoading: false,
            documentsError: '',
            selectedFiles: [],
            isUploading: false,
            uploadProgress: '',
            uploadSteps: [],
            uploadFailures: [],
            uploadStatusUnknowns: [],
            uploadRecoveryBusy: {},
            uploadHistoryLoading: false,
            uploadHistoryError: '',
            uploadBatchCurrent: 0,
            uploadBatchTotal: 0,
            uploadProgressCollapsed: false,
            activeUploadJobId: '',
            uploadPollTimer: null,
            deleteJobs: {},
            deleteHistoryLoading: false,
            deleteHistoryError: '',
            deleteRecoveryBusy: false,
            deletePollTimers: {},
            deleteRemoveTimers: {},
            diagnosticsSummary: emptyDiagnosticResource(),
            diagnosticsChatSummary: emptyDiagnosticResource(),
            diagnosticsAlerts: emptyDiagnosticResource(),
            diagnosticsTraces: emptyDiagnosticResource(),
            diagnosticsJobDetail: emptyDiagnosticResource(),
            diagnosticsTraceDetail: emptyDiagnosticResource(),
            diagnosticsEvents: emptyDiagnosticResource(),
            diagnosticsWindow: '24h',
            diagnosticsAlertState: 'open',
            diagnosticsTab: 'document',
            diagnosticsJobOperation: 'upload',
            diagnosticsJobId: '',
            diagnosticsJobSelection: '',
            diagnosticsAttempt: null,
            diagnosticsTraceId: '',
            diagnosticsTraceSelection: '',
            diagnosticsTimer: null,
            diagnosticsEpoch: 0,
            diagnosticsRequestSequence: 0,
            diagnosticsRequests: {},
            diagnosticsRefreshing: false,
            diagnosticsAcknowledging: {},
            diagnosticsActionError: '',
            toasts: [],
            toastSequence: 0,
            confirmation: null,
            confirmationTrigger: null,
            themePreference: (() => {
                const preference = localStorage.getItem('starmew-theme');
                return ['system', 'light', 'dark'].includes(preference) ? preference : 'system';
            })(),
            systemThemeDark: window.matchMedia('(prefers-color-scheme: dark)').matches,
            themeMediaQuery: null,
            token: localStorage.getItem('accessToken') || '',
            currentUser: null,
            authMode: 'login',
            authForm: {
                username: '',
                password: '',
                role: 'user',
                admin_code: ''
            },
            authLoading: false,
            authError: ''
        };
    },
    computed: {
        diagnosticSelectedAttempt() {
            return this.diagnosticsJobDetail.data?.attempts?.find(item => Number(item.attempt) === this.diagnosticsAttempt) || null;
        },
        diagnosticActiveSummary() {
            return this.diagnosticsTab === 'chat' ? this.diagnosticsChatSummary : this.diagnosticsSummary;
        },
        diagnosticActiveSpans() {
            return this.diagnosticsTab === 'chat' ? (this.diagnosticsTraceDetail.data?.spans || [])
                : (this.diagnosticSelectedAttempt?.spans || []);
        },
        diagnosticFailedSpans() {
            return this.diagnosticActiveSpans.filter(span => span.status === 'failed');
        },
        diagnosticMissingStagesMessage() {
            const resource = this.diagnosticsTab === 'chat' ? this.diagnosticsTraceDetail : this.diagnosticsJobDetail;
            return this.diagnosticIntegrityMessage(resource)
                ? '记录不完整，部分阶段未知；无法从缺少的记录判断这些阶段是否执行。'
                : '暂无阶段记录；可能未执行或诊断缺失，耗时未知。';
        },
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        },
        isAdmin() {
            return this.currentUser?.role === 'admin';
        },
        mobileOverlayVisible() {
            return this.mobileSidebarOpen || this.inspectorOpen;
        },
        mobileSidebarHidden() {
            return this.sidebarInert;
        },
        sidebarInert() {
            return (this.isMobileViewport && !this.mobileSidebarOpen)
                || (this.isOverlayViewport && this.inspectorOpen);
        },
        mainContentInert() {
            return this.isOverlayViewport && (this.mobileSidebarOpen || this.inspectorOpen);
        },
        currentConversationTitle() {
            const current = this.sessions.find(session => session.session_id === this.sessionId);
            return current ? this.formatSessionTitle(current) : '新对话';
        },
        inspectorMessage() {
            if (this.inspectorMessageIndex === null) return null;
            return this.messages[this.inspectorMessageIndex] || null;
        },
        inspectorSources() {
            const trace = this.inspectorMessage?.ragTrace;
            if (!trace) return [];
            if (trace.expanded_retrieved_chunks?.length) return trace.expanded_retrieved_chunks;
            if (trace.initial_retrieved_chunks?.length) return trace.initial_retrieved_chunks;
            return trace.retrieved_chunks || [];
        },
        groupedInspectorSources() {
            const groups = new Map();
            this.inspectorSources.forEach(chunk => {
                const filename = chunk.filename || '未知来源';
                if (!groups.has(filename)) groups.set(filename, { filename, chunks: [] });
                groups.get(filename).chunks.push(chunk);
            });
            return Array.from(groups.values());
        },
        resolvedTheme() {
            if (this.themePreference === 'system') {
                return this.systemThemeDark ? 'dark' : 'light';
            }
            return this.themePreference;
        }
    },
    async mounted() {
        document.addEventListener?.('visibilitychange', this.syncDiagnosticsPolling);
        this.configureMarked();
        window.addEventListener('keydown', this.handleGlobalKeydown);
        this.mobileMediaQuery = window.matchMedia('(max-width: 680px)');
        this.mobileMediaQuery.addEventListener('change', this.syncMobileViewport);
        this.isMobileViewport = this.mobileMediaQuery.matches;
        this.overlayMediaQuery = window.matchMedia('(max-width: 1100px)');
        this.overlayMediaQuery.addEventListener('change', this.syncOverlayViewport);
        this.isOverlayViewport = this.overlayMediaQuery.matches;
        this.themeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        this.themeMediaQuery.addEventListener('change', this.syncSystemTheme);
        this.applyTheme();
        if (this.token) {
            try {
                await this.fetchMe();
                await this.loadSessions();
            } catch (error) {
                if (!this.authError) this.handleLogout({ authError: error.message || '登录已过期，请重新登录' });
            }
        }
    },
    beforeUnmount() {
        document.removeEventListener?.('visibilitychange', this.syncDiagnosticsPolling);
        this.stopDiagnosticsPolling();
        window.removeEventListener('keydown', this.handleGlobalKeydown);
        if (this.mobileMediaQuery) {
            this.mobileMediaQuery.removeEventListener('change', this.syncMobileViewport);
        }
        if (this.overlayMediaQuery) {
            this.overlayMediaQuery.removeEventListener('change', this.syncOverlayViewport);
        }
        if (this.themeMediaQuery) {
            this.themeMediaQuery.removeEventListener('change', this.syncSystemTheme);
        }
        this.stopUploadJobPolling();
        this.stopAllDeleteJobPolling();
        Object.values(this.deleteRemoveTimers).forEach(timer => clearTimeout(timer));
        this.cancelActiveStream();
    },
    methods: {
        diagnosticsCanLoad() {
            return Boolean(this.token && this.currentUser?.role === 'admin'
                && this.activeWorkspace === 'documents' && !document.hidden);
        },

        syncDiagnosticsPolling() {
            if (!this.diagnosticsCanLoad()) {
                this.stopDiagnosticsPolling();
                return;
            }
            if (this.diagnosticsTimer !== null) return;
            if (typeof setInterval !== 'function') return;
            this.diagnosticsTimer = setInterval(() => this.refreshDiagnostics(), 10000);
            this.refreshDiagnostics();
        },

        cancelDiagnosticRequest(key) {
            const pending = this.diagnosticsRequests[key];
            if (pending) pending.controller.abort();
            delete this.diagnosticsRequests[key];
            if (this[key]) this[key].loading = false;
        },

        stopDiagnosticsPolling() {
            if (this.diagnosticsTimer !== null) clearInterval(this.diagnosticsTimer);
            this.diagnosticsTimer = null;
            this.diagnosticsEpoch += 1;
            Object.keys(this.diagnosticsRequests).forEach(key => this.cancelDiagnosticRequest(key));
            this.diagnosticsRefreshing = false;
            this.diagnosticsAcknowledging = {};
            ['diagnosticsSummary', 'diagnosticsChatSummary', 'diagnosticsAlerts', 'diagnosticsTraces',
                'diagnosticsJobDetail', 'diagnosticsTraceDetail', 'diagnosticsEvents'].forEach(key => {
                if (this[key].data) this[key].stale = true;
            });
        },

        clearDiagnostics() {
            this.stopDiagnosticsPolling();
            ['diagnosticsSummary', 'diagnosticsChatSummary', 'diagnosticsAlerts', 'diagnosticsTraces',
                'diagnosticsJobDetail', 'diagnosticsTraceDetail', 'diagnosticsEvents'].forEach(key => {
                this[key] = emptyDiagnosticResource();
            });
            this.diagnosticsJobId = '';
            this.diagnosticsJobSelection = '';
            this.diagnosticsTraceId = '';
            this.diagnosticsTraceSelection = '';
            this.diagnosticsAttempt = null;
            this.diagnosticsActionError = '';
        },

        async requestDiagnostic(key, url, { method = 'GET', merge = false } = {}) {
            if (!this.diagnosticsCanLoad()) return null;
            this.cancelDiagnosticRequest(key);
            const request = { id: ++this.diagnosticsRequestSequence, controller: new AbortController() };
            const epoch = this.diagnosticsEpoch;
            this.diagnosticsRequests[key] = request;
            const resource = this[key];
            if (resource) { resource.loading = true; resource.error = ''; }
            const isCurrent = () => this.diagnosticsEpoch === epoch
                && this.diagnosticsRequests[key]?.id === request.id && this.diagnosticsCanLoad();
            let timer;
            try {
                const data = await Promise.race([
                    (async () => {
                        const response = await this.authFetch(url, { method, signal: request.controller.signal });
                        if (!response.ok) throw new Error(`诊断请求失败（HTTP ${response.status}）`);
                        const value = await response.json();
                        if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('诊断响应格式无效');
                        if (['diagnosticsAlerts', 'diagnosticsTraces', 'diagnosticsEvents'].includes(key) && !Array.isArray(value.items)) throw new Error('诊断列表格式无效');
                        if (['diagnosticsSummary', 'diagnosticsChatSummary'].includes(key)
                            && (!value.counts || typeof value.counts !== 'object' || !Array.isArray(value.groups))) throw new Error('诊断概览格式无效');
                        if (key === 'diagnosticsJobDetail' && !Array.isArray(value.attempts)) throw new Error('执行历史格式无效');
                        if (key === 'diagnosticsTraceDetail' && (!value.trace || !Array.isArray(value.spans))) throw new Error('请求详情格式无效');
                        return value;
                    })(),
                    new Promise((resolve, reject) => {
                        timer = setTimeout(() => { request.controller.abort(); reject(new Error('诊断请求超时')); }, 15000);
                    }),
                ]);
                if (!isCurrent()) return null;
                if (resource) {
                    if (merge && Array.isArray(data.items)) {
                        const items = new Map((resource.data?.items || []).map(item => [item.id ?? item.event_id ?? item.trace_id, item]));
                        data.items.forEach(item => items.set(item.id ?? item.event_id ?? item.trace_id, item));
                        resource.data = { ...data, items: [...items.values()] };
                    } else resource.data = data;
                    resource.stale = false;
                    resource.updatedAt = new Date().toISOString();
                }
                return data;
            } catch (error) {
                if (isCurrent()) {
                    if (resource) { resource.stale = true; resource.error = error.message; }
                    else this.diagnosticsActionError = error.message;
                }
                return null;
            } finally {
                clearTimeout(timer);
                if (this.diagnosticsRequests[key]?.id === request.id) {
                    delete this.diagnosticsRequests[key];
                    if (resource) resource.loading = false;
                }
            }
        },

        async refreshDiagnostics() {
            if (!this.diagnosticsCanLoad() || this.diagnosticsRefreshing) return;
            this.diagnosticsRefreshing = true;
            const epoch = this.diagnosticsEpoch;
            try {
                await Promise.allSettled([
                    this.requestDiagnostic('diagnosticsSummary', `/api/documents/diagnostics/summary?window=${this.diagnosticsWindow}`),
                    this.requestDiagnostic('diagnosticsChatSummary', `/api/diagnostics/chat/summary?window=${this.diagnosticsWindow}`),
                    this.loadDiagnosticAlerts(), this.loadDiagnosticTraces(),
                ]);
            } finally {
                if (epoch === this.diagnosticsEpoch) this.diagnosticsRefreshing = false;
            }
        },

        changeDiagnosticsWindow() {
            this.stopDiagnosticsPolling();
            this.syncDiagnosticsPolling();
        },

        loadDiagnosticAlerts(more = false) {
            if (this.diagnosticsAlerts.loading) return;
            const cursor = more ? this.diagnosticsAlerts.data?.next_cursor : null;
            return this.requestDiagnostic('diagnosticsAlerts', `/api/documents/diagnostics/alerts?state=${this.diagnosticsAlertState}&limit=50`
                + (cursor != null ? `&before_id=${encodeURIComponent(cursor)}` : ''), { merge: more });
        },

        changeDiagnosticAlertState() {
            this.cancelDiagnosticRequest('diagnosticsAlerts');
            this.diagnosticsAlerts = emptyDiagnosticResource();
            return this.loadDiagnosticAlerts();
        },

        loadDiagnosticTraces(more = false) {
            if (this.diagnosticsTraces.loading) return;
            const cursor = more ? this.diagnosticsTraces.data?.next_cursor : null;
            return this.requestDiagnostic('diagnosticsTraces', '/api/diagnostics/chat/traces?limit=50'
                + (cursor != null ? `&before_id=${encodeURIComponent(cursor)}` : ''), { merge: more });
        },

        async acknowledgeDiagnosticAlert(id) {
            if (this.diagnosticsAcknowledging[id]) return;
            this.diagnosticsAcknowledging[id] = true;
            this.diagnosticsActionError = '';
            const data = await this.requestDiagnostic(`ack:${id}`, `/api/documents/diagnostics/alerts/${encodeURIComponent(id)}/acknowledge`, { method: 'POST' });
            if (data) {
                const alert = data.alert || data;
                for (const list of [this.diagnosticsAlerts.data?.items, this.diagnosticsTraceDetail.data?.alerts]) {
                    if (!list) continue;
                    const index = list.findIndex(item => String(item.id) === String(id));
                    if (index >= 0) list[index] = { ...list[index], ...alert };
                }
            }
            delete this.diagnosticsAcknowledging[id];
        },

        async selectDiagnosticJob(operation = this.diagnosticsJobOperation, jobId = this.diagnosticsJobId) {
            jobId = String(jobId || '').trim();
            if (!['upload', 'delete'].includes(operation) || !/^[0-9a-f]{32}$/.test(jobId)) {
                this.diagnosticsJobDetail.error = '请输入 32 位小写十六进制任务 ID';
                return;
            }
            this.diagnosticsTab = 'document';
            const selection = `${operation}/${jobId}`;
            if (this.diagnosticsJobSelection !== selection) {
                this.cancelDiagnosticRequest('diagnosticsEvents');
                this.diagnosticsEvents = emptyDiagnosticResource();
                this.diagnosticsJobDetail = emptyDiagnosticResource();
                this.diagnosticsAttempt = null;
            }
            this.diagnosticsJobSelection = selection;
            this.diagnosticsJobOperation = operation;
            this.diagnosticsJobId = jobId;
            const data = await this.requestDiagnostic('diagnosticsJobDetail', `/api/documents/diagnostics/jobs/${selection}`);
            if (data && this.diagnosticsJobSelection === selection) {
                const attempts = data.attempts || [];
                if (attempts.length) await this.selectDiagnosticAttempt(Math.max(...attempts.map(item => Number(item.attempt))));
            }
        },

        selectDiagnosticAttempt(attempt) {
            this.cancelDiagnosticRequest('diagnosticsEvents');
            this.diagnosticsAttempt = Number(attempt);
            this.diagnosticsEvents = emptyDiagnosticResource();
            return this.loadDiagnosticEvents();
        },

        loadDiagnosticEvents(more = false) {
            if (this.diagnosticsEvents.loading || this.diagnosticsAttempt === null) return;
            const cursor = more ? this.diagnosticsEvents.data?.next_cursor : null;
            const selection = this.diagnosticsJobSelection || `${this.diagnosticsJobOperation}/${encodeURIComponent(this.diagnosticsJobId)}`;
            return this.requestDiagnostic('diagnosticsEvents', `/api/documents/diagnostics/jobs/${selection}/events?attempt=${this.diagnosticsAttempt}&limit=50`
                + (cursor != null ? `&after_id=${encodeURIComponent(cursor)}` : ''), { merge: more });
        },

        selectDiagnosticTrace(traceId) {
            traceId = String(traceId || '').trim();
            if (!traceId) return;
            this.diagnosticsTab = 'chat';
            if (traceId !== this.diagnosticsTraceSelection) this.diagnosticsTraceDetail = emptyDiagnosticResource();
            this.diagnosticsTraceSelection = traceId;
            this.diagnosticsTraceId = traceId;
            return this.requestDiagnostic('diagnosticsTraceDetail', `/api/diagnostics/chat/traces/${encodeURIComponent(traceId)}`);
        },

        openDiagnosticAlert(alert) {
            if (alert.job_id && ['upload', 'delete'].includes(alert.operation)) return this.selectDiagnosticJob(alert.operation, alert.job_id);
            if (alert.scope === 'chat' && alert.trace_id) return this.selectDiagnosticTrace(alert.trace_id);
        },

        formatDiagnosticDuration(value) {
            if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '未知';
            return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
        },

        diagnosticStatusLabel(status) {
            return ({ completed: '已完成', success: '成功', failed: '失败', running: '执行中', pending: '等待中',
                interrupted: '已中断', cancelled: '已取消', canceled: '已取消', not_executed: '未执行', skipped: '未执行',
                blocked: '被阻塞', degraded: '已降级', completed_after_cancel: '取消后后台完成' })[status] || status || '未知';
        },

        diagnosticField(item, field) {
            return item?.[field] ?? item?.attributes?.[field] ?? item?.attrs?.[field] ?? null;
        },

        diagnosticLabel(value) {
            return ({ completed: '完成', failed: '失败', interrupted: '中断', cancelled: '取消', unknown: '未知',
                total: '总数', upload: '上传', delete: '删除', chat: '问答', document: '文档', recovery: '恢复', system: '系统',
                generation: '答案生成', final_answer: '最终回答', retrieval: '检索', rerank: '重排',
                tool_decision: '工具决策', grade: '相关性评分', rewrite: '查询改写', summary: '对话摘要',
                grading: '相关性评分', rewrite_routing: '改写路由', conversation_summary: '对话摘要',
                step_back_question: '上位问题改写', step_back_answer: '上位问题回答', hyde: '假设答案生成',
                model_call: '模型调用', rerank_external: '外部重排', index_wait: '索引等待',
                dense_embedding: '稠密向量化', sparse_embedding: '稀疏编码', initialization: '模型初始化',
                queue_wait: '排队等待', processing: '处理', rollback: '补偿', receive: '接收文件',
                critical: '严重', error: '错误', warning: '警告', info: '信息' })[value] || value || '未知';
        },

        diagnosticResourceStatus(resource) {
            if (resource.stale) return resource.data ? '已过期 · 保留上次数据' : '诊断不可用 · 状态未知';
            if (resource.loading) return '正在加载';
            if (!resource.data) return '尚未加载';
            return `更新于 ${new Date(resource.updatedAt).toLocaleTimeString()}`;
        },

        diagnosticIntegrityMessage(resource) {
            const data = resource?.data;
            if (!data) return '';
            const records = [data, data.trace, ...(data.attempts || []), ...(data.spans || []), ...(data.items || [])];
            if (records.some(item => item?.diagnostics_incomplete === true
                || item?.spans?.some(span => span.diagnostics_incomplete === true))) {
                return '记录不完整，部分阶段未知；缺少记录不代表未执行。';
            }
            if (typeof data.diagnostics_gaps === 'number' && data.diagnostics_gaps > 0) {
                return `诊断历史记录不完整，部分阶段未知。本次服务运行观测到 ${data.diagnostics_gaps} 次记录缺口，请核对具体任务。`;
            }
            return data.diagnostics_degraded ? '诊断当前降级，记录不完整，部分阶段未知。' : '';
        },

        setThemePreference(preference) {
            const allowed = new Set(['system', 'light', 'dark']);
            this.themePreference = allowed.has(preference) ? preference : 'system';
            localStorage.setItem('starmew-theme', this.themePreference);
            this.applyTheme();
        },

        syncSystemTheme(event) {
            this.systemThemeDark = event.matches;
            this.applyTheme();
        },

        syncMobileViewport(event) {
            this.isMobileViewport = event.matches;
            if (!event.matches) this.mobileSidebarOpen = false;
        },

        syncOverlayViewport(event) {
            this.isOverlayViewport = event.matches;
        },

        applyTheme() {
            document.documentElement.dataset.theme = this.resolvedTheme;
        },

        handleGlobalKeydown(event) {
            if (event.key === 'Tab' && !this.confirmation) {
                if (this.inspectorOpen && this.isOverlayViewport) {
                    this.trapPanelFocus(event, this.$refs.inspectorPanel);
                } else if (this.mobileSidebarOpen && this.isMobileViewport) {
                    this.trapPanelFocus(event, this.$refs.mobileSidebarPanel);
                }
                return;
            }
            if (event.key !== 'Escape') return;
            if (this.confirmation) this.cancelPendingAction();
            else if (this.inspectorOpen) this.closeInspector();
            else if (this.mobileSidebarOpen) this.closeMobileSidebar();
        },

        trapPanelFocus(event, panel) {
            if (!panel || typeof panel.querySelectorAll !== 'function') return;
            const focusable = Array.from(panel.querySelectorAll(
                'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [contenteditable="true"], [tabindex]:not([tabindex="-1"])'
            )).filter(element => !element.hasAttribute || !element.hasAttribute('inert'));
            if (!focusable.length) return;
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        },

        showToast(message, tone = 'info') {
            const id = ++this.toastSequence;
            this.toasts.push({ id, message, tone });
            window.setTimeout(() => this.dismissToast(id), 4200);
        },

        dismissToast(id) {
            this.toasts = this.toasts.filter(toast => toast.id !== id);
        },

        requestConfirmation(options) {
            const activeElement = document.activeElement;
            this.confirmationTrigger = activeElement && typeof activeElement.focus === 'function'
                ? activeElement
                : null;
            this.confirmation = {
                title: options.title,
                message: options.message,
                confirmLabel: options.confirmLabel || '确认',
                tone: options.tone || 'danger',
                onConfirm: options.onConfirm
            };
            this.$nextTick(() => {
                const cancelButton = this.$refs.confirmCancelButton;
                if (cancelButton && typeof cancelButton.focus === 'function') cancelButton.focus();
            });
        },

        trapConfirmationFocus(event) {
            const focusable = [this.$refs.confirmCancelButton, this.$refs.confirmActionButton]
                .filter(element => element && typeof element.focus === 'function');
            if (!focusable.length) return;

            event.preventDefault();
            const currentIndex = focusable.indexOf(document.activeElement);
            const nextIndex = event.shiftKey
                ? (currentIndex <= 0 ? focusable.length - 1 : currentIndex - 1)
                : (currentIndex === -1 || currentIndex === focusable.length - 1 ? 0 : currentIndex + 1);
            focusable[nextIndex].focus();
        },

        async confirmPendingAction() {
            const action = this.confirmation?.onConfirm;
            const trigger = this.confirmationTrigger;
            this.confirmation = null;
            this.confirmationTrigger = null;
            if (trigger && typeof trigger.focus === 'function') {
                this.$nextTick(() => trigger.focus());
            }
            if (action) await action();
        },

        cancelPendingAction() {
            const trigger = this.confirmationTrigger;
            this.confirmation = null;
            this.confirmationTrigger = null;
            if (trigger && typeof trigger.focus === 'function') {
                this.$nextTick(() => trigger.focus());
            }
        },

        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },

        parseMarkdown(text) {
            return marked.parse(text);
        },

        formatSessionTitle(session) {
            const ts = parseInt((session.session_id || '').replace('session_', ''));
            if (ts) return new Date(ts).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            const d = new Date(session.updated_at);
            if (!isNaN(d)) return d.toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            return session.session_id;
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        apiOrigin() {
            const configuredOrigin = typeof window !== 'undefined' && typeof window.STARMEW_API_ORIGIN === 'string'
                ? window.STARMEW_API_ORIGIN.trim().replace(/\/$/, '')
                : '';
            if (configuredOrigin) return configuredOrigin;

            const location = typeof window !== 'undefined' ? window.location : null;
            if (!location) return '';
            if (location.protocol === 'file:') return 'http://127.0.0.1:8000';

            const hostname = (location.hostname || '').toLowerCase();
            const localHosts = new Set(['localhost', '127.0.0.1', '::1']);
            if (!localHosts.has(hostname)) return '';

            const currentPort = location.port || (location.protocol === 'https:' ? '443' : '80');
            if (currentPort === '8000') return '';
            const host = hostname === '::1' ? '[::1]' : hostname;
            return `${location.protocol}//${host}:8000`;
        },

        apiPathCandidates(url) {
            const apiUrl = url.startsWith('/api/') ? url : `/api${url.startsWith('/') ? url : `/${url}`}`;
            const rootUrl = apiUrl.slice(4);
            const relativeCandidates = this.apiPrefix === '' ? [rootUrl, apiUrl] : [apiUrl, rootUrl];
            const backendOrigin = this.apiOrigin();
            const backendCandidates = backendOrigin
                ? [`${backendOrigin}${apiUrl}`, `${backendOrigin}${rootUrl}`]
                : [];
            return [...new Set([...relativeCandidates, ...backendCandidates])];
        },

        uploadMutationPathCandidates(url) {
            const candidates = this.apiPathCandidates(url);
            const backendOrigin = this.apiOrigin();
            if (!backendOrigin) return candidates;
            // Static hosts can reject POST with 501. Send to the known backend before any fallback.
            const isBackend = candidate => candidate.startsWith(`${backendOrigin}/`);
            return [...candidates.filter(isBackend), ...candidates.filter(candidate => !isBackend(candidate))];
        },

        rememberApiPrefix(url) {
            this.apiPrefix = url.startsWith('/api/') ? '/api' : '';
        },

        async apiFetch(url, options = {}) {
            const candidates = this.apiPathCandidates(url);
            let response;
            let lastError;
            for (let index = 0; index < candidates.length; index += 1) {
                const candidate = candidates[index];
                try {
                    response = await fetch(candidate, options);
                } catch (error) {
                    lastError = error;
                    continue;
                }
                const mayTryAlternate = (response.status === 404 || response.status === 405 || response.status === 501)
                    && index < candidates.length - 1;
                if (!mayTryAlternate) {
                    this.rememberApiPrefix(candidate);
                    return response;
                }
            }
            if (lastError) throw lastError;
            return response;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await this.apiFetch(url, opts);
            if (response.status === 401) {
                const message = '登录已过期，请重新登录';
                this.handleLogout({ authError: message });
                throw new Error(message);
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch('/api/auth/me');
            if (!response.ok) {
                throw new Error('认证失败');
            }
            this.currentUser = await response.json();
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                this.showToast('用户名和密码不能为空', 'warning');
                return;
            }

            this.authLoading = true;
            this.authError = '';
            try {
                const endpoint = this.authMode === 'login' ? '/api/auth/login' : '/api/auth/register';
                const payload = {
                    username,
                    password
                };
                if (this.authMode === 'register') {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await this.apiFetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || '认证失败');
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                this.authError = '';
                localStorage.setItem('accessToken', this.token);
                this.authForm.password = '';
                this.authForm.admin_code = '';
                this.messages = [];
                this.sessionId = 'session_' + Date.now();
                this.activeWorkspace = 'chat';
                this.closeMobilePanels();
            } catch (error) {
                this.authError = error.message;
                this.showToast(error.message, 'danger');
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout(options = {}) {
            this.clearDiagnostics();
            const nextAuthError = options && typeof options.authError === 'string'
                ? options.authError
                : '';
            this.cancelActiveStream();
            this.token = '';
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.documents = [];
            this.sessionsError = '';
            this.documentsError = '';
            this.authError = nextAuthError;
            this.activeWorkspace = 'chat';
            this.closeMobilePanels(false);
            this.inspectorMessageIndex = null;
            localStorage.removeItem('accessToken');
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        async requestRunCancellation(runId, sessionId, supplement = null) {
            const body = { session_id: sessionId };
            if (supplement) {
                body.supplement_message = supplement.text;
                body.supplement_request_id = supplement.runId;
            }
            const response = await this.authFetch(`/api/chat/runs/${encodeURIComponent(runId)}/cancel`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            if (!response.ok) {
                const data = await response.json().catch(() => ({}));
                const error = new Error('服务端尚未确认中断，请重试');
                error.code = data.detail?.code;
                throw error;
            }
            return response.json();
        },

        async waitForRunEnd(runId, sessionId, generation) {
            while (this.sessionId === sessionId && this.streamGeneration === generation) {
                const response = await this.authFetch(`/api/chat/runs/${encodeURIComponent(runId)}?session_id=${encodeURIComponent(sessionId)}`);
                if (!response.ok) throw new Error('无法确认旧任务是否结束，请重试');
                const state = await response.json();
                if (this.sessionId !== sessionId || this.streamGeneration !== generation) return false;
                if (['completed', 'cancelled', 'failed'].includes(state.status)) return true;
                await new Promise(resolve => setTimeout(resolve, 250));
            }
            return false;
        },

        markStreamInterrupted(message, detail) {
            if (!message) return;
            message.isThinking = false;
            message.status = 'interrupted';
            message.streamError = detail;
            if (!message.text) message.text = '(已终止回答)';
        },

        async handleStop() {
            const pending = this.pendingSupplement;
            const runId = pending?.runId || this.activeRunId;
            const sessionId = this.sessionId;
            if (!runId || (this.isInterrupting && !pending)) return;
            const controller = this.abortController;
            const message = pending?.botMessage || this.messages.findLast(m => !m.isUser && m.requestId === runId);
            const owner = this.createChatRequestId();
            this.interruptionOwner = owner;
            this.isInterrupting = true;
            try {
                if (pending && !pending.confirmed) {
                    await this.requestRunCancellation(pending.sourceRunId, sessionId, pending);
                    pending.confirmed = true;
                }
                const result = await this.requestRunCancellation(runId, sessionId);
                if (this.sessionId !== sessionId || this.interruptionOwner !== owner) return;
                this.pendingSupplement = null;
                const generation = ++this.streamGeneration;
                this.activeRunId = null;
                this.abortController = null;
                if (controller) controller.abort();
                if (result.status === 'completed' && typeof result.response === 'string') {
                    if (message) { message.text = result.response; message.isThinking = false; message.status = 'complete'; message.streamError = ''; }
                } else this.markStreamInterrupted(message, '回答已终止，可重试');
                this.stoppingRunId = pending?.sourceRunId || runId;
                await this.waitForRunEnd(this.stoppingRunId, sessionId, generation);
                if (this.streamGeneration === generation && this.sessionId === sessionId) {
                    this.stoppingRunId = null;
                    this.isLoading = false;
                }
            } catch (error) {
                if (this.interruptionOwner === owner) {
                    this.showToast(error.message, 'warning');
                    if (!this.abortController) this.isLoading = false;
                }
            } finally {
                if (this.interruptionOwner === owner) {
                    this.isInterrupting = false;
                    this.interruptionOwner = null;
                }
            }
        },

        async handleSupplement() {
            const text = this.userInput.trim();
            const sourceRunId = this.activeRunId;
            if (!text || !sourceRunId || this.isInterrupting || this.isComposing) return;
            if (text.length > 32000) { this.showToast('补充信息过长，请控制在 32000 字符内', 'warning'); return; }
            const sessionId = this.sessionId;
            const controller = this.abortController;
            const oldMessage = this.messages.findLast(m => !m.isUser && m.requestId === sourceRunId);
            // Keep the identity even if the commit succeeds but its HTTP response is lost.
            const pending = { runId: this.createChatRequestId(), text, sourceRunId, sessionId,
                confirmed: false, controller, oldMessage,
                botMessage: { text: '', isUser: false, isThinking: true, ragTrace: null,
                    ragSteps: [{ label: '正在提交补充信息' }], streamError: '', retryText: text } };
            pending.botMessage.requestId = pending.runId;
            this.pendingSupplement = pending;
            this.userInput = '';
            this.messages.push({ text, isUser: true });
            this.messages.push(pending.botMessage);
            try {
                await this.resumeSupplement(pending);
            } catch (error) {
                if (this.sessionId === sessionId && this.pendingSupplement?.runId === pending.runId
                    && (!this.interruptionOwner || this.interruptionOwner === pending.runId)) {
                    pending.botMessage.isThinking = false;
                    pending.botMessage.text = pending.confirmed ? '补充已保存，点击重试继续处理。' : '补充提交未确认，点击重试确认并继续处理。';
                    pending.botMessage.streamError = error.message;
                    this.isLoading = false;
                }
            }
        },

        async resumeSupplement(pending) {
            if (this.pendingSupplement?.runId !== pending.runId || this.sessionId !== pending.sessionId) return;
            this.isLoading = true;
            this.isInterrupting = true;
            this.interruptionOwner = pending.runId;
            try {
                if (!pending.confirmed) {
                    const result = await this.requestRunCancellation(pending.sourceRunId, pending.sessionId, pending);
                    if (this.interruptionOwner !== pending.runId) return;
                    if (this.pendingSupplement?.runId !== pending.runId || this.sessionId !== pending.sessionId) {
                        this.requestRunCancellation(pending.runId, pending.sessionId).catch(() => {});
                        return;
                    }
                    pending.confirmed = true;
                    ++this.streamGeneration;
                    this.abortController = null;
                    this.activeRunId = null;
                    if (result.status === 'completed' && typeof result.response === 'string') {
                        if (pending.oldMessage) { pending.oldMessage.text = result.response; pending.oldMessage.isThinking = false; pending.oldMessage.status = 'complete'; pending.oldMessage.streamError = ''; }
                    } else this.markStreamInterrupted(pending.oldMessage, '已根据补充信息中断');
                    if (pending.controller) pending.controller.abort();
                    pending.controller = null;
                }
                pending.botMessage.isThinking = true;
                pending.botMessage.ragSteps = [{ label: '补充已接收，等待上一任务结束' }];
                const generation = this.streamGeneration;
                if (!await this.waitForRunEnd(pending.sourceRunId, pending.sessionId, generation)) return;
                if (this.pendingSupplement?.runId !== pending.runId || this.interruptionOwner !== pending.runId) return;
                this.pendingSupplement = null;
                this.isLoading = false;
                this.isInterrupting = false;
                this.interruptionOwner = null;
                await this.handleSend(pending.runId, pending);
            } finally {
                if (this.interruptionOwner === pending.runId) {
                    this.isInterrupting = false;
                    this.interruptionOwner = null;
                }
            }
        },

        cancelActiveStream() {
            const runId = this.pendingSupplement?.runId || this.activeRunId;
            const sourceSessionId = this.sessionId;
            if (runId) this.requestRunCancellation(runId, sourceSessionId).catch(() => {});
            this.streamGeneration += 1;
            const controller = this.abortController;
            this.abortController = null;
            this.isLoading = false;
            this.activeRunId = null;
            this.isInterrupting = false;
            this.interruptionOwner = null;
            this.pendingSupplement = null;
            this.stoppingRunId = null;
            if (controller) controller.abort();
        },

        isCurrentStream(generation, sessionId, message) {
            return this.streamGeneration === generation
                && this.sessionId === sessionId
                && this.messages.includes(message);
        },

        retryMessage(message) {
            if (this.isLoading || !message?.retryText) return;
            if (this.pendingSupplement?.runId === message.requestId) {
                return this.resumeSupplement(this.pendingSupplement).catch(error => {
                    if (this.pendingSupplement?.runId === message.requestId
                        && (!this.interruptionOwner || this.interruptionOwner === message.requestId)) {
                        message.streamError = error.message;
                        message.isThinking = false;
                        this.isLoading = false;
                        this.isInterrupting = false;
                    }
                });
            }
            this.userInput = message.retryText;
            message.streamError = '';
            return this.handleSend(message.requestId);
        },

        createChatRequestId() {
            if (crypto.randomUUID) return crypto.randomUUID();
            const bytes = crypto.getRandomValues(new Uint8Array(16));
            bytes[6] = (bytes[6] & 15) | 64;
            bytes[8] = (bytes[8] & 63) | 128;
            const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
            return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
        },

        async handleSend(retryRequestId = null, queuedRun = null) {
            if (!this.isAuthenticated) {
                this.showToast('请先登录', 'warning');
                return;
            }

            if (this.isLoading && !queuedRun) {
                if (typeof retryRequestId !== 'string') return this.handleSupplement();
                return;
            }
            if (this.pendingSupplement && !queuedRun) {
                this.showToast('有补充信息等待处理，请先重试该条消息或停止处理', 'warning');
                return;
            }
            const text = queuedRun ? queuedRun.text : this.userInput.trim();
            if (!text || this.isComposing || this.isInterrupting) return;
            if (this.stoppingRunId) {
                const waitingSession = this.sessionId;
                const waitingGeneration = this.streamGeneration;
                this.isLoading = true;
                try {
                    if (!await this.waitForRunEnd(this.stoppingRunId, waitingSession, waitingGeneration)) return;
                    if (this.sessionId !== waitingSession || this.streamGeneration !== waitingGeneration) return;
                    this.stoppingRunId = null;
                } catch (error) {
                    this.showToast(error.message, 'warning');
                    return;
                } finally {
                    if (this.sessionId === waitingSession && this.streamGeneration === waitingGeneration) this.isLoading = false;
                }
            }
            const requestId = typeof retryRequestId === 'string' ? retryRequestId : this.createChatRequestId();

            if (!queuedRun) this.messages.push({
                text: text,
                isUser: true
            });

            if (!queuedRun) this.userInput = '';
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            this.isLoading = true;
            const botMessage = queuedRun?.botMessage || {
                text: '',
                isUser: false,
                isThinking: true,
                ragTrace: null,
                ragSteps: [],
                streamError: '',
                retryText: text,
                requestId,
            };
            botMessage.text = '';
            botMessage.isThinking = true;
            botMessage.streamError = '';
            if (!queuedRun) this.messages.push(botMessage);

            const controller = new AbortController();
            const generation = ++this.streamGeneration;
            const sourceSessionId = this.sessionId;
            this.abortController = controller;
            this.activeRunId = requestId;

            try {
                const response = await this.authFetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        request_id: requestId,
                        session_id: this.sessionId
                    }),
                    signal: controller.signal,
                });

                if (!response.ok) {
                    const body = await response.json().catch(() => ({}));
                    const code = body.detail?.code;
                    if (['REQUEST_FAILED', 'REQUEST_CANCELLED'].includes(code)) botMessage.requestId = null;
                    const message = ['SESSION_BUSY', 'REQUEST_RUNNING'].includes(code)
                        ? '当前会话正在生成回答，请稍后重试'
                        : ['REQUEST_FAILED', 'REQUEST_CANCELLED'].includes(code)
                            ? '上次请求已结束，再次重试将重新生成回答'
                            : `HTTP ${response.status}${code ? ` (${code})` : ''}`;
                    throw new Error(message);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';
                let receivedDone = false;
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    if (!this.isCurrentStream(generation, sourceSessionId, botMessage)) break;

                    buffer += decoder.decode(value, { stream: true });

                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);

                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') { receivedDone = true; continue; }
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.run_id && data.run_id !== requestId) continue;
                                if (!this.isCurrentStream(generation, sourceSessionId, botMessage)) continue;
                                if (data.type === 'content') {
                                    botMessage.isThinking = false;
                                    botMessage.text += data.content;
                                } else if (data.type === 'trace') {
                                    botMessage.ragTrace = data.rag_trace;
                                } else if (data.type === 'rag_step') {
                                    if (!botMessage.ragSteps) botMessage.ragSteps = [];
                                    botMessage.ragSteps.push(data.step);
                                } else if (data.type === 'error') {
                                    // An error can follow a committed answer. Retry
                                    // the same ID until the server confirms its state.
                                    botMessage.isThinking = false;
                                    botMessage.streamError = data.content || '生成失败';
                                    if (botMessage.text) {
                                        botMessage.text += `\n\n_(生成中断：${botMessage.streamError})_`;
                                    } else {
                                        botMessage.text = `星喵遇到一点问题：${botMessage.streamError}`;
                                    }
                                }
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                        }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }
                if (!receivedDone) throw new Error('响应未完整结束');

            } catch (error) {
                if (!this.isCurrentStream(generation, sourceSessionId, botMessage)) return;
                botMessage.isThinking = false;
                if (error.name === 'AbortError') {
                    // Client cancellation cannot establish the server's commit state.
                    botMessage.streamError = '回答已终止，可重试';
                    if (!botMessage.text) {
                        botMessage.text = '(已终止回答)';
                    } else {
                        botMessage.text += '\n\n_(回答已被终止)_';
                    }
                } else {
                    botMessage.streamError = `连接中断：${error.message}`;
                    if (botMessage.text) {
                        botMessage.text += `\n\n_(${botMessage.streamError}，可重试)_`;
                    } else {
                        botMessage.text = `星喵遇到一点问题：${error.message}`;
                    }
                }
            } finally {
                if (this.isCurrentStream(generation, sourceSessionId, botMessage)) {
                    if (!this.isInterrupting) this.isLoading = false;
                    if (this.activeRunId === requestId) this.activeRunId = null;
                    if (this.abortController === controller) this.abortController = null;
                    this.loadSessions();
                    this.$nextTick(() => this.scrollToBottom());
                }
            }
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = 'auto';
            }
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        handleNewChat() {
            if (!this.isAuthenticated) return;
            this.cancelActiveStream();
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeWorkspace = 'chat';
            this.closeInspector(false);
            this.inspectorMessageIndex = null;
            this.closeMobilePanels();
        },

        handleClearChat() {
            this.requestConfirmation({
                title: '开始新对话？',
                message: '将创建一个空白对话，当前会话仍保留在历史记录中。',
                confirmLabel: '开始新对话',
                tone: 'default',
                onConfirm: () => {
                    this.handleNewChat();
                    this.showToast('已开始新对话', 'success');
                }
            });
        },

        async loadSessions() {
            if (!this.isAuthenticated) return;
            this.sessionsLoading = true;
            this.sessionsError = '';
            try {
                const response = await this.authFetch('/api/sessions');
                if (!response.ok) {
                    const payload = await response.json().catch(() => ({}));
                    throw new Error(payload.detail || '服务暂不可用');
                }
                const data = await response.json();
                this.sessions = Array.isArray(data.sessions) ? data.sessions : [];
            } catch (error) {
                this.sessionsError = `加载会话失败：${error.message}`;
            } finally {
                this.sessionsLoading = false;
            }
        },

        async handleHistory() {
            if (!this.isAuthenticated) return;
            await this.loadSessions();
        },

        async loadSession(sessionId) {
            this.cancelActiveStream();
            const historyGeneration = this.streamGeneration;
            this.sessionId = sessionId;
            this.activeWorkspace = 'chat';
            this.closeInspector(false);
            this.inspectorMessageIndex = null;
            this.closeMobilePanels();

            try {
                const response = await this.authFetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) {
                    throw new Error('Failed to load session messages');
                }
                const data = await response.json();
                if (this.sessionId !== sessionId || this.streamGeneration !== historyGeneration) return;
                this.messages = data.messages.map(msg => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null,
                    status: msg.status || 'complete'
                }));
                if (data.pending_run) {
                    const queued = data.pending_run;
                    const botMessage = { text: '补充已保存，点击重试继续处理。', isUser: false,
                        isThinking: false, streamError: '待继续处理', retryText: queued.message, requestId: queued.run_id };
                    this.messages.push(botMessage);
                    this.pendingSupplement = { runId: queued.run_id, sourceRunId: queued.source_run_id,
                        text: queued.message, sessionId, confirmed: true, botMessage };
                }

                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                if (this.sessionId === sessionId && this.streamGeneration === historyGeneration) {
                    this.showToast('加载会话失败：' + error.message, 'danger');
                    this.messages = [];
                }
            }
        },

        deleteSession(sessionId) {
            this.requestConfirmation({
                title: '删除会话？',
                message: '该会话及其消息记录将被永久删除。',
                confirmLabel: '删除会话',
                tone: 'danger',
                onConfirm: () => this.performDeleteSession(sessionId)
            });
        },

        async performDeleteSession(sessionId) {
            if (this.sessionId === sessionId) this.cancelActiveStream();
            try {
                const response = await this.authFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
                    method: 'DELETE'
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Delete failed');
                }

                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                    this.activeWorkspace = 'chat';
                    this.closeMobilePanels();
                }

                this.showToast(payload.message || '会话已删除', 'success');
            } catch (error) {
                this.showToast('删除会话失败：' + error.message, 'danger');
            }
        },

        switchWorkspace(workspace) {
            const allowed = new Set(['chat', 'documents', 'settings']);
            if (!allowed.has(workspace)) return;
            if (workspace === 'documents' && !this.isAdmin) {
                this.showToast('仅管理员可访问文档库', 'warning');
                return;
            }
            if (workspace !== 'chat') this.cancelActiveStream();
            this.activeWorkspace = workspace;
            if (this.mobileSidebarOpen) {
                this.closeMobilePanels();
            } else {
                this.closeInspector(false);
            }
            this.inspectorMessageIndex = null;
            if (workspace === 'documents') this.loadDocuments();
        },

        closeMobilePanels(restoreFocus = true) {
            const sidebarWasOpen = this.mobileSidebarOpen;
            this.closeInspector(restoreFocus && !sidebarWasOpen);
            this.closeMobileSidebar(restoreFocus);
        },

        closeMobileSidebar(restoreFocus = true) {
            const wasOpen = this.mobileSidebarOpen;
            this.mobileSidebarOpen = false;
            const trigger = wasOpen ? this.$refs.mobileNavTrigger : null;
            if (restoreFocus && trigger && typeof trigger.focus === 'function') {
                this.$nextTick(() => trigger.focus());
            }
        },

        openMobileSidebar() {
            this.sidebarCollapsed = false;
            this.mobileSidebarOpen = true;
            this.$nextTick(() => {
                const firstAction = this.$refs.mobileSidebarFirstAction;
                if (firstAction && typeof firstAction.focus === 'function') firstAction.focus();
            });
        },

        openInspector(messageIndex, tab) {
            if (!this.inspectorOpen) {
                const activeElement = document.activeElement;
                this.inspectorTrigger = activeElement && typeof activeElement.focus === 'function'
                    ? activeElement
                    : null;
            }
            this.inspectorMessageIndex = messageIndex;
            this.inspectorTab = tab === 'sources' ? 'sources' : 'process';
            this.inspectorOpen = true;
            this.$nextTick(() => {
                const tabRef = this.inspectorTab === 'sources'
                    ? this.$refs.inspectorSourcesTab
                    : this.$refs.inspectorProcessTab;
                if (tabRef && typeof tabRef.focus === 'function') tabRef.focus();
            });
        },

        focusInspectorTab(tab) {
            this.inspectorTab = tab === 'sources' ? 'sources' : 'process';
            this.$nextTick(() => {
                const tabRef = this.inspectorTab === 'sources'
                    ? this.$refs.inspectorSourcesTab
                    : this.$refs.inspectorProcessTab;
                if (tabRef && typeof tabRef.focus === 'function') tabRef.focus();
            });
        },

        closeInspector(restoreFocus = true) {
            const trigger = this.inspectorTrigger;
            this.inspectorOpen = false;
            this.inspectorTrigger = null;
            if (restoreFocus && trigger && typeof trigger.focus === 'function') {
                this.$nextTick(() => trigger.focus());
            }
        },

        mergeDocumentsWithActiveDeletes(nextDocuments) {
            const merged = Array.isArray(nextDocuments) ? [...nextDocuments] : [];
            Object.keys(this.deleteJobs).forEach(filename => {
                const job = this.deleteJobs[filename];
                if (!job || job.status === 'failed') return;
                const exists = merged.some(doc => doc.filename === filename);
                if (!exists) {
                    const currentDoc = this.documents.find(doc => doc.filename === filename);
                    if (currentDoc) {
                        merged.push(currentDoc);
                    }
                }
            });
            return merged;
        },

        async loadDocuments() {
            this.syncDiagnosticsPolling();
            this.documentsLoading = true;
            this.documentsError = '';
            try {
                const response = await this.authFetch('/api/documents');
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || 'Failed to load documents');
                }
                const data = await response.json();
                this.documents = this.mergeDocumentsWithActiveDeletes(data.documents);
            } catch (error) {
                this.documentsError = '加载文档列表失败：' + error.message;
                this.showToast(this.documentsError, 'danger');
            } finally {
                this.documentsLoading = false;
            }
            await this.loadUploadJobs();
            await this.loadDeleteJobs();
        },

        async loadDeleteJobs() {
            if (this.deleteHistoryLoading) return;
            this.deleteHistoryLoading = true;
            this.deleteHistoryError = '';
            try {
                const response = await this.authFetch('/api/documents/delete/jobs');
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const jobs = await response.json();
                if (!Array.isArray(jobs)) throw new Error('删除任务列表格式无效');
                const seen = new Set();
                // A newer blocked attempt must not hide the original recovery journal.
                const prioritized = [...jobs].sort((a, b) => Number(b.recovery_required === true) - Number(a.recovery_required === true));
                for (const job of prioritized) {
                    if (!job.job_id || !job.filename || seen.has(job.filename)) continue;
                    seen.add(job.filename);
                    const current = this.deleteJobs[job.filename];
                    // A local submission may be newer than the response to this list request.
                    if (!job.recovery_required && current && current.jobId !== job.job_id && ['running', 'unknown'].includes(current.status)) continue;
                    if (job.status === 'completed') {
                        if (current?.jobId === job.job_id && !this.deleteRemoveTimers[job.filename]) {
                            this.stopDeleteJobPolling(job.filename);
                            const { [job.filename]: removed, ...remaining } = this.deleteJobs;
                            this.deleteJobs = remaining;
                        }
                        continue;
                    }
                    this.syncDeleteJob(job.filename, job);
                    if (['pending', 'running'].includes(job.status) && !this.deletePollTimers[job.filename]) {
                        this.startDeleteJobPolling(job.filename, job.job_id);
                    }
                }
            } catch (error) {
                this.deleteHistoryError = '删除任务记录加载失败：' + error.message;
            } finally {
                this.deleteHistoryLoading = false;
            }
        },

        async loadUploadJobs() {
            if (this.uploadHistoryLoading) return;
            this.uploadHistoryLoading = true;
            this.uploadHistoryError = '';
            try {
                const response = await this.authFetch('/api/documents/upload/jobs');
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const jobs = await response.json();
                if (!Array.isArray(jobs)) throw new Error('任务列表格式无效');
                jobs.forEach(job => {
                    if (job.job_id && !(this.isUploading && job.job_id === this.activeUploadJobId)) {
                        this.recordUploadJob(job);
                    }
                });
                if (this.uploadFailures.length || this.uploadStatusUnknowns.length) {
                    if (!this.uploadSteps.length) this.uploadSteps = this.createUploadSteps();
                    this.uploadProgressCollapsed = false;
                }
            } catch (error) {
                this.uploadHistoryError = `加载上传任务记录失败：${error.message}`;
            } finally {
                this.uploadHistoryLoading = false;
            }
        },

        uploadJobRecord(job) {
            return {
                filename: job.filename,
                message: job.message || '',
                jobId: job.job_id,
                retryable: job.retryable === true,
                recoveryRequired: job.recovery_required === true,
                attempt: job.attempt || 1,
            };
        },

        recordUploadJob(job) {
            this.uploadFailures = this.uploadFailures.filter(item => item.jobId !== job.job_id);
            this.uploadStatusUnknowns = this.uploadStatusUnknowns.filter(item => item.jobId !== job.job_id);
            if (job.status === 'failed') this.uploadFailures.push(this.uploadJobRecord(job));
            else if (job.status === 'pending' || job.status === 'running') {
                this.uploadStatusUnknowns.push(this.uploadJobRecord(job));
            }
        },

        uploadStatusUnknownError(jobId, message) {
            const error = new Error(`无法确认后台任务状态：${message}。请重新查询状态，勿重复上传此文件。`);
            error.isUploadStatusUnknown = true;
            error.jobId = jobId;
            return error;
        },

        async fetchUploadJob(jobId, timeoutMs = 15000) {
            return this.withUploadRequestTimeout(async signal => {
                const response = await this.authFetch(
                    `/api/documents/upload/jobs/${encodeURIComponent(jobId)}`, { signal }
                );
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || `上传进度查询失败（HTTP ${response.status}）`);
                }
                const job = await response.json();
                if (job.job_id !== jobId || !['pending', 'running', 'failed', 'completed'].includes(job.status)) {
                    throw new Error('上传任务响应无效');
                }
                return job;
            }, timeoutMs);
        },

        async withUploadRequestTimeout(operation, timeoutMs = 15000) {
            const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
            let timer;
            try {
                return await Promise.race([
                    operation(controller?.signal),
                    new Promise((resolve, reject) => {
                        timer = setTimeout(() => {
                            controller?.abort();
                            reject(new Error('请求等待超时'));
                        }, timeoutMs);
                    }),
                ]);
            } finally {
                clearTimeout(timer);
            }
        },

        async recheckUploadJob(item) {
            if (!item?.jobId || this.isUploading || this.uploadRecoveryBusy[item.jobId]) return;
            this.uploadRecoveryBusy[item.jobId] = true;
            try {
                const job = await this.fetchUploadJob(item.jobId);
                this.syncUploadJob(job);
                if (job.status === 'completed') await this.loadDocuments();
                this.recordUploadJob(job);
                this.uploadProgressCollapsed = false;
                this.showToast(job.message || '已更新任务状态', job.status === 'completed' ? 'success' : 'info');
            } catch (error) {
                this.recordUploadActionError(item, this.uploadStatusUnknownError(item.jobId, error.message));
            } finally {
                delete this.uploadRecoveryBusy[item.jobId];
            }
        },

        async postUploadAction(path, method = 'POST') {
            // Generic apiFetch retries network errors across API paths. A mutation must not be replayed.
            return this.withUploadRequestTimeout(async signal => {
                const candidates = this.uploadMutationPathCandidates(path);
                for (let index = 0; index < candidates.length; index += 1) {
                    const response = await fetch(candidates[index], { method, headers: this.authHeaders(), signal });
                    if ([404, 405].includes(response.status) && index < candidates.length - 1) continue;
                    this.rememberApiPrefix(candidates[index]);
                    if (response.status === 401) {
                        this.handleLogout({ authError: '登录已过期，请重新登录' });
                    }
                    if (!response.ok) {
                        const data = await response.json().catch(() => ({}));
                        const error = new Error(data.detail || `HTTP ${response.status}`);
                        error.isUploadRequestRejected = response.status >= 400 && response.status < 500;
                        throw error;
                    }
                    return await response.json();
                }
            }, 60000);
        },

        recordUploadActionError(item, error) {
            const record = { ...item, message: error.message };
            this.uploadFailures = this.uploadFailures.filter(current => current.jobId !== item.jobId);
            this.uploadStatusUnknowns = this.uploadStatusUnknowns.filter(current => current.jobId !== item.jobId);
            if (error.isUploadStatusUnknown) this.uploadStatusUnknowns.push(record);
            else {
                if (error.isTerminalUploadFailure) {
                    record.retryable = error.retryable;
                    record.recoveryRequired = error.recoveryRequired;
                    record.attempt = error.attempt;
                }
                this.uploadFailures.push(record);
            }
            this.uploadProgress = error.message;
            this.uploadProgressCollapsed = false;
            this.showToast(error.message, error.isUploadStatusUnknown ? 'warning' : 'danger');
        },

        async retryUploadJob(item) {
            if (!item?.jobId || !item.retryable || item.recoveryRequired
                || this.isUploading || this.uploadRecoveryBusy[item.jobId]) return;
            this.uploadRecoveryBusy[item.jobId] = true;
            this.isUploading = true;
            this.activeUploadJobId = item.jobId;
            this.uploadProgressCollapsed = false;
            this.uploadProgress = `正在重试处理 ${item.filename}`;
            try {
                try {
                    await this.postUploadAction(`/api/documents/upload/jobs/${encodeURIComponent(item.jobId)}/retry`);
                } catch (error) {
                    if (error.isUploadRequestRejected) throw error;
                    this.uploadProgress = '重试请求响应未确认，正在核对后台状态';
                }
                const job = await this.waitForUploadJob(item.jobId);
                await this.loadDocuments();
                this.recordUploadJob(job);
                this.showToast(job.message || '文档处理完成', 'success');
            } catch (error) {
                this.recordUploadActionError(item, error);
            } finally {
                this.isUploading = false;
                delete this.uploadRecoveryBusy[item.jobId];
            }
        },

        async recoverUploadData(item) {
            if (!item?.jobId || !item.recoveryRequired
                || this.isUploading || this.uploadRecoveryBusy[item.jobId]) return;
            this.uploadRecoveryBusy[item.jobId] = true;
            this.isUploading = true;
            this.activeUploadJobId = item.jobId;
            this.uploadProgress = `正在恢复 ${item.filename} 的数据`;
            this.uploadProgressCollapsed = false;
            try {
                let actionError;
                try {
                    await this.postUploadAction('/api/documents/upload/recover');
                } catch (error) {
                    if (error.isUploadRequestRejected) throw error;
                    actionError = error;
                }
                const job = await this.fetchUploadJob(item.jobId).catch(error => {
                    throw this.uploadStatusUnknownError(item.jobId, error.message);
                });
                this.syncUploadJob(job);
                this.recordUploadJob(job);
                if (actionError && job.recovery_required) {
                    this.showToast(`数据仍待恢复：${actionError.message}`, 'warning');
                } else {
                    this.showToast(job.recovery_required ? '数据仍待恢复，请稍后再次尝试' : '数据恢复已确认，可重试处理文档',
                        job.recovery_required ? 'warning' : 'success');
                }
                this.uploadProgressCollapsed = false;
            } catch (error) {
                this.recordUploadActionError(item, error);
            } finally {
                this.isUploading = false;
                delete this.uploadRecoveryBusy[item.jobId];
            }
        },

        handleFileSelect(event) {
            const files = Array.from(event.target.files || []);
            const supportedFiles = files.filter(file => this.isSupportedUploadFile(file));
            const skippedCount = files.length - supportedFiles.length;
            if (supportedFiles.length > 0) {
                this.selectedFiles = supportedFiles;
                this.uploadProgress = '';
                this.uploadSteps = this.createUploadSteps();
                this.uploadProgressCollapsed = false;
                this.activeUploadJobId = '';
            }
            if (skippedCount > 0) {
                this.showToast(`已跳过 ${skippedCount} 个不支持的文件`, 'warning');
            }
            if (!supportedFiles.length && files.length) {
                this.selectedFiles = [];
                this.showToast('所选内容中没有支持的文档', 'warning');
            }
        },

        isSupportedUploadFile(file) {
            return /\.(pdf|docx?|xlsx?|md|markdown)$/i.test(file?.name || '');
        },

        createUploadSteps() {
            return [
                { key: 'upload', label: '文档上传', percent: 0, status: 'pending', message: '' },
                { key: 'parse', label: '解析与分块', percent: 0, status: 'pending', message: '' },
                { key: 'cleanup', label: '清理旧版本', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '父级分块入库', percent: 0, status: 'pending', message: '' },
                { key: 'vector_store', label: '向量化入库', percent: 0, status: 'pending', message: '' },
            ];
        },

        updateUploadStep(key, percent, status = 'running', message = '') {
            if (!this.uploadSteps.length) {
                this.uploadSteps = this.createUploadSteps();
            }
            const idx = this.uploadSteps.findIndex(step => step.key === key);
            if (idx === -1) return;
            this.uploadSteps[idx] = {
                ...this.uploadSteps[idx],
                percent: Math.max(0, Math.min(100, Math.round(percent || 0))),
                status,
                message
            };
        },

        uploadFileWithProgress(file) {
            return new Promise((resolve, reject) => {
                const crypto = globalThis.crypto;
                const uploadId = crypto?.randomUUID
                    ? crypto.randomUUID().replace(/-/g, '')
                    : crypto?.getRandomValues
                        ? Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, '0')).join('')
                        : null;
                if (!uploadId) {
                    reject(new Error('当前浏览器无法安全生成上传任务编号，请使用支持 Web Crypto 的浏览器'));
                    return;
                }
                this.activeUploadJobId = uploadId;
                const candidates = this.uploadMutationPathCandidates('/api/documents/upload/async');
                const confirmSubmission = async () => {
                    this.uploadProgress = '上传响应未确认，正在核对后台任务状态';
                    try {
                        resolve(await this.fetchUploadJob(uploadId));
                    } catch (error) {
                        reject(this.uploadStatusUnknownError(uploadId, error.message));
                    }
                };
                const sendAttempt = (index) => {
                    const xhr = new XMLHttpRequest();
                    const formData = new FormData();
                    formData.append('file', file, file.name);
                    formData.append('upload_id', uploadId);
                    const candidate = candidates[index];

                    xhr.open('POST', candidate);
                    xhr.timeout = 10 * 60 * 1000;
                    const headers = this.authHeaders();
                    Object.entries(headers).forEach(([key, value]) => xhr.setRequestHeader(key, value));

                    xhr.upload.onprogress = (event) => {
                        if (!event.lengthComputable) return;
                        const percent = Math.round((event.loaded / event.total) * 100);
                        this.updateUploadStep('upload', percent, 'running', `已上传 ${percent}%`);
                    };

                    xhr.onload = () => {
                        const mayTryAlternate = (xhr.status === 404 || xhr.status === 405)
                            && index < candidates.length - 1;
                        if (mayTryAlternate) {
                            sendAttempt(index + 1);
                            return;
                        }
                        this.rememberApiPrefix(candidate);
                        if (xhr.status === 401) {
                            const message = '登录已过期，请重新登录';
                            this.handleLogout({ authError: message });
                            reject(new Error(message));
                            return;
                        }

                        let data;
                        try {
                            data = JSON.parse(xhr.responseText || '{}');
                        } catch (e) { data = null; }

                        if (xhr.status >= 400 && xhr.status < 500) {
                            reject(new Error(data?.detail || `HTTP ${xhr.status}`));
                            return;
                        }
                        if (xhr.status < 200 || xhr.status >= 300 || !data?.job_id) {
                            confirmSubmission();
                            return;
                        }

                        this.updateUploadStep('upload', 100, 'completed', '文档上传完成');
                        resolve(data);
                    };

                    xhr.onerror = confirmSubmission;
                    xhr.onabort = confirmSubmission;
                    xhr.ontimeout = confirmSubmission;
                    xhr.send(formData);
                };
                sendAttempt(0);
            });
        },

        syncUploadJob(job) {
            this.activeUploadJobId = job.job_id;
            const batchPrefix = this.uploadBatchTotal > 0
                ? `[${this.uploadBatchCurrent}/${this.uploadBatchTotal}] `
                : '';
            this.uploadProgress = batchPrefix + (job.message || '');
            if (Array.isArray(job.steps)) {
                this.uploadSteps = job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                }));
            }
            // 入库成功后自动收起步骤明细，保留摘要供用户再次展开查看。
            if (job.status === 'completed') {
                this.uploadProgressCollapsed = true;
            }
        },

        toggleSidebar() {
            if (window.matchMedia('(max-width: 680px)').matches) {
                this.sidebarCollapsed = false;
                this.mobileSidebarOpen = !this.mobileSidebarOpen;
                return;
            }
            this.sidebarCollapsed = !this.sidebarCollapsed;
        },
        toggleUploadProgressCollapsed() {
            this.uploadProgressCollapsed = !this.uploadProgressCollapsed;
        },

        stopUploadJobPolling() {
            if (this.uploadPollTimer) {
                clearInterval(this.uploadPollTimer);
                this.uploadPollTimer = null;
            }
        },

        startUploadJobPolling(jobId) {
            this.stopUploadJobPolling();

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/api/documents/upload/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load upload job');
                    }

                    const job = await response.json();
                    this.syncUploadJob(job);

                    if (job.status === 'completed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                        this.selectedFiles = [];
                        if (this.$refs.fileInput) {
                            this.$refs.fileInput.value = '';
                        }
                        if (this.$refs.folderInput) {
                            this.$refs.folderInput.value = '';
                        }
                        await this.loadDocuments();
                        this.showToast(job.message || '文档上传完成', 'success');
                    } else if (job.status === 'failed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                        this.showToast(job.message || '文档上传失败', 'danger');
                    }
                } catch (error) {
                    this.uploadProgress = '进度查询失败：' + error.message;
                    this.stopUploadJobPolling();
                    this.isUploading = false;
                    this.showToast(this.uploadProgress, 'danger');
                }
            };

            poll();
            this.uploadPollTimer = setInterval(poll, 1000);
        },

        async waitForUploadJob(jobId, maxWaitMs = 10 * 60 * 1000) {
            let consecutiveErrors = 0;
            const deadline = Date.now() + maxWaitMs;
            while (true) {
                if (Date.now() >= deadline) {
                    throw this.uploadStatusUnknownError(jobId, '已达到等待时限，后台任务可能仍在处理');
                }
                try {
                    const job = await this.fetchUploadJob(jobId, Math.min(15000, deadline - Date.now()));
                    consecutiveErrors = 0;
                    this.syncUploadJob(job);
                    if (job.status === 'completed') return job;
                    if (job.status === 'failed') {
                        const terminalError = new Error(job.message || '文档处理失败');
                        terminalError.isTerminalUploadFailure = true;
                        terminalError.jobId = jobId;
                        terminalError.retryable = job.retryable === true;
                        terminalError.recoveryRequired = job.recovery_required === true;
                        terminalError.attempt = job.attempt || 1;
                        throw terminalError;
                    }
                } catch (error) {
                    if (error.isTerminalUploadFailure) throw error;
                    consecutiveErrors += 1;
                    if (consecutiveErrors >= 3) {
                        throw this.uploadStatusUnknownError(jobId, error.message);
                    }
                    const prefix = this.uploadBatchTotal > 0
                        ? `[${this.uploadBatchCurrent}/${this.uploadBatchTotal}] `
                        : '';
                    this.uploadProgress = `${prefix}进度查询暂时失败，正在重试（${consecutiveErrors}/3）`;
                    await this.waitBeforeUploadPoll(Math.min(500 * (2 ** (consecutiveErrors - 1)), Math.max(0, deadline - Date.now())));
                    continue;
                }
                await this.waitBeforeUploadPoll(Math.min(1000, Math.max(0, deadline - Date.now())));
            }
        },

        waitBeforeUploadPoll(delayMs) {
            return new Promise(resolve => setTimeout(resolve, delayMs));
        },

        async uploadDocument() {
            if (this.isUploading) return;
            if (this.uploadStatusUnknowns.length) {
                this.showToast('请先重新查询待确认的后台任务状态，再继续上传', 'warning');
                return;
            }
            if (!this.selectedFiles.length) {
                this.showToast('请先选择文件或文件夹', 'warning');
                return;
            }

            const files = [...this.selectedFiles];
            this.isUploading = true;
            this.uploadBatchCurrent = 0;
            this.uploadBatchTotal = files.length;
            this.uploadProgressCollapsed = false;
            let successCount = 0;
            const failures = [];
            let pausedIndex = -1;

            for (let index = 0; index < files.length; index += 1) {
                const file = files[index];
                this.uploadBatchCurrent = index + 1;
                this.activeUploadJobId = '';
                this.uploadSteps = this.createUploadSteps();
                this.uploadProgressCollapsed = false;
                this.uploadProgress = `[${index + 1}/${files.length}] 正在上传 ${file.name}`;
                this.updateUploadStep('upload', 0, 'running', `准备上传 ${file.name}`);
                let uploadSubmitted = false;
                try {
                    const data = await this.uploadFileWithProgress(file);
                    uploadSubmitted = true;
                    this.activeUploadJobId = data.job_id;
                    await this.waitForUploadJob(data.job_id);
                    successCount += 1;
                } catch (error) {
                    if (error.isUploadStatusUnknown) {
                        this.uploadStatusUnknowns.push({
                            filename: file.name,
                            message: error.message,
                            jobId: error.jobId || this.activeUploadJobId,
                        });
                        pausedIndex = index;
                        break;
                    }
                    const failure = { filename: file.name, message: error.message };
                    if (error.jobId) {
                        failure.jobId = error.jobId;
                        failure.retryable = error.retryable === true;
                        failure.recoveryRequired = error.recoveryRequired === true;
                        failure.attempt = error.attempt || 1;
                    }
                    failures.push(failure);
                    if (!uploadSubmitted) {
                        this.updateUploadStep('upload', 100, 'failed', `${file.name}：${error.message}`);
                    }
                }
            }

            this.isUploading = false;
            this.selectedFiles = pausedIndex >= 0 ? files.slice(pausedIndex + 1) : [];
            if (pausedIndex < 0) {
                this.activeUploadJobId = '';
                if (this.$refs.fileInput) this.$refs.fileInput.value = '';
                if (this.$refs.folderInput) this.$refs.folderInput.value = '';
            }
            if (successCount > 0) await this.loadDocuments();

            const failedJobIds = new Set(failures.map(item => item.jobId).filter(Boolean));
            this.uploadFailures = this.uploadFailures.filter(item => !failedJobIds.has(item.jobId)).concat(failures);
            this.uploadBatchCurrent = 0;
            this.uploadBatchTotal = 0;
            this.uploadProgress = `批量上传完成：成功 ${successCount} 个，失败 ${failures.length} 个`;
            if (this.uploadStatusUnknowns.length) {
                this.uploadProgress = (
                    `批量上传已暂停：${this.uploadStatusUnknowns[0].filename} 的后台状态待确认，`
                    + `剩余 ${this.selectedFiles.length} 个文件在确认状态后可继续上传`
                );
            }
            this.uploadProgressCollapsed = failures.length === 0 && this.uploadStatusUnknowns.length === 0;
            this.showToast(
                this.uploadProgress,
                failures.length || this.uploadStatusUnknowns.length ? 'warning' : 'success'
            );
        },

        createDeleteSteps() {
            return [
                { key: 'prepare', label: '准备删除', percent: 0, status: 'pending', message: '' },
                { key: 'bm25', label: '同步 BM25 统计', percent: 0, status: 'pending', message: '' },
                { key: 'milvus', label: '删除向量数据', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '删除父级分块', percent: 0, status: 'pending', message: '' },
            ];
        },

        isDeletingDocument(filename) {
            const job = this.deleteJobs[filename];
            return job && job.status === 'running';
        },

        isDeleteActionLocked(filename) {
            const job = this.deleteJobs[filename];
            return this.deleteRecoveryBusy || (job && (['running', 'pending', 'unknown', 'completed'].includes(job.status) || job.recoveryRequired));
        },

        getDeleteButtonIcon(filename) {
            const job = this.deleteJobs[filename];
            if (job?.status === 'running') return 'spinner';
            if (job?.status === 'completed') return 'check';
            return 'trash';
        },

        setDeleteJob(filename, nextJob) {
            this.deleteJobs = {
                ...this.deleteJobs,
                [filename]: {
                    ...(this.deleteJobs[filename] || {}),
                    ...nextJob
                }
            };
        },

        syncDeleteJob(filename, job) {
            const current = this.deleteJobs[filename] || {};
            if (current.jobId && current.jobId !== job.job_id) {
                this.stopDeleteJobPolling(filename);
                this.clearDeleteRemovalTimer(filename);
            }
            // 后端返回统一的步骤结构，前端只负责同步到当前文档行内卡片。
            this.setDeleteJob(filename, {
                jobId: job.job_id,
                status: job.status === 'pending' ? 'running' : job.status,
                message: job.message || '',
                recoveryRequired: job.recovery_required === true,
                retryable: job.retryable === true,
                collapsed: job.status === 'completed' ? true : Boolean(current.collapsed),
                steps: Array.isArray(job.steps) ? job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                })) : this.createDeleteSteps()
            });
        },

        toggleDeleteJobCollapsed(filename) {
            const job = this.deleteJobs[filename];
            if (!job) return;
            this.setDeleteJob(filename, { collapsed: !job.collapsed });
        },

        getDeleteTaskStepsId(filename) {
            return `delete-task-steps-${encodeURIComponent(filename)}`;
        },

        stopDeleteJobPolling(filename) {
            const timer = this.deletePollTimers[filename];
            if (!timer) return;
            clearInterval(timer);
            const { [filename]: _removed, ...rest } = this.deletePollTimers;
            this.deletePollTimers = rest;
        },

        stopAllDeleteJobPolling() {
            Object.keys(this.deletePollTimers).forEach(filename => this.stopDeleteJobPolling(filename));
        },

        clearDeleteRemovalTimer(filename) {
            const timer = this.deleteRemoveTimers[filename];
            if (!timer) return;
            clearTimeout(timer);
            const { [filename]: _removed, ...rest } = this.deleteRemoveTimers;
            this.deleteRemoveTimers = rest;
        },

        scheduleDeletedDocumentRemoval(filename) {
            this.clearDeleteRemovalTimer(filename);
            // 删除完成后先保留 3 秒摘要，再从当前列表移除并刷新后端状态。
            const timer = setTimeout(async () => {
                this.documents = this.documents.filter(doc => doc.filename !== filename);
                const { [filename]: _job, ...jobs } = this.deleteJobs;
                const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
                this.deleteJobs = jobs;
                this.deleteRemoveTimers = timers;
                await this.loadDocuments();
            }, 3000);
            this.deleteRemoveTimers = {
                ...this.deleteRemoveTimers,
                [filename]: timer
            };
        },

        async recheckDeleteJob(filename) {
            const jobId = this.deleteJobs[filename]?.jobId;
            if (!jobId) return;
            try {
                const job = await this.withUploadRequestTimeout(async signal => {
                    const response = await this.authFetch(`/api/documents/delete/jobs/${encodeURIComponent(jobId)}`, { signal });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const value = await response.json();
                    if (value.job_id !== jobId || !['pending', 'running', 'completed', 'failed'].includes(value.status)) {
                        throw new Error('删除任务状态无效');
                    }
                    return value;
                }, 15000);
                if (this.deleteJobs[filename]?.jobId !== jobId) return;
                const previousStatus = this.deleteJobs[filename]?.status;
                this.syncDeleteJob(filename, job);
                if (job.status === 'completed') {
                    this.stopDeleteJobPolling(filename);
                    if (previousStatus !== 'completed') {
                        this.scheduleDeletedDocumentRemoval(filename);
                        this.showToast(job.message || '删除已完成', 'success');
                    }
                } else if (job.status === 'failed') {
                    this.stopDeleteJobPolling(filename);
                    if (previousStatus !== 'failed') this.showToast(job.message || '删除失败，已保留任务记录', 'danger');
                }
                return job;
            } catch (error) {
                if (this.deleteJobs[filename]?.jobId !== jobId) return;
                this.setDeleteJob(filename, {
                    status: 'unknown', collapsed: false,
                    message: '删除状态待确认，请重新查询：' + error.message,
                });
                this.stopDeleteJobPolling(filename);
            }
        },

        startDeleteJobPolling(filename, jobId) {
            this.stopDeleteJobPolling(filename);
            const started = Date.now();
            let checking = false;
            const poll = async () => {
                if (checking || this.deleteJobs[filename]?.jobId !== jobId) return;
                if (Date.now() - started >= 10 * 60 * 1000) {
                    this.setDeleteJob(filename, { status: 'unknown', message: '删除等待超时，请重新查询后台状态' });
                    this.stopDeleteJobPolling(filename);
                    return;
                }
                checking = true;
                try {
                    await this.recheckDeleteJob(filename);
                } finally {
                    checking = false;
                }
            };
            this.deletePollTimers = { ...this.deletePollTimers, [filename]: setInterval(poll, 1000) };
            poll();
        },

        async recoverDeleteData(filename) {
            if (this.deleteRecoveryBusy || !this.deleteJobs[filename]?.jobId) return;
            this.deleteRecoveryBusy = true;
            try {
                let actionError;
                try {
                    await this.postUploadAction('/api/documents/recover');
                } catch (error) {
                    actionError = error;
                }
                const job = await this.recheckDeleteJob(filename);
                if (job && !job.recovery_required) {
                    this.showToast('恢复已确认。如仍需删除，请重新确认删除操作。', 'success');
                    await this.loadDocuments();
                } else {
                    this.showToast(actionError?.message || '恢复尚未确认，请恢复依赖后重新查询', 'warning');
                }
            } finally {
                this.deleteRecoveryBusy = false;
            }
        },

        deleteDocument(filename) {
            if (this.isDeleteActionLocked(filename)) return;
            this.requestConfirmation({
                title: '删除文档索引？',
                message: `将删除“${filename}”当前版本的检索索引；本地原文件保留。`,
                confirmLabel: '删除索引',
                tone: 'danger',
                onConfirm: () => this.performDeleteDocument(filename)
            });
        },

        async performDeleteDocument(filename) {
            if (this.isDeleteActionLocked(filename)) return;
            this.clearDeleteRemovalTimer(filename);
            let jobId;
            try {
                const crypto = globalThis.crypto;
                jobId = crypto?.randomUUID ? crypto.randomUUID().replace(/-/g, '')
                    : crypto?.getRandomValues
                        ? Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, '0')).join('')
                        : null;
                if (!jobId) throw new Error('当前浏览器无法安全生成删除任务编号');
                this.setDeleteJob(filename, {
                    jobId, status: 'running', recoveryRequired: false, retryable: false,
                    message: '正在提交删除任务...', collapsed: false, steps: this.createDeleteSteps(),
                });
                const data = await this.postUploadAction(
                    `/api/documents/delete/async/${encodeURIComponent(filename)}?delete_id=${jobId}`, 'DELETE'
                );
                if (data.job_id !== jobId) throw new Error('删除响应未确认');
                const job = await this.recheckDeleteJob(filename);
                if (job && ['pending', 'running'].includes(job.status)) this.startDeleteJobPolling(filename, jobId);
            } catch (error) {
                if (jobId && !error.isUploadRequestRejected) {
                    // A lost response does not prove failure. Query the original ID,
                    // never issue another DELETE that could target a newer upload.
                    const job = await this.recheckDeleteJob(filename);
                    if (job && ['pending', 'running'].includes(job.status)) this.startDeleteJobPolling(filename, jobId);
                } else {
                    this.setDeleteJob(filename, { status: 'failed', message: error.message, collapsed: false });
                    this.showToast(error.message, 'danger');
                }
            }
        },

        getFileIcon(fileType) {
            if (fileType === 'PDF') {
                return 'file-pdf';
            } else if (fileType === 'Word') {
                return 'file-word';
            } else if (fileType === 'Excel') {
                return 'file-excel';
            }
            return 'file';
        },

        formatFileSize(bytes) {
            const size = Number(bytes) || 0;
            if (size < 1024) return `${size} B`;
            if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
            return `${(size / (1024 * 1024)).toFixed(1)} MB`;
        },

        getThinkingIcon(icon) {
            const aliases = {
                '🔍': 'search',
                '📊': 'chart',
                '✏️': 'edit',
                '🧠': 'brain',
                '📝': 'edit',
                '🔄': 'sync',
                '⚠️': 'warning',
                '🔧': 'tool',
                '✅': 'check',
                '🤔': 'spark',
                '▶': 'spark'
            };
            const allowedIcons = new Set([
                'brand', 'search', 'chart', 'edit', 'brain', 'sync', 'warning',
                'tool', 'check', 'spark'
            ]);
            const resolvedIcon = aliases[icon] || icon;
            return allowedIcons.has(resolvedIcon) ? resolvedIcon : 'spark';
        }
    },
    watch: {
        activeWorkspace() { this.syncDiagnosticsPolling(); },
        isAdmin() {
            if (!this.isAdmin) this.clearDiagnostics();
            else this.syncDiagnosticsPolling();
        },
        token() { this.syncDiagnosticsPolling(); },
        messages: {
            handler() {
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            },
            deep: true
        }
    }
}).mount('#app');
