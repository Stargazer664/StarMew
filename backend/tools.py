from typing import Optional
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from typing import Any
from .tool_contracts import ToolResult, tool_error, tool_exception
from .agent_budget import remaining_timeout
from .run_control import RunCancelled

# 加载 .env 文件
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def tool_is_available(name: str) -> bool:
    """Configuration availability for this turn's model tool binding."""
    if name == 'get_current_weather':
        return bool(AMAP_WEATHER_API and AMAP_API_KEY)
    if name == 'web_search':
        return bool(TAVILY_API_KEY)
    return name == 'search_knowledge_base'

@dataclass
class _RagRequestContext:
    # ContextVar copies across asyncio tasks / LangGraph worker threads share this
    # request-owned object, so tool trace writes are visible to the SSE consumer.
    last_rag_context: Optional[dict] = None
    step_queue: Any = None
    step_loop: Any = None
    answer_selection: Any = None


_RAG_REQUEST: ContextVar[Optional[_RagRequestContext]] = ContextVar("rag_request", default=None)


def _request_context() -> _RagRequestContext:
    context = _RAG_REQUEST.get()
    if context is None:
        context = _RagRequestContext()
        _RAG_REQUEST.set(context)
    return context


def _set_last_rag_context(context: dict):
    request = _request_context()
    new = context.get('rag_trace', {})
    old = (request.last_rag_context or {}).get('rag_trace', {})
    rounds = old.get('retrieval_rounds', []) + [dict(new)]
    records = [record for round_ in rounds for record in round_.get('retrieval_records', [])]
    grades = [record for round_ in rounds for record in round_.get('grading_records', [])]
    filters = [record for round_ in rounds for record in round_.get('relevance_filters', [])]
    chunks, seen = [], set()
    for round_ in rounds:
        for doc in round_.get('retrieved_chunks', []):
            key = (doc.get('filename'), doc.get('page_number'), doc.get('text'))
            if key not in seen:
                seen.add(key)
                chunks.append(doc)
    request.last_rag_context = {'rag_trace': {**new, 'retrieval_rounds': rounds, 'retrieval_records': records, 'grading_records': grades, 'relevance_filters': filters, 'retrieved_chunks': chunks}}


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取本轮聚合 RAG 检索上下文，默认读取后清空。"""
    request = _request_context()
    context = request.last_rag_context
    if clear:
        request.last_rag_context = None
    return context


def set_answer_evidence(selection):
    """Persist the exact final input mapping without replacing retrieval diagnostics."""
    request = _request_context()
    request.answer_selection = selection
    if request.last_rag_context is None and not selection.snippets and not selection.omitted_count:
        return
    context = request.last_rag_context or {}
    trace = context.get('rag_trace', {})
    request.last_rag_context = {**context, 'rag_trace': {**trace,
        'answer_evidence': [asdict(s) for s in selection.snippets],
        'evidence_omitted_count': selection.omitted_count}}


def get_answer_evidence():
    return _request_context().answer_selection


def reset_tool_call_guards():
    """每轮对话创建独立上下文，随后再绑定本轮 SSE 队列。"""
    _RAG_REQUEST.set(_RagRequestContext())


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    context = _request_context()
    context.step_queue = queue
    if queue:
        import asyncio
        try:
            context.step_loop = asyncio.get_running_loop()
        except RuntimeError:
            context.step_loop = None
    else:
        context.step_loop = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    context = _request_context()
    queue, loop = context.step_queue, context.step_loop
    if queue is not None and loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass


@tool("get_current_weather")
def get_current_weather(location: str, extensions: Optional[str] = "base") -> ToolResult:
    """获取天气信息"""
    if not location or not location.strip():
        return tool_error('INVALID_ARGUMENT')
    if extensions not in ("base", "all"):
        return tool_error('INVALID_ARGUMENT')

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return tool_error('NOT_CONFIGURED')

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=remaining_timeout(10))
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return tool_error('UPSTREAM_ERROR')

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return ToolResult('empty', f"未查询到 {location} 的天气数据")
            w = lives[0]
            return ToolResult('ok', (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            ))

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return ToolResult('empty', f"未查询到 {location} 的天气预报数据")
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return ToolResult('ok', "\n".join(out))

    except RunCancelled:
        raise
    except Exception as e:
        return tool_exception(e)


@tool("web_search")
def web_search(query: str) -> ToolResult:
    """Search the web for real-time or external information when the knowledge base is insufficient or the question requires up-to-date facts."""
    if not query or not query.strip():
        return tool_error('INVALID_ARGUMENT')
    if not TAVILY_API_KEY:
        return tool_error('NOT_CONFIGURED')

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
            },
            timeout=remaining_timeout(15),
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ToolResult('empty', f"未搜索到和 '{query}' 相关的结果。", evidence_status='insufficient')

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            content = r.get("content", "")[:300]
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {content}")
        return ToolResult('ok', "\n\n".join(lines), evidence=results, evidence_status='unknown')
    except RunCancelled:
        raise
    except Exception as e:
        return tool_exception(e)


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> ToolResult:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    if not query or not query.strip():
        return tool_error('INVALID_ARGUMENT')

    try:
        from .rag_pipeline import run_rag_graph
        from .telemetry import span
        with span("rag_pipeline", component="rag"):
            rag_result = run_rag_graph(query)
    except RunCancelled:
        raise
    except Exception as exc:
        return tool_exception(exc)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    status = rag_trace.get('retrieval_status')
    evidence_status = rag_trace.get('evidence_status', 'unknown')
    if status == 'expansion_failed':
        result = tool_error(rag_trace.get('expansion_error_code') or 'UPSTREAM_ERROR', retryable=rag_trace.get('expansion_retryable', False))
        result.evidence_status = 'unknown'
        return result
    if status == 'infrastructure_failure':
        failed = next((r for r in reversed(rag_trace.get('retrieval_records', [])) if r['status'] == status), {})
        result = tool_error(failed.get('meta', {}).get('failure_code') or 'UPSTREAM_ERROR', retryable=failed.get('meta', {}).get('retryable', False) or failed.get('meta', {}).get('failure_code') == 'UPSTREAM_TIMEOUT')
        result.evidence_status = 'unknown'
        return result
    if status == 'grading_failed':
        return ToolResult('error', '文档相关性评分不可用，当前资料无法确认。', code='UPSTREAM_ERROR', evidence_status='unknown')
    if status == 'rerank_failed':
        return ToolResult('error', '候选片段相关性打分不可用，当前资料无法确认。', code='UPSTREAM_ERROR', evidence_status='unknown')

    if not docs:
        gap = {'budget_exhausted': '检索预算已用尽，当前资料无法确认。', 'duplicate_query': '扩展查询重复，当前资料无法确认。', 'below_threshold': '候选片段均未达到相关性阈值，当前证据不足。'}.get(status, '知识库没有相关片段，当前资料无法确认。')
        return ToolResult('empty', gap, evidence_status='insufficient')

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    warning = '' if evidence_status == 'sufficient' else '当前资料无法确认；以下片段尚不足以支持答案。' + ('检索预算已用尽。' if status == 'budget_exhausted' else '') + '\n'
    return ToolResult('ok', warning + "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted),
                      evidence=docs, evidence_status=evidence_status)
