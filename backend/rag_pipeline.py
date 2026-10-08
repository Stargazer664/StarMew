from typing import Literal, TypedDict, List, Optional
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .rag_utils import retrieve_documents, step_back_expand, generate_hypothetical_document
from .tools import emit_rag_step
from .agent_budget import AgentRunState, call_signature, current_run_state, use_run_state, model_timeout_kwargs, remaining_timeout
from .llm_config import create_chat_model, resolve_llm_settings
from .chat_diagnostics import model_call, invoke_model
from .telemetry import timed, span
from .conversation_context import ContextBudgetExceeded, model_budget, assert_model_input, model_output_kwargs, structured_output_kwargs
from .evidence_context import aggregate_evidence, select_evidence, render_evidence
from .run_control import RunCancelled, check_cancelled

load_dotenv()

_grader_model = None
_router_model = None


def init_chat_model(*, role="generation", **kwargs):
    """Compatibility injection point used by tests and downstream integrations."""
    return create_chat_model(role=role, **kwargs)


def _get_grader_model():
    global _grader_model
    settings = resolve_llm_settings(role="judge")
    if not settings.model:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            role="judge",
            temperature=0,
            stream_usage=True,
        )
    return _grader_model


def _get_router_model():
    global _router_model
    settings = resolve_llm_settings(role="generation")
    if not settings.model:
        return None
    if _router_model is None:
        _router_model = init_chat_model(
            role="generation",
            temperature=0,
            stream_usage=True,
        )
    return _router_model


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "Assess whether these documents contain explicit evidence sufficient to answer the user question. "
    "Related keywords or general background alone are not sufficient. Do not invent missing facts. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RewriteStrategy(BaseModel):
    """Choose a query expansion strategy."""

    strategy: Literal["step_back", "hyde", "complex"]


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    expansion_type: Optional[str]
    expanded_query: Optional[str]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    hypothetical_doc: Optional[str]
    rag_trace: Optional[dict]


def _retrieve(query, stage, records):
    """Keep each actual attempt, including safe failure/budget metadata."""
    from .tool_contracts import tool_exception
    check_cancelled()
    signature = call_signature('retrieve', {'query': query})
    if any(call_signature('retrieve', {'query': record['query']}) == signature for record in records):
        result = {"docs": [], "meta": {"retrieval_mode": "skipped", "retrieval_status": "duplicate_query"}}
        records.append({"query": query, "stage": stage, "status": "duplicate_query", "actual": False, "docs": [], "meta": result['meta']})
        return result
    try:
        result = retrieve_documents(query, top_k=5)
        remaining_timeout()
    except RunCancelled:
        raise
    except Exception as exc:
        failure = tool_exception(exc)
        result = {"docs": [], "meta": {"retrieval_mode": "failed", "failure_code": failure.code, "retryable": failure.retryable}}
    meta = result.get("meta", {})
    status = meta.get("retrieval_status") or ("infrastructure_failure" if meta.get("retrieval_mode") == "failed" else "retrieved" if result.get("docs") else "zero_results")
    records.append({"query": query, "stage": stage, "status": status, "actual": status not in ('budget_exhausted', 'duplicate_query'), "docs": result.get("docs", []), "meta": dict(meta)})
    return result


def _retrieval_status(records, docs):
    statuses = {record['status'] for record in records}
    for status in ('infrastructure_failure', 'rerank_failed', 'budget_exhausted'):
        if status in statuses:
            return status
    if not docs and 'below_threshold' in statuses:
        return 'below_threshold'
    return 'retrieved' if docs else 'duplicate_query' if 'duplicate_query' in statuses else 'zero_results'


def _relevance_filters(records):
    return [{'stage': record['stage'], **record['meta']['relevance_filter']}
            for record in records if isinstance(record.get('meta', {}).get('relevance_filter'), dict)]


def _emit_retrieval_result(status, count, *, expanded=False):
    if status == 'infrastructure_failure':
        emit_rag_step('❌', '检索服务失败，尚未获取可靠结果')
    elif status == 'rerank_failed':
        emit_rag_step('❌', '相关性打分不可用，当前资料无法确认')
    elif status == 'below_threshold':
        emit_rag_step('⚠️', '候选片段均未达到相关性阈值，当前证据不足')
    elif status in ('budget_exhausted', 'duplicate_query'):
        emit_rag_step('⏭️', '检索预算已用尽' if status == 'budget_exhausted' else '已跳过重复扩展查询')
    else:
        emit_rag_step('✅', f"{'扩展检索' if expanded else '检索'}完成，找到 {count} 个片段")


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{i}] {source} (Page {page}):\n{text}")
    return "\n\n---\n\n".join(chunks)


def retrieve_initial(state: RAGState) -> RAGState:
    check_cancelled()
    remaining_timeout()
    query = state["question"]
    emit_rag_step("🔍", "正在检索知识库...")
    records = []
    retrieved = _retrieve(query, "initial", records)
    results = retrieved.get("docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(results)
    emit_rag_step(
        "🧱",
        "三级分块检索",
        (
            f"叶子层 L{retrieve_meta.get('leaf_retrieve_level', 3)} 召回，"
            f"候选 {retrieve_meta.get('candidate_k', 0)}"
        ),
    )
    emit_rag_step(
        "🧩",
        "Auto-merging 合并",
        (
            f"启用: {bool(retrieve_meta.get('auto_merge_enabled'))}，"
            f"应用: {bool(retrieve_meta.get('auto_merge_applied'))}，"
            f"替换片段: {retrieve_meta.get('auto_merge_replaced_chunks', 0)}"
        ),
    )
    _emit_retrieval_result(records[-1]['status'], len(results))
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": results,
        "initial_retrieved_chunks": results,
        "retrieval_stage": "initial",
        "retrieval_records": records,
        "retrieval_status": records[-1]["status"],
        "relevance_filters": _relevance_filters(records),
        "rerank_enabled": retrieve_meta.get("rerank_enabled"),
        "rerank_applied": retrieve_meta.get("rerank_applied"),
        "rerank_model": retrieve_meta.get("rerank_model"),
        "rerank_endpoint": retrieve_meta.get("rerank_endpoint"),
        "rerank_error": retrieve_meta.get("rerank_error"),
        "retrieval_mode": retrieve_meta.get("retrieval_mode"),
        "candidate_k": retrieve_meta.get("candidate_k"),
        "leaf_retrieve_level": retrieve_meta.get("leaf_retrieve_level"),
        "auto_merge_enabled": retrieve_meta.get("auto_merge_enabled"),
        "auto_merge_applied": retrieve_meta.get("auto_merge_applied"),
        "auto_merge_threshold": retrieve_meta.get("auto_merge_threshold"),
        "auto_merge_replaced_chunks": retrieve_meta.get("auto_merge_replaced_chunks"),
        "auto_merge_steps": retrieve_meta.get("auto_merge_steps"),
    }
    return {
        "query": query,
        "docs": results,
        "context": context,
        "rag_trace": rag_trace,
    }


@timed("grading", component="rag")
def grade_documents_node(state: RAGState) -> RAGState:
    check_cancelled()
    rag_trace = state.get("rag_trace", {}) or {}
    run = current_run_state()
    exhausted = bool(run and run.retrieval_count >= run.budget.max_retrieval_calls)
    terminal = state.get("expansion_type") is not None or exhausted or bool(run and run.remaining() <= 0)
    def graded(route):
        rag_trace.setdefault('grading_records', []).append({
            'stage': rag_trace.get('retrieval_stage'), 'score': rag_trace.get('grade_score'),
            'evidence_status': rag_trace.get('evidence_status'),
            'retrieval_status': rag_trace.get('retrieval_status'),
            'context_chunks': state.get('docs', []),
        })
        return {'route': route, 'rag_trace': rag_trace}
    route = "generate_answer" if terminal else "rewrite_question"
    if (rag_trace.get("retrieval_status") in ("infrastructure_failure", "rerank_failed") or
            (rag_trace.get("retrieval_status") in ("budget_exhausted", "duplicate_query") and not state.get('docs'))):
        rag_trace.update(evidence_status="unknown" if rag_trace["retrieval_status"] in ("infrastructure_failure", "rerank_failed") else "insufficient", grade_score="unknown", grade_route="generate_answer", rewrite_needed=False)
        return graded('generate_answer')
    if not state.get("docs"):
        status = 'below_threshold' if rag_trace.get('retrieval_status') == 'below_threshold' else 'zero_results'
        rag_trace.update(evidence_status="insufficient", retrieval_status=status, grade_score="no", grade_route=route, rewrite_needed=route == "rewrite_question")
        return graded(route)
    remaining_timeout()
    try:
        grader = _get_grader_model()
    except RunCancelled:
        raise
    except Exception:
        grader = None
    emit_rag_step("📊", "正在评估文档相关性...")
    if not grader:
        with span("model_call", component="model", purpose="grading") as skipped:
            skipped.set_result("skipped")
        grade_update = {
            "grade_score": "unknown",
            "grade_route": route,
            "rewrite_needed": route == "rewrite_question",
            "evidence_status": "unknown",
            "retrieval_status": "grading_failed",
        }
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(grade_update)
        return graded(route)
    question = state["question"]
    budget = model_budget('judge')
    fixed_prompt = GRADE_PROMPT.format(question=question, context='')
    fixed_tokens = assert_model_input(fixed_prompt, role='judge', schema=GradeDocuments, budget=budget)
    selection = select_evidence(aggregate_evidence([state['docs']]), budget.input_limit - fixed_tokens, budget)
    context = render_evidence(selection.snippets)
    rag_trace['grading_evidence'] = {'citation_ids': [s.citation_id for s in selection.snippets],
                                    'omitted_count': selection.omitted_count}
    if not selection.snippets:
        rag_trace.update(evidence_status='insufficient', grade_score='no', grade_route=route,
                         rewrite_needed=route == 'rewrite_question', retrieval_status='evidence_omitted')
        return graded(route)
    prompt = GRADE_PROMPT.format(question=question, context=context)
    structured_call = None
    try:
        structured = grader.with_structured_output(GradeDocuments, **structured_output_kwargs())
        assert_model_input(prompt, role='judge', schema=GradeDocuments, budget=budget)
        with model_call(grader, purpose="grading", role="judge", structured=True) as structured_call:
            response = structured.invoke([{"role": "user", "content": prompt}], **model_timeout_kwargs("grading"), **model_output_kwargs(grader, 'judge'))
            structured_call.observe(response)
        score = (response.binary_score or "").strip().lower()
    except ContextBudgetExceeded:
        raise
    except RunCancelled:
        raise
    except Exception:
        try:
            raw = invoke_model(grader, prompt, purpose="grading", role="judge", fallback_of=structured_call.id if structured_call else None)
            score = (raw.content or "").strip().lower()
        except ContextBudgetExceeded:
            raise
        except RunCancelled:
            raise
        except Exception:
            score = "unknown"
    # 评分模型的自由文本不属于公开进度，只允许规范化的二值结果。
    score = score if score in ("yes", "no") else "unknown"
    route = "generate_answer" if score == "yes" or terminal else "rewrite_question"

    if score == "yes":
        emit_rag_step("✅", "文档相关性评估通过", f"评分: {score}")
    else:
        emit_rag_step("⚠️", "证据不足" if terminal else "文档相关性不足，将重写查询", f"评分: {score}")
    grade_update = {
        "grade_score": score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
        "evidence_status": {"yes": "sufficient", "no": "insufficient", "unknown": "unknown"}[score],
        "retrieval_status": "budget_exhausted" if exhausted and score == "no" else {"yes": "relevant", "no": "low_relevance", "unknown": "grading_failed"}[score],
    }
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(grade_update)
    return graded(route)


@timed("rewrite", component="rag")
def rewrite_question_node(state: RAGState) -> RAGState:
    check_cancelled()
    try:
        return _rewrite_question_impl(state)
    except RunCancelled:
        raise
    except Exception as exc:
        return _expansion_failure(state, exc)


def _expansion_failure(state, exc):
    from .tool_contracts import tool_exception
    failure = tool_exception(exc)
    trace = state.get('rag_trace') or {}
    trace.update(expansion_status='failed', expansion_error_code=failure.code,
                 expansion_retryable=failure.retryable, retrieval_status='expansion_failed',
                 evidence_status='unknown', rewrite_needed=False)
    emit_rag_step('❌', '查询扩展失败，当前资料无法确认')
    return {'rag_trace': trace, 'route': 'generate_answer'}


def _rewrite_question_impl(state: RAGState) -> RAGState:
    remaining_timeout()
    question = state["question"]
    emit_rag_step("✏️", "正在重写查询...")
    state['rag_trace']['expansion_stage'] = 'router'
    router = _get_router_model()
    strategy = "step_back"
    if router:
        prompt = (
            "请根据用户问题选择最合适的查询扩展策略，仅输出策略名。\n"
            "- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n"
            "- hyde：模糊、概念性、需要解释或定义的问题。\n"
            "- complex：多步骤、需要分解或综合多种信息的复杂问题。\n"
            f"用户问题：{question}"
        )
        structured_call = None
        try:
            structured = router.with_structured_output(RewriteStrategy, **structured_output_kwargs())
            assert_model_input(prompt, schema=RewriteStrategy)
            with model_call(router, purpose="rewrite_routing", structured=True) as structured_call:
                decision = structured.invoke([{"role": "user", "content": prompt}], **model_timeout_kwargs("rewrite_routing"), **model_output_kwargs(router))
                structured_call.observe(decision)
            strategy = decision.strategy
        except ContextBudgetExceeded:
            raise
        except RunCancelled:
            raise
        except Exception:
            raw = invoke_model(router, prompt, purpose="rewrite_routing", fallback_of=structured_call.id if structured_call else None)
            response_text = (raw.content or "").strip().lower()
            if "step_back" in response_text:
                strategy = "step_back"
            elif "hyde" in response_text:
                strategy = "hyde"
            elif "complex" in response_text:
                strategy = "complex"
            else:
                strategy = "step_back"

    expanded_query = question
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if strategy in ("step_back", "complex"):
        emit_rag_step("🧠", f"使用策略: {strategy}", "生成退步问题")
        state['rag_trace']['expansion_stage'] = 'step_back'
        step_back = step_back_expand(question)
        step_back_question = step_back.get("step_back_question", "")
        step_back_answer = step_back.get("step_back_answer", "")
        expanded_query = step_back.get("expanded_query", question)

    if strategy in ("hyde", "complex"):
        emit_rag_step("📝", "HyDE 假设性文档生成中...")
        state['rag_trace']['expansion_stage'] = 'hyde'
        hypothetical_doc = generate_hypothetical_document(question)

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "rewrite_strategy": strategy,
        "rewrite_query": expanded_query,
        "expansion_status": "completed",
    })

    return {
        "expansion_type": strategy,
        "expanded_query": expanded_query,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    check_cancelled()
    try:
        return _retrieve_expanded_impl(state)
    except RunCancelled:
        raise
    except Exception as exc:
        return _expansion_failure(state, exc)


def _retrieve_expanded_impl(state: RAGState) -> RAGState:
    remaining_timeout()
    strategy = state.get("expansion_type") or "step_back"
    emit_rag_step("🔄", "使用扩展查询重新检索...", f"策略: {strategy}")
    results: List[dict] = []
    rerank_applied_any = False
    rerank_enabled_any = False
    rerank_model = None
    rerank_endpoint = None
    rerank_errors = []
    retrieval_mode = None
    candidate_k = None
    leaf_retrieve_level = None
    auto_merge_enabled = None
    auto_merge_applied = False
    auto_merge_threshold = None
    auto_merge_replaced_chunks = 0
    auto_merge_steps = 0
    records = (state.get("rag_trace") or {}).get("retrieval_records", [])
    record_start = len(records)

    if strategy in ("hyde", "complex"):
        hypothetical_doc = state.get("hypothetical_doc") or generate_hypothetical_document(state["question"])
        retrieved_hyde = _retrieve(hypothetical_doc, "hyde", records)
        results.extend(retrieved_hyde.get("docs", []))
        hyde_meta = retrieved_hyde.get("meta", {})
        emit_rag_step(
            "🧱",
            "HyDE 三级检索",
            (
                f"L{hyde_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {hyde_meta.get('candidate_k', 0)}，"
                f"合并替换 {hyde_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(hyde_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(hyde_meta.get("rerank_enabled"))
        rerank_model = rerank_model or hyde_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or hyde_meta.get("rerank_endpoint")
        if hyde_meta.get("rerank_error"):
            rerank_errors.append(f"hyde:{hyde_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or hyde_meta.get("retrieval_mode")
        candidate_k = candidate_k or hyde_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or hyde_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else hyde_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(hyde_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or hyde_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(hyde_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(hyde_meta.get("auto_merge_steps") or 0)

    if strategy in ("step_back", "complex"):
        expanded_query = state.get("expanded_query") or state["question"]
        retrieved_stepback = _retrieve(expanded_query, "step_back", records)
        results.extend(retrieved_stepback.get("docs", []))
        step_meta = retrieved_stepback.get("meta", {})
        emit_rag_step(
            "🧱",
            "Step-back 三级检索",
            (
                f"L{step_meta.get('leaf_retrieve_level', 3)} 召回，"
                f"候选 {step_meta.get('candidate_k', 0)}，"
                f"合并替换 {step_meta.get('auto_merge_replaced_chunks', 0)}"
            ),
        )
        rerank_applied_any = rerank_applied_any or bool(step_meta.get("rerank_applied"))
        rerank_enabled_any = rerank_enabled_any or bool(step_meta.get("rerank_enabled"))
        rerank_model = rerank_model or step_meta.get("rerank_model")
        rerank_endpoint = rerank_endpoint or step_meta.get("rerank_endpoint")
        if step_meta.get("rerank_error"):
            rerank_errors.append(f"step_back:{step_meta.get('rerank_error')}")
        retrieval_mode = retrieval_mode or step_meta.get("retrieval_mode")
        candidate_k = candidate_k or step_meta.get("candidate_k")
        leaf_retrieve_level = leaf_retrieve_level or step_meta.get("leaf_retrieve_level")
        auto_merge_enabled = auto_merge_enabled if auto_merge_enabled is not None else step_meta.get("auto_merge_enabled")
        auto_merge_applied = auto_merge_applied or bool(step_meta.get("auto_merge_applied"))
        auto_merge_threshold = auto_merge_threshold or step_meta.get("auto_merge_threshold")
        auto_merge_replaced_chunks += int(step_meta.get("auto_merge_replaced_chunks") or 0)
        auto_merge_steps += int(step_meta.get("auto_merge_steps") or 0)

    deduped = []
    seen = set()
    for item in results:
        key = (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # 扩展阶段可能合并了多路召回（如 hyde + step_back），
    # 这里统一重排展示名次，避免出现 1,2,3,4,5,4,5 这类重复名次。
    for idx, item in enumerate(deduped, 1):
        item["rrf_rank"] = idx

    context = _format_docs(deduped)
    status = _retrieval_status(records[record_start:], deduped)
    _emit_retrieval_result(status, len(deduped), expanded=True)
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update({
        "expanded_query": state.get("expanded_query") or state["question"],
        "step_back_question": state.get("step_back_question", ""),
        "step_back_answer": state.get("step_back_answer", ""),
        "hypothetical_doc": state.get("hypothetical_doc", ""),
        "expansion_type": strategy,
        "retrieved_chunks": deduped,
        "expanded_retrieved_chunks": deduped,
        "retrieval_stage": "expanded",
        "retrieval_records": records,
        "retrieval_status": status,
        "relevance_filters": _relevance_filters(records),
        "rerank_enabled": rerank_enabled_any,
        "rerank_applied": rerank_applied_any,
        "rerank_model": rerank_model,
        "rerank_endpoint": rerank_endpoint,
        "rerank_error": "; ".join(rerank_errors) if rerank_errors else None,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "leaf_retrieve_level": leaf_retrieve_level,
        "auto_merge_enabled": auto_merge_enabled,
        "auto_merge_applied": auto_merge_applied,
        "auto_merge_threshold": auto_merge_threshold,
        "auto_merge_replaced_chunks": auto_merge_replaced_chunks,
        "auto_merge_steps": auto_merge_steps,
    })
    return {"docs": deduped, "context": context, "rag_trace": rag_trace}


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)

    graph.set_entry_point("retrieve_initial")
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_conditional_edges("rewrite_question", lambda state: 'failed' if (state.get('rag_trace') or {}).get('expansion_status') == 'failed' else 'continue', {'failed': END, 'continue': 'retrieve_expanded'})
    graph.add_conditional_edges("retrieve_expanded", lambda state: 'failed' if (state.get('rag_trace') or {}).get('expansion_status') == 'failed' else 'continue', {'failed': END, 'continue': 'grade_documents'})
    return graph.compile()


rag_graph = build_rag_graph()


def run_rag_graph(question: str) -> dict:
    if current_run_state() is None:
        with use_run_state(AgentRunState()):
            return run_rag_graph(question)
    return rag_graph.invoke({
        "question": question,
        "query": question,
        "context": "",
        "docs": [],
        "route": None,
        "expansion_type": None,
        "expanded_query": None,
        "step_back_question": None,
        "step_back_answer": None,
        "hypothetical_doc": None,
        "rag_trace": None,
    })
