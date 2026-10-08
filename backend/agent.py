from dotenv import load_dotenv
import os
import json
import re
import asyncio
import uuid
from openai import APITimeoutError
from typing import Any, Literal, TypedDict, List, Annotated
from contextlib import aclosing
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage, BaseMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, END
from .tools import get_current_weather, search_knowledge_base, web_search, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue, emit_rag_step, tool_is_available
from .tool_contracts import ToolResult, tool_error
from .tools import set_answer_evidence, get_answer_evidence
from .evidence_context import CitationFilter
from .schemas import public_rag_trace as _public_rag_trace
from datetime import datetime
from dataclasses import asdict
from threading import RLock
from .cache import cache
from .llm_config import create_chat_model, resolve_llm_settings
from .database import SessionLocal
from .models import User, ChatSession, ChatMessage, ChatTurn
from .run_control import check_cancelled, mark_cancelled, RunCancelled
from .conversation_turns import admitted_chat, locked_session, admitted_session, TurnIdentity, input_digest, TurnConflict
from .memory import memory_service
from .conversation_context import ContextBudget, ContextBudgetExceeded, model_budget, allocate_context, fit_context, plan_compaction, finish_compaction, summary_batches, summary_prompt
from .chat_diagnostics import run_blocking, model_call, invoke_model, ainvoke_model, invoke_tool, traced_chat, mark_public_answer, mark_chat_failed
from .telemetry import span
from .agent_budget import AgentRunState, AgentDeadline, current_run_state, use_run_state, model_timeout_kwargs

# 加载 LangSmith 配置
from . import langsmith_config

# LangSmith 追踪标签
LANGCHAIN_TAGS = ["starmew", "react-agent"]

load_dotenv()


def init_chat_model(**kwargs):
    """兼容现有测试/扩展的生成模型注入点，实际配置由统一工厂解析。"""
    return create_chat_model(role="generation", **kwargs)

# ============ ReAct Agent 状态定义 ============

class ReactAgentState(TypedDict):
    """ReAct Agent 状态"""
    messages: Annotated[List[BaseMessage], "对话历史"]
    step_count: int  # 当前步数
    tool_output: str | None  # 上次工具调用结果
    memory_context: str  # 长期记忆上下文
    run_state: AgentRunState  # same request object in graph and nested SDK calls
    source_policy: Literal['internal_only', 'external_allowed', 'unrestricted']
    answer_ready: bool  # 异步工具决策明确完成后才进入最终回答节点
    context_state: dict  # compacted history/checkpoints may carry unfinished context
    router_fallback_used: bool
    evidence_selection: Any


def _stream_model_response(
    model_with_tools,
    full_messages: list[BaseMessage],
    *, purpose="tool_decision", role="generation",
) -> AIMessageChunk:
    """同步聊天使用的模型消息聚合；不承担 SSE 投递。"""
    accumulated = None
    with model_call(model_with_tools, purpose=purpose, role=role, streaming=True) as call:
        try:
            for chunk in model_with_tools.stream(full_messages, **model_timeout_kwargs(purpose)):
                if chunk is not None:
                    call.observe(chunk)
                    accumulated = chunk if accumulated is None else accumulated + chunk
                run = current_run_state()
                if run:
                    run.timeout(final=purpose == "final_answer")
        except Exception as exc:
            if purpose in ('tool_decision', 'tool_decision_fallback'):
                exc.agent_decision_response = accumulated
            raise
        if accumulated is not None:
            call.observe(accumulated)

    return accumulated or AIMessageChunk(content="")


async def _astream_model_response(model, messages, *, purpose="tool_decision", role="generation") -> AIMessageChunk:
    """直接消费供应商异步流，取消会传播到正在等待网络数据的调用。"""
    accumulated = None
    with model_call(model, purpose=purpose, role=role, streaming=True) as call:
        try:
            timeout = model_timeout_kwargs(purpose)
            async with asyncio.timeout(timeout.get('timeout')):
                async with aclosing(model.astream(messages, **timeout)) as stream:
                    async for chunk in stream:
                        if chunk is not None:
                            call.observe(chunk)
                            accumulated = chunk if accumulated is None else accumulated + chunk
                        run = current_run_state()
                        if run:
                            run.timeout(final=purpose == 'final_answer')
        except Exception as exc:
            if purpose in ('tool_decision', 'tool_decision_fallback'):
                exc.agent_decision_response = accumulated
            raise
        if accumulated is not None:
            call.observe(accumulated)
    return accumulated or AIMessageChunk(content="")


def _message_text(message: BaseMessage | AIMessageChunk | Any) -> str:
    """提取消息中的可展示文本，兼容不同 LangChain 版本。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)

    text_attr = getattr(message, "text", "")
    if callable(text_attr):
        text_attr = text_attr()
    return text_attr if isinstance(text_attr, str) else ""


def _streamed_final_text(message: Any, metadata: dict[str, Any] | None) -> str:
    """筛选图级 messages 流中允许发送给客户端的文本。"""
    if not isinstance(message, AIMessageChunk):
        return ""
    if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "final":
        return ""
    if getattr(message, "tool_call_chunks", None) or getattr(message, "tool_calls", None):
        return ""
    return _message_text(message)


def _tool_decision_message(response):
    """Strip decision text and give missing provider IDs the same paired ID."""
    return response.model_copy(update={
        'content': '',
        **{field: [{**call, 'id': call.get('id') or f'call_{uuid.uuid4().hex}'}
                   for call in getattr(response, field, [])]
           for field in ('tool_calls', 'invalid_tool_calls')},
    })


def resolve_source_policy(messages, initial='unrestricted'):
    """Carry explicit user source choices until the user changes that choice."""
    policy = initial if initial in ('internal_only', 'external_allowed') else 'unrestricted'
    restricted = r'(?:仅|只)(?:允许|能|可|使用|用|根据|依据|基于|从|查|参考|限于|限|所)*\s*(?:内部|知识库|上传|提供的|文档)|(?:不要|禁止|不允许|不得)(?:联网|搜索网络|网络搜索|使用外部|使用网络)'
    allowed = r'(?:解除|取消)(?:内部知识|内部来源|知识库|联网|网络|来源).{0,8}(?:限制|约束|禁令)|不再(?:仅|只)?(?:允许|限于|限|使用)*(?:内部知识|内部来源|知识库)|(?:允许|可以|可|请)(?:使用|搜索|查询|参考|访问)?(?:联网|上网|网络|外部来源|外部信息)|(?:使用|参考)(?:网络|外部)(?:来源|信息|资料)'
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        text = _message_text(message)
        directives = []
        for pattern in (restricted, r'\b(?:only (?:use |using |from )?(?:internal|uploaded|provided)|no (?:web|internet|external))\b'):
            for match in re.finditer(pattern, text, re.I):
                if not re.search(r'(?:不再|no longer)\s*$', text[:match.start()], re.I):
                    directives.append((match.start(), 'internal_only'))
        for pattern in (allowed, r'\b(?:allow (?:web|internet|external)|(?:use|search) (?:the )?(?:web|internet)|remove (?:the )?(?:source|internal) restriction)\b'):
            for match in re.finditer(pattern, text, re.I):
                if not re.search(r'(?:不|禁止|不得|不要|not|do not)\s*$', text[:match.start()], re.I):
                    directives.append((match.start(), 'external_allowed'))
        if directives:
            policy = max(directives, key=lambda directive: directive[0])[1]
    return policy


class ConversationStorage:
    """PostgreSQL-authoritative conversation reads; legacy Redis keys only expire.

    Mutable histories/lists cannot trust delete-based cache invalidation: a late
    reader or failed DELETE can resurrect old data, including after a restart.
    Reads deliberately use the database; unrelated Redis caches are unaffected.
    """

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def append_turn(self, user_id: str, session_id: str, user_text: str, answer: str,
                    *, context_state: dict, rag_trace: dict | None, request_id: str | None = None,
                    turn_identity: TurnIdentity | None = None) -> dict[str, int]:
        """原子追加一轮，返回两条消息 ID 与提交后的消息总数。"""
        db = SessionLocal()
        try:
            turn = None
            if turn_identity is not None:
                if request_id and str(request_id) != turn_identity.request_id:
                    raise TurnConflict('REQUEST_INPUT_CONFLICT')
                request_id = turn_identity.request_id
                session, turn = admitted_session(db, user_id, session_id, turn_identity, lock=True)
                if turn.input_digest != input_digest(user_text):
                    raise TurnConflict('REQUEST_INPUT_CONFLICT')
                if turn.status == 'completed':
                    return {"user_message_id": turn.user_message_id,
                            "assistant_message_id": turn.assistant_message_id,
                            "message_count": turn.message_count}
                if turn.status != 'running':
                    raise TurnConflict('REQUEST_' + turn.status.upper())
            else:
                if request_id:
                    raise TurnConflict('REQUEST_IDENTITY_REQUIRED')
                session = locked_session(db, user_id, session_id)
            session.metadata_json = {**(session.metadata_json or {}), "context_state": dict(context_state)}
            now = datetime.utcnow()
            user_message = db.get(ChatMessage, turn.user_message_id) if turn and turn.user_message_id else None
            if user_message is None:
                user_message = ChatMessage(session_ref_id=session.id, message_type="human",
                    content=str(user_text), timestamp=now)
            assistant_message = ChatMessage(session_ref_id=session.id, message_type="ai",
                content=str(answer), timestamp=now, rag_trace=rag_trace)
            db.add_all([user_message, assistant_message])
            session.updated_at = now
            db.flush()
            result = {
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "message_count": db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).count(),
            }
            if turn is not None:
                turn.status = 'completed'
                turn.user_message_id = user_message.id
                turn.assistant_message_id = assistant_message.id
                turn.message_count = result['message_count']
                turn.error_code = None
            from .memory_jobs import enqueue_memory_update
            enqueue_memory_update(db, session.id, result['message_count'], request_id=request_id)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        # 仅提交成功后失效；不要将调用者的旧历史快照写回缓存。
        cache.delete(self._messages_cache_key(user_id, session_id))
        cache.delete(self._sessions_cache_key(user_id))
        return result

    def save_interrupted(self, user_id, session_id, content, *, turn_identity, status='interrupted'):
        """Retain visible partial output without advancing inference or memory checkpoints."""
        if not content:
            return
        with SessionLocal() as db:
            session, turn = admitted_session(db, user_id, session_id, turn_identity, lock=True)
            if turn.status == 'completed' or turn.assistant_message_id is not None:
                return
            message = ChatMessage(session_ref_id=session.id, message_type='ai', content=content, status=status)
            db.add(message)
            db.flush()
            turn.assistant_message_id = message.id
            session.updated_at = datetime.utcnow()
            db.commit()

    def replace_all_messages(self, user_id: str, session_id: str, messages: list,
                             metadata: dict = None, extra_message_data: list = None):
        """导入/修复专用：全量替换历史；正常聊天必须使用 append_turn。"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return

            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .with_for_update()
                .first()
            )
            if not session:
                session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
                db.add(session)
                db.flush()
            else:
                session.metadata_json = {**(session.metadata_json or {}), **(metadata or {})}

            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            now = datetime.utcnow()
            for idx, msg in enumerate(messages):
                rag_trace = None
                if extra_message_data and idx < len(extra_message_data):
                    extra = extra_message_data[idx] or {}
                    rag_trace = extra.get("rag_trace")

                db.add(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=msg.type,
                        content=str(msg.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
            session.updated_at = now
            db.commit()

            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def load(self, user_id: str, session_id: str, *, turn_identity: TurnIdentity | None = None) -> list:
        """加载对话"""
        records = self.get_session_messages(user_id, session_id, turn_identity=turn_identity)
        records = [r for r in records if r['type'] != 'ai' or r.get('status', 'complete') == 'complete']
        if turn_identity is not None:
            with SessionLocal() as db:
                _, turn = admitted_session(db, user_id, session_id, turn_identity)
                if turn.user_message_id is not None:
                    records = [r for r in records if r['type'] != 'human' or r['id'] < turn.user_message_id]
        return self._to_langchain_messages(records)

    def load_context_state(self, user_id: str, session_id: str, *, turn_identity: TurnIdentity | None = None) -> dict:
        """Read the summary checkpoint independently of the message cache."""
        with SessionLocal() as db:
            session = (admitted_session(db, user_id, session_id, turn_identity)[0] if turn_identity is not None else
                       db.query(ChatSession).join(User).filter(User.username == user_id, ChatSession.session_id == session_id).first())
            return dict((session.metadata_json or {}).get("context_state", {})) if session else {}

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []

            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = []
            for s in sessions:
                count = db.query(ChatMessage).filter(ChatMessage.session_ref_id == s.id).count()
                result.append(
                    {
                        "session_id": s.session_id,
                        "updated_at": s.updated_at.isoformat(),
                        "message_count": count,
                    }
                )
            return result
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str, *, turn_identity: TurnIdentity | None = None) -> list[dict]:
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return []
            session = admitted_session(db, user_id, session_id, turn_identity)[0] if turn_identity is not None else (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return []

            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "id": row.id,
                    "status": row.status,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            return result
        finally:
            db.close()

    def get_pending_run(self, user_id, session_id):
        from .conversation_turns import pending_run
        return pending_run(user_id, session_id, session_factory=SessionLocal)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == user_id).first()
            if not user:
                return False
            session = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            cache.delete(self._messages_cache_key(user_id, session_id))
            cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()


# ============ ReAct Agent 实现 ============

def _create_react_agent():
    """创建 ReAct Agent（LangGraph StateGraph）"""

    budget = ContextBudget.from_env()
    model = init_chat_model(
        temperature=0.3,
        stream_usage=True,
        max_tokens=budget.output_reserve,
    )

    # 绑定工具
    tools = [get_current_weather, search_knowledge_base, web_search]
    finish_tool = {
        'type': 'function', 'function': {
            'name': 'finish',
            'description': '已有足够信息或无需工具时选择此操作，开始最终回答。不要在此生成答案。',
            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
        },
    }
    router_models, bound_models, schema_versions = {}, {}, {}
    binding_lock = RLock()
    tool_schemas = {id(tool): convert_to_openai_tool(tool) for tool in tools + [finish_tool]}

    def _binding_kwargs(asynchronous):
        return {'tool_choice': 'required', 'parallel_tool_calls': False} if asynchronous else {}

    def _schema_version(bound_tools, asynchronous=False):
        key = (tuple(id(tool) for tool in bound_tools), asynchronous)
        with binding_lock:
            if key not in schema_versions:
                envelope = {'tools': [tool_schemas[id(tool)] for tool in bound_tools], **_binding_kwargs(asynchronous)}
                schema = json.dumps(envelope, ensure_ascii=False)
                schema_versions[key] = (schema, budget.count_text(schema))
            return schema_versions[key]

    def _bound_router(bound_tools, asynchronous, *, fallback=False):
        settings = resolve_llm_settings(role='generation' if fallback else 'router')
        separate = not fallback and bool(os.getenv('AGENT_ROUTER_MODEL', '').strip())
        schema, _ = _schema_version(bound_tools, asynchronous)
        key = (settings, separate, asynchronous, schema)
        with binding_lock:
            if key not in bound_models:
                if separate:
                    if settings not in router_models:
                        router_models.clear()  # retain only the current configured provider/model
                        router_models[settings] = create_chat_model(role='router', temperature=0.3,
                            stream_usage=True, max_tokens=budget.output_reserve)
                    selected = router_models[settings]
                else:
                    selected = model
                if len(bound_models) >= 16:
                    bound_models.clear()
                bound_models[key] = selected.bind_tools(bound_tools, **_binding_kwargs(asynchronous))
            return bound_models[key], separate

    def _direct_greeting(state):
        if os.getenv('AGENT_DIRECT_CHAT_ENABLED', 'false').strip().lower() not in ('true', '1', 'yes', 'on'):
            return False
        messages = state['messages']
        if len(messages) != 1 or not isinstance(messages[0], HumanMessage) or messages[0].content != '你好':
            return False
        context = state.get('context_state') or {}
        if state.get('memory_context', '').strip() or context.get('count', 0) or context.get('summary'):
            return False
        # Empty checkpoint metadata is normal even for a brand new session.
        return not any(value for key, value in context.items() if key not in ('count', 'digest', 'summary', 'source_policy'))

    def _invalid_route(response, bound_tools):
        if response.invalid_tool_calls:
            return True
        schemas = {item['function']['name']: item['function']['parameters']
                   for item in (tool_schemas[id(tool)] for tool in bound_tools)}
        for call in response.tool_calls:
            schema = schemas.get(call.get('name'))
            args = call.get('args')
            if schema is None or not isinstance(args, dict):
                return True
            properties = schema.get('properties', {})
            if set(args) - set(properties) or not set(schema.get('required', [])) <= set(args):
                return True
            if call['name'] == 'finish' and args != {}:
                return True
            tool = next((tool for tool in tools if tool.name == call['name']), None)
            validator = getattr(tool, 'args_schema', None)
            if validator:
                try:
                    validator.model_validate(args)
                except (ValueError, TypeError):
                    return True
        return not response.tool_calls and not _message_text(response).strip()

    def _reject_route(state, response):
        # No rejected batch is executed, including valid siblings. All model calls
        # stay paired before the fallback request; decision text is discarded.
        if response is None or not (response.tool_calls or response.invalid_tool_calls):
            return state
        decision = _tool_decision_message(response)
        pairs = []
        for call in decision.tool_calls + decision.invalid_tool_calls:
            result = _source_policy_rejection(call.get('name'), state) or tool_error('INVALID_ARGUMENT')
            pairs.append(ToolMessage(content=result.to_json(include_evidence=False),
                                     artifact=asdict(result), tool_call_id=call['id']))
        return {**state, 'messages': state['messages'] + [decision] + pairs}

    def _decision_result(state, response):
        calls = response.tool_calls + response.invalid_tool_calls
        if not calls and not _message_text(response).strip():
            raise ValueError('模型未返回有效的工具决策')
        return {**state, 'messages': state['messages'] + [_tool_decision_message(response)] if calls else state['messages'],
                'step_count': _run(state).decision_count, 'answer_ready': not calls}

    def _internal_knowledge_only(state):
        if 'source_policy' not in state:
            state['source_policy'] = resolve_source_policy(state['messages'])
        return state['source_policy'] == 'internal_only'

    def _source_policy_allows(tool_name, state):
        return not (tool_name in ('web_search', 'get_current_weather') and _internal_knowledge_only(state))

    def _source_policy_rejection(tool_name, state):
        if not _source_policy_allows(tool_name, state):
            return ToolResult('skipped', '用户仅允许内部知识，未执行外部工具。', evidence_status='insufficient')
        return None

    def _available_tools(state):
        return [tool for tool in tools if tool_is_available(tool.name) and _source_policy_allows(tool.name, state)]

    # ReAct System Prompt
    REACT_SYSTEM_PROMPT = """你是星喵，一位温和可靠的智能助手，喜欢帮助用户解决问题。

你可以使用工具来辅助回答：
- search_knowledge_base：用于查询用户上传的文档/内部知识库。
- web_search：用于获取实时、最新或知识库未覆盖的外部信息。
- get_current_weather：用于查询天气。

【ReAct 推理模式】
对于每个问题，请按以下步骤思考：
1. Thought（思考）：分析问题，决定是否需要使用工具
2. Action（行动）：如果需要工具，选择并调用合适的工具
3. Observation（观察）：查看工具返回的结果
4. 如果工具结果足够，回答问题
5. 如果结果不足，可以继续思考或直接基于已知信息回答

规则：
- 如果检索到的信息不足，请诚实地说不知道，不要编造事实。
- 知识库会先执行内部查询扩展并再次评分；evidence_status 为 insufficient/unknown 时，才可尝试一次不同查询。禁止重复同一查询。
- 用户只允许内部知识时，不能使用网络搜索。允许外部来源且网络工具可用时，可在剩余预算内检索。
- 文档非空不代表证据充分，评分失败或检索失败不能视为成功。证据不足时明确说明“当前资料无法确认”。
- 回答引用必须来自工具实际提供的上下文，标明文件及页码，不引用仅在检索记录中出现的文档。
- 回答时不要暴露推理过程，只输出最终答案。
{memory_context}
"""

    def _model_messages(state, *, suffix="", bound_tools=(), separate=False, asynchronous=False):
        request_budget = model_budget('router') if separate else budget
        system_content = REACT_SYSTEM_PROMPT.format(memory_context='') + suffix
        if _internal_knowledge_only(state):
            system_content += '\n用户当前的有效来源限制：仅允许内部知识和提供的文档，禁止联网。'
        tool_tokens = request_budget.count_text(_schema_version(bound_tools, asynchronous)[0]) if bound_tools else 0
        fitted, selection = allocate_context(system_content, state['messages'], request_budget,
            memory=state.get('memory_context', ''), tool_tokens=tool_tokens)
        state['evidence_selection'] = selection
        return fitted

    def _should_continue(state: ReactAgentState) -> Literal["continue", "final", "end"]:
        """判断是否继续推理循环"""
        if state.get("answer_ready"):
            return "final"
        last_msg = state["messages"][-1]

        # 如果有工具调用，继续执行工具
        if getattr(last_msg, "tool_calls", None) or getattr(last_msg, "invalid_tool_calls", None):
            return "continue"

        # 如果是最终回答，结束
        return "end"

    def _run(state):
        run = state.get('run_state') or current_run_state() or AgentRunState()
        state['run_state'] = run
        return run

    def _claim_route_attempt(state, fallback):
        run = _run(state)
        if run.check(decision=True):
            return None
        run.decision_count += 1
        return (not fallback and bool(os.getenv('AGENT_ROUTER_MODEL', '').strip()), {
            'role': 'generation' if fallback else 'router',
            'purpose': 'tool_decision_fallback' if fallback else 'tool_decision',
        })

    def _route_outcome(state, bound_tools, *, fallback, separate, response, error):
        """Classify one attempt and advance the shared fallback state machine."""
        if isinstance(error, ContextBudgetExceeded):
            raise error
        if error is None:
            try:
                if not separate or not _invalid_route(response, bound_tools):
                    return _decision_result(state, response), False
            except Exception as exc:
                error = exc
        if error is not None:
            response = getattr(error, 'agent_decision_response', None)
            if isinstance(error, (TimeoutError, APITimeoutError)):
                state = _reject_route(state, response)
                _run(state).stop_reason = 'deadline'
                return {**state, 'answer_ready': True}, False
            if fallback:
                return {**_reject_route(state, response), 'answer_ready': True}, False
            if not separate:
                raise error
        state = _reject_route(state, response)
        if state.get('router_fallback_used'):
            return {**state, 'answer_ready': True}, False
        return {**state, 'router_fallback_used': True}, True

    def _reason(state: ReactAgentState) -> ReactAgentState:
        run = _run(state)
        if state.get('answer_ready') or run.check(decision=True) or _direct_greeting(state):
            return {**state, 'answer_ready': True}
        bound_tools = _available_tools(state) + [finish_tool]
        for fallback in (False, True):
            attempt = _claim_route_attempt(state, fallback)
            if attempt is None:
                return {**state, 'answer_ready': True}
            separate, call_kwargs = attempt
            response = error = None
            try:
                router, separate = _bound_router(bound_tools, False, fallback=fallback)
                with use_run_state(run):
                    response = _stream_model_response(router, _model_messages(state, bound_tools=bound_tools, separate=separate),
                        **call_kwargs)
            except Exception as exc:
                error = exc
            state, retry = _route_outcome(state, bound_tools, fallback=fallback,
                separate=separate, response=response, error=error)
            if not retry:
                return state

    async def _reason_async(state: ReactAgentState) -> ReactAgentState:
        run = _run(state)
        if state.get('answer_ready') or run.check(decision=True) or _direct_greeting(state):
            return {**state, 'answer_ready': True}
        # 工具决策流可能先输出文本后输出 tool call，整轮内容均不可公开。
        # finish 是控制信号，不含答案；最终答案由无工具的 final 节点生成一次。
        bound_tools = _available_tools(state) + [finish_tool]
        for fallback in (False, True):
            attempt = _claim_route_attempt(state, fallback)
            if attempt is None:
                return {**state, 'answer_ready': True}
            separate, call_kwargs = attempt
            response = error = None
            try:
                router, separate = _bound_router(bound_tools, True, fallback=fallback)
                with use_run_state(run):
                    response = await _astream_model_response(router, _model_messages(state, bound_tools=bound_tools, separate=separate, asynchronous=True,
                        suffix='\n本轮只选择工具。无需更多工具时调用 finish，不要生成答案或推理文本。'),
                        **call_kwargs)
            except Exception as exc:
                error = exc
            state, retry = _route_outcome(state, bound_tools, fallback=fallback,
                separate=separate, response=response, error=error)
            if not retry:
                return state

    def _final_messages(state):
        run = _run(state)
        run.timeout(final=True)
        suffix = ''
        if run.stop_reason and run.stop_reason != 'evidence_sufficient':
            suffix = ('\n本轮工具或时间预算已用尽。请仅基于已有证据给出有限总结，明确尚未解决的缺口。'
                      '没有证据时必须说明未获得检索支持，不能声称已检索证实。')
        messages = _model_messages(state, suffix=suffix)
        if state['evidence_selection'].omitted_count and not state['evidence_selection'].snippets:
            run.stop_reason = 'evidence_omitted'
        set_answer_evidence(state['evidence_selection'])
        return messages

    def _finish_response(state, response):
        run = _run(state)
        run.timeout(final=True)
        if not _message_text(response).strip() or response.tool_calls or response.invalid_tool_calls:
            raise ValueError('模型未返回完整的最终回答')
        selection = state.get('evidence_selection')
        if selection and (selection.snippets or selection.omitted_count):
            citations = CitationFilter(s.citation_id for s in selection.snippets)
            response = response.model_copy(update={'content': citations.feed(_message_text(response)) + citations.finish()})
        run.answer_status = 'partial' if run.stop_reason and run.stop_reason != 'evidence_sufficient' else 'complete'
        return {**state, 'messages': state['messages'] + [response], 'answer_ready': False}

    async def _final(state: ReactAgentState) -> ReactAgentState:
        run = _run(state)
        try:
            with use_run_state(run):
                response = await _astream_model_response(model, _final_messages(state), purpose='final_answer')
            return _finish_response(state, response)
        except (TimeoutError, APITimeoutError):
            run.stop_reason = 'deadline'
            raise AgentDeadline() from None

    def _final_sync(state: ReactAgentState) -> ReactAgentState:
        run = _run(state)
        try:
            with use_run_state(run):
                response = _stream_model_response(model, _final_messages(state), purpose='final_answer')
            return _finish_response(state, response)
        except (TimeoutError, APITimeoutError):
            run.stop_reason = 'deadline'
            raise AgentDeadline() from None

    def _act(state: ReactAgentState) -> ReactAgentState:
        """行动节点：执行工具并返回 ToolMessage"""
        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        calls = (getattr(last_msg, 'tool_calls', []) or []) + (getattr(last_msg, 'invalid_tool_calls', []) or [])
        if not calls:
            return state

        tool_messages = []
        tool_results = []
        finish_requested = False
        for tool_call in calls:
            check_cancelled()
            # 每一个 tool_call_id 都必须对应一个 ToolMessage；不能只处理
            # 列表中的第一个调用，否则下一次模型请求会被接口拒绝。
            tool_name = tool_call.get("name", "")
            tool_call_id = (
                tool_call.get("id") or
                f"{tool_name or 'unknown'}_{uuid.uuid4().hex[:8]}"
            )

            # 获取参数 - tool_calls 中的 args 可能是字典
            tool_args = tool_call.get("args", {})
            if not tool_args and hasattr(tool_call, "input"):
                tool_args = tool_call.input or {}

            catalog = {
                'get_current_weather': (get_current_weather, 'weather'),
                'search_knowledge_base': (search_knowledge_base, 'knowledge_search'),
                'web_search': (web_search, 'web_search'),
            }
            if tool_name == 'finish':
                # Internal control gets a matching response but no external span
                # or progress; decision content never enters the answer stream.
                if tool_call in getattr(last_msg, 'invalid_tool_calls', []) or tool_args != {}:
                    result = tool_error('INVALID_ARGUMENT')
                else:
                    result = ToolResult('skipped', '开始最终回答。')
                    finish_requested = True
            else:
                display_name = tool_name if tool_name in catalog else '未知工具'
                emit_rag_step('🔧', f'正在执行工具: {display_name}')
                selected, purpose = catalog.get(tool_name, (None, 'unknown'))
                if tool_call in getattr(last_msg, 'invalid_tool_calls', []):
                    tool_args = None
                policy_rejection = _source_policy_rejection(tool_name, state)
                if policy_rejection is not None:
                    result = policy_rejection
                else:
                    with use_run_state(_run(state)):
                        result = invoke_tool(selected, tool_args, purpose=purpose, tool_call_id=tool_call_id)
                if tool_name == 'search_knowledge_base' and result.evidence_status == 'sufficient':
                    _run(state).stop_reason = 'evidence_sufficient'
                icon, label = {
                    'ok': ('✅', '执行完成'),
                    'empty': ('ℹ️', '无结果'),
                    'error': ('❌', '执行失败'),
                    'skipped': ('⏭️', '已跳过'),
                }[result.status]
                emit_rag_step(icon, f'工具 {display_name} {label}')

            result_text = result.to_json(include_evidence=False)
            tool_results.append(result_text)

            # OpenAI 兼容接口要求每个工具调用都有同 ID 的响应消息。
            tool_messages.append(ToolMessage(
                content=result_text,
                tool_call_id=tool_call_id,
                artifact=asdict(result),
            ))

        return {
            **state,
            "messages": messages + tool_messages,
            "tool_output": "\n".join(tool_results),
            "answer_ready": finish_requested or bool(_run(state).check(decision=True)),
        }

    async def _act_async(state):
        # A cancelled executor Future does not stop its synchronous tool. Keep
        # this graph node alive until the worker exits, retaining turn admission.
        worker = asyncio.create_task(asyncio.to_thread(_act, state))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                mark_cancelled()
                cancelled = True
            except Exception:
                break
        if cancelled:
            worker.exception()
            raise asyncio.CancelledError
        return worker.result()

    # 构建图
    graph = StateGraph(ReactAgentState)
    graph.add_node("reason", RunnableLambda(_reason, afunc=_reason_async))
    graph.add_node("act", RunnableLambda(_act, afunc=_act_async))
    graph.add_node("final", RunnableLambda(_final_sync, afunc=_final))

    graph.set_entry_point("reason")

    # 条件边：reason -> (act | end)
    graph.add_conditional_edges(
        "reason",
        _should_continue,
        {
            "continue": "act",
            "final": "final",
            "end": END,
        },
    )

    # act -> reason（循环）
    graph.add_edge("act", "reason")
    graph.add_edge("final", END)

    # LangSmith tracing 已在 langsmith_config.py 中自动启用
    # 不使用 checkpointer，避免需要每次传入 thread_id
    compiled_graph = graph.compile()

    return compiled_graph, model


# 初始化 ReAct Agent
react_graph, model = _create_react_agent()

storage = ConversationStorage()

def summarize_old_messages(model, messages: list, previous="", budget=None) -> str:
    """Roll only newly evicted history into the persisted summary in bounded batches."""
    budget = budget or ContextBudget.from_env()
    summary = previous
    for batch in summary_batches(messages, budget):
        response = invoke_model(model, summary_prompt(summary, batch), purpose="conversation_summary")
        summary = budget.clip(_message_text(response), budget.summary)
        if not summary.strip():
            raise ValueError("对话摘要为空，未推进摘要进度")
    return summary


async def _summarize_old_messages_async(model, messages, previous, budget):
    summary = previous
    for batch in summary_batches(messages, budget):
        response = await ainvoke_model(model, summary_prompt(summary, batch), purpose="conversation_summary")
        summary = budget.clip(_message_text(response), budget.summary)
        if not summary.strip():
            raise ValueError("对话摘要为空，未推进摘要进度")
    return summary


def _history_context(history, state, user_text):
    messages = list(history[state["count"]:])
    if state["summary"]:
        messages.insert(0, SystemMessage(content="之前的对话摘要：\n" + state["summary"]))
    return messages + [HumanMessage(content=user_text)]


@admitted_chat
@traced_chat
def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session", *, request_id=None, turn_identity: TurnIdentity | None = None):
    """使用 ReAct Agent 处理用户消息并返回响应"""
    with span("conversation_load", component="postgres"):
        all_messages = list(storage.load(user_id, session_id, turn_identity=turn_identity))
    messages = list(all_messages)

    # 清理可能残留的 RAG 上下文
    reset_tool_call_guards()
    get_last_rag_context(clear=True)

    # 获取长期记忆上下文
    with span("memory_load", component="memory"):
        current_run_state().timeout(final=True)
        memory_context = '' if current_run_state().check() else memory_service.get_memory_for_context(
            user_id, query=user_text, session_id=session_id,
            session_ref_id=turn_identity.session_ref_id if turn_identity else None)

    current_run_state().timeout(final=True)
    budget = ContextBudget.from_env()
    saved_context = storage.load_context_state(user_id, session_id, turn_identity=turn_identity)
    source_policy = resolve_source_policy(messages + [HumanMessage(content=user_text)], saved_context.get('source_policy'))
    plan = plan_compaction(messages, saved_context, budget)
    try:
        current_run_state().timeout()
        summary = summarize_old_messages(model, plan['pending'], plan['summary'], budget) if plan['pending'] else plan['summary']
        context_state = finish_compaction(messages, plan, summary, budget)
    except (TimeoutError, APITimeoutError):
        current_run_state().stop_reason = 'deadline'
        # Never advance a summary checkpoint when its model did not finish.
        context_state = finish_compaction(messages, {**plan, 'end': plan['start']}, plan['summary'], budget)
    context_state['source_policy'] = source_policy
    messages = _history_context(messages, context_state, user_text)

    # 调用 ReAct Agent
    result = react_graph.invoke({
        "messages": messages,
        "step_count": 0,
        "tool_output": None,
        "memory_context": memory_context,
        "run_state": current_run_state(),
        "source_policy": source_policy,
        "context_state": {**context_state, **saved_context},
    }, config={"recursion_limit": current_run_state().recursion_limit})

    # 提取响应
    response_messages = result.get("messages", [])
    final_msg = response_messages[-1] if response_messages else None
    response_content = getattr(final_msg, "content", str(final_msg)) if final_msg else ""

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    all_messages.extend([HumanMessage(content=user_text), AIMessage(content=response_content)])
    with span("conversation_save", component="postgres"):
        storage.append_turn(user_id, session_id, user_text, response_content,
                            context_state=context_state, rag_trace=rag_trace, request_id=request_id, turn_identity=turn_identity)

    if current_run_state().answer_status == 'failed':
        current_run_state().answer_status = 'complete'
    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


@admitted_chat
@traced_chat
async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session", *, request_id=None, turn_identity: TurnIdentity | None = None):
    """实时发送最终回答；持久化成功后结束协议，取消不保存未完成回答。"""
    output_queue = asyncio.Queue()
    agent_task = None
    active = True
    public_parts = []
    partial_status = 'interrupted'

    class _RagStepProxy:
        """将 RAG 步骤实时推送到前端"""
        def put_nowait(self, step):
            if active:
                output_queue.put_nowait({"type": "rag_step", "step": step})

    agent_result_holder = {"final_state": None}

    async def _agent_worker():
        """后台任务：消费 LangGraph 原生消息和值流。"""
        citations = None
        try:
            async with aclosing(react_graph.astream(
                {
                    "messages": messages,
                    "step_count": 0,
                    "tool_output": None,
                    "memory_context": memory_context,
                    "answer_ready": False,
                    "run_state": current_run_state(),
                    "source_policy": source_policy,
                    "context_state": {**context_state, **saved_context},
                },
                stream_mode=["messages", "values"],
                config={"recursion_limit": current_run_state().recursion_limit},
            )) as graph_stream:
                async for stream_event in graph_stream:
                    if not isinstance(stream_event, tuple) or len(stream_event) != 2:
                        continue
                    mode, data = stream_event
                    if mode == "messages":
                        try:
                            message, metadata = data
                        except (TypeError, ValueError):
                            continue
                        content = _streamed_final_text(message, metadata)
                        selection = get_answer_evidence()
                        if content and selection and (selection.snippets or selection.omitted_count):
                            if citations is None:
                                citations = CitationFilter(s.citation_id for s in selection.snippets)
                            content = citations.feed(content)
                        if content:
                            await output_queue.put({"type": "content", "content": content})
                    elif mode == "values" and isinstance(data, dict):
                        agent_result_holder["final_state"] = data

            if citations:
                tail = citations.finish()
                if tail:
                    await output_queue.put({'type': 'content', 'content': tail})

            # 让跨线程排入的 RAG 步骤先于结束哨兵入队。
            await asyncio.sleep(0)
        finally:
            output_queue.put_nowait(None)

    try:
        # 初始化也属于流的生命周期，任何失败都走同一个 error/DONE 协议。
        reset_tool_call_guards()
        set_rag_step_queue(_RagStepProxy())
        get_last_rag_context(clear=True)
        with span("conversation_load", component="postgres"):
            messages = await run_blocking(storage.load, user_id, session_id, turn_identity=turn_identity)
        current_run_state().timeout(final=True)
        with span("memory_load", component="memory"):
            memory_context = '' if current_run_state().check() else await run_blocking(
                memory_service.get_memory_for_context, user_id, query=user_text, session_id=session_id,
                session_ref_id=turn_identity.session_ref_id if turn_identity else None)
        current_run_state().timeout(final=True)
        budget = ContextBudget.from_env()
        saved_context = await run_blocking(storage.load_context_state, user_id, session_id, turn_identity=turn_identity)
        source_policy = resolve_source_policy(messages + [HumanMessage(content=user_text)], saved_context.get('source_policy'))
        current_run_state().timeout(final=True)
        plan = plan_compaction(messages, saved_context, budget)
        try:
            current_run_state().timeout()
            summary = await _summarize_old_messages_async(model, plan['pending'], plan['summary'], budget) if plan['pending'] else plan['summary']
            context_state = finish_compaction(messages, plan, summary, budget)
        except (TimeoutError, APITimeoutError):
            current_run_state().stop_reason = 'deadline'
            context_state = finish_compaction(messages, {**plan, 'end': plan['start']}, plan['summary'], budget)
        context_state['source_policy'] = source_policy
        messages = _history_context(messages, context_state, user_text)
        emit_rag_step("🤔", "星喵开始思考...")
        agent_task = asyncio.create_task(_agent_worker())

        while True:
            event = await output_queue.get()
            if event is None:
                break
            if event.get("type") == "content" and event.get("content"):
                public_parts.append(event['content'])
                mark_public_answer()
            yield f"data: {json.dumps(event)}\n\n"
        # 显式取回图异常，保证错误发生在 DONE 之前。
        await agent_task
        final_state = agent_result_holder["final_state"] or {}
        response_messages = final_state.get("messages", [])
        final_msg = response_messages[-1] if response_messages else None
        if not isinstance(final_msg, (AIMessage, AIMessageChunk)) or getattr(final_msg, "tool_calls", None):
            raise ValueError("模型未返回完整的最终回答")
        rag_context = get_last_rag_context(clear=True)
        rag_trace = rag_context.get("rag_trace") if rag_context else None
        # A cancelled await still drains the real commit before releasing admission.
        await asyncio.sleep(0)
        with span("conversation_save", component="postgres"):
            await run_blocking(storage.append_turn, user_id, session_id, user_text, _message_text(final_msg),
                               context_state=context_state, rag_trace=rag_trace, request_id=request_id, turn_identity=turn_identity)
        if current_run_state().answer_status == 'failed':
            current_run_state().answer_status = 'complete'
        yield f"data: {json.dumps({'type': 'rag_step', 'step': {'icon': '✅', 'label': '回答完成', 'detail': ''}})}\n\n"
        if rag_trace:
            yield f"data: {json.dumps({'type': 'trace', 'rag_trace': _public_rag_trace(rag_trace)})}\n\n"
        yield "data: [DONE]\n\n"
    except RunCancelled:
        mark_cancelled()
        raise asyncio.CancelledError from None
    except Exception as exc:
        partial_status = 'failed'
        mark_chat_failed()
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        mark_cancelled()
        active = False
        set_rag_step_queue(None)
        if agent_task is not None:
            if not agent_task.done():
                agent_task.cancel()
            # Repeated disconnect cancellation must not interrupt graph cleanup
            # or send a second cancellation into a tool that is draining a thread.
            while not agent_task.done():
                try:
                    await asyncio.shield(agent_task)
                except (asyncio.CancelledError, Exception):
                    pass
            if not agent_task.cancelled():
                agent_task.exception()
        if public_parts and turn_identity is not None:
            try:
                await run_blocking(ConversationStorage().save_interrupted, user_id, session_id,
                    ''.join(public_parts), turn_identity=turn_identity, status=partial_status)
            except TurnConflict:
                # Deleted sessions never receive a replacement or stale output.
                pass
