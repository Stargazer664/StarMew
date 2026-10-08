"""Versioned, network-denied characterization of the current local agent.

Run: python -X utf8 -m scripts.agent_baseline_offline --rounds 5 --output PATH
External provider/retrieval/memory/cache boundaries are deterministic; real
LangGraph, SSE, context fitting and ConversationStorage remain in the path.
The harness records defects rather than making them acceptance expectations.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from importlib.metadata import version

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = Path(__file__).with_name('agent_baseline_cases_v1.json')
_request = ContextVar('baseline_request', default='')
HARNESS_VERSION = 'offline-real-graph-v1.2'
FIXED_ENV = {
    'DATABASE_URL': 'sqlite:///:memory:', 'LANGCHAIN_TRACING_V2': 'false',
    'LANGSMITH_TRACING': 'false', 'CHAT_CONTEXT_WINDOW_TOKENS': '8192',
    'CHAT_OUTPUT_RESERVE_TOKENS': '2048', 'CHAT_HISTORY_MAX_TOKENS': '3072',
    'CHAT_SUMMARY_MAX_TOKENS': '768', 'CHAT_MEMORY_MAX_TOKENS': '768',
    'CHAT_TOKENIZER_PATH': '', 'MODEL': 'offline-scripted-v1',
    'BASE_URL': 'http://127.0.0.1:1/v1', 'ARK_API_KEY': 'offline-only',
}


def load_cases():
    return json.loads(CASES_PATH.read_text(encoding='utf-8'))['cases']


def inspect_tool_protocol(messages):
    """Check ordered AI call/ToolMessage groups in one actual model input."""
    expected, matched, missing, unexpected, duplicate, invalid = set(), set(), set(), set(), set(), []
    pending = {}
    seen_responses = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            identifier = message.tool_call_id
            if identifier in seen_responses:
                duplicate.add(identifier)
            seen_responses.add(identifier)
            if identifier not in pending:
                unexpected.add(identifier)
            else:
                pending[identifier] += 1
            continue
        for identifier, count in pending.items():
            if count == 0: missing.add(identifier)
            elif count == 1: matched.add(identifier)
            else: duplicate.add(identifier)
        pending = {}
        for call in getattr(message, 'tool_calls', []):
            identifier = call.get('id')
            if not identifier:
                invalid.append('tool call lacks id')
                continue
            if identifier in expected:
                invalid.append('duplicate tool call id: ' + identifier)
            expected.add(identifier)
            pending[identifier] = 0
    for identifier, count in pending.items():
        if count == 0: missing.add(identifier)
        elif count == 1: matched.add(identifier)
        else: duplicate.add(identifier)
    failed = bool(missing or unexpected or duplicate or invalid)
    return {'status': 'failed' if failed else 'passed' if expected else 'not_applicable',
            'expected_call_ids': sorted(expected), 'matched_call_ids': sorted(matched),
            'missing_response_ids': sorted(missing), 'unexpected_response_ids': sorted(unexpected),
            'duplicate_response_ids': sorted(duplicate), 'invalid_calls': invalid}


def tool_protocol_result(observations, emitted_calls):
    boundaries = [{'purpose': o['purpose'], **o['input_tool_protocol']} for o in observations if 'input_tool_protocol' in o]
    ordinary = [call for call in emitted_calls if call['name'] != 'finish']
    controls = [call for call in emitted_calls if call['name'] == 'finish']
    matched = {identifier for boundary in boundaries for identifier in boundary['matched_call_ids']}
    unobserved = sorted({call['id'] for call in emitted_calls} - matched)
    if any(boundary['status'] == 'failed' for boundary in boundaries):
        status, result = 'failed', False
    elif not boundaries:
        status, result = 'unobserved', None
    elif unobserved:
        status, result = 'unverified', None
    elif emitted_calls or any(boundary['expected_call_ids'] for boundary in boundaries):
        status, result = 'passed', True
    else:
        status, result = 'not_applicable', None
    return {'matched_tool_protocol': result, 'tool_protocol_status': status,
            'tool_protocol_observations': boundaries, 'tool_protocol_unobserved_call_ids': unobserved,
            'tool_protocol_emitted_calls': ordinary, 'tool_protocol_control_calls': controls}


class ScriptedModel(BaseChatModel):
    case: dict = Field(default_factory=dict)
    observations: list = Field(default_factory=list)
    final_inputs: dict = Field(default_factory=dict)
    emitted_calls: list = Field(default_factory=list)
    entered: set = Field(default_factory=set)
    gate: object = None
    overlap: bool = False

    @property
    def _llm_type(self):
        return 'offline-scripted-v1'

    def bind_tools(self, tools, **kwargs):
        return self.bind(route=True)

    def record(self, purpose, messages):
        from backend.conversation_context import ContextBudget
        budget = ContextBudget.from_env()
        self.observations.append({'request': _request.get(), 'purpose': purpose,
                                  'input_tool_protocol': inspect_tool_protocol(messages),
                                  'input_utf8_budget_estimate': budget.count_messages(messages)})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.record('summary', messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='已有会话的固定离线摘要。'))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        query = _request.get()
        if kwargs.get('route'):
            self.record('decision', messages)
            category = self.case['category']
            has_tool = any(isinstance(m, ToolMessage) for m in messages)
            needs_tool = category in {'tool_success', 'tool_unavailable', 'tool_exception',
                                      'tool_repeat', 'no_evidence', 'long_evidence'}
            if needs_tool and (not has_tool or category == 'tool_repeat'):
                name = {'tool_unavailable': 'get_current_weather', 'tool_exception': 'web_search'}.get(category, 'search_knowledge_base')
                args = {'location' if name == 'get_current_weather' else 'query': query}
            else:
                name, args = 'finish', {}
            ordinal = sum(o['request'] == query and o['purpose'] == 'decision' for o in self.observations)
            self.emitted_calls.append({'request': query, 'name': name, 'id': f'call-{ordinal}'})
            yield ChatGenerationChunk(message=AIMessageChunk(content='', tool_call_chunks=[{
                'name': name, 'args': json.dumps(args, ensure_ascii=False),
                'id': f'call-{ordinal}', 'index': 0}]))
            return
        self.record('final', messages)
        self.final_inputs[query] = list(messages)
        activation = object()
        self.entered.add(activation)
        try:
            async for chunk in self._final_chunks(messages, query):
                yield chunk
        finally:
            self.entered.discard(activation)

    async def _final_chunks(self, messages, query):
        if self.gate is not None:
            if len(self.entered) >= 2:
                self.overlap = True
                self.gate.set()
            # Bounded barrier also allows T09 serialized/rejected implementations.
            try:
                await asyncio.wait_for(self.gate.wait(), timeout=0.15)
            except asyncio.TimeoutError:
                self.gate.set()
        text = '\n'.join(str(m.content) for m in messages)
        if self.case['category'] == 'no_evidence':
            answer = '没有足够的相关证据，无法确认。'
        elif self.case['category'] == 'tool_unavailable':
            answer = '天气服务未配置，无法获取实时天气。'
        elif self.case['category'] == 'tool_exception':
            answer = '工具执行失败，无法确认外部信息。'
        else:
            kept = [doc['filename'] for doc in fixture_docs(self.case) if doc['filename'] in text]
            answer = '离线回答：' + query[:128] + ('；来源：' + '、'.join(kept) if kept else '')
        midpoint = max(1, len(answer) // 2)
        for piece in (answer[:midpoint], answer[midpoint:]):
            await asyncio.sleep(0)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def fixture_docs(case):
    category = case['category']
    if category not in {'tool_success', 'tool_repeat', 'long_evidence', 'no_evidence'}:
        return []
    if case.get('evidence') == 'empty':
        return []
    count = 3 if category == 'long_evidence' else 1
    text = ('证据正文，审批须提交申请并由负责人审核。' * 1200
            if category == 'long_evidence' else
            '无关设备目录，不能证明审批规则。' if case.get('evidence') == 'irrelevant' else
            '审批步骤：提交申请，由负责人审核，保存验收记录。')
    return [{'chunk_id': f'{case["id"]}-doc-{i}', 'filename': f'{case["id"]}-source-{i}.pdf',
             'page_number': i, 'text': text, 'score': 1.0 - i / 10, 'rrf_rank': i}
            for i in range(1, count + 1)]


@contextmanager
def boundaries(case, db_url=None):
    """Never opens the configured production DB or remote provider/retriever."""
    attempts = []
    original_connect = socket.socket.connect
    def deny(*args, **kwargs):
        attempts.append('network attempt denied')
        raise AssertionError('offline benchmark attempted network I/O')
    pg_url = make_url(db_url) if db_url else None
    if pg_url and (pg_url.host != '127.0.0.1' or pg_url.port != 55432 or pg_url.database != 'agent_optimization_test'):
        raise ValueError('PostgreSQL probe is restricted to the task-only local test database')
    def guarded_connect(sock, address):
        if pg_url and isinstance(address, tuple) and address[:2] == ('127.0.0.1', 55432):
            return original_connect(sock, address)
        return deny(sock, address)
    cache = SimpleNamespace(get_json=lambda *a: None, set_json=lambda *a, **k: None,
                            delete=lambda *a: None)
    fake_model = ScriptedModel(case=case)
    with tempfile.TemporaryDirectory(prefix='supermew-t00-') as temp, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, FIXED_ENV))
        stack.enter_context(patch.object(socket.socket, 'connect', guarded_connect))
        if db_url is None: stack.enter_context(patch.object(socket, 'create_connection', deny))
        from backend import llm_config
        stack.enter_context(patch.object(llm_config, 'create_chat_model', return_value=fake_model))
        from backend import agent, database
        from backend.models import User, ChatMessage, ChatSession
        schema = None
        if pg_url:
            schema = 't00_' + uuid.uuid4().hex
            raw_engine = create_engine(db_url, connect_args={'connect_timeout': 3,
                'options': '-c statement_timeout=5000 -c lock_timeout=2000'})
            with raw_engine.begin() as connection:
                connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            engine = raw_engine.execution_options(schema_translate_map={None: schema})
            def cleanup_schema():
                # Exact generated task schema; never drop a database/public schema.
                assert schema.startswith('t00_') and len(schema) == 36
                with raw_engine.begin() as connection:
                    connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                raw_engine.dispose()
            stack.callback(cleanup_schema)
        else:
            engine = create_engine(f'sqlite:///{Path(temp) / "baseline.db"}')
        database.Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        with sessions() as db:
            db.add(User(username='baseline-user', password_hash='offline-only'))
            db.commit()
        stack.callback(engine.dispose)
        stack.enter_context(patch.object(agent, 'SessionLocal', sessions))
        from backend import conversation_turns, memory as memory_module, memory_jobs
        stack.enter_context(patch.object(memory_module, "SessionLocal", sessions))
        stack.enter_context(patch.object(memory_jobs, "SessionLocal", sessions))
        stack.enter_context(patch.object(conversation_turns, 'SessionLocal', sessions))
        stack.enter_context(patch.object(database, 'SessionLocal', sessions))
        stack.enter_context(patch.object(agent, 'cache', cache))
        storage = agent.ConversationStorage()
        stack.enter_context(patch.object(agent, 'storage', storage))
        memory_calls = []
        def extract_memory(messages):
            last_user = next((m['content'] for m in reversed(messages) if m['type'] == 'human'), '')
            memory_calls.append(last_user)
            if case.get('memory_failure') and last_user.endswith('第三轮'):
                raise RuntimeError('offline injected memory extraction failure')
            return {}
        job_service = memory_module.UserMemoryService()
        stack.enter_context(patch.object(job_service, '_extract_facts_by_llm', extract_memory))
        stack.enter_context(patch.object(memory_module, 'cache', cache))
        memory = SimpleNamespace(get_memory_for_context=lambda *a: '用户偏好简洁回答。')
        stack.enter_context(patch.object(agent, 'memory_service', memory))
        stack.enter_context(patch.object(agent, 'init_chat_model', return_value=fake_model))
        stack.enter_context(patch.object(agent, 'model', fake_model))
        graph, _ = agent._create_react_agent()
        stack.enter_context(patch.object(agent, 'react_graph', graph))
        retrieval_calls = []
        def retrieve(query, **kwargs):
            from backend.agent_budget import current_run_state
            run = current_run_state()
            if run and not run.claim_retrieval():
                return {'docs': [], 'meta': {'retrieval_mode': 'skipped', 'retrieval_status': 'budget_exhausted'}}
            retrieval_calls.append({'request': _request.get(), 'query': query})
            return {'docs': fixture_docs(case), 'meta': {'retrieval_mode': 'offline-fixture', 'candidate_k': 5}}
        fake_utils = SimpleNamespace(retrieve_documents=retrieve,
            step_back_expand=lambda query: {'expanded_query': '固定扩展查询', 'step_back_question': '固定退步问题', 'step_back_answer': '固定扩展文本'},
            generate_hypothetical_document=lambda query: '固定假设文档')
        spec = importlib.util.spec_from_file_location('backend.rag_pipeline', ROOT / 'backend/rag_pipeline.py')
        pipeline = importlib.util.module_from_spec(spec)
        stack.enter_context(patch.dict(sys.modules, {'backend.rag_utils': fake_utils, 'backend.rag_pipeline': pipeline}))
        spec.loader.exec_module(pipeline)
        grade_calls = []
        class Grader:
            def with_structured_output(self, schema, **kwargs):
                return self
            def invoke(self, messages, **kwargs):
                grade_calls.append({'request': _request.get(), 'purpose': 'grading',
                                    'input_utf8_budget_estimate': len(str(messages).encode('utf-8'))})
                return SimpleNamespace(binary_score='no' if case['category'] == 'no_evidence' else 'yes')
        stack.enter_context(patch.object(pipeline, '_get_grader_model', return_value=Grader()))
        stack.enter_context(patch.object(pipeline, '_get_router_model', return_value=None))
        from backend import tools
        stack.enter_context(patch.object(tools, 'AMAP_WEATHER_API', None))
        stack.enter_context(patch.object(tools, 'AMAP_API_KEY', None))
        stack.enter_context(patch.object(tools, 'TAVILY_API_KEY', 'offline-only'))
        def failed_web_request(*args, **kwargs):
            """Use the real web tool contract, replacing only its HTTP boundary."""
            raise RuntimeError('offline injected tool exception')
        stack.enter_context(patch.object(tools.requests, 'post', failed_web_request))
        tool_calls = []
        original_tool = agent.invoke_tool
        def measured_tool(tool, inputs, **kwargs):
            call = {'request': _request.get(), 'name': tool.name if tool else 'unknown', 'args': inputs}
            tool_calls.append(call)
            result = original_tool(tool, inputs, **kwargs)
            call.update(status=result.status, code=result.code, retryable=result.retryable,
                        evidence_status=result.evidence_status)
            return result
        stack.enter_context(patch.object(agent, 'invoke_tool', measured_tool))
        yield SimpleNamespace(agent=agent, model=fake_model, sessions=sessions, storage=storage,
            attempts=attempts, retrieval_calls=retrieval_calls, grade_calls=grade_calls,
            tool_calls=tool_calls, memory_calls=memory_calls, memory_jobs=memory_jobs, job_service=job_service, ChatMessage=ChatMessage,
            ChatSession=ChatSession)


def source_keys(docs):
    return {(d['filename'], d['page_number']) for d in docs}


async def measure_request(env, case, prompt):
    token = _request.set(prompt)
    start, first = time.perf_counter(), None
    events = []
    admission_conflict = None
    from backend.conversation_turns import TurnConflict
    try:
        async with __import__('contextlib').aclosing(env.agent.chat_with_agent_stream(prompt, 'baseline-user', case['id'])) as stream:
            async for chunk in stream:
                payload = chunk.removeprefix('data: ').strip()
                event = '[DONE]' if payload == '[DONE]' else json.loads(payload)
                events.append(event)
                if first is None and isinstance(event, dict) and event.get('type') == 'content' and event.get('content'):
                    first = time.perf_counter()
    except TurnConflict as exc:
        admission_conflict = str(exc)
    finally:
        end = time.perf_counter()
        _request.reset(token)
    calls = [o for o in env.model.observations + env.grade_calls if o['request'] == prompt]
    final_messages = env.model.final_inputs.get(prompt, [])
    final_text = '\n'.join(str(m.content) for m in final_messages)
    docs = fixture_docs(case)
    expected_sources = source_keys(docs)
    preserved = {(filename, page) for filename, page in expected_sources
                 if f'{filename} (Page {page})' in final_text}
    trace = next((e['rag_trace'] for e in events if isinstance(e, dict) and e.get('type') == 'trace'), {})
    trace_sources = source_keys(trace.get('retrieved_chunks', []))
    protocol = tool_protocol_result(calls, [call for call in env.model.emitted_calls if call['request'] == prompt])
    return {'prompt': prompt, 'admission_conflict': admission_conflict, 'model_calls': len(calls), 'model_calls_by_purpose': {
                p: sum(o['purpose'] == p for o in calls) for p in sorted({o['purpose'] for o in calls})},
            'tool_calls': sum(c['request'] == prompt for c in env.tool_calls),
            'tool_results': [{k: c[k] for k in ('name', 'status', 'code', 'retryable', 'evidence_status')}
                             for c in env.tool_calls if c['request'] == prompt],
            'retrieval_calls': sum(c['request'] == prompt for c in env.retrieval_calls),
            'first_public_answer_ms': (first - start) * 1000 if first is not None else None,
            'completion_ms': (end - start) * 1000, 'input_tokens': None, 'output_tokens': None,
            'token_usage_status': 'missing: fake boundary returns no provider usage',
            'model_input_utf8_budget_estimates': [o['input_utf8_budget_estimate'] for o in calls],
            'final_input_source_retention': len(preserved) / len(expected_sources) if expected_sources else None,
            'trace_source_retention': len(trace_sources & expected_sources) / len(expected_sources) if expected_sources else None,
            'expected_sources': sorted(expected_sources), 'final_input_sources': sorted(preserved),
            'public_answer': ''.join(e['content'] for e in events if isinstance(e, dict) and e.get('type') == 'content'),
            'errors': [e['content'] for e in events if isinstance(e, dict) and e.get('type') == 'error'],
            'done_count': events.count('[DONE]'), **protocol,
            'final_input_human_messages': sum(isinstance(m, HumanMessage) for m in final_messages),
            'events': events}


async def run_case(case, db_url=None):
    setup = time.perf_counter()
    with boundaries(case, db_url=db_url) as env:
        setup_ms = (time.perf_counter() - setup) * 1000
        historical_timestamps = []
        if case['category'] == 'same_session_concurrent':
            env.model.gate = asyncio.Event()
            requests = await asyncio.wait_for(asyncio.gather(
                measure_request(env, case, case['prompt'] + '：请求一'),
                measure_request(env, case, case['prompt'] + '：请求二')), timeout=10)
        elif case['category'] == 'multi_turn':
            requests = []
            for ordinal in ('第一轮', '第二轮', '第三轮'):
                prompt = case['prompt'] + ordinal
                if case.get('first_turn_suffix') and ordinal == '第一轮':
                    prompt += case['first_turn_suffix']['text'] * case['first_turn_suffix']['repeat']
                requests.append(await asyncio.wait_for(measure_request(env, case, prompt), timeout=10))
                with env.sessions() as db:
                    first_row = db.query(env.ChatMessage).order_by(env.ChatMessage.id).first()
                    historical_timestamps.append(first_row.timestamp.isoformat() if first_row else None)
        else:
            requests = [await asyncio.wait_for(measure_request(env, case, case['prompt']), timeout=10)]
        # Explicitly observe the outbox AFTER all foreground timing has ended.
        # Use the real incremental memory service with an offline extraction boundary.
        while True:
            job = await asyncio.to_thread(env.memory_jobs.claim_next_job)
            if job is None:
                break
            await asyncio.to_thread(env.memory_jobs.process_job, job, env.job_service)
        with env.sessions() as db:
            from backend.models import MemoryUpdateJob
            memory_jobs = [{'target_message_count': j.target_message_count, 'status': j.status,
                'attempts': j.attempts, 'last_error_code': j.last_error_code, 'request_id': j.request_id}
                for j in db.query(MemoryUpdateJob).order_by(MemoryUpdateJob.id).all()]
        with env.sessions() as db:
            rows = db.query(env.ChatMessage).order_by(env.ChatMessage.id).all()
            persisted = [{'type': row.message_type, 'content': row.content,
                          'timestamp': row.timestamp.isoformat(), 'rag_trace': row.rag_trace} for row in rows]
        findings = []
        if any(r['admission_conflict'] and (case['category'] != 'same_session_concurrent' or r['admission_conflict'] != 'SESSION_BUSY') for r in requests):
            findings.append('unexpected admission conflict')
        if any(r['tool_protocol_status'] == 'failed' for r in requests): findings.append('tool protocol mismatch at model input')
        if any(r['tool_protocol_status'] == 'unverified' for r in requests): findings.append('tool protocol has unverified emitted calls')
        if any(r['tool_protocol_status'] == 'unobserved' and not r['admission_conflict'] for r in requests): findings.append('tool protocol model input not observed')
        if any(r['errors'] for r in requests): findings.append('SSE error: ' + '; '.join(e for r in requests for e in r['errors']))
        if any(r['done_count'] != 1 and not r['admission_conflict'] for r in requests): findings.append('DONE count is not one')
        if any(r['final_input_source_retention'] is not None and r['final_input_source_retention'] < 1 for r in requests): findings.append('source/page lost before final model')
        if case['category'] in {'tool_unavailable', 'tool_exception'} and any(
            isinstance(e, dict) and e.get('type') == 'rag_step' and e['step'].get('label', '').startswith('工具 ') and '执行完成' in e['step']['label']
            for r in requests for e in r['events']):
            findings.append('failed/unavailable tool reported execution complete')
        if case['category'] == 'no_evidence' and any('forced: has retrieval results' in str(r['events']) for r in requests):
            findings.append('irrelevant evidence forced past no grade')
        if len(set(historical_timestamps)) > 1: findings.append('earliest historical timestamp overwritten')
        if env.model.overlap: findings.append('same session entered final nodes concurrently')
        expected_count = 2 * sum(not r['admission_conflict'] and (not r['errors'] or bool(r['public_answer'])) for r in requests)
        if len(persisted) != expected_count: findings.append('persisted message count differs from completed turns')
        if len(memory_jobs) != len(persisted) // 2: findings.append('committed answer missing durable memory job')
        if case.get('memory_failure') and not any(j['last_error_code'] == 'MEMORY_UPDATE_FAILED' for j in memory_jobs):
            findings.append('background memory failure injection was not observed')
        return {'id': case['id'], 'category': case['category'], 'expected': case['expected'],
                'setup_ms': setup_ms, 'requests': requests, 'target_passed': not findings,
                'findings': findings, 'persisted_message_count': len(persisted),
                'persisted_messages': persisted, 'historical_first_timestamps': historical_timestamps,
                'overlapping_final_nodes': env.model.overlap, 'network_attempts': len(env.attempts),
                'memory_jobs': memory_jobs, 'background_memory_extraction_calls': len(env.memory_calls),
                'background_memory_observation': 'real outbox and checkpoint service; scripted extraction outside foreground timing'}


def quantile(values, q):
    values = sorted(v for v in values if v is not None)
    return values[max(0, math.ceil(len(values) * q) - 1)] if values else None


def aggregate(records):
    groups = {}
    for group, selected in [('all', records), ('pure_chat', [r for r in records if r['category'] == 'chat']),
                             ('retrieval', [r for r in records if r['category'] in {'tool_success', 'tool_repeat', 'no_evidence', 'long_evidence'}])]:
        requests = [q for record in selected for q in record['requests']]
        groups[group] = {'scenario_n': len(selected), 'request_n': len(requests),
            'failure_rate': sum(bool(q['errors']) for q in requests) / len(requests) if requests else None,
            'target_failure_rate': sum(not r['target_passed'] for r in selected) / len(selected) if selected else None,
            'model_calls': sum(q['model_calls'] for q in requests), 'tool_calls': sum(q['tool_calls'] for q in requests),
            'first_public_answer_ms_p50': quantile([q['first_public_answer_ms'] for q in requests], .5),
            'first_public_answer_ms_p95': quantile([q['first_public_answer_ms'] for q in requests], .95),
            'completion_ms_p50': quantile([q['completion_ms'] for q in requests], .5),
            'completion_ms_p95': quantile([q['completion_ms'] for q in requests], .95),
            'input_tokens': None, 'output_tokens': None,
            'source_retention_request_n': sum(q['final_input_source_retention'] is not None for q in requests),
            'final_input_source_retention_mean': (sum(q['final_input_source_retention'] for q in requests if q['final_input_source_retention'] is not None) /
                sum(q['final_input_source_retention'] is not None for q in requests)) if any(q['final_input_source_retention'] is not None for q in requests) else None}
    return groups


async def benchmark(rounds, selected_ids=None):
    start = time.perf_counter()
    cases = [c for c in load_cases() if not selected_ids or c['id'] in selected_ids]
    records = []
    for round_index in range(1, rounds + 1):
        for index, case in enumerate(cases, 1):
            print(f'昕 [{round_index}/{rounds} 轮][{index}/{len(cases)}] {case["id"]}', flush=True)
            record = await run_case(case)
            record.update(round=round_index, process_phase='first_round' if round_index == 1 else 'warm_imports')
            records.append(record)
    config = json.loads(CASES_PATH.read_text(encoding='utf-8'))
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [Path(__file__), CASES_PATH] + [ROOT / 'backend' / name for name in
              ('agent.py', 'tools.py', 'tool_contracts.py', 'chat_diagnostics.py', 'telemetry.py',
               'rag_pipeline.py', 'conversation_context.py', 'memory.py', 'models.py')]}
    return {'harness_version': HARNESS_VERSION, 'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'packages': {name: version(name) for name in ('langchain-core', 'langgraph', 'SQLAlchemy', 'pytest')},
                        'database': 'fresh isolated SQLite file per scenario', 'network': 'socket connect denied'},
        'fixed_environment': FIXED_ENV, 'case_config': config, 'sha256': hashes,
        'limitations': ['No real provider/model, Milvus, Redis, embedding or reranker is contacted.',
            'No provider input/output token usage exists; null is missing, not zero. UTF-8 estimates are budget estimates only.',
            'Latency covers local graph/SSE/SQLite and instrumentation, not network/model inference latency.',
            'First SSE content chunk is proxy for first public token, not tokenizer-level timing.',
            'first_round vs warm_imports only separates process import state; every case gets a fresh graph and DB. Real cold provider/cache startup is unverified.',
            'Scripted model answers do not prove semantic model quality; source/page retention and grade routing are structural measurements.',
            'SQLite cannot validate PostgreSQL row locking, isolation or crash recovery.',
            'p50/p95 are nearest-rank descriptive statistics; these small samples do not establish stable percentiles.'],
        'metric_definitions': {
            'model_calls': 'actual scripted generation/decision/summary calls plus structured grader calls; expansion/router model disabled and recorded',
            'tool_calls': 'invoke_tool dispatch count, including repeated, unavailable and failing tool attempts',
            'first_public_answer_ms': 'request stream consumption start to first nonempty public SSE content; null if absent',
            'completion_ms': 'request start to generator exhaustion after DONE and cleanup',
            'failure_rate': 'requests with SSE error / completed benchmark requests',
            'target_failure_rate': 'scenarios with at least one specified target violation / measured scenarios',
            'source_retention': 'distinct fixture filename/page pairs in final model input or public trace / expected fixture source pairs; null if denominator empty',
            'matched_tool_protocol': 'ordered same-ID tool pairing at every observed model input, including internal finish; true only when observed and all emitted calls verified, false for mismatch, null for unobserved/unverified/not_applicable',
            'tool_protocol_control_calls': 'finish decisions are internal control signals with paired ToolMessages, never external execution; listed separately from ordinary tool calls',
            'overlapping_final_nodes': 'two final streams active at the same time; active membership removed in finally on completion or cancellation',
            'target_passed': 'structural target checks only; does not certify LLM factual correctness or PostgreSQL safety'},
        'rounds': rounds, 'total_ms': (time.perf_counter() - start) * 1000,
        'summary': aggregate(records),
        'phase_summary': {phase: aggregate([r for r in records if r['process_phase'] == phase]) for phase in ('first_round', 'warm_imports')},
        'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--postgres-env', help='Environment variable holding the task-only PostgreSQL URL; only concurrency probes run')
    args = parser.parse_args()
    if args.rounds < 1: parser.error('--rounds must be positive')
    if args.postgres_env:
        async def pg_probe():
            records = []
            for index, case in enumerate([c for c in load_cases() if c['category'] == 'same_session_concurrent'], 1):
                print(f'昕 [PostgreSQL {index}/3] {case["id"]}', flush=True)
                records.append(await run_case(case, db_url=os.environ[args.postgres_env]))
            return {'harness_version': HARNESS_VERSION, 'database': 'task-only PostgreSQL 127.0.0.1:55432; random schema per case, dropped after measurement',
                    'timeouts': {'connect_seconds': 3, 'statement_ms': 5000, 'lock_ms': 2000},
                    'records': records, 'summary': aggregate(records)}
        data = asyncio.run(pg_probe())
    else:
        data = asyncio.run(benchmark(args.rounds, args.case))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print('昕 baseline saved: ' + str(args.output), flush=True)
    print(json.dumps(data['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
