"""Tool outcomes are machine readable; external services remain offline."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'

import pytest
import requests
from backend import tools
from backend.chat_diagnostics import invoke_tool
from backend import telemetry
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from test_agent_streaming import agent, ToolDecisionBoundaryModel, FakeMemoryService, FakeStorage, patched_stream_dependencies, parse_sse


def response(data):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)


@pytest.mark.parametrize('inputs,configured,data,error,want', [
    ({'location': ''}, True, {}, None, ('error', 'INVALID_ARGUMENT', False)),
    ({'location': '  '}, True, {}, None, ('error', 'INVALID_ARGUMENT', False)),
    ({'location': '北京', 'extensions': 'wrong'}, True, {}, None, ('error', 'INVALID_ARGUMENT', False)),
    ({'location': '北京'}, False, {}, None, ('error', 'NOT_CONFIGURED', False)),
    ({'location': '北京'}, True, {}, requests.exceptions.Timeout('private token'), ('error', 'UPSTREAM_TIMEOUT', True)),
    ({'location': '北京'}, True, {}, requests.exceptions.ConnectionError('private token'), ('error', 'UPSTREAM_ERROR', True)),
    ({'location': '北京'}, True, {'status': '0', 'info': 'private token'}, None, ('error', 'UPSTREAM_ERROR', False)),
    ({'location': '北京'}, True, {'status': '1', 'lives': []}, None, ('empty', None, False)),
    ({'location': '北京', 'extensions': 'all'}, True, {'status': '1', 'forecasts': []}, None, ('empty', None, False)),
    ({'location': '北京'}, True, {'status': '1', 'lives': [{'city': '北京', 'weather': '晴'}]}, None, ('ok', None, False)),
])
def test_weather_returns_explicit_status_without_exception_details(monkeypatch, inputs, configured, data, error, want):
    monkeypatch.setattr(tools, 'AMAP_WEATHER_API', 'https://example.invalid' if configured else None)
    monkeypatch.setattr(tools, 'AMAP_API_KEY', 'test' if configured else None)
    def get(*args, **kwargs):
        if error:
            raise error
        return response(data)
    monkeypatch.setattr(tools.requests, 'get', get)
    result = tools.get_current_weather.invoke(inputs)
    assert (getattr(result, 'status', None), getattr(result, 'code', None), getattr(result, 'retryable', None)) == want
    assert 'private token' not in result.content


@pytest.mark.parametrize('query,configured,data,error,want', [
    ('', True, {}, None, ('error', 'INVALID_ARGUMENT', False)),
    ('   ', True, {}, None, ('error', 'INVALID_ARGUMENT', False)),
    ('新闻', False, {}, None, ('error', 'NOT_CONFIGURED', False)),
    ('新闻', True, {}, requests.exceptions.Timeout('private token'), ('error', 'UPSTREAM_TIMEOUT', True)),
    ('新闻', True, {}, requests.exceptions.ConnectionError('private token'), ('error', 'UPSTREAM_ERROR', True)),
    ('新闻', True, {'results': []}, None, ('empty', None, False)),
    ('新闻', True, {'results': [{'title': '今日', 'url': 'https://example.invalid', 'content': '文本'}]}, None, ('ok', None, False)),
])
def test_web_returns_explicit_status(monkeypatch, query, configured, data, error, want):
    monkeypatch.setattr(tools, 'TAVILY_API_KEY', 'test' if configured else None)
    def post(*args, **kwargs):
        if error:
            raise error
        return response(data)
    monkeypatch.setattr(tools.requests, 'post', post)
    result = tools.web_search.invoke({'query': query})
    assert (getattr(result, 'status', None), getattr(result, 'code', None), getattr(result, 'retryable', None)) == want
    assert 'private token' not in result.content


def test_knowledge_base_returns_evidence_and_limit_as_skipped(monkeypatch):
    docs = [{'filename': 'rules.pdf', 'page_number': 2, 'text': '审批由负责人审核'}]
    monkeypatch.setitem(sys.modules, 'backend.rag_pipeline', SimpleNamespace(run_rag_graph=lambda query: {'docs': docs, 'rag_trace': {'request': query}}))
    tools.reset_tool_call_guards()
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    from backend.chat_diagnostics import invoke_tool
    with use_run_state(AgentRunState(AgentBudget(max_kb_calls=1))):
        first = invoke_tool(tools.search_knowledge_base, {'query': '审批'}, purpose='knowledge_search')
        second = invoke_tool(tools.search_knowledge_base, {'query': '新问题'}, purpose='knowledge_search')
    assert getattr(first, 'status', None) == 'ok'
    assert first.evidence == docs
    assert (second.status, second.code, second.retryable) == ('skipped', 'CALL_LIMIT_REACHED', False)
    assert tools.get_last_rag_context()['rag_trace']['request'] == '审批'


def test_knowledge_base_empty_and_invalid_are_distinct(monkeypatch):
    monkeypatch.setitem(sys.modules, 'backend.rag_pipeline', SimpleNamespace(run_rag_graph=lambda query: {'docs': [], 'rag_trace': {}}))
    tools.reset_tool_call_guards()
    invalid = tools.search_knowledge_base.invoke({'query': ' '})
    empty = tools.search_knowledge_base.invoke({'query': '问题'})
    assert (getattr(invalid, 'status', None), getattr(invalid, 'code', None)) == ('error', 'INVALID_ARGUMENT')
    assert (empty.status, empty.evidence_status, empty.evidence) == ('empty', 'insufficient', [])


def test_invoke_tool_classifies_missing_arguments_and_unknown_tools():
    invalid = invoke_tool(tools.web_search, {}, purpose='web_search')
    unknown = invoke_tool(None, {}, purpose='unknown')
    assert (invalid.status, invalid.code) == ('error', 'INVALID_ARGUMENT')
    assert (unknown.status, unknown.code) == ('error', 'UNKNOWN_TOOL')


def test_knowledge_result_and_trace_remain_request_local(monkeypatch):
    monkeypatch.setitem(sys.modules, 'backend.rag_pipeline', SimpleNamespace(run_rag_graph=lambda query: {'docs': [{'filename': query, 'text': query}], 'rag_trace': {'request': query}}))
    async def request(query):
        tools.reset_tool_call_guards()
        result = await asyncio.to_thread(tools.search_knowledge_base.invoke, {'query': query})
        await asyncio.sleep(0)
        return result, tools.get_last_rag_context()
    async def run():
        return await asyncio.gather(request('A'), request('B'))
    for query, (result, trace) in zip(['A', 'B'], asyncio.run(run())):
        assert getattr(result, 'status', None) == 'ok'
        assert result.evidence[0]['filename'] == query
        assert trace['rag_trace']['request'] == query
        assert trace['rag_trace']['retrieval_rounds'] == [{'request': query}]
        assert trace['rag_trace']['retrieval_records'] == []


@pytest.fixture
def diagnostics(tmp_path):
    telemetry.start_diagnostics(tmp_path / 'diagnostics.db', tmp_path / 'logs')
    yield telemetry
    telemetry.stop_diagnostics()


@pytest.mark.parametrize('status,code,retryable,evidence_status,icon,label,span_status', [
    ('ok', None, False, 'unknown', '✅', '执行完成', 'completed'),
    ('empty', None, False, 'insufficient', 'ℹ️', '无结果', 'empty'),
    ('error', 'UPSTREAM_TIMEOUT', True, None, '❌', '执行失败', 'failed'),
    ('skipped', 'CALL_LIMIT_REACHED', False, None, '⏭️', '已跳过', 'skipped'),
])
def test_model_progress_and_persisted_diagnostics_agree(diagnostics, status, code, retryable, evidence_status, icon, label, span_status):
    from backend.tool_contracts import ToolResult
    from langchain_core.tools import tool
    @tool('web_search')
    def fixture_tool(query: str) -> ToolResult:
        """Offline service outcome at the provider boundary."""
        return ToolResult(status, 'private result', code=code, retryable=retryable, evidence_status=evidence_status)
    async def run():
        model = ToolDecisionBoundaryModel(decisions=[
            AIMessageChunk(content='private decision', tool_calls=[{'name': 'web_search', 'args': {'query': 'private args'}, 'id': 'call-outcome'}]),
            AIMessageChunk(content='private finish decision', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'call-finish'}]),
        ])
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        with telemetry.bind_context(trace_id='outcome'), patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()), patch.object(agent, 'web_search', fixture_tool):
            events = parse_sse([event async for event in agent.chat_with_agent_stream('问题')])
        messages = [m for m in model.final_inputs[0] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in messages] == ['call-outcome', 'call-finish']
        outcome = json.loads(messages[0].content)
        assert (outcome['status'], outcome['code'], outcome['retryable'], outcome['evidence_status']) == (status, code, retryable, evidence_status)
        steps = [e['step'] for e in events if isinstance(e, dict) and e.get('type') == 'rag_step']
        assert any(s['icon'] == icon and label in s['label'] for s in steps)
        if status != 'ok':
            assert not any(s['icon'] == '✅' and s['label'].startswith('工具 ') for s in steps)
        assert all('private' not in str(e) for e in events)
        assert all(m.content != 'private finish decision' for m in model.final_inputs[0])
    asyncio.run(run())
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('outcome')
    spans = [s for s in detail['spans'] if s['stage'] == 'tool']
    assert len(spans) == 1  # finish is internal control, never an external operation.
    outcome = spans[0]
    assert (outcome['status'], outcome['tool_status'], outcome.get('code'), outcome['retryable'], outcome.get('evidence_status')) == (span_status, status, code, retryable, evidence_status)
    assert 'private' not in json.dumps(detail)


@pytest.mark.parametrize('weather,web,want', [
    (False, False, ['search_knowledge_base']),
    (True, False, ['get_current_weather', 'search_knowledge_base']),
    (False, True, ['search_knowledge_base', 'web_search']),
    (True, True, ['get_current_weather', 'search_knowledge_base', 'web_search']),
])
def test_current_round_binding_excludes_unconfigured_tools(monkeypatch, weather, web, want):
    class BindingModel(ToolDecisionBoundaryModel):
        bindings: list = []
        def bind_tools(self, tools, **kwargs):
            self.bindings.append([t['function']['name'] if isinstance(t, dict) else t.name for t in tools])
            return super().bind_tools(tools, **kwargs)
    async def run():
        model = BindingModel(decisions=[AIMessageChunk(content='', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish'}])])
        # The compiled graph survives configuration changes between chat turns.
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        monkeypatch.setattr(tools, 'AMAP_WEATHER_API', 'https://example.invalid' if weather else None)
        monkeypatch.setattr(tools, 'AMAP_API_KEY', 'test' if weather else None)
        monkeypatch.setattr(tools, 'TAVILY_API_KEY', 'test' if web else None)
        with patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()):
            await graph.ainvoke({'messages': [HumanMessage(content='问题')], 'step_count': 0, 'memory_context': '', 'tool_output': None})
        assert model.bindings[-1] == want + ['finish']
    asyncio.run(run())


def test_one_failure_does_not_break_other_call_pairs_or_leak_exception(diagnostics, monkeypatch):
    from langchain_core.tools import tool
    from backend.tool_contracts import ToolResult
    @tool('web_search')
    def exploding(query: str) -> ToolResult:
        """Provider exception boundary."""
        raise RuntimeError('private token traceback details')
    monkeypatch.setattr(tools, 'AMAP_WEATHER_API', 'https://example.invalid')
    monkeypatch.setattr(tools, 'AMAP_API_KEY', 'test')
    monkeypatch.setattr(tools.requests, 'get', lambda *a, **kw: response({'status': '1', 'lives': [{'weather': '晴'}]}))
    async def run():
        model = ToolDecisionBoundaryModel(decisions=[
            AIMessageChunk(content='', tool_calls=[
                {'name': 'web_search', 'args': {'query': '问题'}, 'id': 'failed'},
                {'name': 'unknown', 'args': {}, 'id': 'unknown'},
                {'name': 'web_search', 'args': {}, 'id': 'invalid'},
                {'name': 'get_current_weather', 'args': {'location': '北京'}, 'id': 'success'},
            ], invalid_tool_calls=[{'name': 'web_search', 'args': '{not-json', 'id': 'malformed', 'error': 'private parse error'}]),
            AIMessageChunk(content='', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish'}]),
        ])
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        with telemetry.bind_context(trace_id='mixed-outcomes'), patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()), patch.object(agent, 'web_search', exploding):
            events = parse_sse([event async for event in agent.chat_with_agent_stream('问题')])
        messages = [m for m in model.final_inputs[0] if isinstance(m, ToolMessage)]
        assert {m.tool_call_id for m in messages} == {'failed', 'unknown', 'invalid', 'success', 'malformed', 'finish'}
        results = {m.tool_call_id: json.loads(m.content) for m in messages}
        assert results['failed']['code'] == 'UPSTREAM_ERROR'
        assert results['unknown']['code'] == 'UNKNOWN_TOOL'
        assert results['invalid']['code'] == results['malformed']['code'] == 'INVALID_ARGUMENT'
        assert results['success']['status'] == 'ok'
        assert 'private' not in json.dumps(results)
        assert not any(e.get('type') == 'error' for e in events if isinstance(e, dict))
    asyncio.run(run())
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('mixed-outcomes')
    assert sorted(s['status'] for s in detail['spans'] if s['stage'] == 'tool') == ['completed', 'failed', 'failed', 'failed', 'failed']
    assert 'private' not in json.dumps(detail)


def test_model_gets_evidence_text_once_and_full_tool_artifact(monkeypatch):
    text = '审批须由负责人审核。'
    docs = [{'filename': 'rules.pdf', 'page_number': 2, 'text': text}]
    monkeypatch.setitem(sys.modules, 'backend.rag_pipeline', SimpleNamespace(run_rag_graph=lambda query: {'docs': docs, 'rag_trace': {}}))
    async def run():
        tools.reset_tool_call_guards()
        model = ToolDecisionBoundaryModel(decisions=[
            AIMessageChunk(content='', tool_calls=[{'name': 'search_knowledge_base', 'args': {'query': '审批'}, 'id': 'kb'}]),
            AIMessageChunk(content='', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish'}]),
        ])
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        result = await graph.ainvoke({'messages': [HumanMessage(content='审批')], 'step_count': 0, 'memory_context': '', 'tool_output': None})
        output = next(m for m in model.final_inputs[0] if isinstance(m, ToolMessage) and m.tool_call_id == 'kb')
        assert output.content.count(text) == 1
        assert json.loads(output.content)['status'] == 'ok'
        assert output.artifact['evidence'] == docs
        assert next(m for m in result['messages'] if isinstance(m, ToolMessage) and m.tool_call_id == 'kb').artifact['evidence'] == docs
    asyncio.run(run())


def test_sync_finish_pairs_control_before_final_without_external_tool(diagnostics):
    class Model:
        calls = 0
        final_input = None
        def bind_tools(self, tools, **kwargs):
            return self
        def stream(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(content='private decision', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish-sync'}])
            else:
                self.final_input = messages
                yield AIMessageChunk(content='答案')
    model = Model()
    with patch.object(agent, 'init_chat_model', return_value=model):
        graph, _ = agent._create_react_agent()
    with telemetry.trace('chat', 'answer', trace_id='sync-finish'):
        result = graph.invoke({'messages': [HumanMessage(content='你好')], 'step_count': 0, 'memory_context': '', 'tool_output': None})
    assert result['messages'][-1].content == '答案'
    assert [m.tool_call_id for m in model.final_input if isinstance(m, ToolMessage)] == ['finish-sync']
    assert all(m.content != 'private decision' for m in model.final_input)
    diagnostics.flush()
    assert not any(s['stage'] == 'tool' for s in diagnostics.get_store().trace_detail('sync-finish')['spans'])


def test_provider_call_without_id_receives_matching_normalized_pair():
    async def run():
        model = ToolDecisionBoundaryModel(decisions=[
            AIMessageChunk(content='', tool_calls=[{'name': 'unknown', 'args': {}, 'id': None}]),
            AIMessageChunk(content='', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish'}]),
        ])
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        await graph.ainvoke({'messages': [HumanMessage(content='你好')], 'step_count': 0, 'memory_context': '', 'tool_output': None})
        call = next(call for message in model.final_inputs[0] for call in getattr(message, 'tool_calls', []) if call['name'] == 'unknown')
        result = next(message for message in model.final_inputs[0] if isinstance(message, ToolMessage) and json.loads(message.content).get('code') == 'UNKNOWN_TOOL')
        assert call['id']
        assert result.tool_call_id == call['id']
    asyncio.run(run())


@pytest.mark.parametrize('status,code,retryable,evidence_status', [
    ('ok', None, False, 'unknown'),
    ('empty', None, False, 'insufficient'),
    ('error', 'UPSTREAM_TIMEOUT', True, None),
    ('skipped', 'CALL_LIMIT_REACHED', False, None),
])
def test_long_contract_keeps_metadata_at_actual_model_and_provider_boundaries(monkeypatch, status, code, retryable, evidence_status):
    from backend.tool_contracts import ToolResult
    from backend.conversation_context import ContextBudget
    from langchain_core.tools import tool
    from langchain_openai.chat_models.base import _convert_message_to_dict
    from pydantic import Field
    body = '证据\n"\\' * 5000
    @tool('web_search')
    def long_tool(query: str) -> ToolResult:
        """Deterministic long provider result."""
        return ToolResult(status, body, code=code, retryable=retryable, evidence=[{'text': body}], evidence_status=evidence_status)
    class BoundaryModel(ToolDecisionBoundaryModel):
        inputs: list = Field(default_factory=list)
        async def _astream(self, messages, **kwargs):
            self.inputs.append(list(messages))
            async for chunk in super()._astream(messages, **kwargs):
                yield chunk
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '8192')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '2048')
    monkeypatch.setattr(tools, 'TAVILY_API_KEY', 'offline-only')
    async def run():
        model = BoundaryModel(decisions=[AIMessageChunk(content='', tool_calls=[{'name': 'web_search', 'args': {'query': 'query'}, 'id': 'long'}]),
                                        AIMessageChunk(content='', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish'}])])
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        with patch.object(agent, 'web_search', long_tool):
            await graph.ainvoke({'messages': [HumanMessage(content='current question')], 'step_count': 0, 'memory_context': '', 'tool_output': None})
        for messages in model.inputs[1:]:
            wire = [_convert_message_to_dict(message) for message in messages]
            output = next(item for item in wire if item.get('tool_call_id') == 'long')
            contract = json.loads(output['content'])
            assert (contract['status'], contract['code'], contract['retryable'], contract['evidence_status']) == (status, code, retryable, evidence_status)
            assert len(contract['content']) < len(body)
            assert ContextBudget.from_env().count_messages(messages) <= 6144
            assert next(item['content'] for item in wire if item['role'] == 'user') == 'current question'
            assert 'artifact' not in output and 'evidence' not in contract
            artifact = next(message.artifact for message in messages if isinstance(message, ToolMessage) and message.tool_call_id == 'long')
            assert artifact['content'] == body and artifact['evidence'] == [{'text': body}]
    asyncio.run(run())
