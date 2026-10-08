import asyncio
import os
from unittest.mock import patch
from types import SimpleNamespace

os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
import pytest
from pydantic import Field
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from test_agent_streaming import AsyncBoundaryModel, FakeStorage, FakeMemoryService, patched_stream_dependencies, collect_sse, parse_sse
from backend import agent
from backend.tool_contracts import ToolResult, tool_error


class BudgetModel(AsyncBoundaryModel):
    def bind_tools(self, tools, **kwargs):
        return self.bind(route=True)

    different: bool = False
    received: list = Field(default_factory=list)

    def _stream(self, messages, **kwargs):
        self.received.append(kwargs)
        if kwargs.get('route'):
            self.calls.append('route')
            query = str(self.calls.count('route')) if self.different else ' same  query '
            yield ChatGenerationChunk(message=AIMessageChunk(content='', tool_call_chunks=[{
                'name': 'web_search', 'args': '{"query":' + __import__('json').dumps(query) + '}',
                'id': 'call-' + str(len(self.calls)), 'index': 0}]))
        else:
            self.calls.append('final')
            yield ChatGenerationChunk(message=AIMessageChunk(content='有限结果'))

    async def _astream(self, messages, **kwargs):
        for chunk in self._stream(messages, **kwargs):
            yield chunk


def graph_for(model):
    with patch.object(agent, 'init_chat_model', return_value=model):
        return agent._create_react_agent()[0]


@pytest.mark.parametrize('asynchronous', [False, True])
def test_duplicate_loop_finishes_with_paired_messages(asynchronous):
    model = BudgetModel()
    graph = graph_for(model)
    executed = []
    tool = SimpleNamespace(invoke=lambda args: executed.append(args) or ToolResult('empty', '无结果'))
    initial = dict(messages=[HumanMessage(content='q')], step_count=0, memory_context='', tool_output=None)
    with patch.object(agent, 'web_search', tool):
        result = asyncio.run(graph.ainvoke(initial, config={'recursion_limit': 32})) if asynchronous else graph.invoke(initial, config={'recursion_limit': 32})
    assert len(executed) == 1
    assert model.calls == ['route', 'route', 'final']
    assert result['messages'][-1].content == '有限结果'
    assert [m.tool_call_id for m in result['messages'] if isinstance(m, ToolMessage)] == ['call-1', 'call-2']
    assert result['run_state'].stop_reason == 'duplicate_call'


@pytest.mark.parametrize('retryable,expected', [(True, 2), (False, 1)])
def test_failure_retry_is_bounded_and_counted(retryable, expected):
    model = BudgetModel()
    graph = graph_for(model)
    executed = []
    tool = SimpleNamespace(invoke=lambda args: executed.append(args) or tool_error('UPSTREAM_TIMEOUT', retryable=retryable))
    with patch.object(agent, 'web_search', tool):
        result = asyncio.run(graph.ainvoke(dict(messages=[HumanMessage(content='q')], step_count=0, memory_context='', tool_output=None)))
    assert len(executed) == expected
    assert result['run_state'].tool_count == expected
    assert model.calls[-1] == 'final'


def test_total_tool_limit_finishes_before_recursion():
    model = BudgetModel(different=True)
    with patch.dict(os.environ, {'AGENT_MAX_TOOL_CALLS': '2'}):
        graph = graph_for(model)
        with patch.object(agent, 'web_search', SimpleNamespace(invoke=lambda args: ToolResult('ok', 'evidence'))):
            result = asyncio.run(graph.ainvoke(dict(messages=[HumanMessage(content='q')], step_count=0, memory_context='', tool_output=None)))
    assert model.calls == ['route', 'route', 'final']
    assert result['run_state'].stop_reason == 'tool_limit'


def test_sdk_timeout_and_no_hidden_retry():
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    from backend.llm_config import build_chat_model_kwargs
    from backend.chat_diagnostics import invoke_model
    now = [100.0]
    state = AgentRunState(AgentBudget(), clock=lambda: now[0])
    now[0] += 95
    model = SimpleNamespace(invoke=lambda inputs, **kwargs: SimpleNamespace(content=str(kwargs)))
    with use_run_state(state):
        assert "'timeout': 5.0" in invoke_model(model, [], purpose='grading').content
        assert "'timeout': 25.0" in invoke_model(model, [], purpose='final_answer').content
    assert build_chat_model_kwargs(role='generation')['max_retries'] == 0


def test_normalized_signature_and_distinct_kb_cap():
    from backend.agent_budget import AgentBudget, AgentRunState, call_signature, use_run_state
    from backend.chat_diagnostics import invoke_tool
    assert call_signature('x', {'b': 1, 'q': 'Ａ  B'}) == call_signature('x', {'q': 'A B', 'b': 1})
    run = AgentRunState(AgentBudget())
    tool = SimpleNamespace(name='search_knowledge_base', invoke=lambda args: ToolResult('empty', 'none'))
    with use_run_state(run):
        assert invoke_tool(tool, {'query': 'first'}, purpose='knowledge_search').status == 'empty'
        assert invoke_tool(tool, {'query': 'second'}, purpose='knowledge_search').status == 'empty'
        assert invoke_tool(tool, {'query': 'third'}, purpose='knowledge_search').code == 'CALL_LIMIT_REACHED'
    assert len(run.kb_signatures) == run.tool_count == 2


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('elapsed,expected_calls', [(105, ['final']), (121, [])])
def test_history_loading_consumes_deadline(asynchronous, elapsed, expected_calls):
    from backend.agent_budget import AgentBudget, AgentRunState
    from backend import chat_diagnostics
    now = [0.0]
    state = AgentRunState(AgentBudget(), clock=lambda: now[0])
    model = BudgetModel()
    graph = graph_for(model)
    storage = FakeStorage()
    def slow_load(*args, **kwargs):
        now[0] = elapsed
        return []
    storage.load = slow_load
    with patch.object(chat_diagnostics, 'AgentRunState', return_value=state), patched_stream_dependencies(graph, storage, FakeMemoryService()):
        if asynchronous:
            events = parse_sse(asyncio.run(collect_sse(agent.chat_with_agent_stream('q'))))
            assert events.count('[DONE]') == 1
            assert any(isinstance(e, dict) and e['type'] == 'error' for e in events) == (elapsed > 120)
        elif elapsed > 120:
            with pytest.raises(TimeoutError):
                agent.chat_with_agent('q')
        else:
            assert agent.chat_with_agent('q')['response'] == '有限结果'
    assert model.calls == expected_calls
    assert bool(storage.saved) == (elapsed < 120)
    assert state.stop_reason == 'deadline'


def test_final_timeout_has_one_done_and_no_save():
    from backend.agent_budget import AgentBudget, AgentRunState
    from backend import chat_diagnostics
    class HangingFinal(BudgetModel):
        async def _astream(self, messages, **kwargs):
            self.calls.append('final')
            self.received.append(kwargs)
            await asyncio.Event().wait()
            yield
    async def run():
        state = AgentRunState(AgentBudget(total_timeout_seconds=0.2, final_reserve_seconds=0.1))
        state.stop_reason = 'tool_limit'
        model = HangingFinal()
        storage = FakeStorage()
        with patch.object(chat_diagnostics, 'AgentRunState', return_value=state), patched_stream_dependencies(graph_for(model), storage, FakeMemoryService()):
            events = parse_sse(await collect_sse(agent.chat_with_agent_stream('q')))
        assert model.calls == ['final']
        assert 0 < model.received[0]['timeout'] <= 0.2
        assert events.count('[DONE]') == 1
        assert events[-2]['type'] == 'error'
        assert not storage.saved
        assert state.answer_status == 'failed'
        assert state.stop_reason == 'deadline'
    asyncio.run(run())


def test_tool_deadline_stops_subsequent_calls_and_preserves_pairs():
    from backend.agent_budget import AgentBudget, AgentRunState
    now = [0.0]
    state = AgentRunState(AgentBudget(), clock=lambda: now[0])
    from test_agent_streaming import ToolDecisionBoundaryModel
    model = ToolDecisionBoundaryModel(decisions=[AIMessageChunk(content='', tool_calls=[
        {'name': 'web_search', 'args': {'query': str(i)}, 'id': str(i)} for i in range(3)])])
    executed = []
    def slow_tool(args):
        executed.append(args)
        now[0] = 105
        return ToolResult('ok', 'evidence')
    graph = graph_for(model)
    with patch.object(agent, 'web_search', SimpleNamespace(invoke=slow_tool)):
        result = asyncio.run(graph.ainvoke(dict(messages=[HumanMessage(content='q')], run_state=state, step_count=0, memory_context='', tool_output=None)))
    assert len(executed) == 1
    assert state.stop_reason == 'deadline'
    assert state.answer_status == 'partial'
    assert [m.tool_call_id for m in result['messages'] if isinstance(m, ToolMessage)] == ['0', '1', '2']
    assert '没有证据时' in model.final_inputs[0][0].content


def test_decision_limit_and_programming_recursion_error_are_separate():
    from backend.agent_budget import AgentBudget, AgentRunState
    model = BudgetModel(different=True)
    graph = graph_for(model)
    with patch.object(agent, 'web_search', SimpleNamespace(invoke=lambda args: ToolResult('ok', 'data'))):
        state = AgentRunState(AgentBudget(max_decisions=1))
        result = asyncio.run(graph.ainvoke(dict(messages=[HumanMessage(content='q')], run_state=state, step_count=0, memory_context='', tool_output=None)))
    assert model.calls == ['route', 'final']
    assert result['run_state'].stop_reason == 'decision_limit'
    from langgraph.errors import GraphRecursionError
    class BrokenGraph:
        async def astream(self, *args, **kwargs):
            raise GraphRecursionError('program error')
            yield
    storage = FakeStorage()
    with patched_stream_dependencies(BrokenGraph(), storage, FakeMemoryService()):
        events = parse_sse(asyncio.run(collect_sse(agent.chat_with_agent_stream('q'))))
    assert events.count('[DONE]') == 1
    assert events[-2]['type'] == 'error'
    assert not storage.saved


def test_real_openai_sdk_receives_remaining_timeout_without_retries():
    import httpx
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    from backend.llm_config import create_chat_model
    from backend.chat_diagnostics import invoke_model
    seen = []
    def handle(request):
        seen.append(request.extensions['timeout'])
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0,
            'model': 'offline', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
    client = httpx.Client(transport=httpx.MockTransport(handle))
    model = create_chat_model(role='generation', http_client=client)
    now = [0.0]
    run = AgentRunState(AgentBudget(), clock=lambda: now[0])
    now[0] = 96
    with use_run_state(run):
        assert invoke_model(model, 'hello', purpose='grading').content == 'ok'
    assert len(seen) == 1
    assert seen[0] == dict(connect=4.0, read=4.0, write=4.0, pool=4.0)
    assert model.max_retries == 0
    client.close()


@pytest.mark.parametrize('tool_name,cap', [('get_current_weather', 10), ('web_search', 15)])
def test_tool_http_timeout_is_remaining_budget_and_respects_cap(monkeypatch, tool_name, cap):
    from backend import tools
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    from backend.chat_diagnostics import invoke_tool
    seen = []
    def send(*args, **kwargs):
        seen.append(kwargs['timeout'])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'status': '1', 'lives': [], 'results': []})
    monkeypatch.setattr(tools, 'AMAP_WEATHER_API', 'https://offline.invalid')
    monkeypatch.setattr(tools, 'AMAP_API_KEY', 'x')
    monkeypatch.setattr(tools, 'TAVILY_API_KEY', 'x')
    monkeypatch.setattr(tools.requests, 'get', send)
    monkeypatch.setattr(tools.requests, 'post', send)
    for elapsed in (0, 97):
        now = [0.0]
        state = AgentRunState(AgentBudget(), clock=lambda: now[0])
        now[0] = elapsed
        with use_run_state(state):
            invoke_tool(getattr(tools, tool_name), {'location': '北京'} if tool_name == 'get_current_weather' else {'query': 'q'}, purpose=tool_name)
    assert seen == [cap, 3.0]


def test_budget_diagnostics_are_persisted_without_signature_content(tmp_path):
    from backend import telemetry
    model = BudgetModel()
    storage = FakeStorage()
    graph = graph_for(model)
    telemetry.start_diagnostics(tmp_path / 'budget.db', tmp_path / 'logs')
    try:
        with telemetry.bind_context(trace_id='budget-turn'), patched_stream_dependencies(graph, storage, FakeMemoryService()), patch.object(agent, 'web_search', SimpleNamespace(invoke=lambda args: ToolResult('empty', 'private evidence'))):
            asyncio.run(collect_sse(agent.chat_with_agent_stream('private question')))
        telemetry.flush()
        detail = telemetry.get_store().trace_detail('budget-turn')
        budget = next(s for s in detail['spans'] if s['stage'] == 'agent_budget')
        assert {k: budget[k] for k in ('stop_reason', 'decision_count', 'tool_count', 'kb_call_count', 'answer_status')} == dict(stop_reason='duplicate_call', decision_count=2, tool_count=1, kb_call_count=0, answer_status='partial')
        assert 'private' not in __import__('json').dumps(detail)
        assert 'same  query' not in __import__('json').dumps(detail)
    finally:
        telemetry.stop_diagnostics()


def test_milvus_read_rpc_and_reconnect_receive_fresh_deadline(monkeypatch):
    from backend import milvus_client
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    now = [0.0]
    state = AgentRunState(AgentBudget(), clock=lambda: now[0])
    now[0] = 95
    seen = []
    class Client:
        def __init__(self, **kwargs):
            seen.append(('connect', kwargs.get('timeout')))
        def hybrid_search(self, **kwargs):
            seen.append(('hybrid', kwargs.get('timeout')))
            now[0] = 98
            raise RuntimeError('closed channel') if len([x for x in seen if x[0] == 'hybrid']) == 1 else AssertionError('second-hybrid')
        def search(self, **kwargs):
            seen.append(('dense', kwargs.get('timeout')))
            return []
        def close(self):
            pass
    monkeypatch.setattr(milvus_client, 'MilvusClient', Client)
    manager = milvus_client.MilvusManager()
    manager._bm25_validated = True
    with use_run_state(state):
        with pytest.raises(AssertionError, match='second-hybrid'):
            manager.hybrid_retrieve([1.0], 'query')
        manager.dense_retrieve([1.0])
    assert seen == [('connect', 5.0), ('hybrid', 5.0), ('connect', 2.0), ('hybrid', 2.0), ('dense', 2.0)]


def test_provider_sdk_decision_timeout_uses_final_reserve():
    from openai import APITimeoutError
    import httpx
    class TimeoutModel(BudgetModel):
        async def _astream(self, messages, **kwargs):
            if kwargs.get('route'):
                self.calls.append('route')
                raise APITimeoutError(request=httpx.Request('POST', 'http://offline.invalid'))
            for chunk in self._stream(messages, **kwargs):
                yield chunk
    model = TimeoutModel()
    result = asyncio.run(graph_for(model).ainvoke(dict(messages=[HumanMessage(content='q')], step_count=0, memory_context='', tool_output=None)))
    assert model.calls == ['route', 'final']
    assert result['run_state'].stop_reason == 'deadline'
    assert result['run_state'].answer_status == 'partial'


def test_nested_structured_grading_rewrite_and_rerank_receive_deadline(monkeypatch):
    from test_agent_streaming import local_rag_pipeline
    from backend.agent_budget import AgentBudget, AgentRunState, use_run_state
    from backend import rag_utils
    now = [0.0]
    state = AgentRunState(AgentBudget(), clock=lambda: now[0])
    now[0] = 97
    seen = []
    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self
        def invoke(self, inputs, **kwargs):
            seen.append(('model', kwargs.get('timeout')))
            return SimpleNamespace(binary_score='yes', strategy='step_back')
    with local_rag_pipeline() as pipeline, use_run_state(state), patch.object(pipeline, '_get_grader_model', return_value=Model()), patch.object(pipeline, '_get_router_model', return_value=Model()):
        pipeline.grade_documents_node(dict(question='q', context='c', docs=[{'text': 'c'}], rag_trace={}))
        pipeline.rewrite_question_node(dict(question='q', context='c', docs=[], rag_trace={}))
    def post(*args, **kwargs):
        seen.append(('rerank', kwargs.get('timeout')))
        return SimpleNamespace(status_code=200, json=lambda: {'results': [{'index': 0, 'relevance_score': 1}]})
    monkeypatch.setattr(rag_utils, 'RERANK_MODEL', 'rank')
    monkeypatch.setattr(rag_utils, 'RERANK_BINDING_HOST', 'https://offline.invalid')
    monkeypatch.setattr(rag_utils, 'RERANK_API_KEY', 'x')
    monkeypatch.setattr(rag_utils.requests, 'post', post)
    with use_run_state(state):
        result, _ = rag_utils._rerank_documents('q', [{'text': 'test'}], 1)
    assert len(result) == 1
    assert seen == [('model', 3.0), ('model', 3.0), ('rerank', 3.0)]


def test_timeout_during_summary_does_not_advance_checkpoint():
    from backend.agent_budget import AgentBudget, AgentRunState
    from backend import chat_diagnostics
    now = [0.0]
    state = AgentRunState(AgentBudget(), clock=lambda: now[0])
    storage = FakeStorage()
    storage.load = lambda *args, **kwargs: [HumanMessage(content='old ' * 1800)]
    async def summarize(inputs, **kwargs):
        assert kwargs['timeout'] == 100
        now[0] = 105
        return SimpleNamespace(content='not a finished summary')
    model = BudgetModel()
    with patch.object(chat_diagnostics, 'AgentRunState', return_value=state), patched_stream_dependencies(graph_for(model), storage, FakeMemoryService()), patch.object(agent, 'model', SimpleNamespace(ainvoke=summarize)):
        events = parse_sse(asyncio.run(collect_sse(agent.chat_with_agent_stream('q'))))
        assert events.count('[DONE]') == 1
        assert model.calls == ['final']
        with agent.SessionLocal() as db:
            from backend.models import ChatSession
            saved = db.query(ChatSession).one().metadata_json['context_state']
        assert saved['count'] == 0
        assert saved['summary'] == ''
