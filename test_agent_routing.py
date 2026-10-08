"""Real graph, provider configuration and persisted routing diagnostics boundaries."""
import asyncio
import hashlib
import os
from unittest.mock import patch

os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
import pytest
from pydantic import Field
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from test_agent_streaming import AsyncBoundaryModel, FakeStorage, FakeMemoryService, patched_stream_dependencies, collect_sse, parse_sse
from backend import agent, telemetry
from backend.agent_budget import AgentBudget, AgentRunState
from backend.llm_config import resolve_llm_settings, build_chat_model_kwargs


class RoutingModel(AsyncBoundaryModel):
    model_name: str = 'generation-test'
    mode: str = 'finish'
    inputs: list = Field(default_factory=list)
    bindings: list = Field(default_factory=list)
    timeouts: list = Field(default_factory=list)
    before_decision_chunk: object = None

    def bind_tools(self, tools, **kwargs):
        self.bindings.append([tool['function']['name'] if isinstance(tool, dict) else tool.name for tool in tools])
        return self.bind(route=True)

    def _stream(self, messages, **kwargs):
        self.inputs.append(list(messages))
        self.timeouts.append(kwargs.get('timeout'))
        route = kwargs.get('route')
        self.calls.append('decision' if route else 'final')
        if route and self.mode == 'failure':
            raise RuntimeError('offline provider failure')
        if route:
            if self.mode == 'text':
                response = AIMessageChunk(content='private router prose')
            else:
                name, args = {'tool': ('web_search', '{"query":"offline"}'), 'unknown': ('bad_tool', '{}'), 'invalid': ('web_search', '{broken'),
                              'bad_finish': ('finish', '{"answer":"secret"}'),
                              'bad_args': ('web_search', '{"wrong":"field"}')}.get(self.mode, ('finish', '{}'))
                response = AIMessageChunk(content='private router prose', tool_call_chunks=[{
                    'name': name, 'args': args, 'id': 'router-call', 'index': 0}])
        else:
            response = AIMessageChunk(content='公开答案')
        if route and self.before_decision_chunk:
            self.before_decision_chunk()
        yield ChatGenerationChunk(message=response)
        if route and self.mode == 'partial_failure':
            raise RuntimeError('stream failed after emitting a call')

    async def _astream(self, messages, **kwargs):
        for item in self._stream(messages, **kwargs):
            yield item


def make_graph(generation, router=None):
    with patch.object(agent, 'init_chat_model', return_value=generation), patch.object(agent, 'create_chat_model', return_value=router):
        # Router creation may be lazy; keep factory patched while the graph runs.
        return agent._create_react_agent()[0]


def execute(graph, asynchronous=True, **state):
    initial = dict(messages=[HumanMessage(content='你好')], step_count=0, memory_context='', tool_output=None)
    initial.update(state)
    return asyncio.run(graph.ainvoke(initial)) if asynchronous else graph.invoke(initial)


@pytest.mark.parametrize('provider', ['ollama', 'ark', 'aliyun'])
def test_router_inherits_selected_provider_credentials_and_preserves_other_roles(provider):
    env = dict(LLM_PROVIDER=provider, OLLAMA_MODEL='local-gen', OLLAMA_FAST_MODEL='local-fast', OLLAMA_JUDGE_MODEL='local-judge',
               OLLAMA_API_KEY='local-key', OLLAMA_BASE_URL='http://localhost:11434/v1/', MODEL='remote-gen',
               FAST_MODEL='remote-fast', GRADE_MODEL='remote-judge', BASE_URL='http://remote/v1/', ARK_API_KEY='remote-key', AGENT_ROUTER_MODEL='router-small')
    settings = resolve_llm_settings(role='router', environ=env)
    generation = resolve_llm_settings(role='generation', environ=env)
    assert (settings.model, settings.provider, settings.base_url, settings.api_key) == ('router-small', provider, generation.base_url, generation.api_key)
    assert resolve_llm_settings(role='fast', environ=env).model == ('local-fast' if provider == 'ollama' else 'remote-fast')
    assert resolve_llm_settings(role='judge', environ=env).model == ('local-judge' if provider == 'ollama' else 'remote-judge')
    assert build_chat_model_kwargs(role='router', environ=env)['max_retries'] == 0
    env['AGENT_ROUTER_MODEL'] = ''
    assert resolve_llm_settings(role='router', environ=env) == generation


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('enabled,expected', [('true', ['final']), ('false', ['decision', 'final'])])
def test_exact_greeting_only_skips_decision_when_enabled(asynchronous, enabled, expected):
    generation = RoutingModel()
    with patch.dict(os.environ, AGENT_DIRECT_CHAT_ENABLED=enabled, AGENT_ROUTER_MODEL=''):
        result = execute(make_graph(generation), asynchronous, context_state={'count': 0, 'digest': hashlib.sha256(b'').hexdigest(), 'summary': '', 'source_policy': 'unrestricted'})
    assert generation.calls == expected
    assert result['messages'][-1].content == '公开答案'
    assert result['run_state'].decision_count == (0 if enabled == 'true' else 1)


@pytest.mark.parametrize('state', [
    {'messages': [HumanMessage(content='你好，查询知识库')]},
    {'messages': [HumanMessage(content='北京今天的天气')]},
    {'messages': [HumanMessage(content='你好'), AIMessage(content='继续'), HumanMessage(content='你好')]},
    {'messages': [HumanMessage(content='你好\n只输出一句话')]},
    {'context_state': {'count': 2, 'summary': ''}},
    {'context_state': {'count': 0, 'summary': '待办：继续分析'}},
    {'context_state': {'pending': ['查文档']}},
    {'memory_context': '有待办任务'},
])
def test_greeting_shortcut_does_not_skip_effective_context_or_queries(state):
    generation = RoutingModel()
    with patch.dict(os.environ, AGENT_DIRECT_CHAT_ENABLED='true', AGENT_ROUTER_MODEL=''):
        execute(make_graph(generation), **state)
    assert generation.calls == ['decision', 'final']


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('mode', ['failure', 'invalid', 'unknown', 'bad_finish', 'bad_args'])
def test_light_router_fallback_is_once_and_pairs_rejected_calls(asynchronous, mode):
    generation, router = RoutingModel(), RoutingModel(model_name='router-small', mode=mode)
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small', AGENT_DIRECT_CHAT_ENABLED='false'), patch.object(agent, 'create_chat_model', return_value=router):
        result = execute(make_graph(generation, router), asynchronous)
    assert router.calls == ['decision']
    assert generation.calls == ['decision', 'final']
    assert result['run_state'].decision_count == 2
    assert result['messages'][-1].content == '公开答案'
    pairs = [m.tool_call_id for m in generation.inputs[0] if isinstance(m, ToolMessage)]
    assert pairs == ([] if mode == 'failure' else ['router-call'])
    assert all(m.content == '' for m in generation.inputs[0] if isinstance(m, AIMessageChunk))


def test_fallback_does_not_cross_decision_limit():
    generation, router = RoutingModel(), RoutingModel(mode='failure')
    run = AgentRunState(AgentBudget(max_decisions=1))
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', return_value=router):
        result = execute(make_graph(generation, router), run_state=run)
    assert generation.calls == ['final']
    assert result['run_state'].stop_reason == 'decision_limit'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_partial_router_failure_retains_pair_before_fallback(asynchronous):
    generation, router = RoutingModel(), RoutingModel(mode='partial_failure')
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', return_value=router):
        result = execute(make_graph(generation, router), asynchronous)
    assert generation.calls == ['decision', 'final']
    assert [m.tool_call_id for m in generation.inputs[0] if isinstance(m, ToolMessage)] == ['router-call']
    assert result['run_state'].decision_count == 2


def test_router_factory_failure_still_has_one_budgeted_generation_fallback():
    generation = RoutingModel()
    graph = make_graph(generation)
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', side_effect=RuntimeError('offline init failure')):
        result = execute(graph)
    assert generation.calls == ['decision', 'final']
    assert result['run_state'].decision_count == 2


@pytest.mark.parametrize('asynchronous', [False, True])
def test_call_received_at_decision_deadline_still_has_pair_in_final(asynchronous):
    now = [0.0]
    generation = RoutingModel()
    router = RoutingModel(before_decision_chunk=lambda: now.__setitem__(0, 105.0))
    run = AgentRunState(clock=lambda: now[0])
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', return_value=router):
        result = execute(make_graph(generation, router), asynchronous, run_state=run)
    assert generation.calls == ['final']
    assert [m.tool_call_id for m in generation.inputs[0] if isinstance(m, ToolMessage)] == ['router-call']
    assert result['run_state'].stop_reason == 'deadline'


def test_cached_router_has_fresh_remaining_timeout_for_each_request():
    generation, router = RoutingModel(), RoutingModel()
    now = [0.0]
    first, second = AgentRunState(clock=lambda: now[0]), AgentRunState(clock=lambda: now[0])
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', return_value=router):
        graph = make_graph(generation, router)
        execute(graph, run_state=first)
        now[0] = 30.0
        execute(graph, run_state=second)
    assert router.timeouts == [100.0, 70.0]
    assert generation.timeouts == [120.0, 90.0]
    assert len(router.bindings) == 1


def test_second_light_router_failure_cannot_trigger_second_fallback():
    from types import SimpleNamespace
    from backend.tool_contracts import ToolResult
    generation, router = RoutingModel(mode='tool'), RoutingModel(mode='failure')
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', return_value=router):
        graph = make_graph(generation, router)
        with patch.object(agent, 'web_search', SimpleNamespace(invoke=lambda args: ToolResult('empty', '无证据'))):
            result = execute(graph)
    assert generation.calls == ['decision', 'final']
    assert router.calls == ['decision', 'decision']
    assert result['run_state'].decision_count == 3
    assert result['run_state'].tool_count == 1


@pytest.mark.parametrize('saved_context,expected', [
    ({'count': 0, 'summary': '', 'digest': hashlib.sha256(b'').hexdigest()}, ['final']),
    ({'count': 0, 'summary': '待办：读完文档'}, ['decision', 'final']),
    ({'pending': ['后续查询']}, ['decision', 'final']),
])
def test_public_chat_greeting_honors_persisted_context_even_if_compaction_drops_it(saved_context, expected):
    generation = RoutingModel()
    storage = FakeStorage()
    storage.load_context_state = lambda *args, **kwargs: saved_context
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='', AGENT_DIRECT_CHAT_ENABLED='true'):
        with patched_stream_dependencies(make_graph(generation), storage, FakeMemoryService()):
            events = parse_sse(asyncio.run(collect_sse(agent.chat_with_agent_stream('你好'))))
    assert generation.calls == expected
    assert ''.join(e['content'] for e in events if isinstance(e, dict) and e['type'] == 'content') == '公开答案'
    assert len(storage.saved) == 1


def test_binding_cache_versions_tool_availability_policy_and_model_configuration():
    generation, router = RoutingModel(), RoutingModel(model_name='router-small')
    with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', return_value=router):
        graph = make_graph(generation, router)
        with patch.object(agent, 'tool_is_available', return_value=True):
            execute(graph)
            execute(graph)
            execute(graph, source_policy='internal_only')
            execute(graph, source_policy='external_allowed')
        with patch.object(agent, 'tool_is_available', side_effect=lambda name: name != 'get_current_weather'):
            execute(graph)
        with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-new'), patch.object(agent, 'tool_is_available', return_value=True):
            execute(graph)
    assert router.bindings == [
        ['get_current_weather', 'search_knowledge_base', 'web_search', 'finish'],
        ['search_knowledge_base', 'finish'],
        ['search_knowledge_base', 'web_search', 'finish'],
        ['get_current_weather', 'search_knowledge_base', 'web_search', 'finish']]


def test_router_generation_roles_persist_and_public_sse_is_final_only(tmp_path):
    generation, router = RoutingModel(), RoutingModel(model_name='router-small')
    telemetry.start_diagnostics(tmp_path / 'routing.db', tmp_path / 'logs')
    try:
        with patch.dict(os.environ, AGENT_ROUTER_MODEL='router-small'), patch.object(agent, 'create_chat_model', return_value=router):
            graph = make_graph(generation, router)
            with telemetry.bind_context(trace_id='routing-role'), patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()):
                events = parse_sse(asyncio.run(collect_sse(agent.chat_with_agent_stream('问题'))))
        assert ''.join(e['content'] for e in events if isinstance(e, dict) and e['type'] == 'content') == '公开答案'
        telemetry.flush()
        calls = [s for s in telemetry.get_store().trace_detail('routing-role')['spans'] if s.get('component') == 'model']
        assert [(s['role'], s['purpose'], s['model']) for s in calls] == [('router', 'tool_decision', 'router-small'), ('generation', 'final_answer', 'generation-test')]
    finally:
        telemetry.stop_diagnostics()
