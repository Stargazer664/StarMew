"""Retrieval policy through the real RAG graph and retrieval snapshot boundary."""
import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
from types import SimpleNamespace
import pytest
from backend import rag_pipeline as pipeline, rag_utils, tools
from backend.agent_budget import AgentRunState, use_run_state
from backend.chat_diagnostics import invoke_tool
from backend.schemas import RagTrace


def doc(name):
    return {'chunk_id': name, 'filename': name + '.pdf', 'page_number': 1, 'text': name}


@pytest.fixture
def boundary(monkeypatch):
    tools.reset_tool_call_guards()
    calls, grades = [], []
    monkeypatch.setattr(pipeline, '_get_router_model', lambda: None)
    monkeypatch.setattr(pipeline, 'step_back_expand', lambda q: {'expanded_query': q + ' expanded'})
    monkeypatch.setattr(rag_utils, '_finish_retrieval', lambda q, snapshot, k: snapshot)
    monkeypatch.setattr(rag_utils, 'assert_index_available', lambda: None)
    def configure(results, scores):
        queue = iter(results)
        def snapshot(q, k):
            calls.append(q)
            result = next(queue)
            if isinstance(result, Exception):
                raise result
            return {'docs': result, 'meta': {'retrieval_mode': 'hybrid'}}
        monkeypatch.setattr(rag_utils, '_retrieve_snapshot', snapshot)
        answers = iter(scores)
        class Grader:
            def with_structured_output(self, schema, **kwargs): return self
            def invoke(self, prompt, **kwargs):
                grades.append(prompt)
                return SimpleNamespace(binary_score=next(answers))
        monkeypatch.setattr(pipeline, '_get_grader_model', lambda: Grader())
    yield configure, calls, grades
    tools.reset_tool_call_guards()


def test_nonempty_no_grade_is_not_forced_and_expansion_is_regraded(boundary):
    configure, calls, grades = boundary
    configure([[doc('unrelated')], [doc('relevant')]], ['no', 'yes'])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert calls == ['q', 'q expanded']
    assert len(grades) == 2
    assert result.evidence_status == 'sufficient'
    assert [d['chunk_id'] for d in result.evidence] == ['relevant']
    trace = tools.get_last_rag_context()['rag_trace']
    assert trace['grade_score'] == 'yes'
    assert len(trace['retrieval_records']) == 2
    assert [r['score'] for r in trace['grading_records']] == ['no', 'yes']
    assert 'unrelated' not in result.content


def test_two_insufficient_queries_use_one_shared_actual_retrieval_cap(boundary):
    configure, calls, _ = boundary
    configure([[doc('same')]] * 4, ['no'] * 4)
    state = AgentRunState()
    with use_run_state(state):
        first = invoke_tool(tools.search_knowledge_base, {'query': 'q'}, purpose='knowledge_search')
        second = invoke_tool(tools.search_knowledge_base, {'query': 'other'}, purpose='knowledge_search')
        blocked = rag_utils.retrieve_documents('third')
    assert first.evidence_status == second.evidence_status == 'insufficient'
    assert state.retrieval_count == len(calls) == 4
    assert blocked['meta']['retrieval_status'] == 'budget_exhausted'
    assert '当前资料无法确认' in second.content
    trace = tools.get_last_rag_context()['rag_trace']
    assert len(trace['retrieval_records']) == 4
    assert len(trace['retrieval_rounds']) == 2
    assert len(trace['retrieved_chunks']) == 1
    assert len(RagTrace.model_validate(trace).retrieval_rounds) == 2


@pytest.mark.parametrize('failure,status,code', [(TimeoutError('secret'), 'infrastructure_failure', 'UPSTREAM_TIMEOUT'), (RuntimeError('secret'), 'infrastructure_failure', 'UPSTREAM_ERROR')])
def test_first_retrieval_failure_is_error_and_retains_trace(boundary, failure, status, code):
    configure, calls, grades = boundary
    configure([failure], [])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'error'
    assert result.code == code
    assert result.evidence_status == 'unknown'
    assert len(calls) == 1 and not grades
    trace = tools.get_last_rag_context()['rag_trace']
    assert trace['retrieval_status'] == status
    assert len(trace['retrieval_records']) == 1


def test_retrieval_failure_progress_does_not_claim_success(boundary, monkeypatch):
    configure, _, _ = boundary
    configure([TimeoutError('secret')], [])
    steps = []
    monkeypatch.setattr(pipeline, 'emit_rag_step', lambda icon, label, detail='': steps.append((icon, label)))
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'error'
    assert not any(icon == '✅' and '检索完成' in label for icon, label in steps)
    assert any(icon == '❌' for icon, label in steps)


def test_grader_unavailable_is_unknown_not_success(boundary, monkeypatch):
    configure, _, _ = boundary
    configure([[doc('a')], [doc('b')]], [])
    monkeypatch.setattr(pipeline, '_get_grader_model', lambda: None)
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'error'
    assert result.evidence_status == 'unknown'
    assert tools.get_last_rag_context()['rag_trace']['retrieval_status'] == 'grading_failed'


def test_zero_results_is_empty_after_expansion(boundary):
    configure, calls, grades = boundary
    configure([[], []], [])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'empty' and result.evidence_status == 'insufficient'
    assert len(calls) == 2 and not grades
    assert tools.get_last_rag_context()['rag_trace']['retrieval_status'] == 'zero_results'


def test_last_round_references_only_supplied_context_and_previous_trace_survives(boundary):
    configure, _, _ = boundary
    configure([[doc('a')], [doc('b')]], ['yes', 'yes'])
    state = AgentRunState()
    with use_run_state(state):
        first = invoke_tool(tools.search_knowledge_base, {'query': 'one'}, purpose='knowledge_search')
        second = invoke_tool(tools.search_knowledge_base, {'query': 'two'}, purpose='knowledge_search')
    assert first.evidence == [doc('a')] and second.evidence == [doc('b')]
    trace = tools.get_last_rag_context()['rag_trace']
    assert [d['chunk_id'] for d in trace['retrieved_chunks']] == ['a', 'b']


def test_complex_expansion_cannot_cross_remaining_actual_retrieval_budget(boundary, monkeypatch):
    configure, calls, _ = boundary
    configure([[doc('a')]] * 4, ['no'] * 4)
    class Router:
        def with_structured_output(self, schema, **kwargs): return self
        def invoke(self, *args, **kwargs): return SimpleNamespace(strategy='complex')
    monkeypatch.setattr(pipeline, '_get_router_model', lambda: Router())
    monkeypatch.setattr(pipeline, 'generate_hypothetical_document', lambda q: q + ' hyde')
    state = AgentRunState()
    with use_run_state(state):
        invoke_tool(tools.search_knowledge_base, {'query': 'one'}, purpose='knowledge_search')
        second = invoke_tool(tools.search_knowledge_base, {'query': 'two'}, purpose='knowledge_search')
    assert state.retrieval_count == len(calls) == 4
    assert second.evidence_status == 'insufficient'
    assert '预算' in second.content


def test_unconfigured_web_has_no_http_call(monkeypatch):
    monkeypatch.setattr(tools, 'TAVILY_API_KEY', None)
    monkeypatch.setattr(tools.requests, 'post', lambda *a, **k: pytest.fail('unexpected network'))
    assert not tools.tool_is_available('web_search')
    assert tools.web_search.invoke({'query': 'q'}).code == 'NOT_CONFIGURED'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_agent_enforces_internal_only_and_pairs_blocked_web(monkeypatch, asynchronous):
    import asyncio
    from backend import agent
    from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
    from test_agent_streaming import ToolDecisionBoundaryModel
    from langchain_core.outputs import ChatGenerationChunk
    class Model(ToolDecisionBoundaryModel):
        def _stream(self, messages, **kwargs):
            if kwargs.get('route'):
                yield ChatGenerationChunk(message=self.decisions[self.route_count])
                self.route_count += 1
            else:
                self.final_inputs.append(list(messages))
                yield ChatGenerationChunk(message=AIMessageChunk(content='当前资料无法确认'))
    model = Model(decisions=[AIMessageChunk(content='', tool_calls=[{'name': 'web_search', 'args': {'query': 'q'}, 'id': 'blocked'}]), AIMessageChunk(content='', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish'}])])
    monkeypatch.setattr(agent, 'init_chat_model', lambda **kwargs: model)
    monkeypatch.setattr(tools, 'TAVILY_API_KEY', 'offline-only')
    monkeypatch.setattr(tools.requests, 'post', lambda *a, **k: pytest.fail('unexpected network'))
    graph, _ = agent._create_react_agent()
    initial = dict(messages=[HumanMessage(content='只允许内部知识，不要联网')], step_count=0, memory_context='')
    result = asyncio.run(graph.ainvoke(initial)) if asynchronous else graph.invoke(initial)
    paired = [m for m in result['messages'] if isinstance(m, ToolMessage)]
    assert paired[0].tool_call_id == 'blocked'
    assert paired[0].artifact['status'] == 'skipped'
    assert result['run_state'].tool_count == 0
    assert '当前资料无法确认' in model.final_inputs[0][0].content


def test_failed_expansion_keeps_both_attempts_without_claiming_success(boundary):
    configure, _, _ = boundary
    configure([[doc('a')], TimeoutError('secret')], ['no'])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'error' and result.evidence_status == 'unknown'
    trace = tools.get_last_rag_context()['rag_trace']
    assert len(trace['retrieval_records']) == 2
    assert trace['initial_retrieved_chunks'] == [doc('a')]


@pytest.mark.parametrize('score', ['unavailable', 'yes but no evidence', ''])
def test_invalid_grader_output_is_unknown(boundary, score):
    configure, _, _ = boundary
    configure([[doc('a')], [doc('b')]], [score, score])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'error' and result.evidence_status == 'unknown'


def test_rewrite_that_returns_same_query_does_not_repeat_retrieval(boundary, monkeypatch):
    configure, calls, _ = boundary
    configure([[doc('a')]], ['no'])
    monkeypatch.setattr(pipeline, 'step_back_expand', lambda q: {'expanded_query': q})
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert len(calls) == 1
    assert result.evidence_status == 'insufficient'
    assert '重复' in result.content


def test_retrieval_budget_exhaustion_does_not_generate_another_rewrite(boundary, monkeypatch):
    configure, calls, _ = boundary
    configure([[doc('a')]], ['no'])
    state = AgentRunState()
    state.retrieval_count = 3
    monkeypatch.setattr(pipeline, 'step_back_expand', lambda q: pytest.fail('rewrite after budget exhausted'))
    with use_run_state(state):
        result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.evidence_status == 'insufficient'
    assert '预算' in result.content
    assert state.retrieval_count == 4 and len(calls) == 1


def test_real_chat_persists_aggregated_rounds_and_final_has_supplied_evidence(boundary, monkeypatch):
    import asyncio
    from backend import agent
    from langchain_core.messages import AIMessageChunk, ToolMessage
    from test_agent_streaming import ToolDecisionBoundaryModel, FakeStorage, FakeMemoryService, isolated_admission
    from unittest.mock import patch
    configure, calls, _ = boundary
    configure([[doc('irrelevant')], [doc('first')], [doc('second')]], ['no', 'no', 'yes'])
    model = ToolDecisionBoundaryModel(decisions=[AIMessageChunk(content='', tool_calls=[{'name': 'search_knowledge_base', 'args': {'query': q}, 'id': q}]) for q in ['one', 'two']])
    monkeypatch.setattr(agent, 'init_chat_model', lambda **kwargs: model)
    graph, _ = agent._create_react_agent()
    storage = FakeStorage()
    async def run():
        with isolated_admission(storage), patch.object(agent, 'react_graph', graph), patch.object(agent, 'storage', storage), patch.object(agent, 'memory_service', FakeMemoryService()):
            events = [event async for event in agent.chat_with_agent_stream('只允许内部知识')]
            from backend.models import ChatMessage
            with agent.SessionLocal() as db:
                saved = db.query(ChatMessage).filter(ChatMessage.message_type == 'ai').one()
                assert len(saved.rag_trace['retrieval_rounds']) == 2
            return events
    events = asyncio.run(run())
    assert events[-1] == 'data: [DONE]\n\n'
    assert len(calls) == 3
    assert model.route_count == 2  # sufficient evidence starts final directly
    trace = storage.saved[0][3][1]['rag_trace']
    assert len(trace['retrieval_rounds']) == 2 and len(trace['retrieval_records']) == 3
    assert [d['chunk_id'] for d in trace['retrieved_chunks']] == ['first', 'second']
    responses = [m for m in model.final_inputs[0] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in responses] == ['one', 'two']
    assert 'first.pdf' in responses[0].content and 'second.pdf' in responses[1].content
    assert 'irrelevant.pdf' not in ''.join(m.content for m in responses)
    import json
    public_trace = next(json.loads(event[6:])['rag_trace'] for event in events if event.startswith('data: {') and json.loads(event[6:]).get('type') == 'trace')
    assert 'one expanded' not in json.dumps(public_trace)
    assert 'query' not in public_trace['retrieval_records'][0]


def test_public_trace_recursively_hides_internal_queries():
    from backend.agent import _public_rag_trace
    trace = {'query': 'secret', 'retrieval_records': [{'query': 'secret-expanded', 'docs': [doc('safe')]}], 'retrieval_rounds': [{'query': 'secret', 'rewrite_query': 'secret-rewrite', 'retrieved_chunks': [doc('safe')]}]}
    public = _public_rag_trace(trace)
    assert 'secret' not in str(public)
    assert public['retrieval_records'][0]['docs'] == [doc('safe')]
    assert trace['query'] == 'secret'


def test_retrieval_timeout_at_turn_deadline_preserves_failed_attempt(boundary, monkeypatch):
    _, _, _ = boundary
    from backend.agent_budget import AgentBudget
    now = [0.0]
    state = AgentRunState(AgentBudget(), clock=lambda: now[0])
    def slow_snapshot(q, k):
        now[0] = 105
        raise TimeoutError('secret')
    monkeypatch.setattr(rag_utils, '_retrieve_snapshot', slow_snapshot)
    with use_run_state(state):
        result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'error' and result.code == 'UPSTREAM_TIMEOUT'
    trace = tools.get_last_rag_context()
    assert trace is not None
    assert trace['rag_trace']['retrieval_records'][0]['status'] == 'infrastructure_failure'
    assert state.retrieval_count == 1


@pytest.mark.parametrize('expanded_score', ['yes', 'no'])
def test_partial_complex_budget_still_grades_successful_branch(boundary, monkeypatch, expanded_score):
    configure, calls, grades = boundary
    configure([[doc('background')], [doc('confirmed')]], ['no', expanded_score])
    class Router:
        def with_structured_output(self, schema, **kwargs): return self
        def invoke(self, *args, **kwargs): return SimpleNamespace(strategy='complex')
    monkeypatch.setattr(pipeline, '_get_router_model', lambda: Router())
    monkeypatch.setattr(pipeline, 'generate_hypothetical_document', lambda q: q + ' hyde')
    state = AgentRunState()
    state.retrieval_count = 2
    with use_run_state(state):
        result = tools.search_knowledge_base.invoke({'query': 'second'})
    assert calls == ['second', 'second hyde'] and state.retrieval_count == 4
    assert len(grades) == 2
    assert result.evidence == [dict(doc('confirmed'), rrf_rank=1)]
    assert result.evidence_status == ('sufficient' if expanded_score == 'yes' else 'insufficient')
    trace = tools.get_last_rag_context()['rag_trace']
    assert [r['score'] for r in trace['grading_records']] == ['no', expanded_score]
    assert trace['retrieval_records'][-1]['status'] == 'budget_exhausted'
    if expanded_score == 'no':
        assert '预算' in result.content


@pytest.mark.parametrize('stage', ['step_back', 'hyde', 'router_init', 'router_call'])
@pytest.mark.parametrize('exception,code', [(TimeoutError('private timeout'), 'UPSTREAM_TIMEOUT'), (RuntimeError('private failure'), 'UPSTREAM_ERROR')])
def test_expansion_failure_retains_prior_and_current_round(boundary, monkeypatch, stage, exception, code):
    configure, calls, grades = boundary
    configure([[doc('previous')], [doc('background')]], ['yes', 'no'])
    def fail(*args, **kwargs): raise exception
    class Router:
        def with_structured_output(self, schema, **kwargs): return self
        def invoke(self, *args, **kwargs):
            if stage == 'router_call': raise exception
            return SimpleNamespace(strategy='hyde' if stage == 'hyde' else 'step_back')
    monkeypatch.setattr(pipeline, '_get_router_model', fail if stage == 'router_init' else lambda: Router())
    if stage == 'step_back': monkeypatch.setattr(pipeline, 'step_back_expand', fail)
    if stage == 'hyde': monkeypatch.setattr(pipeline, 'generate_hypothetical_document', fail)
    state = AgentRunState()
    with use_run_state(state):
        tools.search_knowledge_base.invoke({'query': 'previous'})
        result = tools.search_knowledge_base.invoke({'query': 'current'})
    assert result.status == 'error' and result.code == code and result.evidence_status == 'unknown'
    trace = tools.get_last_rag_context()['rag_trace']
    assert len(trace['retrieval_rounds']) == 2
    assert [r['query'] for r in trace['retrieval_records']] == ['previous', 'current']
    assert [r['score'] for r in trace['grading_records']] == ['yes', 'no']
    assert trace['retrieval_rounds'][-1]['expansion_status'] == 'failed'
    assert len(calls) == state.retrieval_count == 2 and len(grades) == 2


@pytest.mark.parametrize('stage', ['step_back_question', 'step_back_answer', 'hyde'])
@pytest.mark.parametrize('exception,code', [(TimeoutError('private timeout'), 'UPSTREAM_TIMEOUT'), (RuntimeError('private failure'), 'UPSTREAM_ERROR')])
def test_real_expansion_helpers_report_model_failure_without_retrieving_again(boundary, monkeypatch, stage, exception, code):
    configure, calls, grades = boundary
    configure([[doc('background')]], ['no'])
    class Router:
        def with_structured_output(self, schema, **kwargs): return self
        def invoke(self, *args, **kwargs): return SimpleNamespace(strategy='hyde' if stage == 'hyde' else 'step_back')
    model_calls = []
    class Model:
        def invoke(self, prompt, **kwargs):
            model_calls.append(prompt)
            if stage == 'step_back_answer' and len(model_calls) == 1:
                return SimpleNamespace(content='generated abstract question')
            raise exception
    monkeypatch.setattr(pipeline, '_get_router_model', lambda: Router())
    monkeypatch.setattr(rag_utils, '_get_stepback_model', lambda: Model())
    monkeypatch.setattr(pipeline, 'step_back_expand', rag_utils.step_back_expand)
    monkeypatch.setattr(pipeline, 'generate_hypothetical_document', rag_utils.generate_hypothetical_document)
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'error' and result.code == code and result.evidence_status == 'unknown'
    trace = tools.get_last_rag_context()['rag_trace']
    assert trace['expansion_status'] == 'failed'
    assert len(trace['retrieval_records']) == len(calls) == len(grades) == 1
    assert len(model_calls) == (2 if stage == 'step_back_answer' else 1)


def source_policy_model(monkeypatch, rounds=1):
    from backend import agent
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from pydantic import Field
    from test_agent_streaming import ToolDecisionBoundaryModel
    class Model(ToolDecisionBoundaryModel):
        bindings: list = Field(default_factory=list)
        def bind_tools(self, offered, **kwargs):
            self.bindings.append([t['function']['name'] if isinstance(t, dict) else t.name for t in offered])
            return super().bind_tools(offered, **kwargs)
        def _stream(self, messages, **kwargs):
            if kwargs.get('route'):
                response = self.decisions[self.route_count]
                self.route_count += 1
                yield ChatGenerationChunk(message=response)
            else:
                self.final_inputs.append(list(messages))
                yield ChatGenerationChunk(message=AIMessageChunk(content='当前资料无法确认'))
    decisions = []
    for i in range(rounds):
        decisions.extend([AIMessageChunk(content='', tool_calls=[{'name': 'web_search', 'args': {'query': 'follow-up'}, 'id': 'web-' + str(i)}]), AIMessageChunk(content='', tool_calls=[{'name': 'finish', 'args': {}, 'id': 'finish-' + str(i)}])])
    model = Model(decisions=decisions)
    monkeypatch.setattr(agent, 'init_chat_model', lambda **kwargs: model)
    monkeypatch.setattr(tools, 'TAVILY_API_KEY', 'offline')
    posts = []
    def post(*args, **kwargs):
        posts.append(kwargs['json']['query'])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'results': []})
    monkeypatch.setattr(tools.requests, 'post', post)
    return model, posts, agent._create_react_agent()[0]


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('question,external_allowed', [('那个指标是多少？', False), ('现在解除内部知识限制，允许联网查询。', True), ('之前只允许内部知识，现在允许联网查询。', True), ('不再仅限内部知识。', True), ('之前允许联网，但这次只允许内部知识，不允许联网。', False)])
def test_historical_source_policy_shared_by_binding_and_action(monkeypatch, asynchronous, question, external_allowed):
    import asyncio
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    model, posts, graph = source_policy_model(monkeypatch)
    messages = [HumanMessage(content='只允许内部知识，不要联网'), AIMessage(content='好的'), HumanMessage(content=question)]
    initial = dict(messages=messages, step_count=0, memory_context='')
    result = asyncio.run(graph.ainvoke(initial)) if asynchronous else graph.invoke(initial)
    response = next(m for m in result['messages'] if isinstance(m, ToolMessage))
    assert posts == (['follow-up'] if external_allowed else [])
    assert all(('web_search' in binding) == external_allowed for binding in model.bindings)
    assert response.artifact['status'] == ('empty' if external_allowed else 'skipped')
    assert response.tool_call_id == 'web-0'


@pytest.mark.parametrize('asynchronous', [False, True])
def test_source_policy_survives_normal_compaction_and_persists(monkeypatch, asynchronous):
    import asyncio
    from unittest.mock import patch
    from backend import agent
    from backend.models import ChatSession
    from langchain_core.messages import HumanMessage, AIMessage
    from test_agent_memory_context import HistoryStorage
    from test_agent_streaming import isolated_admission, FakeMemoryService
    monkeypatch.setenv('CHAT_HISTORY_MAX_TOKENS', '400')
    model, posts, graph = source_policy_model(monkeypatch, rounds=3)
    storage = HistoryStorage([HumanMessage(content='只允许内部知识，不要联网。' + 'OLD' * 500), AIMessage(content='好的')])
    summaries = []
    def summarize(*args, **kwargs):
        summaries.append(args)
        return SimpleNamespace(content='压缩摘要未包含来源限制')
    async def summarize_async(*args, **kwargs): return summarize(*args, **kwargs)
    async def stream(question): return [e async for e in agent.chat_with_agent_stream(question)]
    with isolated_admission(storage), patch.object(agent, 'react_graph', graph), patch.object(agent, 'storage', storage), patch.object(agent, 'memory_service', FakeMemoryService()), patch.object(agent, 'invoke_model', summarize), patch.object(agent, 'ainvoke_model', summarize_async):
        for question in ['那个指标是多少？', '那个趋势呢？']:
            asyncio.run(stream(question)) if asynchronous else agent.chat_with_agent(question)
        assert posts == []
        assert storage.context_state['source_policy'] == 'internal_only'
        assert storage.context_state['count'] == 2 and len(summaries) == 1
        assert all('web_search' not in binding for binding in model.bindings)
        assert all('只允许内部知识' not in str(m.content) for m in model.final_inputs[0] if isinstance(m, HumanMessage))
        with agent.SessionLocal() as db:
            assert db.query(ChatSession).one().metadata_json['context_state']['source_policy'] == 'internal_only'
        question = '解除内部知识限制，现在允许联网查询。'
        asyncio.run(stream(question)) if asynchronous else agent.chat_with_agent(question)
        assert posts == ['follow-up']
        assert storage.context_state['source_policy'] == 'external_allowed'


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('released', [False, True])
@pytest.mark.parametrize('separate_router', [False, True])
def test_internal_policy_blocks_weather_binding_and_forced_execution(monkeypatch, asynchronous, released, separate_router):
    import asyncio
    from backend import agent
    from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
    model, _, _ = source_policy_model(monkeypatch)
    monkeypatch.setenv('AGENT_ROUTER_MODEL', 'policy-router-test' if separate_router else '')
    monkeypatch.setattr(agent, 'create_chat_model', lambda **kwargs: model)
    model.decisions[0] = AIMessageChunk(content='', tool_calls=[{
        'name': 'get_current_weather', 'args': {'location': '110000'}, 'id': 'weather-forced'}])
    monkeypatch.setattr(tools, 'AMAP_WEATHER_API', 'https://weather.invalid')
    monkeypatch.setattr(tools, 'AMAP_API_KEY', 'test-only')
    calls = []
    def get(*args, **kwargs):
        calls.append(kwargs['params']['city'])
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'status': '1', 'lives': [{'city': '北京', 'weather': '晴'}]})
    monkeypatch.setattr(tools.requests, 'get', get)
    graph, _ = agent._create_react_agent()
    messages = [HumanMessage(content='只允许内部知识，不要联网')]
    if released:
        messages.append(HumanMessage(content='解除内部知识限制，现在允许联网查询。'))
    initial = dict(messages=messages, step_count=0, memory_context='')
    result = asyncio.run(graph.ainvoke(initial)) if asynchronous else graph.invoke(initial)
    paired = next(m for m in result['messages'] if isinstance(m, ToolMessage))
    assert calls == (['110000'] if released else [])
    assert all(('get_current_weather' in binding) == released for binding in model.bindings)
    assert paired.tool_call_id == 'weather-forced'
    import json
    assert json.loads(paired.content)['status'] == ('ok' if released else 'skipped')
    assert paired.artifact['status'] == ('ok' if released else 'skipped')
