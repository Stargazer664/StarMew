"""Acceptance harness checks; no assertions bless existing agent defects."""
import asyncio
from unittest.mock import patch
from langchain_core.messages import HumanMessage, ToolMessage


def test_sequential_final_generations_do_not_report_overlap():
    from scripts.agent_baseline_offline import ScriptedModel, _request
    async def run():
        model = ScriptedModel(case={'category': 'chat'}, gate=asyncio.Event())
        for prompt in ('first', 'second'):
            token = _request.set(prompt)
            try:
                async for _ in model._astream([HumanMessage(content=prompt)]):
                    pass
            finally:
                _request.reset(token)
        assert model.overlap is False
        assert model.entered == set()
    asyncio.run(run())


def test_simultaneous_final_generations_report_overlap_and_cleanup():
    from scripts.agent_baseline_offline import ScriptedModel, _request
    async def run():
        model = ScriptedModel(case={'category': 'chat'}, gate=asyncio.Event())
        async def consume(prompt):
            token = _request.set(prompt)
            try:
                async for _ in model._astream([HumanMessage(content=prompt)]):
                    pass
            finally:
                _request.reset(token)
        await asyncio.gather(consume('first'), consume('second'))
        assert model.overlap is True
        assert model.entered == set()
    asyncio.run(run())


def test_cancelled_final_generation_removes_active_request():
    from scripts.agent_baseline_offline import ScriptedModel, _request
    async def run():
        model = ScriptedModel(case={'category': 'chat'}, gate=asyncio.Event())
        token = _request.set('cancelled')
        try:
            stream = model._astream([HumanMessage(content='cancelled')])
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            assert len(model.entered) == 1
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
            await stream.aclose()
            assert model.entered == set()
        finally:
            _request.reset(token)
    asyncio.run(run())


def test_missing_tool_response_at_actual_final_boundary_fails_target_without_sse_error():
    from scripts.agent_baseline_offline import ScriptedModel, run_case, load_cases
    original = ScriptedModel._astream
    async def drop_response(self, messages, **kwargs):
        if not kwargs.get('route'):
            messages = [m for m in messages if not isinstance(m, ToolMessage)]
        async for chunk in original(self, messages, **kwargs):
            yield chunk
    case = next(c for c in load_cases() if c['category'] == 'tool_success')
    with patch.object(ScriptedModel, '_astream', drop_response):
        result = asyncio.run(run_case(case))
    request = result['requests'][0]
    assert request['errors'] == []
    assert request['matched_tool_protocol'] is False
    assert request['tool_protocol_status'] == 'failed'
    assert result['target_passed'] is False
    assert 'tool protocol mismatch at model input' in result['findings']


def test_no_final_with_pending_call_is_unverified_not_passed():
    from scripts.agent_baseline_offline import ScriptedModel, _request, tool_protocol_result
    async def controlled_decision_only():
        model = ScriptedModel(case={'category': 'tool_success'})
        token = _request.set('controlled pending call')
        try:
            # Intentionally stop at the external model boundary: no graph,
            # executed tool or subsequent model input is part of this fixture.
            chunks = [chunk async for chunk in model._astream(
                [HumanMessage(content='controlled pending call')], route=True)]
        finally:
            _request.reset(token)
        assert chunks[0].message.tool_calls[0]['id'] == 'call-1'
        assert model.final_inputs == {}
        return tool_protocol_result(model.observations, model.emitted_calls)
    result = asyncio.run(controlled_decision_only())
    assert result['matched_tool_protocol'] is None
    assert result['tool_protocol_status'] == 'unverified'
    assert result['tool_protocol_unobserved_call_ids'] == ['call-1']


def test_no_model_input_is_explicitly_unobserved():
    from scripts.agent_baseline_offline import tool_protocol_result
    result = tool_protocol_result([], [])
    assert result['matched_tool_protocol'] is None
    assert result['tool_protocol_status'] == 'unobserved'


def test_same_prompt_final_streams_are_still_distinct_active_requests():
    from scripts.agent_baseline_offline import ScriptedModel, _request
    async def run():
        model = ScriptedModel(case={'category': 'chat'}, gate=asyncio.Event())
        async def consume():
            token = _request.set('identical prompt')
            try:
                async for _ in model._astream([HumanMessage(content='identical prompt')]):
                    pass
            finally:
                _request.reset(token)
        await asyncio.gather(consume(), consume())
        assert model.overlap is True
        assert model.entered == set()
    asyncio.run(run())


def test_orphan_and_duplicate_tool_messages_are_detected_at_input():
    from scripts.agent_baseline_offline import inspect_tool_protocol
    from langchain_core.messages import AIMessage
    assert inspect_tool_protocol([ToolMessage(content='orphan', tool_call_id='x')])['unexpected_response_ids'] == ['x']
    result = inspect_tool_protocol([
        AIMessage(content='', tool_calls=[{'name': 'web_search', 'args': {}, 'id': 'x'}]),
        ToolMessage(content='one', tool_call_id='x'), ToolMessage(content='two', tool_call_id='x')])
    assert result['duplicate_response_ids'] == ['x']
    assert result['status'] == 'failed'


def test_manifest_has_three_cases_per_required_category():
    from scripts.agent_baseline_offline import load_cases
    cases = load_cases()
    categories = {c['category'] for c in cases}
    assert categories == {'chat', 'tool_success', 'tool_unavailable', 'tool_exception',
                          'tool_repeat', 'no_evidence', 'long_evidence', 'multi_turn', 'same_session_concurrent'}
    assert len(cases) == 27
    assert all(sum(c['category'] == category for c in cases) == 3 for category in categories)
    assert len({c['id'] for c in cases}) == 27
    assert all(c['expected'] for c in cases)


def test_real_graph_answer_and_sqlite_persistence_are_measured():
    from scripts.agent_baseline_offline import run_case, load_cases
    result = asyncio.run(run_case(load_cases()[0]))
    request = result['requests'][0]
    assert request['public_answer']
    assert request['done_count'] == 1
    assert request['model_calls'] == 2
    assert request['input_tokens'] is None and request['output_tokens'] is None
    assert request['first_public_answer_ms'] <= request['completion_ms']
    assert result['persisted_message_count'] == 2
    assert result['network_attempts'] == 0


def test_repeat_is_recorded_as_target_failure_without_blessing_recursion_error():
    from scripts.agent_baseline_offline import run_case, load_cases
    case = next(c for c in load_cases() if c['category'] == 'tool_repeat')
    result = asyncio.run(run_case(case))
    request = result['requests'][0]
    assert request['done_count'] == 1
    if request['errors']:
        assert result['target_passed'] is False
        assert any(f.startswith('SSE error:') for f in result['findings'])
    else:
        assert request['public_answer']


def test_source_retention_uses_final_model_input_and_trace_separately():
    from scripts.agent_baseline_offline import run_case, load_cases
    case = next(c for c in load_cases() if c['category'] == 'long_evidence')
    result = asyncio.run(run_case(case))
    request = result['requests'][0]
    assert request['trace_source_retention'] == 1
    assert 0 <= request['final_input_source_retention'] <= 1
    if request['final_input_source_retention'] < 1:
        assert result['target_passed'] is False
        assert 'source/page lost before final model' in result['findings']


def test_same_session_overlap_and_lost_turn_are_honestly_recorded():
    from scripts.agent_baseline_offline import run_case, load_cases
    case = next(c for c in load_cases() if c['category'] == 'same_session_concurrent')
    result = asyncio.run(run_case(case))
    assert len(result['requests']) == 2
    admitted = [r for r in result['requests'] if not r.get('admission_conflict')]
    if result['persisted_message_count'] < 2 * len(admitted):
        assert 'persisted message count differs from completed turns' in result['findings']
        assert result['target_passed'] is False
    if result['overlapping_final_nodes']:
        assert 'same session entered final nodes concurrently' in result['findings']
        assert result['target_passed'] is False


def test_long_first_turn_reuses_summary_checkpoint():
    from scripts.agent_baseline_offline import run_case, load_cases
    case = next(c for c in load_cases() if c['id'] == 'multi-2')
    result = asyncio.run(run_case(case))
    purposes = [r['model_calls_by_purpose'] for r in result['requests']]
    assert purposes[1].get('summary', 0) > 0
    assert purposes[2].get('summary', 0) == 0
    assert result['persisted_message_count'] == 6


def test_real_weather_and_web_failures_have_contract_and_failure_progress():
    from scripts.agent_baseline_offline import run_case, load_cases
    for category, code in [('tool_unavailable', 'NOT_CONFIGURED'), ('tool_exception', 'UPSTREAM_ERROR')]:
        case = next(c for c in load_cases() if c['category'] == category)
        result = asyncio.run(run_case(case))
        request = result['requests'][0]
        assert request['tool_results'][0]['status'] == 'error'
        assert request['tool_results'][0]['code'] == code
        assert request['matched_tool_protocol'] is True
        steps = [e['step'] for e in request['events'] if isinstance(e, dict) and e.get('type') == 'rag_step']
        assert any(s['icon'] == '❌' and '执行失败' in s['label'] for s in steps)
        assert not any(s['icon'] == '✅' and s['label'].startswith('工具 ') for s in steps)
        assert result['network_attempts'] == 0


def test_missing_finish_response_is_unverified_even_when_external_call_is_paired():
    from scripts.agent_baseline_offline import tool_protocol_result
    result = tool_protocol_result([{'purpose': 'final', 'input_tool_protocol': {
        'status': 'passed', 'matched_call_ids': ['external'], 'expected_call_ids': ['external'],
        'missing_response_ids': [], 'unexpected_response_ids': [], 'duplicate_response_ids': [], 'invalid_calls': []}}],
        [{'name': 'web_search', 'id': 'external'}, {'name': 'finish', 'id': 'control'}])
    assert result['tool_protocol_status'] == 'unverified'
    assert result['tool_protocol_unobserved_call_ids'] == ['control']
