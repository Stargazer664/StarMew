from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.conversation_context import ContextBudget, fit_context, plan_compaction, finish_compaction


def budget(**kwargs):
    return ContextBudget(window=1000, output_reserve=200, history=300, summary=100, memory=100, **kwargs)


def test_long_messages_trigger_compaction_before_fifty_messages():
    history = [HumanMessage(content="历史" * 100), AIMessage(content="answer" * 100),
               HumanMessage(content="recent"), AIMessage(content="ok")]
    plan = plan_compaction(history, {}, budget())
    assert plan["end"] == 2
    state = finish_compaction(history, plan, "summary", budget())
    next_plan = plan_compaction(history, state, budget())
    assert next_plan["pending"] == []
    assert next_plan["summary"] == "summary"


def test_incremental_summary_never_reprocesses_already_summarized_prefix():
    old = [HumanMessage(content="old" * 150), AIMessage(content="old answer" * 50)]
    initial = plan_compaction(old, {}, budget())
    saved = finish_compaction(old, initial, "old summary", budget())
    new = old + [HumanMessage(content="new" * 200), AIMessage(content="new answer" * 40)]
    plan = plan_compaction(new, saved, budget())
    assert plan["pending"] == new[len(old):]
    assert plan["summary"] == "old summary"


def test_short_history_is_not_summarized_just_because_it_has_many_messages():
    history = [HumanMessage(content="x"), AIMessage(content="y")] * 26
    large = ContextBudget(window=10000, output_reserve=1000, history=8000, summary=100, memory=100)
    assert plan_compaction(history, {}, large)["pending"] == []


def test_changed_history_invalidates_summary_checkpoint():
    old = [HumanMessage(content="old" * 200)]
    saved = finish_compaction(old, plan_compaction(old, {}, budget()), "old summary", budget())
    plan = plan_compaction([HumanMessage(content="edited" * 200)], saved, budget())
    assert plan["start"] == 0
    assert plan["summary"] == ""


def test_full_context_budget_includes_system_and_tool_schema():
    b = budget()
    messages = [HumanMessage(content="old" * 100), AIMessage(content="answer" * 100), HumanMessage(content="current")]
    result = fit_context("system", messages, b, tool_tokens=150)
    assert b.count_messages(result) + 150 <= b.input_limit
    assert result[-1].content == "current"
    assert result[0].type == "system"


def test_oversized_tool_result_is_trimmed_without_breaking_call_response_pair():
    b = budget()
    messages = [HumanMessage(content="current"), AIMessage(content="", tool_calls=[
        {"name": "lookup", "args": {}, "id": "call-1"}]),
        ToolMessage(content="知识" * 2000, tool_call_id="call-1")]
    before = deepcopy(messages)
    result = fit_context("system", messages, b)
    assert b.count_messages(result) <= b.input_limit
    assert [m.type for m in result] == ["system", "human", "ai", "tool"]
    assert result[-1].tool_call_id == result[-2].tool_calls[0]["id"]
    assert messages == before


def test_current_question_is_never_silently_cut():
    with pytest.raises(ValueError, match="token"):
        fit_context("system", [HumanMessage(content="question" * 2000)], budget())


def test_recent_raw_history_is_evicted_before_losing_rolling_summary():
    b = budget()
    result = fit_context("system", [SystemMessage(content="important rolling summary"),
        HumanMessage(content="old" * 150), AIMessage(content="reply" * 50),
        HumanMessage(content="current")], b)
    assert any(m.content == "important rolling summary" for m in result)
    assert not any(m.content.startswith("oldold") for m in result)


@pytest.mark.parametrize('status,code,retryable,evidence_status', [
    ('ok', None, False, 'unknown'),
    ('empty', None, False, 'insufficient'),
    ('error', 'UPSTREAM_TIMEOUT', True, None),
    ('skipped', 'CALL_LIMIT_REACHED', False, None),
])
def test_budget_trims_contract_content_preserving_valid_provider_json(status, code, retryable, evidence_status):
    from backend.tool_contracts import ToolResult
    from langchain_openai.chat_models.base import _convert_message_to_dict
    b = budget()
    body = '知识\n"\\' * 2000
    contract = ToolResult(status, body, code=code, retryable=retryable, evidence_status=evidence_status)
    artifact = {'status': status, 'content': body, 'evidence': [{'text': body}]}
    messages = [HumanMessage(content='current'), AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {}, 'id': 'call'}]),
                ToolMessage(content=contract.to_json(include_evidence=False), artifact=artifact, tool_call_id='call')]
    before = deepcopy(messages)
    result = fit_context('system', messages, b)
    wire = [_convert_message_to_dict(message) for message in result]
    output = json.loads(wire[-1]['content'])
    assert (output['status'], output['code'], output['retryable'], output['evidence_status']) == (status, code, retryable, evidence_status)
    assert len(output['content']) < len(body)
    assert b.count_messages(result) <= b.input_limit
    assert result[-1].artifact == artifact
    assert 'artifact' not in wire[-1] and 'evidence' not in output
    assert wire[-1]['tool_call_id'] == wire[-2]['tool_calls'][0]['id']
    assert wire[1]['content'] == 'current'
    assert messages == before


def test_fixed_contract_metadata_too_large_raises_instead_of_corrupting_json():
    from backend.tool_contracts import ToolResult
    messages = [HumanMessage(content='current'), AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {}, 'id': 'call'}]),
                ToolMessage(content=ToolResult('error', 'x' * 1000, code='UPSTREAM_ERROR').to_json(include_evidence=False), tool_call_id='call')]
    tiny = ContextBudget(window=300, output_reserve=100, history=100, summary=50, memory=50)
    with pytest.raises(ValueError, match='token'):
        fit_context('system', messages, tiny)


def test_oversized_invalid_call_is_counted_and_rejected_before_provider_request():
    from langchain_openai.chat_models.base import _convert_message_to_dict
    b = ContextBudget()
    invalid = AIMessage(content='', invalid_tool_calls=[{'name': 'lookup', 'args': 'not-json' * 1250, 'id': 'bad', 'error': 'parse failed'}])
    wire = _convert_message_to_dict(invalid)
    sent_calls = json.dumps(wire['tool_calls'], ensure_ascii=False)
    assert b.count_message(invalid) >= b.count_text(sent_calls)
    messages = [HumanMessage(content='current'), invalid, ToolMessage(content='invalid arguments', tool_call_id='bad')]
    with pytest.raises(ValueError, match='token'):
        fit_context('system', messages, b)


def test_within_budget_invalid_call_keeps_original_arguments_and_matching_response():
    from langchain_openai.chat_models.base import _convert_message_to_dict
    invalid = AIMessage(content='', invalid_tool_calls=[{'name': 'lookup', 'args': 'not-json', 'id': 'bad', 'error': 'parse failed'}])
    result = fit_context('system', [HumanMessage(content='current'), invalid, ToolMessage(content='invalid arguments', tool_call_id='bad')], budget())
    wire = [_convert_message_to_dict(message) for message in result]
    assert wire[-2]['tool_calls'][0]['function']['arguments'] == 'not-json'
    assert wire[-1]['tool_call_id'] == wire[-2]['tool_calls'][0]['id'] == 'bad'
    assert budget().count_messages(result) <= budget().input_limit
