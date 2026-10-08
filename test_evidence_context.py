import asyncio
import json
import os
from dataclasses import replace
from types import SimpleNamespace

os.environ['DATABASE_URL'] = 'sqlite:///:memory:'

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from backend.conversation_context import ContextBudget


@pytest.fixture(params=['bytes', 'local'])
def budget(request, tmp_path):
    path = ''
    if request.param == 'local':
        # Real local tokenizer round trip through AutoTokenizer, without downloads.
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        from transformers import PreTrainedTokenizerFast
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        from tokenizers.trainers import BpeTrainer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(['中文 evidence source sentence. 标题 第1页。'],
            BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['[UNK]']))
        path = str(tmp_path / 'tokenizer')
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(path)
    return ContextBudget(window=2400, output_reserve=400, history=300, summary=150, memory=150, tokenizer_path=path)


def test_stable_citations_dedup_and_rank_ignore_raw_scores(budget):
    from backend.evidence_context import aggregate_evidence, select_evidence
    snippets = aggregate_evidence([[{'filename': '甲.pdf', 'page_number': 2, 'text': '第一事实。', 'score': .1},
        {'filename': '甲.pdf', 'page_number': 3, 'text': '第二事实。', 'score': 999}],
        [{'filename': '甲.pdf', 'page_number': 2, 'text': '第一事实。'},
         {'url': 'https://example.test', 'title': '网页', 'content': '补充事实。', 'score': 10000},
         {'text': '未知事实。'}]])
    selected = select_evidence(snippets, 1000, budget)
    assert [s.citation_id for s in snippets] == ['1', '2', '1', '3', '4']
    assert [s.citation_id for s in selected.snippets] == ['1', '3', '4', '2']
    assert selected.snippets[2].source == '未知来源'
    assert selected.omitted_count == 1


@pytest.mark.parametrize('heading', ['标题：设备指南。', '资料：设备指南。', 'Title: Equipment guide.', 'Document: Equipment guide.'])
def test_compressed_diversity_reserves_body_not_metadata(budget, heading):
    from backend.evidence_context import aggregate_evidence, select_evidence, render_evidence
    snippets = aggregate_evidence([[{'filename': 'a.pdf', 'text': heading + ' 保持时间为47分钟。' + '后续细节。' * 1000},
        {'filename': 'b.pdf', 'text': heading + ' 校验液体积为320毫升。' + '其他细节。' * 1000}]])
    bodies = [replace(snippets[0], text='保持时间为47分钟。'), replace(snippets[1], text='校验液体积为320毫升。')]
    limit = budget.count_text(render_evidence(bodies))
    selected = select_evidence(snippets, limit, budget)
    assert [s.citation_id for s in selected.snippets] == ['1', '2']
    assert [s.text for s in selected.snippets] == [s.text for s in bodies]
    assert selected.used_tokens <= limit and selected.omitted_count == 0


@pytest.mark.parametrize('text', ['版本3要求320毫升。', '版本：3.1。', 'Version: 3.1.', '操作说明规定47分钟。', 'The version 3 specification requires 320 ml.'])
def test_ordinary_body_words_are_not_metadata(budget, text):
    from backend.evidence_context import aggregate_evidence, select_evidence, render_evidence
    snippets = aggregate_evidence([[{'filename': 'a.pdf', 'text': text + ' ' + '后续内容。' * 1000}]])
    limit = budget.count_text(render_evidence([replace(snippets[0], text=text)]))
    assert select_evidence(snippets, limit, budget).snippets[0].text == text


def test_metadata_only_is_omitted_but_complete_source_keeps_original(budget):
    from backend.evidence_context import aggregate_evidence, select_evidence
    full = '标题：设备指南。版本：3，来源文件：a.pdf。操作说明规定47分钟。'
    snippets = aggregate_evidence([[{'filename': 'a.pdf', 'text': full},
        {'filename': 'b.pdf', 'text': '标题：设备指南。来源文件：b.pdf。'}]])
    result = select_evidence(snippets, 2000, budget)
    assert [s.text for s in result.snippets] == [full]
    assert result.omitted_count == 1


@pytest.mark.parametrize('text', ['Title: Header', '标题：设备指南', 'Title: Header\nSource file: a.pdf'])
def test_metadata_only_without_sentence_punctuation_is_omitted(budget, text):
    from backend.evidence_context import aggregate_evidence, select_evidence
    result = select_evidence(aggregate_evidence([[{'filename': 'a.pdf', 'text': text}]]), 2000, budget)
    assert result.snippets == () and result.omitted_count == 1


def test_heading_line_does_not_consume_following_body_sentence(budget):
    from backend.evidence_context import aggregate_evidence, select_evidence, render_evidence
    snippets = aggregate_evidence([[{'filename': 'a.pdf', 'text': 'Title: Header\n保持时间为47分钟。' + '后续内容。' * 1000}]])
    limit = budget.count_text(render_evidence([replace(snippets[0], text='保持时间为47分钟。')]))
    assert select_evidence(snippets, limit, budget).snippets[0].text == '保持时间为47分钟。'


@pytest.mark.parametrize('text', ['Title: Equipment guide. Hold time is 47 minutes', '标题：设备指南。保持时间为47分钟'])
def test_complete_snippet_keeps_unpunctuated_body_after_inline_heading(budget, text):
    from backend.evidence_context import aggregate_evidence, select_evidence, render_evidence
    snippets = aggregate_evidence([[{'filename': 'a.pdf', 'text': text}]])
    assert budget.count_text(render_evidence(snippets)) < 2000
    selection = select_evidence(snippets, 2000, budget)
    assert selection.snippets == snippets and selection.omitted_count == 0


def test_supplement_reserves_later_affordable_sentence_after_long_first(budget):
    from backend.evidence_context import aggregate_evidence, select_evidence, render_evidence
    snippets = aggregate_evidence([[{'filename': 'a.pdf', 'text': 'Primary fact. ' + 'More detail. ' * 100},
        {'filename': 'b.pdf', 'text': 'B' * 500 + '. Second fact.'}]])
    minimum = [replace(snippets[0], text='Primary fact.'), replace(snippets[1], text='Second fact.')]
    assert budget.count_text(render_evidence(minimum)) <= 130
    selection = select_evidence(snippets, 130, budget)
    assert [s.source for s in selection.snippets] == ['a.pdf', 'b.pdf']
    assert selection.snippets[0].text.startswith('Primary fact.')
    assert selection.snippets[1].text == 'Second fact.'
    assert selection.used_tokens <= 130 and selection.omitted_count == 0


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('route_mode', ['default', 'independent', 'fallback'])
def test_actual_sdk_router_envelope_is_inside_near_limit_budget(budget, monkeypatch, asynchronous, route_mode):
    import httpx
    from langchain_openai import ChatOpenAI
    from langchain_openai.chat_models.base import _convert_dict_to_message
    from backend import agent
    from backend.tool_contracts import ToolResult
    from dataclasses import asdict
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '4096')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '1024')
    monkeypatch.setenv('CHAT_TOKENIZER_PATH', budget.tokenizer_path)
    monkeypatch.setenv('AGENT_ROUTER_MODEL', '' if route_mode == 'default' else 'router-budget-test')
    monkeypatch.setenv('AGENT_DIRECT_CHAT_ENABLED', 'false')
    monkeypatch.setattr(agent, 'tool_is_available', lambda name: name == 'search_knowledge_base')
    bodies = []
    def transport(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get('tools'):
            name = 'not_a_tool' if route_mode == 'fallback' and body['model'] == 'router-budget-test' else 'finish'
            delta = {'tool_calls': [{'index': 0, 'id': 'sdk-' + str(len(bodies)), 'type': 'function',
                'function': {'name': name, 'arguments': '{}'}}]}
        else:
            delta = {'content': 'Answer [1].'}
        chunk = {'id': 'offline', 'object': 'chat.completion.chunk', 'created': 0, 'model': body['model'],
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
            content='data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n')
    sync = httpx.Client(transport=httpx.MockTransport(transport))
    asynchronous_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    def model(name):
        return ChatOpenAI(model=name, api_key='offline-only', base_url='https://sdk.invalid/v1',
            max_tokens=1024, http_client=sync, http_async_client=asynchronous_client)
    generation, router = model('generation-budget-test'), model('router-budget-test')
    monkeypatch.setattr(agent, 'init_chat_model', lambda **kwargs: generation)
    monkeypatch.setattr(agent, 'create_chat_model', lambda **kwargs: router)
    docs = [{'filename': 'a.pdf', 'page_number': 1, 'text': 'Body fact. ' * 3000}]
    contract = ToolResult('ok', '', evidence=docs, evidence_status='sufficient')
    initial = dict(messages=[HumanMessage(content='Compare the facts.'), AIMessage(content='', tool_calls=[{
        'name': 'search_knowledge_base', 'args': {'query': 'facts'}, 'id': 'existing'}]),
        ToolMessage(content=contract.to_json(include_evidence=False), artifact=asdict(contract), tool_call_id='existing')],
        step_count=0, memory_context='')
    graph, _ = agent._create_react_agent()
    try:
        async def run():
            try:
                return await graph.ainvoke(initial)
            finally:
                await asynchronous_client.aclose()
        result = asyncio.run(run()) if asynchronous else graph.invoke(initial)
        assert result['messages'][-1].content == 'Answer [1].'
        routed = [b for b in bodies if b.get('tools')]
        assert len(routed) == (2 if route_mode == 'fallback' else 1)
        costs = []
        for body in routed:
            counter = replace(budget, window=4096, output_reserve=1024,
                tokenizer_path='' if body['model'] == 'router-budget-test' else budget.tokenizer_path)
            messages = [_convert_dict_to_message(m) for m in body['messages']]
            envelope = {k: body[k] for k in ('tools', 'tool_choice', 'parallel_tool_calls', 'response_format') if k in body}
            cost = counter.count_messages(messages) + counter.count_text(json.dumps(envelope, ensure_ascii=False))
            costs.append(cost)
            assert body.get('tool_choice') == ('required' if asynchronous else None)
            assert body.get('parallel_tool_calls') == (False if asynchronous else None)
            assert cost <= counter.input_limit, (cost, counter.input_limit, envelope.keys())
        assert max(costs) >= 3040  # Actual wire payloads exercise a tight boundary.
    finally:
        sync.close()
        if not asynchronous:
            asyncio.run(asynchronous_client.aclose())


@pytest.mark.parametrize('operation', ['grade', 'rewrite'])
def test_actual_structured_sdk_payload_budget_and_output_cap(budget, monkeypatch, operation, record_property):
    import httpx
    from langchain_openai import ChatOpenAI
    from langchain_openai.chat_models.base import _convert_dict_to_message
    from backend import rag_pipeline as rag
    from backend.conversation_context import assert_model_input, ContextBudgetExceeded
    counter = replace(budget, window=4096, output_reserve=1024,
                      tokenizer_path='' if operation == 'grade' else budget.tokenizer_path)
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '4096')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '1024')
    monkeypatch.setenv('CHAT_TOKENIZER_PATH', budget.tokenizer_path)
    bodies = []
    answer = {'binary_score': 'yes'} if operation == 'grade' else {'strategy': 'hyde'}
    def transport(request):
        body = json.loads(request.content)
        bodies.append(body)
        message = {'role': 'assistant', 'content': json.dumps(answer)}
        if body.get('tools'):
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'structured', 'type': 'function',
                'function': {'name': body['tools'][0]['function']['name'], 'arguments': json.dumps(answer)}}]}
        return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 0,
            'model': body['model'], 'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        model = ChatOpenAI(model='offline-structured', api_key='offline', base_url='https://sdk.invalid/v1',
                           max_tokens=1800, http_client=client)
        monkeypatch.setattr(rag, '_get_grader_model', lambda: model)
        monkeypatch.setattr(rag, '_get_router_model', lambda: model)
        monkeypatch.setattr(rag, 'generate_hypothetical_document', lambda question: 'offline expansion')
        if operation == 'grade':
            result = rag.grade_documents_node({'question': 'What is the fact?', 'docs': [
                {'filename': 'fact.pdf', 'text': 'Valid fact. ' * 3000}], 'rag_trace': {}})
            assert result['rag_trace']['grade_score'] == 'yes'
        else:
            prefix = ('请根据用户问题选择最合适的查询扩展策略，仅输出策略名。\n'
                '- step_back：包含具体名称、日期、代码等细节，需要先理解通用概念的问题。\n'
                '- hyde：模糊、概念性、需要解释或定义的问题。\n'
                '- complex：多步骤、需要分解或综合多种信息的复杂问题。\n用户问题：')
            # Size the fixture at the public admission bound, then independently
            # verify what the real SDK sent rather than trusting that bound.
            n = 0
            for length in range(4096):
                try:
                    assert_model_input(prefix + 'Q' * length, schema=rag.RewriteStrategy, budget=counter)
                except ContextBudgetExceeded:
                    break
                n = length
            result = rag._rewrite_question_impl({'question': 'Q' * n, 'rag_trace': {}})
            assert result['expansion_type'] == 'hyde'
    assert len(bodies) == 1  # No swallowed parse failure/plain-text fallback.
    body = bodies[0]
    envelope = {key: body[key] for key in ('tools', 'tool_choice', 'parallel_tool_calls', 'response_format') if key in body}
    used = counter.count_messages([_convert_dict_to_message(m) for m in body['messages']]) + counter.count_text(json.dumps(envelope, ensure_ascii=False))
    record_property('structured_sdk_boundary', json.dumps({'operation': operation,
        'counter': 'local' if counter.tokenizer_path else 'utf8', 'input_estimate': used,
        'envelope': envelope, 'output_cap': body.get('max_completion_tokens', body.get('max_tokens'))}, ensure_ascii=False))
    assert 3025 <= used <= counter.input_limit, (used, counter.input_limit)
    assert body.get('max_completion_tokens', body.get('max_tokens')) == 1024


def test_long_sentence_table_url_omission_and_source_preservation(budget):
    from backend.evidence_context import aggregate_evidence, select_evidence, render_evidence
    snippets = aggregate_evidence([[{'filename': 'a.pdf', 'page_number': 1, 'text': '关键结论。\n' + '巨型表格|' * 4000},
        {'title': '网页', 'url': 'https://example.test/' + 'path' * 2000, 'content': 'valid sentence.'},
        {'filename': 'b.pdf', 'page_number': 7, 'text': 'English finding. 中文结论。' * 2000},
        {'filename': 'empty.pdf', 'text': ' '}]])
    selected = select_evidence(snippets, 400, budget)
    assert {s.source for s in selected.snippets} == {'a.pdf', 'b.pdf'}
    assert selected.snippets[0].text == '关键结论。'
    assert selected.snippets[1].text.endswith(('。', '.'))
    assert selected.omitted_count == 2
    assert budget.count_text(render_evidence(selected.snippets)) == selected.used_tokens <= 400


def test_final_allocation_keeps_json_pairs_and_only_selected_citations(budget):
    from backend.conversation_context import allocate_context
    from backend.tool_contracts import ToolResult
    from langchain_openai.chat_models.base import _convert_message_to_dict
    docs = [{'filename': 'a.pdf', 'page_number': 2, 'text': '重要结论。' * 1000},
            {'filename': 'b.pdf', 'page_number': 3, 'text': '补充结论。'},
            {'filename': 'omitted.pdf', 'text': 'table|' * 5000}]
    contract = ToolResult('ok', 'raw should disappear', evidence=docs, evidence_status='sufficient')
    from dataclasses import asdict
    messages = [HumanMessage(content='old' * 1000), AIMessage(content='history' * 1000),
        HumanMessage(content='完整当前问题'), AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {}, 'id': 'c'}]),
        ToolMessage(content=contract.to_json(include_evidence=False), artifact=asdict(contract), tool_call_id='c')]
    result, selection = allocate_context('system', messages, budget, memory='记忆' * 1000, tool_tokens=150)
    assert budget.count_messages(result) + 150 <= budget.input_limit
    assert next(m.content for m in result if m.type == 'human') == '完整当前问题'
    tool = result[-1]
    payload = json.loads(_convert_message_to_dict(tool)['content'])
    assert payload['status'] == 'ok' and payload['evidence_status'] == 'sufficient'
    assert 'raw should disappear' not in payload['content'] and 'omitted.pdf' not in payload['content']
    assert {s.source for s in selection.snippets} == {'a.pdf', 'b.pdf'}
    assert all(f'[{s.citation_id}] {s.source}' in payload['content'] and s.text in payload['content'] for s in selection.snippets)
    assert tool.tool_call_id == result[-2].tool_calls[0]['id']
    assert tool.artifact['evidence'] == docs and 'artifact' not in _convert_message_to_dict(tool)


@pytest.mark.parametrize('purpose,role', [('grading', 'judge'), ('rewrite_routing', 'generation'),
    ('step_back_question', 'generation'), ('step_back_answer', 'generation'), ('hyde', 'generation'),
    ('rerank', 'generation'), ('conversation_summary', 'generation'), ('memory_extraction', 'fast'), ('memory_summary', 'fast')])
def test_model_boundary_rejects_oversized_inputs_before_provider(purpose, role, monkeypatch):
    from backend.chat_diagnostics import invoke_model, ainvoke_model
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '1000')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '200')
    calls = []
    class Provider:
        def invoke(self, value, **kwargs):
            calls.append(value)
            return SimpleNamespace(content='yes')
        async def ainvoke(self, value, **kwargs):
            return self.invoke(value, **kwargs)
    with pytest.raises(ValueError, match='token'):
        invoke_model(Provider(), '超长' * 1000, purpose=purpose, role=role)
    with pytest.raises(ValueError, match='token'):
        asyncio.run(ainvoke_model(Provider(), [HumanMessage(content='超长' * 1000)], purpose=purpose, role=role))
    assert calls == []


def test_structured_grader_receives_bounded_evidence_and_schema(monkeypatch):
    from backend import rag_pipeline as rag
    from langchain_core.messages import convert_to_messages
    from langchain_core.utils.function_calling import convert_to_openai_tool
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '4096')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '1024')
    inputs = []
    class Grader:
        def with_structured_output(self, schema, **kwargs):
            return self
        def invoke(self, value, **kwargs):
            inputs.append(value)
            return SimpleNamespace(binary_score='yes')
    monkeypatch.setattr(rag, '_get_grader_model', lambda: Grader())
    docs = [{'filename': 'a.pdf', 'page_number': 1, 'text': '相关事实。' * 20000},
            {'filename': 'b.pdf', 'page_number': 2, 'text': '补充事实。'}]
    output = rag.grade_documents_node({'question': '事实是什么', 'docs': docs, 'context': str(docs), 'rag_trace': {}})
    b = ContextBudget.from_env()
    schema = b.count_text(json.dumps(convert_to_openai_tool(rag.GradeDocuments), ensure_ascii=False))
    assert inputs and all(b.count_messages(convert_to_messages(value)) + schema <= b.input_limit for value in inputs)
    assert 'a.pdf' in str(inputs) and 'b.pdf' in str(inputs)
    assert output['rag_trace']['grade_score'] == 'yes'


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('range_case', [False, True])
def test_actual_final_model_and_public_citations_share_retained_fragments(monkeypatch, streaming, range_case):
    from test_agent_streaming import agent, ToolDecisionBoundaryModel, patched_stream_dependencies, FakeStorage, FakeMemoryService, collect_sse, parse_sse
    from langchain_core.messages import AIMessageChunk
    from backend.tool_contracts import ToolResult
    from unittest.mock import patch
    from langchain_core.outputs import ChatGenerationChunk
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '4096')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '1024')
    monkeypatch.setenv('AGENT_DIRECT_CHAT_ENABLED', 'false')
    class CitationModel(ToolDecisionBoundaryModel):
        def _stream(self, messages, **kwargs):
            if kwargs.get('route'):
                yield ChatGenerationChunk(message=self.decisions[0])
            else:
                self.final_inputs.append(messages)
                chunks = (['事实[', '1', ']，补充[', '3]，范围[', '1-', '3', ']。末尾[99'] if range_case else
                          ['事实[', '1', ']，补充[', '2]，无依据[', '3', ']。末尾[99'])
                for text in chunks:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        async def _astream(self, messages, **kwargs):
            for chunk in self._stream(messages, **kwargs):
                yield chunk
    model = CitationModel(decisions=[AIMessageChunk(content='', tool_call_chunks=[
        {'name': 'search_knowledge_base', 'args': '{"query":"facts"}', 'id': 'c', 'index': 0}])])
    docs = [{'filename': 'a.pdf', 'page_number': 2, 'text': '重要事实。' * 2000},
            {'filename': 'b.pdf', 'page_number': 4, 'text': '补充事实。'},
            {'filename': 'omitted.pdf', 'page_number': 9, 'text': 'table|' * 5000}]
    if range_case:
        docs = [docs[0], docs[2], docs[1]]
    class Tool:
        def invoke(self, args):
            return ToolResult('ok', str(docs), evidence=docs, evidence_status='sufficient')
    with patch.object(agent, 'init_chat_model', return_value=model):
        graph, _ = agent._create_react_agent()
    storage = FakeStorage()
    async def run():
        from backend import tools
        with patched_stream_dependencies(graph, storage, FakeMemoryService()), patch.object(agent, 'search_knowledge_base', Tool()), \
             patch.object(agent, 'get_last_rag_context', tools.get_last_rag_context), \
             patch.object(agent, 'reset_tool_call_guards', tools.reset_tool_call_guards):
            if streaming:
                events = parse_sse(await collect_sse(agent.chat_with_agent_stream('资料中的事实是什么', 'xin', 't08')))
                text = ''.join(e['content'] for e in events if isinstance(e, dict) and e.get('type') == 'content')
                trace = next(e['rag_trace'] for e in events if isinstance(e, dict) and e.get('type') == 'trace')
                forbidden = ('1-', '[1-', '[1-3]', '99', '[99') if range_case else ('3', '[3]', '99', '[99')
                assert all(e.get('content') not in forbidden for e in events if isinstance(e, dict))
                return text, trace
            result = agent.chat_with_agent('资料中的事实是什么', 'xin', 't08')
            return result['response'], result['rag_trace']
    response, trace = asyncio.run(run())
    assert model.final_inputs
    final = model.final_inputs[-1]
    b = ContextBudget.from_env()
    assert b.count_messages(final) <= b.input_limit
    assert [s['source'] for s in trace['answer_evidence']] == ['a.pdf', 'b.pdf']
    assert trace['evidence_omitted_count'] == 1
    body = '\n'.join(json.loads(m.content)['content'] for m in final if m.type == 'tool')
    assert all(s['text'] in body and f"[{s['citation_id']}] {s['source']}" in body for s in trace['answer_evidence'])
    assert 'omitted.pdf' not in body
    assert response == ('事实[1]，补充[3]，范围。末尾' if range_case else '事实[1]，补充[2]，无依据。末尾')
    assert storage.saved[-1][2][-1].content == response
    assert storage.saved[-1][3][-1]['rag_trace']['answer_evidence'] == trace['answer_evidence']
    if streaming:
        assert trace['retrieved_chunks'][0]['page_number'] == 2


def test_impossible_router_fixed_budget_is_not_swallowed_as_fallback(monkeypatch):
    from test_agent_streaming import agent
    from unittest.mock import patch
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '4096')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '2048')
    monkeypatch.setenv('AGENT_DIRECT_CHAT_ENABLED', 'false')
    monkeypatch.setattr(agent, 'tool_is_available', lambda name: True)
    calls = []
    class Model:
        def bind_tools(self, tools, **kwargs): return self
        async def astream(self, messages, **kwargs):
            calls.append(messages)
            yield AIMessage(content='not reachable')
    with patch.object(agent, 'init_chat_model', return_value=Model()):
        graph, _ = agent._create_react_agent()
    with pytest.raises(ValueError, match='token'):
        asyncio.run(graph.ainvoke({'messages': [HumanMessage(content='你好')], 'step_count': 0, 'memory_context': ''}))
    assert calls == []


def test_rerank_uses_bounded_optional_material(monkeypatch):
    from backend import rag_utils, memory
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '1600')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '400')
    inputs = []
    class Model:
        def invoke(self, prompt, **kwargs):
            inputs.append(prompt)
            assert ContextBudget.from_env().count_text(prompt) + 24 <= 1200
            return SimpleNamespace(content='[{"doc_id":1,"score":9}]')
    monkeypatch.setattr(rag_utils, '_get_stepback_model', lambda: Model())
    docs = [{'filename': f'{i}.pdf', 'text': '相关结论。' * 1000} for i in range(9)]
    result, meta = rag_utils._llm_self_rerank('结论', docs, 3, {})
    assert meta.get('rerank_applied') is True


def test_memory_summary_uses_bounded_optional_material(monkeypatch):
    from backend import memory
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '1600')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '400')
    inputs = []
    class Model:
        def invoke(self, prompt, **kwargs):
            inputs.append(prompt)
            assert ContextBudget.from_env().count_text(prompt) + 24 <= 1200
            return SimpleNamespace(content='摘要')
    service = memory.UserMemoryService()
    monkeypatch.setattr(service, '_get_extract_llm', lambda: Model())
    service.summarize_session([{'type': 'human', 'content': '历史。' * 1000}] * 10)
    assert len(inputs) == 1


def test_promptvalue_and_schema_fixed_overflow_are_checked(monkeypatch):
    from backend.chat_diagnostics import invoke_model
    from langchain_core.prompt_values import ChatPromptValue
    from backend.rag_pipeline import RewriteStrategy
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '500')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '100')
    class Model:
        def invoke(self, *args, **kwargs): raise AssertionError('provider must not be called')
    with pytest.raises(ValueError, match='token'):
        invoke_model(Model(), ChatPromptValue(messages=[HumanMessage(content='x' * 200)]),
                     purpose='rewrite_routing', schema=RewriteStrategy)


def test_citations_split_across_provider_chunks_never_leak_omitted_ids():
    from backend.evidence_context import CitationFilter
    stream = CitationFilter({'1', '3'})
    assert stream.feed('事实[') == '事实'
    assert stream.feed('2') == ''
    assert stream.feed(']，补充[') == '，补充'
    assert stream.feed('3]。') == '[3]。'
    assert stream.feed('[说明](https://example.test)') == '[说明](https://example.test)'
    assert stream.feed('[1, 99]') == ''
    assert stream.feed('[999') == ''
    assert stream.finish() == ''


@pytest.mark.parametrize('configured, reserve, expected', [(None, 400, 400), (512, 400, 400), (256, 400, 256), (512, 1024, 512)])
def test_auxiliary_provider_output_cap_matches_reserved_budget(monkeypatch, configured, reserve, expected):
    from backend.chat_diagnostics import invoke_model
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '1600')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', str(reserve))
    observed = []
    class Model:
        max_tokens = configured
        def invoke(self, prompt, **kwargs):
            observed.append(kwargs)
            return SimpleNamespace(content='ok')
    invoke_model(Model(), 'small prompt', purpose='memory_summary', role='fast')
    assert observed[0]['max_tokens'] == expected


def test_judge_fast_router_never_reuse_generation_tokenizer(budget, monkeypatch):
    if not budget.tokenizer_path:
        return
    from backend.conversation_context import model_budget
    monkeypatch.setenv('CHAT_TOKENIZER_PATH', budget.tokenizer_path)
    assert model_budget('generation').tokenizer_path == budget.tokenizer_path
    assert all(model_budget(role).tokenizer_path == '' for role in ('judge', 'fast', 'router'))


def test_independent_router_enforces_bytes_with_generation_local_tokenizer(budget, monkeypatch):
    if not budget.tokenizer_path:
        return
    from test_agent_streaming import agent
    monkeypatch.setenv('CHAT_TOKENIZER_PATH', budget.tokenizer_path)
    monkeypatch.setenv('CHAT_CONTEXT_WINDOW_TOKENS', '4096')
    monkeypatch.setenv('CHAT_OUTPUT_RESERVE_TOKENS', '2048')
    monkeypatch.setenv('AGENT_ROUTER_MODEL', 'independent-router')
    monkeypatch.setenv('AGENT_DIRECT_CHAT_ENABLED', 'false')
    monkeypatch.setattr(agent, 'tool_is_available', lambda name: True)
    calls = []
    class Model:
        def bind_tools(self, *args, **kwargs): return self
        async def astream(self, messages, **kwargs):
            calls.append(messages)
            yield AIMessage(content='done')
    monkeypatch.setattr(agent, 'init_chat_model', lambda **kwargs: Model())
    monkeypatch.setattr(agent, 'create_chat_model', lambda **kwargs: Model())
    graph, _ = agent._create_react_agent()
    with pytest.raises(ValueError, match='token'):
        asyncio.run(graph.ainvoke({'messages': [HumanMessage(content='你好')], 'step_count': 0, 'memory_context': ''}))
    assert calls == []


def test_all_omitted_evidence_is_explicit_even_when_retrieval_was_sufficient():
    from backend.conversation_context import allocate_context
    from backend.tool_contracts import ToolResult
    from dataclasses import asdict
    result = ToolResult('ok', 'old body', evidence=[{'filename': 'table.pdf', 'text': 'cell|' * 10000}], evidence_status='sufficient')
    messages = [HumanMessage(content='question'), AIMessage(content='', tool_calls=[{'name': 'lookup', 'args': {}, 'id': 'c'}]),
        ToolMessage(content=result.to_json(include_evidence=False), artifact=asdict(result), tool_call_id='c')]
    fitted, selection = allocate_context('system', messages, ContextBudget(window=1000, output_reserve=200))
    payload = json.loads(fitted[-1].content)
    assert not selection.snippets and selection.omitted_count == 1
    assert payload['evidence_status'] == 'sufficient'
    assert '未保留有效证据' in payload['content']


def test_rerank_dedup_keeps_original_candidate_indices(monkeypatch):
    from backend import rag_utils
    class Model:
        def invoke(self, prompt, **kwargs):
            return SimpleNamespace(content='[{"doc_id":3,"score":9}]')
    monkeypatch.setattr(rag_utils, '_get_stepback_model', lambda: Model())
    docs = [{'filename': 'same.pdf', 'text': 'same evidence.'}] * 2 + [{'filename': 'best.pdf', 'text': 'best evidence.'}]
    result, meta = rag_utils._llm_self_rerank('question', docs, 1, {})
    assert result[0]['filename'] == 'best.pdf'
    assert meta['rerank_applied']


def test_multiple_summary_messages_share_one_component_budget():
    from backend.conversation_context import allocate_context
    from langchain_core.messages import SystemMessage
    b = ContextBudget(window=2000, output_reserve=200, summary=80, memory=80, history=300)
    messages = [SystemMessage(content='旧摘要' * 50), SystemMessage(content='补充摘要' * 50), HumanMessage(content='当前问题')]
    result, _ = allocate_context('fixed system', messages, b)
    assert sum(b.count_text(m.content) for m in result[1:] if m.type == 'system') <= 80


@pytest.mark.parametrize('primary_text', ['Primary fact.', 'Primary fact. ' + 'overflow' * 1000], ids=['complete', 'first-sentence'])
def test_primary_evidence_survives_tight_budget_before_diversity(budget, primary_text):
    from backend.evidence_context import EvidenceSnippet, select_evidence
    snippets = (EvidenceSnippet('1', 'a' * 40, '1', primary_text, 1),
                EvidenceSnippet('2', 'b', '1', 'Supplement.', 2))
    result = select_evidence(snippets, 80, budget)
    assert result.snippets[0].citation_id == '1'
    assert result.snippets[0].source == 'a' * 40
    assert result.snippets[0].text == 'Primary fact.'
    assert result.used_tokens <= 80


@pytest.mark.parametrize('body', [
    'See https://example.test/?private=' + 'x' * 1000,
    'See https://example.test/!private=' + 'x' * 1000,
    '| Key | Value 1. ' + 'x' * 1000 + ' |',
    'First cell. | ' + 'x' * 1000 + ' | Last cell.',
    'First cell.\t' + 'x' * 1000 + '\tLast cell.',
    '关键结论。' + '巨型表格|' * 4000,
], ids=['url-query', 'url-bang', 'pipe-row', 'first-cell-period', 'tab-row', 'unseparated-prose-row'])
def test_body_url_and_table_rows_are_never_partially_selected(budget, body):
    from backend.evidence_context import EvidenceSnippet, select_evidence
    result = select_evidence([EvidenceSnippet('1', 'a', '1', body, 1)], 80, budget)
    assert result.snippets == ()
    assert result.omitted_count == 1


@pytest.mark.parametrize('body', ['See https://example.test/?a=1!b=2', '| Key | Value 1. |'])
def test_complete_body_url_or_table_still_retained_when_it_fits(budget, body):
    from backend.evidence_context import EvidenceSnippet, select_evidence
    result = select_evidence([EvidenceSnippet('1', 'a', '1', body, 1)], 200, budget)
    assert result.snippets[0].text == body
    assert result.omitted_count == 0


def test_numeric_ranges_are_suppressed_across_chunks():
    from backend.evidence_context import CitationFilter
    stream = CitationFilter({'1', '3'})
    assert stream.feed('Evidence [1') == 'Evidence '
    assert stream.feed('-') == ''
    assert stream.feed('3') == ''
    assert stream.feed('] and [3]') == ' and [3]'
    assert stream.finish() == ''
