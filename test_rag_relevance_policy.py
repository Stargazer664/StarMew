"""Offline graph coverage for score-threshold outcomes at the retrieval boundary."""
import os
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'

from copy import deepcopy
from types import SimpleNamespace

import pytest

from backend import rag_pipeline as pipeline, tools
from backend.schemas import RagTrace


def doc(name):
    return {'chunk_id': name, 'filename': name + '.md', 'text': name, 'llm_rerank_score': 8}


def retrieved(status, docs=()):
    failed = status == 'rerank_failed'
    return {'docs': list(docs), 'meta': {
        'retrieval_mode': 'hybrid', 'retrieval_status': status,
        'rerank_error': 'secret-provider-detail' if failed else None,
        'relevance_filter': {
            'enabled': True, 'score_field': 'llm_rerank_score', 'threshold': 6,
            'input_count': 1, 'scored_count': 0 if failed else 1,
            'kept_count': len(docs), 'rejected_count': int(not failed and not docs),
            'unscored_count': int(failed), 'status': status,
        },
    }}


@pytest.fixture
def boundary(monkeypatch):
    tools.reset_tool_call_guards()
    calls, grades, steps = [], [], []
    monkeypatch.setattr(pipeline, '_get_router_model', lambda: None)
    monkeypatch.setattr(pipeline, 'step_back_expand', lambda q: {'expanded_query': q + ' expanded'})
    monkeypatch.setattr(pipeline, 'generate_hypothetical_document', lambda q: q + ' hyde')
    monkeypatch.setattr(pipeline, 'emit_rag_step', lambda icon, label, detail='': steps.append((icon, label, detail)))

    def configure(results, scores=()):
        queue, answers = iter(results), iter(scores)
        def retrieve(query, top_k):
            calls.append(query)
            return deepcopy(next(queue))
        monkeypatch.setattr(pipeline, 'retrieve_documents', retrieve)
        class Grader:
            def with_structured_output(self, schema, **kwargs): return self
            def invoke(self, messages, **kwargs):
                grades.append(messages)
                return SimpleNamespace(binary_score=next(answers))
        monkeypatch.setattr(pipeline, '_get_grader_model', lambda: Grader())
    yield configure, calls, grades, steps
    tools.reset_tool_call_guards()


@pytest.mark.parametrize('expanded', [False, True])
def test_below_threshold_keeps_reason_and_routes_by_stage(expanded):
    state = {'docs': [], 'expansion_type': 'step_back' if expanded else None,
             'rag_trace': {'retrieval_status': 'below_threshold'}}
    result = pipeline.grade_documents_node(state)
    assert result['route'] == ('generate_answer' if expanded else 'rewrite_question')
    assert result['rag_trace']['retrieval_status'] == 'below_threshold'
    assert result['rag_trace']['evidence_status'] == 'insufficient'
    assert result['rag_trace']['grade_score'] == 'no'


def test_initial_rerank_failure_is_unknown_error_without_rewrite(boundary):
    configure, calls, grades, steps = boundary
    configure([retrieved('rerank_failed')])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert (result.status, result.evidence_status, result.code) == ('error', 'unknown', 'UPSTREAM_ERROR')
    assert result.evidence == [] and 'secret' not in result.content
    assert len(calls) == 1 and not grades
    assert not any(icon == '✅' and '检索' in label for icon, label, _ in steps)
    trace = tools.get_last_rag_context()['rag_trace']
    assert trace['retrieval_status'] == 'rerank_failed'
    assert trace['grade_score'] == 'unknown'


def test_all_filtered_rewrites_once_and_reports_threshold_not_empty_corpus(boundary):
    configure, calls, grades, _ = boundary
    configure([retrieved('below_threshold'), retrieved('below_threshold')])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert (result.status, result.evidence_status, result.evidence) == ('empty', 'insufficient', [])
    assert '阈值' in result.content and '没有相关片段' not in result.content
    assert calls == ['q', 'q expanded'] and not grades
    trace = tools.get_last_rag_context()['rag_trace']
    assert trace['retrieval_status'] == 'below_threshold'
    assert [r['status'] for r in trace['retrieval_records']] == ['below_threshold'] * 2
    assert [f['stage'] for f in trace['relevance_filters']] == ['initial', 'step_back']
    assert [f['rejected_count'] for f in trace['relevance_filters']] == [1, 1]
    assert RagTrace.model_validate(trace).model_dump()['relevance_filters'] == trace['relevance_filters']


def test_accepted_expansion_does_not_restore_rejected_initial_docs(boundary):
    configure, _, grades, _ = boundary
    configure([retrieved('retrieved', [doc('old')]), retrieved('below_threshold')], ['no'])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.evidence == [] and 'old' not in result.content
    assert result.evidence_status == 'insufficient' and len(grades) == 1
    trace = tools.get_last_rag_context()['rag_trace']
    assert trace['retrieval_status'] == 'below_threshold'
    assert trace['initial_retrieved_chunks'] == [doc('old')]
    assert trace['retrieved_chunks'] == []


def test_filtered_initial_can_recover_with_scored_expansion(boundary):
    configure, calls, grades, _ = boundary
    configure([retrieved('below_threshold'), retrieved('retrieved', [doc('kept')])], ['yes'])
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert result.status == 'ok' and result.evidence_status == 'sufficient'
    assert [d['chunk_id'] for d in result.evidence] == ['kept']
    assert calls == ['q', 'q expanded'] and len(grades) == 1
    trace = tools.get_last_rag_context()['rag_trace']
    assert [f['kept_count'] for f in trace['relevance_filters']] == [0, 1]


@pytest.mark.parametrize('failed_branch', ['hyde', 'step_back'])
def test_complex_partial_scoring_failure_is_conservative_unknown(boundary, monkeypatch, failed_branch):
    configure, calls, grades, _ = boundary
    branches = [retrieved('rerank_failed') if stage == failed_branch else retrieved('retrieved', [doc('kept')])
                for stage in ('hyde', 'step_back')]
    configure([retrieved('below_threshold'), *branches], ['yes'])
    class Router:
        def with_structured_output(self, schema, **kwargs): return self
        def invoke(self, messages, **kwargs): return SimpleNamespace(strategy='complex')
    monkeypatch.setattr(pipeline, '_get_router_model', lambda: Router())
    result = tools.search_knowledge_base.invoke({'query': 'q'})
    assert (result.status, result.evidence_status, result.evidence) == ('error', 'unknown', [])
    assert 'secret' not in result.content and 'kept' not in result.content
    assert len(calls) == 3 and not grades
    trace = tools.get_last_rag_context()['rag_trace']
    assert trace['retrieval_status'] == 'rerank_failed'
    assert [f['stage'] for f in trace['relevance_filters']] == ['initial', 'hyde', 'step_back']
    assert sum(f['unscored_count'] for f in trace['relevance_filters']) == 1
    assert sum(f['kept_count'] for f in trace['relevance_filters']) == 1


def test_tool_does_not_forward_unscored_documents_on_rerank_failure(monkeypatch):
    tools.reset_tool_call_guards()
    monkeypatch.setattr(pipeline, 'run_rag_graph', lambda query: {
        'docs': [{'filename': 'secret.md', 'text': 'raw unscored secret'}],
        'rag_trace': {'retrieval_status': 'rerank_failed', 'evidence_status': 'unknown',
                      'rerank_error': 'secret-provider-detail'},
    })
    try:
        result = tools.search_knowledge_base.invoke({'query': 'q'})
        assert result.status == 'error' and result.evidence_status == 'unknown'
        assert result.evidence == [] and 'secret' not in result.content
    finally:
        tools.reset_tool_call_guards()


def test_filter_metadata_survives_multiple_tool_rounds_and_public_schema(boundary):
    configure, _, _, _ = boundary
    configure([retrieved('retrieved', [doc('a')]), retrieved('retrieved', [doc('b')])], ['yes', 'yes'])
    tools.search_knowledge_base.invoke({'query': 'first'})
    tools.search_knowledge_base.invoke({'query': 'second'})
    trace = tools.get_last_rag_context()['rag_trace']
    assert len(trace['retrieval_rounds']) == len(trace['relevance_filters']) == 2
    assert all(f['threshold'] == 6 and f['kept_count'] == 1 for f in trace['relevance_filters'])
    public = RagTrace.model_validate(trace).model_dump()
    assert public['relevance_filters'] == trace['relevance_filters']
    assert [r['meta']['relevance_filter']['kept_count'] for r in public['retrieval_records']] == [1, 1]
