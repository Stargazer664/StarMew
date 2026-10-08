"""Threshold policy against real rerank/merge code, with only model I/O isolated."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from backend import rag_utils as rag
from backend.conversation_context import ContextBudget


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.delenv('RERANK_MIN_SCORE', raising=False)
    monkeypatch.delenv('LLM_RERANK_MIN_SCORE', raising=False)
    monkeypatch.setattr(rag, 'RERANK_MODEL', None)
    monkeypatch.setattr(rag, 'RERANK_API_KEY', None)
    monkeypatch.setattr(rag, 'RERANK_BINDING_HOST', None)


@pytest.fixture
def docs():
    return [dict(chunk_id=key, filename=key + '.md', text='Evidence ' + key + '.',
                 chunk_level=3, score=100-i, parent_chunk_id='P') for i, key in enumerate('ABCD')]


@pytest.fixture
def score_model(monkeypatch):
    def configure(items):
        model = SimpleNamespace(invoke=lambda *a, **k: SimpleNamespace(content=json.dumps(items)))
        monkeypatch.setattr(rag, '_get_stepback_model', lambda: model)
    return configure


@pytest.fixture
def external(monkeypatch):
    monkeypatch.setattr(rag, 'RERANK_MODEL', 'fixture')
    monkeypatch.setattr(rag, 'RERANK_API_KEY', 'fixture')
    monkeypatch.setattr(rag, 'RERANK_BINDING_HOST', 'https://example.invalid')
    def configure(items=None, status=200):
        monkeypatch.setattr(rag.requests, 'post', lambda *a, **k: SimpleNamespace(
            status_code=status, text='unavailable', json=lambda: {'results': items}))
    return configure


def test_llm_threshold_never_refills_low_or_unscored_chunks(docs, score_model, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    score_model([{'doc_id': 1, 'score': 9}, {'doc_id': 2, 'score': 5}, {'doc_id': 3, 'score': 6}])
    before = deepcopy(docs)
    result, meta = rag._rerank_documents('question', docs, 4)
    assert [d['chunk_id'] for d in result] == ['A', 'C']
    policy = meta['relevance_filter']
    assert (policy['scored_count'], policy['kept_count'], policy['rejected_count'], policy['unscored_count']) == (3, 2, 1, 1)
    assert docs == before


def test_even_one_candidate_is_scored_and_can_be_rejected(docs, score_model, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    score_model([{'doc_id': 1, 'score': 5}])
    result, meta = rag._rerank_documents('question', docs[:1], 5)
    assert result == []
    assert meta['relevance_filter']['status'] == 'filtered_all'
    assert meta['retrieval_status'] == 'below_threshold'


def test_zero_threshold_is_active_and_equal_score_passes(docs, score_model, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '0')
    score_model([{'doc_id': 2, 'score': 0}])
    result, meta = rag._rerank_documents('question', docs, 3)
    assert [d['chunk_id'] for d in result] == ['B']
    assert meta['relevance_filter']['enabled'] is True


def test_partial_below_threshold_does_not_claim_all_candidates_irrelevant(docs, score_model, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    score_model([{'doc_id': 1, 'score': 2}])
    result, meta = rag._rerank_documents('question', docs, 2)
    assert result == []
    assert meta['retrieval_status'] == 'rerank_failed'
    assert meta['relevance_filter']['status'] == 'incomplete'


@pytest.mark.parametrize('items', [[], [{'doc_id': 99, 'score': 9}], [{'doc_id': 1, 'score': float('nan')}]])
def test_invalid_llm_scoring_never_returns_raw_recall(docs, score_model, monkeypatch, items):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    score_model(items)
    result, meta = rag._rerank_documents('question', docs, 2)
    assert result == []
    assert meta['retrieval_status'] == 'rerank_failed'


def test_unavailable_model_is_unknown_not_below_threshold(docs, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    monkeypatch.setattr(rag, '_get_stepback_model', lambda: None)
    result, meta = rag._rerank_documents('question', docs, 2)
    assert result == []
    assert meta['relevance_filter']['status'] == 'scoring_failed'
    assert meta['retrieval_status'] == 'rerank_failed'


def test_external_uses_own_scale_and_does_not_refill(docs, external, monkeypatch):
    monkeypatch.setenv('RERANK_MIN_SCORE', '-0.5')
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '9')
    external([{'index': 0, 'relevance_score': -0.5}, {'index': 1, 'relevance_score': -0.6}])
    result, meta = rag._rerank_documents('question', docs, 3)
    assert [d['chunk_id'] for d in result] == ['A']
    assert meta['relevance_filter']['score_field'] == 'rerank_score'


def test_external_failure_does_not_certify_stale_or_rrf_scores(docs, external, monkeypatch):
    monkeypatch.setenv('RERANK_MIN_SCORE', '0.5')
    docs[0]['rerank_score'] = 1.0
    external(status=503)
    result, meta = rag._rerank_documents('question', docs, 3)
    assert result == []
    assert meta['retrieval_status'] == 'rerank_failed'


@pytest.mark.parametrize('value', ['nan', 'inf', 'true', 'invalid', '-1', '11'])
def test_invalid_llm_threshold_fails_closed_without_model_call(docs, monkeypatch, value):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', value)
    monkeypatch.setattr(rag, '_get_stepback_model', lambda: pytest.fail('invalid configuration must not call model'))
    result, meta = rag._rerank_documents('question', docs, 1)
    assert result == []
    assert meta['retrieval_status'] == 'rerank_failed'
    assert meta['relevance_filter']['status'] == 'invalid_configuration'


def test_only_inactive_scorer_threshold_does_not_change_legacy_behavior(docs, score_model, monkeypatch):
    monkeypatch.setenv('RERANK_MIN_SCORE', 'invalid')
    score_model([{'doc_id': 2, 'score': 0}])
    result, _ = rag._rerank_documents('question', docs, 3)
    assert [d['chunk_id'] for d in result] == ['B', 'A', 'C']


def test_threshold_keeps_scored_leaves_instead_of_unscored_parent(docs, score_model, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    monkeypatch.setattr(rag, 'AUTO_MERGE_ENABLED', True)
    monkeypatch.setattr(rag, 'AUTO_MERGE_THRESHOLD', 2)
    score_model([{'doc_id': 1, 'score': 9}, {'doc_id': 2, 'score': 8}])
    snapshot = {'docs': docs[:2], 'parents': {'P': {'chunk_id': 'P', 'text': 'unscored tail'}},
                'meta': {'retrieval_mode': 'hybrid'}}
    result = rag._finish_retrieval('question', snapshot, 5)
    assert [d['chunk_id'] for d in result['docs']] == ['A', 'B']
    assert result['meta']['auto_merge_applied'] is False
    assert result['meta']['auto_merge_skip_reason'] == 'relevance_filter'


def test_threshold_mode_does_not_fetch_unused_parents(docs, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    monkeypatch.setattr(rag, 'AUTO_MERGE_ENABLED', True)
    monkeypatch.setattr(rag, '_parent_chunk_store', SimpleNamespace(
        get_documents_by_ids=lambda ids: pytest.fail('threshold mode must not query parents')))
    snapshot = rag._snapshot_candidates(docs, 'hybrid', 30, 5)
    assert snapshot['parents'] == {}
    assert snapshot['docs'] == docs


def test_oversized_candidate_cannot_pass_by_scoring_a_compressed_prefix(docs, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    monkeypatch.setattr(rag, 'model_budget', lambda: ContextBudget(window=1800, output_reserve=300))
    docs = [dict(docs[0], text='Useful opening. ' + 'Unseen tail. ' * 200), docs[1]]
    def invoke(prompt, **kwargs):
        return SimpleNamespace(content=json.dumps([{'doc_id': 1, 'score': 10}, {'doc_id': 2, 'score': 8}]))
    monkeypatch.setattr(rag, '_get_stepback_model', lambda: SimpleNamespace(invoke=invoke))
    result, meta = rag._rerank_documents('question', docs, 5)
    assert [d['chunk_id'] for d in result] == ['B']
    assert result[0]['text'] == docs[1]['text']
    assert meta['relevance_filter']['unscored_count'] == 1


def test_trace_preserves_all_scored_candidates_before_top_k(docs, score_model, monkeypatch):
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '6')
    score_model([{'doc_id': i + 1, 'score': score} for i, score in enumerate([9, 8, 7, 1])])
    result, meta = rag._rerank_documents('question', docs, 1)
    assert [d['chunk_id'] for d in result] == ['A']
    decisions = meta['relevance_filter']['candidates']
    assert [r['decision'] for r in decisions] == ['kept', 'top_k_limit', 'top_k_limit', 'below_threshold']
    assert all(len(r['text_sha256']) == 64 for r in decisions)
