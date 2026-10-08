"""Offline contract tests with hand-calculated retrieval outcomes, no services."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent / 'scripts' / 'summarize_relevance_retest.py'


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def module():
    if not SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location('relevance_retest', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def inputs(tmp_path):
    chunks = [{'chunk_id': f'a{i}', 'filename': 'a.md', 'text': f'事实 {i}。', 'chunk_level': 3}
              for i in range(1, 7)]
    chunks += [{'chunk_id': 'b1', 'filename': 'b.md', 'text': '其他问题的证据。', 'chunk_level': 3}]
    chunks += [{'chunk_id': f'x{i}', 'filename': 'other.md', 'text': f'背景 {i}。', 'chunk_level': 3}
               for i in range(1, 1345)]
    by_id = {c['chunk_id']: c for c in chunks}
    samples = [{'id': f'q{i}', 'question': f'问题 {i}？', 'expected_docs': ['a.md' if i == 1 else 'b.md'],
                'metadata': {'answerability': 'answerable' if i <= 50 else 'unanswerable'}}
               for i in range(1, 61)]
    dataset = {'schema_version': 1, 'split': 'exploratory', 'formal_holdout': False, 'samples': samples}
    corpus = {'schema_version': 1, 'collection': 'fixture', 'index_identity_sha256': 'a' * 64, 'chunks': chunks}
    qrels = {'schema_version': 1, 'review_method': 'ai_source_checked', 'human_reviewed': False,
             'dataset_sha256': sha(encode(dataset)), 'corpus_sha256': sha(encode(corpus)), 'samples': []}
    for sample in samples[:50]:
        scope = chunks[:6] if sample['id'] == 'q1' else chunks[6:7]
        qrels['samples'].append({'sample_id': sample['id'], 'expected_docs': sample['expected_docs'],
            'scope_complete': True, 'reference_facts_verified': True,
            'relevant_chunk_ids': [c['chunk_id'] for c in scope],
            'judgments': [{'chunk_id': c['chunk_id'], 'relevant': True, 'reason': '原文证据', 'quote': c['text']}
                          for c in scope]})
    records = [{'sample_id': s['id'], 'question': s['question'], 'answerability': s['metadata']['answerability'],
                'status': 'exception', 'raw_available': False, 'raw_docs': [], 'docs': [], 'meta': {},
                'elapsed_seconds': 0.25, 'error_type': 'TimeoutError'} for s in samples]
    raw = [copy.deepcopy(by_id[i]) for i in ['x1', 'a1', 'a2', 'a3', 'x2', 'x3', 'x4', 'a4', 'x5', 'a5']]
    # Eight pass; only a1 is in production top5. a2 is low, a3 unscored,
    # a4/a5 are topK losses, and a6 was not recalled.
    scores = [9, 8, 5, None, 7, 7, 7, 6, 6, 6]
    decisions = ['kept', 'kept', 'below_threshold', 'unscored', 'kept', 'kept', 'kept',
                 'top_k_limit', 'top_k_limit', 'top_k_limit']
    policy = {'enabled': True, 'score_field': 'llm_rerank_score', 'threshold': 6, 'status': 'passed',
              'input_count': 10, 'scored_count': 9, 'kept_count': 5, 'rejected_count': 1,
              'unscored_count': 1, 'top_k_omitted_count': 3,
              'candidates': [{'index': i, 'chunk_id': c['chunk_id'], 'filename': c['filename'],
                              'text_sha256': sha(c['text'].encode('utf-8')), 'score': scores[i],
                              'decision': decisions[i]} for i, c in enumerate(raw)]}
    records[0].update(status='success', raw_available=True, raw_docs=raw, error_type=None,
        docs=[dict(raw[i], llm_rerank_score=scores[i]) for i in [0, 1, 4, 5, 6]],
        meta={'retrieval_mode': 'hybrid', 'relevance_filter': policy})
    measurement = {'protocol': {'top_k': 5, 'threshold': 6, 'score_field': 'llm_rerank_score'},
                   'status': 'complete', 'dataset_sha256': sha(encode(dataset)),
                   'corpus_sha256': sha(encode(corpus)), 'qrels_sha256': sha(encode(qrels)), 'records': records}
    values = dict(dataset=dataset, corpus=corpus, qrels=qrels, measurement=measurement)
    paths = {key: tmp_path / f'{key}.json' for key in values}
    for key, value in values.items():
        paths[key].write_bytes(encode(value))
    return values, paths


def save(inputs, *names, rehash=False):
    values, paths = inputs
    if rehash:
        values['qrels']['dataset_sha256'] = sha(encode(values['dataset']))
        values['qrels']['corpus_sha256'] = sha(encode(values['corpus']))
        for name in ('dataset', 'corpus', 'qrels'):
            values['measurement'][name + '_sha256'] = sha(encode(values[name]))
        names = tuple(values)
    for name in names:
        paths[name].write_bytes(encode(values[name]))


def build(module, inputs):
    assert module is not None, 'offline retest summarizer is not implemented'
    return module.summarize(**{name + '_path': path for name, path in inputs[1].items()})


def test_fixed_denominators_and_actual_precision_are_not_conflated(module, inputs):
    result = build(module, inputs)
    actual = result['answerable']['filtered_top5']
    assert actual['question_count'] == 50
    assert actual['precision_at_k_macro'] == pytest.approx(.2 / 50)
    assert actual['recall_at_k_macro'] == pytest.approx(1 / 6 / 50)
    assert actual['actual_precision_micro'] == pytest.approx(.2)
    assert actual['actual_precision_macro'] == pytest.approx(.2 / 50)
    assert actual['actual_precision_nonempty_macro'] == pytest.approx(.2)
    assert actual['f1_actual_macro'] == pytest.approx((2 / 11) / 50)
    assert actual['mean_returned_count'] == pytest.approx(.1)
    assert actual['failure_count'] == actual['empty_count'] == 49
    raw = result['answerable']['raw_top5']
    assert raw['precision_at_k_macro'] == pytest.approx(.6 / 50)
    assert raw['recall_at_k_macro'] == pytest.approx(.5 / 50)
    assert raw['raw_unavailable_count'] == 49


def test_shadow_top10_uses_same_scores_not_actual_five_or_new_generation(module, inputs):
    result = build(module, inputs)
    shadow = result['answerable']['shadow_filtered_top10']
    assert shadow['precision_at_k_macro'] == pytest.approx(.3 / 50)
    assert shadow['recall_at_k_macro'] == pytest.approx(.5 / 50)
    assert shadow['actual_precision_micro'] == pytest.approx(3 / 8)
    assert shadow['mean_returned_count'] == pytest.approx(8 / 50)
    assert result['answerable']['raw_top10']['recall_at_k_macro'] == pytest.approx(5 / 6 / 50)
    row = result['questions'][0]
    assert row['loss_attribution'] == {'kept': ['a1'], 'not_recalled': ['a6'], 'unscored': ['a3'],
                                       'below_threshold': ['a2'], 'top_k_limit': ['a4', 'a5'],
                                       'retrieval_unavailable': [], 'processing_failure': []}
    assert result['scoring_coverage']['scored_count'] == 9
    assert result['scoring_coverage']['unscored_count'] == 1
    assert result['scoring_coverage']['scored_fraction'] == pytest.approx(.9)


def test_exception_keeps_observed_raw_diagnostic_and_zero_filtered_denominator(module, inputs):
    values = inputs[0]
    row = values['measurement']['records'][0]
    row.update(status='exception', docs=[], meta={}, error_type='TimeoutError')
    save(inputs, 'measurement')
    result = build(module, inputs)
    assert result['answerable']['raw_top5']['recall_at_k_macro'] == pytest.approx(.5 / 50)
    assert result['answerable']['filtered_top5']['recall_at_k_macro'] == 0
    assert result['answerable']['filtered_top5']['failure_count'] == 50
    assert result['questions'][0]['loss_attribution']['unscored'] == ['a1', 'a2', 'a3', 'a4', 'a5']


def test_infrastructure_failure_without_filter_is_counted_not_rejected(module, inputs):
    row = inputs[0]['measurement']['records'][0]
    row.update(status='success', raw_available=False, raw_docs=[], docs=[], error_type=None,
               meta={'retrieval_mode': 'failed'})
    save(inputs, 'measurement')
    result = build(module, inputs)
    assert result['answerable']['filtered_top5']['failure_count'] == 50
    assert result['answerable']['filtered_top5']['actual_precision_nonempty_macro'] is None
    assert result['questions'][0]['loss_attribution']['retrieval_unavailable'] == ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']


def test_negative_questions_only_report_retrieval_outcomes(module, inputs):
    result = build(module, inputs)
    negative = result['unanswerable']
    assert negative['question_count'] == negative['empty_count'] == negative['failure_count'] == 10
    assert negative['filtered_all_count'] == negative['returned_count'] == 0
    assert not any('recall' in key or 'accuracy' in key for key in negative)


def test_incomplete_is_separate_from_scoring_errors_and_complete_low_scores(module, inputs):
    v = inputs[0]
    original = v['measurement']['records'][0]
    for offset, status, scores in [(50, 'incomplete', [1, None]),
                                   (51, 'filtered_all', [1]), (52, 'scoring_failed', [None])]:
        row = v['measurement']['records'][offset]
        raw = copy.deepcopy(original['raw_docs'][:len(scores)])
        policy = {'enabled': True, 'score_field': 'llm_rerank_score', 'threshold': 6, 'status': status,
                  'input_count': len(raw), 'scored_count': sum(s is not None for s in scores),
                  'kept_count': 0, 'rejected_count': sum(s is not None for s in scores),
                  'unscored_count': sum(s is None for s in scores), 'top_k_omitted_count': 0,
                  'candidates': [dict(original['meta']['relevance_filter']['candidates'][i],
                                      score=score, decision='unscored' if score is None else 'below_threshold')
                                 for i, score in enumerate(scores)]}
        row.update(status='success', raw_available=True, raw_docs=raw, docs=[], error_type=None,
                   meta={'retrieval_mode': 'hybrid', 'relevance_filter': policy,
                         'retrieval_status': 'below_threshold' if status == 'filtered_all' else 'rerank_failed'})
    save(inputs, 'measurement')
    result = build(module, inputs)
    no = result['unanswerable']
    assert no['incomplete_count'] == no['scoring_failed_count'] == no['filtered_all_count'] == 1
    assert no['failure_count'] == 9
    assert result['audit']['exception_count'] == 56  # 49 positives and seven negatives


def test_short_return_actual_precision_uses_actual_count_and_degraded_path_is_explicit(module, inputs):
    row = inputs[0]['measurement']['records'][0]
    raw = copy.deepcopy(row['raw_docs'][1])
    candidate = dict(row['meta']['relevance_filter']['candidates'][1], index=0)
    row.update(raw_docs=[raw], docs=[dict(raw, llm_rerank_score=8)],
               meta={'retrieval_mode': 'dense_fallback', 'relevance_filter': {
                   'enabled': True, 'score_field': 'llm_rerank_score', 'threshold': 6, 'status': 'passed',
                   'input_count': 1, 'scored_count': 1, 'kept_count': 1, 'rejected_count': 0,
                   'unscored_count': 0, 'top_k_omitted_count': 0, 'candidates': [candidate]}})
    save(inputs, 'measurement')
    result = build(module, inputs)
    actual = result['answerable']['filtered_top5']
    assert actual['actual_precision_macro'] == pytest.approx(1 / 50)
    assert actual['precision_at_k_macro'] == pytest.approx(.2 / 50)
    assert actual['f1_actual_macro'] == pytest.approx(2 / 7 / 50)
    assert actual['dense_fallback_count'] == result['audit']['dense_fallback_count'] == 1


@pytest.mark.parametrize('mutation', [
    'missing_record', 'duplicate_record', 'question', 'answerability', 'incomplete_run', 'hash',
    'raw_identity', 'kept_identity', 'candidate_hash', 'candidate_index', 'candidate_score',
    'candidate_decision', 'count', 'kept_order', 'kept_score', 'missing_policy', 'threshold',
    'qrels_scope', 'qrels_unverified', 'qrels_quote', 'wrong_sample_count',
])
def test_invalid_or_misaligned_inputs_are_rejected(module, inputs, mutation):
    v = inputs[0]
    record = v['measurement']['records'][0]
    policy = record['meta']['relevance_filter']
    if mutation == 'missing_record': v['measurement']['records'].pop()
    elif mutation == 'duplicate_record': v['measurement']['records'].append(copy.deepcopy(record))
    elif mutation == 'question': record['question'] += 'changed'
    elif mutation == 'answerability': record['answerability'] = 'unanswerable'
    elif mutation == 'incomplete_run': v['measurement']['status'] = 'running'
    elif mutation == 'hash': v['measurement']['dataset_sha256'] = '0' * 64
    elif mutation == 'raw_identity': record['raw_docs'][0]['text'] += ' fabricated'
    elif mutation == 'kept_identity': record['docs'][0]['filename'] = 'a.md'
    elif mutation == 'candidate_hash': policy['candidates'][0]['text_sha256'] = '0' * 64
    elif mutation == 'candidate_index': policy['candidates'][0]['index'] = 1
    elif mutation == 'candidate_score': policy['candidates'][0]['score'] = True
    elif mutation == 'candidate_decision': policy['candidates'][2]['decision'] = 'kept'
    elif mutation == 'count': policy['scored_count'] = 10
    elif mutation == 'kept_order': record['docs'].reverse()
    elif mutation == 'kept_score': record['docs'][0]['llm_rerank_score'] = 8
    elif mutation == 'missing_policy': record['meta'].pop('relevance_filter')
    elif mutation == 'threshold': policy['threshold'] = 7
    elif mutation == 'qrels_scope': v['qrels']['samples'][0]['judgments'].pop()
    elif mutation == 'qrels_unverified': v['qrels']['samples'][0]['reference_facts_verified'] = False
    elif mutation == 'qrels_quote': v['qrels']['samples'][0]['judgments'][0]['quote'] = '不存在的证据'
    elif mutation == 'wrong_sample_count': v['dataset']['samples'].pop()
    save(inputs, *v, rehash=mutation.startswith('qrels_') or mutation == 'wrong_sample_count')
    with pytest.raises(ValueError):
        build(module, inputs)


def test_cli_writes_three_new_artifacts_and_refuses_overwrite(module, inputs, tmp_path):
    output = tmp_path / 'result'
    command = [sys.executable, '-X', 'utf8', str(SCRIPT)]
    for key, value in inputs[1].items(): command += ['--' + key, str(value)]
    command += ['--output-dir', str(output)]
    first = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=20)
    assert first.returncode == 0, first.stderr
    assert {p.name for p in output.iterdir()} == {'summary.json', 'summary.md', 'questions.json'}
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    second = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=20)
    assert second.returncode != 0
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert 'shadow' in before['summary.md'].decode('utf-8')
