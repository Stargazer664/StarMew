"""Exercise evaluator contracts with model I/O isolated, including partial failure."""
import asyncio
import csv
import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from backend import evaluation as ev


KEYS = tuple(ev._METRICS_DEFINITION)
INPUT = dict(question='question', answer='answer', retrieved_docs=[{'text': 'evidence'}], ground_truth='reference')


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setattr(ev, '_RAGAS_AVAILABLE', True)
    return ev.RAGASEvaluator()


def test_sdk_exception_is_missing_not_a_midpoint(evaluator, monkeypatch):
    def fail(**kwargs): raise TimeoutError('private endpoint details')
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', fail)
    result = evaluator.evaluate(**INPUT)
    assert result['metrics'] == dict.fromkeys(KEYS)
    assert result['overall_score'] is None
    assert result['evaluation_status'] == 'failed'
    assert set(result['metric_status'].values()) == {'failed'}
    assert 'TimeoutError' in result['error'] and 'private' not in result['error']
    assert evaluator.get_history()[0]['metric_errors'] == result['metric_errors']


@pytest.mark.parametrize('invalid', [None, float('nan'), float('inf'), -0.01, 1.01, True, '0.5', [], {}, 10**400, -(10**400)])
def test_partial_invalid_metric_does_not_discard_valid_zero(evaluator, monkeypatch, invalid):
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', lambda **kw: {'faithfulness': 0, 'factual_correctness': invalid})
    result = evaluator.evaluate(**INPUT)
    assert result['metrics']['faithfulness'] == 0
    assert result['metrics']['factual_correctness'] is None
    assert result['overall_score'] == 0
    assert result['evaluation_status'] == 'partial'
    assert result['metric_status']['faithfulness'] == 'success'
    assert result['metric_status']['factual_correctness'] == 'failed'
    assert len(evaluator.history) == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('payload', [None, [], 'invalid'])
def test_malformed_sdk_result_is_recorded_as_failure(evaluator, monkeypatch, payload):
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', lambda **kw: payload)
    result = evaluator.evaluate(**INPUT)
    assert result['evaluation_status'] == 'failed'
    assert all(value is None for value in result['metrics'].values())


@pytest.mark.parametrize('response', [None, 'not JSON', '{"faithfulness": NaN}', '{"faithfulness": 3}', '{"faithfulness": true}'])
def test_fallback_unavailable_or_invalid_never_fabricates_midpoint(evaluator, monkeypatch, response):
    monkeypatch.setattr(ev, '_RAGAS_AVAILABLE', False)
    model = None if response is None else SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content=response))
    monkeypatch.setattr(evaluator, '_get_fallback_llm', lambda: model)
    result = evaluator.evaluate(**INPUT)
    assert result['ragas_native'] is False
    assert result['evaluation_status'] == 'failed'
    assert result['metrics'] == dict.fromkeys(KEYS)
    assert result['overall_score'] is None


def test_partial_fallback_retains_only_actual_valid_scores(evaluator, monkeypatch):
    monkeypatch.setattr(ev, '_RAGAS_AVAILABLE', False)
    monkeypatch.setattr(evaluator, '_get_fallback_llm', lambda: SimpleNamespace(
        invoke=lambda prompt: SimpleNamespace(content='{"faithfulness":0.5,"answer_relevancy":0}')))
    result = evaluator.evaluate(**INPUT)
    assert result['metrics']['faithfulness'] == .5  # A real midpoint remains valid.
    assert result['metrics']['answer_relevancy'] == 0
    assert result['metrics']['context_precision'] is None
    assert result['evaluation_status'] == 'partial' and result['overall_score'] == .25


def test_inapplicable_metrics_are_distinct_from_failures(evaluator, monkeypatch):
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', lambda **kw: {'answer_relevancy': .8})
    result = evaluator.evaluate(question='q', answer='a', retrieved_docs=[], ground_truth=None)
    assert result['evaluation_status'] == 'success'
    assert result['overall_score'] == .8
    assert result['metric_status'] == {k: 'success' if k == 'answer_relevancy' else 'not_applicable' for k in KEYS}
    assert result['metric_errors'] == {}


def test_sdk_selects_only_metrics_whose_inputs_exist(evaluator, monkeypatch):
    captured = []
    monkeypatch.setattr(evaluator, '_get_ragas_llm', lambda: object())
    monkeypatch.setattr(evaluator, '_get_ragas_embeddings', lambda: object())
    def factory(name):
        return lambda **kw: SimpleNamespace(name=name)
    for name in ['Faithfulness', 'AnswerRelevancy', 'ContextPrecision', 'ContextRecall', 'FactualCorrectness']:
        monkeypatch.setattr(ev, name, factory(name))
    monkeypatch.setattr(ev, 'ragas_evaluate', lambda dataset, **kw: captured.extend(m.name for m in kw['metrics']) or {})
    evaluator._evaluate_with_ragas('q', 'a', [], None)
    assert captured == ['AnswerRelevancy']
    captured.clear()
    evaluator._evaluate_with_ragas('q', 'a', ['context'], None)
    assert set(captured) == {'Faithfulness', 'AnswerRelevancy'}


def test_factual_precision_and_recall_are_not_renamed_to_f1(evaluator, monkeypatch):
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', lambda **kw: {'factual_correctness(mode=precision)': .9})
    result = evaluator.evaluate(**INPUT)
    assert result['metrics']['factual_correctness'] is None
    assert result['metric_status']['factual_correctness'] == 'failed'


def test_averages_include_valid_zero_exclude_missing_and_disclose_counts(evaluator, monkeypatch):
    results = iter([{'faithfulness': 0}, {'faithfulness': 1}, {}])
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', lambda **kw: next(results))
    for _ in range(3): evaluator.evaluate(**INPUT)
    summary = evaluator.get_average_scores()
    assert summary['avg_faithfulness'] == .5
    assert summary['avg_context_recall'] is None
    assert summary['avg_overall_score'] == .5
    assert summary['valid_counts']['faithfulness'] == 2
    assert summary['valid_counts']['context_recall'] == 0
    assert summary['valid_overall_count'] == 2 and summary['total_evaluations'] == 3
    assert summary['evaluation_status_counts'] == {'partial': 2, 'failed': 1}


def test_empty_average_and_overall_are_unavailable(evaluator):
    assert ev.calculate_overall_score({}) is None
    assert ev.calculate_overall_score({'faithfulness': float('inf'), 'answer_relevancy': True}) is None
    assert evaluator.get_average_scores()['avg_overall_score'] is None


def test_statistics_ignore_numeric_scores_marked_failed_or_errored(evaluator):
    evaluator.history = [
        ev.EvalResult(sample=ev.EvalSample('q'), metrics={'faithfulness': .9},
                      overall_score=.9, metric_status={'faithfulness': 'failed'}),
        ev.EvalResult(sample=ev.EvalSample('q'), metrics={'faithfulness': .8},
                      overall_score=.8, metric_errors={'faithfulness': 'judge_error'}),
        ev.EvalResult(sample=ev.EvalSample('q'), metrics={'faithfulness': 0}, overall_score=.7),
    ]
    summary = evaluator.get_average_scores()
    assert summary['avg_faithfulness'] == summary['avg_overall_score'] == 0
    assert summary['valid_counts']['faithfulness'] == summary['valid_overall_count'] == 1
    distribution = evaluator.generate_visualization_data()['metric_distributions']['faithfulness']
    assert distribution == {'min': 0, 'max': 0, 'avg': 0, 'count': 1}


def test_no_applicable_metrics_do_not_call_model(evaluator, monkeypatch):
    def unexpected(**kwargs): raise AssertionError('model should not be called')
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', unexpected)
    result = evaluator.evaluate(question='', answer='', retrieved_docs=[])
    assert result['evaluation_status'] == 'not_applicable'
    assert result['overall_score'] is None and result['metric_errors'] == {}


def test_pipeline_failure_stays_in_history_and_batch_denominator(evaluator):
    def fail(question): raise RuntimeError('upstream offline')
    samples = [ev.EvalSample(question='a'), ev.EvalSample(question='b')]
    rows = asyncio.run(evaluator.evaluate_batch(samples, rag_pipeline_fn=fail, max_concurrency=2))
    assert len(evaluator.history) == len(rows) == 2
    assert all(row.overall_score is None and row.evaluation_status == 'failed' for row in rows)
    assert evaluator.get_average_scores()['total_evaluations'] == 2


def test_sample_returns_its_own_record_when_another_history_append_interleaves(evaluator, monkeypatch):
    class InterleavingHistory(list):
        def append(self, item):
            super().append(item)
            super().append(ev.EvalResult(sample=ev.EvalSample(question='other'), metrics={'faithfulness': .1}))
    evaluator.history = InterleavingHistory()
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', lambda **kw: {'faithfulness': .9})
    result = evaluator.evaluate_sample(ev.EvalSample(question='mine', ground_truth='r'), 'a', [{'text': 'c'}])
    assert result.metrics['faithfulness'] == .9
    assert evaluator.history[-1].sample.question == 'other'


def test_json_csv_console_keep_missing_scores_and_status(evaluator, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator, '_evaluate_with_ragas', lambda **kw: {})
    evaluator.evaluate(**INPUT)
    evaluator.export_results(str(tmp_path/'result.json'))
    evaluator.export_csv(str(tmp_path/'result.csv'))
    result = json.loads((tmp_path/'result.json').read_text(encoding='utf-8'))
    assert result['evaluations'][0]['overall_score'] is None
    assert result['evaluations'][0]['evaluation_status'] == 'failed'
    with (tmp_path/'result.csv').open(encoding='utf-8') as f: row = next(csv.DictReader(f))
    assert row['overall_score'] == row['faithfulness'] == ''
    assert row['evaluation_status'] == 'failed'
    assert json.loads(row['metric_status'])['faithfulness'] == 'failed'
    stream = io.StringIO()
    monkeypatch.setattr(ev, 'console', Console(file=stream, force_terminal=False))
    evaluator.print_summary()
    assert 'N/A' in stream.getvalue()
