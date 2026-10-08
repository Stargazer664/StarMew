"""Exercise public evaluation JSON through real FastAPI response serialization."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def evaluation_client(monkeypatch):
    # Importing the router does not start the application or any infrastructure.
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *a, **kw: False)
    monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
    from backend import api

    app = FastAPI()
    app.include_router(api.router)
    user = lambda: SimpleNamespace(username='xin', role='admin')
    app.dependency_overrides[api.get_current_user] = user
    app.dependency_overrides[api.require_admin] = user

    def configure(payload):
        result = deepcopy(payload)
        batch = SimpleNamespace(
            sample=SimpleNamespace(question='question'), answer='answer', retrieved_docs=[],
            metrics=result.get('metrics', {}), overall_score=result.get('overall_score'),
            error=result.get('error'), latency=0.0,
            **{key: result[key] for key in (
                'evaluation_status', 'metric_status', 'metric_errors', 'ragas_native', 'ragas_version'
            ) if key in result},
        )
        fake = SimpleNamespace(
            evaluate=lambda **kwargs: deepcopy(result),
            evaluate_batch_sync=lambda **kwargs: [batch],
            get_history=lambda **kwargs: [deepcopy(result)],
            get_average_scores=lambda: {'factual_correctness': None},
            export_results=lambda path: None,
        )
        monkeypatch.setattr(api, 'evaluator', fake)

    with TestClient(app) as client:
        yield client, configure


def request_result(client, route):
    if route == 'single':
        response = client.post('/evaluation/run', json={'question': 'question', 'answer': 'answer'})
        key = 'evaluation'
    elif route == 'batch':
        response = client.post('/evaluation/batch', json={'questions': ['question']})
        key = 'results'
    else:
        response = client.get('/evaluation/history')
        key = 'evaluations'
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['average_scores'] == {'factual_correctness': None}
    if route == 'single':
        return body[key]
    assert len(body[key]) == 1
    return body[key][0]


@pytest.mark.parametrize('route', ['single', 'batch', 'history'])
def test_factual_and_legacy_correctness_remain_independent(evaluation_client, route):
    client, configure = evaluation_client
    configure({
        'metrics': {'factual_correctness': 0.73, 'answer_correctness': 0.21},
        'overall_score': 0.6, 'evaluation_status': 'success',
        'metric_status': {'factual_correctness': 'success'}, 'metric_errors': {},
        'ragas_native': True, 'ragas_version': '0.4.3',
    })
    result = request_result(client, route)
    assert result['metrics']['factual_correctness'] == 0.73
    assert result['metrics']['answer_correctness'] == 0.21
    assert result['evaluation_status'] == 'success'
    assert result['metric_status'] == {'factual_correctness': 'success'}
    assert result['metric_errors'] == {}
    assert result['ragas_native'] is True and result['ragas_version'] == '0.4.3'


@pytest.mark.parametrize('route', ['single', 'batch', 'history'])
def test_zero_score_and_partial_failure_survive_serialization(evaluation_client, route):
    client, configure = evaluation_client
    configure({
        'metrics': {'factual_correctness': 0.0, 'faithfulness': None},
        'overall_score': 0.0, 'evaluation_status': 'partial',
        'metric_status': {'factual_correctness': 'success', 'faithfulness': 'failed'},
        'metric_errors': {'faithfulness': 'judge_timeout'},
        'ragas_native': True, 'ragas_version': '0.4.3', 'error': 'Some metrics failed',
    })
    result = request_result(client, route)
    assert result['metrics']['factual_correctness'] == 0.0
    assert result['metrics']['faithfulness'] is None
    assert result['metrics']['answer_correctness'] is None
    assert result['overall_score'] == 0.0
    assert result['evaluation_status'] == 'partial'
    assert result['metric_status'] == {'factual_correctness': 'success', 'faithfulness': 'failed'}
    assert result['metric_errors'] == {'faithfulness': 'judge_timeout'}
    assert result['error'] == 'Some metrics failed'


@pytest.mark.parametrize('route', ['single', 'batch', 'history'])
def test_failed_evaluation_returns_null_scores_and_failure_details(evaluation_client, route):
    client, configure = evaluation_client
    configure({
        'metrics': {'factual_correctness': None}, 'overall_score': None,
        'evaluation_status': 'failed', 'metric_status': {'factual_correctness': 'failed'},
        'metric_errors': {'factual_correctness': 'judge_unavailable'},
        'ragas_native': False, 'ragas_version': '', 'error': 'Evaluation unavailable',
    })
    result = request_result(client, route)
    assert result['overall_score'] is None
    assert result['metrics']['factual_correctness'] is None
    assert result['metrics']['answer_correctness'] is None
    assert result['evaluation_status'] == 'failed'
    assert result['metric_status'] == {'factual_correctness': 'failed'}
    assert result['metric_errors'] == {'factual_correctness': 'judge_unavailable'}
    assert result['error'] == 'Evaluation unavailable'
    assert result['ragas_native'] is False and result['ragas_version'] == ''


@pytest.mark.parametrize('route', ['single', 'batch', 'history'])
def test_legacy_only_record_does_not_invent_factual_or_success(evaluation_client, route):
    client, configure = evaluation_client
    configure({'metrics': {'answer_correctness': 0.0}})
    result = request_result(client, route)
    assert result['metrics']['answer_correctness'] == 0.0
    assert result['metrics']['factual_correctness'] is None
    assert result['overall_score'] is None
    assert result['evaluation_status'] == 'unknown'
    assert result['metric_status'] == {} and result['metric_errors'] == {}
    assert result['ragas_native'] is None and result['ragas_version'] == ''


def test_openapi_distinguishes_factual_metric_and_deprecated_legacy_field(evaluation_client):
    client, _ = evaluation_client
    schema = client.get('/openapi.json').json()['components']['schemas']
    metrics = schema['EvaluationMetrics']['properties']
    assert {item['type'] for item in metrics['factual_correctness']['anyOf']} == {'number', 'null'}
    assert metrics['answer_correctness']['deprecated'] is True
    result = schema['EvaluationResult']['properties']
    assert {item['type'] for item in result['overall_score']['anyOf']} == {'number', 'null'}
    assert {item['type'] for item in result['ragas_native']['anyOf']} == {'boolean', 'null'}
    assert result['evaluation_status']['default'] == 'unknown'
    assert result['metric_status']['additionalProperties'] == {'type': 'string'}
    assert result['metric_errors']['additionalProperties'] == {'type': 'string'}
    assert result['ragas_version']['default'] == ''
