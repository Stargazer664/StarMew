from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_diagnostics_routes_require_admin_and_keep_ack_separate(tmp_path):
    assert Path('backend/diagnostics_api.py').exists(), 'Missing admin diagnostic routes'
    from backend import telemetry
    from backend.auth import get_current_user
    api=importlib.import_module('backend.diagnostics_api')
    app=FastAPI()
    app.include_router(api.router,prefix='/api')
    app.dependency_overrides[get_current_user]=lambda:SimpleNamespace(username='reader',role='user')
    with TestClient(app) as client:
        assert client.get('/api/diagnostics/chat/traces').status_code==403
        app.dependency_overrides[get_current_user]=lambda:SimpleNamespace(username='admin',role='admin')
        assert client.get('/api/diagnostics/chat/traces').status_code==503
        telemetry.start_diagnostics(tmp_path/'jobs.sqlite3',tmp_path/'logs')
        try:
            with telemetry.trace('chat','answer',user_id='private-user',session_id='private-session') as execution:
                telemetry.event('model_failed',status='failed',code='MODEL_FAILED')
            telemetry.flush()
            response=client.get('/api/diagnostics/chat/traces')
            assert response.status_code==200
            assert 'private-user' not in response.text and 'private-session' not in response.text
            assert response.json()['items'][0]['id']==execution.id
            assert client.get('/api/diagnostics/chat/summary?window=bad').status_code==422
            assert client.get('/api/diagnostics/chat/traces/missing').status_code==404
            alert=client.get('/api/documents/diagnostics/alerts').json()['items'][0]
            response=client.post(f"/api/documents/diagnostics/alerts/{alert['id']}/acknowledge")
            assert response.status_code==200 and response.json()['acknowledged_at']
            assert response.json()['resolved_at'] is None
        finally:
            telemetry.stop_diagnostics()


def test_request_correlation_headers_are_server_generated():
    assert Path('backend/diagnostics_api.py').exists(), 'Missing request correlation middleware'
    from backend.diagnostics_api import RequestCorrelationMiddleware
    from backend.telemetry import get_context
    app=FastAPI()
    app.add_middleware(RequestCorrelationMiddleware)
    @app.get('/api/chat')
    def demo():
        return {k:get_context().get(k) for k in ('request_id','trace_id')}
    with TestClient(app) as client:
        response=client.get('/api/chat',headers={'X-Request-ID':'injected'})
        assert response.headers['x-request-id']!='injected'
        assert response.json()['request_id']==response.headers['x-request-id']
        assert response.json()['trace_id']==response.headers['x-trace-id']
