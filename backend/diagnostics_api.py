"""Admin-only, content-free diagnostic views independent of the search index."""
import sqlite3
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.datastructures import MutableHeaders

from .auth import require_admin
from . import telemetry

router=APIRouter(dependencies=[Depends(require_admin)])


class RequestCorrelationMiddleware:
    def __init__(self,app):
        self.app=app

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        fields={'request_id':uuid4().hex}
        if scope.get('path') in ('/api/chat','/api/chat/stream'):
            fields['trace_id']=uuid4().hex
        async def correlated_send(message):
            if message['type']=='http.response.start':
                headers=MutableHeaders(scope=message)
                headers['X-Request-ID']=fields['request_id']
                if fields.get('trace_id'):
                    headers['X-Trace-ID']=fields['trace_id']
            await send(message)
        with telemetry.bind_context(**fields):
            await self.app(scope,receive,correlated_send)


def query(action):
    try:
        result=action(telemetry.get_store())
    except (sqlite3.Error,OSError,RuntimeError,ValueError):
        raise HTTPException(status_code=503,detail='诊断记录暂不可用，请查看本机运行日志') from None
    if result is None:
        raise HTTPException(status_code=404,detail='未找到诊断记录')
    return {**result,**telemetry.diagnostics_status()}


@router.get('/documents/diagnostics/jobs/{operation}/{job_id}')
def document_details(operation:Literal['upload','delete'],job_id:str):
    if len(job_id)!=32 or any(char not in '0123456789abcdef' for char in job_id):
        raise HTTPException(status_code=422,detail='任务 ID 无效')
    def read(store):
        attempts=store.document_attempts(operation,job_id)
        if not attempts:
            from .upload_jobs import upload_job_manager,delete_job_manager
            manager=upload_job_manager if operation=='upload' else delete_job_manager
            if manager.get_job(job_id) is None:
                return None
        return {'attempts':attempts,'diagnostics_available':bool(attempts)}
    return query(read)


@router.get('/documents/diagnostics/jobs/{operation}/{job_id}/events')
def document_events(operation:Literal['upload','delete'],job_id:str,
                    attempt:int|None=Query(None,ge=1),after_id:int=Query(0,ge=0),limit:int=Query(50,ge=1,le=100)):
    if len(job_id)!=32 or any(char not in '0123456789abcdef' for char in job_id):
        raise HTTPException(status_code=422,detail='任务 ID 无效')
    return query(lambda store:store.events(operation=operation,job_id=job_id,attempt=attempt,after_id=after_id,limit=limit))


@router.get('/documents/diagnostics/alerts')
def alerts(state:Literal['open','resolved','all']='open',before_id:int|None=Query(None,ge=1),
           limit:int=Query(50,ge=1,le=100),scope:Literal['document','chat','recovery','system']|None=None):
    return query(lambda store:store.alerts(state=state,before_id=before_id,limit=limit,scope=scope))


@router.post('/documents/diagnostics/alerts/{alert_id}/acknowledge')
def acknowledge(alert_id:int,admin=Depends(require_admin)):
    return query(lambda store:store.acknowledge(alert_id,admin.username))


@router.get('/documents/diagnostics/summary')
def document_summary(window:Literal['24h','7d','30d']='24h'):
    return query(lambda store:store.summary(scope='document',hours={'24h':24,'7d':168,'30d':720}[window]))


@router.get('/diagnostics/chat/traces')
def chat_traces(before_id:int|None=Query(None,ge=1),limit:int=Query(50,ge=1,le=100),
                status:Literal['pending','running','completed','failed','cancelled','interrupted']|None=None):
    return query(lambda store:store.traces(scope='chat',before_id=before_id,limit=limit,status=status))


@router.get('/diagnostics/chat/summary')
def chat_summary(window:Literal['24h','7d','30d']='24h'):
    return query(lambda store:store.summary(scope='chat',hours={'24h':24,'7d':168,'30d':720}[window]))


@router.get('/diagnostics/chat/traces/{trace_id}')
def chat_details(trace_id:str):
    def read(store):
        detail=store.trace_detail(trace_id)
        return detail if detail and detail['trace'].get('scope')=='chat' else None
    return query(read)
