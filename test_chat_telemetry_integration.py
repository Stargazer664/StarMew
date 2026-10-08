import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend import telemetry
from backend.chat_diagnostics import request_metrics
from backend.tool_contracts import ToolResult
from test_agent_streaming import AsyncBoundaryModel, KnowledgeBoundaryModel, FakeMultiToolModel, FakeMemoryService, FakeStorage, agent, patched_stream_dependencies, isolated_admission


@pytest.fixture(autouse=True)
def disable_relevance_thresholds(monkeypatch):
    """Keep legacy telemetry cases independent of local threshold settings."""
    monkeypatch.setenv('LLM_RERANK_MIN_SCORE', '')
    monkeypatch.setenv('RERANK_MIN_SCORE', '')


@pytest.fixture
def diagnostics(tmp_path):
    telemetry.start_diagnostics(tmp_path / 'jobs.db', tmp_path / 'logs')
    yield telemetry
    telemetry.stop_diagnostics()


def test_real_graph_generation_success_save_failure_stays_separate(diagnostics):
    class BrokenStorage(FakeStorage):
        def append_turn(self, *args, **kwargs):
            raise RuntimeError('private db contents')
    async def run():
        model = AsyncBoundaryModel()
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        with telemetry.bind_context(trace_id='save-failure'), patched_stream_dependencies(graph, BrokenStorage(), FakeMemoryService()):
            events = [event async for event in agent.chat_with_agent_stream('private question')]
        assert '"type": "content"' in ''.join(events)
        assert '"type": "error"' in ''.join(events)
        assert events[-1] == 'data: [DONE]\n\n'
        assert telemetry.current_trace() is None
    asyncio.run(run())
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('save-failure')
    assert detail['trace']['status'] == 'failed'
    assert detail['trace']['answer_first_content_ms'] >= 0
    calls = [item for item in detail['spans'] if item.get('component') == 'model']
    assert [item['purpose'] for item in calls] == ['tool_decision', 'final_answer']
    assert all(item['status'] == 'completed' for item in calls)
    assert calls[0].get('first_answer_ms') is None
    assert calls[1]['first_answer_ms'] >= 0
    assert next(item for item in detail['spans'] if item['stage'] == 'conversation_save')['status'] == 'failed'
    assert 'private' not in json.dumps(detail)


def test_hidden_decision_first_content_is_not_public_answer(diagnostics):
    async def run():
        model = AsyncBoundaryModel(mixed=True)
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        with telemetry.bind_context(trace_id='hidden-decision'), patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()), patch.object(agent, 'web_search', SimpleNamespace(invoke=lambda inputs: ToolResult('ok', 'safe tool result'))):
            events = [event async for event in agent.chat_with_agent_stream('private')]
        assert 'secret-args' not in ''.join(events)
    asyncio.run(run())
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('hidden-decision')
    calls = [s for s in detail['spans'] if s.get('component') == 'model']
    assert [s['purpose'] for s in calls] == ['tool_decision', 'tool_decision', 'final_answer']
    assert calls[0]['first_content_ms'] >= 0
    assert calls[0].get('first_answer_ms') is None
    assert calls[2]['first_answer_ms'] >= 0
    assert detail['trace']['answer_first_content_ms'] >= 0
    assert 'secret-args' not in json.dumps(detail)


def test_cancelled_real_model_stream_is_not_complete_latency(diagnostics):
    async def run():
        model = AsyncBoundaryModel(blocked=asyncio.Event(), cancelled=asyncio.Event(), release=asyncio.Event())
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        storage = FakeStorage()
        with telemetry.bind_context(trace_id='cancelled-stream'), patched_stream_dependencies(graph, storage, FakeMemoryService()):
            stream = agent.chat_with_agent_stream('private')
            async for event in stream:
                if '"type": "content"' in event:
                    break
            await stream.aclose()
            assert model.cancelled.is_set()
            assert not storage.saved
    asyncio.run(run())
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('cancelled-stream')
    assert detail['trace']['status'] == 'cancelled'
    call = next(item for item in detail['spans'] if item.get('purpose') == 'final_answer')
    assert call['status'] == 'cancelled'
    assert call.get('generation_total_ms') is None
    assert call.get('answer_stream_ms') is None


@pytest.mark.parametrize('hybrid_fails', [True, False])
def test_hybrid_failure_dense_fallback_keeps_actual_calls_and_parentage(diagnostics, hybrid_fails):
    from backend import rag_utils
    class Embeddings:
        def get_embeddings(self, texts):
            return [[1.0]]
        def get_sparse_embedding(self, text):
            return {1: 1.0}
    class Milvus:
        calls = []
        def hybrid_retrieve(self, **kwargs):
            self.calls.append('hybrid')
            if hybrid_fails:
                raise RuntimeError('private query')
            return [{'text': 'private result'}]
        def dense_retrieve(self, **kwargs):
            self.calls.append('dense')
            return [{'text': 'private result'}]
    milvus = Milvus()
    with patch.object(rag_utils, '_embedding_service', Embeddings()), patch.object(rag_utils, '_milvus_manager', milvus), patch.object(rag_utils, 'assert_index_available', lambda: None), patch.object(rag_utils, 'RERANK_MODEL', None):
        with diagnostics.trace('chat', 'answer') as execution, request_metrics():
            result = rag_utils.retrieve_documents('private')
    assert result['meta']['retrieval_mode'] == ('dense_fallback' if hybrid_fails else 'hybrid')
    assert milvus.calls == (['hybrid', 'dense'] if hybrid_fails else ['hybrid'])
    diagnostics.flush()
    spans = diagnostics.get_store().trace_detail(execution.id)['spans']
    calls = [s for s in spans if s['stage'] == 'milvus_recall']
    assert [s['status'] for s in calls] == (['failed', 'completed'] if hybrid_fails else ['completed'])
    if hybrid_fails:
        assert calls[1]['fallback_of'] == calls[0]['id']
        assert calls[0].get('candidate_count') is None
    assert all(s['top_k'] == 30 for s in calls)
    assert calls[-1]['candidate_count'] == 1
    assert all(s['retrieval_round'] == 1 for s in calls)
    retrieval = next(s for s in spans if s['stage'] == 'retrieval')
    assert retrieval['status'] == ('degraded' if hybrid_fails else 'completed')
    assert all(s['parent_span_id'] == retrieval['id'] for s in calls)
    assert next(s for s in spans if s['stage'] == 'rerank')['status'] == 'skipped'


@pytest.mark.parametrize('mode', ['external', 'llm', 'skipped'])
def test_rerank_records_mode_and_actual_input_output_counts(diagnostics, mode):
    from backend import rag_utils
    calls = []
    def llm_invoke(prompt, **kwargs):
        calls.append('llm')
        return SimpleNamespace(content='[{"doc_id": 2, "score": 9}]')
    def external_post(*args, **kwargs):
        calls.append('external')
        return SimpleNamespace(status_code=200, json=lambda: {'results': [{'index': 1, 'relevance_score': 0.9}]})
    docs = [] if mode == 'skipped' else [{'text': str(i)} for i in range(3)]
    with patch.object(rag_utils, 'RERANK_MODEL', 'rank-model' if mode == 'external' else None), patch.object(rag_utils, 'RERANK_BINDING_HOST', 'https://example.invalid'), patch.object(rag_utils, 'RERANK_API_KEY', 'test'), patch.object(rag_utils.requests, 'post', external_post), patch.object(rag_utils, '_get_stepback_model', return_value=SimpleNamespace(model_name='real-chat-model', invoke=llm_invoke)):
        with diagnostics.trace('chat', 'answer') as execution:
            output, _ = rag_utils._rerank_documents('private', docs, 1)
    assert calls == ([] if mode == 'skipped' else [mode])
    diagnostics.flush()
    spans = diagnostics.get_store().trace_detail(execution.id)['spans']
    parent = next(s for s in spans if s['stage'] == 'rerank')
    assert parent['mode'] == mode
    assert parent['input_count'] == (0 if mode == 'skipped' else 3)
    assert parent['output_count'] == len(output) == (0 if mode == 'skipped' else 1)
    if mode == 'external':
        assert parent['model'] == 'rank-model'
    elif mode == 'llm':
        assert next(s for s in spans if s.get('purpose') == 'rerank')['model'] == 'real-chat-model'


def test_embedding_initialization_only_on_cold_call(diagnostics, tmp_path):
    from backend.embedding import EmbeddingService
    embedder = SimpleNamespace(embed_documents=lambda texts: [[1.0] for _ in texts])
    service = EmbeddingService(tmp_path / 'bm25.json')
    with patch('backend.embedding._create_dense_embedder', return_value=embedder):
        with diagnostics.trace('chat', 'answer') as execution:
            assert service.get_embeddings(['private']) == [[1.0]]
            assert service.get_embeddings(['private']) == [[1.0]]
    diagnostics.flush()
    spans = diagnostics.get_store().trace_detail(execution.id)['spans']
    dense = [s for s in spans if s['stage'] == 'embedding_dense']
    init = [s for s in spans if s['stage'] == 'initialization']
    assert [s['cold_start'] for s in dense] == [True, False]
    assert len(init) == 1 and init[0]['parent_span_id'] == dense[0]['id']


def test_cancelled_request_does_not_pretend_tool_thread_stopped(diagnostics):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def blocking_tool(inputs):
        with diagnostics.span('retrieval', component='rag'):
            entered.set()
            assert release.wait(5)
        finished.set()
        return ToolResult('ok', 'safe')
    async def run():
        model = KnowledgeBoundaryModel()
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        with telemetry.bind_context(trace_id='cancelled-thread'), patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()), patch.object(agent, 'search_knowledge_base', SimpleNamespace(invoke=blocking_tool)):
            stream = agent.chat_with_agent_stream('private')
            async def consume():
                async for _ in stream:
                    pass
            task = asyncio.create_task(consume())
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                task.cancel()
                await asyncio.sleep(0.03)
                assert not task.done()  # Admission remains owned while real tool runs.
                diagnostics.flush()
                detail = diagnostics.get_store().trace_detail('cancelled-thread')
                assert detail['trace']['status'] == 'running'
                assert next(s for s in detail['spans'] if s['stage'] == 'retrieval')['status'] == 'running'
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await stream.aclose()
                assert await asyncio.to_thread(finished.wait, 3)
    asyncio.run(run())
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('cancelled-thread')
    assert detail['trace']['status'] == 'cancelled'
    assert next(s for s in detail['spans'] if s['stage'] == 'retrieval')['status'] == 'completed'


def test_sync_final_model_call_is_generation_without_public_sse_metric(diagnostics):
    with patch.object(agent, 'init_chat_model', return_value=FakeMultiToolModel()):
        graph, _ = agent._create_react_agent()
    with telemetry.bind_context(trace_id='sync-answer'), patch.object(agent, 'react_graph', graph), isolated_admission(FakeStorage()), patch.object(agent, 'storage', FakeStorage()), patch.object(agent, 'memory_service', FakeMemoryService()):
        assert agent.chat_with_agent('private')['response'] == '完成'
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('sync-answer')
    call = next(s for s in detail['spans'] if s.get('purpose') == 'final_answer')
    assert call['stage'] == 'generation'
    assert detail['trace'].get('answer_first_content_ms') is None


def test_request_uuid_reaches_actual_diagnostic_trace_and_model_spans(diagnostics):
    from uuid import uuid4
    request_id = str(uuid4())
    async def run():
        model = AsyncBoundaryModel()
        with patch.object(agent, 'init_chat_model', return_value=model):
            graph, _ = agent._create_react_agent()
        with telemetry.bind_context(trace_id='request-uuid'), patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()):
            events = [event async for event in agent.chat_with_agent_stream('private', request_id=request_id)]
        assert events[-1] == 'data: [DONE]\n\n'
    asyncio.run(run())
    diagnostics.flush()
    detail = diagnostics.get_store().trace_detail('request-uuid')
    assert detail['trace']['request_id'] == request_id
    assert all(span['request_id'] == request_id for span in detail['spans'])
