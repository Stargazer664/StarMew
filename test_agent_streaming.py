import asyncio
import json
import unittest
import os
import ast
import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager, aclosing
from starlette.responses import StreamingResponse
from unittest.mock import patch, AsyncMock, Mock
from typing import Any
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
from backend import agent, tools as agent_tools
from backend.tool_contracts import ToolResult


class AsyncBoundaryModel(BaseChatModel):
    """Exercise the real graph and LangChain stream callbacks without networking."""
    calls: list = Field(default_factory=list)
    blocked: Any = None
    cancelled: Any = None
    release: Any = None
    mixed: bool = False
    route_count: int = 0

    @property
    def _llm_type(self):
        return "async-boundary"

    def bind_tools(self, tools, **kwargs):
        self.calls.append(("bind", kwargs))
        return self.bind(route=True)

    def _generate(self, *args, **kwargs):
        raise AssertionError("SSE must not use synchronous model generation")

    def _stream(self, *args, **kwargs):
        raise AssertionError("SSE must not use synchronous model streaming")
        yield

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        route = kwargs.get("route", False)
        self.calls.append("route" if route else "final")
        if route:
            self.route_count += 1
            if self.mixed and self.route_count == 1:
                yield ChatGenerationChunk(message=AIMessageChunk(content="秘密中间推理"))
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "web_search", "args": '{"query":"secret-args"}',
                    "id": "call-search", "index": 0,
                }]))
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[{
                    "name": "finish", "args": "{}", "id": "call-finish", "index": 0,
                }]))
            return
        yield ChatGenerationChunk(message=AIMessageChunk(content="最终"))
        if self.blocked is not None:
            self.blocked.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        yield ChatGenerationChunk(message=AIMessageChunk(content="答案"))


class ToolDecisionBoundaryModel(AsyncBoundaryModel):
    """Replay provider decisions through the real graph and SSE boundary."""
    decisions: list[AIMessageChunk] = Field(default_factory=list)
    final_inputs: list = Field(default_factory=list)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        if kwargs.get("route"):
            self.calls.append("route")
            response = self.decisions[self.route_count]
            self.route_count += 1
            yield ChatGenerationChunk(message=response)
            return
        self.final_inputs.append(list(messages))
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk


class FakeStreamingModel:
    def __init__(self, chunks):
        self.chunks = chunks

    def stream(self, messages, **kwargs):
        assert messages == [HumanMessage(content="你好")]
        yield from self.chunks


class KnowledgeBoundaryModel(AsyncBoundaryModel):
    repeat: bool = False
    final_gates: Any = None
    final_started: Any = None

    def decision(self, messages):
        query = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))
        if self.repeat or not any(isinstance(msg, ToolMessage) for msg in messages):
            name, args = "search_knowledge_base", json.dumps({"query": query})
        else:
            name, args = "finish", "{}"
        return AIMessageChunk(content="", tool_call_chunks=[{"name": name, "args": args, "id": "search", "index": 0}])

    def _stream(self, messages, **kwargs):
        if kwargs.get('route'):
            self.calls.append('route')
            yield ChatGenerationChunk(message=self.decision(messages))
        else:
            self.calls.append('final')
            yield ChatGenerationChunk(message=AIMessageChunk(content='安全回答'))

    async def _astream(self, messages, **kwargs):
        if kwargs.get("route"):
            self.calls.append("route")
            yield ChatGenerationChunk(message=self.decision(messages))
        else:
            query = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))
            if self.final_started:
                self.final_started[query].set()
            if self.final_gates:
                await self.final_gates[query].wait()
                agent_tools.emit_rag_step("✅", f"after-{query}")
            yield ChatGenerationChunk(message=AIMessageChunk(content="安全回答"))


@contextmanager
def local_rag_pipeline():
    """Load the real RAG graph with only external retrieval/LLM boundaries replaced."""
    retrieval = SimpleNamespace(
        retrieve_documents=Mock(return_value={"docs": [], "meta": {"candidate_k": 5, "retrieval_mode": "hybrid"}}),
        step_back_expand=Mock(return_value={"expanded_query": "sensitive-rewritten-query", "step_back_question": "sensitive-question", "step_back_answer": "sensitive-answer"}),
        generate_hypothetical_document=Mock(return_value="sensitive-hypothesis"),
    )
    path = Path(__file__).parent / "backend" / "rag_pipeline.py"
    spec = importlib.util.spec_from_file_location("backend.rag_pipeline", path)
    pipeline = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"backend.rag_utils": retrieval, "backend.rag_pipeline": pipeline}):
        spec.loader.exec_module(pipeline)
        with patch.object(pipeline, "_get_grader_model", return_value=None), patch.object(pipeline, "_get_router_model", return_value=None):
            yield pipeline


class FakeMultiToolModel:
    def __init__(self):
        self.stream_calls = 0

    def bind_tools(self, tools):
        return self

    def stream(self, messages, **kwargs):
        self.stream_calls += 1
        if self.stream_calls == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "unknown_a", "args": "{}", "id": "call-a", "index": 0},
                    {"name": "unknown_b", "args": "{}", "id": "call-b", "index": 1},
                ],
            )
            return
        yield AIMessageChunk(content="完成")


class FakeStorage:
    def __init__(self, actions=None):
        self.actions = actions if actions is not None else []
        self.saved = []

    def load(self, user_id, session_id, *, turn_identity=None):
        return []

    def load_context_state(self, user_id, session_id, *, turn_identity=None):
        return {}

    def append_turn(self, user_id, session_id, user_text, answer, *, context_state, rag_trace, request_id=None, turn_identity=None):
        agent.ConversationStorage().append_turn(user_id, session_id, user_text, answer, context_state=context_state, rag_trace=rag_trace, request_id=request_id, turn_identity=turn_identity)
        self.saved.append((user_id, session_id, [HumanMessage(content=user_text), AIMessage(content=answer)],
                           [None, {"rag_trace": rag_trace}]))
        self.actions.append("save")
        return {"user_message_id": len(self.saved) * 2 - 1,
                "assistant_message_id": len(self.saved) * 2, "message_count": len(self.saved) * 2}


class FakeMemoryService:
    def __init__(self, actions=None):
        self.actions = actions if actions is not None else []
        self.updated = []

    def get_memory_for_context(self, user_id, **kwargs):
        return ""

    def update_from_conversation(self, user_id, messages, **kwargs):
        self.updated.append((user_id, list(messages)))
        self.actions.append("memory")


class BlockingGraph:
    def __init__(self, rag_emitter=None):
        self.rag_emitter = rag_emitter
        self.started = asyncio.Event()
        self.completed = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.stream_mode = None

    async def astream(self, state, *, stream_mode=None, **kwargs):
        self.stream_mode = stream_mode
        self.started.set()
        if self.rag_emitter:
            self.rag_emitter("🔎", "检索中", "尚未完成")
        try:
            yield (
                "messages",
                (AIMessageChunk(content="早"), {"langgraph_node": "final"}),
            )
            await self.release.wait()
            self.completed.set()
            yield (
                "values",
                {"messages": [HumanMessage(content="你好"), AIMessageChunk(content="早到") ]},
            )
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class CompleteGraph:
    def __init__(self, rag_emitter=None):
        self.rag_emitter = rag_emitter
        self.completed = asyncio.Event()
        self.stream_mode = None

    async def astream(self, state, *, stream_mode=None, **kwargs):
        self.stream_mode = stream_mode
        if self.rag_emitter:
            self.rag_emitter("🔎", "检索完成", "")
        yield (
            "messages",
            (AIMessageChunk(content="完"), {"langgraph_node": "final"}),
        )
        yield (
            "messages",
            (AIMessageChunk(content="整"), {"langgraph_node": "final"}),
        )
        self.completed.set()
        yield (
            "values",
            {
                "messages": [
                    HumanMessage(content="你好"),
                    AIMessageChunk(content="完整回答"),
                ]
            },
        )


class FilteringGraph:
    def __init__(self):
        self.stream_called = False

    async def astream(self, state, *, stream_mode=None, **kwargs):
        self.stream_called = True
        yield (
            "messages",
            (AIMessageChunk(content="先出现的工具决策文本"), {"langgraph_node": "reason"}),
        )
        yield (
            "messages",
            (AIMessageChunk(content="内部输出"), {"langgraph_node": "act"}),
        )
        yield (
            "messages",
            (
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"name": "secret_tool", "args": '{"secret":"不应泄露"}'},
                    ],
                ),
                {"langgraph_node": "reason"},
            ),
        )
        yield (
            "values",
            {
                "messages": [
                    HumanMessage(content="你好"),
                    AIMessageChunk(content="可见答案"),
                ]
            },
        )


@contextmanager
def isolated_admission(storage):
    from backend import conversation_turns
    from backend.database import Base
    from backend import memory as memory_module, memory_jobs
    from backend.models import User
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    if not isinstance(storage, FakeStorage):
        with patch.object(conversation_turns, "SessionLocal", agent.SessionLocal), patch.object(memory_module, "SessionLocal", agent.SessionLocal), patch.object(memory_jobs, "SessionLocal", agent.SessionLocal):
            yield
        return
    from tempfile import TemporaryDirectory
    temp = TemporaryDirectory()
    engine = create_engine('sqlite:///' + str(Path(temp.name) / 'chat.sqlite'), connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        db.add_all([User(username=name, password_hash='test') for name in ('default_user', 'u1', 'u2', 'xin', 'A', 'B')])
        db.commit()
    try:
        with patch.object(agent, 'SessionLocal', sessions), patch.object(memory_module, 'SessionLocal', sessions), patch.object(memory_jobs, 'SessionLocal', sessions), patch.object(conversation_turns, 'SessionLocal', sessions), patch.object(agent.cache, 'delete', lambda *args: None):
            yield
    finally:
        engine.dispose()
        temp.cleanup()


@contextmanager
def patched_stream_dependencies(graph, storage, memory, rag_trace=None):
    # Import consumers before patching their source module so a first import
    # cannot permanently capture a FakeStorage/FakeMemoryService singleton.
    from backend import api
    rag_proxy = {}

    def set_queue(proxy):
        rag_proxy["queue"] = proxy
        rag_proxy["loop"] = asyncio.get_running_loop()

    def emit_rag_step(icon, label, detail=""):
        queue = rag_proxy.get("queue")
        if queue is not None:
            rag_proxy["loop"].call_soon_threadsafe(queue.put_nowait, {"icon": icon, "label": label, "detail": detail})

    rag_calls = []

    def get_last_rag_context(clear=True):
        rag_calls.append(clear)
        if len(rag_calls) == 1:
            return None
        return {"rag_trace": rag_trace} if rag_trace is not None else None

    with isolated_admission(storage), patch.object(agent, "react_graph", graph), \
         patch.object(agent, "storage", storage), \
         patch.object(agent, "memory_service", memory), \
         patch.object(agent, "set_rag_step_queue", set_queue), \
         patch.object(agent, "emit_rag_step", emit_rag_step), \
         patch.object(agent, "get_last_rag_context", get_last_rag_context), \
         patch.object(agent, "reset_tool_call_guards", lambda: None):
        yield


async def collect_sse(stream):
    events = []
    try:
        while True:
            events.append(await stream.__anext__())
    except StopAsyncIteration:
        return events


def parse_sse(events):
    parsed = []
    for event in events:
        payload = event.removeprefix("data: ").strip()
        parsed.append(payload if payload == "[DONE]" else json.loads(payload))
    return parsed


def load_stream_endpoint(stream_function):
    # Exercise the actual route and response subclass; unrelated dependencies stay lazy.
    from backend import api
    async def endpoint(request, user):
        from backend.schemas import ChatRequest
        with patch.object(api, 'chat_with_agent_stream', stream_function):
            return await api.chat_stream_endpoint(ChatRequest(message=request.message, session_id=request.session_id), user)
    return endpoint


class AgentStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_only_decision_generates_and_saves_only_final_answer(self):
        for decision in (
            AIMessageChunk(content="决策阶段的问候"),
            AIMessageChunk(content=[{"type": "text", "text": "决策阶段的问候"}]),
        ):
            with self.subTest(content=decision.content):
                fake_model = ToolDecisionBoundaryModel(decisions=[decision])
                with patch.object(agent, "init_chat_model", return_value=fake_model):
                    graph, _ = agent._create_react_agent()
                storage, memory = FakeStorage(), FakeMemoryService()
                with patched_stream_dependencies(graph, storage, memory):
                    events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream("你好")), 5))
                errors = [event for event in events if isinstance(event, dict) and event["type"] == "error"]
                self.assertEqual(errors, [])
                self.assertEqual([event["content"] for event in events if isinstance(event, dict) and event["type"] == "content"], ["最终", "答案"])
                self.assertEqual(events.count("[DONE]"), 1)
                self.assertEqual(events[-1], "[DONE]")
                self.assertEqual(fake_model.calls.count("final"), 1)
                self.assertNotIn("决策阶段的问候", str(events))
                self.assertNotIn("决策阶段的问候", str(fake_model.final_inputs))
                self.assertEqual([msg.content for msg in storage.saved[0][2]], ["你好", "最终答案"])
                self.assertEqual(memory.updated, [])

    async def test_text_only_decision_after_tool_preserves_tool_result_for_final(self):
        fake_model = ToolDecisionBoundaryModel(decisions=[
            AIMessageChunk(content="", tool_call_chunks=[{
                "name": "get_current_weather", "args": '{"location":"北京"}', "id": "weather", "index": 0,
            }]),
            AIMessageChunk(content="工具执行后直接给出的决策文字"),
        ])
        with patch.object(agent, "init_chat_model", return_value=fake_model):
            graph, _ = agent._create_react_agent()
        storage = FakeStorage()
        with patched_stream_dependencies(graph, storage, FakeMemoryService()), \
             patch.object(agent, "get_current_weather", Mock(invoke=Mock(return_value=ToolResult('ok', '北京晴，25度')))):
            events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream("北京天气")), 5))
        self.assertFalse(any(isinstance(event, dict) and event["type"] == "error" for event in events), events)
        tool_results = [msg for msg in fake_model.final_inputs[0] if isinstance(msg, ToolMessage)]
        self.assertEqual([msg.tool_call_id for msg in tool_results], ['weather'])
        self.assertEqual(json.loads(tool_results[0].content)['status'], 'ok')
        self.assertEqual(json.loads(tool_results[0].content)['content'], '北京晴，25度')
        self.assertNotIn("工具执行后直接给出的决策文字", str(fake_model.final_inputs))
        self.assertEqual(storage.saved[0][2][-1].content, "最终答案")

    async def test_empty_decision_still_errors_without_saving(self):
        for decision in (
            AIMessageChunk(content=""),
            AIMessageChunk(content=" \n\t"),
            AIMessageChunk(content=[{"type": "reasoning", "text": "仅有内部推理"}]),
        ):
            with self.subTest(decision=decision):
                fake_model = ToolDecisionBoundaryModel(decisions=[decision])
                with patch.object(agent, "init_chat_model", return_value=fake_model):
                    graph, _ = agent._create_react_agent()
                storage, memory = FakeStorage(), FakeMemoryService()
                with patched_stream_dependencies(graph, storage, memory):
                    events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream("你好")), 5))
                self.assertEqual([event if isinstance(event, str) else event["type"] for event in events][-2:], ["error", "[DONE]"])
                self.assertFalse(any(isinstance(event, dict) and event["type"] == "content" for event in events))
                self.assertEqual(fake_model.final_inputs, [])
                self.assertEqual(storage.saved, [])
                self.assertEqual(memory.updated, [])

    async def test_malformed_and_mixed_finish_calls_are_paired_before_final(self):
        malformed = {'name': 'web_search', 'args': 'not-json', 'id': 'bad', 'index': 1}
        finish = {'name': 'finish', 'args': '{}', 'id': 'finish', 'index': 0}
        for first_calls in ([malformed], [finish, malformed]):
            with self.subTest(first_calls=first_calls):
                decisions = [AIMessageChunk(content='秘密决策文字', tool_call_chunks=first_calls)]
                if len(first_calls) == 1:
                    decisions.append(AIMessageChunk(content='', tool_call_chunks=[finish]))
                fake_model = ToolDecisionBoundaryModel(decisions=decisions)
                with patch.object(agent, 'init_chat_model', return_value=fake_model):
                    graph, _ = agent._create_react_agent()
                with patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()):
                    events = parse_sse(await collect_sse(agent.chat_with_agent_stream('你好')))
                self.assertFalse(any(e.get('type') == 'error' for e in events if isinstance(e, dict)))
                messages = [m for m in fake_model.final_inputs[0] if isinstance(m, ToolMessage)]
                self.assertEqual({m.tool_call_id for m in messages}, {'bad', 'finish'})
                bad = next(m for m in messages if m.tool_call_id == 'bad')
                self.assertEqual(json.loads(bad.content)['code'], 'INVALID_ARGUMENT')
                self.assertNotIn('秘密决策文字', str(events))

    async def test_long_invalid_arguments_cannot_reach_next_model_or_public_stream(self):
        malformed = {'name': 'web_search', 'args': 'not-json' * 1250, 'id': 'bad', 'index': 0}
        fake_model = ToolDecisionBoundaryModel(decisions=[
            AIMessageChunk(content='秘密决策文字', tool_call_chunks=[malformed]),
            AIMessageChunk(content='', tool_call_chunks=[{'name': 'finish', 'args': '{}', 'id': 'finish', 'index': 0}]),
        ])
        with patch.object(agent, 'init_chat_model', return_value=fake_model):
            graph, _ = agent._create_react_agent()
        storage = FakeStorage()
        with patched_stream_dependencies(graph, storage, FakeMemoryService()):
            events = parse_sse(await collect_sse(agent.chat_with_agent_stream('current question')))
        self.assertEqual(fake_model.route_count, 1)
        self.assertEqual(fake_model.final_inputs, [])
        self.assertTrue(any(e.get('type') == 'error' for e in events if isinstance(e, dict)))
        self.assertEqual(events[-1], '[DONE]')
        self.assertNotIn('not-json', str(events))
        self.assertNotIn('秘密决策文字', str(events))
        self.assertEqual(storage.saved, [])

    async def test_real_search_pipeline_does_not_expose_query_or_rewrite_in_sse(self):
        fake_model = KnowledgeBoundaryModel()
        storage = FakeStorage()
        with patch.object(agent, "init_chat_model", return_value=fake_model):
            graph, _ = agent._create_react_agent()
        with local_rag_pipeline(), patch.object(agent, "react_graph", graph), \
             isolated_admission(storage), patch.object(agent, "storage", storage), patch.object(agent, "memory_service", FakeMemoryService()):
            events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream("sensitive-original-query")), 5))
        for secret in ("sensitive-original-query", "sensitive-rewritten-query", "sensitive-question", "sensitive-answer"):
            self.assertNotIn(secret, str(events))
        self.assertTrue(any(isinstance(event, dict) and event.get("type") == "rag_step" and "扩展检索完成" in event["step"]["label"] for event in events))
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(storage.saved[0][3][-1]["rag_trace"]["query"], "sensitive-original-query")
        self.assertEqual(storage.saved[0][3][-1]["rag_trace"]["expanded_query"], "sensitive-rewritten-query")

    async def test_pipeline_unexpected_grader_text_cannot_leak_as_progress_score(self):
        with patch.object(agent, "init_chat_model", return_value=KnowledgeBoundaryModel()):
            graph, _ = agent._create_react_agent()
        grader = Mock()
        grader.with_structured_output.return_value.invoke.return_value = SimpleNamespace(binary_score="sensitive-grader-echo")
        with local_rag_pipeline() as pipeline, patch.object(pipeline, "_get_grader_model", return_value=grader), \
             patch.object(agent, "react_graph", graph), isolated_admission(FakeStorage()), patch.object(agent, "storage", FakeStorage()), \
             patch.object(agent, "memory_service", FakeMemoryService()):
            events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream("question")), 5))
        self.assertNotIn("sensitive-grader-echo", str(events))
        self.assertEqual(events[-1], "[DONE]")

    async def test_concurrent_real_tool_threads_keep_steps_traces_and_cleanup_isolated(self):
        gates = {key: asyncio.Event() for key in ("A", "B")}
        started = {key: asyncio.Event() for key in ("A", "B")}
        fake_model = KnowledgeBoundaryModel(final_gates=gates, final_started=started)
        with patch.object(agent, "init_chat_model", return_value=fake_model):
            graph, _ = agent._create_react_agent()
        both_tools = threading.Barrier(2)
        threads = []

        def pipeline_run(query):
            threads.append(threading.get_ident())
            both_tools.wait(timeout=3)
            agent_tools.emit_rag_step("🔎", f"only-{query}")
            return {"docs": [], "rag_trace": {"request": query}}

        with local_rag_pipeline() as pipeline, patch.object(pipeline, "run_rag_graph", side_effect=pipeline_run), \
             patch.object(agent, "react_graph", graph), isolated_admission(FakeStorage()), patch.object(agent, "storage", FakeStorage()), \
             patch.object(agent, "memory_service", FakeMemoryService()):
            requests = {key: asyncio.create_task(collect_sse(agent.chat_with_agent_stream(key, key, key))) for key in ("A", "B")}
            try:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 5)
                # Both tool traces have been written, then A ends while B remains active.
                gates["A"].set()
                events_a = parse_sse(await asyncio.wait_for(requests["A"], 5))
                gates["B"].set()
                events_b = parse_sse(await asyncio.wait_for(requests["B"], 5))
            finally:
                for gate in gates.values():
                    gate.set()
                for task in requests.values():
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*requests.values(), return_exceptions=True)
        self.assertEqual(len(threads), 2)
        self.assertTrue(all(thread_id != threading.get_ident() for thread_id in threads))
        for key, events in (("A", events_a), ("B", events_b)):
            traces = [e["rag_trace"] for e in events if isinstance(e, dict) and e.get("type") == "trace"]
            self.assertEqual([trace['request'] for trace in traces], [key])
            self.assertEqual(traces[0]['retrieval_rounds'], [{'request': key}])
            labels = [e["step"]["label"] for e in events if isinstance(e, dict) and e.get("type") == "rag_step"]
            self.assertIn(f"only-{key}", labels)
            self.assertIn(f"after-{key}", labels)
            self.assertNotIn(f"only-{'B' if key == 'A' else 'A'}", labels)

    async def test_repeated_tool_loop_finishes_with_one_done_and_save(self):
        fake_model = KnowledgeBoundaryModel(repeat=True)
        with patch.object(agent, "init_chat_model", return_value=fake_model):
            graph, _ = agent._create_react_agent()
        storage = FakeStorage()
        with local_rag_pipeline() as pipeline, patch.object(agent, "react_graph", graph), \
             isolated_admission(storage), patch.object(agent, "storage", storage), patch.object(agent, "memory_service", FakeMemoryService()):
            try:
                events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream("repeat")), 5))
            except TimeoutError:
                self.fail(f"Unbounded tool loop: {fake_model.calls.count('route')} model calls within 5 seconds")
        self.assertEqual(fake_model.calls.count("route"), 2)
        self.assertEqual(pipeline.retrieve_documents.call_count, 2)  # one initial + one expanded search for this request
        self.assertFalse(any(isinstance(e, dict) and e["type"] == "error" for e in events))
        self.assertEqual(events.count("[DONE]"), 1)
        self.assertEqual(len(storage.saved), 1)

    def test_sync_repeated_tool_loop_finishes_and_saves(self):
        fake_model = KnowledgeBoundaryModel(repeat=True)
        with patch.object(agent, "init_chat_model", return_value=fake_model):
            graph, _ = agent._create_react_agent()
        storage = FakeStorage()
        with local_rag_pipeline() as pipeline, patch.object(agent, "react_graph", graph), \
             isolated_admission(storage), patch.object(agent, "storage", storage), patch.object(agent, "memory_service", FakeMemoryService()):
            self.assertEqual(agent.chat_with_agent("repeat")["response"], "安全回答")
        self.assertEqual(fake_model.calls.count("route"), 2)
        self.assertEqual(pipeline.retrieve_documents.call_count, 2)
        self.assertEqual(len(storage.saved), 1)

    def test_sync_nonempty_memory_and_summary_do_not_pollute_saved_history(self):
        for history_length in (0, 51):
            with self.subTest(history_length=history_length):
                fake_model = FakeMultiToolModel()
                with patch.object(agent, "init_chat_model", return_value=fake_model):
                    graph, _ = agent._create_react_agent()
                storage, memory = FakeStorage(), FakeMemoryService()
                history = [HumanMessage(content=f"old-{n}-" + "x" * 100) for n in range(history_length)]
                storage.load = lambda *args, **kwargs: list(history)
                memory.get_memory_for_context = lambda *args, **kwargs: "内部长期记忆"
                with patched_stream_dependencies(graph, storage, memory), patch.object(agent, "summarize_old_messages", return_value="内部摘要"):
                    agent.chat_with_agent("你好")
                self.assertEqual([m.content for m in storage.saved[0][2]], ["你好", "完成"])
                self.assertEqual(memory.updated, [])

    def test_sync_chat_contract_returns_answer_trace_and_persists(self):
        with patch.object(agent, "init_chat_model", return_value=FakeMultiToolModel()):
            graph, _ = agent._create_react_agent()
        storage, memory = FakeStorage(), FakeMemoryService()
        trace = {"steps": []}
        with patched_stream_dependencies(graph, storage, memory, rag_trace=trace):
            result = agent.chat_with_agent("你好", "u1", "s1")
        self.assertEqual(result, {"response": "完成", "rag_trace": trace})
        self.assertEqual([msg.content for msg in storage.saved[0][2]], ["你好", "完成"])

    async def test_extraction_failure_is_deferred_and_answer_has_durable_job(self):
        storage, memory = FakeStorage(), FakeMemoryService()
        memory.update_from_conversation = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("memory write failed"))
        with patched_stream_dependencies(CompleteGraph(), storage, memory):
            events = parse_sse(await collect_sse(agent.chat_with_agent_stream("你好")))
            from backend.models import MemoryUpdateJob
            with agent.SessionLocal() as db:
                self.assertEqual(db.query(MemoryUpdateJob).one().status, 'pending')
        self.assertFalse(any(isinstance(e, dict) and e['type'] == 'error' for e in events))
        self.assertEqual(events.count("[DONE]"), 1)
        self.assertEqual(len(storage.saved), 1)

    async def test_cancellation_while_awaiting_next_event_skips_persistence(self):
        graph = BlockingGraph()
        storage, memory = FakeStorage(), FakeMemoryService()
        with patched_stream_dependencies(graph, storage, memory):
            stream = agent.chat_with_agent_stream("你好")
            await stream.__anext__()
            await stream.__anext__()
            pending = asyncio.create_task(stream.__anext__())
            await asyncio.sleep(0)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertTrue(graph.cancelled.is_set())
        self.assertEqual(storage.saved, [])
        self.assertEqual(memory.updated, [])

    async def test_http_wrapper_close_immediately_cancels_underlying_graph(self):
        graph = BlockingGraph()
        storage, memory = FakeStorage(), FakeMemoryService()
        with patched_stream_dependencies(graph, storage, memory):
            endpoint = load_stream_endpoint(agent.chat_with_agent_stream)
            response = await endpoint(SimpleNamespace(message="你好", session_id="s1"), SimpleNamespace(username="u1"))
            await response.body_iterator.__anext__()
            await response.body_iterator.__anext__()
            await response.body_iterator.aclose()
            self.assertTrue(graph.cancelled.is_set(), "HTTP wrapper did not close the agent generator")
        self.assertEqual(storage.saved, [])

    async def test_http_wrapper_fallback_error_also_ends_protocol(self):
        async def failed_stream(*args, **kwargs):
            raise RuntimeError("startup failed")
            yield
        endpoint = load_stream_endpoint(failed_stream)
        with isolated_admission(FakeStorage()):
            response = await endpoint(SimpleNamespace(message="你好", session_id="s1"), SimpleNamespace(username="u1"))
            events = parse_sse(await collect_sse(response.body_iterator))
            response.admission.close()
        self.assertEqual([e if isinstance(e, str) else e["type"] for e in events], ["error", "[DONE]"])

    async def test_done_is_yielded_after_save_and_durable_job_even_if_closed_immediately(self):
        actions = []
        storage, memory = FakeStorage(actions), FakeMemoryService(actions)
        with patched_stream_dependencies(CompleteGraph(), storage, memory):
            stream = agent.chat_with_agent_stream("你好", "u1", "s1")
            try:
                async for event in stream:
                    if event == "data: [DONE]\n\n":
                        self.assertEqual(actions, ["save"])
                        from backend.models import MemoryUpdateJob
                        with agent.SessionLocal() as db:
                            self.assertEqual(db.query(MemoryUpdateJob).one().target_message_count, 2)
                        break
            finally:
                await stream.aclose()

    async def test_initialization_failures_emit_error_done_and_clear_queue(self):
        for target in ("load", "memory", "summary"):
            with self.subTest(target=target):
                storage, memory = FakeStorage(), FakeMemoryService()
                changes = []
                with patched_stream_dependencies(CompleteGraph(), storage, memory), \
                     patch.object(agent, "set_rag_step_queue", changes.append):
                    if target == "load":
                        storage.load = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed"))
                    elif target == "memory":
                        memory.get_memory_for_context = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("memory failed"))
                    else:
                        storage.load = lambda *args, **kwargs: [HumanMessage(content="old" * 1500)] * 2
                    with patch.object(agent, "summarize_old_messages", side_effect=RuntimeError("summary failed")), \
                         patch.object(agent, "model", Mock(ainvoke=AsyncMock(side_effect=RuntimeError("summary failed")))):
                        try:
                            events = parse_sse(await collect_sse(agent.chat_with_agent_stream("你好")))
                        except RuntimeError as exc:
                            self.fail(f"Initialization exception escaped SSE protocol: {exc}")
                self.assertEqual([e if isinstance(e, str) else e["type"] for e in events][-2:], ["error", "[DONE]"])
                self.assertIsNone(changes[-1])
                self.assertEqual(storage.saved, [])

    async def test_save_failure_emits_error_then_single_done(self):
        storage, memory = FakeStorage(), FakeMemoryService()
        storage.append_turn = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("save failed"))
        with patched_stream_dependencies(CompleteGraph(), storage, memory):
            try:
                events = parse_sse(await collect_sse(agent.chat_with_agent_stream("你好")))
            except RuntimeError as exc:
                self.fail(f"Save exception escaped SSE protocol: {exc}")
        self.assertEqual([e if isinstance(e, str) else e["type"] for e in events][-2:], ["error", "[DONE]"])
        self.assertEqual(events.count("[DONE]"), 1)
        self.assertEqual(memory.updated, [])

    async def test_real_async_model_is_cancelled_and_answer_is_not_saved(self):
        fake_model = AsyncBoundaryModel(blocked=asyncio.Event(), cancelled=asyncio.Event(), release=asyncio.Event())
        with patch.object(agent, "init_chat_model", return_value=fake_model):
            graph, _ = agent._create_react_agent()
        storage, memory = FakeStorage(), FakeMemoryService()
        with patched_stream_dependencies(graph, storage, memory):
            stream = agent.chat_with_agent_stream("你好")
            try:
                async for event in stream:
                    payload = parse_sse([event])[0]
                    if isinstance(payload, dict) and payload.get("type") == "content":
                        self.assertEqual(payload["content"], "最终")
                        break
                self.assertTrue(fake_model.blocked.is_set(), "Async model never reached its suspended stream")
            finally:
                await stream.aclose()
        self.assertTrue(fake_model.cancelled.is_set(), "Cancellation never reached underlying async model")
        self.assertEqual(storage.saved, [])
        self.assertEqual(memory.updated, [])

    async def test_mixed_text_then_tool_call_never_leaks_and_final_generated_once(self):
        fake_model = AsyncBoundaryModel(mixed=True)
        with patch.object(agent, "init_chat_model", return_value=fake_model):
            graph, _ = agent._create_react_agent()
        with patched_stream_dependencies(graph, FakeStorage(), FakeMemoryService()), \
             patch.object(agent, "web_search", Mock(invoke=Mock(return_value=ToolResult('ok', 'secret-result')))):
            events = parse_sse(await asyncio.wait_for(collect_sse(agent.chat_with_agent_stream("你好")), timeout=5))
        content = [event["content"] for event in events if isinstance(event, dict) and event["type"] == "content"]
        self.assertEqual(content, ["最终", "答案"])
        self.assertNotIn("秘密中间推理", str(events))
        self.assertNotIn("secret-args", str(events))
        self.assertNotIn("secret-result", str(events))
        self.assertEqual(fake_model.calls.count("final"), 1)

    def test_act_steps_do_not_include_tool_arguments_or_results(self):
        fake_model = FakeMultiToolModel()
        steps = []
        with patch.object(agent, "init_chat_model", return_value=fake_model), \
             patch.object(agent, "emit_rag_step", side_effect=lambda *args: steps.append(args)):
            graph, _ = agent._create_react_agent()
            graph.invoke({"messages": [HumanMessage(content="你好")], "step_count": 0, "memory_context": "", "tool_output": None})
        self.assertTrue(steps)
        self.assertTrue(all(len(step) < 3 or step[2] == "" for step in steps), steps)

    def test_sync_stream_model_response_aggregates_message(self):
        model = FakeStreamingModel([
            AIMessageChunk(content="你"),
            AIMessageChunk(content="好"),
        ])

        result = agent._stream_model_response(
            model,
            [HumanMessage(content="你好")],
        )

        self.assertEqual(result.content, "你好")

    def test_stream_model_response_keeps_tool_call_chunks_for_react_loop(self):
        model = FakeStreamingModel([
            AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "search_knowledge_base",
                    "args": '{"query":',
                    "id": "call-1",
                    "index": 0,
                }],
            ),
            AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": None,
                    "args": '"星喵"}',
                    "id": None,
                    "index": 0,
                }],
            ),
        ])

        result = agent._stream_model_response(model, [HumanMessage(content="你好")])

        self.assertEqual(result.tool_calls[0]["name"], "search_knowledge_base")
        self.assertEqual(result.tool_calls[0]["args"], {"query": "星喵"})

    def test_react_agent_returns_one_tool_message_for_each_tool_call(self):
        fake_model = FakeMultiToolModel()
        original_init = agent.init_chat_model
        original_emit = agent.emit_rag_step
        agent.init_chat_model = lambda **kwargs: fake_model
        agent.emit_rag_step = lambda *args, **kwargs: None
        try:
            graph, _ = agent._create_react_agent()
            result = graph.invoke({
                "messages": [HumanMessage(content="请同时处理两个动作")],
                "step_count": 0,
                "tool_output": None,
                "memory_context": "",
            })
        finally:
            agent.init_chat_model = original_init
            agent.emit_rag_step = original_emit

        tool_messages = [
            message for message in result["messages"] if isinstance(message, ToolMessage)
        ]
        self.assertEqual(
            [message.tool_call_id for message in tool_messages],
            ["call-a", "call-b"],
        )

    async def test_first_content_arrives_before_graph_finishes(self):
        graph = BlockingGraph()
        storage = FakeStorage()
        memory = FakeMemoryService()
        with patched_stream_dependencies(graph, storage, memory):
            stream = agent.chat_with_agent_stream("你好", "u1", "s1")
            first = await stream.__anext__()
            content = await stream.__anext__()

            self.assertEqual(json.loads(content[6:])["content"], "早")
            self.assertFalse(graph.completed.is_set())
            self.assertEqual(graph.stream_mode, ["messages", "values"])

            graph.release.set()
            await collect_sse(stream)

    async def test_rag_step_arrives_before_graph_finishes(self):
        graph = BlockingGraph(rag_emitter=lambda *args: agent.emit_rag_step(*args))
        storage = FakeStorage()
        memory = FakeMemoryService()
        with patched_stream_dependencies(graph, storage, memory):
            stream = agent.chat_with_agent_stream("你好", "u1", "s1")
            await stream.__anext__()
            for _ in range(3):
                rag_event = await asyncio.wait_for(stream.__anext__(), timeout=1)
                payload = json.loads(rag_event[6:])
                if payload.get("type") == "rag_step":
                    break
            self.assertEqual(payload["type"], "rag_step")
            self.assertEqual(payload["step"]["label"], "检索中")
            self.assertFalse(graph.completed.is_set())

            graph.release.set()
            await collect_sse(stream)

    async def test_cancelled_generator_cancels_graph_and_skips_persistence(self):
        graph = BlockingGraph()
        storage = FakeStorage()
        memory = FakeMemoryService()
        with patched_stream_dependencies(graph, storage, memory):
            stream = agent.chat_with_agent_stream("你好", "u1", "s1")
            await stream.__anext__()
            await stream.__anext__()

            await stream.aclose()
            await asyncio.wait_for(graph.cancelled.wait(), timeout=1)

        self.assertEqual(storage.saved, [])
        self.assertEqual(memory.updated, [])

    async def test_normal_completion_uses_final_values_and_preserves_trace_done_protocol(self):
        actions = []
        graph = CompleteGraph(rag_emitter=lambda *args: None)
        storage = FakeStorage(actions)
        memory = FakeMemoryService(actions)
        trace = {"steps": [{"label": "检索完成"}]}
        with patched_stream_dependencies(graph, storage, memory, rag_trace=trace):
            events = await collect_sse(
                agent.chat_with_agent_stream("你好", "u1", "s1")
            )

        parsed = parse_sse(events)
        self.assertEqual("".join(item["content"] for item in parsed if isinstance(item, dict) and item["type"] == "content"), "完整")
        self.assertEqual({key: value for key, value in parsed[-2].items() if key != 'run_id'}, {"type": "trace", "rag_trace": trace})
        self.assertTrue(parsed[-2]['run_id'])
        self.assertEqual(parsed[-1], "[DONE]")
        self.assertEqual(actions, ["save"])
        saved_messages = storage.saved[0][2]
        self.assertEqual(saved_messages[-2].content, "你好")
        self.assertEqual(saved_messages[-1].content, "完整回答")

    async def test_only_final_node_text_chunks_become_content(self):
        graph = FilteringGraph()
        storage = FakeStorage()
        memory = FakeMemoryService()
        with patched_stream_dependencies(graph, storage, memory):
            events = await collect_sse(
                agent.chat_with_agent_stream("你好", "u1", "s1")
            )

        self.assertTrue(graph.stream_called)
        parsed = parse_sse(events)
        content = [
            item["content"]
            for item in parsed
            if isinstance(item, dict) and item.get("type") == "content"
        ]
        self.assertEqual(content, [])
