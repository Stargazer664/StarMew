from types import SimpleNamespace
from unittest.mock import patch
import asyncio

from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage

from test_agent_streaming import agent, FakeMemoryService, FakeStorage, patched_stream_dependencies
from test_agent_streaming import CompleteGraph, collect_sse, parse_sse


class RecordingModel:
    def __init__(self):
        self.inputs = []

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages, **kwargs):
        self.inputs.append(messages)
        yield AIMessageChunk(content="完成")


class HistoryStorage(FakeStorage):
    def __init__(self, history):
        super().__init__()
        self.history = list(history)
        self.context_state = {}

    def load(self, *args, **kwargs):
        return list(self.history)

    def load_context_state(self, *args, **kwargs):
        return dict(self.context_state)

    def append_turn(self, user, session, user_text, answer, *, context_state, rag_trace, request_id=None, turn_identity=None):
        result = super().append_turn(user, session, user_text, answer, context_state=context_state, rag_trace=rag_trace, request_id=request_id, turn_identity=turn_identity)
        self.history.extend([HumanMessage(content=user_text), AIMessage(content=answer)])
        self.context_state = dict(context_state)
        return {**result, "message_count": len(self.history)}


def test_long_term_memory_is_injected_once_and_sync_defers_extraction():
    model, storage, memory = RecordingModel(), HistoryStorage([]), FakeMemoryService()
    memory.get_memory_for_context = lambda *args, **kwargs: "UNIQUE_USER_MEMORY"
    with patch.object(agent, "init_chat_model", return_value=model):
        graph, _ = agent._create_react_agent()
    with patched_stream_dependencies(graph, storage, memory):
        agent.chat_with_agent("hello", "xin", "one")
    assert sum(str(m.content).count("UNIQUE_USER_MEMORY") for m in model.inputs[0]) == 1
    assert memory.updated == []
    assert [m.type for m in storage.history] == ["human", "ai"]


def test_summary_checkpoint_is_reused_and_only_new_prefix_is_summarized(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_MAX_TOKENS", "100")
    model = RecordingModel()
    history = [HumanMessage(content="OLD_PREFIX" * 100), AIMessage(content="OLD_REPLY" * 100)]
    storage, memory, prompts = HistoryStorage(history), FakeMemoryService(), []
    def summarize(model, prompt, **kwargs):
        prompts.append(prompt)
        return SimpleNamespace(content="saved summary")
    with patch.object(agent, "init_chat_model", return_value=model):
        graph, _ = agent._create_react_agent()
    with patched_stream_dependencies(graph, storage, memory), patch.object(agent, "invoke_model", summarize):
        agent.chat_with_agent("one", "xin", "session")
        first_count = len(prompts)
        assert first_count > 0
        agent.chat_with_agent("two", "xin", "session")
        assert len(prompts) == first_count
        storage.history.extend([HumanMessage(content="NEW_PREFIX" * 100), AIMessage(content="NEW_REPLY" * 100)])
        agent.chat_with_agent("three", "xin", "session")
    assert len(prompts) > first_count
    assert all("OLD_PREFIX" not in p and "OLD_REPLY" not in p for p in prompts[first_count:])
    assert any("NEW_PREFIX" in p for p in prompts[first_count:])
    assert storage.history[0].content == history[0].content


def test_streaming_path_reuses_persisted_summary_and_preserves_complete_history(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_MAX_TOKENS", "100")
    old = [HumanMessage(content="old" * 300), AIMessage(content="reply" * 100)]
    storage, memory, prompts = HistoryStorage(old), FakeMemoryService(), []
    async def summarize(model, prompt, **kwargs):
        prompts.append(prompt)
        return SimpleNamespace(content="rolling summary")
    async def run():
        with patched_stream_dependencies(CompleteGraph(), storage, memory), patch.object(agent, "ainvoke_model", summarize):
            first = parse_sse(await collect_sse(agent.chat_with_agent_stream("one", "xin", "session")))
            count = len(prompts)
            assert count > 0
            second = parse_sse(await collect_sse(agent.chat_with_agent_stream("two", "xin", "session")))
            assert len(prompts) == count
        assert first[-1] == second[-1] == "[DONE]"
        assert not any(isinstance(e, dict) and e.get("type") == "error" for e in first + second)
    asyncio.run(run())
    assert storage.context_state["summary"] == "rolling summary"
    assert storage.history[:2] == old
    assert memory.updated == []
