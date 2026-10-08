import backend.langsmith_config as langsmith_config


def test_ollama_provider_forces_external_tracing_off(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "legacy-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    langsmith_config.setup_langsmith()

    assert langsmith_config.os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert langsmith_config.os.environ["LANGSMITH_TRACING"] == "false"
