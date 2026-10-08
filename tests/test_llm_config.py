import os
import importlib
import pytest

from backend.llm_config import (
    build_chat_model_kwargs,
    create_chat_model,
    resolve_llm_settings,
)


def test_cloud_evaluation_is_isolated_from_all_runtime_roles():
    env = {
        "LLM_PROVIDER": "ollama",
        "EVALUATION_PROVIDER": "aliyun",
        "OLLAMA_MODEL": "qwen3.5:4b",
        "GRADE_MODEL": "qwen3.7-flash",
        "BASE_URL": "https://cloud.example/compatible-mode/v1/",
        "ARK_API_KEY": "test-cloud-key",
    }
    settings = resolve_llm_settings(role="evaluation", environ=env)
    assert settings.provider == "aliyun"
    assert settings.model == "qwen3.7-flash"
    assert settings.base_url == "https://cloud.example/compatible-mode/v1"
    assert settings.api_key == "test-cloud-key"
    for role in ("generation", "judge", "fast"):
        local = resolve_llm_settings(role=role, environ=env)
        assert local.provider == "ollama"
        assert local.model == "qwen3.5:4b"
        assert local.base_url == "http://127.0.0.1:11434/v1"
        assert local.api_key == "ollama"


def test_evaluation_defaults_to_existing_judge_without_explicit_opt_in():
    settings = resolve_llm_settings(role="evaluation", environ={
        "OLLAMA_JUDGE_MODEL": "local-judge", "GRADE_MODEL": "cloud-judge",
    })
    assert settings.provider == "ollama"
    assert settings.model == "local-judge"


def test_aliyun_evaluation_uses_its_own_thinking_parameter():
    kwargs = build_chat_model_kwargs(role="evaluation", environ={
        "EVALUATION_PROVIDER": "aliyun", "GRADE_MODEL": "qwen3.7-flash",
        "BASE_URL": "https://cloud.example/v1", "ARK_API_KEY": "test-key",
    })
    assert kwargs["extra_body"] == {"enable_thinking": False}
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("missing", ["GRADE_MODEL", "BASE_URL", "ARK_API_KEY"])
def test_cloud_evaluation_rejects_missing_configuration(missing):
    env = {
        "EVALUATION_PROVIDER": "aliyun", "GRADE_MODEL": "qwen3.7-flash",
        "BASE_URL": "https://cloud.example/v1", "ARK_API_KEY": "test-key",
    }
    del env[missing]
    with pytest.raises(ValueError, match="aliyun.*requires"):
        resolve_llm_settings(role="evaluation", environ=env)


def test_ollama_defaults_ignore_legacy_ark_credentials():
    settings = resolve_llm_settings(
        role="generation",
        environ={
            "ARK_API_KEY": "legacy-secret",
            "BASE_URL": "https://legacy.example/v1",
            "MODEL": "legacy-model",
        },
    )

    assert settings.provider == "ollama"
    assert settings.model == "qwen3.5:4b"
    assert settings.base_url == "http://127.0.0.1:11434/v1"
    assert settings.api_key == "ollama"


def test_ollama_uses_separate_local_judge_model_when_configured():
    settings = resolve_llm_settings(
        role="judge",
        environ={
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "qwen3.5:4b",
            "OLLAMA_JUDGE_MODEL": "qwen3.5:9b",
            "OLLAMA_BASE_URL": "http://localhost:11434/v1/",
        },
    )

    assert settings.model == "qwen3.5:9b"
    assert settings.base_url == "http://localhost:11434/v1"


def test_ollama_chat_kwargs_do_not_send_ark_thinking_extension():
    kwargs = build_chat_model_kwargs(
        role="generation",
        environ={"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3.5:4b"},
        temperature=0,
        timeout=180,
        max_retries=0,
    )

    assert kwargs == {
        "model": "qwen3.5:4b",
        "model_provider": "openai",
        "api_key": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "temperature": 0,
        "timeout": 180,
        "max_retries": 0,
        "reasoning_effort": "none",
    }


def test_ark_remains_an_explicit_opt_in_provider():
    kwargs = build_chat_model_kwargs(
        role="judge",
        environ={
            "LLM_PROVIDER": "ark",
            "ARK_API_KEY": "secret",
            "BASE_URL": "https://ark.example/v1",
            "MODEL": "generation-model",
            "GRADE_MODEL": "judge-model",
        },
        temperature=0,
    )

    assert kwargs["model"] == "judge-model"
    assert kwargs["api_key"] == "secret"
    assert kwargs["base_url"] == "https://ark.example/v1"
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_create_chat_model_initializes_langchain_with_ollama_settings(monkeypatch):
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return "local-model"

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init_chat_model)

    result = create_chat_model(
        role="generation",
        environ={"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3.5:4b"},
        temperature=0.3,
        stream_usage=True,
    )

    assert result == "local-model"
    assert captured["model"] == "qwen3.5:4b"
    assert captured["base_url"] == "http://127.0.0.1:11434/v1"
    assert captured["api_key"] == "ollama"
    assert captured["temperature"] == 0.3
    assert captured["stream_usage"] is True
    assert captured["reasoning_effort"] == "none"
    assert "extra_body" not in captured


def test_create_ollama_model_forces_langsmith_tracing_off(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", lambda **kwargs: kwargs
    )

    create_chat_model(role="generation")

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_importing_ollama_config_disables_tracing_before_model_creation(monkeypatch):
    import backend.llm_config as llm_config

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    importlib.reload(llm_config)

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGSMITH_TRACING"] == "false"
