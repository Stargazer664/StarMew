"""Shared configuration boundary for OpenAI-compatible chat models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Local-only inference must never attempt an outbound LangSmith trace.  This is
# intentionally applied at import time because RAG modules can create their
# own chains before the first chat model is constructed.
if os.getenv("LLM_PROVIDER", "ollama").strip().lower() == "ollama":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    base_url: str
    api_key: str


def _clean_url(value: str) -> str:
    return value.rstrip("/")


def resolve_llm_settings(
    *, role: str, environ: Mapping[str, str] | None = None
) -> LLMSettings:
    env = os.environ if environ is None else environ
    provider = env.get("LLM_PROVIDER", "ollama").strip().lower()
    # Evaluation may use a remote judge without changing the RAG grader or
    # any other runtime model. With no opt-in, preserve the old judge default.
    if role == "evaluation":
        provider = (env.get("EVALUATION_PROVIDER") or provider).strip().lower()
        role = "judge"
    if role not in {"generation", "judge", "fast", "router"}:
        raise ValueError(f"unsupported LLM role: {role}")

    if provider == "ollama":
        generation_model = env.get("OLLAMA_MODEL", "qwen3.5:4b")
        role_models = {
            "generation": generation_model,
            "router": env.get("AGENT_ROUTER_MODEL", "").strip() or generation_model,
            "judge": env.get("OLLAMA_JUDGE_MODEL") or generation_model,
            "fast": env.get("OLLAMA_FAST_MODEL") or generation_model,
        }
        return LLMSettings(
            provider=provider,
            model=role_models[role],
            base_url=_clean_url(
                env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
            ),
            api_key=env.get("OLLAMA_API_KEY", "ollama"),
        )

    if provider in {"ark", "aliyun"}:
        generation_model = env.get("MODEL", "")
        role_models = {
            "generation": generation_model,
            "router": env.get("AGENT_ROUTER_MODEL", "").strip() or generation_model,
            "judge": env.get("GRADE_MODEL") or env.get("FAST_MODEL") or generation_model,
            "fast": env.get("FAST_MODEL") or generation_model,
        }
        if provider == "aliyun" and not all(
            str(value).strip() for value in (
                role_models[role], env.get("BASE_URL", ""), env.get("ARK_API_KEY", "")
            )
        ):
            raise ValueError("aliyun requires a model, BASE_URL and ARK_API_KEY")
        return LLMSettings(
            provider=provider,
            model=role_models[role],
            base_url=_clean_url(env.get("BASE_URL", "")),
            api_key=env.get("ARK_API_KEY", ""),
        )

    raise ValueError(f"unsupported LLM_PROVIDER: {provider}")


def build_chat_model_kwargs(
    *, role: str, environ: Mapping[str, str] | None = None, **overrides: Any
) -> dict[str, Any]:
    settings = resolve_llm_settings(role=role, environ=environ)
    kwargs = {
        "model": settings.model,
        "model_provider": "openai",
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "max_retries": 0,
    }
    if settings.provider == "ollama":
        kwargs["reasoning_effort"] = "none"
    elif settings.provider == "aliyun":
        kwargs["extra_body"] = {"enable_thinking": False}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    kwargs.update(overrides)
    return kwargs


def create_chat_model(
    *, role: str, environ: Mapping[str, str] | None = None, **overrides: Any
) -> Any:
    settings = resolve_llm_settings(role=role, environ=environ)
    if settings.provider == "ollama":
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        **build_chat_model_kwargs(role=role, environ=environ, **overrides)
    )
