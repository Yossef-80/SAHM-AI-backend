"""Ollama provider (local, or remote over ngrok / Google Colab).

Selected when ``AI_LLM_PROVIDER=ollama``. Reads from ``.env``:

    OLLAMA_BASE_URL=https://xxxx.ngrok-free.dev   (or http://localhost:11434)
    OLLAMA_MODEL=hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF
    OLLAMA_NUM_CTX=8192        OLLAMA_KEEP_ALIVE=30m
    OLLAMA_TIMEOUT_SECONDS=300 OLLAMA_REASONING=false   (optional)

All Ollama-specific knowledge (ngrok header, context size, keep-alive) lives in
this one file. Nothing else in the codebase knows Ollama exists.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from app.ai.enums import StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.settings import AISettings

#: Ollama enforces the JSON schema natively via its ``format`` field -- the
#: most reliable option for small local models.
DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.JSON_SCHEMA

#: Free ngrok tunnels return an HTML interstitial page without this header.
_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true", "User-Agent": "sahm-ai/1.0"}


def build_chat_model(
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - installed by default
        raise ProviderNotInstalledError("ollama", "langchain-ollama") from exc

    resolved_base_url = (base_url.rstrip("/") if base_url else None) or (
        settings.ollama_base_url.rstrip("/") if getattr(settings, "ollama_base_url", None) else None
    )

    if not resolved_base_url:
        raise ProviderNotConfiguredError(
            "AI_LLM_PROVIDER=ollama but Ollama needs OLLAMA_BASE_URL. Start your Ollama "
            "server / Colab ngrok tunnel and paste its URL into .env."
        )

    kwargs: dict[str, Any] = {
        "model": model or settings.ollama_model,
        "base_url": resolved_base_url,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": settings.llm_max_tokens,
        "keep_alive": settings.ollama_keep_alive,
        "client_kwargs": {"headers": _NGROK_HEADERS, "timeout": settings.ollama_timeout_seconds},
        "async_client_kwargs": {"headers": _NGROK_HEADERS, "timeout": settings.ollama_timeout_seconds},
    }
    if settings.ollama_reasoning is not None:
        kwargs["reasoning"] = settings.ollama_reasoning
    return ChatOllama(**kwargs)
