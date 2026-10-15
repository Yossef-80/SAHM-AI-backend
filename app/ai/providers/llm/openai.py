"""OpenAI provider (also any OpenAI-compatible server via ``OPENAI_BASE_URL``).

Selected when ``AI_LLM_PROVIDER=openai``. Needs ``uv add langchain-openai``.

Two model families live behind this one endpoint, and they do not accept the
same request. GPT-4-era models take ``max_tokens`` and any ``temperature``;
reasoning models (o-series, GPT-5, GPT-6) reject both -- they take
``max_completion_tokens`` and the default temperature only. From GPT-5.4 on, a
request that carries **function tools** is refused unless ``reasoning_effort``
is ``none``, and GPT-6 Astra/6.1 Sol refuse tools on Chat Completions at all
(``app/core/llm/model_compat.py`` holds those rules and is unit-tested).

Every agent in this product binds tools, so a reasoning model is built with
``reasoning_effort="none"`` unless the deployment asked for something else --
that is the setting under which tools work on this endpoint at all.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.ai.enums import StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.settings import AISettings
from app.core.llm.model_compat import (
    chat_completions_tools_unsupported,
    recommended_reasoning_effort,
    supports_temperature,
)

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING


def build_chat_model(
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ProviderNotInstalledError("openai", "langchain-openai") from exc

    if not settings.openai_api_key and not settings.openai_base_url:
        raise ProviderNotConfiguredError("AI_LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")

    resolved_model = model or settings.openai_model
    resolved_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    resolved_temperature = settings.llm_temperature if temperature is None else temperature
    effort = recommended_reasoning_effort(
        resolved_model,
        reasoning_effort or settings.llm_reasoning_effort,
    )

    if chat_completions_tools_unsupported(resolved_model):
        # Not fatal here -- a text-only turn still works -- but every agent in
        # this product binds tools, so the first tool-using turn would fail with
        # Meta-grade obscurity. Say it once, where the model is built.
        from app.core.logging import get_logger

        get_logger(__name__).warning(
            "openai_chat_tools_unsupported",
            extra={
                "model": resolved_model,
                "detail": "tool calling requires the Responses API for this model; "
                "select a model that supports tools on /chat/completions",
            },
        )

    # langchain-openai renamed this field: its ``max_tokens`` is an alias of
    # ``max_completion_tokens`` on the wire, so the value below is already the
    # parameter reasoning models require. Naming it in the comment keeps the
    # next reader from "fixing" it back to the deprecated field.
    kwargs: dict[str, object] = {
        "model": resolved_model,
        "api_key": settings.openai_api_key or "not-needed",
        "base_url": settings.openai_base_url,
        "max_tokens": resolved_tokens,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if supports_temperature(resolved_model):
        kwargs["temperature"] = resolved_temperature
    if effort is not None:
        kwargs["reasoning_effort"] = effort

    return ChatOpenAI(**kwargs)  # type: ignore[arg-type]
