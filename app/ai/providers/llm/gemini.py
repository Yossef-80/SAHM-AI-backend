"""Gemini chat-model provider for the general AI layer.

The separate VLM integration remains isolated; this adapter only follows the
existing ``AI_LLM_PROVIDER=gemini`` selection for LangGraph text/structured calls.
"""

from __future__ import annotations

import warnings
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.ai.enums import StructuredOutputMethod
from app.ai.errors import ProviderNotConfiguredError, ProviderNotInstalledError
from app.ai.settings import AISettings

# The wrapper warns "uses fixed sampling defaults" for some model families. It
# is informational -- the API accepts and applies the value (verified against
# gemini-3.1-flash-lite: four headlines at temperature 0.0 collapsed to two
# distinct strings, the same four at 1.6 produced four) -- so it is silenced
# rather than acted on by dropping the sampling config.
warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*")

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.FUNCTION_CALLING


def build_chat_model(
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float | None = None,
    #: Accepted for a uniform `get_llm` signature. None of the non-OpenAI
    #: providers has a separate "reasoning effort" knob, and their own token
    #: limit already comes from settings, so both are intentionally unused.
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ProviderNotInstalledError("gemini", "langchain-google-genai") from exc

    if not settings.gemini_api_key:
        raise ProviderNotConfiguredError(
            "AI_LLM_PROVIDER=gemini but GEMINI_API_KEY/GOOGLE_API_KEY is empty."
        )

    chosen_model = model or settings.gemini_model
    # An explicit temperature always wins -- that is what a role asks for
    # (ad copy at 0.85, extraction at 0.0). The old code threw it away for every
    # flash-lite model and replaced it with the wrapper default, which is why
    # "three variations" of an ad came back as one sentence three times.
    temp = temperature if temperature is not None else settings.llm_temperature
    # A per-call budget (role max_tokens, or a workspace row) beats the process
    # one; omitting it keeps the historical `settings.llm_max_tokens`.
    max_out = max_tokens if max_tokens is not None else settings.llm_max_tokens

    return ChatGoogleGenerativeAI(
        model=chosen_model,
        api_key=SecretStr(settings.gemini_api_key),
        temperature=temp,
        max_tokens=max_out,
        request_timeout=settings.llm_timeout_seconds,
        retries=settings.llm_max_retries,
    )
