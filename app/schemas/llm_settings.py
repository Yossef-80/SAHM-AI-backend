"""Workspace-scoped LLM settings API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr

from app.schemas.common import ApiModel


Provider = Literal[
    "mock",
    "openai",
    "claude",
    "groq",
    "gemini",
    "deepseek",
    "mistral",
    "openrouter",
    "together",
    "fireworks",
    "perplexity",
    "xai",
    "custom",
]


class LlmTierUpdate(ApiModel):
    provider: Provider | None = None
    model: str | None = Field(default=None, max_length=120)
    api_key: SecretStr | None = None
    base_url: str | None = Field(default=None, max_length=500)
    extra_headers: dict[str, str] | None = None
    #: Output ceiling in tokens (reasoning tokens included on reasoning models).
    #: ``null`` clears the override and returns the tier to the process default.
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    #: ``none`` | ``low`` | ``medium`` | ``high``. Only OpenAI reasoning models
    #: read it; ``none`` is what allows function tools on Chat Completions.
    reasoning_effort: str | None = Field(default=None, max_length=16)


class LlmSettingsUpdate(ApiModel):
    fast: LlmTierUpdate | None = None
    strong: LlmTierUpdate | None = None


class LlmTierOut(ApiModel):
    provider: Provider
    model: str
    base_url: str | None = None
    #: Effective values: the workspace override when set, else ``None`` -- which
    #: the UI renders as "process default".
    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    key_configured: bool = False
    key_source: Literal["workspace", "environment", "missing"] = "missing"
    extra_headers_configured: bool = False
    extra_headers_source: Literal["workspace", "environment", "missing"] = "missing"


class LlmSettingsOut(ApiModel):
    business_id: str
    fast: LlmTierOut
    strong: LlmTierOut
    has_workspace_override: bool = False
    encryption_key_configured: bool = True
