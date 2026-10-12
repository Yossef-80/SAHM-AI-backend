"""Provider selection. The ONLY place a provider is chosen.

``get_llm(tier)`` reads the provider->tier and model->tier maps from config and
returns a ready client. Changing provider or model is an env change: nothing
else in the repo contains a provider or model string.
"""

from __future__ import annotations

import hashlib
import json
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.errors import LLMNotConfiguredError
from app.core.llm.base import LLMClient, TierName
from app.core.llm.claude_client import API_URL as CLAUDE_API_URL, ClaudeClient
from app.core.llm.mock_client import MockClient
from app.core.llm.openai_client import API_URL, OpenAIClient
from app.core.logging import get_logger
from app.core.security import TokenCipher
from app.db.models import LlmProviderSettings

_logger = get_logger(__name__)

# Defaults are full Chat Completions URLs because OpenAIClient posts directly
# to the configured URL. All listed providers expose an OpenAI-compatible API.
OPENAI_COMPATIBLE_URLS = {
    "openai": API_URL,
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
    "fireworks": "https://api.fireworks.ai/inference/v1/chat/completions",
    "perplexity": "https://api.perplexity.ai/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "custom": API_URL,
}

#: Credentials are included only as a one-way digest, never logged.
_ClientKey = tuple[str, str, str, str, tuple[tuple[str, str], ...], str]

_clients: dict[_ClientKey, LLMClient] = {}

#: In-memory 60s TTL cache for workspace settings to eliminate redundant DB SELECTs
_WORKSPACE_SETTINGS_CACHE: dict[str, tuple[LlmProviderSettings | None, float]] = {}
_LLM_SETTINGS_CACHE_TTL = 60.0


def invalidate_workspace_llm_cache(business_id: str | None = None) -> None:
    """Drop cached workspace LLM settings. If business_id is None, clears all."""
    if business_id is None:
        _WORKSPACE_SETTINGS_CACHE.clear()
    else:
        _WORKSPACE_SETTINGS_CACHE.pop(business_id, None)


def build_client(tier: TierName, settings: Settings | None = None) -> LLMClient:
    """Construct a client for ``tier`` from config. Not cached."""
    cfg = settings or get_settings()
    provider = cfg.provider_for_tier[tier]
    model = cfg.model_for_tier[tier]
    # The endpoint and the extra headers are part of the identity of an
    # OpenAI-compatible client: changing LLM_BASE_URL / LLM_EXTRA_HEADERS must
    # build a new client instead of silently reusing a stale one.
    base_url = cfg.llm_base_url or (
        CLAUDE_API_URL if provider == "claude" else OPENAI_COMPATIBLE_URLS.get(provider, API_URL)
    )
    header_fp = tuple(sorted((str(k), str(v)) for k, v in cfg.llm_extra_headers.items()))
    credential = (
        cfg.anthropic_api_key
        if provider == "claude"
        else (cfg.gemini_api_key or cfg.openai_api_key if provider == "gemini" else cfg.openai_api_key)
    )
    credential_fp = hashlib.sha256((credential or "").encode()).hexdigest()
    key: _ClientKey = (tier, provider, model, base_url, header_fp, credential_fp)

    cached = _clients.get(key)
    if cached is not None:
        return cached

    # Loud warning if mock client is selected when real provider keys are present
    if provider == "mock" and (cfg.anthropic_api_key or cfg.openai_api_key or getattr(cfg, "gemini_api_key", None)):
        _logger.warning(
            "llm_mock_selected_despite_available_keys",
            extra={
                "tier": tier,
                "provider": provider,
                "reason": "mock_configured_for_tier",
            },
        )

    client: LLMClient
    if provider == "mock":
        client = MockClient(model=model)
    elif provider == "claude":
        client = ClaudeClient(
            api_key=cfg.anthropic_api_key,
            model=model,
            timeout=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
            base_delay=cfg.llm_retry_base_delay,
            base_url=base_url,
        )
    elif provider in OPENAI_COMPATIBLE_URLS:
        client = OpenAIClient(
            api_key=credential,
            model=model,
            timeout=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
            base_delay=cfg.llm_retry_base_delay,
            base_url=base_url,
            extra_headers=cfg.llm_extra_headers,
        )
    else:  # pragma: no cover - Literal makes this unreachable via config
        raise LLMNotConfiguredError(f"unknown llm provider '{provider}'")

    _clients[key] = client
    _logger.info(
        "llm_client_built",
        extra={
            "tier": tier,
            "provider": provider,
            "model": model,
            "base_url_configured": bool(cfg.llm_base_url),
            # Header names only: values may be credentials.
            "extra_header_names": [name for name, _ in header_fp],
        },
    )
    return client


async def resolve_workspace_tier(
    db: AsyncSession,
    business_id: str,
    tier: TierName,
    settings: Settings | None = None,
) -> dict[str, object]:
    """Return effective, masked metadata for a workspace LLM tier."""
    cfg = settings or get_settings()

    # In-memory TTL cache (60s) check to eliminate redundant database queries
    now_mono = time.monotonic()
    cached = _WORKSPACE_SETTINGS_CACHE.get(business_id)
    if cached is not None and now_mono < cached[1]:
        row = cached[0]
    else:
        row = await db.scalar(
            select(LlmProviderSettings).where(
                LlmProviderSettings.business_id == business_id
            )
        )
        _WORKSPACE_SETTINGS_CACHE[business_id] = (row, now_mono + _LLM_SETTINGS_CACHE_TTL)

    provider_value = getattr(row, f"{tier}_provider", None) if row else None
    model_value = getattr(row, f"{tier}_model", None) if row else None
    base_url_value = getattr(row, f"{tier}_base_url", None) if row else None
    encrypted_key = getattr(row, f"{tier}_api_key_enc", None) if row else None
    encrypted_headers = getattr(row, f"{tier}_extra_headers_enc", None) if row else None
    provider = provider_value or cfg.provider_for_tier[tier]
    model = model_value or cfg.model_for_tier[tier]
    if provider == "openai":
        env_key = cfg.openai_api_key
    elif provider == "claude":
        env_key = cfg.anthropic_api_key
    elif provider == "gemini":
        env_key = getattr(cfg, "gemini_api_key", None) or cfg.openai_api_key
    else:
        env_key = None
    cipher = TokenCipher(cfg.token_encryption_key)
    workspace_key = cipher.decrypt(encrypted_key) if encrypted_key else None
    workspace_headers = (
        json.loads(cipher.decrypt(encrypted_headers)) if encrypted_headers else None
    )
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url_value
        or (cfg.llm_base_url if provider == "openai" else None),
        "api_key": workspace_key or env_key,
        "key_source": "workspace" if workspace_key else "environment" if env_key else "missing",
        "extra_headers": workspace_headers or cfg.llm_extra_headers,
        "headers_source": "workspace" if workspace_headers else "environment" if cfg.llm_extra_headers else "missing",
        "has_override": row is not None,
    }


async def get_workspace_llm(
    tier: TierName,
    *,
    db: AsyncSession,
    business_id: str,
    settings: Settings | None = None,
) -> LLMClient:
    """Build/cache a client from this workspace's encrypted provider settings."""
    cfg = settings or get_settings()
    values = await resolve_workspace_tier(db, business_id, tier, cfg)
    updates: dict[str, object] = {
        f"llm_provider_{tier}": values["provider"],
        f"llm_model_{tier}": values["model"],
        # Claude accepts its native Messages endpoint; OpenAI may use any
        # OpenAI-compatible endpoint configured for this workspace.
        "llm_base_url": values["base_url"],
        "llm_extra_headers": values["extra_headers"],
    }
    if values["provider"] == "openai":
        updates["openai_api_key"] = values["api_key"]
    elif values["provider"] == "claude":
        updates["anthropic_api_key"] = values["api_key"]
    elif values["provider"] in OPENAI_COMPATIBLE_URLS:
        # Groq, Gemini, OpenRouter, and the other compatible gateways all use
        # OpenAIClient, so their workspace-scoped secret must reach its key
        # field as well. Without this, the key is saved/decrypted but discarded.
        updates["openai_api_key"] = values["api_key"]
    return get_llm(tier, cfg.model_copy(update=updates))


def get_llm(tier: TierName = "strong", settings: Settings | None = None) -> LLMClient:
    """Return a client for ``tier``. Cached per tier, provider, model and endpoint."""
    return build_client(tier, settings)


def reset_clients() -> None:
    """Drop cached clients and workspace settings. Used by tests and on settings reload."""
    _clients.clear()
    invalidate_workspace_llm_cache()


async def close_clients() -> None:
    """Close any held HTTP connections. Called on app shutdown."""
    for client in list(_clients.values()):
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover - shutdown best effort
                pass
    _clients.clear()
