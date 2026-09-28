"""Provider selection. The ONLY place a provider is chosen.

``get_llm(tier)`` reads the provider->tier and model->tier maps from config and
returns a ready client. Changing provider or model is an env change: nothing
else in the repo contains a provider or model string.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.core.errors import LLMNotConfiguredError
from app.core.llm.base import LLMClient, TierName
from app.core.llm.claude_client import ClaudeClient
from app.core.llm.mock_client import MockClient
from app.core.llm.openai_client import OpenAIClient
from app.core.logging import get_logger

_logger = get_logger(__name__)

_clients: dict[tuple[str, str, str], LLMClient] = {}


def build_client(tier: TierName, settings: Settings | None = None) -> LLMClient:
    """Construct a client for ``tier`` from config. Not cached."""
    cfg = settings or get_settings()
    provider = cfg.provider_for_tier[tier]
    model = cfg.model_for_tier[tier]
    key = (tier, provider, model)

    cached = _clients.get(key)
    if cached is not None:
        return cached

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
        )
    elif provider == "openai":
        client = OpenAIClient(
            api_key=cfg.openai_api_key,
            model=model,
            timeout=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
            base_delay=cfg.llm_retry_base_delay,
        )
    else:  # pragma: no cover - Literal makes this unreachable via config
        raise LLMNotConfiguredError(f"unknown llm provider '{provider}'")

    _clients[key] = client
    _logger.info(
        "llm_client_built",
        extra={"tier": tier, "provider": provider, "model": model},
    )
    return client


def get_llm(tier: TierName = "strong", settings: Settings | None = None) -> LLMClient:
    """Return a client for ``tier``. Cached per (tier, provider, model)."""
    return build_client(tier, settings)


def reset_clients() -> None:
    """Drop cached clients. Used by tests and on settings reload."""
    _clients.clear()


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
