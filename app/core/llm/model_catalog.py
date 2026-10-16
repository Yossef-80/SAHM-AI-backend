"""What models a credential can actually use, and how to say so usefully.

Picking a model by name is a guessing game that the app used to lose silently:
``gemini-3-flash`` looks plausible and returns ``404 NOT_FOUND``, a free OpenRouter
model returns ``403`` because it is gated to coding agents, and the user only finds
out when the next chat turn dies. This module asks the provider for its own list and
turns a miss into a message naming the real alternatives.

Everything here **fails open**. A catalog we cannot fetch (no key, no network, a
provider without a listing endpoint, an unknown gateway) is never a reason to refuse
a save -- an unverifiable choice is the user's to make.
"""

from __future__ import annotations

import difflib
from typing import Any

import httpx

from app.core.logging import get_logger

_logger = get_logger(__name__)

#: Long enough for a slow listing, short enough that saving settings stays snappy.
LIST_TIMEOUT_SECONDS = 8.0

_LISTING_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
}

#: Providers whose listing endpoint is OpenAI-compatible (``{"data": [{"id": ...}]}``).
_OPENAI_SHAPED = frozenset({"openai", "groq", "openrouter"})


def _extract(provider: str, payload: dict[str, Any]) -> list[str]:
    if provider == "gemini":
        names = []
        for item in payload.get("models", []) or []:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue  # embeddings, TTS, live-audio: not chat models
            names.append(str(item.get("name") or "").replace("models/", ""))
        return [name for name in names if name]
    return [
        str(item.get("id") or "")
        for item in payload.get("data", []) or []
        if isinstance(item, dict) and str(item.get("id") or "")
    ]


async def list_models(
    provider: str, api_key: str, *, base_url: str | None = None
) -> list[str] | None:
    """The chat models this credential offers, or ``None`` when unknowable."""
    provider = (provider or "").strip().lower()
    if base_url:
        # A custom gateway or proxy: its catalog is its own business.
        return None
    url = _LISTING_URLS.get(provider)
    if not url or not api_key:
        return None
    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict[str, str] = {}
    if provider == "gemini":
        headers = {"x-goog-api-key": api_key}
        params = {"pageSize": "200"}
    try:
        async with httpx.AsyncClient(timeout=LIST_TIMEOUT_SECONDS, trust_env=False) as client:
            response = await client.get(url, headers=headers, params=params)
        if response.status_code != 200:
            _logger.info(
                "model_catalog_unavailable",
                extra={"provider": provider, "status": response.status_code},
            )
            return None
        names = _extract(provider, response.json())
    except Exception as exc:  # noqa: BLE001 - an unverifiable choice is allowed
        _logger.info(
            "model_catalog_unavailable",
            extra={"provider": provider, "error": f"{type(exc).__name__}: {str(exc)[:120]}"},
        )
        return None
    return names or None


def suggest_models(chosen: str, available: list[str], *, limit: int = 8) -> list[str]:
    """The closest available names to what was typed, best first.

    Family matches (``gemini-3-flash`` -> ``gemini-3.6-flash``, ``gemini-3.5-flash``)
    beat a fuzzy match on the whole string, because that is what the user meant.
    """
    model = (chosen or "").strip()
    if not model or not available:
        return []
    lowered = model.casefold()
    parts = [part for part in lowered.replace(":", "-").split("-") if part]
    same_family = [
        name
        for name in available
        if parts and parts[0] in name.casefold() and all(part in name.casefold() for part in parts[:2])
    ]
    close = difflib.get_close_matches(model, available, n=limit, cutoff=0.5)
    ordered: list[str] = []
    for name in [*same_family, *close]:
        if name not in ordered:
            ordered.append(name)
    return ordered[:limit]


__all__ = ["LIST_TIMEOUT_SECONDS", "list_models", "suggest_models"]
