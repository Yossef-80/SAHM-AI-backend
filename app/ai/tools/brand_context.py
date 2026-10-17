"""Read the saved business profile and durable brand memories for this business.

Source of truth
---------------
The **database** is authoritative: ``businesses`` + ``brand_profiles`` for the
identity fields and ``brand_memory_items`` (via ``BrandMemoryStore``) for the
distilled facts, identity rows and recent learnings. The conversation bridge
already injects ``business_profile_context`` on every turn, so when that is
present it wins (it is the row the Settings page just wrote).

The LangGraph ``BaseStore`` path is kept only as a fallback for callers that run
an agent graph with no database attached (scripts, offline tests). It used to be
the *primary* path, which meant the agents read an in-process, empty store while
the real memory lived in Postgres -- one of the reasons "the AI forgot my brand"
was a recurring report.

Retrieval itself is deterministic (BM25 + recency + kind/task affinity); there
is no embedding call in this tool.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from app.ai.loggers.tools.get_brand_context import logger
from app.ai.memory.brand_memory import (
    business_id_from,
    get_brand_profile,
    learnings_namespace,
)
from app.ai.providers.db import current_store
from app.ai.security import redact_sensitive_data
from app.ai.settings import get_ai_settings


def _bounded_json(result: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text

    result["learnings"] = []
    result["truncated"] = True
    profile = result.get("profile")
    if isinstance(profile, dict):
        compact: dict[str, Any] = {}
        for key, value in profile.items():
            if isinstance(value, list):
                compact[key] = [str(item)[:120] for item in value[:6] if str(item).strip()]
            elif value is not None:
                compact[key] = str(value)[:400]
        result["profile"] = compact
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text

    # Keep a small but still useful identity/market snapshot rather than emit
    # malformed JSON by cutting an encoded response at an arbitrary byte.
    if isinstance(result.get("profile"), dict):
        source_profile = result["profile"]
        result["profile"] = {
            key: source_profile[key]
            for key in ("name", "industry", "country", "website_url", "target_audience")
            if key in source_profile
        }
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text

    result.pop("profile", None)
    return json.dumps(result, ensure_ascii=False, default=str)


def _has_profile_values(profile: Any) -> bool:
    if not isinstance(profile, dict):
        return False
    for value in profile.values():
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
        if value is not None and not isinstance(value, list) and str(value).strip():
            return True
    return False


def _learning_payload(item: Any) -> dict[str, Any]:
    """Serialise a ``BrandMemoryItem`` learning row for the agent."""
    return redact_sensitive_data(
        {
            "id": getattr(item, "id", None),
            "text": getattr(item, "text", ""),
            "kind": getattr(item, "kind", "learning"),
            "metadata": dict(getattr(item, "metadata_", None) or {}),
            "created_at": (
                item.created_at.isoformat()
                if getattr(item, "created_at", None) is not None
                else None
            ),
        }
    )


async def _from_database(configurable: dict[str, Any], business_id: str) -> dict[str, Any] | None:
    """Read the authoritative profile + memories from the workspace DB."""
    db = configurable.get("tool_db")
    if db is None:
        return None

    from app.core.memory.store import BrandMemoryStore

    store = BrandMemoryStore(db)
    profile = await store.get_profile(business_id)
    distilled = await store.get_distilled_profile(business_id)
    learnings = await store.recent_learnings(business_id=business_id, limit=20)

    profile_payload: dict[str, Any] = {}
    settings_profile = configurable.get("business_profile_context")
    if isinstance(settings_profile, dict) and _has_profile_values(settings_profile):
        profile_payload = dict(settings_profile)
        source = "business_settings"
    elif profile is not None:
        profile_payload = {
            "brand_voice": profile.brand_voice,
            "visual_style": profile.visual_style,
            "target_audience": profile.target_audience,
            "key_offers": list(profile.key_offers or []),
            "goals": list(profile.goals or []),
            "competitors": list(profile.competitors or []),
        }
        source = "brand_profile"
    else:
        source = "empty"

    return {
        "status": "ok" if profile_payload or distilled else "partial",
        "business_id": business_id,
        "profile_source": source,
        "profile": redact_sensitive_data(profile_payload),
        "distilled_facts": distilled or None,
        "learnings": [_learning_payload(item) for item in learnings],
    }


@tool
async def get_brand_context_tool(config: RunnableConfig) -> str:
    """Retrieve the saved business profile plus durable brand memories.

    Returns, in order of authority: the Settings profile supplied by the
    conversation bridge, the stored brand profile, the distilled brand facts
    maintained by the background LTM, and recent learnings (newest first).
    Call this before any task that should sound like the brand.
    """
    business_id = business_id_from(config)
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    max_chars = max(512, get_ai_settings().memory_global_profile_max_tokens * 6)

    try:
        db_result = await _from_database(configurable, business_id)
    except Exception as exc:  # a memory read must never fail the turn
        logger.warning(
            "brand_context_db_read_failed",
            business_id=business_id,
            error_type=type(exc).__name__,
        )
        db_result = None

    if db_result is not None:
        return _bounded_json(db_result, max_chars)

    # ------------------------------------------------- legacy fallback (no DB) --
    settings_profile = (
        configurable.get("business_profile_context")
        if isinstance(configurable, dict)
        else None
    )

    if _has_profile_values(settings_profile):
        result: dict[str, Any] = {
            "status": "ok",
            "business_id": business_id,
            "profile_source": "business_settings",
            "profile": redact_sensitive_data(settings_profile),
            "learnings": [],
        }
        try:
            store = current_store()
            learnings = store.search(learnings_namespace(business_id), limit=20)
            result["learnings"] = [
                redact_sensitive_data(item.value if isinstance(item.value, dict) else {})
                for item in learnings
            ]
        except Exception as exc:
            logger.warning(
                "brand_context_learning_read_failed",
                business_id=business_id,
                error_type=type(exc).__name__,
            )
        return _bounded_json(result, max_chars)

    try:
        store = current_store()
        profile = get_brand_profile(store, business_id)
        if not profile:
            return json.dumps(
                {
                    "status": "partial",
                    "business_id": business_id,
                    "unavailable_reason": (
                        "No saved Business Settings profile or long-term brand profile "
                        "is stored for this business."
                    ),
                    "learnings": [],
                },
                ensure_ascii=False,
            )

        learnings = store.search(learnings_namespace(business_id), limit=20)
        result = {
            "status": "ok",
            "business_id": business_id,
            "profile_source": "ai_memory",
            "profile": redact_sensitive_data(profile.model_dump(mode="json")),
            "learnings": [
                redact_sensitive_data(item.value if isinstance(item.value, dict) else {})
                for item in learnings
            ],
        }
        return _bounded_json(result, max_chars)
    except Exception as exc:
        logger.error(
            "brand_context_read_failed",
            business_id=business_id,
            error_type=type(exc).__name__,
        )
        return json.dumps(
            {
                "status": "unavailable",
                "business_id": business_id,
                "unavailable_reason": (
                    "Brand memory could not be read; do not invent brand facts."
                ),
                "learnings": [],
            },
            ensure_ascii=False,
        )


__all__ = ["get_brand_context_tool"]
