"""Read the current settings profile and durable brand learnings for this business."""

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


@tool
def get_brand_context_tool(config: RunnableConfig) -> str:
    """Retrieve saved business settings plus durable learnings before planning.

    The current Settings profile is supplied by the conversation bridge and is
    authoritative. The AI store remains a backwards-compatible fallback for
    older workflows and a source of additional learnings.
    """
    business_id = business_id_from(config)
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    settings_profile = (
        configurable.get("business_profile_context")
        if isinstance(configurable, dict)
        else None
    )
    max_chars = max(512, get_ai_settings().memory_global_profile_max_tokens * 4)

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
            result = {
                "status": "partial",
                "business_id": business_id,
                "unavailable_reason": "No saved Business Settings profile or long-term brand profile is stored for this business.",
                "learnings": [],
            }
            return json.dumps(result, ensure_ascii=False)

        learnings = store.search(learnings_namespace(business_id), limit=20)
        safe_learnings: list[dict[str, Any]] = []
        for item in learnings:
            value = item.value if isinstance(item.value, dict) else {}
            safe_learnings.append(redact_sensitive_data(value))

        result = {
            "status": "ok",
            "business_id": business_id,
            "profile_source": "ai_memory",
            "profile": redact_sensitive_data(profile.model_dump(mode="json")),
            "learnings": safe_learnings,
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
                "unavailable_reason": "The saved profile and long-term brand context store could not be read.",
            },
            ensure_ascii=False,
        )


__all__ = ["get_brand_context_tool"]
