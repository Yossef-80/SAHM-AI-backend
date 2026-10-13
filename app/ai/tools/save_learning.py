"""Explicit, business-scoped long-term learning storage for the AI agents."""

from __future__ import annotations

import json
from uuid import uuid4

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.enums import LearningCategory
from app.ai.memory.brand_memory import Learning, business_id_from, learnings_namespace
from app.ai.providers.db import current_store
from app.ai.security import redact_sensitive_data
from app.ai.settings import get_ai_settings
from app.ai.loggers.tools.save_learning import logger


class SaveLearningInput(BaseModel):
    insight: str = Field(min_length=1, max_length=1200)
    category: LearningCategory = LearningCategory.MARKET
    source_url: str | None = Field(default=None, max_length=1000)


@tool(args_schema=SaveLearningInput)
def save_learning_tool(
    insight: str,
    category: LearningCategory = LearningCategory.MARKET,
    source_url: str | None = None,
    config: RunnableConfig | None = None,
) -> str:
    """Save one durable, non-sensitive finding to this business's AI memory.

    Use this only when the user explicitly asks the agent to remember or save a
    finding. Do not store credentials, personal data, temporary questions, or
    unverified claims. Include a source URL when the finding came from the web.
    """
    safe_insight = str(redact_sensitive_data(insight)).strip()
    safe_url = str(redact_sensitive_data(source_url)).strip() if source_url else None
    normalized_category = LearningCategory(category)
    settings = get_ai_settings()
    max_chars = max(1, settings.memory_item_max_tokens * 4)
    safe_insight = safe_insight[:max_chars]
    if safe_url and len(safe_url) > 1000:
        safe_url = safe_url[:1000]

    learning = Learning(
        insight=safe_insight,
        category=normalized_category,
        source_url=safe_url,
        agent=(config or {}).get("configurable", {}).get("agent"),
    )
    key = f"learning_{uuid4().hex}"
    business_id = business_id_from(config)
    current_store().put(
        learnings_namespace(business_id),
        key,
        learning.model_dump(mode="json"),
    )
    logger.info(
        "learning_saved",
        business_id=business_id,
        category=normalized_category.value,
        source_url_present=bool(safe_url),
        insight_chars=len(safe_insight),
    )
    return json.dumps(
        {
            "status": "ok",
            "id": key,
            "business_id": business_id,
            "category": normalized_category.value,
            "insight": safe_insight,
            "source_url": safe_url,
        },
        ensure_ascii=False,
    )


__all__ = ["SaveLearningInput", "save_learning_tool"]
