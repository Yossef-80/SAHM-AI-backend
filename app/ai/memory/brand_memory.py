"""Long-term brand memory layout on top of a LangGraph ``BaseStore``.

Shared by every agent and by the ``get_brand_context`` / ``save_learning`` tools,
so all agents read and write the SAME memory for a business.

    ("brands", <business_id>)               key "profile"  -> BrandProfile
    ("brands", <business_id>, "learnings")  key <uuid>     -> Learning
"""

from __future__ import annotations

from datetime import UTC, datetime

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from app.ai.enums import LearningCategory
from app.ai.security import redact_sensitive_data

DEFAULT_BUSINESS_ID = "default"


class BrandProfile(BaseModel):
    """Customer identity injected into every agent prompt."""

    name: str = ""
    industry: str = ""
    market: str = Field("", description="Country / city the brand sells in")
    description: str = ""
    audience: str = ""
    tone: str = ""
    competitors_known: list[str] = Field(default_factory=list)
    do_not_use: list[str] = Field(default_factory=list, description="Forbidden words/claims")


class Learning(BaseModel):
    """One finding an agent decided was worth remembering."""

    insight: str
    category: LearningCategory = LearningCategory.MARKET
    source_url: str | None = None
    agent: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def profile_namespace(business_id: str) -> tuple[str, ...]:
    return ("brands", business_id)


def learnings_namespace(business_id: str) -> tuple[str, ...]:
    return ("brands", business_id, "learnings")


def put_brand_profile(store: BaseStore, business_id: str, profile: BrandProfile) -> None:
    safe_profile = redact_sensitive_data(profile.model_dump(mode="json"))
    store.put(profile_namespace(business_id), "profile", safe_profile)


def get_brand_profile(store: BaseStore, business_id: str) -> BrandProfile | None:
    item = store.get(profile_namespace(business_id), "profile")
    if item is None:
        return None
    return BrandProfile.model_validate(redact_sensitive_data(item.value))


def business_id_from(config: dict | None) -> str:
    """Read ``business_id`` from a LangChain ``RunnableConfig``."""
    configurable = (config or {}).get("configurable") or {}
    return str(configurable.get("business_id") or DEFAULT_BUSINESS_ID)


def agent_from(config: dict | None) -> str | None:
    configurable = (config or {}).get("configurable") or {}
    return configurable.get("agent")


__all__ = [
    "DEFAULT_BUSINESS_ID",
    "BrandProfile",
    "Learning",
    "agent_from",
    "business_id_from",
    "get_brand_profile",
    "learnings_namespace",
    "profile_namespace",
    "put_brand_profile",
]
