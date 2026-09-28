"""Memory tools: get_brand_context (read) and save_learning (draft_write)."""

from __future__ import annotations

from typing import Any

from app.core.tools.base import Tool, ToolContext, ToolOutput
from app.core.memory.store import BrandMemoryStore, get_brand_context
from app.schemas.tools import (
    BrandContext,
    GetBrandContextInput,
    Learning,
    SaveLearningInput,
)
from app.schemas.common import ToolSideEffect


async def _handle_get_brand_context(payload: GetBrandContextInput, ctx: ToolContext) -> BrandContext:
    return await get_brand_context(payload, ctx)


async def _handle_save_learning(payload: SaveLearningInput, ctx: ToolContext) -> Learning:
    """Write a learning back to brand memory.

    ``side_effect="draft_write"``: it only ever writes to our own memory table,
    never to an external system. The runtime's autonomy policy still governs
    whether the write is committed for this run.
    """
    store = BrandMemoryStore(ctx.db)
    return await store.save_learning(business_id=ctx.business_id, payload=payload)


get_brand_context_tool = Tool(
    name="get_brand_context",
    description=(
        "Retrieve the compact brand context: brand voice, visual style, offers, "
        "audience, competitors, and the most relevant memory items and learnings "
        "for the task at hand. Call this FIRST for any task that should sound like "
        "the brand or reference its offers/audience. Do not call it to look up "
        "campaign performance numbers (use get_campaign_insights) or to research "
        "the market (use web_search)."
    ),
    input_model=GetBrandContextInput,
    output_model=BrandContext,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_brand_context,
)

save_learning_tool = Tool(
    name="save_learning",
    description=(
        "Persist a durable learning to brand memory: what worked, what did not, and "
        "whether it applies to future campaigns. Use this at the end of an analysis "
        "or after a campaign result is known. Do NOT use it for transient notes or "
        "for anything the user has not established from real data."
    ),
    input_model=SaveLearningInput,
    output_model=Learning,
    side_effect=ToolSideEffect.DRAFT_WRITE,
    handler=_handle_save_learning,
)

MEMORY_TOOLS: list[Tool] = [get_brand_context_tool, save_learning_tool]

__all__ = ["MEMORY_TOOLS", "get_brand_context_tool", "save_learning_tool", "ToolOutput", "Any"]
