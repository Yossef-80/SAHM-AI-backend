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
        "Retrieve the compact brand context: identity (voice, visual style, "
        "offers, audience, competitors), the active strategy plan, the most "
        "relevant research/competitor notes for this task, and recent learnings. "
        "The result is pre-budgeted (~900 tokens) and ranked deterministically "
        "(lexical relevance + recency + kind affinity), so it is safe to call on "
        "every task. Call this FIRST for any task that should sound like the "
        "brand. Do not call it for campaign performance numbers (use "
        "get_campaign_insights) or to research the market (use web_search)."
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
