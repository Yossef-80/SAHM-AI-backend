"""Run-local structured planning tools for Omar.

The original Drive bundle included strategy generators that called a second LLM
through a module absent from the app archive. These adapters keep the useful
structured outputs but let the already-running Omar model populate them through
its normal tool calls. They only update the current LangGraph state: they do not
write to the database, call Meta, or trigger publication.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.ai.agents.omar.schema import (
    CampaignStrategyOutput,
    ContentCalendarOutput,
    MarketingStrategyOutput,
)


def _state_update(
    tool_call_id: str,
    content: str,
    *,
    tool_name: str,
    **state_fields: Any,
) -> Command:
    """Return a named tool result and update only this graph invocation's state."""

    return Command(
        update={
            **state_fields,
            "messages": [
                ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
            ],
        }
    )


def _model_dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


def _brand_context_was_loaded(state: dict[str, Any]) -> bool:
    return any(
        getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "get_brand_context_tool"
        for message in state.get("messages", [])
    )


@tool(args_schema=MarketingStrategyOutput)
def generate_marketing_strategy(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
    **kwargs: Any,
) -> Command:
    """Store a structured marketing strategy for the current Omar run.

    Use brand context and any research already gathered. Unknown facts belong in
    ``missing_inputs``; unsupported budgets, audience sizes, and KPI targets must
    not be made up. Call this before ``generate_campaign_strategy``.
    """

    if not _brand_context_was_loaded(state):
        return _state_update(
            tool_call_id,
            "ERROR: call get_brand_context_tool first; if no profile is available, record that as a missing input.",
            tool_name="generate_marketing_strategy",
        )

    strategy = MarketingStrategyOutput.model_validate(kwargs)
    data = _model_dump(strategy)
    return _state_update(
        tool_call_id,
        "MARKETING_STRATEGY_SAVED\n" + json.dumps(data, ensure_ascii=False),
        tool_name="generate_marketing_strategy",
        marketing_strategy=data,
    )


@tool(args_schema=CampaignStrategyOutput)
def generate_campaign_strategy(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
    **kwargs: Any,
) -> Command:
    """Store the campaign plan and creative brief that can be handed to Layla.

    Requires a marketing strategy earlier in this run. Keep objective, audience,
    budget guidance, and CTA unset or listed as missing when the user/evidence
    did not provide them. This tool creates no Meta campaign or ad.
    """

    if not state.get("marketing_strategy"):
        return _state_update(
            tool_call_id,
            "ERROR: call generate_marketing_strategy first; do not invent a plan.",
            tool_name="generate_campaign_strategy",
        )

    strategy = CampaignStrategyOutput.model_validate(kwargs)
    data = _model_dump(strategy)
    return _state_update(
        tool_call_id,
        "CAMPAIGN_STRATEGY_SAVED\n" + json.dumps(data, ensure_ascii=False),
        tool_name="generate_campaign_strategy",
        campaign_strategy=data,
    )


@tool(args_schema=ContentCalendarOutput)
def generate_content_calendar(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict[str, Any], InjectedState],
    **kwargs: Any,
) -> Command:
    """Store a structured content calendar aligned to this run's campaign plan.

    Requires a campaign strategy earlier in this run. Use only when the user
    requests a calendar; do not invent dates, budgets, or performance targets.
    """

    if not state.get("campaign_strategy"):
        return _state_update(
            tool_call_id,
            "ERROR: call generate_campaign_strategy first; do not invent a plan.",
            tool_name="generate_content_calendar",
        )

    calendar = ContentCalendarOutput.model_validate(kwargs)
    data = _model_dump(calendar)
    return _state_update(
        tool_call_id,
        "CONTENT_CALENDAR_SAVED\n" + json.dumps(data, ensure_ascii=False),
        tool_name="generate_content_calendar",
        content_calendar=data,
    )
