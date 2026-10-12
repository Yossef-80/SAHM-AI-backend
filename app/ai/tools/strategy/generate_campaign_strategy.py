import json
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .schemas import GenerateCampaignStrategyInput, CampaignStrategyOutput as CampaignStrategy
from ._common import state_update, structured_call


@tool(args_schema=GenerateCampaignStrategyInput)
def generate_campaign_strategy(
    objective: str,
    budget: float | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = {},
) -> Command:
    """Produce a campaign (big idea, channels, budget allocation). Requires marketing strategy first."""
    if not state.get("strategy"):
        return state_update(tool_call_id, "ERROR: call generate_marketing_strategy first.")

    out: CampaignStrategy = structured_call(
        CampaignStrategy,
        "You are a campaign strategist. Channel budget_share_pct values MUST sum to 100.",
        {
            "objective": objective,
            "budget": budget,
            "brand": state.get("brand_context"),
            "strategy": state["strategy"],
        },
    )
    data = out.model_dump()
    total = sum(c["budget_share_pct"] for c in data["channels"])
    warn = "" if abs(total - 100) < 0.5 else f"\nWARNING: channel shares sum to {total}; regenerate to fix."
    return state_update(
        tool_call_id,
        "CAMPAIGN_STRATEGY_SAVED" + warn + "\n" + json.dumps(data, ensure_ascii=False),
        campaign=data,
    )
