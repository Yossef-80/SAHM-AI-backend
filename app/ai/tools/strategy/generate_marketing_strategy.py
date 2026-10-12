import json
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .schemas import GenerateMarketingStrategyInput, MarketingStrategyOutput as MarketingStrategy
from ._common import state_update, structured_call


@tool(args_schema=GenerateMarketingStrategyInput)
def generate_marketing_strategy(
    focus: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Produce positioning, target segments, value propositions and a 90-day roadmap.
    `focus` = any specific instruction/angle (e.g. 'premium positioning vs. Toyota')."""
    if not state.get("brand_context"):
        return state_update(tool_call_id, "ERROR: call get_brand_context first.")

    out: MarketingStrategy = structured_call(
        MarketingStrategy,
        "You are a senior brand strategist. Build a concrete, evidence-based marketing strategy. "
        "Use only the provided brand context and research; flag assumptions explicitly.",
        {"focus": focus, "brand": state["brand_context"], "research": state.get("research_notes", [])},
    )
    data = out.model_dump()
    return state_update(
        tool_call_id,
        "MARKETING_STRATEGY_SAVED\n" + json.dumps(data, ensure_ascii=False),
        strategy=data,
    )
