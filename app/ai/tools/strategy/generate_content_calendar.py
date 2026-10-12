import json
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .schemas import GenerateContentCalendarInput, ContentCalendarOutput as ContentCalendar
from ._common import state_update, structured_call


@tool(args_schema=GenerateContentCalendarInput)
def generate_content_calendar(
    duration_days: int,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = {},
) -> Command:
    """Produce a day-by-day content calendar. Requires campaign strategy first."""
    if not state.get("campaign"):
        return state_update(tool_call_id, "ERROR: call generate_campaign_strategy first.")

    out: ContentCalendar = structured_call(
        ContentCalendar,
        "You are a content strategist. Align every item to the campaign channels and big idea.",
        {
            "duration_days": duration_days,
            "brand": state.get("brand_context"),
            "strategy": state.get("strategy"),
            "campaign": state["campaign"],
        },
    )
    data = out.model_dump()
    return state_update(
        tool_call_id,
        "CONTENT_CALENDAR_SAVED\n" + json.dumps(data, ensure_ascii=False),
        content_calendar=data,
    )
