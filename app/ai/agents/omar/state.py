from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OmarState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    brand_context: Any
    research_notes: list[str] | None
    strategy: dict[str, Any] | None
    campaign: dict[str, Any] | None
    marketing_strategy: dict[str, Any]
    campaign_strategy: dict[str, Any]
    content_calendar: dict[str, Any]
    final_report: dict[str, Any]
