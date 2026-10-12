from typing import Any, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class OmarState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    brand_context: Any
    research_notes: list[str] | None
    strategy: dict[str, Any] | None
    campaign: dict[str, Any] | None
    content_calendar: dict[str, Any] | None
