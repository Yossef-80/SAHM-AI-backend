from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class OmarState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
