from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class NourState(TypedDict):
    """
    The state for the Nour agent.
    We keep track of the conversation messages.
    Additional state like business_id is passed via RunnableConfig.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
