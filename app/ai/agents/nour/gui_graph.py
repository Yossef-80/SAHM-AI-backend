"""Optional single-agent UI formatter that follows the configured AI provider."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings
from app.schemas.blocks import MessageWithBlocks


class NourState(TypedDict):
    messages: list[BaseMessage]
    final_blocks: MessageWithBlocks | None


def call_llm(state: NourState, config: RunnableConfig) -> dict[str, MessageWithBlocks]:
    """Format the existing conversation; never selects a provider/model itself."""
    handle = get_llm(provider=get_ai_settings().llm_provider)
    prompt = [
        SystemMessage(
            content=(
                "You are Nour, a researcher. Convert the completed answer into the "
                "MessageWithBlocks schema. Preserve evidence URLs, missing inputs, "
                "and uncertainty. Do not invent facts."
            )
        ),
        *state.get("messages", []),
    ]
    formatted = handle.structured(MessageWithBlocks).invoke(prompt, config)
    return {"final_blocks": formatted}


graph_builder = StateGraph(NourState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)
graph = graph_builder.compile()
