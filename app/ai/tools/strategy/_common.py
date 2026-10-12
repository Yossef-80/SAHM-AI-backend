"""Shared utilities for strategy tools (LangGraph Command state updates and LLM structured output)."""
from __future__ import annotations

import json
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.ai.providers.llm import get_llm
from app.ai.settings import get_ai_settings

T = TypeVar("T", bound=BaseModel)


def state_update(tool_call_id: str, content: str, **kwargs: Any) -> Command:
    """Create a LangGraph Command that yields a ToolMessage and updates state fields."""
    return Command(
        update={
            "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            **kwargs,
        }
    )


def structured_call(schema: type[T], system_prompt: str, context: dict[str, Any]) -> T:
    """Call configured LLM with structured output constraint against schema."""
    ai_settings = get_ai_settings()
    llm = get_llm(provider=ai_settings.llm_provider)
    runnable = llm.structured(schema)

    formatted_context = json.dumps(context, ensure_ascii=False, indent=2, default=str)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context and Requirements:\n{formatted_context}"),
    ]
    return runnable.invoke(messages)
