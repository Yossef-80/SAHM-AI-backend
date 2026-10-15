"""Mock provider -- deterministic, offline chat model for tests and dry runs.

Selected when ``AI_LLM_PROVIDER=mock``. It never calls tools and fills any
structured-output schema with clearly-labelled ``[mock]`` values, so the full
LangGraph pipeline can be exercised without a GPU, tunnel or API key.
"""

from __future__ import annotations

import types
import typing
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from app.ai.enums import StructuredOutputMethod
from app.ai.settings import AISettings

DEFAULT_STRUCTURED_METHOD = StructuredOutputMethod.JSON_SCHEMA


def _last_human_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):  # prompt-value dicts
        messages = messages.get("messages", [])
    if hasattr(messages, "to_messages"):
        messages = messages.to_messages()
    for msg in reversed(list(messages or [])):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""


def _fake_value(annotation: Any, field_name: str, hint: str) -> Any:
    """Produce a schema-valid placeholder for one annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        return _fake_value(non_none[0], field_name, hint) if non_none else None
    if origin is typing.Literal:
        return args[0]
    if origin in (list, tuple, set):
        inner = args[0] if args else str
        return [_fake_value(inner, field_name, hint)]
    if origin is dict:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return fake_instance(annotation, hint)
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 0.5
    return f"[mock] {field_name}: {hint[:80]}".strip()


def fake_instance(schema: type[BaseModel], hint: str = "") -> BaseModel:
    """Fill every field of ``schema`` with a valid ``[mock]`` placeholder."""
    data = {
        name: _fake_value(field.annotation, name, hint)
        for name, field in schema.model_fields.items()
    }
    return schema.model_validate(data)


class MockChatModel(BaseChatModel):
    """Echo-style chat model. Text replies are prefixed with ``[mock]``."""

    model: str = "mock-chat"

    @property
    def _llm_type(self) -> str:
        return "sahm-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        hint = _last_human_text(messages)
        reply = AIMessage(
            content=f"[mock] Analysis notes for: {hint[:200]}",
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:  # noqa: ARG002
        """Accept tools but never call them -- keeps the ReAct loop to one turn."""
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:  # noqa: ARG002
        return RunnableLambda(lambda inp: fake_instance(schema, _last_human_text(inp)))


def build_chat_model(  # noqa: ARG001
    settings: AISettings,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> BaseChatModel:
    return MockChatModel(model=model or "mock-chat")
