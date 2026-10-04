"""LangChain callback that routes runtime events to the per-component loggers.

Attach it once per run (``config={"callbacks": [AILoggingCallback()]}``) and:

* every tool call      -> ``logs/tools/<tool_name>.log``   (input, duration, output preview, errors)
* every chat-model call -> ``logs/providers/llm.log``       (model, latency, token usage)

This is why tool files contain NO start/finish/timing boilerplate -- the
callback records it automatically for every tool, current and future.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger
from app.ai.loggers.providers.llm import logger as llm_logger

_PREVIEW_CHARS = 400


def _preview(value: Any) -> str:
    text = value.content if hasattr(value, "content") else value
    text = str(text)
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "..."


class AILoggingCallback(BaseCallbackHandler):
    """Routes tool + LLM events to their own log files."""

    #: Don't let a logging problem ever break an agent run.
    raise_error = False

    def __init__(self, agent: str | None = None) -> None:
        self.agent = agent
        self._starts: dict[UUID, tuple[float, str]] = {}

    # ------------------------------------------------------------- tools --
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown_tool"
        self._starts[run_id] = (time.perf_counter(), name)
        get_component_logger(LogKind.TOOL, name).info(
            "tool_started", agent=self.agent, input=inputs if inputs is not None else input_str
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        started, name = self._starts.pop(run_id, (time.perf_counter(), kwargs.get("name", "unknown_tool")))
        get_component_logger(LogKind.TOOL, name).info(
            "tool_finished",
            agent=self.agent,
            duration_ms=int((time.perf_counter() - started) * 1000),
            output=_preview(output),
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        started, name = self._starts.pop(run_id, (time.perf_counter(), kwargs.get("name", "unknown_tool")))
        get_component_logger(LogKind.TOOL, name).error(
            "tool_failed",
            agent=self.agent,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(error).__name__}: {error}",
        )

    # --------------------------------------------------------------- LLM --
    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], *, run_id: UUID, **kwargs: Any
    ) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or (serialized or {}).get("name")
        self._starts[run_id] = (time.perf_counter(), str(model))
        llm_logger.info(
            "llm_started",
            agent=self.agent,
            model=model,
            node=(kwargs.get("metadata") or {}).get("langgraph_node"),
            messages=sum(len(batch) for batch in messages),
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        started, model = self._starts.pop(run_id, (time.perf_counter(), "?"))
        usage: dict[str, Any] = {}
        tool_calls = 0
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    usage = dict(getattr(msg, "usage_metadata", None) or usage)
                    tool_calls += len(getattr(msg, "tool_calls", None) or [])
        llm_logger.info(
            "llm_finished",
            agent=self.agent,
            model=model,
            duration_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            tool_calls=tool_calls,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        started, model = self._starts.pop(run_id, (time.perf_counter(), "?"))
        llm_logger.error(
            "llm_failed",
            agent=self.agent,
            model=model,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(error).__name__}: {error}",
        )


__all__ = ["AILoggingCallback"]
