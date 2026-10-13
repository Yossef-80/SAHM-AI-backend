"""LangChain callback that routes runtime events to the per-component loggers.

Attach it once per run (``config={"callbacks": [AILoggingCallback()]}``) and:

* every tool call      -> ``logs/tools/<tool_name>.log``   (input, duration, output preview, errors)
* every chat-model call -> ``logs/providers/llm.log``       (model, latency, token usage)

This is why tool files contain NO start/finish/timing boilerplate -- the
callback records it automatically for every tool, current and future.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger
from app.ai.loggers.providers.llm import logger as llm_logger
from app.ai.security import redact_sensitive_data

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


class AIToolAuditCallback(BaseCallbackHandler):
    """Collect sanitized tool telemetry for a later async database flush.

    LangChain callbacks are synchronous, while SQLAlchemy uses an async session.
    This collector buffers only bounded, redacted metadata; the orchestrator
    persists it after each graph run without constructing a core ToolContext.
    """

    raise_error = False
    _MAX_CHARS = 2000

    def __init__(self, agent: str | None = None) -> None:
        self.agent = agent
        self._lock = threading.Lock()
        self._starts: dict[UUID, tuple[float, str, Any]] = {}
        self._records: list[dict[str, Any]] = []

    @staticmethod
    def _safe_input(value: Any) -> Any:
        safe = redact_sensitive_data(value)
        if isinstance(safe, str):
            return safe[:AIToolAuditCallback._MAX_CHARS]
        if isinstance(safe, (dict, list)):
            try:
                encoded = json.dumps(safe, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(safe)[:AIToolAuditCallback._MAX_CHARS]
            if len(encoded) > AIToolAuditCallback._MAX_CHARS:
                return {"preview": encoded[:AIToolAuditCallback._MAX_CHARS]}
        return safe

    @staticmethod
    def _status(output: Any) -> str:
        value = getattr(output, "content", output)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return "ok"
            if isinstance(parsed, dict):
                status = parsed.get("status")
                if isinstance(status, str) and status:
                    return status[:32]
        return "ok"

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
        safe_input = self._safe_input(inputs if inputs is not None else input_str)
        with self._lock:
            self._starts[run_id] = (time.perf_counter(), str(name), safe_input)

    def _finish(
        self,
        run_id: UUID,
        *,
        output: Any = "",
        status: str | None = None,
        error: BaseException | None = None,
        name: str | None = None,
    ) -> None:
        with self._lock:
            started, saved_name, safe_input = self._starts.pop(
                run_id, (time.perf_counter(), name or "unknown_tool", {})
            )
            value = getattr(output, "content", output)
            summary = str(redact_sensitive_data(value))[: self._MAX_CHARS]
            self._records.append(
                {
                    "tool": saved_name,
                    "input": safe_input if isinstance(safe_input, dict) else {"value": safe_input},
                    "output_summary": summary,
                    "status": status or self._status(output),
                    "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
                    "error": (
                        f"{type(error).__name__}: {redact_sensitive_data(str(error))}"[: self._MAX_CHARS]
                        if error is not None
                        else None
                    ),
                }
            )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(run_id, output=output, name=kwargs.get("name"))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._finish(
            run_id,
            status="error",
            error=error,
            name=kwargs.get("name"),
        )

    def drain(self) -> list[dict[str, Any]]:
        """Return completed events exactly once; unfinished calls are discarded."""
        with self._lock:
            records, self._records = self._records, []
            self._starts.clear()
        return records


__all__ = ["AILoggingCallback", "AIToolAuditCallback"]
