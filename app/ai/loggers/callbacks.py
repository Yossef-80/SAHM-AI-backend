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
from app.ai.loggers.cost import estimate_token_cost, format_cost
from app.ai.loggers.providers.llm import logger as llm_logger
from app.ai.security import redact_sensitive_data

_PREVIEW_CHARS = 400


def _preview(value: Any) -> str:
    text = value.content if hasattr(value, "content") else value
    text = str(text)
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "..."


def _diagnose_llm_error(error: BaseException, model: str | None) -> str:
    err_str = str(error).lower()
    err_type = type(error).__name__
    if "connect" in err_str or "connection" in err_str or "unreachable" in err_str:
        return "Network/Connection Failure: verify the LLM server or Ollama instance is running and reachable."
    if "timeout" in err_str or "timed out" in err_str:
        return "LLM Request Timed Out: increase AI_LLM_TIMEOUT_SECONDS or reduce context size."
    if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
        return "Rate Limit or Quota Exceeded: check your provider credit balance or switch to free local Ollama/Mock."
    if "auth" in err_str or "401" in err_str or "api_key" in err_str or "unauthorized" in err_str:
        return "Authentication Error: verify provider API key is set and valid in .env."
    if "context_length" in err_str or "maximum context" in err_str or "token limit" in err_str:
        return "Context Window Exceeded: prompt is too large. Enable session compression or reduce history tokens."
    return f"Model call failed ({err_type}). Inspect stack trace and check input payload."


class AILoggingCallback(BaseCallbackHandler):
    """Routes tool, LLM, and agent events to their own diagnostic log files."""

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
        node = (kwargs.get("metadata") or {}).get("langgraph_node")
        self._starts[run_id] = (time.perf_counter(), str(model))

        prompt_preview = None
        if messages and messages[0]:
            prompt_preview = _preview(getattr(messages[0][-1], "content", str(messages[0][-1])))

        llm_logger.info(
            "llm_started",
            agent=self.agent,
            model=model,
            node=node,
            messages=sum(len(batch) for batch in messages),
            prompt_preview=prompt_preview,
        )
        if self.agent:
            get_component_logger(LogKind.AGENT, self.agent).info(
                "llm_call_initiated",
                model=model,
                node=node,
                messages_count=sum(len(batch) for batch in messages),
                prompt_preview=prompt_preview,
            )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        started, model = self._starts.pop(run_id, (time.perf_counter(), "?"))
        usage: dict[str, Any] = {}
        tool_calls: list[str] = []
        response_preview = ""
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    usage = dict(getattr(msg, "usage_metadata", None) or usage)
                    t_calls = getattr(msg, "tool_calls", None) or []
                    tool_calls.extend([tc.get("name", "unknown") for tc in t_calls if isinstance(tc, dict)])
                    if not response_preview and hasattr(msg, "content"):
                        response_preview = _preview(msg.content)
                elif hasattr(gen, "text") and not response_preview:
                    response_preview = _preview(gen.text)

        duration_ms = int((time.perf_counter() - started) * 1000)
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        tot_tok = usage.get("total_tokens") or ((in_tok or 0) + (out_tok or 0) if in_tok or out_tok else None)

        cost_usd = estimate_token_cost(model, in_tok, out_tok)
        cost_str = format_cost(cost_usd)

        llm_logger.info(
            "llm_finished",
            agent=self.agent,
            model=model,
            duration_ms=duration_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            estimated_cost_usd=cost_usd,
            cost=cost_str,
            tool_calls=len(tool_calls),
            tool_names=tool_calls if tool_calls else None,
            preview=response_preview if response_preview else None,
        )
        if self.agent:
            get_component_logger(LogKind.AGENT, self.agent).info(
                "llm_call_completed",
                model=model,
                duration_ms=duration_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost=cost_str,
                tool_calls=tool_calls if tool_calls else None,
            )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        started, model = self._starts.pop(run_id, (time.perf_counter(), "?"))
        duration_ms = int((time.perf_counter() - started) * 1000)
        diagnosis = _diagnose_llm_error(error, model)

        llm_logger.error(
            "llm_failed",
            agent=self.agent,
            model=model,
            duration_ms=duration_ms,
            error=f"{type(error).__name__}: {error}",
            diagnosis=diagnosis,
        )
        if self.agent:
            get_component_logger(LogKind.AGENT, self.agent).error(
                "llm_call_failed",
                model=model,
                duration_ms=duration_ms,
                error=f"{type(error).__name__}: {error}",
                diagnosis=diagnosis,
            )
        get_component_logger(LogKind.AGENT, "diagnostic").error(
            "llm_failure_diagnosed",
            agent=self.agent,
            model=model,
            duration_ms=duration_ms,
            error_type=type(error).__name__,
            error_message=str(error),
            remediation=diagnosis,
        )

    # ----------------------------------------------------------- agent graph --
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "agent_node"
        node = (kwargs.get("metadata") or {}).get("langgraph_node") or name
        self._starts[run_id] = (time.perf_counter(), str(node))
        if self.agent:
            get_component_logger(LogKind.AGENT, self.agent).info(
                "agent_step_started",
                node=node,
                agent=self.agent,
            )

    def on_chain_end(self, outputs: dict[str, Any] | Any, *, run_id: UUID, **kwargs: Any) -> None:
        started, node = self._starts.pop(run_id, (time.perf_counter(), "agent_node"))
        duration_ms = int((time.perf_counter() - started) * 1000)
        if self.agent:
            get_component_logger(LogKind.AGENT, self.agent).info(
                "agent_step_finished",
                node=node,
                agent=self.agent,
                duration_ms=duration_ms,
            )

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        started, node = self._starts.pop(run_id, (time.perf_counter(), "agent_node"))
        duration_ms = int((time.perf_counter() - started) * 1000)
        if self.agent:
            get_component_logger(LogKind.AGENT, self.agent).error(
                "agent_step_failed",
                node=node,
                agent=self.agent,
                duration_ms=duration_ms,
                error=f"{type(error).__name__}: {error}",
            )
        get_component_logger(LogKind.AGENT, "diagnostic").error(
            "agent_node_failed",
            node=node,
            agent=self.agent,
            duration_ms=duration_ms,
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
