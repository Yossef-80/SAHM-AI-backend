"""Specialized diagnostic logger for agent execution, graph states, and issue troubleshooting.

Writes to `logs/agents/diagnostic.log`. Used by the runtime, graphs, and callback handlers
to record:
- Agent step-by-step reasoning & LangGraph node transitions
- Router branch choices & criteria
- Tool invocation parameters, outputs, and errors
- Agent handoffs and state persistence
- Root-cause diagnostics when an agent run fails or falls back

Import it with::

    from app.ai.loggers.agents.diagnostic import logger
"""

from __future__ import annotations

from typing import Any

from app.ai.enums import LogKind
from app.ai.loggers.base import get_component_logger

logger = get_component_logger(LogKind.AGENT, "diagnostic")


def log_agent_diagnosis(
    event: str,
    agent: str | None,
    node: str | None = None,
    status: str = "ok",
    detail: str | None = None,
    error: BaseException | None = None,
    **extra: Any,
) -> None:
    """Convenience helper to record structured diagnostic context for troubleshooting."""
    payload: dict[str, Any] = {
        "agent": agent or "unknown",
        "node": node or "root",
        "status": status,
        **extra,
    }
    if detail:
        payload["detail"] = detail
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
        logger.error(event, **payload)
    elif status == "error":
        logger.error(event, **payload)
    elif status in ("warning", "fallback", "retry"):
        logger.warning(event, **payload)
    else:
        logger.info(event, **payload)


__all__ = ["logger", "log_agent_diagnosis"]
