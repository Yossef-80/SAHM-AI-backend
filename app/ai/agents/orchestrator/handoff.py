"""Small, dependency-free helpers for structured agent-to-agent handoffs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


_STRATEGY_FIELDS = (
    "marketing_strategy",
    "campaign_strategy",
    "content_calendar",
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_tool_args(messages: Sequence[Any], tool_name: str) -> dict[str, Any] | None:
    """Find args for the named submit tool, even if other tools ran in parallel."""

    for message in reversed(list(messages or [])):
        calls = _field(message, "tool_calls", []) or []
        for call in reversed(list(calls)):
            if _field(call, "name") != tool_name:
                continue
            args = _field(call, "args")
            if isinstance(args, Mapping):
                return dict(args)
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (TypeError, ValueError):
                    return None
                return dict(parsed) if isinstance(parsed, Mapping) else None
    return None


def strategy_handoff_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """Collect only the run-local structured strategy needed by Layla."""

    report = state.get("final_report")
    report = report if isinstance(report, Mapping) else {}
    payload: dict[str, Any] = {}

    for field in _STRATEGY_FIELDS:
        value = state.get(field) or report.get(field)
        if value:
            payload[field] = value

    for field in ("summary", "details", "missing_inputs"):
        value = report.get(field)
        if value:
            payload[field] = value

    # A text-only fallback keeps the handoff honest if the model did not call
    # one of the structured planning tools.
    if not payload and report:
        payload.update(report)
    return payload


def creative_handoff_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """Trim Omar's report to what Layla needs and make missing strategy explicit."""

    payload = strategy_handoff_payload(state)
    payload.pop("content_calendar", None)
    if not payload.get("campaign_strategy"):
        missing = list(payload.get("missing_inputs") or [])
        missing.append(
            "Omar did not return a structured campaign strategy; do not infer campaign settings from the marketing strategy alone."
        )
        payload["missing_inputs"] = list(dict.fromkeys(missing))
    return payload


def render_output(value: Any) -> str:
    """Render Pydantic/dict/message content consistently for prompts and UI."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, indent=2, default=str)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value or "")


def agent_output(state: Mapping[str, Any], submit_tool_name: str) -> Any:
    """Prefer the final report stored in state, then the named submit call."""

    if submit_tool_name == "submit_strategy_report" and state.get("final_report"):
        return state["final_report"]
    messages = state.get("messages") or []
    args = extract_tool_args(messages, submit_tool_name)
    if args is not None:
        return args
    if messages:
        last_message = messages[-1]
        return _field(last_message, "content", "")
    return ""
