"""Deterministic fake LLM.

Two jobs:
1. Let the whole graph run with no API keys (offline dev + CI).
2. Behave predictably in tests, including a scripted mode for the
   structured-output repair retry.

It never calls a network. Given a JSON schema it can synthesise a *valid*
instance for any Pydantic model, which is what makes the "every TaskSpec
output validates" test possible without a real provider.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from app.core.llm.base import BaseLLMClient, LLMResult, Msg, ToolCall, ToolSpec, TierName
from app.schemas.common import Usage

# ------------------------------------------------------------------ samples --


def _sample_string(schema: dict[str, Any], depth: int) -> str:
    min_len = int(schema.get("minLength", 1) or 1)
    fmt = schema.get("format", "")
    if fmt == "date":
        return "2026-01-15"
    if fmt == "date-time":
        return "2026-01-15T10:00:00Z"
    if fmt in ("email",):
        return "brand@example.com"
    if fmt in ("uri", "url"):
        return "https://example.com"
    enum = schema.get("enum")
    if enum:
        return str(enum[0])
    if min_len > 1:
        return "x" * min_len
    return "sample"


def _sample_number(schema: dict[str, Any]) -> float:
    if "minimum" in schema:
        return float(schema["minimum"])
    if "maximum" in schema:
        return float(min(schema["maximum"], 1.0))
    return 1.0


def _sample_int(schema: dict[str, Any]) -> int:
    if "minimum" in schema:
        return int(schema["minimum"])
    if "maximum" in schema:
        return int(min(schema["maximum"], 1))
    return 1


def sample_from_schema(schema: dict[str, Any], depth: int = 0) -> Any:
    """Build a value that satisfies ``schema``. Deterministic, no randomness."""
    if depth > 6:
        return None
    # A declared default is by definition a valid value, so prefer it.
    if "default" in schema and schema.get("type") != "object":
        return schema["default"]
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            if option.get("type") != "null":
                return sample_from_schema(option, depth + 1)
        return None
    if "oneOf" in schema:
        return sample_from_schema(schema["oneOf"][0], depth + 1)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]

    kind = schema.get("type")
    if kind is None:
        # Untyped node: guess from the keys present.
        if "properties" in schema:
            kind = "object"
        elif "items" in schema:
            kind = "array"
        else:
            return None

    if kind == "string":
        return _sample_string(schema, depth)
    if kind == "integer":
        return _sample_int(schema)
    if kind == "number":
        return _sample_number(schema)
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    if kind == "array":
        items = schema.get("items", {})
        if not isinstance(items, dict):
            return []
        min_items = int(schema.get("minItems", 1) or 0)
        count = max(min_items, 1) if min_items else 0
        return [sample_from_schema(items, depth + 1) for _ in range(count)]
    if kind == "object":
        out: dict[str, Any] = {}
        for name, sub in (schema.get("properties") or {}).items():
            out[name] = sample_from_schema(sub, depth + 1)
        return out
    return None


def sample_for_model(model: type[BaseModel]) -> dict[str, Any]:
    """A valid dict payload for a Pydantic model."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    resolved = _inline(schema, defs)
    return sample_from_schema(resolved)


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            target = defs.get(node["$ref"].rsplit("/", 1)[-1], {})
            merged = {k: v for k, v in node.items() if k != "$ref"}
            return _inline({**target, **merged}, defs)
        return {
            k: _inline(v, defs)
            for k, v in node.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


# ------------------------------------------------------------------- client --


class MockClient(BaseLLMClient):
    """Deterministic LLM stand-in."""

    def __init__(
        self,
        model: str = "mock-strong-1",
        *,
        emit_tool_calls: bool = True,
        script: list[dict[str, Any]] | None = None,
        latency_ms: int = 0,
        max_tool_calls_per_turn: int = 4,
    ) -> None:
        super().__init__(
            model=model, timeout=1.0, max_retries=0, base_delay=0.0
        )
        self.emit_tool_calls = emit_tool_calls
        self.script = list(script or [])
        self.latency_ms = latency_ms
        # Real models batch independent tool calls in one turn; so does this.
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.calls: list[dict[str, Any]] = []

    # -- helpers -------------------------------------------------------------

    def _called_tools(self, messages: list[Msg]) -> set[str]:
        called: set[str] = set()
        for msg in messages:
            if msg.role == "tool" and msg.name:
                called.add(msg.name)
            for tc in msg.tool_calls:
                called.add(tc.name)
        return called

    # -- the one abstract method --------------------------------------------

    async def _complete_once(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolSpec] | None,
        response_schema: type[BaseModel] | None,
        tier: TierName,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        self.calls.append(
            {
                "tier": tier,
                "has_tools": bool(tools),
                "has_schema": response_schema.__name__ if response_schema else None,
                "message_count": len(messages),
            }
        )
        usage = Usage(
            tokens_in=max(1, len(system) // 4 + sum(len(m.content or "") for m in messages) // 4),
            tokens_out=240,
            model=self.model,
            latency_ms=self.latency_ms,
        )

        # Scripted responses take precedence (tests drive this explicitly).
        if self.script:
            item = self.script.pop(0)
            return LLMResult(
                text=item.get("text", ""),
                tool_calls=[ToolCall(**tc) for tc in item.get("tool_calls", [])],
                usage=usage,
                model=self.model,
            )

        # Emit a batch of tool calls per turn until every offered tool has been
        # tried. Matches how real models behave, and keeps multi-tool personas
        # inside the iteration cap.
        if tools and self.emit_tool_calls and response_schema is None:
            pending = [t for t in tools if t.name not in self._called_tools(messages)]
            if pending:
                batch = pending[: self.max_tool_calls_per_turn]
                calls = [
                    ToolCall(
                        id=f"mock-{spec.name}",
                        name=spec.name,
                        arguments=sample_from_schema(spec.input_schema),
                    )
                    for spec in batch
                ]
                return LLMResult(
                    text=f"Calling {', '.join(c.name for c in calls)}.",
                    tool_calls=calls,
                    usage=usage,
                    model=self.model,
                )

        if response_schema is not None:
            payload = sample_for_model(response_schema)
            return LLMResult(
                text=json.dumps(payload, ensure_ascii=False),
                usage=usage,
                model=self.model,
            )

        return LLMResult(
            text="This is a deterministic mock response. Configure a real provider "
            "to get model-generated content.",
            usage=usage,
            model=self.model,
        )


__all__ = ["MockClient", "sample_for_model", "sample_from_schema"]
