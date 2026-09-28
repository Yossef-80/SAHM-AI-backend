"""Layer 1: the single LLM abstraction.

No agent, tool, or router in this codebase imports a provider SDK. They all
talk to ``LLMClient``. ``claude_client`` / ``openai_client`` are the only
modules that know what a provider's HTTP API looks like, and they get their
model strings from ``config.py`` only.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from app.core.errors import LLMOutputError, LLMProviderError
from app.schemas.common import Usage

TierName = Literal["fast", "strong"]


# ------------------------------------------------------------------ models ----


class ToolCall(BaseModel):
    """A provider-agnostic tool invocation request from the model."""

    id: str
    name: str
    arguments: dict[str, Any] = {}


class Msg(BaseModel):
    """One message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    # Set on assistant messages that request tool calls.
    tool_calls: list[ToolCall] = []
    # Set on tool result messages, to match them back to their call.
    tool_call_id: str | None = None
    name: str | None = None


class ToolSpec(BaseModel):
    """What the LLM sees of a tool: name, description, JSON schema."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def from_model(cls, name: str, description: str, model: type[BaseModel]) -> ToolSpec:
        return cls(name=name, description=description, input_schema=json_schema_for(model))


class LLMResult(BaseModel):
    """Everything a caller needs from one completion."""

    text: str = ""
    #: Validated instance of ``response_schema`` when one was requested.
    parsed: BaseModel | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage = Usage()
    model: str = ""
    latency_ms: int = 0

    model_config = {"arbitrary_types_allowed": True}


@runtime_checkable
class LLMClient(Protocol):
    """The only interface the rest of the system may depend on."""

    async def complete(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        tier: TierName = "strong",
        max_tokens: int = 2000,
        temperature: float = 0.4,
    ) -> LLMResult:
        """One completion. Retries transient errors; validates structured output."""
        ...


# ---------------------------------------------------------------- helpers ----


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a Pydantic model, with ``additionalProperties`` off.

    Providers are stricter about schema shape than Pydantic's default, so we
    strip the Pydantic-specific keys and force a closed object.
    """
    schema = model.model_json_schema()
    # Resolve $defs/$ref inlining so providers that dislike $ref still work.
    defs = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                target = defs.get(ref.rsplit("/", 1)[-1], {})
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return _resolve({**target, **merged})
            # Pydantic emits a human-readable "title" on every node. Drop those,
            # but never drop a *property* literally named "title".
            return {
                k: _resolve(v)
                for k, v in node.items()
                if not (k == "title" and isinstance(v, str))
            }
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    resolved = _resolve(schema)
    resolved.pop("$defs", None)
    resolved.pop("title", None)
    if resolved.get("type") == "object":
        resolved.setdefault("additionalProperties", False)
    return resolved


def schema_instruction(model: type[BaseModel]) -> str:
    """The output-schema instruction appended to the system prompt."""
    schema = json.dumps(json_schema_for(model), ensure_ascii=False, indent=2)
    return (
        "You must reply with a single JSON object and nothing else -- no prose, "
        "no markdown fences. It must validate against this JSON schema:\n"
        f"{schema}\n"
        "If a value is not available from the tool results or the user's input, "
        "do not invent it: use null, an empty list, or the string \"not available\"."
    )


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models occasionally wrap JSON in prose or ```json fences despite
    instructions. This is deliberately tolerant on the way in and strict on
    validation afterwards.
    """
    if not text or not text.strip():
        raise LLMOutputError("empty response from model")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fall back to a brace-balanced scan.
    start = candidate.find("{")
    if start == -1:
        raise LLMOutputError("no JSON object found in model response")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = candidate[start : idx + 1]
                try:
                    parsed = json.loads(blob)
                except json.JSONDecodeError as exc:
                    raise LLMOutputError(f"malformed JSON in response: {exc}") from exc
                if isinstance(parsed, dict):
                    return parsed
                break
    raise LLMOutputError("no JSON object found in model response")


def format_validation_error(exc: ValidationError) -> str:
    """Compact, model-readable rendering of a Pydantic validation error."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']} (got {err['input']!r:.200})")
    return "; ".join(parts)


# ------------------------------------------------------------- base client ----


class BaseLLMClient(ABC):
    """Template method implementing retries, structured output and usage.

    Subclasses implement exactly one method: ``_complete_once``. That keeps
    retry/backoff/validation semantics identical across providers, which is
    where subtle per-provider drift would otherwise creep in.
    """

    def __init__(self, *, model: str, timeout: float, max_retries: int, base_delay: float) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay

    # -- to implement --------------------------------------------------------

    @abstractmethod
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
        """A single provider call. No retries, no validation."""

    # -- public API ----------------------------------------------------------

    async def complete(
        self,
        *,
        system: str,
        messages: list[Msg],
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        tier: TierName = "strong",
        max_tokens: int = 2000,
        temperature: float = 0.4,
    ) -> LLMResult:
        started = time.monotonic()
        result = await self._call_with_retries(
            system=system,
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            tier=tier,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result.latency_ms = int((time.monotonic() - started) * 1000)
        result.model = self.model
        if result.usage.model == "":
            result.usage.model = self.model
        self._log_usage(result)
        return result

    # -- internals -----------------------------------------------------------

    async def _call_with_retries(
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
        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.max_retries:
            try:
                result = await self._complete_once(
                    system=system,
                    messages=messages,
                    tools=tools,
                    response_schema=response_schema,
                    tier=tier,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if response_schema is not None:
                    return await self._validate(
                        result,
                        response_schema,
                        system,
                        messages,
                        tools,
                        tier,
                        max_tokens,
                        temperature,
                    )
                return result
            except LLMOutputError:
                # Structured output failed after the repair retry -- do not retry.
                raise
            except LLMProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.max_retries:
                    raise
                delay = self.base_delay * (2**attempt)
                await _sleep(delay)
                attempt += 1
        raise last_error or LLMProviderError("llm call failed")

    async def _validate(
        self,
        result: LLMResult,
        schema: type[BaseModel],
        system: str,
        messages: list[Msg],
        tools: list[ToolSpec] | None,
        tier: TierName,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        """Validate structured output; on failure retry ONCE with the error."""
        try:
            payload = extract_json(result.text)
            result.parsed = schema.model_validate(payload)
            return result
        except (ValidationError, LLMOutputError) as exc:
            reason = (
                format_validation_error(exc)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
        # One repair attempt: append the validation error and ask again.
        repair_messages = list(messages) + [
            Msg(role="assistant", content=result.text),
            Msg(
                role="user",
                content=(
                    "Your previous reply did not satisfy the required output schema. "
                    f"Problem: {reason}\n"
                    "Reply again with a single valid JSON object only."
                ),
            ),
        ]
        try:
            repaired = await self._complete_once(
                system=system,
                messages=repair_messages,
                tools=None,
                response_schema=schema,
                tier=tier,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except LLMProviderError as exc:
            raise LLMOutputError(
                f"repair attempt failed: {exc.message}", detail={"original": reason}
            ) from exc
        try:
            repaired.parsed = schema.model_validate(extract_json(repaired.text))
            return repaired
        except (ValidationError, LLMOutputError) as exc2:
            final = (
                format_validation_error(exc2)
                if isinstance(exc2, ValidationError)
                else str(exc2)
            )
            raise LLMOutputError(
                "model failed to produce schema-valid output after one repair retry",
                detail={"first_error": reason, "second_error": final},
            ) from exc2

    def _log_usage(self, result: LLMResult) -> None:
        from app.core.logging import get_logger

        get_logger(__name__).info(
            "llm_call",
            extra={
                "model": result.model,
                "tokens_in": result.usage.tokens_in,
                "tokens_out": result.usage.tokens_out,
                "latency_ms": result.latency_ms,
            },
        )


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def is_retryable_status(status_code: int) -> bool:
    """429 and 5xx are transient; everything else is a real error."""
    return status_code == 429 or 500 <= status_code <= 599


def is_retryable_exception(exc: Exception) -> bool:
    """Timeouts and connection failures are worth another attempt."""
    import httpx

    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError))


__all__ = [
    "BaseLLMClient",
    "LLMClient",
    "LLMResult",
    "Msg",
    "TierName",
    "ToolCall",
    "ToolSpec",
    "extract_json",
    "format_validation_error",
    "json_schema_for",
    "schema_instruction",
]
