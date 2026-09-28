"""Layer 2: the Tool model, ToolContext, and the registry.

Tools are typed functions: a Pydantic input model, a Pydantic output model, and
an async handler. Everything the handler needs arrives in ``ToolContext`` --
never from globals -- which is what makes them unit-testable and MCP-ready.

MCP is an *adapter*, not a dependency: ``to_mcp_server`` wraps the same Tool
objects in a FastMCP server for external exposure, but the agent runtime calls
handlers through the registry directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.llm.base import LLMClient, ToolSpec
from app.core.logging import get_logger
from app.schemas.common import ToolSideEffect, ToolStatus

_logger = get_logger(__name__)


# ------------------------------------------------------------------ context --


@dataclass
class ToolContext:
    """Everything a tool handler is allowed to know.

    Handlers never read globals or ambient state. ``llm`` and ``db`` are passed
    in so a test can substitute either.
    """

    workspace_id: str
    business_id: str
    db: AsyncSession
    llm: LLMClient
    request_id: str
    campaign_id: str | None = None
    creative_id: str | None = None
    insight_id: str | None = None
    persona_id: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None
    settings: Settings | None = None
    #: Data the user pasted/uploaded, for tools that support a manual fallback
    #: (e.g. competitor ads the Meta Ad Library API does not cover).
    user_supplied: dict[str, Any] = field(default_factory=dict)

    def require_settings(self) -> Settings:
        if self.settings is None:
            from app.config import get_settings

            self.settings = get_settings()
        return self.settings


# -------------------------------------------------------------- tool output --


class ToolOutput(BaseModel):
    """Base for every tool result.

    The brief's ``-> list[SearchResult]`` shapes are carried as *fields* on
    these wrappers, so that every result can also report ``status`` and
    ``source``. A tool that cannot reach its source returns
    ``status="unavailable"`` with a human-readable reason and never fabricates
    data.
    """

    status: ToolStatus = ToolStatus.OK
    source: str = ""
    unavailable_reason: str | None = None

    @classmethod
    def unavailable(cls, *, source: str, reason: str) -> "ToolOutput":
        return cls(status=ToolStatus.UNAVAILABLE, source=source, unavailable_reason=reason)

    @classmethod
    def partial(cls, *, source: str, reason: str = "") -> "ToolOutput":
        return cls(status=ToolStatus.PARTIAL, source=source, unavailable_reason=reason or None)


# -------------------------------------------------------------------- tool ----


class Tool(BaseModel):
    """A typed, MCP-ready tool. Not MCP-dependent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    #: Written for the LLM: when to use it, and when not to.
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    side_effect: ToolSideEffect = ToolSideEffect.NONE
    handler: Callable[[BaseModel, ToolContext], Awaitable[BaseModel]]

    def to_spec(self) -> ToolSpec:
        return ToolSpec.from_model(self.name, self.description, self.input_model)

    def is_external_write(self) -> bool:
        return self.side_effect == ToolSideEffect.EXTERNAL_WRITE


ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[BaseModel]]


# ---------------------------------------------------------------- registry ----


class ToolRegistry:
    """Name -> Tool. Agents only ever see a filtered view of this."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.is_external_write():
            # Structural guarantee: external-write tools are never reachable
            # from any agent, so they must never enter the registry at all.
            raise ValueError(
                f"tool '{tool.name}' has side_effect='external_write' and cannot be "
                "registered for agent use. It belongs in tools/protected.py and may "
                "only be called from approvals/service.py."
            )
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name '{tool.name}'")
        self._tools[tool.name] = tool
        return tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        from app.core.errors import ToolNotFoundError

        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"tool '{name}' is not registered")
        return tool

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def for_names(self, names: list[str]) -> list[Tool]:
        """Resolve tool names to Tools, ignoring unknown ones.

        Unknown names are a persona-config bug, not a runtime error; the runtime
        reports them rather than crashing the request.
        """
        resolved: list[Tool] = []
        for name in names:
            if name in self._tools:
                resolved.append(self._tools[name])
            else:
                _logger.warning("unknown_tool_in_allowlist", extra={"tool": name})
        return resolved

    def specs_for(self, names: list[str]) -> list[ToolSpec]:
        return [tool.to_spec() for tool in self.for_names(names)]


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """The process-wide registry."""
    return _registry


def reset_registry() -> None:
    _registry._tools.clear()


# ----------------------------------------------------------------- logging ----


async def log_tool_call(
    ctx: ToolContext,
    *,
    tool: str,
    tool_input: BaseModel | dict[str, Any],
    output_summary: str,
    status: str,
    duration_ms: int,
    error: str | None = None,
) -> None:
    """Append one row to agent_tool_calls.

    This is what powers the "what the AI did" view and post-hoc debugging. It
    never blocks a run: a logging failure is swallowed.
    """
    from app.db.models import AgentToolCall
    from app.core.logging import redact
    import uuid

    payload = (
        tool_input.model_dump(mode="json")
        if isinstance(tool_input, BaseModel)
        else dict(tool_input)
    )
    try:
        ctx.db.add(
            AgentToolCall(
                id=f"tc_{uuid.uuid4().hex[:24]}",
                business_id=ctx.business_id,
                task_id=ctx.task_id,
                conversation_id=ctx.conversation_id,
                persona_id=ctx.persona_id,
                tool=tool,
                input=redact(payload),
                output_summary=output_summary[:2000],
                status=status,
                duration_ms=duration_ms,
                error=(error or "")[:2000] or None,
                request_id=ctx.request_id,
            )
        )
        await ctx.db.flush()
    except Exception as exc:  # pragma: no cover - logging must never break a run
        _logger.warning("tool_call_log_failed", extra={"tool": tool, "error": str(exc)})


# --------------------------------------------------------------- MCP adapter --


def to_mcp_server(tools: list[Tool], *, name: str = "sahm-tools") -> Any:
    """Wrap registry tools in a FastMCP server.

    MCP is an adapter, not a dependency: the agent runtime never goes through
    this. It exists so the same tool set can be exposed to external MCP clients
    without duplicating definitions. Imported lazily so the core never needs
    fastmcp installed.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "fastmcp is not installed; install the 'dev' extra to use to_mcp_server()"
        ) from exc

    server = FastMCP(name)

    for tool in tools:
        if tool.is_external_write():
            # Same structural rule as the registry: never expose external writes.
            raise ValueError(
                f"refusing to expose external_write tool '{tool.name}' over MCP"
            )

        def _make(t: Tool):
            async def _handler(payload: dict[str, Any]) -> dict[str, Any]:
                # An MCP caller supplies a ToolContext; without one we cannot
                # run the handler, so we expose the schema and validate input.
                validated = t.input_model.model_validate(payload)
                return validated.model_dump(mode="json")

            _handler.__name__ = t.name
            _handler.__doc__ = t.description
            return _handler

        server.tool(name=tool.name, description=tool.description)(_make(tool))

    return server


__all__ = [
    "Tool",
    "ToolContext",
    "ToolHandler",
    "ToolOutput",
    "ToolRegistry",
    "get_registry",
    "log_tool_call",
    "reset_registry",
    "to_mcp_server",
]
