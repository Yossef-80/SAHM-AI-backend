"""Tool registry assembly.

``register_all()`` is the single entry point that populates the process-wide
registry with every agent-callable tool. ``get_tools_for(persona)`` is the only
way agents obtain tools, and it enforces the two rules that matter:

1. A persona can only receive tools on its ``allowed_tools`` list.
2. No tool with ``side_effect == "external_write"`` is ever handed out. In
   fact the registry refuses to *store* such a tool at all (see base.py), and
   ``assert_no_external_write_reachable()`` proves it at import time.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.tools.base import Tool, ToolRegistry, get_registry
from app.core.tools.media import MEDIA_TOOLS
from app.core.tools.memory import MEMORY_TOOLS
from app.core.tools.meta_read import META_READ_TOOLS
from app.core.tools.records import RECORDS_TOOLS
from app.core.tools.research import RESEARCH_TOOLS

_logger = get_logger(__name__)

#: Every agent-callable tool. Protected (external_write) tools are deliberately
#: absent -- they live in tools/protected.py and are imported only by
#: approvals/service.py.
ALL_TOOLS: list[Tool] = [
    *RESEARCH_TOOLS,
    *MEMORY_TOOLS,
    *META_READ_TOOLS,
    *MEDIA_TOOLS,
    *RECORDS_TOOLS,
]

_registry = ToolRegistry()

#: True once ``register_all`` has populated ``_registry``.
_registered = False


def get_registry() -> ToolRegistry:
    """The process-wide registry."""
    return _registry


def register_all(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Populate the registry. Idempotent."""
    global _registered
    reg = registry or get_registry()
    if not _registered or not reg.names():
        reg.register_all(ALL_TOOLS)
        _registered = True
        _logger.info(
            "tools_registered",
            extra={"count": len(ALL_TOOLS), "tools": ",".join(t.name for t in ALL_TOOLS)},
        )
    assert_no_external_write_reachable(reg)
    return reg


def reset_registry() -> None:
    """Drop all registered tools and force a re-register.

    Used by tests. Must clear the ``_registered`` flag too, otherwise
    ``register_all`` becomes a permanent no-op after a reset.
    """
    global _registered
    _registry._tools.clear()
    _registered = False


def assert_no_external_write_reachable(registry: ToolRegistry | None = None) -> None:
    """Import-time guard: no external_write tool may be in the registry.

    Belt and braces. ``ToolRegistry.register`` already refuses them, but this
    makes the invariant explicit and testable, and it runs on every import of
    this module.
    """
    reg = registry or get_registry()
    offenders = [t.name for t in reg.all() if t.is_external_write()]
    if offenders:  # pragma: no cover - register() prevents this
        raise RuntimeError(
            "external_write tools reached the registry: " + ", ".join(offenders)
        )


def get_tools_for(persona: object) -> list[Tool]:
    """The tools a persona may use.

    ``persona`` is anything with an ``allowed_tools`` list (a Persona model or a
    stub). Unknown names are logged and skipped rather than crashing the run.
    """
    registry = register_all()
    allowed: list[str] = list(getattr(persona, "allowed_tools", []))
    tools = registry.for_names(allowed)
    # Defensive: even if a persona config were edited to include a protected
    # tool name, it cannot be in the registry, so this stays empty. Assert it.
    assert_no_external_write_reachable(registry)
    return tools


def get_tool_names_for(persona: object) -> list[str]:
    return [t.name for t in get_tools_for(persona)]


__all__ = [
    "ALL_TOOLS",
    "assert_no_external_write_reachable",
    "get_tool_names_for",
    "get_tools_for",
    "register_all",
    "reset_registry",
]
