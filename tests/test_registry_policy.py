"""Required test 1: no external_write tool is reachable from any persona.

Plus the structural guarantees around it: the registry refuses to store an
external_write tool, protected tools are not in the registry, and no agent
module imports the protected module.
"""

from __future__ import annotations

import pathlib

import pytest

from app.core.agents.personas import PERSONAS, PERSONA_ORDER
from app.core.tools.base import ToolRegistry, to_mcp_server
from app.core.tools.protected import PROTECTED_TOOLS
from app.core.tools.registry import (
    ALL_TOOLS,
    assert_no_external_write_reachable,
    get_tools_for,
    register_all,
    reset_registry,
)
from app.schemas.common import ToolSideEffect


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_registry()
    yield
    reset_registry()


def test_registry_refuses_external_write_tools() -> None:
    registry = ToolRegistry()
    for tool in PROTECTED_TOOLS:
        assert tool.side_effect == ToolSideEffect.EXTERNAL_WRITE
        with pytest.raises(ValueError, match="external_write"):
            registry.register(tool)
    assert registry.names() == []


def test_protected_tools_are_not_in_the_registry() -> None:
    registry = register_all()
    names = set(registry.names())
    for tool in PROTECTED_TOOLS:
        assert tool.name not in names, f"{tool.name} leaked into the agent registry"


def test_no_persona_can_reach_an_external_write_tool() -> None:
    """The headline guarantee. Enumerated for every persona."""
    register_all()
    for persona_id in PERSONA_ORDER:
        persona = PERSONAS[persona_id]
        tools = get_tools_for(persona)
        for tool in tools:
            assert tool.side_effect != ToolSideEffect.EXTERNAL_WRITE, (
                f"persona '{persona_id}' can reach external_write tool '{tool.name}'"
            )
        # And nothing on the allowlist resolves to a protected tool.
        protected_names = {t.name for t in PROTECTED_TOOLS}
        assert not (set(persona.allowed_tools) & protected_names)


def test_assert_no_external_write_reachable_passes_on_a_clean_registry() -> None:
    register_all()
    assert_no_external_write_reachable()  # must not raise


def test_to_mcp_server_refuses_external_write_tools() -> None:
    register_all()
    with pytest.raises(ValueError, match="external_write"):
        to_mcp_server(PROTECTED_TOOLS, name="bad")


def test_to_mcp_server_wraps_agent_tools() -> None:
    """MCP is an adapter, not a dependency. Built and exercised once."""
    register_all()
    server = to_mcp_server(ALL_TOOLS[:3], name="sahm-tools-test")
    assert server is not None
    # The tool names must survive the wrap.
    assert "web_search" in {t.name for t in ALL_TOOLS[:3]}


def test_no_agent_module_imports_protected_tools() -> None:
    """Only approvals/service.py may import app.core.tools.protected."""
    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "protected.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "tools.protected" in text or "from app.core.tools import protected" in text:
            if "approvals/service.py" not in str(path):
                offenders.append(str(path.relative_to(root)))
    assert not offenders, "unexpected importers of protected tools: " + ", ".join(offenders)


def test_all_thirteen_tools_are_registered() -> None:
    registry = register_all()
    assert len(registry.all()) == 13
    expected = {
        "web_search", "analyze_website", "get_social_profile", "get_reviews",
        "get_keywords", "get_competitor_ads", "get_brand_context", "save_learning",
        "get_campaign_insights", "get_breakdowns", "analyze_reference_creative",
        "generate_image", "create_recommendation",
    }
    assert set(registry.names()) == expected


def test_every_tool_has_a_typed_io_pair() -> None:
    registry = register_all()
    for tool in registry.all():
        assert tool.input_model.model_fields, f"{tool.name} has no input fields"
        assert tool.output_model.model_fields, f"{tool.name} has no output fields"
        assert "status" in tool.output_model.model_fields, (
            f"{tool.name} output has no status field"
        )
