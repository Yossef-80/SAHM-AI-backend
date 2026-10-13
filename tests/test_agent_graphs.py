"""Compiled LangGraph smoke tests using scripted offline chat responses."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from tests.test_omar_layla_handoff_flow import _ScriptedChat, _tool_call


@pytest.mark.parametrize(
    ("agent", "module_path", "submit_tool", "arguments"),
    [
        (
            "nour",
            "app.ai.agents.nour.graph",
            "submit_research_report",
            {
                "market_overview": "A static offline report with source-backed observations.",
                "competitor_summary": [
                    {
                        "brand": "River Roast",
                        "positioning": "Small-batch Cairo coffee",
                        "strengths": ["The page states free delivery."],
                        "weaknesses_or_gaps": [],
                    }
                ],
                "pricing_insights": ["An explicit EGP price was present."],
                "offer_insights": [],
                "shipping_insights": ["The page mentions free delivery."],
                "search_insights": [],
                "social_insights": [],
                "opportunities_for_target_brand": [],
                "risks": [],
            },
        ),
        (
            "layla",
            "app.ai.agents.layla.graph",
            "submit_creative_brief",
            {"summary": "A warm workplace creative brief.", "details": ["Keep the CTA explicit."]},
        ),
        (
            "karim",
            "app.ai.agents.karim.graph",
            "submit_performance_report",
            {"summary": "Only supplied metrics are reported.", "details": ["No live Meta call made."]},
        ),
        (
            "salma",
            "app.ai.agents.salma.graph",
            "submit_optimization_plan",
            {"summary": "A proposal only.", "details": ["Human approval is still required."]},
        ),
    ],
)
async def test_compiled_specialist_graph_submits_its_named_report(
    agent: str,
    module_path: str,
    submit_tool: str,
    arguments: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(module_path)
    events: list[str] = []
    chat = _ScriptedChat(agent, [_tool_call(submit_tool, arguments, f"{agent}-submit")], events)
    monkeypatch.setattr(module, "get_ai_settings", lambda: SimpleNamespace(llm_provider="mock"))
    monkeypatch.setattr(module, "get_llm", lambda **_kwargs: SimpleNamespace(chat=chat))

    graph = getattr(module, f"{agent}_graph")
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content=f"Complete one offline {agent} task.")]},
        {"configurable": {"thread_id": f"thread-{agent}", "business_id": "biz_test"}},
    )

    assert submit_tool in chat.bound_tools
    assert events == [agent]
    final_message = state["messages"][-1]
    assert final_message.type == "tool"
    assert final_message.name == submit_tool
    assert final_message.content


def test_langgraph_submit_tools_remain_separate_from_meta_write_tools() -> None:
    """The legacy LangGraphs expose only their declared, non-Meta tool sets."""
    graph_modules = [
        importlib.import_module(f"app.ai.agents.{agent}.graph")
        for agent in ("nour", "omar", "layla", "karim", "salma")
    ]
    for module in graph_modules:
        tool_names = [tool.name for tool in module.NOUR_TOOLS] if hasattr(module, "NOUR_TOOLS") else None
        if tool_names is None:
            tool_names = [
                tool.name
                for attr in ("OMAR_TOOLS", "LAYLA_TOOLS", "KARIM_TOOLS", "SALMA_TOOLS")
                for tool in getattr(module, attr, [])
            ]
        assert tool_names
        assert not any("meta" in name.lower() and "submit" not in name.lower() for name in tool_names)
