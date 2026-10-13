from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.ai.agents.orchestrator import brain
from app.ai.memory.brand_memory import BrandProfile
from app.db.models import AgentToolCall


class _FakeRouter:
    def __init__(self, decision: brain.RoutingDecision, events: list[str] | None = None) -> None:
        self.decision = decision
        self.events = events

    async def ainvoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> brain.RoutingDecision:
        if self.events is not None:
            self.events.append("router")
        return self.decision


class _FailingFormatter:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    async def ainvoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> Any:
        if self.events is not None:
            self.events.append("formatter")
        raise RuntimeError("formatter is intentionally mocked")


class _FakeChat:
    def __init__(self, decision: brain.RoutingDecision, events: list[str] | None = None) -> None:
        self.events = events
        self.router = _FakeRouter(decision, events)

    def with_structured_output(self, schema: type[Any]) -> Any:
        if schema is brain.RoutingDecision:
            return self.router
        return _FailingFormatter(self.events)


def _fake_handle(chat: _FakeChat) -> SimpleNamespace:
    return SimpleNamespace(chat=chat, structured=chat.with_structured_output)


class _FakeGraph:
    def __init__(self, name: str, result: dict[str, Any], order: list[str]) -> None:
        self.name = name
        self.result = result
        self.order = order
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.order.append(self.name)
        self.calls.append((payload, config))
        return self.result


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    decision: brain.RoutingDecision,
    *,
    omar_result: dict[str, Any],
    layla_result: dict[str, Any],
) -> tuple[list[str], _FakeGraph, _FakeGraph]:
    order: list[str] = []
    omar = _FakeGraph("omar", omar_result, order)
    layla = _FakeGraph("layla", layla_result, order)
    chat = _FakeChat(decision)
    handle = SimpleNamespace(chat=chat, structured=chat.with_structured_output)

    monkeypatch.setattr(brain, "get_ai_settings", lambda: SimpleNamespace(llm_provider="mock"))
    monkeypatch.setattr(brain, "get_llm", lambda **kwargs: handle)
    monkeypatch.setattr(brain, "omar_graph", omar)
    monkeypatch.setattr(brain, "layla_graph", layla)
    monkeypatch.setattr(brain, "nour_graph", _FakeGraph("nour", {}, order))
    monkeypatch.setattr(brain, "karim_graph", _FakeGraph("karim", {}, order))
    monkeypatch.setattr(brain, "salma_graph", _FakeGraph("salma", {}, order))
    return order, omar, layla


def test_combined_workflow_passes_omars_strategy_to_layla(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = brain.RoutingDecision(
        agent="layla",
        reason="The user requested campaign strategy and ad creative.",
        workflow="omar_then_layla",
    )
    omar_result = {
        "final_report": {
            "summary": "Focus on office subscriptions.",
            "details": ["Lead with convenience."],
            "marketing_strategy": {"positioning": "Reliable office coffee."},
            "campaign_strategy": {
                "objective": "leads",
                "key_messages": ["Fresh coffee at work."],
                "missing_inputs": ["confirmed CTA"],
            },
        },
        "marketing_strategy": {"positioning": "Reliable office coffee."},
        "campaign_strategy": {
            "objective": "leads",
            "key_messages": ["Fresh coffee at work."],
            "missing_inputs": ["confirmed CTA"],
        },
    }
    layla_result = {
        "messages": [SimpleNamespace(content={"summary": "Three ad concepts."}, tool_calls=[])]
    }
    order, _omar, layla = _install_fakes(
        monkeypatch,
        decision,
        omar_result=omar_result,
        layla_result=layla_result,
    )

    result = asyncio.run(
        brain.orchestrate(
            "اعمل استراتيجية الحملة والإعلان",
            [],
            business_id="biz_test",
        )
    )

    assert order == ["omar", "layla"]
    layla_messages, layla_config = layla.calls[0]
    handoff_text = layla_messages["messages"][-1].content
    assert '"objective": "leads"' in handoff_text
    assert "confirmed CTA" in handoff_text
    assert layla_config["configurable"]["business_id"] == "biz_test"
    assert "لم يتم إنشاء أو نشر" in result.blocks[0].text


def test_saved_settings_profile_reaches_agent_prompt_and_tool_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = brain.RoutingDecision(agent="nour", reason="Competitor research request.")
    _install_fakes(
        monkeypatch,
        decision,
        omar_result={},
        layla_result={"messages": [SimpleNamespace(content="unused", tool_calls=[])]},
    )
    saved_profile = {
        "name": "Sotra",
        "industry": "Coffee shop",
        "website_url": "https://sotra.example",
        "country": "EG",
        "target_audience": "Office workers in Cairo",
        "competitors": ["Known competitor"],
    }

    asyncio.run(
        brain.orchestrate(
            "Identify competitors and analyze their weaknesses.",
            [],
            business_id="biz_test",
            forced_agent="nour",
            context={"business_profile_context": saved_profile},
        )
    )

    nour_graph = brain.nour_graph
    payload, graph_config = nour_graph.calls[0]
    prompt_text = " ".join(str(getattr(message, "content", "")) for message in payload["messages"])
    assert "Business name: Sotra" in prompt_text
    assert "Industry: Coffee shop" in prompt_text
    assert "Country / market: EG" in prompt_text
    assert "Do not ask the user to repeat them" in prompt_text
    assert graph_config["configurable"]["business_profile_context"] == saved_profile


def test_combined_campaign_reply_is_markdown_not_raw_json(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = brain.RoutingDecision(
        agent="layla",
        reason="The user requested a campaign plan and an ad draft.",
        workflow="omar_then_layla",
    )
    omar_result = {
        "final_report": {
            "summary": "A draft campaign plan.",
            "marketing_strategy": {
                "summary": "Reach MG owners in Egypt.",
                "channels": [{"channel": "Meta Ads", "budget_share_pct": 60}],
                "evidence": ["Business profile from Settings."],
            },
            "campaign_strategy": {
                "objective": "Increase inquiries.",
                "missing_inputs": ["Monthly booking target."],
            },
        },
    }
    layla_result = {
        "messages": [
            SimpleNamespace(
                content={
                    "headline": "Need MG service?",
                    "primary_text": "Draft copy text.",
                    "call_to_action": "Book now",
                    "aspect_ratio": "1:1",
                },
                tool_calls=[],
            )
        ]
    }
    _install_fakes(
        monkeypatch,
        decision,
        omar_result=omar_result,
        layla_result=layla_result,
    )

    result = asyncio.run(
        brain.orchestrate(
            "اعمللي حملة إعلانية",
            [],
            business_id="biz_test",
        )
    )

    text = result.blocks[0].text
    assert "## خطة الحملة الإعلانية" in text
    assert "### الاستراتيجية التسويقية — عمر" in text
    assert "### مسودة الإعلان — ليلى" in text
    assert "حصة مقترحة: 60%" in text
    assert '"summary"' not in text
    assert "لم يتم إنشاء أو نشر حملة" in text


def test_simple_creative_request_does_not_run_omar(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = brain.RoutingDecision(
        agent="layla",
        reason="The user requested a single ad variation.",
        workflow="single",
    )
    order, omar, layla = _install_fakes(
        monkeypatch,
        decision,
        omar_result={"final_report": {"summary": "Must not run."}},
        layla_result={
            "messages": [SimpleNamespace(content={"summary": "One variation."}, tool_calls=[])]
        },
    )

    asyncio.run(brain.orchestrate("اكتب نسخة إعلان واحدة", [], business_id="biz_test"))

    assert order == ["layla"]
    assert not omar.calls
    assert len(layla.calls) == 1


class _ScriptedChat:
    """Offline chat double that returns real LangChain tool-call messages."""

    def __init__(self, label: str, responses: list[AIMessage], events: list[str]) -> None:
        self.label = label
        self.responses = list(responses)
        self.events = events
        self.inputs: list[list[Any]] = []
        self.input_tails: list[tuple[str | None, str | None]] = []
        self.bound_tools: list[str] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _ScriptedChat:
        self.bound_tools = [tool.name for tool in tools]
        return self

    def invoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> AIMessage:
        self.inputs.append(list(messages))
        last = messages[-1]
        self.input_tails.append((getattr(last, "type", None), getattr(last, "name", None)))
        self.events.append(self.label)
        if not self.responses:
            raise AssertionError(f"{self.label} model was invoked more times than scripted")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


async def test_actual_graphs_run_omar_tools_then_layla_with_handoff(
    monkeypatch: pytest.MonkeyPatch, seeded_db
) -> None:
    """Exercise compiled LangGraphs and ToolNodes without a provider/network call."""
    omar_module = importlib.import_module("app.ai.agents.omar.graph")
    layla_module = importlib.import_module("app.ai.agents.layla.graph")
    brand_context_module = importlib.import_module("app.ai.tools.brand_context")

    decision = brain.RoutingDecision(
        agent="layla",
        reason="The request asks for campaign strategy and a matching ad concept.",
        workflow="omar_then_layla",
    )
    events: list[str] = []
    omar_model = _ScriptedChat(
        "omar",
        [
            _tool_call("get_brand_context_tool", {}, "omar-brand"),
            _tool_call(
                "generate_marketing_strategy",
                {
                    "summary": "Position as reliable office coffee.",
                    "positioning": "Fresh coffee delivered to Cairo offices.",
                    "value_propositions": ["Convenient weekly delivery."],
                    "target_segments": ["Office managers in Cairo."],
                    "missing_inputs": ["Monthly budget."],
                },
                "omar-marketing",
            ),
            _tool_call(
                "generate_campaign_strategy",
                {
                    "summary": "Generate office subscription leads.",
                    "objective": "leads",
                    "audience": "Office managers in Cairo.",
                    "key_messages": ["Fresh coffee, delivered to work."],
                    "creative_direction": "Warm, practical workplace moments.",
                    "missing_inputs": ["Confirmed CTA."],
                },
                "omar-campaign",
            ),
            _tool_call(
                "submit_strategy_report",
                {
                    "summary": "Lead with convenient office subscriptions.",
                    "details": ["Use a warm, practical tone."],
                    "missing_inputs": ["Monthly budget.", "Confirmed CTA."],
                },
                "omar-report",
            ),
        ],
        events,
    )
    layla_model = _ScriptedChat(
        "layla",
        [
            _tool_call(
                "submit_creative_brief",
                {
                    "summary": "A warm workplace scene built around fresh coffee delivery.",
                    "details": ["Headline: Fresh coffee, right at work."],
                },
                "layla-report",
            )
        ],
        events,
    )

    class _EmptyStore:
        def search(self, namespace: tuple[str, ...], **_kwargs: Any) -> list[Any]:
            return []

    fake_profile = BrandProfile(
        name="Cairo Coffee Co",
        industry="Food and beverage",
        market="Cairo, Egypt",
        description="Office coffee subscriptions.",
        audience="Office managers.",
        tone="Warm and practical.",
    )
    brand_ids: list[str] = []

    def _get_fake_profile(store: Any, business_id: str) -> BrandProfile:
        brand_ids.append(business_id)
        return fake_profile

    monkeypatch.setattr(brand_context_module, "current_store", lambda: _EmptyStore())
    monkeypatch.setattr(brand_context_module, "get_brand_profile", _get_fake_profile)

    monkeypatch.setattr(brain, "get_ai_settings", lambda: SimpleNamespace(llm_provider="mock"))
    monkeypatch.setattr(
        brain,
        "get_llm",
        lambda **kwargs: _fake_handle(_FakeChat(decision, events)),
    )
    monkeypatch.setattr(
        omar_module,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="mock"),
    )
    monkeypatch.setattr(
        layla_module,
        "get_ai_settings",
        lambda: SimpleNamespace(llm_provider="mock"),
    )
    monkeypatch.setattr(
        omar_module,
        "get_llm",
        lambda **kwargs: SimpleNamespace(chat=omar_model),
    )
    monkeypatch.setattr(
        layla_module,
        "get_llm",
        lambda **kwargs: SimpleNamespace(chat=layla_model),
    )

    result = await brain.orchestrate(
        "اعمل استراتيجية حملة leads وإعلان مناسب لها",
        [],
        business_id="biz_test",
        workspace_id="ws_test",
        conversation_id="conv_chat_audit",
        request_id="req_chat_audit",
        db=seeded_db,
    )

    # Agent reports are rendered as Markdown by app.ai.presentation; do not make
    # an extra LLM formatting call over facts the agents already supplied.
    assert events == ["router", "omar", "omar", "omar", "omar", "layla"]
    assert result is not None and result.blocks
    assert "خطة" in result.blocks[0].text or "استراتيجية" in result.blocks[0].text
    assert not omar_model.responses
    assert not layla_model.responses
    assert "get_brand_context_tool" in omar_model.bound_tools
    assert "generate_marketing_strategy" in omar_model.bound_tools
    assert "generate_campaign_strategy" in omar_model.bound_tools
    assert "submit_strategy_report" in omar_model.bound_tools
    assert "submit_creative_brief" in layla_model.bound_tools
    assert not any("meta" in name.lower() for name in omar_model.bound_tools + layla_model.bound_tools)
    assert omar_model.input_tails == [
        ("human", None),
        ("tool", "get_brand_context_tool"),
        ("tool", "generate_marketing_strategy"),
        ("tool", "generate_campaign_strategy"),
    ]
    assert layla_model.input_tails == [("human", None)]
    assert brand_ids == ["biz_test"]

    tool_rows = (
        await seeded_db.execute(
            select(AgentToolCall).where(
                AgentToolCall.conversation_id == "conv_chat_audit",
                AgentToolCall.request_id == "req_chat_audit",
            )
        )
    ).scalars().all()
    tool_names = {row.tool for row in tool_rows}
    assert {
        "get_brand_context_tool",
        "generate_marketing_strategy",
        "generate_campaign_strategy",
        "submit_strategy_report",
        "submit_creative_brief",
    } <= tool_names
    assert all(row.business_id == "biz_test" for row in tool_rows)
    assert all(row.status for row in tool_rows)
    assert all(row.output_summary for row in tool_rows)
    assert all(row.persona_id in {"omar", "layla"} for row in tool_rows)

    layla_input = "\n".join(str(message.content) for message in layla_model.inputs[0])
    assert '"objective": "leads"' in layla_input
    assert "Confirmed CTA" in layla_input
    assert "Monthly budget" in layla_input
    assert "content_calendar" not in layla_input
    assert "strategy" in result.blocks[0].text.lower() or "استراتيجية" in result.blocks[0].text
    assert "إبداع" in result.blocks[0].text
    assert "لم يتم إنشاء أو نشر" in result.blocks[0].text


@pytest.mark.parametrize("agent", ["nour", "omar", "layla", "karim", "salma"])
def test_orchestrator_routes_single_agent_requests_without_implicit_handoff(
    agent: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = brain.RoutingDecision(
        agent=agent,
        reason="One specialist owns this request.",
        workflow="single",
    )
    order, _omar, _layla = _install_fakes(
        monkeypatch,
        decision,
        omar_result={"final_report": {"summary": "Omar must not run unless selected."}},
        layla_result={"messages": [SimpleNamespace(content="Layla must not run unless selected.", tool_calls=[])]},
    )

    result = asyncio.run(
        brain.orchestrate("Handle this single-agent marketing request.", [], business_id="biz_test")
    )

    assert order == [agent]
    assert result is not None
    assert result.persona_id == ("researcher" if agent == "nour" else agent)
    assert result.blocks


async def test_tool_context_is_created_only_for_explicit_competitor_ads_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.tools.base import ToolContext

    decision = brain.RoutingDecision(agent="nour", reason="Research request.")
    _, _omar, layla = _install_fakes(
        monkeypatch,
        decision,
        omar_result={},
        layla_result={"messages": [SimpleNamespace(content="Creative brief.", tool_calls=[])]},
    )
    dummy_db = object()
    dummy_core_llm = object()

    await brain.orchestrate(
        "Write one ad creative brief.",
        [],
        business_id="biz_test",
        workspace_id="ws_test",
        db=dummy_db,
        core_llm=dummy_core_llm,
        forced_agent="layla",
    )
    assert layla.calls[0][1]["configurable"].get("tool_context") is None

    _, _omar, _layla = _install_fakes(
        monkeypatch,
        decision,
        omar_result={},
        layla_result={"messages": [SimpleNamespace(content="Research report.", tool_calls=[])]},
    )
    await brain.orchestrate(
        "Find competitor ads in Meta Ad Library for Cairo coffee brands.",
        [],
        business_id="biz_test",
        workspace_id="ws_test",
        conversation_id="conv_competitor",
        request_id="req_competitor",
        db=dummy_db,
        core_llm=dummy_core_llm,
        forced_agent="researcher",
    )
    # Nour's fake graph is installed by _install_fakes; inspect it via the module.
    nour_graph = brain.nour_graph
    tool_context = nour_graph.calls[-1][1]["configurable"]["tool_context"]
    assert isinstance(tool_context, ToolContext)
    assert tool_context.user_supplied["competitor_ads_only"] is True
    assert tool_context.user_supplied["latest_user_message"].startswith("Find competitor ads")


def test_absent_campaign_plan_is_flagged_in_layla_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = brain.RoutingDecision(
        agent="layla",
        reason="The request asks for a new campaign and a creative.",
        workflow="omar_then_layla",
    )
    order, _omar, layla = _install_fakes(
        monkeypatch,
        decision,
        omar_result={
            "marketing_strategy": {"positioning": "Evidence-backed position."},
            "final_report": {"summary": "Marketing strategy only."},
        },
        layla_result={
            "messages": [SimpleNamespace(content={"summary": "Draft concept."}, tool_calls=[])]
        },
    )

    asyncio.run(brain.orchestrate("Build a campaign and its ad", [], business_id="biz_test"))

    assert order == ["omar", "layla"]
    handoff_text = layla.calls[0][0]["messages"][-1].content
    assert "Omar did not return a structured campaign strategy" in handoff_text
    assert "do not silently guess" in handoff_text


def test_groq_default_model_uses_supported_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from app.ai.settings import AISettings

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert AISettings(_env_file=None).groq_model == "openai/gpt-oss-20b"
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    assert "GROQ_MODEL=openai/gpt-oss-20b" in env_example.read_text(encoding="utf-8")
