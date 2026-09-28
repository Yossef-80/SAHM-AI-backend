"""Required test 6: intent routing for at least eight sample messages, including
a missing-info case that yields a question_with_options block.
"""

from __future__ import annotations

import pytest

from app.core.agents.personas import PERSONAS
from app.core.agents.runtime import RunOptions
from app.core.llm.mock_client import MockClient
from app.core.orchestrator.blocks import safe_block, validate_blocks
from app.core.orchestrator.graph import OrchestratorDeps, build_graph
from app.core.orchestrator.router import (
    INTENT_ROUTES,
    classify_with_rules,
    clarification_question,
)
from app.core.orchestrator.state import initial_state
from app.schemas.common import BlockKind, IntentKind


# ------------------------------------------------------------- routing table --

#: (message, expected intent). At least eight, per the brief.
ROUTING_CASES: list[tuple[str, IntentKind]] = [
    ("Who are my main competitors and what do they offer?", IntentKind.RESEARCH),
    ("Research the market for coffee subscriptions in Cairo", IntentKind.RESEARCH),
    ("What keywords should I target for coffee delivery?", IntentKind.RESEARCH),
    ("Build me a marketing strategy for the next quarter", IntentKind.STRATEGY),
    ("Write a content calendar for the next month", IntentKind.STRATEGY),
    ("Write 5 ad copy variations for our subscription offer", IntentKind.CREATIVE),
    ("Give me an image prompt for a Ramadan campaign", IntentKind.CREATIVE),
    ("How did my last campaign perform?", IntentKind.ANALYZE),
    ("What's my CTR and CPC breakdown by placement?", IntentKind.ANALYZE),
    ("What should I change to improve performance?", IntentKind.OPTIMIZE),
    ("Should I pause the underperforming ad set?", IntentKind.OPTIMIZE),
]


@pytest.mark.parametrize("message,expected", ROUTING_CASES)
def test_rule_based_routing(message: str, expected: IntentKind) -> None:
    guess = classify_with_rules(message, {})
    assert guess.intent == expected, (
        f"{message!r} routed to {guess.intent.value}, expected {expected.value}"
    )
    persona, _task = INTENT_ROUTES[expected]
    assert guess.task_name is not None
    # The picked task must belong to the routed persona.
    assert guess.task_name in PERSONAS[persona].allowed_tasks


def test_missing_information_routes_to_clarification() -> None:
    """No campaign in the message and none in context -> ask, do not guess."""
    guess = classify_with_rules("How is my campaign performing?", {})
    assert guess.intent == IntentKind.ANALYZE
    assert guess.missing_info == ["campaign_id"]
    assert guess.task_name == "performance_analysis"


def test_context_campaign_id_clears_the_missing_info_flag() -> None:
    guess = classify_with_rules("How is my campaign performing?", {"campaign_id": "cmp_123"})
    assert guess.missing_info == []


def test_missing_budget_routes_to_clarification_for_campaign_strategy() -> None:
    guess = classify_with_rules("Create a campaign strategy for Ramadan", {})
    assert guess.intent == IntentKind.STRATEGY
    assert guess.task_name == "campaign_strategy"
    assert "budget" in guess.missing_info


def test_budget_in_the_message_clears_the_missing_flag() -> None:
    guess = classify_with_rules("Create a campaign strategy for Ramadan with 50000 EGP", {})
    assert "budget" not in guess.missing_info


def test_clarification_question_shape() -> None:
    payload = clarification_question(["campaign_id", "budget"])
    assert payload["question"]
    assert [o["id"] for o in payload["options"]] == ["pick_campaign", "give_budget"]
    assert payload["missing_info"] == ["campaign_id", "budget"]


# --------------------------------------------------------------- the graph ----


async def test_graph_routes_a_research_message_to_the_researcher(seeded_db) -> None:
    deps = OrchestratorDeps(
        db=seeded_db,
        llm=MockClient(model="mock-strong-1"),
        business_id="biz_test",
        workspace_id="ws_test",
        request_id="req-orch-1",
        options=RunOptions(max_iterations=6, timeout_seconds=60),
    )
    graph = build_graph(deps)
    state = initial_state(
        conversation_id="conv_1",
        business_id="biz_test",
        workspace_id="ws_test",
        user_message="Who are my main competitors and what do they offer?",
        request_id="req-orch-1",
        context={},
    )
    final = await graph.ainvoke(state)
    assert final["intent"] == IntentKind.RESEARCH.value
    assert final["persona_id"] == "researcher"
    assert final["task_name"] == "competitor_research"
    assert final["agent_result"]["persona_id"] == "researcher"
    # Blocks were produced and every one validates unchanged.
    assert final["blocks"]
    assert validate_blocks(final["blocks"]) == final["blocks"]


async def test_graph_missing_info_produces_question_with_options(seeded_db) -> None:
    deps = OrchestratorDeps(
        db=seeded_db,
        llm=MockClient(model="mock-strong-1"),
        business_id="biz_test",
        workspace_id="ws_test",
        request_id="req-orch-2",
        options=RunOptions(),
    )
    graph = build_graph(deps)
    state = initial_state(
        conversation_id="conv_2",
        business_id="biz_test",
        workspace_id="ws_test",
        user_message="How is my campaign performing?",
        request_id="req-orch-2",
        context={},
    )
    final = await graph.ainvoke(state)
    # Routed to ask_clarification, never to karim with no data.
    assert final["intent"] == IntentKind.ANALYZE.value
    assert final["missing_info"] == ["campaign_id"]
    assert final["blocks"], "a question block must be produced"
    question = final["blocks"][0]
    assert question["type"] == BlockKind.QUESTION_WITH_OPTIONS.value
    assert question["options"]
    assert "campaign" in question["question"].lower()
    # No agent ran, so there is no fabricated analysis.
    assert final.get("agent_result") is None


async def test_graph_forced_persona_bypasses_classification(seeded_db) -> None:
    """An action card arrives as a forced persona/task."""
    deps = OrchestratorDeps(
        db=seeded_db,
        llm=MockClient(model="mock-strong-1"),
        business_id="biz_test",
        workspace_id="ws_test",
        request_id="req-orch-3",
        forced_persona_id="layla",
        forced_task="ad_variations",
        options=RunOptions(),
    )
    graph = build_graph(deps)
    state = initial_state(
        conversation_id="conv_3",
        business_id="biz_test",
        workspace_id="ws_test",
        user_message="Generate new creative variations",
        request_id="req-orch-3",
        context={},
    )
    final = await graph.ainvoke(state)
    assert final["persona_id"] == "layla"
    assert final["task_name"] == "ad_variations"
    assert final["agent_result"]["persona_id"] == "layla"


async def test_graph_degrades_on_an_unroutable_message(seeded_db) -> None:
    """A message nothing matches must not crash the graph."""
    deps = OrchestratorDeps(
        db=seeded_db,
        llm=MockClient(model="mock-strong-1"),
        business_id="biz_test",
        workspace_id="ws_test",
        request_id="req-orch-4",
        options=RunOptions(),
    )
    graph = build_graph(deps)
    state = initial_state(
        conversation_id="conv_4",
        business_id="biz_test",
        workspace_id="ws_test",
        user_message="hello",
        request_id="req-orch-4",
        context={},
    )
    final = await graph.ainvoke(state)
    assert "blocks" in final
    assert isinstance(final["blocks"], list)


def test_safe_block_degrades_instead_of_raising() -> None:
    block = safe_block({"type": "not_a_real_block", "junk": 1}, fallback_text="oops")
    assert block.type == BlockKind.TEXT
    assert block.text == "oops"


def test_validate_blocks_replaces_a_bad_block() -> None:
    blocks = validate_blocks([{"type": "text", "text": "fine"}, {"type": "bogus", "x": 1}])
    assert blocks[0]["type"] == BlockKind.TEXT.value
    assert blocks[1]["type"] == BlockKind.TEXT.value
