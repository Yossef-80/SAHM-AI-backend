"""Required tests 2, 3 and 4: runtime allowlist, task validation, output schema
validation with exactly one repair retry, and autonomy policy.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.core.agents.persona import Persona
from app.core.agents.personas import KARIM, LAYLA, OMAR, PERSONAS, SALMA
from app.core.agents.runtime import (
    RunOptions,
    _apply_autonomy_policy,
    describe_autonomy,
    run_agent,
    run_agent_stream,
)
from app.core.agents.tasks import TASK_SPECS, TASK_OUTPUT_MODELS, get_task, validate_task_catalog
from app.core.errors import TaskNotAllowedError
from app.core.llm.base import ToolCall
from app.core.llm.mock_client import MockClient
from app.core.tools.base import ToolContext
from app.schemas.agent import AgentEvent, ToolCallRecord
from app.schemas.common import Autonomy, ToolSideEffect

ALL_TASK_NAMES = [
    "competitor_research", "market_analysis", "market_gaps", "marketing_strategy",
    "campaign_strategy", "content_calendar", "creative_brief", "commercial_script",
    "image_prompt", "video_prompt", "ad_copy", "ad_variations", "performance_analysis",
    "optimization_proposal",
]


# ============================================================ test 2: allowlist


async def test_persona_cannot_run_a_task_outside_its_allowlist(tool_ctx: ToolContext) -> None:
    """Karim only runs performance_analysis; ad_copy must be rejected."""
    llm = MockClient()
    with pytest.raises(TaskNotAllowedError) as excinfo:
        await run_agent(KARIM, "ad_copy", {}, tool_ctx, llm=llm)
    assert "may not run task" in str(excinfo.value)
    assert excinfo.value.detail["allowed_tasks"] == ["performance_analysis"]


async def test_persona_cannot_call_a_tool_outside_its_allowlist(tool_ctx: ToolContext) -> None:
    """Script the model into calling a tool the persona does not own.

    Layla may not call web_search. The runtime must refuse to run it and report
    it back to the model as a blocked call, not execute it.
    """
    scripted = MockClient(
        model="mock-strong-1",
        script=[
            # Turn 1: Layla illegally asks for web_search.
            {
                "text": "Let me search the web.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "web_search",
                        "arguments": {"query": "coffee subscriptions"},
                    }
                ],
            },
            # Turn 2: also tries a protected tool. Also must be blocked.
            {
                "text": "Let me apply a change.",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "apply_change",
                        "arguments": {"recommendation_id": "rec_x", "confirmed_by": "attacker"},
                    }
                ],
            },
            # Turn 3: gives up on tools and produces the final output.
            {"text": json.dumps({"primary_texts": ["Beans, delivered."], "headlines": ["Fresh weekly"],
                                 "descriptions": ["Subscription coffee"], "ctas": ["Subscribe"],
                                 "summary": "Ad copy produced.", "evidence": [],
                                 "unavailable_sources": [], "notes": ""})},
        ],
    )
    result = await run_agent(LAYLA, "ad_copy", {}, tool_ctx, llm=scripted)

    blocked = [c for c in result.tool_calls_made if c.status == "blocked"]
    assert len(blocked) == 2, [c.tool for c in result.tool_calls_made]
    assert {c.tool for c in blocked} == {"web_search", "apply_change"}
    for call in blocked:
        assert "allowlist" in call.error or "not registered" in call.error
    # Neither tool actually ran: no research rows, no Meta write.
    assert result.output.get("primary_texts")


async def test_allowlist_block_is_reported_to_the_model(tool_ctx: ToolContext) -> None:
    """The blocked call is appended as a tool result so the model can adapt."""
    # Three replies: the blocked tool call, a final-text turn, and the
    # structured-output turn.
    scripted = MockClient(
        model="mock-strong-1",
        script=[
            {"text": "x", "tool_calls": [{"id": "c1", "name": "get_campaign_insights",
                                          "arguments": {"campaign_id": "cmp_1"}}]},
            {"text": "I cannot read campaign performance, so I will write the script anyway."},
            {"text": json.dumps({"hook": "Fresh beans.", "scenes": [], "voiceover": "v",
                                 "cta": "Subscribe", "duration_seconds": 30.0,
                                 "summary": "Script.", "evidence": [],
                                 "unavailable_sources": [], "notes": ""})},
        ],
    )
    # Layla cannot read campaign insights (that's karim's tool).
    result = await run_agent(LAYLA, "commercial_script", {}, tool_ctx, llm=scripted)
    assert any(c.tool == "get_campaign_insights" and c.status == "blocked"
               for c in result.tool_calls_made)
    assert result.output.get("hook") == "Fresh beans."


# ================================================== test 3: output validation


@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
async def test_every_task_output_validates_with_mock_output(
    task_name: str, tool_ctx: ToolContext
) -> None:
    """Each TaskSpec's output_model accepts the mock LLM's output."""
    validate_task_catalog()
    persona_id = TASK_SPECS[task_name].owner_persona
    persona = PERSONAS[persona_id]
    result = await run_agent(persona, task_name, {}, tool_ctx, llm=MockClient())
    assert result.error is None, f"{task_name} errored: {result.error}"
    assert result.incomplete is False, f"{task_name} incomplete: {result.incomplete_reason}"
    # The output must re-validate against the declared model.
    model = TASK_OUTPUT_MODELS[task_name]
    validated = model.model_validate(result.output)
    assert validated is not None


async def test_invalid_output_triggers_exactly_one_repair_retry(tool_ctx: ToolContext) -> None:
    """Bad output first, good output second -> exactly two provider calls."""
    llm = MockClient(
        model="mock-strong-1",
        script=[
            # Missing required fields -> schema violation.
            {"text": json.dumps({"positioning": "x"})},
            # Repair: a complete, valid MarketingStrategyOutput.
            {"text": json.dumps({
                "positioning": "The subscription for people who care about coffee.",
                "value_propositions": ["Fresh", "Convenient"],
                "messaging_pillars": ["Freshness", "Convenience"],
                "channels": [{"channel": "meta", "role": "acquisition", "budget_share": 0.6,
                              "kpis": ["cpa"]}],
                "primary_kpis": ["subscriptions"],
                "ninety_day_phases": ["launch", "iterate", "scale"],
                "summary": "Strategy.", "evidence": [], "unavailable_sources": [], "notes": "",
            })},
        ],
    )
    result = await run_agent(OMAR, "marketing_strategy", {}, tool_ctx, llm=llm)
    assert result.error is None
    assert result.output["positioning"].startswith("The subscription")
    # Two calls: the original plus exactly one repair.
    assert len(llm.calls) == 2


async def test_unrepairable_output_sets_incomplete(tool_ctx: ToolContext) -> None:
    # Three scripted replies: the tools turn, the first (bad) structured
    # attempt, and the (also bad) repair attempt. After that the mock would
    # fall back to generating a valid sample, so the script must cover all
    # three calls to prove the run really gives up.
    llm = MockClient(
        model="mock-strong-1",
        script=[
            {"text": "not json"},
            {"text": "still not json"},
            {"text": "and still not json"},
        ],
    )
    result = await run_agent(OMAR, "marketing_strategy", {}, tool_ctx, llm=llm)
    assert result.incomplete is True
    assert result.error is not None
    assert "repair" in result.error.lower()
    assert len(llm.calls) == 3  # tools turn + original + one repair, then stop


# ======================================================= test 4: autonomy policy


def test_suggest_persists_nothing() -> None:
    persona = OMAR
    assert persona.autonomy == Autonomy.SUGGEST
    task = get_task("marketing_strategy")
    persisted, requires_approval = _apply_autonomy_policy(
        persona, task, tool_calls=[], output={"summary": "a plan"}
    )
    assert persisted == []
    assert requires_approval is False
    assert "nothing" in describe_autonomy(Autonomy.SUGGEST)


def test_auto_draft_persists_drafts_but_never_external_writes() -> None:
    persona = LAYLA
    assert persona.autonomy == Autonomy.AUTO_DRAFT
    task = get_task("ad_copy")
    calls = [ToolCallRecord(tool="generate_image", input={}, output_summary="", status="ok")]
    persisted, requires_approval = _apply_autonomy_policy(
        persona, task, tool_calls=calls, output={"summary": "copy"}
    )
    assert "task_output:ad_copy" in persisted
    assert "tool:generate_image" in persisted
    assert requires_approval is False
    # And no protected tool is reachable from this persona at all.
    from app.core.tools.registry import get_tools_for

    assert all(t.side_effect != ToolSideEffect.EXTERNAL_WRITE for t in get_tools_for(persona))


def test_ask_first_requires_approval() -> None:
    persona = SALMA
    assert persona.autonomy == Autonomy.ASK_FIRST
    task = get_task("optimization_proposal")
    calls = [ToolCallRecord(tool="create_recommendation", input={}, output_summary="",
                            status="ok")]
    persisted, requires_approval = _apply_autonomy_policy(
        persona, task, tool_calls=calls, output={"action_type": "pause_ad"}
    )
    assert requires_approval is True
    assert persisted == ["pending:create_recommendation"]


async def test_ask_first_leaves_the_recommendation_pending(tool_ctx: ToolContext) -> None:
    """Salma's create_recommendation writes status='pending', never executed."""
    from sqlalchemy import select

    from app.db.models import Recommendation

    result = await run_agent(SALMA, "optimization_proposal", {}, tool_ctx, llm=MockClient())
    assert result.error is None
    rows = (await tool_ctx.db.execute(select(Recommendation))).scalars().all()
    assert rows, "salma should have created a recommendation"
    for row in rows:
        assert row.status == "pending"
        assert row.requires_approval is True
        assert row.executed_at is None


async def test_suggest_persona_creates_no_durable_records(tool_ctx: ToolContext) -> None:
    """Omar (suggest) must not persist a plan artifact."""
    from sqlalchemy import select

    from app.db.models import Creative, Recommendation

    await run_agent(OMAR, "marketing_strategy", {}, tool_ctx, llm=MockClient())
    creatives = (await tool_ctx.db.execute(select(Creative))).scalars().all()
    recommendations = (await tool_ctx.db.execute(select(Recommendation))).scalars().all()
    assert creatives == []
    assert recommendations == []


async def test_auto_draft_persona_persists_creative_drafts(tool_ctx: ToolContext) -> None:
    """Layla (auto_draft) may persist generated assets."""
    from sqlalchemy import select

    from app.db.models import Creative

    result = await run_agent(
        LAYLA, "image_prompt", {"prompt": "a cup of coffee"}, tool_ctx, llm=MockClient()
    )
    assert result.error is None
    assert "task_output:image_prompt" in result.persisted


# ============================================================ hard caps


async def test_iteration_cap_returns_partial_result(tool_ctx: ToolContext) -> None:
    """A model that always asks for tools must hit the cap, not hang."""
    llm = MockClient(model="mock-strong-1")
    # Every turn asks for a tool that never gets marked as called.
    original = llm._complete_once

    async def always_call_tools(**kwargs):  # type: ignore[no-untyped-def]
        from app.core.llm.base import LLMResult
        from app.schemas.common import Usage

        return LLMResult(
            text="calling",
            tool_calls=[ToolCall(id="x", name="get_brand_context", arguments={})],
            usage=Usage(tokens_in=1, tokens_out=1, model="mock-strong-1"),
            model="mock-strong-1",
        )

    llm._complete_once = always_call_tools  # type: ignore[method-assign]
    options = RunOptions(max_iterations=3, timeout_seconds=30)
    result = await run_agent(LAYLA, "ad_copy", {}, tool_ctx, llm=llm, options=options)
    assert result.incomplete is True
    assert "iteration cap" in (result.incomplete_reason or "")
    assert result.output == {}


async def test_run_timeout_cap_returns_partial_result(tool_ctx: ToolContext) -> None:
    llm = MockClient(model="mock-strong-1")
    options = RunOptions(max_iterations=10, timeout_seconds=0.0001)
    result = await run_agent(OMAR, "marketing_strategy", {}, tool_ctx, llm=llm, options=options)
    assert result.incomplete is True


# ============================================================ streaming


async def test_streaming_yields_the_documented_events(tool_ctx: ToolContext) -> None:
    events: list[AgentEvent] = []
    async for event in run_agent_stream(LAYLA, "ad_copy", {}, tool_ctx, llm=MockClient()):
        events.append(event)
    kinds = [e.type for e in events]
    assert "step_started" in kinds
    assert "tool_called" in kinds
    assert "tool_result" in kinds
    assert "completed" in kinds
    terminal = [e for e in events if e.type == "completed"]
    assert terminal and terminal[-1].data and "result" in terminal[-1].data
    # The streamed result matches the non-streaming path.
    assert terminal[-1].data["result"]["task"] == "ad_copy"


# ============================================================ persona data


def test_persona_catalog_matches_the_brief() -> None:
    assert set(PERSONAS) == {"researcher", "omar", "layla", "karim", "salma"}
    assert PERSONAS["researcher"].name == "Nour"
    assert PERSONAS["researcher"].autonomy == Autonomy.AUTO_DRAFT
    assert PERSONAS["omar"].tone.value == "bold"
    assert PERSONAS["omar"].autonomy == Autonomy.SUGGEST
    assert PERSONAS["layla"].tone.value == "playful"
    assert PERSONAS["layla"].autonomy == Autonomy.AUTO_DRAFT
    assert PERSONAS["karim"].tone.value == "minimal"
    assert PERSONAS["karim"].autonomy == Autonomy.AUTO_DRAFT
    assert PERSONAS["salma"].tone.value == "professional"
    assert PERSONAS["salma"].autonomy == Autonomy.ASK_FIRST


def test_persona_editable_fields_are_closed() -> None:
    for persona in PERSONAS.values():
        assert persona.editable_fields == ["name", "tone", "autonomy", "language"]
        assert "allowed_tools" not in persona.editable_fields
        assert "allowed_tasks" not in persona.editable_fields


def test_persona_overrides_cannot_widen_permissions() -> None:
    persona = PERSONAS["omar"]
    merged = persona.with_overrides({"name": "O", "allowed_tools": ["apply_change"]})
    assert merged.name == "O"
    # allowed_tools is not editable, so it is untouched.
    assert merged.allowed_tools == persona.allowed_tools


def test_every_persona_task_is_a_real_task() -> None:
    for persona in PERSONAS.values():
        for task in persona.allowed_tasks:
            assert task in TASK_SPECS, f"{persona.id} references unknown task {task}"


def test_salma_cannot_generate_creatives() -> None:
    """Structural: salma has no creative tools and no creative tasks."""
    assert "generate_image" not in SALMA.allowed_tools
    assert "analyze_reference_creative" not in SALMA.allowed_tools
    assert not set(SALMA.allowed_tasks) & {
        "creative_brief", "ad_copy", "ad_variations", "image_prompt", "commercial_script",
        "video_prompt",
    }
