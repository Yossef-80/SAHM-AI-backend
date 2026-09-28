"""Required test 8: every block validates against the frontend's schema.

Also: the full ``Message.blocks`` discriminated union is exercised, every block
kind round-trips through JSON, and the API's assistant response only ever emits
blocks that validate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.agents.personas import PERSONAS
from app.core.agents.runtime import run_agent
from app.core.agents.tasks import TASK_SPECS
from app.core.llm.mock_client import MockClient
from app.core.orchestrator.blocks import (
    action_card,
    compose_blocks,
    context_used_block,
    error_block,
    findings_block,
    metric_block,
    question_block,
    recommendation_block,
    safe_block,
    task_progress_block,
    text_block,
    validate_block,
    validate_blocks,
)
from app.schemas.blocks import (
    BLOCK_TYPES,
    ActionCard,
    ContextUsedBlock,
    Finding,
    FindingsBlock,
    Message,
    ErrorBlock,
    MessageWithBlocks,
    MetricBlock,
    QuestionWithOptionsBlock,
    RecommendationBlock,
    TaskProgressBlock,
    TextBlock,
)
from app.schemas.common import BlockKind
from app.core.tools.base import ToolContext


# ------------------------------------------------------- every block kind ----


def test_every_declared_block_type_has_a_model() -> None:
    models = {
        TextBlock, FindingsBlock, RecommendationBlock, MetricBlock, ActionCard,
        TaskProgressBlock, ContextUsedBlock, QuestionWithOptionsBlock, ErrorBlock,
    }
    assert len(models) == len(BLOCK_TYPES)
    for model in models:
        assert model.model_fields["type"].default is not None


@pytest.mark.parametrize(
    "builder",
    [
        lambda: text_block("hello"),
        lambda: findings_block("Findings", [Finding(title="t", detail="d", source="s")]),
        lambda: metric_block("CTR", 0.42, unit="%"),
        lambda: action_card(action_type="run_task", label="Go", target_persona="layla"),
        lambda: task_progress_block(
            task_id="t1", task_name="ad_copy", persona_id="layla", status="running"
        ),
        lambda: context_used_block([]),
        lambda: error_block("boom"),
        lambda: recommendation_block(
            recommendation_id="r1", action_type="pause_ad", title="Pause it"
        ),
        lambda: question_block("Which campaign?", [{"id": "a", "label": "A"}], ["campaign_id"]),
    ],
)
def test_every_block_kind_validates(builder) -> None:
    block = builder()
    validated = validate_block(block)
    assert validated is not None
    # And it survives a JSON round-trip, which is what crosses the wire.
    raw = json.loads(json.dumps(block))
    assert validate_block(raw) is not None


def test_blocks_discriminate_on_type() -> None:
    """A wrong payload for a type must fail validation, not silently coerce."""
    with pytest.raises(ValidationError):
        MessageWithBlocks.model_validate([{"type": "text", "value": 1}])
    with pytest.raises(ValidationError):
        MessageWithBlocks.model_validate([{"type": "hologram", "text": "hi"}])


def test_message_blocks_round_trip_with_the_discriminator() -> None:
    blocks = [
        text_block("hi"),
        findings_block("F", [Finding(title="t", detail="d", source="s")]),
        metric_block("CTR", 0.42),
    ]
    message = Message(
        id="m1", role="assistant", content="hi", blocks=blocks,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    raw = message.model_dump(mode="json", by_alias=True)
    assert [b["type"] for b in raw["blocks"]] == ["text", "findings", "metric"]
    restored = Message.model_validate(raw)
    assert [b.type for b in restored.blocks] == [
        BlockKind.TEXT,
        BlockKind.FINDINGS,
        BlockKind.METRIC,
    ]


# ------------------------------------------------- composed blocks validate --


@pytest.mark.parametrize(
    "task_name",
    [
        "competitor_research", "market_analysis", "market_gaps", "marketing_strategy",
        "campaign_strategy", "content_calendar", "creative_brief", "commercial_script",
        "image_prompt", "video_prompt", "ad_copy", "ad_variations", "performance_analysis",
        "optimization_proposal",
    ],
)
async def test_composed_blocks_validate_for_every_task(
    task_name: str, tool_ctx: ToolContext
) -> None:
    """The composer must only ever emit blocks the frontend can render."""
    persona_id = TASK_SPECS[task_name].owner_persona
    persona = PERSONAS[persona_id]
    result = await run_agent(persona, task_name, {}, tool_ctx, llm=MockClient())

    blocks = compose_blocks(
        result.model_dump(mode="json"),
        persona_id=persona.id,
        task_name=task_name,
    )
    assert blocks, f"{task_name} produced no blocks"
    # Every block validates unchanged -- this is the contract gate.
    assert validate_blocks(blocks) == blocks


async def test_action_card_from_salma_routes_to_layla(tool_ctx: ToolContext) -> None:
    """Salma hands creative work to layla through a clickable action card."""
    import json as _json

    scripted = MockClient(
        model="mock-strong-1",
        script=[
            {"text": "checking performance",
             "tool_calls": [{"id": "c1", "name": "get_campaign_insights",
                             "arguments": {"campaign_id": "cmp_1"}}]},
            {"text": "the creative is fatiguing, so new variations are needed"},
            {"text": _json.dumps({
                "action_type": "refresh_creative",
                "title": "Refresh the Ramadan reel",
                "rationale": "Frequency is 6.1 and CTR fell to 0.4%.",
                "payload": {"creative_ids": ["cr_1"]},
                "estimated_impact": "not available",
                "needs_creative_variations": True,
                "creative_brief_for_layla": (
                    "Three angles: price, convenience, and office gifting. "
                    "Keep the earthy palette and natural light."
                ),
                "summary": "The Ramadan reel is fatiguing and needs refreshing.",
                "evidence": [], "unavailable_sources": ["get_campaign_insights"],
                "notes": "Meta is not connected.",
            })},
        ],
    )
    salma = PERSONAS["salma"]
    result = await run_agent(
        salma, "optimization_proposal", {}, tool_ctx, llm=scripted
    )
    assert result.output.get("needs_creative_variations") is True
    blocks = compose_blocks(
        result.model_dump(mode="json"), persona_id="salma", task_name="optimization_proposal"
    )
    cards = [b for b in blocks if b.get("type") == BlockKind.ACTION_CARD.value]
    assert cards, [b.get("type") for b in blocks]
    card = ActionCard.model_validate(cards[0])
    assert card.target_persona == "layla"
    assert card.task == "ad_variations"
    # Salma must never be the target of her own creative action.
    assert card.target_persona != "salma"


async def test_blocks_from_a_failed_run_still_validate(tool_ctx: ToolContext) -> None:
    """An incomplete or errored result must degrade, never 500."""
    broken = {
        "task": "performance_analysis",
        "persona_id": "karim",
        "output": {},
        "tool_calls_made": [],
        "unavailable_sources": ["get_campaign_insights: no token"],
        "usage": {"tokens_in": 0, "tokens_out": 0, "model": "m", "latency_ms": 0},
        "request_id": "r",
        "incomplete": True,
        "incomplete_reason": "hit the iteration cap",
        "error": "LLM call failed: timeout",
        "persisted": [],
    }
    blocks = compose_blocks(broken, persona_id="karim", task_name="performance_analysis")
    assert blocks
    assert validate_blocks(blocks) == blocks
    assert any(b["type"] == BlockKind.ERROR.value for b in blocks)


async def test_unknown_task_degrades_to_text(tool_ctx: ToolContext) -> None:
    """A task the composer does not know must not break the response."""
    result = {
        "task": "some_future_task",
        "persona_id": "omar",
        "output": {"summary": "Did the thing."},
        "tool_calls_made": [],
        "unavailable_sources": [],
        "incomplete": False,
        "error": None,
    }
    blocks = compose_blocks(result, persona_id="omar", task_name="some_future_task")
    assert blocks
    assert blocks[0]["type"] == BlockKind.TEXT.value
    assert validate_blocks(blocks) == blocks


def test_safe_block_never_raises_on_garbage() -> None:
    for garbage in (
        {"type": "nope"},
        {"type": "text", "text": {"not": "a string"}},
        {},
        {"type": "findings", "findings": "not a list"},
    ):
        block = safe_block(garbage, fallback_text="degraded")
        assert block.type == BlockKind.TEXT


# --------------------------------------------- the API only emits valid blocks --


async def test_assistant_response_blocks_are_all_valid(api_client) -> None:
    """The wire response is checked against the frontend union."""
    client = api_client
    resp = await client.post(
        "/api/v1/assistant/message",
        json={"message": "Who are my main competitors and what do they offer?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    blocks = body["message"]["blocks"]
    assert blocks
    for block in blocks:
        assert "type" in block
        assert validate_block(block) is not None
    # The whole message validates as a Message.
    Message.model_validate(body["message"])


async def test_api_error_shape_matches_the_contract(api_client) -> None:
    client = api_client
    resp = await client.get("/api/v1/campaigns/does_not_exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["message"]
    assert body["error"]["requestId"]
    assert "requestId" in body["error"]


async def test_personas_endpoint_returns_camel_case(api_client) -> None:
    client = api_client
    resp = await client.get("/api/v1/personas")
    assert resp.status_code == 200
    personas = resp.json()
    assert len(personas) == 5
    for persona in personas:
        # camelCase aliases, exactly what the frontend's Zod types use.
        assert "allowedTools" in persona
        assert "allowedTasks" in persona
        assert "editableFields" in persona
        assert "isOverridden" in persona
        assert "allowed_tools" not in persona
