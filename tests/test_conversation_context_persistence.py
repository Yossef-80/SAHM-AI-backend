"""The conversation's sticky context must actually reach the database.

Regression for the Layla prompt interview looping forever: `handle_message`
assigned the *working* dict to `conversation.context` early in the turn, and a
plain (non-Mutable) JSON column keeps that same object as the ORM's
committed-state baseline. Every key added after `orchestrate()` therefore mutated
the baseline as well, so the later re-assignment compared equal, emitted no
UPDATE, and was silently dropped.

The visible symptom: `creative_prompt_stage` never persisted, so each turn
restarted at `collecting_details` -- Layla re-asked questions the user had already
answered and re-presented a plan they had already approved.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from app.schemas.blocks import TextBlock


def _layla_result(stage: str):
    from app.ai.agents.orchestrator import brain

    return brain.OrchestratorResult(
        blocks=[TextBlock(text="plan")],
        persona_id="layla",
        creative_prompt_stage=stage,
        creative_prompt_workflow=True,
        creative_prompt_type="image",
    )


async def _stored_context(db_session, conversation_id: str) -> dict:
    """Read the row straight from SQL so the ORM identity map cannot hide a miss."""
    row = (
        await db_session.execute(
            text("SELECT context FROM conversations WHERE id = :i"),
            {"i": conversation_id},
        )
    ).scalar_one()
    return json.loads(row) if isinstance(row, str) else row


@pytest.mark.asyncio
async def test_keys_added_after_orchestrate_are_persisted(api_client, monkeypatch, db_session) -> None:
    from app.ai.agents.orchestrator import brain

    stages_seen: list[object] = []

    async def fake_orchestrate(*_args, **kwargs):
        stages_seen.append((kwargs.get("context") or {}).get("creative_prompt_stage"))
        return _layla_result("awaiting_approval")

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)

    first = await api_client.post(
        "/api/v1/assistant/message",
        json={
            "message": "ليلى عايز برومبت صورة",
            "context": {"workflow": "creative_prompt", "promptType": "image"},
        },
    )
    assert first.status_code == 200, first.text
    conversation_id = first.json()["conversationId"]

    stored = await _stored_context(db_session, conversation_id)
    assert stored.get("creative_prompt_stage") == "awaiting_approval", stored
    assert stored.get("persona_id") == "layla", stored

    second = await api_client.post(
        "/api/v1/assistant/message",
        json={"conversationId": conversation_id, "message": "موافق، اكتب البرومبت"},
    )
    assert second.status_code == 200, second.text

    # Turn 1 has no prior stage; turn 2 must see the one turn 1 persisted.
    assert stages_seen == [None, "awaiting_approval"], stages_seen


@pytest.mark.asyncio
async def test_an_approved_plan_is_not_reopened_on_the_next_turn(
    api_client, monkeypatch, db_session
) -> None:
    """The user-visible contract: approving ends the interview, it does not restart it."""
    from app.ai.agents.orchestrator import brain

    plan = {
        "summary": "خطة إعلان فيسبوك لخدمة عمرة موتور لسيارات إم جي.",
        "details": ["الهدف: إعلان فيسبوك", "المنصة: فيسبوك 1:1"],
        "preserved_elements": ["الثيم اللوني الأحمر والأبيض"],
        "proposed_changes": ["استبدال الفتيس بمحرك سيارة إم جي"],
        "ready_for_approval": True,
        "image_prompt": "A cinematic 1:1 studio shot of an MG engine overhaul.",
    }

    async def fake_orchestrate(*_args, **kwargs):
        return _layla_result("awaiting_approval")

    # The stage machine lives in brain.py; drive it directly for the second turn so
    # the assertion is about persistence, not about the model.
    from app.ai.agents.orchestrator.brain import (
        _activate_creative_prompt_workflow,
        _creative_prompt_blocks,
        _prepare_creative_prompt_turn,
    )

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)

    first = await api_client.post(
        "/api/v1/assistant/message",
        json={
            "message": "ليلى عايز برومبت صورة",
            "context": {"workflow": "creative_prompt", "promptType": "image"},
        },
    )
    conversation_id = first.json()["conversationId"]
    stored = await _stored_context(db_session, conversation_id)

    # Replay the next turn through the real state machine with the persisted context.
    context = _activate_creative_prompt_workflow(dict(stored), "موافق، اكتب البرومبت", [])
    context = _prepare_creative_prompt_turn(context, "موافق، اكتب البرومبت")
    assert context["creative_prompt_stage"] == "approved", context

    blocks, next_stage = _creative_prompt_blocks(
        plan,
        context=context,
        prompt="موافق، اكتب البرومبت",
        stage=context["creative_prompt_stage"],
        history=[],
    )
    assert next_stage == "prompt_ready", next_stage
    assert "البرومبت النهائي" in blocks[0].text


@pytest.mark.asyncio
async def test_delete_conversation_endpoint(api_client, monkeypatch) -> None:
    from app.ai.agents.orchestrator import brain

    async def fake_orchestrate(*_args, **kwargs):
        return _layla_result("awaiting_approval")

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)

    res = await api_client.post(
        "/api/v1/assistant/message",
        json={"message": "محادثة للتجربة"},
    )
    assert res.status_code == 200
    conv_id = res.json()["conversationId"]

    # Read conversation detail -> 200
    get_res = await api_client.get(f"/api/v1/assistant/conversations/{conv_id}")
    assert get_res.status_code == 200

    # Delete conversation -> 204
    del_res = await api_client.delete(f"/api/v1/assistant/conversations/{conv_id}")
    assert del_res.status_code == 204

    # Subsequent read -> 404
    get_after = await api_client.get(f"/api/v1/assistant/conversations/{conv_id}")
    assert get_after.status_code == 404

