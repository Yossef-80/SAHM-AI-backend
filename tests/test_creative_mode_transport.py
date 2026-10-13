"""The mode picked in the UI must survive the trip to the tool gate.

`test_creative_mode_gate.py` pins the gate itself. This pins the *transport*,
which is the part that was hand-wired: a camelCase `creativeMode` in the request
body has to become `creative_mode` in the sticky conversation context, survive
into the next turn, and arrive in the dict that `brain.py:2328` copies into
`user_supplied` — because that copy is the only thing the gate can see.

If any link broke, the picker would look like it worked in the UI while Layla
kept guessing from keywords.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from app.ai.tools.core_bridge import _validate_explicit_intent
from app.schemas.blocks import TextBlock


def _layla_result():
    from app.ai.agents.orchestrator import brain

    return brain.OrchestratorResult(
        blocks=[TextBlock(text="plan")],
        persona_id="layla",
        creative_prompt_workflow=True,
        creative_prompt_type="image",
    )


async def _stored_context(db_session, conversation_id: str) -> dict:
    row = (
        await db_session.execute(
            text("SELECT context FROM conversations WHERE id = :i"),
            {"i": conversation_id},
        )
    ).scalar_one()
    return json.loads(row) if isinstance(row, str) else row


async def _post_mode(api_client, monkeypatch, mode: str) -> tuple[str, list[dict]]:
    """Send one turn carrying `mode`; return (conversation_id, contexts seen)."""
    from app.ai.agents.orchestrator import brain

    seen: list[dict] = []

    async def fake_orchestrate(*_args, **kwargs):
        seen.append(dict(kwargs.get("context") or {}))
        return _layla_result()

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)

    response = await api_client.post(
        "/api/v1/assistant/message",
        json={
            "message": "عايز تصميم للبوتاجاز ده",
            "context": {"workflow": "creative_prompt", "promptType": "image", "creativeMode": mode},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["conversationId"], seen


@pytest.mark.asyncio
async def test_camel_case_mode_lands_in_the_persisted_context(
    api_client, monkeypatch, db_session
) -> None:
    conversation_id, seen = await _post_mode(api_client, monkeypatch, "prompt")

    # The orchestrator received it under the snake_case field name...
    assert seen[0].get("creative_mode") == "prompt", seen[0]
    # ...and it was written to the row, not just held in memory.
    stored = await _stored_context(db_session, conversation_id)
    assert stored.get("creative_mode") == "prompt", stored


@pytest.mark.asyncio
async def test_mode_is_sticky_across_turns(api_client, monkeypatch, db_session) -> None:
    """Choosing design once must not silently revert on the next message."""
    conversation_id, _ = await _post_mode(api_client, monkeypatch, "design")

    from app.ai.agents.orchestrator import brain

    seen: list[dict] = []

    async def fake_orchestrate(*_args, **kwargs):
        seen.append(dict(kwargs.get("context") or {}))
        return _layla_result()

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)

    follow_up = await api_client.post(
        "/api/v1/assistant/message",
        json={"conversationId": conversation_id, "message": "كمّل"},
    )
    assert follow_up.status_code == 200, follow_up.text
    assert seen[0].get("creative_mode") == "design", seen[0]


@pytest.mark.asyncio
async def test_the_persisted_mode_actually_reaches_the_tool_gate(
    api_client, monkeypatch
) -> None:
    """End to end: HTTP body -> context -> user_supplied -> gate decision."""
    _, seen = await _post_mode(api_client, monkeypatch, "prompt")

    # brain.py:2328 does exactly this copy before handing tools their context.
    supplied = dict(seen[0])
    supplied["latest_user_message"] = "اعمل صورة للبوتاجاز"
    gate_context = SimpleNamespace(user_supplied=supplied)

    # Prompt mode refuses pixels even though the wording asks for them.
    assert _validate_explicit_intent("generate_image", gate_context) is not None

    supplied["creative_mode"] = "design"
    assert _validate_explicit_intent("generate_image", gate_context) is None


@pytest.mark.asyncio
async def test_an_unknown_mode_is_rejected(api_client) -> None:
    response = await api_client.post(
        "/api/v1/assistant/message",
        json={"message": "أهلاً", "context": {"creativeMode": "teleport"}},
    )
    assert response.status_code == 422, response.text
