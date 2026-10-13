"""Shared assistant wire fixtures must validate in backend Pydantic and frontend Zod."""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.assistant import AssistantMessageResponse


def test_assistant_message_fixture_matches_backend_response_and_block_contract() -> None:
    fixture = Path(__file__).parent / "fixtures" / "assistant_message_response.json"
    wire = json.loads(fixture.read_text(encoding="utf-8"))
    parsed = AssistantMessageResponse.model_validate(wire)

    assert parsed.conversation_id == "conv_assistant_contract"
    assert parsed.user_message is not None
    assert parsed.user_message.role == "user"
    assert parsed.message.role == "assistant"
    assert parsed.message.persona_id == "researcher"
    assert [block.type.value for block in parsed.message.blocks] == [
        "text",
        "findings",
        "action_card",
    ]
    action = parsed.message.blocks[2]
    assert action.task == "ad_variations"
    assert action.target_persona == "layla"

    # The exact serialized aliases are consumed by frontend/scripts/smoke-assistant-contract.cjs.
    serialized = parsed.model_dump(mode="json", by_alias=True)
    assert serialized["conversationId"] == wire["conversationId"]
    assert serialized["message"]["personaId"] == "researcher"
