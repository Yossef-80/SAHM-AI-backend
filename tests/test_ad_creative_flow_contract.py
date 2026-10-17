from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select

from app.ai.tools.ad_previews import AdPreviewNotFoundError, read_ad_preview, store_ad_preview
from app.core.errors import NotFoundError, ValidationFailed
from app.core.orchestrator import service
from app.db.models import Business, Conversation, Message
from app.schemas.assistant import AssistantAnswer, AssistantMessageRequest
from app.schemas.blocks import AdStyleAnalysisBlock, AdStyleSpec, MessageWithBlocks, TextBlock


@pytest.mark.asyncio
async def test_answer_is_resolved_from_persisted_option_payload_and_message_id_is_stable(
    seeded_db, monkeypatch
):
    business = await seeded_db.get(Business, "biz_test")
    assert business is not None
    conversation = Conversation(
        id="conv_ad_flow",
        business_id=business.id,
        title="Competitor ad selection",
        context={},
    )
    question = Message(
        id="msg_ad_choice",
        conversation_id=conversation.id,
        role="assistant",
        content="Choose one image",
        blocks=[
            {
                "type": "question_with_options",
                "question": "Choose one image",
                "options": [
                    {
                        "id": "ad-option-1",
                        "label": "Acme — Spring ad",
                        "payload": {
                            "kind": "meta_ad_preview",
                            "previewId": "1" * 32,
                            "adId": "ad-42",
                            "pageName": "Acme",
                        },
                    }
                ],
                "answeredOptionIds": [],
            }
        ],
    )
    seeded_db.add_all([conversation, question])
    await seeded_db.commit()

    captured: dict[str, object] = {}

    async def fake_orchestrate(*args, **kwargs):
        captured.update(kwargs)
        return MessageWithBlocks(
            blocks=[
                TextBlock(text="Selected image analysis complete."),
                AdStyleAnalysisBlock(
                    source_ad_id="ad-42",
                    source_name="Acme",
                    analysis=AdStyleSpec(summary="A minimal product-focused ad."),
                ),
            ]
        )

    monkeypatch.setattr("app.ai.agents.orchestrator.brain.orchestrate", fake_orchestrate)
    monkeypatch.setattr(
        "app.ai.memory.compressor.manage_session_memory_with_token_cap",
        AsyncMock(return_value=([{"role": "assistant", "content": "Choose one image"}], None)),
    )
    monkeypatch.setattr(
        "app.ai.memory.consolidator.memory_consolidator.schedule",
        Mock(),
    )

    response = await service.handle_message(
        seeded_db,
        AssistantMessageRequest(
            conversation_id=conversation.id,
            message="Acme — Spring ad",
            answer=AssistantAnswer(message_id=question.id, option_ids=["ad-option-1"]),
        ),
        llm=object(),
        fast_llm=None,
        request_id="req-ad-flow-test",
        business=business,
    )

    assert captured["ad_flow_selection"] == {
        "kind": "meta_ad_preview",
        "previewId": "1" * 32,
        "adId": "ad-42",
        "pageName": "Acme",
    }
    assert captured["ad_tool_context_factory"]
    updated_question = await seeded_db.get(Message, question.id)
    assert updated_question.blocks[0]["answeredOptionIds"] == ["ad-option-1"]
    with pytest.raises(ValidationFailed):
        await service._resolve_answer_selection(
            seeded_db,
            conversation_id=conversation.id,
            answer=AssistantAnswer(message_id=question.id, option_ids=["ad-option-1"]),
        )
    assistant_row = await seeded_db.scalar(
        select(Message).where(
            Message.conversation_id == conversation.id,
            Message.role == "assistant",
            Message.id != question.id,
        )
    )
    assert assistant_row is not None
    assert response.message.id == assistant_row.id


@pytest.mark.asyncio
async def test_failed_vlm_attempt_leaves_image_choice_retryable(seeded_db):
    business = await seeded_db.get(Business, "biz_test")
    assert business is not None
    conversation = Conversation(
        id="conv_retry_ad_flow",
        business_id=business.id,
        title="Retry selected ad",
        context={},
    )
    question = Message(
        id="msg_retry_ad_choice",
        conversation_id=conversation.id,
        role="assistant",
        content="Choose one image",
        blocks=[
            {
                "type": "question_with_options",
                "question": "Choose one image",
                "options": [
                    {
                        "id": "ad-option-1",
                        "label": "Acme — Spring ad",
                        "payload": {
                            "kind": "meta_ad_preview",
                            "previewId": "2" * 32,
                            "adId": "ad-42",
                        },
                    }
                ],
                "answeredOptionIds": [],
            }
        ],
    )
    seeded_db.add_all([conversation, question])
    await seeded_db.commit()

    await service._resolve_answer_selection(
        seeded_db,
        conversation_id=conversation.id,
        answer=AssistantAnswer(message_id=question.id, option_ids=["ad-option-1"]),
    )
    unchanged_question = await seeded_db.get(Message, question.id)
    assert unchanged_question.blocks[0]["answeredOptionIds"] == []


@pytest.mark.asyncio
async def test_conversation_ids_cannot_cross_business_boundaries(seeded_db):
    business = await seeded_db.get(Business, "biz_test")
    assert business is not None
    conversation = Conversation(
        id="conv_owned_by_test",
        business_id=business.id,
        title="Scoped conversation",
        context={},
    )
    seeded_db.add(conversation)
    await seeded_db.commit()

    with pytest.raises(NotFoundError):
        await service.get_or_create_conversation(
            seeded_db, conversation.id, business_id="biz_other"
        )


@pytest.mark.asyncio
async def test_preview_endpoint_requires_auth_and_returns_scoped_image(api_client, tmp_path):
    preview_id = store_ad_preview(
        upload_dir=tmp_path,
        business_id="biz_test",
        image_bytes=b"scoped preview bytes",
        mime_type="image/png",
    )
    unauthenticated = await api_client.get(
        f"/api/v1/assistant/ad-previews/{preview_id}",
        headers={"X-Workspace-Id": "biz_test"},
    )
    assert unauthenticated.status_code == 401

    from app.api.deps import get_current_principal
    from app.config import get_settings

    app = api_client._transport.app
    prior_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_current_principal] = lambda: object()
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(
        update={"upload_dir": str(tmp_path)}
    )
    try:
        response = await api_client.get(
            f"/api/v1/assistant/ad-previews/{preview_id}",
            headers={"X-Workspace-Id": "biz_test"},
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(prior_overrides)

    assert response.status_code == 200
    assert response.content == b"scoped preview bytes"
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["cache-control"] == "private, max-age=60"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_preview_bytes_are_tenant_scoped_expiring_and_not_exposed_as_paths(tmp_path):
    created_at = time.time()
    preview_id = store_ad_preview(
        upload_dir=tmp_path,
        business_id="biz_one",
        image_bytes=b"fake png bytes",
        mime_type="image/png",
        now=created_at,
    )
    data, mime_type = read_ad_preview(
        upload_dir=tmp_path,
        business_id="biz_one",
        preview_id=preview_id,
        now=created_at + 1,
    )
    assert data == b"fake png bytes"
    assert mime_type == "image/png"
    with pytest.raises(AdPreviewNotFoundError):
        read_ad_preview(
            upload_dir=tmp_path,
            business_id="biz_two",
            preview_id=preview_id,
            now=created_at + 1,
        )
    with pytest.raises(AdPreviewNotFoundError):
        read_ad_preview(
            upload_dir=tmp_path,
            business_id="biz_one",
            preview_id=preview_id,
            now=created_at + 24 * 60 * 60 + 1,
        )
    with pytest.raises(AdPreviewNotFoundError):
        read_ad_preview(
            upload_dir=tmp_path,
            business_id="biz_one",
            preview_id="../secret",
            now=created_at + 1,
        )
