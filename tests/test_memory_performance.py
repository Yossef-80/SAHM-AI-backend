"""Tests for memory pipeline performance: token cap raising, 25% growth gating, and composite indexes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy import Index

from app.ai.memory.compressor import count_tokens, manage_session_memory_with_token_cap
from app.db.models import BrandMemoryItem, Conversation, Message


def test_composite_indexes_declared_on_models() -> None:
    """Verify that all three hot-path composite indexes are declared."""
    # 1. BrandMemoryItem(business_id, kind)
    bmi_indexes = {
        idx.name: [col.name if hasattr(col, "name") else str(col) for col in idx.columns]
        for idx in BrandMemoryItem.__table__.indexes
    }
    assert "ix_brand_memory_items_biz_kind" in bmi_indexes
    assert bmi_indexes["ix_brand_memory_items_biz_kind"] == ["business_id", "kind"]

    # 2. Conversation(business_id, updated_at)
    conv_indexes = {
        idx.name: [col.name if hasattr(col, "name") else str(col) for col in idx.columns]
        for idx in Conversation.__table__.indexes
    }
    assert "ix_conversations_biz_updated" in conv_indexes
    assert conv_indexes["ix_conversations_biz_updated"] == ["business_id", "updated_at"]

    # 3. Message(conversation_id, created_at)
    msg_indexes = {
        idx.name: [col.name if hasattr(col, "name") else str(col) for col in idx.columns]
        for idx in Message.__table__.indexes
    }
    assert "ix_messages_conv_created" in msg_indexes
    assert msg_indexes["ix_messages_conv_created"] == ["conversation_id", "created_at"]


@pytest.mark.asyncio
async def test_session_memory_below_cap_returns_raw_history() -> None:
    """Conversations below max_tokens return complete history without LLM call."""
    conv = Conversation(
        id="conv_1",
        business_id="biz_1",
        title="Test",
        context={},
        summary=None,
    )
    messages = [
        Message(id="m1", conversation_id="conv_1", role="user", content="Hello", blocks=[]),
        Message(id="m2", conversation_id="conv_1", role="assistant", content="Hi there!", blocks=[]),
    ]
    mock_db = AsyncMock()

    with patch("app.ai.memory.compressor.get_llm") as mock_get_llm:
        history, summary = await manage_session_memory_with_token_cap(
            mock_db,
            conversation=conv,
            messages=messages,
            max_tokens=10000,
        )
        assert len(history) == 2
        assert summary is None
        mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_session_memory_growth_gating_prevents_redundant_llm_calls() -> None:
    """When conversation has a summary and token history grows < 25%, reuse summary without LLM."""
    # Build the transcript with the *current* tokenizer instead of a hardcoded
    # character count: the watermark is a token count, and a change to the
    # counter must not silently turn a 5%-growth test into a 50%-growth test.
    long_text = "word " * 10500
    total = count_tokens(long_text) + count_tokens("Short reply")
    conv = Conversation(
        id="conv_gate",
        business_id="biz_1",
        title="Test Growth Gating",
        context={"summary_token_watermark": int(total / 1.05)},
        summary="Existing high quality summary.",
    )
    assert total > 10_000
    messages = [
        Message(id="m1", conversation_id="conv_gate", role="user", content=long_text, blocks=[]),
        Message(id="m2", conversation_id="conv_gate", role="assistant", content="Short reply", blocks=[]),
    ]
    mock_db = AsyncMock()

    with patch("app.ai.memory.compressor.get_llm") as mock_get_llm:
        history, summary = await manage_session_memory_with_token_cap(
            mock_db,
            conversation=conv,
            messages=messages,
            max_tokens=10000,
        )
        # Gated! Reused existing summary, did NOT invoke fast_llm
        assert summary == "Existing high quality summary."
        mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_session_memory_regenerates_when_growth_exceeds_twenty_five_percent() -> None:
    """When history grows >= 25% over the watermark, regenerate summary and update watermark."""
    long_text = "word " * 13000
    total = count_tokens(long_text) + count_tokens("Recent turn")
    conv = Conversation(
        id="conv_regen",
        business_id="biz_1",
        title="Test Regen",
        context={"summary_token_watermark": int(total / 1.30)},
        summary="Old summary.",
    )
    messages = [
        Message(id="m1", conversation_id="conv_regen", role="user", content=long_text, blocks=[]),
        Message(id="m2", conversation_id="conv_regen", role="assistant", content="Recent turn", blocks=[]),
    ]
    mock_db = AsyncMock()

    mock_chat = AsyncMock()
    mock_chat.ainvoke.return_value = MagicMock(content="Brand new updated 30% summary.")
    mock_llm_obj = MagicMock(chat=mock_chat)

    with patch("app.ai.memory.compressor.get_llm", return_value=mock_llm_obj):
        history, summary = await manage_session_memory_with_token_cap(
            mock_db,
            conversation=conv,
            messages=messages,
            max_tokens=10000,
        )
        # Regenerated!
        assert "Brand new updated 30% summary" in summary
        assert mock_chat.ainvoke.called
        # Watermark updated in context!
        watermark = conv.context.get("summary_token_watermark")
        assert watermark is not None
        assert watermark >= total


def test_conversation_messages_ordering_model_declaration() -> None:
    """Ensure Conversation.messages relationship orders by created_at and role desc."""
    from app.db.models import Conversation
    rel = Conversation.__mapper__.relationships["messages"]
    order_clauses = [str(x) for x in rel.order_by]
    assert any("created_at" in c for c in order_clauses)
    assert any("role DESC" in c for c in order_clauses)

