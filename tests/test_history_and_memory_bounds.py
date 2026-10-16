"""Bounds that keep a long-lived workspace fast.

Three invariants, all of which used to be violated and all of which showed up as
"the app got slow":

1. Opening a conversation loads a *page* of messages, not the entire history.
2. A message row never carries an inline base64 payload into the database.
3. The long-term-memory debouncer leaves no sleeping task behind at shutdown.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.orchestrator.service import (
    MAX_MESSAGE_BLOCK_CHARS,
    MAX_MESSAGE_CONTENT_CHARS,
    _persist_message,
    _strip_inline_blobs,
    get_conversation,
)
from app.db.models import Conversation, Message


async def _conversation_with_messages(db: AsyncSession, count: int) -> str:
    conversation = Conversation(id="conv_page", business_id="biz_test", title="Paging", context={})
    db.add(conversation)
    for index in range(count):
        db.add(
            Message(
                id=f"msg_{index:03d}",
                conversation_id=conversation.id,
                role="user" if index % 2 == 0 else "assistant",
                content=f"message {index}",
                blocks=[{"type": "text", "text": f"message {index}"}],
            )
        )
    await db.commit()
    return conversation.id


class TestConversationPaging:
    async def test_only_the_newest_page_is_returned(self, seeded_db: AsyncSession):
        conversation_id = await _conversation_with_messages(seeded_db, 120)

        detail = await get_conversation(seeded_db, conversation_id, business_id="biz_test", limit=50)

        assert len(detail.messages) == 50
        # Newest page... oldest page. Chronological order is preserved for the
        # client; the window simply starts later.
        contents = [message.content for message in detail.messages]
        assert contents == sorted(contents, key=lambda text: int(text.split()[-1]))
        assert contents[0] == "message 70"
        assert contents[-1] == "message 119"

    async def test_the_before_cursor_walks_backwards_without_gaps(self, seeded_db: AsyncSession):
        conversation_id = await _conversation_with_messages(seeded_db, 120)

        newest = await get_conversation(seeded_db, conversation_id, business_id="biz_test", limit=50)
        older = await get_conversation(
            seeded_db,
            conversation_id,
            business_id="biz_test",
            limit=50,
            before=newest.messages[0].id,
        )

        assert len(older.messages) == 50
        assert older.messages[-1].content == "message 69"
        assert older.messages[0].content == "message 20"

    async def test_the_page_size_is_capped(self, seeded_db: AsyncSession):
        conversation_id = await _conversation_with_messages(seeded_db, 300)

        detail = await get_conversation(seeded_db, conversation_id, business_id="biz_test", limit=10_000)

        assert len(detail.messages) == 200

    async def test_another_workspace_gets_a_404_not_a_partial_read(self, seeded_db: AsyncSession):
        conversation_id = await _conversation_with_messages(seeded_db, 5)

        from app.core.errors import NotFoundError

        with pytest.raises(NotFoundError):
            await get_conversation(seeded_db, conversation_id, business_id="biz_other")


class TestMessagePayloadGuard:
    def test_inline_data_urls_are_replaced_not_stored(self):
        payload = "data:image/png;base64," + "A" * (MAX_MESSAGE_BLOCK_CHARS + 10)

        cleaned, removed = _strip_inline_blobs([{"type": "image_url", "image_url": payload}], limit=MAX_MESSAGE_BLOCK_CHARS)

        assert removed > 0
        assert "base64" not in str(cleaned)
        assert "inline asset removed" in str(cleaned)

    def test_normal_blocks_are_untouched(self):
        blocks = [{"type": "text", "text": "a normal assistant answer"}]

        cleaned, removed = _strip_inline_blobs(blocks, limit=MAX_MESSAGE_BLOCK_CHARS)

        assert cleaned == blocks
        assert removed == 0

    async def test_persisted_messages_respect_the_limits(self, seeded_db: AsyncSession):
        conversation = Conversation(id="conv_guard", business_id="biz_test", title="", context={})
        seeded_db.add(conversation)
        await seeded_db.commit()

        blob = "data:image/png;base64," + "B" * (MAX_MESSAGE_BLOCK_CHARS + 50)
        row = await _persist_message(
            seeded_db,
            conversation_id=conversation.id,
            role="assistant",
            content="x" * (MAX_MESSAGE_CONTENT_CHARS + 500),
            blocks=[{"type": "image_url", "image_url": blob}],
        )
        await seeded_db.commit()

        refreshed = await seeded_db.get(Message, row.id)
        assert refreshed is not None
        assert len(refreshed.content) == MAX_MESSAGE_CONTENT_CHARS
        assert "base64" not in str(refreshed.blocks)


class TestDebouncerLifecycle:
    async def test_shutdown_cancels_pending_timers(self):
        from app.ai.memory.compressor import MemoryDebouncer

        debouncer = MemoryDebouncer(delay_seconds=3600)
        debouncer.schedule_ltm_update("biz_a")
        debouncer.schedule_ltm_update("biz_b")
        assert debouncer.pending == 2

        await debouncer.shutdown()

        assert debouncer.pending == 0
        assert debouncer._timers == {}
        # Nothing left to wake up and write to a database we are closing.
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]

    async def test_rescheduling_replaces_the_previous_timer(self):
        from app.ai.memory.compressor import MemoryDebouncer

        debouncer = MemoryDebouncer(delay_seconds=3600)
        try:
            debouncer.schedule_ltm_update("biz_a")
            debouncer.schedule_ltm_update("biz_a")
            assert debouncer.pending == 1
        finally:
            await debouncer.shutdown()

    async def test_scheduling_after_shutdown_is_a_no_op(self):
        from app.ai.memory.compressor import MemoryDebouncer

        debouncer = MemoryDebouncer(delay_seconds=3600)
        await debouncer.shutdown()

        debouncer.schedule_ltm_update("biz_a")

        assert debouncer.pending == 0
