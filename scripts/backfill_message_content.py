"""One-off idempotent backfill script for Message.content and conversation summaries.

Regenerates Message.content from Message.blocks using canonical blocks_to_plaintext().
Invalidates Conversation.summary for affected conversations so summaries are rebuilt
losslessly on subsequent turns.

Usage:
    python scripts/backfill_message_content.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models import Conversation, Message
from app.db.session import get_sessionmaker
from app.schemas.blocks import blocks_to_plaintext

logger = get_logger("backfill_message_content")


async def run_backfill(*, dry_run: bool = False, session: AsyncSession | None = None) -> dict[str, int]:
    """Regenerate Message.content for assistant messages from blocks.
    
    Returns a dictionary of counts:
        {
            "total_inspected": int,
            "updated": int,
            "unchanged": int,
            "lacking_blocks": int,
            "summaries_invalidated": int,
        }
    """
    counts = {
        "total_inspected": 0,
        "updated": 0,
        "unchanged": 0,
        "lacking_blocks": 0,
        "summaries_invalidated": 0,
    }

    own_session = session is None
    session_maker = get_sessionmaker() if own_session else None

    async def _process(sess: AsyncSession) -> None:
        stmt = (
            select(Message)
            .where(Message.role == "assistant")
            .order_by(Message.created_at.asc())
        )
        res = await sess.execute(stmt)
        messages = res.scalars().all()

        counts["total_inspected"] = len(messages)
        affected_conv_ids: set[str] = set()

        for msg in messages:
            raw_blocks = msg.blocks
            if isinstance(raw_blocks, str):
                try:
                    raw_blocks = json.loads(raw_blocks)
                except Exception:
                    raw_blocks = None

            if not raw_blocks:
                counts["lacking_blocks"] += 1
                logger.warning(
                    "assistant_message_lacking_blocks",
                    extra={"message_id": msg.id, "conversation_id": msg.conversation_id},
                )
                continue

            new_content = blocks_to_plaintext(raw_blocks)
            if (msg.content or "") != new_content:
                msg.content = new_content
                affected_conv_ids.add(msg.conversation_id)
                counts["updated"] += 1
            else:
                counts["unchanged"] += 1

        # Invalidate conversation summaries for conversations with updated messages
        if affected_conv_ids:
            for conv_id in affected_conv_ids:
                conv_res = await sess.execute(select(Conversation).where(Conversation.id == conv_id))
                conv = conv_res.scalar_one_or_none()
                if conv and conv.summary:
                    conv.summary = None
                    counts["summaries_invalidated"] += 1

        if dry_run:
            logger.info("backfill_dry_run_rolling_back", extra=counts)
            await sess.rollback()
        else:
            await sess.commit()
            logger.info("backfill_committed", extra=counts)

    if own_session and session_maker:
        async with session_maker() as sess:
            await _process(sess)
    else:
        assert session is not None
        await _process(session)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill message content from blocks.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without committing changes.")
    args = parser.parse_args()

    print(f"Starting backfill (dry_run={args.dry_run})...")
    results = asyncio.run(run_backfill(dry_run=args.dry_run))
    print(f"Backfill finished:")
    print(f"  Total assistant messages: {results['total_inspected']}")
    print(f"  Updated:                 {results['updated']}")
    print(f"  Unchanged:               {results['unchanged']}")
    print(f"  Lacking blocks:          {results['lacking_blocks']}")
    print(f"  Summaries invalidated:   {results['summaries_invalidated']}")


if __name__ == "__main__":
    main()
