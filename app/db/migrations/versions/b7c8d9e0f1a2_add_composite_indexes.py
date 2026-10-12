"""add composite indexes for hot query paths

Revision ID: b7c8d9e0f1a2
Revises: e5a1b2c3d4e5
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "e5a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_brand_memory_items_biz_kind",
        "brand_memory_items",
        ["business_id", "kind"],
        unique=False,
    )
    op.create_index(
        "ix_conversations_biz_updated",
        "conversations",
        ["business_id", "updated_at"],
        unique=False,
    )
    op.create_index(
        "ix_messages_conv_created",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conv_created", table_name="messages")
    op.drop_index("ix_conversations_biz_updated", table_name="conversations")
    op.drop_index("ix_brand_memory_items_biz_kind", table_name="brand_memory_items")
