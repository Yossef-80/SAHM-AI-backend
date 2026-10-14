"""add composite indexes for hot query paths

Revision ID: b7c8d9e0f1a2
Revises: d2f1a7c04b18
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "d2f1a7c04b18"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ensure_index(name: str, table_name: str, columns: list[str]) -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {index.get("name") for index in inspector.get_indexes(table_name)}
    if name not in existing:
        op.create_index(name, table_name, columns, unique=False)


def upgrade() -> None:
    # create_all() can materialize current-model indexes before Alembic's
    # version table catches up. Make the pending revision safe in that case.
    _ensure_index(
        "ix_brand_memory_items_biz_kind",
        "brand_memory_items",
        ["business_id", "kind"],
    )
    _ensure_index(
        "ix_conversations_biz_updated",
        "conversations",
        ["business_id", "updated_at"],
    )
    _ensure_index(
        "ix_messages_conv_created",
        "messages",
        ["conversation_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_messages_conv_created", table_name="messages")
    op.drop_index("ix_conversations_biz_updated", table_name="conversations")
    op.drop_index("ix_brand_memory_items_biz_kind", table_name="brand_memory_items")
