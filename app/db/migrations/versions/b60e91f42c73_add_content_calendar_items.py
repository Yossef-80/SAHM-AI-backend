"""add workspace-scoped content calendar items

Revision ID: b60e91f42c73
Revises: a7d8c3e219b4
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b60e91f42c73"
down_revision: Union[str, None] = "a7d8c3e219b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "content_items",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("business_id", sa.String(length=64), nullable=False),
        sa.Column("campaign_id", sa.String(length=64), nullable=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("time", sa.String(length=5), nullable=True),
        sa.Column("channel", sa.String(length=20), nullable=False),
        sa.Column("format", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("hook", sa.Text(), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("hashtags", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("pillar", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "channel IN ('instagram','facebook','tiktok','email','blog')",
            name="ck_content_items_channel",
        ),
        sa.CheckConstraint(
            "format IN ('post','reel','story','ad','email','article')",
            name="ck_content_items_format",
        ),
        sa.CheckConstraint(
            "status IN ('idea','draft','scheduled','published')",
            name="ck_content_items_status",
        ),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["campaign_id"], ["campaigns.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_content_items_business_date", "content_items", ["business_id", "date"])
    op.create_index("ix_content_items_business_id", "content_items", ["business_id"])
    op.create_index("ix_content_items_campaign_id", "content_items", ["campaign_id"])
    op.create_index("ix_content_items_date", "content_items", ["date"])
    op.create_index("ix_content_items_status", "content_items", ["status"])


def downgrade() -> None:
    op.drop_index("ix_content_items_status", table_name="content_items")
    op.drop_index("ix_content_items_date", table_name="content_items")
    op.drop_index("ix_content_items_campaign_id", table_name="content_items")
    op.drop_index("ix_content_items_business_id", table_name="content_items")
    op.drop_index("ix_content_items_business_date", table_name="content_items")
    op.drop_table("content_items")
