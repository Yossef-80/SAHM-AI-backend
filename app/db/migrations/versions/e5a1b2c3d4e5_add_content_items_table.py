"""add content_items table for content calendar

Revision ID: e5a1b2c3d4e5
Revises: f4b271c90e12
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "e5a1b2c3d4e5"
down_revision: Union[str, None] = "f4b271c90e12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB, "postgresql")
    op.create_table(
        "content_items",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("business_id", sa.String(length=64), nullable=False),
        sa.Column("date", sa.String(length=10), nullable=False),
        sa.Column("time", sa.String(length=5), nullable=True),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("format", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("hook", sa.Text(), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("hashtags", json_type, nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="scheduled"),
        sa.Column("campaign_id", sa.String(length=64), nullable=True),
        sa.Column("creative_id", sa.String(length=64), nullable=True),
        sa.Column("pillar", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["creative_id"], ["creatives.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_content_items_business_id"), "content_items", ["business_id"], unique=False)
    op.create_index(op.f("ix_content_items_date"), "content_items", ["date"], unique=False)
    op.create_index(op.f("ix_content_items_campaign_id"), "content_items", ["campaign_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_content_items_campaign_id"), table_name="content_items")
    op.drop_index(op.f("ix_content_items_date"), table_name="content_items")
    op.drop_index(op.f("ix_content_items_business_id"), table_name="content_items")
    op.drop_table("content_items")
