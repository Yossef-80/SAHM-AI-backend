"""add workspace LLM provider settings

Revision ID: f4b271c90e12
Revises: 9ab6bffb43a6
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f4b271c90e12"
down_revision: Union[str, None] = "9ab6bffb43a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_provider_settings",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("business_id", sa.String(length=64), nullable=False),
        sa.Column("fast_provider", sa.String(length=20), nullable=True),
        sa.Column("fast_model", sa.String(length=120), nullable=True),
        sa.Column("fast_base_url", sa.String(length=500), nullable=True),
        sa.Column("fast_api_key_enc", sa.Text(), nullable=True),
        sa.Column("fast_extra_headers_enc", sa.Text(), nullable=True),
        sa.Column("strong_provider", sa.String(length=20), nullable=True),
        sa.Column("strong_model", sa.String(length=120), nullable=True),
        sa.Column("strong_base_url", sa.String(length=500), nullable=True),
        sa.Column("strong_api_key_enc", sa.Text(), nullable=True),
        sa.Column("strong_extra_headers_enc", sa.Text(), nullable=True),
        sa.Column("updated_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("business_id", name="uq_llm_provider_settings_business"),
    )
    op.create_index(
        "ix_llm_provider_settings_business_id",
        "llm_provider_settings",
        ["business_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_provider_settings_business_id", table_name="llm_provider_settings")
    op.drop_table("llm_provider_settings")
