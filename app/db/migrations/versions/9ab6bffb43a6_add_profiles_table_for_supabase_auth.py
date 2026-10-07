"""add_profiles_table_for_supabase_auth

Revision ID: 9ab6bffb43a6
Revises: cfe093781e86
Create Date: 2026-10-05 22:48:57.896719
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9ab6bffb43a6'
down_revision: Union[str, None] = 'cfe093781e86'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "profiles",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("supabase_user_id", sa.UUID(), nullable=False),
        sa.Column("business_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_profiles"),
        sa.UniqueConstraint("supabase_user_id", name="uq_profiles_supabase_user_id"),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], ondelete="CASCADE", name="fk_profiles_business_id"),
        sa.CheckConstraint("role IN ('owner','member','viewer')", name="ck_profiles_role"),
    )
    op.create_index("ix_profiles_supabase_user_id", "profiles", ["supabase_user_id"], unique=False)
    op.create_index("ix_profiles_business_id", "profiles", ["business_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_profiles_business_id", table_name="profiles")
    op.drop_index("ix_profiles_supabase_user_id", table_name="profiles")
    op.drop_table("profiles")
