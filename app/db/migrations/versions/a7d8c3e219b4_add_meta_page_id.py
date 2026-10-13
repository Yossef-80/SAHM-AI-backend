"""add a workspace Meta Page ID for paused ad creation

Revision ID: a7d8c3e219b4
Revises: f4b271c90e12
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a7d8c3e219b4"
down_revision: Union[str, None] = "f4b271c90e12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("tool_settings", sa.Column("meta_page_id", sa.String(length=120), nullable=True))


def downgrade() -> None:
    op.drop_column("tool_settings", "meta_page_id")
