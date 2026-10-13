"""usage_records: per-action attribution and an estimated flag

Revision ID: c1f4a9d27e35
Revises: b7c8d9e0f1a2
Create Date: 2026-10-13

Two additions to ``usage_records``, both needed before the product charges a
customer for token consumption:

``action``
    The billable act (``chat``, ``research.website``, ``creative.text`` ...).
    Every LLM call the AI layer makes is now metered (see
    ``app/core/ai_usage.py``); without an action label the usage screen can only
    show a per-model total, which cannot answer "what is spending my budget?".

``estimated``
    ``true`` when the provider returned no usage block and the tokens were
    derived from text length. A billing report must be able to separate a
    measured number from a guess, so this is stored rather than inferred.

Both columns are nullable/defaulted, so existing rows and existing inserts keep
working unchanged.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c1f4a9d27e35"
down_revision: Union[str, None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "usage_records", sa.Column("action", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "usage_records",
        sa.Column(
            "estimated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index("ix_usage_records_action", "usage_records", ["action"])


def downgrade() -> None:
    op.drop_index("ix_usage_records_action", table_name="usage_records")
    op.drop_column("usage_records", "estimated")
    op.drop_column("usage_records", "action")
