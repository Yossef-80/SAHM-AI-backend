"""widen creatives.url to text

Revision ID: d2f1a7c04b18
Revises: b60e91f42c73
Create Date: 2026-10-12

``creatives.url`` holds an inline data URL: generated images and uploads both
store their bytes there, because Meta's ``/adimages`` edge is handed the raw
``bytes`` rather than a URL to fetch. A real photo is hundreds of kilobytes, so
the encoded value runs into the millions of characters.

The column was ``String(1000)``. SQLite ignores VARCHAR length entirely, which
is why this never surfaced in development or in the test suite — every test runs
on SQLite. Postgres enforces it, so any creative whose data URL exceeded 1000
characters failed at INSERT with ``StringDataRightTruncationError``.

The model already declares ``Text``; this migration brings existing databases in
line with it. ``create_all()`` only creates *missing* tables, so a database that
already has ``creatives`` keeps the old type until this runs.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d2f1a7c04b18"
down_revision: Union[str, None] = "b60e91f42c73"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # Widening only: every value that fitted before still fits, so no data
        # is rewritten and nothing can be truncated.
        op.alter_column(
            "creatives",
            "url",
            existing_type=sa.String(length=1000),
            type_=sa.Text(),
            existing_nullable=False,
        )
    # On SQLite a VARCHAR(1000) column already accepts any length, so there is
    # nothing to change — and SQLite cannot ALTER a column type in place anyway.


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.alter_column(
            "creatives",
            "url",
            existing_type=sa.Text(),
            type_=sa.String(length=1000),
            existing_nullable=False,
        )
