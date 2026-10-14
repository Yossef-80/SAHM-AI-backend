"""add reusable business-scoped saved audiences

Revision ID: c3e81b7a4d26
Revises: b7c8d9e0f1a2
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

_JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")

revision: str = "c3e81b7a4d26"
down_revision: Union[str, None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_foreign_key(
    table_name: str,
    local_columns: list[str],
    referred_table: str,
    referred_columns: list[str],
) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(
        fk.get("constrained_columns") == local_columns
        and fk.get("referred_table") == referred_table
        and fk.get("referred_columns") == referred_columns
        for fk in inspector.get_foreign_keys(table_name)
    )


def _ensure_index(table_name: str, name: str, columns: list[str]) -> None:
    inspector = sa.inspect(op.get_bind())
    existing = {index.get("name") for index in inspector.get_indexes(table_name)}
    if name not in existing:
        op.create_index(name, table_name, columns)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("saved_audiences"):
        op.create_table(
            "saved_audiences",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("business_id", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("targeting", _JSON_TYPE, nullable=False),
            sa.Column(
                "source", sa.String(length=20), server_default="manual", nullable=False
            ),
            sa.Column(
                "is_archived",
                sa.Boolean(),
                server_default=sa.false(),
                nullable=False,
            ),
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
            sa.ForeignKeyConstraint(
                ["business_id"], ["businesses.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
        )
    else:
        # create_all() may have made this table at dev startup while Alembic's
        # revision stayed behind. Preserve its data and only adopt a complete
        # table with the expected ORM/migration columns.
        actual = {
            column["name"]
            for column in sa.inspect(bind).get_columns("saved_audiences")
        }
        expected = {
            "id", "business_id", "name", "description", "targeting", "source",
            "is_archived", "created_at", "updated_at",
        }
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(
                "saved_audiences already exists but is missing columns required by "
                f"revision {revision}: {', '.join(missing)}. No rows were changed."
            )

    # A create_all()-generated table uses client-side ORM defaults. Add the
    # equivalent PostgreSQL defaults so inserts outside the ORM stay compatible
    # with a table created by this migration. This does not rewrite existing rows.
    if bind.dialect.name == "postgresql":
        columns = {
            column["name"]: column
            for column in sa.inspect(bind).get_columns("saved_audiences")
        }
        if not columns["source"].get("default"):
            op.alter_column(
                "saved_audiences",
                "source",
                existing_type=sa.String(length=20),
                existing_nullable=False,
                server_default=sa.text("'manual'"),
            )
        if not columns["is_archived"].get("default"):
            op.alter_column(
                "saved_audiences",
                "is_archived",
                existing_type=sa.Boolean(),
                existing_nullable=False,
                server_default=sa.false(),
            )

    if not _has_foreign_key(
        "saved_audiences", ["business_id"], "businesses", ["id"]
    ):
        with op.batch_alter_table("saved_audiences") as batch_op:
            batch_op.create_foreign_key(
                "fk_saved_audiences_business_id_businesses",
                "businesses",
                ["business_id"],
                ["id"],
                ondelete="CASCADE",
            )

    for name, columns in (
        ("ix_saved_audiences_business_id", ["business_id"]),
        ("ix_saved_audiences_is_archived", ["is_archived"]),
        ("ix_saved_audiences_business_updated", ["business_id", "updated_at"]),
    ):
        _ensure_index("saved_audiences", name, columns)

    campaign_inspector = sa.inspect(bind)
    if not campaign_inspector.has_table("campaigns"):
        raise RuntimeError(
            "campaigns table is missing; refusing to apply saved-audience migration."
        )
    campaign_columns = {
        column["name"]
        for column in campaign_inspector.get_columns("campaigns")
    }
    has_audience_fk = _has_foreign_key(
        "campaigns", ["audience_id"], "saved_audiences", ["id"]
    )
    campaign_indexes = {
        index.get("name")
        for index in campaign_inspector.get_indexes("campaigns")
    }

    # Add only absent nullable columns and missing constraints/indexes. This
    # keeps existing campaigns and audience rows intact when create_all() has
    # partially materialized the current ORM schema.
    with op.batch_alter_table("campaigns") as batch_op:
        if "audience_id" not in campaign_columns:
            batch_op.add_column(sa.Column("audience_id", sa.String(length=64), nullable=True))
        if "audience_snapshot" not in campaign_columns:
            batch_op.add_column(sa.Column("audience_snapshot", _JSON_TYPE, nullable=True))
        if not has_audience_fk:
            batch_op.create_foreign_key(
                "fk_campaigns_audience_id_saved_audiences",
                "saved_audiences",
                ["audience_id"],
                ["id"],
                ondelete="SET NULL",
            )
        if "ix_campaigns_audience_id" not in campaign_indexes:
            batch_op.create_index("ix_campaigns_audience_id", ["audience_id"])


def downgrade() -> None:
    with op.batch_alter_table("campaigns") as batch_op:
        batch_op.drop_index("ix_campaigns_audience_id")
        batch_op.drop_constraint("fk_campaigns_audience_id_saved_audiences", type_="foreignkey")
        batch_op.drop_column("audience_snapshot")
        batch_op.drop_column("audience_id")
    op.drop_index("ix_saved_audiences_business_updated", table_name="saved_audiences")
    op.drop_index("ix_saved_audiences_is_archived", table_name="saved_audiences")
    op.drop_index("ix_saved_audiences_business_id", table_name="saved_audiences")
    op.drop_table("saved_audiences")
