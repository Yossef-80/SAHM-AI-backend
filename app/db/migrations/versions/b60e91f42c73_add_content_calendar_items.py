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


def _existing_foreign_key(
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

    if not inspector.has_table("content_items"):
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
    else:
        # In dev, SQLAlchemy's create_all() may have created this table before
        # Alembic ran. Keep its rows; only accept it as this revision's table if
        # it has the full expected set of columns. Never silently bless an
        # unrelated or incomplete table.
        actual = {
            column["name"]
            for column in sa.inspect(bind).get_columns("content_items")
        }
        expected = {
            "id", "business_id", "campaign_id", "date", "time", "channel",
            "format", "title", "hook", "caption", "hashtags", "status",
            "pillar", "created_at", "updated_at",
        }
        missing = sorted(expected - actual)
        if missing:
            raise RuntimeError(
                "content_items already exists but is missing columns required by "
                f"revision {revision}: {', '.join(missing)}. No rows were changed."
            )

    # Repair only missing, non-destructive objects. This also covers a table
    # that create_all() made from the current ORM metadata.
    checks = (
        (
            "ck_content_items_channel",
            "channel IN ('instagram','facebook','tiktok','email','blog')",
        ),
        (
            "ck_content_items_format",
            "format IN ('post','reel','story','ad','email','article')",
        ),
        (
            "ck_content_items_status",
            "status IN ('idea','draft','scheduled','published')",
        ),
    )
    inspector = sa.inspect(bind)
    existing_checks = {
        check.get("name")
        for check in inspector.get_check_constraints("content_items")
    }
    for name, condition in checks:
        if name not in existing_checks:
            op.create_check_constraint(name, "content_items", condition)

    if not _existing_foreign_key(
        "content_items", ["business_id"], "businesses", ["id"]
    ):
        op.create_foreign_key(
            "fk_content_items_business_id_businesses",
            "content_items",
            "businesses",
            ["business_id"],
            ["id"],
            ondelete="CASCADE",
        )
    if not _existing_foreign_key(
        "content_items", ["campaign_id"], "campaigns", ["id"]
    ):
        op.create_foreign_key(
            "fk_content_items_campaign_id_campaigns",
            "content_items",
            "campaigns",
            ["campaign_id"],
            ["id"],
            ondelete="SET NULL",
        )

    for name, columns in (
        ("ix_content_items_business_date", ["business_id", "date"]),
        ("ix_content_items_business_id", ["business_id"]),
        ("ix_content_items_campaign_id", ["campaign_id"]),
        ("ix_content_items_date", ["date"]),
        ("ix_content_items_status", ["status"]),
    ):
        _ensure_index("content_items", name, columns)


def downgrade() -> None:
    op.drop_index("ix_content_items_status", table_name="content_items")
    op.drop_index("ix_content_items_date", table_name="content_items")
    op.drop_index("ix_content_items_campaign_id", table_name="content_items")
    op.drop_index("ix_content_items_business_id", table_name="content_items")
    op.drop_index("ix_content_items_business_date", table_name="content_items")
    op.drop_table("content_items")
