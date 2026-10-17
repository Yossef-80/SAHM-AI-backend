"""brand memory: drop the vector column, add the lifecycle columns

Revision ID: b3f1c2d4e5a6
Revises: d4a91f7c2e60
Create Date: 2026-10-17 13:05:00.000000

What this does
--------------
Brand memory stopped being a similarity-search table and became a
lifecycle-managed one (``docs/MEMORY_ARCHITECTURE.md``). Concretely:

* ``brand_memory_items.embedding`` is **dropped**. Nothing embeds text any more;
  retrieval is BM25 + recency + kind affinity over the same rows.
* ``dedupe_digest``, ``importance``, ``pinned``, ``source``, ``use_count``,
  ``last_used_at``, ``expires_at``, ``archived_at``, ``updated_at`` are added.
  They replace the reindex/prune-by-hand workflow: expiry is an indexed
  predicate, dedupe is a digest, and pruning has a ranking to sort by.
* Four composite indexes replace per-column single indexes that could not serve
  the two predicates every query carries (tenant + active + kind).

Why it is defensive
-------------------
``cfe093781e86`` explicitly documents that it may already have been applied, and
that its older text created the ``embedding`` column. So this revision inspects
the live schema and only performs the parts that are missing. On a fresh
database (``create_all()`` or a clean ``alembic upgrade head``) every operation
is a no-op; on a database provisioned from the old initial revision it repairs
the table in place. Re-running it is harmless.

Downgrade restores the columns but **cannot restore vectors**: they were never
semantically meaningful (a hashed bag of words), and re-creating the column as
JSON keeps the schema shape without pretending otherwise.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3f1c2d4e5a6"
down_revision: Union[str, None] = "d4a91f7c2e60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "brand_memory_items"

def _add_columns(dialect: str) -> tuple[sa.Column, ...]:
    """Columns added by this revision, with dialect-correct server defaults.

    ``ALTER TABLE ... ADD COLUMN x BOOLEAN NOT NULL`` requires a default on both
    SQLite and Postgres, and ``false`` is not a valid literal on SQLite, so the
    default is chosen per dialect. The defaults are dropped afterwards by the
    model layer? No: they stay, which is what makes a backfill unnecessary for
    rows inserted by an older application instance during a rolling deploy.
    """
    boolean_false = sa.text("false") if dialect == "postgresql" else sa.text("0")
    return (
        sa.Column("dedupe_digest", sa.String(length=64), nullable=True),
        sa.Column("importance", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=boolean_false),
        sa.Column("source", sa.String(length=40), nullable=False, server_default=sa.text("''")),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


#: Names added by :func:`_add_columns`, in order (used by ``downgrade``).
_ADDED_COLUMN_NAMES: tuple[str, ...] = (
    "dedupe_digest",
    "importance",
    "pinned",
    "source",
    "use_count",
    "last_used_at",
    "expires_at",
    "archived_at",
    "updated_at",
)

_INDEXES: tuple[tuple[str, list[str]], ...] = (
    ("ix_brand_memory_items_biz_kind", ["business_id", "kind"]),
    ("ix_brand_memory_items_biz_active", ["business_id", "archived_at"]),
    ("ix_brand_memory_items_biz_created", ["business_id", "created_at"]),
    ("ix_brand_memory_items_dedupe", ["business_id", "dedupe_digest"]),
)


def _columns(bind: sa.engine.Connection, table: str) -> dict[str, dict]:
    return {col["name"]: col for col in sa.inspect(bind).get_columns(table)}


def _index_names(bind: sa.engine.Connection, table: str) -> set[str]:
    return {idx["name"] for idx in sa.inspect(bind).get_indexes(table)}


def _table_exists(bind: sa.engine.Connection, table: str) -> bool:
    return table in sa.inspect(bind).get_table_names()


def upgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, TABLE):
        # A database that has no brand_memory_items at all is built by the
        # model metadata; nothing to repair.
        return

    existing = _columns(bind, TABLE)

    # ---- 1. drop the vector column -----------------------------------------
    if "embedding" in existing:
        with op.batch_alter_table(TABLE) as batch:
            batch.drop_column("embedding")

    # ---- 2. add lifecycle columns ------------------------------------------
    for column in _add_columns(bind.dialect.name):
        if column.name in existing:
            continue
        with op.batch_alter_table(TABLE) as batch:
            batch.add_column(column)

    # ---- 3. backfill --------------------------------------------------------
    # Every pre-existing row predates the lifecycle, so it gets a stable digest
    # and an importance derived from its kind by the running code. Backfilling
    # ``updated_at`` from ``created_at`` keeps "never edited" truthful.
    if "updated_at" in _columns(bind, TABLE):
        op.execute(
            sa.text(
                f"UPDATE {TABLE} SET updated_at = created_at WHERE updated_at IS NULL"
            )
        )
    if "source" in _columns(bind, TABLE):
        op.execute(sa.text(f"UPDATE {TABLE} SET source = 'legacy' WHERE source = ''"))
    if "importance" in _columns(bind, TABLE):
        op.execute(sa.text(f"UPDATE {TABLE} SET importance = 0.5 WHERE importance IS NULL"))

    # ---- 4. indexes ---------------------------------------------------------
    present = _index_names(bind, TABLE)
    for name, columns in _INDEXES:
        if name in present:
            continue
        op.create_index(name, TABLE, columns, unique=False)

    if "expires_at" in _columns(bind, TABLE) and (
        "ix_brand_memory_items_expires_at" not in _index_names(bind, TABLE)
    ):
        op.create_index(
            "ix_brand_memory_items_expires_at", TABLE, ["expires_at"], unique=False
        )


def downgrade() -> None:
    """Restore the old shape. Vectors are *not* recoverable -- see module docstring."""
    bind = op.get_bind()
    if not _table_exists(bind, TABLE):
        return

    present = _index_names(bind, TABLE)
    for name, _columns_ in _INDEXES:
        if name in present:
            op.drop_index(name, table_name=TABLE)
    if "ix_brand_memory_items_expires_at" in present:
        op.drop_index("ix_brand_memory_items_expires_at", table_name=TABLE)

    existing = _columns(bind, TABLE)
    for name in reversed(_ADDED_COLUMN_NAMES):
        if name in existing:
            with op.batch_alter_table(TABLE) as batch:
                batch.drop_column(name)

    if "embedding" not in existing:
        with op.batch_alter_table(TABLE) as batch:
            batch.add_column(sa.Column("embedding", sa.JSON(), nullable=True))
