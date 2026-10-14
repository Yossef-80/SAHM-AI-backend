"""merge saved-audience and usage-metering revision branches

Revision ID: d4a91f7c2e60
Revises: c1f4a9d27e35, c3e81b7a4d26

Both parent revisions are retained as ordinary Alembic migrations. This merge
revision has no schema operations; it only creates one unambiguous head for
future ``alembic upgrade head`` runs.
"""
from typing import Sequence, Union

revision: str = "d4a91f7c2e60"
down_revision: Union[str, Sequence[str], None] = (
    "c1f4a9d27e35",
    "c3e81b7a4d26",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op: both parent migrations apply their own schema changes."""
    pass


def downgrade() -> None:
    """No-op: Alembic downgrades the selected parent branch separately."""
    pass
