"""sampling controls (per LLM tier) and image quality (image connector)

Revision ID: b3d51c7a9f02
Revises: d4a91f7c2e60
Create Date: 2026-10-14

The product could pick a provider and a model but not how much the model was
allowed to say, or how good an image was allowed to be. Both were compiled-in
constants: the assistant's temperature came from ``AI_LLM_TEMPERATURE`` and the
image quality was whatever the provider's default happened to be (``high`` on
GPT Image, which is the expensive one).

``llm_provider_settings``: ``{tier}_max_tokens``, ``{tier}_temperature``,
``{tier}_reasoning_effort``
    Per-tier, nullable overrides. NULL means "keep the process default", so an
    existing workspace's behaviour does not change when this migration lands.
    ``reasoning_effort`` is only meaningful for OpenAI reasoning models, where it
    is what makes tool calling possible on Chat Completions.

``tool_settings.image_quality``
    ``auto`` (default) / ``low`` / ``medium`` / ``high`` for the image connector.
    Stored on the connector row because it is provider configuration, exactly
    like the model and the base URL.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3d51c7a9f02"
down_revision: Union[str, None] = "d4a91f7c2e60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for tier in ("fast", "strong"):
        op.add_column(
            "llm_provider_settings",
            sa.Column(f"{tier}_max_tokens", sa.Integer(), nullable=True),
        )
        op.add_column(
            "llm_provider_settings",
            sa.Column(f"{tier}_temperature", sa.Float(), nullable=True),
        )
        op.add_column(
            "llm_provider_settings",
            sa.Column(f"{tier}_reasoning_effort", sa.String(length=16), nullable=True),
        )
    op.add_column(
        "tool_settings",
        sa.Column("image_quality", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tool_settings", "image_quality")
    for tier in ("fast", "strong"):
        op.drop_column("llm_provider_settings", f"{tier}_reasoning_effort")
        op.drop_column("llm_provider_settings", f"{tier}_temperature")
        op.drop_column("llm_provider_settings", f"{tier}_max_tokens")
