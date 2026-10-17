"""SQLAlchemy 2 models.

One schema for PostgreSQL (production) and SQLite (dev/test). Brand memory is
stored as short text rows and retrieved with deterministic lexical ranking --
see ``app/core/memory/store.py`` -- so there is no vector column, no pgvector
dependency and no embedding provider to keep in sync.

Every table that a user can inspect (tasks, tool calls, approvals, usage) is
queryable so "what did the AI do" is answerable from the DB.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import date as date_type
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

#: Native JSONB on PostgreSQL / Supabase, falling back to JSON on SQLite.
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class DateString(TypeDecorator):
    """Date column stored as DATE in Postgres/SQLite, exposed as YYYY-MM-DD str in Python."""

    impl = Date
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            import datetime
            try:
                return datetime.date.fromisoformat(value)
            except Exception:
                return value
        return value

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return str(value)

from app.schemas.common import (
    Autonomy,
    Language,
    RecommendationStatus,
    TaskStatus,
    Tone,
)

class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------- tables --


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Business(Base, TimestampMixin):
    __tablename__ = "businesses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # A workspace owns one or more businesses. Workspace scoping is what the
    # per-workspace token budget and persona overrides hang off.
    workspace_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    industry: Mapped[str | None] = mapped_column(String(120))
    website_url: Mapped[str | None] = mapped_column(String(500))
    country: Mapped[str | None] = mapped_column(String(2))
    language: Mapped[str] = mapped_column(String(2), default=Language.EN.value)
    currency: Mapped[str | None] = mapped_column(String(8))
    monthly_budget: Mapped[float | None] = mapped_column(Float)
    # P2.7 / P4.1: the per-workspace override for ``Settings.daily_token_budget``.
    # NULL means "inherit the process default", which is why this is a nullable
    # override and not a second mandatory number: two workspaces can hold
    # different budgets while a fresh workspace still boots with the env value.
    daily_token_budget: Mapped[int | None] = mapped_column(Integer)

    profile: Mapped["BrandProfile"] = relationship(
        back_populates="business", uselist=False, cascade="all, delete-orphan",
        lazy="selectin",
    )
    memory_items: Mapped[list["BrandMemoryItem"]] = relationship(
        back_populates="business", cascade="all, delete-orphan", lazy="selectin"
    )
    campaigns: Mapped[list["Campaign"]] = relationship(
        back_populates="business", cascade="all, delete-orphan", lazy="selectin"
    )


class BrandProfile(Base, TimestampMixin):
    """Stable brand fields. Always part of BrandContext (the compact part)."""

    __tablename__ = "brand_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), unique=True, index=True
    )
    brand_voice: Mapped[str | None] = mapped_column(Text)
    visual_style: Mapped[str | None] = mapped_column(Text)
    target_audience: Mapped[str | None] = mapped_column(Text)
    key_offers: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)
    goals: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)
    competitors: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)

    business: Mapped[Business] = relationship(back_populates="profile")


class SavedAudience(Base, TimestampMixin):
    """Reusable targeting profile scoped to one business/workspace."""

    __tablename__ = "saved_audiences"
    __table_args__ = (
        Index("ix_saved_audiences_business_updated", "business_id", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    targeting: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict, nullable=False)
    source: Mapped[str] = mapped_column(String(20), default="manual", nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)


class BrandMemoryItem(Base):
    """One unit of brand memory.

    Retrieval is deterministic lexical ranking (see ``app/core/memory``): there
    is no embedding column and no vector index. The lifecycle columns below are
    what replaced the reindex workflow:

    * ``dedupe_digest``  -- digest of (kind, workspace, campaign, normalised text);
      a repeat write updates the same row instead of adding a near-duplicate.
    * ``importance`` / ``pinned`` -- ranking inputs; the kind registry supplies
      the default importance.
    * ``expires_at`` -- materialised TTL for episodic kinds, so expiry is an
      indexed predicate instead of Python arithmetic over the whole table.
    * ``archived_at`` -- soft delete: out of retrieval, still auditable.
    * ``use_count`` / ``last_used_at`` -- proof an item was actually injected
      into a prompt; pruning spares rows that earn their place.
    """

    __tablename__ = "brand_memory_items"
    __table_args__ = (
        Index("ix_brand_memory_items_biz_kind", "business_id", "kind"),
        Index("ix_brand_memory_items_biz_active", "business_id", "archived_at"),
        Index("ix_brand_memory_items_biz_created", "business_id", "created_at"),
        Index("ix_brand_memory_items_dedupe", "business_id", "dedupe_digest"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(
        String(40),
        index=True,
        comment=(
            "See app/core/memory/kinds.py: brand_identity|voice_example|"
            "visual_style|offer|audience|strategy_blueprint|competitor_note|"
            "market_research|market_gaps|website_analysis|reference_analysis|"
            "chat_attachment|learning"
        ),
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Free-form: campaign ids, scores, tags, reference analysis payloads.
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    campaign_id: Mapped[str | None] = mapped_column(String(64), index=True)

    # -- lifecycle / ranking --------------------------------------------------
    dedupe_digest: Mapped[str | None] = mapped_column(String(64))
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    use_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    business: Mapped[Business] = relationship(back_populates="memory_items")


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_biz_updated", "business_id", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300), default="")
    # Sticky context: campaign_id / creative_id / insight_id.
    context: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    summary: Mapped[str | None] = mapped_column(Text)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at, desc(Message.role)",
        lazy="selectin",
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conv_created", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, default=list)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True)
    persona_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Campaign(Base, TimestampMixin):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="draft", index=True)
    objective: Mapped[str | None] = mapped_column(String(200))
    channel: Mapped[str | None] = mapped_column(String(60))
    budget: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(8))
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    target_audience: Mapped[str | None] = mapped_column(Text)
    audience_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("saved_audiences.id", ondelete="SET NULL"), index=True
    )
    # Immutable-at-campaign-creation copy; later edits to the reusable audience
    # must not silently rewrite the targeting shown on historical campaigns.
    audience_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    meta_campaign_id: Mapped[str | None] = mapped_column(String(120), index=True)
    notes: Mapped[str | None] = mapped_column(Text)

    business: Mapped[Business] = relationship(back_populates="campaigns")
    creatives: Mapped[list["Creative"]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan", lazy="selectin"
    )
    insights: Mapped[list["Insight"]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan", lazy="selectin"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan", lazy="selectin"
    )


class ContentItem(Base, TimestampMixin):
    """One editable, workspace-owned content-calendar entry."""

    __tablename__ = "content_items"
    __table_args__ = (
        Index("ix_content_items_business_date", "business_id", "date"),
        CheckConstraint(
            "channel IN ('instagram','facebook','tiktok','email','blog')",
            name="ck_content_items_channel",
        ),
        CheckConstraint(
            "format IN ('post','reel','story','ad','email','article')",
            name="ck_content_items_format",
        ),
        CheckConstraint(
            "status IN ('idea','draft','scheduled','published')",
            name="ck_content_items_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    campaign_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("campaigns.id", ondelete="SET NULL"), index=True
    )
    date: Mapped[date_type] = mapped_column(Date, nullable=False, index=True)
    time: Mapped[str | None] = mapped_column(String(5))
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    format: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    hook: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    hashtags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="idea", nullable=False, index=True)
    pillar: Mapped[str | None] = mapped_column(String(200))


class Creative(Base, TimestampMixin):
    """A generated or uploaded creative asset."""

    __tablename__ = "creatives"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(String(64), index=True)
    campaign_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("campaigns.id", ondelete="SET NULL"), index=True
    )
    persona_id: Mapped[str | None] = mapped_column(String(64))
    task_name: Mapped[str | None] = mapped_column(String(60))
    kind: Mapped[str] = mapped_column(String(30), default="image")
    # Text, not String(1000): this holds an inline data URL. Generated and
    # uploaded creatives both store their bytes here, and Meta is sent those
    # bytes, so the value runs to megabytes. SQLite ignores VARCHAR length,
    # which hid the overflow; Postgres would have rejected every creative.
    url: Mapped[str] = mapped_column(Text, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, default="")
    aspect_ratio: Mapped[str] = mapped_column(String(10), default="1:1")
    provider: Mapped[str | None] = mapped_column(String(40))
    model: Mapped[str | None] = mapped_column(String(80))
    style_spec: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    reference_analysis_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="ok")

    campaign: Mapped[Campaign | None] = relationship(back_populates="creatives")


class Insight(Base):
    __tablename__ = "insights"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("campaigns.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(60))
    title: Mapped[str] = mapped_column(String(400))
    detail: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(20), default="info", index=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    persona_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    campaign: Mapped[Campaign] = relationship(back_populates="insights")
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="insight"
    )


class Recommendation(Base, TimestampMixin):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    campaign_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("campaigns.id", ondelete="CASCADE"), index=True
    )
    # P1: the owning business, denormalised. Before this column the tenant had to
    # be reached through ``campaign_id`` -> ``Campaign.business_id``, which is
    # ambiguous for a recommendation raised outside a campaign (no owner at all)
    # and costs a join on every read. Nullable with ``ondelete="SET NULL"``: a
    # NULL row belongs to nobody and stays invisible, so a deleted business fails
    # closed rather than orphaning rows into someone's list.
    workspace_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="SET NULL"), index=True
    )
    insight_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("insights.id", ondelete="SET NULL"), index=True
    )
    action_type: Mapped[str] = mapped_column(String(60), index=True)
    title: Mapped[str] = mapped_column(String(400))
    rationale: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, default=list)
    status: Mapped[str] = mapped_column(
        String(20), default=RecommendationStatus.PENDING.value, index=True
    )
    persona_id: Mapped[str | None] = mapped_column(String(64))
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=True)
    estimated_impact: Mapped[str | None] = mapped_column(String(200))
    # before -> after captured at approval time, shown in the review dialog.
    diff: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    approved_by: Mapped[str | None] = mapped_column(String(120))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    campaign: Mapped[Campaign | None] = relationship(back_populates="recommendations")
    insight: Mapped[Insight | None] = relationship(back_populates="recommendations")


class AuditLog(Base):
    """Every approval and execution. Append-only by convention."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    actor: Mapped[str] = mapped_column(String(120), default="system")
    action: Mapped[str] = mapped_column(String(80), index=True)
    entity_type: Mapped[str] = mapped_column(String(60), default="recommendation")
    entity_id: Mapped[str | None] = mapped_column(String(64), index=True)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    business_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


class PersonaOverride(Base, TimestampMixin):
    """Per-workspace persona edits. Only editable_fields are ever stored."""

    __tablename__ = "persona_overrides"
    __table_args__ = (UniqueConstraint("workspace_id", "persona_id", name="uq_persona_override"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    persona_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(100))
    tone: Mapped[str | None] = mapped_column(String(20))
    autonomy: Mapped[str | None] = mapped_column(String(20))
    language: Mapped[str | None] = mapped_column(String(2))


class AgentTask(Base, TimestampMixin):
    """A long-running agent run: queued|running|succeeded|failed|cancelled."""

    __tablename__ = "agent_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str | None] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    task_name: Mapped[str] = mapped_column(String(60), index=True)
    persona_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default=TaskStatus.QUEUED.value, index=True)
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, default=list)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    incomplete: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, default=list)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(80))
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentToolCall(Base):
    """Every tool call the agents make. Powers the "what the AI did" UI."""

    __tablename__ = "agent_tool_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str | None] = mapped_column(String(64), index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), index=True)
    persona_id: Mapped[str | None] = mapped_column(String(64), index=True)
    tool: Mapped[str] = mapped_column(String(80), index=True)
    input: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    output_summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="ok")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


class Integration(Base, TimestampMixin):
    """Third-party connections. Tokens are encrypted at rest."""

    __tablename__ = "integrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(40), index=True)
    access_token_enc: Mapped[str | None] = mapped_column(Text)
    refresh_token_enc: Mapped[str | None] = mapped_column(Text)
    account_id: Mapped[str | None] = mapped_column(String(120))
    account_name: Mapped[str | None] = mapped_column(String(200))
    scopes: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="connected")
    __table_args__ = (
        UniqueConstraint("business_id", "provider", name="uq_integration_provider"),
    )


# ------------------------------------------------------------ identity (P1) ---


class User(Base, TimestampMixin):
    """A person who can sign in. The identity root for every other P1 table.

    ``email`` is stored already normalised (see ``app/core/auth.py``:
    ``normalise_email``) and carries a plain UNIQUE constraint. That is the
    citext behaviour without a Postgres extension, and it is what makes
    ``Foo@Example.com`` and ``foo@example.com`` one account rather than two --
    which is also why login can never disclose whether an address is registered.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    #: ``pbkdf2_sha256$<iterations>$<b64 salt>$<b64 hash>``. Never logged.
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Membership(Base):
    """The user <-> workspace join that makes multi-tenancy real (P1.7).

    One row per (user, workspace). ``business_id`` records which business inside
    that workspace the membership was created against; both unique constraints
    are deliberate -- a user cannot end up with two memberships for the same
    workspace, and therefore cannot hold two roles in it.
    """

    __tablename__ = "memberships"
    __table_args__ = (
        UniqueConstraint("user_id", "business_id", name="uq_membership_user_business"),
        UniqueConstraint("user_id", "workspace_id", name="uq_membership_user_workspace"),
        CheckConstraint(
            "role IN ('owner','member','viewer')", name="ck_membership_role"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    #: An id namespace, not yet a foreign key -- see the roadmap's note on why.
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    #: ``owner | member | viewer`` (enforced by ``ck_membership_role``).
    role: Mapped[str] = mapped_column(String(20), default="member", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Session(Base):
    """One opaque bearer token. The raw token is returned once and never stored.

    Only ``sha256(token)`` lands here, so a database leak -- a stolen backup, a
    SQL console -- yields hashes that cannot be replayed as sessions. Revocation
    is instant because every request reads this row; rotation (``/auth/refresh``)
    is therefore just "insert a new row, stamp ``revoked_at`` on the old one".
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    #: sha256 hex of the raw token. Unique, so a collision is a 500 not a
    #: silent account takeover.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_agent: Mapped[str | None] = mapped_column(String(400))
    ip: Mapped[str | None] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )


class Profile(Base, TimestampMixin):
    """Links a Supabase Auth user to a business + role.

    Replaces the old User + Membership tables for authentication.
    Supabase owns the identity (email, password, OAuth); we own the
    business/role mapping via this table.
    """

    __tablename__ = "profiles"
    __table_args__ = (
        UniqueConstraint("supabase_user_id", name="uq_profiles_supabase_user_id"),
        ForeignKeyConstraint(
            ["business_id"], ["businesses.id"], ondelete="CASCADE", name="fk_profiles_business_id"
        ),
        CheckConstraint("role IN ('owner','member','viewer')", name="ck_profiles_role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    supabase_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), unique=True, index=True, nullable=False
    )
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), default="member", index=True, nullable=False)
    user_context_summary: Mapped[str | None] = mapped_column(Text)


class ToolSettings(Base, TimestampMixin):
    """P2.4: per-workspace connector configuration, one row per (business, connector).

    ``integrations`` stores OAuth *grants*; this table stores *configuration* --
    the provider, the non-secret ids and the encrypted secrets a workspace's
    tools need. The split matters: disconnecting Meta must not delete the app id
    an agency typed in, and rotating a search key must not touch the Meta grant.

    One row per connector (not one wide row per workspace) is deliberate even
    though the roadmap sketch in §5.1 shows a wide ``tool_settings`` row: it
    makes ``DELETE /settings/connectors/{connector}`` a single-row delete, keeps
    an added provider from needing a nullable column pair, and lets the unique
    constraint below do the isolation work in the database instead of in a
    ``WHERE`` clause somebody can forget.

    Every ``*_enc`` column is Fernet ciphertext produced by ``TokenCipher``
    (``app/core/security.py``). A secret is write-only over HTTP: no response
    body, log line or audit row ever contains the plaintext.
    """

    __tablename__ = "tool_settings"
    __table_args__ = (
        UniqueConstraint(
            "business_id", "connector", name="uq_tool_settings_business_connector"
        ),
        CheckConstraint(
            "connector IN ('meta','google_places','dataforseo','image','search')",
            name="ck_tool_settings_connector",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    connector: Mapped[str] = mapped_column(String(40), index=True)
    # --- non-secret configuration -------------------------------------------
    #: Meta: the app and the ad account this workspace manages.
    meta_app_id: Mapped[str | None] = mapped_column(String(120))
    meta_ad_account_id: Mapped[str | None] = mapped_column(String(120))
    meta_page_id: Mapped[str | None] = mapped_column(String(120))
    #: Google Places / DataForSEO.
    google_places_base_url: Mapped[str | None] = mapped_column(String(500))
    dataforseo_login: Mapped[str | None] = mapped_column(String(200))
    #: Provider selection for the two swappable connectors.
    image_provider: Mapped[str | None] = mapped_column(String(30))
    image_model: Mapped[str | None] = mapped_column(String(80))
    search_provider: Mapped[str | None] = mapped_column(String(30))
    keyword_provider: Mapped[str | None] = mapped_column(String(30))
    #: Repointing column: the connector's base URL, when the map says the
    #: connector is ``configurable_via`` a base URL.
    base_url: Mapped[str | None] = mapped_column(String(500))
    # --- encrypted secrets ---------------------------------------------------
    meta_app_secret_enc: Mapped[str | None] = mapped_column(Text)
    meta_access_token_enc: Mapped[str | None] = mapped_column(Text)
    google_places_api_key_enc: Mapped[str | None] = mapped_column(Text)
    dataforseo_password_enc: Mapped[str | None] = mapped_column(Text)
    image_api_key_enc: Mapped[str | None] = mapped_column(Text)
    search_api_key_enc: Mapped[str | None] = mapped_column(Text)
    #: Who last changed it, for the audit trail. Never a secret.
    updated_by: Mapped[str | None] = mapped_column(String(64))


class LlmProviderSettings(Base, TimestampMixin):
    """Workspace-level LLM routing and credentials; secrets are encrypted."""

    __tablename__ = "llm_provider_settings"
    __table_args__ = (
        UniqueConstraint("business_id", name="uq_llm_provider_settings_business"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    fast_provider: Mapped[str | None] = mapped_column(String(20))
    fast_model: Mapped[str | None] = mapped_column(String(120))
    fast_base_url: Mapped[str | None] = mapped_column(String(500))
    fast_api_key_enc: Mapped[str | None] = mapped_column(Text)
    fast_extra_headers_enc: Mapped[str | None] = mapped_column(Text)
    strong_provider: Mapped[str | None] = mapped_column(String(20))
    strong_model: Mapped[str | None] = mapped_column(String(120))
    strong_base_url: Mapped[str | None] = mapped_column(String(500))
    strong_api_key_enc: Mapped[str | None] = mapped_column(Text)
    strong_extra_headers_enc: Mapped[str | None] = mapped_column(Text)
    updated_by: Mapped[str | None] = mapped_column(String(64))


class UsageRecord(Base):
    """Per-call token accounting, used for the daily budget guard and billing.

    ``action`` is the billable *act* (``chat``, ``research.website``,
    ``creative.text`` ...). It is what lets the usage screen answer "which part
    of the product is spending my tokens?", which a bare per-model total cannot.
    ``estimated`` marks a row whose token counts were derived from text length
    because the provider reported no usage -- a billing report must be able to
    tell a measured number from a guess.
    """

    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(String(64), index=True)
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True)
    persona_id: Mapped[str | None] = mapped_column(String(64))
    tier: Mapped[str | None] = mapped_column(String(10))
    action: Mapped[str | None] = mapped_column(String(64), index=True)
    model: Mapped[str | None] = mapped_column(String(80))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    estimated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )





__all__ = [
    "AgentTask",
    "AgentToolCall",
    "AuditLog",
    "Base",
    "BrandMemoryItem",
    "BrandProfile",
    "Business",
    "Campaign",
    "ContentItem",
    "Conversation",
    "Creative",
    "Integration",
    "Insight",
    "Message",
    "PersonaOverride",
    "ToolSettings",
    "Recommendation",
    "UsageRecord",
    "Profile",
    "Tone",
    "Autonomy",
    "Language",
    # Legacy (kept for migration, will be removed):
    "User",
    "Membership",
    "Session",
]
