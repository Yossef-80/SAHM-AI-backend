"""SQLAlchemy 2 models.

PostgreSQL + pgvector is the production target. Every model here also works on
SQLite (dev/test) via the ``Embedding`` type decorator, which stores vectors as
JSON and does cosine similarity in Python when pgvector is not available.

Every table that a user can inspect (tasks, tool calls, approvals, usage) is
queryable so "what did the AI do" is answerable from the DB.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.schemas.common import (
    Autonomy,
    Language,
    RecommendationStatus,
    TaskStatus,
    Tone,
)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------- embeddings --


class Embedding(TypeDecorator):
    """Vector column: pgvector on Postgres, JSON elsewhere.

    Keeps one model definition working against both backends. On Postgres the
    column is a real ``vector`` column so similarity search happens in the
    database; on SQLite the vector is stored as JSON and similarity is computed
    in Python (see ``app/core/memory/store.py``).
    """

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector

                return dialect.type_descriptor(Vector(self.dim))
            except Exception:  # pragma: no cover - pgvector not installed
                pass
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return list(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return list(value)

    class comparator_factory(TypeDecorator.Comparator):
        """Expose pgvector's distance operators through the decorator.

        Without this, ``BrandMemoryItem.embedding.cosine_distance(...)`` resolves
        against the JSON impl and raises AttributeError, silently forcing every
        search onto the slow Python fallback.

        The operators are emitted directly rather than delegating to
        ``self.expr.cosine_distance``: ``self.expr`` has our own type, so
        delegating recurses forever. ``<=>`` is pgvector's cosine distance,
        ``<->`` L2, ``<#>`` negative inner product.
        """

        def cosine_distance(self, other: Any) -> Any:
            return self.expr.op("<=>")(other)

        def l2_distance(self, other: Any) -> Any:
            return self.expr.op("<->")(other)

        def max_inner_product(self, other: Any) -> Any:
            return self.expr.op("<#>")(other)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Used on non-pgvector backends."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


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
    key_offers: Mapped[list[str]] = mapped_column(JSON, default=list)
    goals: Mapped[list[str]] = mapped_column(JSON, default=list)
    competitors: Mapped[list[str]] = mapped_column(JSON, default=list)

    business: Mapped[Business] = relationship(back_populates="profile")


class BrandMemoryItem(Base):
    """A single memory unit, vector-embedded for similarity retrieval."""

    __tablename__ = "brand_memory_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(
        String(40),
        index=True,
        comment="voice_example|visual_style|offer|audience|competitor_note|learning|reference_analysis",
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Embedding())
    # Free-form: campaign ids, scores, tags, reference analysis payloads.
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    campaign_id: Mapped[str | None] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    business: Mapped[Business] = relationship(back_populates="memory_items")


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("businesses.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(300), default="")
    # Sticky context: campaign_id / creative_id / insight_id.
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
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
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
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
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
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
    insight_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("insights.id", ondelete="SET NULL"), index=True
    )
    action_type: Mapped[str] = mapped_column(String(60), index=True)
    title: Mapped[str] = mapped_column(String(400))
    rationale: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
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
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    incomplete: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(80))
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
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
    input: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
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
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="connected")
    __table_args__ = (
        UniqueConstraint("business_id", "provider", name="uq_integration_provider"),
    )


class UsageRecord(Base):
    """Per-call token accounting, used for the daily budget guard."""

    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[str] = mapped_column(String(64), index=True)
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True)
    persona_id: Mapped[str | None] = mapped_column(String(64))
    tier: Mapped[str | None] = mapped_column(String(10))
    model: Mapped[str | None] = mapped_column(String(80))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
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
    "Conversation",
    "Creative",
    "Embedding",
    "Integration",
    "Insight",
    "Message",
    "PersonaOverride",
    "Recommendation",
    "UsageRecord",
    "Tone",
    "Autonomy",
    "Language",
    "cosine",
]
