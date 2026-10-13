"""Schemas for POST /business and GET /business/profile."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import AliasChoices, Field

from app.schemas.common import ApiModel

#: P2.1 caps. Every length is a product decision, not a technical one: these are
#: the fields an agent embeds verbatim into a prompt
#: (``app/core/memory/store.py::build_context``), so an unbounded text field is
#: both a prompt-injection surface and a way to write a 10 MB JSON blob through
#: an HTML form. 4 000 characters is roughly a page of prose per field, which is
#: more than any brand voice document needs.
MAX_BRAND_TEXT = 4_000
#: A tag input (competitors, offers, goals) has no business sending 200 entries.
MAX_LIST_ITEMS = 50
MAX_LIST_ITEM = 200


class BusinessCreateRequest(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    industry: str | None = None
    website_url: str | None = None
    country: str | None = Field(default=None, max_length=2)
    language: str | None = Field(default=None, description="en | ar")
    currency: str | None = Field(default=None, max_length=8)
    monthly_budget: float | None = None
    goals: list[str] = Field(default_factory=list)


class BrandProfile(ApiModel):
    """Stable, always-in-context brand fields (the compact part of memory)."""

    business_id: str
    name: str
    industry: str | None = None
    website_url: str | None = None
    country: str | None = None
    language: str = "en"
    currency: str | None = None
    monthly_budget: float | None = None
    #: P2.7/P4.1. ``None`` means "inherit the process default"; the same row is
    #: editable through ``PATCH /business/profile`` and readable through
    #: ``GET /usage``, so the two can never disagree.
    daily_token_budget: int | None = None
    goals: list[str] = Field(default_factory=list)
    brand_voice: str | None = None
    visual_style: str | None = None
    target_audience: str | None = None
    key_offers: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class BusinessProfileResponse(ApiModel):
    business: BrandProfile
    memory_item_count: int = 0
    learning_count: int = 0


class BusinessProfileUpdateRequest(ApiModel):
    """P2.1: partial update for every writable profile field.

    Every field is optional and semantics are **exclude-unset**: a field the
    caller omitted is left untouched, and a field sent as ``null`` is stored as
    ``NULL`` (which is how a user clears "competitors"). Sending an omitted
    field and sending ``null`` are genuinely different instructions, so
    ``model_fields_set`` is the whole mechanism -- a plain default of ``None``
    would make them indistinguishable and silently wipe data.

    Sending ``[]`` or ``null`` for a list clears it; omitting the key leaves it
    alone. A list column is never NULL in the database, so ``null`` and ``[]``
    are the same instruction here -- unlike the scalar fields, where ``null`` is
    stored as NULL. That difference is deliberate and is the reason the list
    fields are nullable in the schema rather than defaulting to ``[]``: a
    defaulted list cannot distinguish "not sent" from "sent empty".

    Requires ``owner`` or ``member``: a brand profile is a write, and the role
    matrix in the roadmap puts viewers on read-only for all of them.
    """

    # -- business row --
    name: str | None = Field(default=None, min_length=1, max_length=200)
    industry: str | None = Field(default=None, max_length=120)
    website_url: str | None = Field(default=None, max_length=500)
    country: str | None = Field(default=None, max_length=2)
    language: str | None = Field(default=None, max_length=2)
    currency: str | None = Field(default=None, max_length=8)
    monthly_budget: float | None = Field(default=None, ge=0)
    #: P2.7/P4.1: this workspace's own daily token budget. ``null`` restores the
    #: process-wide ``DAILY_TOKEN_BUDGET``.
    daily_token_budget: int | None = Field(default=None, ge=0)
    # -- brand profile row --
    brand_voice: str | None = Field(default=None, max_length=MAX_BRAND_TEXT)
    visual_style: str | None = Field(default=None, max_length=MAX_BRAND_TEXT)
    target_audience: str | None = Field(default=None, max_length=MAX_BRAND_TEXT)
    key_offers: list[Annotated[str, Field(max_length=MAX_LIST_ITEM)]] | None = Field(
        default=None, max_length=MAX_LIST_ITEMS
    )
    goals: list[Annotated[str, Field(max_length=MAX_LIST_ITEM)]] | None = Field(
        default=None, max_length=MAX_LIST_ITEMS
    )
    competitors: list[Annotated[str, Field(max_length=MAX_LIST_ITEM)]] | None = Field(
        default=None, max_length=MAX_LIST_ITEMS
    )

    def provided(self) -> set[str]:
        """Field names the caller actually sent (exclude-unset)."""
        return set(self.model_fields_set)


class ColorPalette(ApiModel):
    primary: str = Field(default="#2563EB", description="Primary brand color hex")
    secondary: str = Field(default="#1E293B", description="Secondary brand color hex")
    accent: str = Field(default="#F59E0B", description="Accent highlight color hex")
    background: str = Field(default="#F8FAFC", description="Base background hex")
    text: str = Field(default="#0F172A", description="Base text color hex")
    palette: list[str] = Field(default_factory=list, description="Curated 5-color palette")


class SWOTAnalysis(ApiModel):
    strengths: list[str] = Field(default_factory=list, description="Internal advantages and differentiators")
    weaknesses: list[str] = Field(default_factory=list, description="Internal limitations or areas to improve")
    opportunities: list[str] = Field(default_factory=list, description="Market opportunities, gaps to seize")
    threats: list[str] = Field(default_factory=list, description="Competitive risks, market barriers")


class IdentityEssence(ApiModel):
    tagline: str | None = None
    mission: str | None = None
    vision: str | None = None
    brand_voice: str | None = None
    tone_keywords: list[str] = Field(default_factory=list)
    archetype: str | None = None


class TargetAudienceProfile(ApiModel):
    summary: str | None = None
    demographics: str | None = None
    psychographics: str | None = None
    pain_points: list[str] = Field(default_factory=list)
    core_segments: list[str] = Field(default_factory=list)


class BrandIdentityResponse(ApiModel):
    business_id: str
    name: str = Field(validation_alias=AliasChoices("name", "brand_name", "brandName"))
    industry: str | None = None
    website_url: str | None = None
    country: str | None = None
    language: str = "en"
    currency: str | None = "EGP"
    monthly_budget: float | None = None
    colors: ColorPalette = Field(default_factory=ColorPalette, validation_alias=AliasChoices("colors", "palette"))
    essence: IdentityEssence = Field(default_factory=IdentityEssence)
    audience: TargetAudienceProfile = Field(default_factory=TargetAudienceProfile, validation_alias=AliasChoices("audience", "target_audience", "targetAudience"))
    swot: SWOTAnalysis = Field(default_factory=SWOTAnalysis)
    key_offers: list[str] = Field(default_factory=list, validation_alias=AliasChoices("key_offers", "keyOffers"))
    competitors: list[str] = Field(default_factory=list)
    market_gaps: list[str] = Field(default_factory=list, validation_alias=AliasChoices("market_gaps", "marketGaps"))
    last_synced_at: datetime | None = Field(default=None, validation_alias=AliasChoices("last_synced_at", "lastSyncedAt", "synced_at", "syncedAt"))
    sources_count: int = Field(default=0, validation_alias=AliasChoices("sources_count", "sourcesCount"))


class BrandIdentityUpdateRequest(ApiModel):
    name: str | None = Field(default=None, validation_alias=AliasChoices("name", "brand_name", "brandName"))
    industry: str | None = None
    website_url: str | None = None
    currency: str | None = None
    monthly_budget: float | None = None
    colors: ColorPalette | None = Field(default=None, validation_alias=AliasChoices("colors", "palette"))
    essence: IdentityEssence | None = None
    audience: TargetAudienceProfile | None = Field(default=None, validation_alias=AliasChoices("audience", "target_audience", "targetAudience"))
    swot: SWOTAnalysis | None = None
    key_offers: list[str] | None = Field(default=None, validation_alias=AliasChoices("key_offers", "keyOffers"))
    competitors: list[str] | None = None
    market_gaps: list[str] | None = Field(default=None, validation_alias=AliasChoices("market_gaps", "marketGaps"))


