"""Reusable, business-scoped audience profiles and AI audience ideas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import ApiModel
from app.schemas.targeting import AudienceTargetingTarget, TargetingSearchTerm


class AudienceInterest(ApiModel):
    """A Meta interest selected from the official Targeting Search response."""

    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=200)


class AudienceTargeting(ApiModel):
    age_min: int = Field(default=18, ge=13, le=65)
    age_max: int = Field(default=65, ge=13, le=65)
    genders: list[Literal["all", "male", "female"]] = Field(default_factory=lambda: ["all"], max_length=3)
    locations: list[str] = Field(default_factory=list, max_length=30)
    countries: list[str] = Field(default_factory=list, max_length=12)
    interests: list[str] = Field(default_factory=list, max_length=20)
    interest_targets: list[AudienceInterest] = Field(default_factory=list, max_length=20)
    #: New categorized selections. Old `interest_targets` remains readable for saved audiences.
    targeting_targets: list[AudienceTargetingTarget] = Field(default_factory=list, max_length=20)
    behaviors: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_age_range(self) -> "AudienceTargeting":
        if self.age_min > self.age_max:
            raise ValueError("age_min must be less than or equal to age_max")
        if "all" in self.genders and len(self.genders) > 1:
            raise ValueError("'all' cannot be combined with a specific gender")
        return self


class SavedAudienceCreateRequest(ApiModel):
    name: str = Field(min_length=2, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    targeting: AudienceTargeting
    source: Literal["manual", "ai"] = "manual"


class SavedAudienceUpdateRequest(ApiModel):
    name: str | None = Field(default=None, min_length=2, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    targeting: AudienceTargeting | None = None
    source: Literal["manual", "ai"] | None = None


class SavedAudienceResponse(SavedAudienceCreateRequest):
    id: str
    business_id: str
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime


class SavedAudienceListResponse(ApiModel):
    audiences: list[SavedAudienceResponse]
    total: int


class AudienceSuggestionRequest(ApiModel):
    campaign_name: str = Field(min_length=2, max_length=200)
    objective: Literal["awareness", "traffic", "engagement", "leads", "sales", "app_installs"]
    brief: str | None = Field(default=None, max_length=4000)
    countries: list[str] = Field(default_factory=list, max_length=12)
    language: Literal["ar", "en"] = "ar"


class AudienceIdea(ApiModel):
    name: str = Field(min_length=2, max_length=200)
    summary: str = Field(min_length=2, max_length=600)
    approach: Literal["broad", "targeting_hypothesis", "interest_hypothesis"] = "targeting_hypothesis"
    #: Legacy interest-only suggestions remain for backwards-compatible responses.
    interest_search_terms: list[str] = Field(default_factory=list, max_length=5)
    targeting_search_terms: list[TargetingSearchTerm] = Field(default_factory=list, max_length=12)
    rationale: str = Field(min_length=2, max_length=600)


class AudienceSuggestionResponse(ApiModel):
    suggestions: list[AudienceIdea] = Field(default_factory=list, max_length=3)
    missing_inputs: list[str] = Field(default_factory=list, max_length=8)
