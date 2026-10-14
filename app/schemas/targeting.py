"""Shared Meta detailed-targeting categories and selected target values."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import ApiModel


TargetingCategory = Literal[
    "interests",
    "behaviors",
    "job_titles",
    "employers",
    "education_majors",
    "education_schools",
    "life_events",
    "industries",
    "income",
    "family_statuses",
    "devices",
    "operating_systems",
]


class TargetingSearchTerm(ApiModel):
    """A natural-language phrase suggested for Meta search, not a targeting value."""

    category: TargetingCategory = "interests"
    term: str = Field(min_length=2, max_length=120)


class AudienceTargetingTarget(ApiModel):
    """One user-selected, Meta-resolved targeting option with its target category."""

    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=200)
    category: TargetingCategory = "interests"
