"""Evidence-bearing response schema for Nour's research reports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchEvidence(BaseModel):
    claim: str = ""
    source_url: str = ""
    observed_text: str = ""
    source_type: str = "web"


class CompetitorSummary(BaseModel):
    brand: str = ""
    positioning: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses_or_gaps: list[str] = Field(default_factory=list)
    evidence: list[ResearchEvidence] = Field(default_factory=list)


class MarketResearchReport(BaseModel):
    """Final research output; separate observed evidence from interpretation."""

    market_overview: str = ""
    competitor_summary: list[CompetitorSummary] = Field(default_factory=list)
    pricing_insights: list[str] = Field(default_factory=list)
    offer_insights: list[str] = Field(default_factory=list)
    shipping_insights: list[str] = Field(default_factory=list)
    search_insights: list[str] = Field(default_factory=list)
    social_insights: list[str] = Field(default_factory=list)
    opportunities_for_target_brand: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[ResearchEvidence] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
