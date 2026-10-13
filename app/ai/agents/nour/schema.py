"""Evidence-bearing response schema for Nour's research reports."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ResearchEvidence(BaseModel):
    claim: str = ""
    source_url: str = ""
    observed_text: str = ""
    source_type: str = "web"


class CompetitorSummary(BaseModel):
    brand: str = Field(default="", description="The name of the competitor brand")
    positioning: str = Field(default="", description="How they position themselves in the market")
    strengths: list[str] = Field(default_factory=list, description="Key strengths and advantages")
    weaknesses_or_gaps: list[str] = Field(default_factory=list, description="Weaknesses or gaps in their offering")
    source: str | None = Field(default=None, description="Primary website URL or source for this competitor")
    evidence: list[ResearchEvidence | dict[str, Any]] = Field(default_factory=list, description="Evidence items supporting this competitor")


class MarketResearchReport(BaseModel):
    """Final research output; separate observed evidence from interpretation."""

    market_overview: str = Field(default="", description="A high-level summary of the current market state")
    competitor_summary: list[CompetitorSummary] = Field(default_factory=list, description="Detailed breakdown of key competitors")
    pricing_insights: list[str] = Field(default_factory=list, description="Actionable insights regarding market pricing")
    offer_insights: list[str] = Field(default_factory=list, description="Insights on discounts, bundles, and offers")
    shipping_insights: list[str] = Field(default_factory=list, description="Insights on delivery and shipping trends")
    search_insights: list[str] = Field(default_factory=list, description="SEO and search visibility insights")
    social_insights: list[str] = Field(default_factory=list, description="Social media presence and strategy insights")
    opportunities_for_target_brand: list[str] = Field(default_factory=list, description="Strategic opportunities for our brand")
    risks: list[str] = Field(default_factory=list, description="Potential risks and threats in the market")
    sources: list[str] = Field(default_factory=list, description="Primary source or competitor URLs consulted during research")
    evidence: list[ResearchEvidence | dict[str, Any]] = Field(default_factory=list, description="Supporting evidence items")
    missing_inputs: list[str] = Field(default_factory=list, description="Missing inputs or unobserved facts")
    assumptions: list[str] = Field(default_factory=list, description="Key working assumptions")
