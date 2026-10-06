from pydantic import BaseModel, Field

class CompetitorSummary(BaseModel):
    brand: str = Field(description="The name of the competitor brand")
    positioning: str = Field(description="How they position themselves in the market")
    strengths: list[str] = Field(description="Key strengths and advantages")
    weaknesses_or_gaps: list[str] = Field(description="Weaknesses or gaps in their offering")

class MarketResearchReport(BaseModel):
    """Call this tool to submit your final, highly structured competitive market research report."""
    market_overview: str = Field(description="A high-level summary of the current market state")
    competitor_summary: list[CompetitorSummary] = Field(description="Detailed breakdown of key competitors")
    pricing_insights: list[str] = Field(description="Actionable insights regarding market pricing")
    offer_insights: list[str] = Field(description="Insights on discounts, bundles, and offers")
    shipping_insights: list[str] = Field(description="Insights on delivery and shipping trends")
    search_insights: list[str] = Field(description="SEO and search visibility insights")
    social_insights: list[str] = Field(description="Social media presence and strategy insights")
    opportunities_for_target_brand: list[str] = Field(description="Strategic opportunities for our brand")
    risks: list[str] = Field(description="Potential risks and threats in the market")
