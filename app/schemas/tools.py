"""Input and output models for every tool.

The brief's shorthand (``web_search(query, max_results) -> list[SearchResult]``)
is realised as a wrapper carrying ``results`` plus the mandatory ``status`` /
``source`` fields from ``ToolOutput``.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.core.tools.base import ToolOutput
from app.schemas.common import ApiModel, Evidence, ToolStatus

# ============================================================== research ====


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    published_date: str | None = None
    score: float | None = None


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    country: str | None = Field(default=None, max_length=2)


class WebSearchOutput(ToolOutput):
    results: list[SearchResult] = Field(default_factory=list)
    query: str = ""


class WebsiteProfileInput(BaseModel):
    url: str = Field(min_length=1, max_length=2000)


class Offer(BaseModel):
    name: str = ""
    detail: str = ""
    price: str | None = None


class WebsiteProfile(ToolOutput):
    url: str = ""
    title: str = ""
    description: str = ""
    language: str | None = None
    headings: list[str] = Field(default_factory=list)
    offers: list[Offer] = Field(default_factory=list)
    ctas: list[str] = Field(default_factory=list)
    social_links: list[str] = Field(default_factory=list)
    contact: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    word_count: int = 0


class SocialProfileInput(BaseModel):
    platform: str = Field(description="instagram | facebook | tiktok | x | linkedin")
    handle: str = Field(min_length=1, max_length=200)


class SocialPost(BaseModel):
    text: str = ""
    likes: int | None = None
    comments: int | None = None
    posted_at: str | None = None


class SocialProfile(ToolOutput):
    platform: str = ""
    handle: str = ""
    display_name: str | None = None
    bio: str | None = None
    follower_count: int | None = None
    following_count: int | None = None
    post_count: int | None = None
    recent_posts: list[SocialPost] = Field(default_factory=list)
    profile_url: str | None = None


class GetReviewsInput(BaseModel):
    place_query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=10, ge=1, le=50)


class Review(BaseModel):
    author: str = ""
    rating: float | None = None
    text: str = ""
    relative_time: str | None = None


class GetReviewsOutput(ToolOutput):
    reviews: list[Review] = Field(default_factory=list)
    place_name: str | None = None
    average_rating: float | None = None
    total_ratings: int | None = None


class KeywordMetric(BaseModel):
    term: str
    volume: int | None = None
    competition: float | None = Field(default=None, ge=0.0, le=1.0)
    cpc: float | None = None
    currency: str | None = None
    trend: str | None = None


class GetKeywordsInput(BaseModel):
    seed_terms: list[str] = Field(min_length=1, max_length=25)
    country: str = Field(default="EG", max_length=2)
    language: str | None = Field(default=None, max_length=5)


class GetKeywordsOutput(ToolOutput):
    keywords: list[KeywordMetric] = Field(default_factory=list)
    country: str = ""
    seed_terms: list[str] = Field(default_factory=list)


class AdRecord(BaseModel):
    page_name: str = ""
    ad_id: str | None = None
    body: str = ""
    headline: str | None = None
    cta: str | None = None
    media_type: str | None = None
    started_running: str | None = None
    platforms: list[str] = Field(default_factory=list)
    snapshot_url: str | None = None


class GetCompetitorAdsInput(BaseModel):
    page_or_query: str = Field(min_length=1, max_length=300)
    country: str = Field(default="EG", max_length=2)
    max_results: int = Field(default=10, ge=1, le=50)
    #: Manual fallback: ads the user pasted/uploaded when the API has no
    #: coverage for the country. Never silently substituted.
    user_supplied_ads: list[AdRecord] = Field(default_factory=list)


class GetCompetitorAdsOutput(ToolOutput):
    ads: list[AdRecord] = Field(default_factory=list)
    page_or_query: str = ""
    country: str = ""
    #: True when the result came from user-supplied data, not the API.
    used_user_supplied: bool = False


# ================================================================= memory ====


class MemoryScope(BaseModel):
    """What get_brand_context should retrieve for this task."""

    query: str = Field(default="", description="free text used for similarity search")
    kinds: list[str] = Field(default_factory=list)
    top_k: int = Field(default=6, ge=1, le=20)
    include_learnings: bool = True
    max_learnings: int = Field(default=3, ge=0, le=10)


class GetBrandContextInput(BaseModel):
    scope: MemoryScope = Field(default_factory=MemoryScope)
    campaign_id: str | None = None
    task_name: str | None = None


class BrandContextItem(BaseModel):
    kind: str
    label: str
    detail: str = ""
    similarity: float | None = None
    memory_id: str | None = None


class BrandContext(ToolOutput):
    business_id: str = ""
    business_name: str = ""
    industry: str | None = None
    country: str | None = None
    language: str = "en"
    currency: str | None = None
    monthly_budget: float | None = None
    brand_voice: str | None = None
    visual_style: str | None = None
    target_audience: str | None = None
    key_offers: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    items: list[BrandContextItem] = Field(default_factory=list)
    learnings: list[BrandContextItem] = Field(default_factory=list)
    token_estimate: int = 0
    truncated: bool = False


class SaveLearningInput(BaseModel):
    campaign_id: str | None = None
    what_worked: str = Field(min_length=1)
    what_didnt: str = ""
    applies_to_future: bool = True
    tags: list[str] = Field(default_factory=list)


class Learning(ToolOutput):
    id: str = ""
    business_id: str = ""
    campaign_id: str | None = None
    what_worked: str = ""
    what_didnt: str = ""
    applies_to_future: bool = True
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


# ============================================================== meta_read ====


class GetCampaignInsightsInput(BaseModel):
    campaign_id: str = Field(min_length=1)
    meta_campaign_id: str | None = None
    date_start: date | None = None
    date_end: date | None = None
    #: Metrics to request. Empty means the default set.
    metrics: list[str] = Field(default_factory=list)


class GetBreakdownsInput(BaseModel):
    campaign_id: str = Field(min_length=1)
    meta_campaign_id: str | None = None
    by: str = Field(description="ad | placement | audience | age | gender")
    date_start: date | None = None
    date_end: date | None = None


# ================================================================== media ====


class AnalyzeReferenceInput(BaseModel):
    image_url: str | None = None
    upload_id: str | None = None
    campaign_id: str | None = None
    focus: str | None = None


class GenerateImageInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    style_spec: dict[str, object] = Field(default_factory=dict)
    aspect_ratio: str = Field(default="1:1", max_length=10)
    campaign_id: str | None = None
    num_images: int = Field(default=1, ge=1, le=4)
    reference_analysis_id: str | None = None


class GeneratedAssetRecord(BaseModel):
    id: str = ""
    url: str = ""
    prompt: str = ""
    aspect_ratio: str = "1:1"
    provider: str = ""
    model: str = ""
    style_spec: dict[str, object] = Field(default_factory=dict)
    reference_analysis_id: str | None = None


class GenerateImageOutput(ToolOutput):
    assets: list[GeneratedAssetRecord] = Field(default_factory=list)


# ================================================================ records ====


class CreateRecommendationInput(BaseModel):
    campaign_id: str | None = None
    insight_id: str | None = None
    action_type: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=400)
    payload: dict[str, object] = Field(default_factory=dict)
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    estimated_impact: str | None = None


class CreateRecommendationOutput(ToolOutput):
    """``status`` is the tool status; ``recommendation_status`` is the record's."""

    recommendation_id: str = ""
    recommendation_status: str = "pending"
    requires_approval: bool = True
    title: str = ""
    action_type: str = ""


# ============================================================ shared views ====


class ContextUsedView(ApiModel):
    """What the runtime reports back as a context_used block."""

    items: list[BrandContextItem] = Field(default_factory=list)
    token_estimate: int = 0
    truncated: bool = False


__all__ = [
    "AdRecord",
    "AnalyzeReferenceInput",
    "BrandContext",
    "BrandContextItem",
    "ContextUsedView",
    "CreateRecommendationInput",
    "CreateRecommendationOutput",
    "GeneratedAssetRecord",
    "GenerateImageInput",
    "GenerateImageOutput",
    "GetBrandContextInput",
    "GetBreakdownsInput",
    "GetCampaignInsightsInput",
    "GetCompetitorAdsInput",
    "GetCompetitorAdsOutput",
    "GetKeywordsInput",
    "GetKeywordsOutput",
    "GetReviewsInput",
    "GetReviewsOutput",
    "KeywordMetric",
    "Learning",
    "MemoryScope",
    "Offer",
    "ReferenceAnalysis",
    "Review",
    "SaveLearningInput",
    "SearchResult",
    "SocialPost",
    "SocialProfile",
    "SocialProfileInput",
    "ToolStatus",
    "WebSearchInput",
    "WebSearchOutput",
    "WebsiteProfile",
    "WebsiteProfileInput",
]

# Imported at the bottom to avoid a circular import at module load.
from app.schemas.creatives import ReferenceAnalysis  # noqa: E402
