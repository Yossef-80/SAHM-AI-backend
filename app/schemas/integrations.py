"""Schemas for integration status/connect endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ApiModel
from app.schemas.targeting import TargetingCategory


class MetaInterestSuggestion(ApiModel):
    """One resolved Meta detailed-targeting value with its category and ID."""

    id: str
    name: str
    category: TargetingCategory = "interests"
    audience_size: int | None = None
    description: str | None = None
    path: list[str] = Field(default_factory=list)


class MetaTargetingSearchResponse(ApiModel):
    available: bool = True
    unavailable_reason: str | None = None
    results: list[MetaInterestSuggestion] = Field(default_factory=list)
    has_more: bool = False


class MetaAccountCampaign(ApiModel):
    """Read-only campaign summary from the connected Meta ad account."""

    meta_campaign_id: str
    name: str = ""
    objective: str | None = None
    status: str | None = None
    effective_status: str | None = None
    created_time: str | None = None
    start_time: str | None = None
    stop_time: str | None = None


class MetaAccountCampaignsResponse(ApiModel):
    """One page of account campaigns; this is a view, not a local import."""

    status: str = "ok"
    source: str = "meta_marketing_api"
    unavailable_reason: str | None = None
    account_id: str | None = None
    campaigns: list[MetaAccountCampaign] = Field(default_factory=list)
    has_more: bool = False
    next_cursor: str | None = None
    retrieved_at: datetime | None = None


class MetaIntegrationStatus(ApiModel):
    connected: bool = False
    campaign_ready: bool = False
    page_configured: bool = False
    ad_account_id: str | None = None
    ad_account_name: str | None = None
    token_expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)
    last_synced_at: datetime | None = None
    # Never includes the token itself.
    message: str = ""
    unavailable_reason: str | None = None


class MetaConnectRequest(ApiModel):
    access_token: str = Field(min_length=1)
    ad_account_id: str | None = None
    # When the token is short-lived, a long-lived exchange is attempted.
    app_id: str | None = None
    app_secret: str | None = None


class MetaConnectResponse(ApiModel):
    status: MetaIntegrationStatus
    message: str = ""


class IntegrationInfo(ApiModel):
    provider: str
    connected: bool
    detail: str = ""
