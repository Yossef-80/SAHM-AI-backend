"""Schemas for integration status/connect endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ApiModel


class MetaInterestSuggestion(ApiModel):
    """One resolved interest, in the shape Meta's `flexible_spec` expects."""

    id: str
    name: str
    audience_size: int | None = None
    path: list[str] = Field(default_factory=list)


class MetaTargetingSearchResponse(ApiModel):
    available: bool = True
    unavailable_reason: str | None = None
    results: list[MetaInterestSuggestion] = Field(default_factory=list)


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
