"""Persistent, workspace-scoped content calendar endpoints."""

from __future__ import annotations

import calendar
import uuid
from collections import Counter
from datetime import date
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import delete, select

from app.ai.agents.omar import service as omar_service
from app.ai.errors import ProviderError
from app.ai.security import redact_sensitive_data
from app.api.deps import (
    AuthDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    assert_campaign_owned,
    require_role,
)
from app.core.errors import NotFoundError, ToolUnavailable, ValidationFailed
from app.db.models import BrandProfile, Business
from app.db.models import ContentItem as ContentItemRow
from app.schemas.content import (
    ContentCalendarGenerateRequest,
    ContentItemDeleteResponse,
    ContentItemPatchRequest,
    ContentItemResponse,
)

router = APIRouter(prefix="/content", tags=["content calendar"])


def _month_dates(month: str) -> tuple[date, date, int]:
    """Return the first/last dates and number of days in a validated month."""
    try:
        year, month_number = (int(part) for part in month.split("-"))
        days = calendar.monthrange(year, month_number)[1]
        first = date(year, month_number, 1)
        last = date(year, month_number, days)
    except (TypeError, ValueError) as exc:
        raise ValidationFailed("month must be a real calendar month in YYYY-MM format") from exc
    return first, last, days


def _to_schema(row: ContentItemRow) -> ContentItemResponse:
    return ContentItemResponse(
        id=row.id,
        date=row.date,
        time=row.time,
        channel=row.channel,
        format=row.format,
        title=row.title,
        hook=row.hook,
        caption=row.caption,
        hashtags=row.hashtags or [],
        status=row.status,
        campaign_id=row.campaign_id,
        creative_id=None,
        pillar=row.pillar,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _business_profile(business: Business, profile: BrandProfile | None) -> dict[str, Any]:
    """Build only the saved, relevant business context for Omar."""
    return redact_sensitive_data(
        {
            "name": str(business.name or "")[:200],
            "industry": str(business.industry or "")[:120],
            "country": str(business.country or "")[:2],
            "language": str(business.language or "")[:8],
            "brand_voice": str(profile.brand_voice or "")[:2000] if profile else "",
            "visual_style": str(profile.visual_style or "")[:2000] if profile else "",
            "target_audience": str(profile.target_audience or "")[:2000] if profile else "",
            "key_offers": [str(item)[:200] for item in (profile.key_offers or [])[:20]]
            if profile
            else [],
            "goals": [str(item)[:200] for item in (profile.goals or [])[:20]]
            if profile
            else [],
        }
    )


def _campaign_context(campaign: Any | None) -> dict[str, Any] | None:
    if campaign is None:
        return None
    return {
        "name": str(campaign.name or "")[:200],
        "objective": str(campaign.objective or "")[:200],
        "channel": str(campaign.channel or "")[:60],
        "target_audience": str(campaign.target_audience or "")[:2000],
        "notes": str(campaign.notes or "")[:2000],
        "start_date": campaign.start_date.date().isoformat() if campaign.start_date else None,
        "end_date": campaign.end_date.date().isoformat() if campaign.end_date else None,
    }


def _output_language(business_language: str | None, brief: str | None) -> str:
    """Prefer an Arabic brief, otherwise follow the saved profile language."""
    if brief and any("\u0600" <= character <= "\u06ff" for character in brief):
        return "ar"
    return "ar" if (business_language or "").lower().startswith("ar") else "en"


def _scheduled_date(
    month_start: date,
    *,
    week: int,
    position: int,
    posts_per_week: int,
    days_in_month: int,
) -> date:
    """Spread each week's posts across that seven-day bucket, inside the month."""
    first_day = (week - 1) * 7 + 1
    days_in_bucket = min(7, days_in_month - (first_day - 1))
    offset = ((position * 2 + 1) * days_in_bucket) // (2 * posts_per_week)
    offset = min(offset, days_in_bucket - 1)
    return date(month_start.year, month_start.month, first_day + offset)


@router.get("", response_model=list[ContentItemResponse])
async def list_content_items(
    ws: WorkspaceDep,
    db: SessionDep,
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    campaign_id: str | None = Query(default=None, alias="campaignId"),
) -> list[ContentItemResponse]:
    """Read only this workspace's entries for one month."""
    first, last, _ = _month_dates(month)
    stmt = select(ContentItemRow).where(
        ContentItemRow.business_id == ws.business_id,
        ContentItemRow.date >= first,
        ContentItemRow.date <= last,
    )
    if campaign_id is not None:
        stmt = stmt.where(ContentItemRow.campaign_id == campaign_id)
    rows = (
        await db.execute(
            stmt.order_by(ContentItemRow.date, ContentItemRow.created_at, ContentItemRow.id)
        )
    ).scalars().all()
    return [_to_schema(row) for row in rows]


@router.post("/generate", response_model=list[ContentItemResponse])
async def generate_calendar(
    payload: ContentCalendarGenerateRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> list[ContentItemResponse]:
    """Generate and persist editable ideas; no campaign strategy is required."""
    await require_role(principal, settings, ws, minimum="member")
    campaign = await assert_campaign_owned(db, payload.campaign_id, ws)
    business = await db.get(Business, ws.business_id)
    if business is None:
        raise NotFoundError("business not found")
    profile = await db.scalar(
        select(BrandProfile).where(BrandProfile.business_id == ws.business_id).limit(1)
    )

    calendar_input = payload.model_dump(mode="json", by_alias=False, exclude_none=False)
    calendar_input["language"] = _output_language(business.language, payload.brief)
    try:
        draft = await omar_service.generate_content_calendar(
            calendar_input=redact_sensitive_data(calendar_input),
            business_profile=_business_profile(business, profile),
            campaign_context=redact_sensitive_data(_campaign_context(campaign)),
        )
    except ProviderError as exc:
        raise ToolUnavailable(
            "Omar could not produce a usable content calendar. Check the configured AI provider; "
            "no items were changed.",
            detail={"service": "omar_content_calendar", "failure": "provider_unavailable"},
        ) from exc

    first, last, days_in_month = _month_dates(payload.month)
    # Re-generating a month replaces only unscheduled ideas in the same campaign
    # scope. Drafted/scheduled/published entries remain untouched.
    delete_stmt = delete(ContentItemRow).where(
        ContentItemRow.business_id == ws.business_id,
        ContentItemRow.date >= first,
        ContentItemRow.date <= last,
        ContentItemRow.status == "idea",
    )
    if payload.campaign_id is None:
        delete_stmt = delete_stmt.where(ContentItemRow.campaign_id.is_(None))
    else:
        delete_stmt = delete_stmt.where(ContentItemRow.campaign_id == payload.campaign_id)
    await db.execute(delete_stmt.execution_options(synchronize_session=False))

    positions: Counter[int] = Counter()
    rows: list[ContentItemRow] = []
    for item in draft.items:
        # app.ai validates the week and cadence before returning this draft.
        week = int(item.week or 1)
        item_date = _scheduled_date(
            first,
            week=week,
            position=positions[week],
            posts_per_week=payload.posts_per_week,
            days_in_month=days_in_month,
        )
        positions[week] += 1
        rows.append(
            ContentItemRow(
                id=f"content_{uuid.uuid4().hex[:20]}",
                business_id=ws.business_id,
                campaign_id=payload.campaign_id,
                date=item_date,
                time=None,
                channel=item.channel,
                format=item.format,
                title=item.title,
                hook=item.hook or None,
                caption=item.caption or None,
                hashtags=item.hashtags,
                status="idea",
                pillar=item.pillar or None,
            )
        )

    db.add_all(rows)
    await db.commit()
    generated_ids = [row.id for row in rows]
    saved = (
        await db.execute(
            select(ContentItemRow)
            .where(
                ContentItemRow.business_id == ws.business_id,
                ContentItemRow.id.in_(generated_ids),
            )
            .order_by(ContentItemRow.date, ContentItemRow.created_at, ContentItemRow.id)
        )
    ).scalars().all()
    return [_to_schema(row) for row in saved]


@router.patch("/{content_id}", response_model=ContentItemResponse)
async def update_content_item(
    content_id: str,
    payload: ContentItemPatchRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> ContentItemResponse:
    """Edit an item in this workspace; changing its date moves its month."""
    await require_role(principal, settings, ws, minimum="member")
    row = await db.scalar(
        select(ContentItemRow).where(
            ContentItemRow.id == content_id,
            ContentItemRow.business_id == ws.business_id,
        )
    )
    if row is None:
        raise NotFoundError("content item not found")

    for field_name, value in payload.model_dump(exclude_unset=True).items():
        setattr(row, field_name, value)
    await db.commit()
    await db.refresh(row)
    return _to_schema(row)


@router.delete("/{content_id}", response_model=ContentItemDeleteResponse)
async def delete_content_item(
    content_id: str,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> ContentItemDeleteResponse:
    """Delete one item after checking workspace ownership; never publish anything."""
    await require_role(principal, settings, ws, minimum="member")
    row = await db.scalar(
        select(ContentItemRow).where(
            ContentItemRow.id == content_id,
            ContentItemRow.business_id == ws.business_id,
        )
    )
    if row is None:
        raise NotFoundError("content item not found")
    await db.delete(row)
    await db.commit()
    return ContentItemDeleteResponse(id=content_id)


__all__ = [
    "delete_content_item",
    "generate_calendar",
    "list_content_items",
    "router",
    "update_content_item",
]
