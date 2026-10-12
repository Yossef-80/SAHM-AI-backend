"""Content calendar endpoints: GET /content, POST /content/generate, PATCH /content/{id}, DELETE /content/{id}."""

from __future__ import annotations

import calendar
import datetime
import uuid
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import LLMDep, SessionDep, WorkspaceDep
from app.core.errors import NotFoundError
from app.db.models import BrandProfile, Campaign, ContentItem
from app.schemas.content import (
    ContentItemOut,
    GenerateContentCalendarRequest,
    UpdateContentItemRequest,
)

router = APIRouter(prefix="/content", tags=["content"])


def _new_content_id() -> str:
    return f"cnt_{uuid.uuid4().hex[:20]}"


@router.get("", response_model=list[ContentItemOut])
async def list_content_items(
    ws: WorkspaceDep,
    db: SessionDep,
    month: str | None = Query(None, description="Month in YYYY-MM format"),
    campaign_id: str | None = Query(None, alias="campaignId"),
) -> list[ContentItemOut]:
    """List scheduled or drafted content calendar items for a given month/campaign."""
    stmt = (
        select(ContentItem)
        .where(ContentItem.business_id == ws.business_id)
        .order_by(ContentItem.date.asc(), ContentItem.time.asc())
    )
    if month:
        try:
            y, m = [int(x) for x in month.split("-")]
            last_day = calendar.monthrange(y, m)[1]
            start_date = f"{month}-01"
            end_date = f"{month}-{last_day:02d}"
            stmt = stmt.where(ContentItem.date >= start_date, ContentItem.date <= end_date)
        except Exception:
            pass
    if campaign_id:
        stmt = stmt.where(ContentItem.campaign_id == campaign_id)

    rows = (await db.execute(stmt)).scalars().all()
    return [ContentItemOut.model_validate(row) for row in rows]


@router.post("/generate", response_model=list[ContentItemOut])
async def generate_content_calendar(
    payload: GenerateContentCalendarRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> list[ContentItemOut]:
    """Generate a content calendar for a given month and persist to DB."""
    # Check if brand profile exists for context
    brand_profile = await db.scalar(
        select(BrandProfile).where(BrandProfile.business_id == ws.business_id)
    )

    campaign_name = "General"
    if payload.campaign_id:
        c_row = await db.scalar(
            select(Campaign).where(
                Campaign.id == payload.campaign_id,
                Campaign.business_id == ws.business_id,
            )
        )
        if c_row:
            campaign_name = c_row.name

    year_str, month_str = payload.month.split("-")
    year = int(year_str)
    month_num = int(month_str)
    num_days = calendar.monthrange(year, month_num)[1]

    # Target number of posts for this month based on posts_per_week
    total_posts = max(1, int(round((num_days / 7.0) * payload.posts_per_week)))
    step = max(1, num_days // total_posts)

    channels = payload.channels or ["instagram", "facebook"]
    formats_map = {
        "instagram": ["post", "reel", "story"],
        "facebook": ["post", "ad"],
        "tiktok": ["reel"],
        "email": ["email"],
        "blog": ["article"],
    }

    pillars = ["Educational", "Engagement", "Product Spotlight", "Customer Story", "Promotional"]
    brand_voice = brand_profile.brand_voice if brand_profile else "Engaging & Professional"

    created_items: list[ContentItem] = []
    day = 1
    post_idx = 0

    while day <= num_days and post_idx < total_posts:
        channel = channels[post_idx % len(channels)]
        fmt_options = formats_map.get(channel, ["post"])
        fmt = fmt_options[post_idx % len(fmt_options)]
        pillar = pillars[post_idx % len(pillars)]
        date_str = f"{year:04d}-{month_num:02d}-{day:02d}"
        time_str = "12:00" if post_idx % 2 == 0 else "18:30"

        title_detail = payload.brief or f"{campaign_name} - {pillar} {fmt.title()}"
        if payload.brief:
            title = f"{payload.brief[:40]} #{post_idx + 1}"
        else:
            title = f"{pillar}: {channel.title()} {fmt.title()}"

        hook = f"Hook for {title}: Why you need to focus on {pillar.lower()} today!"
        caption = (
            f"Here is your scheduled {fmt} for {channel}.\n\n"
            f"Focus: {title_detail}\n"
            f"Tone: {brand_voice}\n\n"
            f"Drop your thoughts below! 👇"
        )
        hashtags = [f"#{channel}", f"#{pillar.replace(' ', '')}", f"#{campaign_name.replace(' ', '')}"]

        item = ContentItem(
            id=_new_content_id(),
            business_id=ws.business_id,
            date=date_str,
            time=time_str,
            channel=channel,
            format=fmt,
            title=title,
            hook=hook,
            caption=caption,
            hashtags=hashtags,
            status="scheduled",
            campaign_id=payload.campaign_id,
            pillar=pillar,
        )
        db.add(item)
        created_items.append(item)

        day += step
        post_idx += 1

    await db.commit()
    for item in created_items:
        await db.refresh(item)

    return [ContentItemOut.model_validate(item) for item in created_items]


@router.patch("/{item_id}", response_model=ContentItemOut)
async def update_content_item(
    item_id: str,
    patch: UpdateContentItemRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> ContentItemOut:
    """Update date, time, title, status, etc. of a calendar entry."""
    item = await db.scalar(
        select(ContentItem).where(
            ContentItem.id == item_id,
            ContentItem.business_id == ws.business_id,
        )
    )
    if not item:
        raise NotFoundError(f"Content item '{item_id}' not found")

    if patch.date is not None:
        item.date = patch.date
    if patch.time is not None:
        item.time = patch.time
    if patch.channel is not None:
        item.channel = patch.channel
    if patch.format is not None:
        item.format = patch.format
    if patch.title is not None:
        item.title = patch.title
    if patch.hook is not None:
        item.hook = patch.hook
    if patch.caption is not None:
        item.caption = patch.caption
    if patch.hashtags is not None:
        item.hashtags = patch.hashtags
    if patch.status is not None:
        item.status = patch.status
    if patch.pillar is not None:
        item.pillar = patch.pillar

    await db.commit()
    await db.refresh(item)
    return ContentItemOut.model_validate(item)


@router.delete("/{item_id}")
async def delete_content_item(
    item_id: str,
    ws: WorkspaceDep,
    db: SessionDep,
) -> dict[str, str]:
    """Delete a calendar entry."""
    item = await db.scalar(
        select(ContentItem).where(
            ContentItem.id == item_id,
            ContentItem.business_id == ws.business_id,
        )
    )
    if not item:
        raise NotFoundError(f"Content item '{item_id}' not found")

    await db.delete(item)
    await db.commit()
    return {"id": item_id}
