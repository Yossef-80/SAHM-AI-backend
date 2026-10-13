"""Build bounded, tenant-scoped context for creative and campaign generation."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.db.models import BrandMemoryItem, BrandProfile, Business, Campaign


async def load_marketing_context(
    db: AsyncSession,
    *,
    business_id: str,
    campaign_id: str | None = None,
    reference_analysis_id: str | None = None,
) -> dict[str, Any]:
    """Return facts that belong to this business; fail closed on foreign ids."""
    business = await db.get(Business, business_id)
    if business is None:
        raise NotFoundError("business not found")

    profile = await db.scalar(
        select(BrandProfile).where(BrandProfile.business_id == business_id)
    )
    context: dict[str, Any] = {
        "business": {
            "name": business.name,
            "industry": business.industry,
            "country": business.country,
            "language": business.language,
            "website": business.website_url,
        },
        "brand": {
            "voice": profile.brand_voice if profile else None,
            "visual_style": profile.visual_style if profile else None,
            "audience": profile.target_audience if profile else None,
            "offers": list(profile.key_offers or []) if profile else [],
            "goals": list(profile.goals or []) if profile else [],
        },
    }

    memories = (
        await db.execute(
            select(BrandMemoryItem)
            .where(
                BrandMemoryItem.business_id == business_id,
                BrandMemoryItem.kind.in_(("offer", "audience", "voice_example", "visual_style")),
            )
            .order_by(BrandMemoryItem.created_at.desc())
            .limit(16)
        )
    ).scalars().all()
    context["verified_memory"] = [
        {"kind": item.kind, "text": item.text[:1200]} for item in memories
    ]

    if campaign_id:
        campaign = await db.scalar(
            select(Campaign).where(
                Campaign.id == campaign_id,
                Campaign.business_id == business_id,
            )
        )
        if campaign is None:
            raise NotFoundError(f"campaign '{campaign_id}' not found")
        context["campaign"] = {
            "id": campaign.id,
            "name": campaign.name,
            "objective": campaign.objective,
            "channel": campaign.channel,
            "budget": campaign.budget,
            "currency": campaign.currency,
            "start_date": campaign.start_date.isoformat() if campaign.start_date else None,
            "end_date": campaign.end_date.isoformat() if campaign.end_date else None,
        }
        plan = await db.get(BrandMemoryItem, f"campcfg_{campaign_id}")
        if plan and plan.business_id == business_id and plan.kind == "campaign_plan":
            context["campaign_plan"] = dict(plan.metadata_ or {})

    if reference_analysis_id:
        reference = await db.scalar(
            select(BrandMemoryItem).where(
                BrandMemoryItem.id == reference_analysis_id,
                BrandMemoryItem.business_id == business_id,
                BrandMemoryItem.kind == "reference_analysis",
            )
        )
        if reference is None:
            raise NotFoundError(f"reference style '{reference_analysis_id}' not found")
        metadata = reference.metadata_ or {}
        style = metadata.get("style")
        if not isinstance(style, dict):
            raise NotFoundError(f"reference style '{reference_analysis_id}' is unavailable")
        context["reference_style"] = {
            "style_spec": style,
            "summary": str(metadata.get("summary") or reference.text)[:1200],
            "source_name": str(metadata.get("source_name") or "Saved reference")[:160],
        }

    return context


__all__ = ["load_marketing_context"]
