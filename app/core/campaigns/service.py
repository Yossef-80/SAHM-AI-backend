"""Storage bridge for the campaign wizard's AI strategy endpoint."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.security import redact_sensitive_data
from app.core.errors import NotFoundError
from app.db.models import BrandProfile as BrandProfileRow
from app.db.models import Business
from app.schemas.campaigns import (
    CampaignStrategyGenerationRequest,
    GeneratedCampaignStrategy,
)


async def generate_campaign_strategy_draft(
    db: AsyncSession,
    *,
    business_id: str,
    request: CampaignStrategyGenerationRequest,
) -> GeneratedCampaignStrategy:
    """Load saved workspace context, then delegate to Omar's structured strategy generator."""
    business = await db.get(Business, business_id)
    if business is None:
        raise NotFoundError("business not found")

    brand_profile = await db.scalar(
        select(BrandProfileRow)
        .where(BrandProfileRow.business_id == business.id)
        .limit(1)
    )
    business_profile = {
        "name": str(business.name or "")[:200],
        "industry": str(business.industry or "")[:120],
        "website_url": str(business.website_url or "")[:500],
        "country": str(business.country or "")[:2],
        "language": str(business.language or "")[:8],
        "currency": str(business.currency or "")[:8],
        "monthly_budget": business.monthly_budget,
        "brand_voice": str(brand_profile.brand_voice or "")[:2000] if brand_profile else "",
        "visual_style": str(brand_profile.visual_style or "")[:2000] if brand_profile else "",
        "target_audience": str(brand_profile.target_audience or "")[:2000] if brand_profile else "",
        "key_offers": [str(item)[:200] for item in (brand_profile.key_offers or [])[:20]] if brand_profile else [],
        "goals": [str(item)[:200] for item in (brand_profile.goals or [])[:20]] if brand_profile else [],
        "competitors": [str(item)[:200] for item in (brand_profile.competitors or [])[:20]] if brand_profile else [],
    }
    safe_profile = redact_sensitive_data(business_profile)
    campaign_input: dict[str, Any] = request.model_dump(
        mode="json", by_alias=False, exclude_none=True
    )

    from app.ai.agents.omar.service import generate_campaign_strategy

    result = await generate_campaign_strategy(
        campaign_input=campaign_input,
        business_profile=safe_profile,
    )
    return GeneratedCampaignStrategy.model_validate(
        result.model_dump(mode="json", exclude_none=False)
    )


__all__ = ["generate_campaign_strategy_draft"]
