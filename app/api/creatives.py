"""Creative endpoints: analyze-reference and generate."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import SessionDep
from app.core.tools.base import ToolContext
from app.core.tools.media import _handle_analyze_reference, _handle_generate_image
from app.db.models import Business, Creative
from app.schemas.creatives import (
    AnalyzeReferenceRequest,
    GenerateCreativeRequest,
    GenerateCreativeResponse,
    ReferenceAnalysis,
)
from app.schemas.tools import AnalyzeReferenceInput, GenerateImageInput


router = APIRouter(prefix="/creatives", tags=["creatives"])


async def _business(db) -> Business:
    stmt = select(Business).order_by(Business.created_at).limit(1)
    business = (await db.execute(stmt)).scalars().first()
    if business is None:
        import uuid

        business = Business(id=f"biz_{uuid.uuid4().hex[:20]}", name="My Business")
        db.add(business)
        await db.flush()
    return business


def _ctx(db, business_id: str) -> ToolContext:
    from app.core.llm.router import get_llm

    return ToolContext(
        workspace_id="default",
        business_id=business_id,
        db=db,
        llm=get_llm("strong"),
        request_id="",
    )


@router.post("/analyze-reference", response_model=ReferenceAnalysis)
async def analyze_reference(
    payload: AnalyzeReferenceRequest, db: SessionDep
) -> ReferenceAnalysis:
    """Analyse a reference creative into a structured StyleSpec."""
    business = await _business(db)
    analysis = await _handle_analyze_reference(
        AnalyzeReferenceInput(
            image_url=payload.image_url,
            upload_id=payload.upload_id,
            campaign_id=payload.campaign_id,
            focus=payload.focus,
        ),
        _ctx(db, business.id),
    )
    await db.commit()
    return analysis


@router.post("/generate", response_model=GenerateCreativeResponse)
async def generate_creative(
    payload: GenerateCreativeRequest, db: SessionDep
) -> GenerateCreativeResponse:
    """Generate image creatives from a prompt and an optional style spec."""
    business = await _business(db)

    analysis: ReferenceAnalysis | None = None
    if payload.reference_analysis_id and not payload.style_spec:
        from app.db.models import BrandMemoryItem

        item = await db.get(BrandMemoryItem, payload.reference_analysis_id)
        if item is not None and (item.metadata_ or {}).get("style"):
            payload = payload.model_copy(
                update={"style_spec": dict(item.metadata_["style"])}
            )

    result = await _handle_generate_image(
        GenerateImageInput(
            prompt=payload.prompt,
            style_spec=payload.style_spec.model_dump() if payload.style_spec else {},
            aspect_ratio=payload.aspect_ratio,
            campaign_id=payload.campaign_id,
            num_images=payload.num_images,
            reference_analysis_id=payload.reference_analysis_id,
        ),
        _ctx(db, business.id),
    )
    await db.commit()

    from app.schemas.creatives import GeneratedAsset, StyleSpec

    assets = [
        GeneratedAsset(
            id=a.id,
            url=a.url,
            prompt=a.prompt,
            aspect_ratio=a.aspect_ratio,
            provider=a.provider,
            model=a.model,
            style_spec=StyleSpec(**a.style_spec) if a.style_spec else None,
            reference_analysis_id=a.reference_analysis_id,
            status=a.status,
            source=a.source,
            created_at=None,
        )
        for a in result.assets
    ]
    return GenerateCreativeResponse(assets=assets, reference_analysis=analysis)


@router.get("", response_model=list[dict])
async def list_creatives(db: SessionDep) -> list[dict]:
    business = await _business(db)
    rows = (
        await db.execute(
            select(Creative)
            .where(Creative.business_id == business.id)
            .order_by(Creative.created_at.desc())
            .limit(100)
        )
    ).scalars().all()
    return [
        {
            "id": c.id,
            "url": c.url,
            "prompt": c.prompt,
            "aspectRatio": c.aspect_ratio,
            "provider": c.provider,
            "campaignId": c.campaign_id,
            "createdAt": c.created_at.isoformat() if c.created_at else None,
        }
        for c in rows
    ]


__all__ = ["analyze_reference", "generate_creative", "list_creatives"]
