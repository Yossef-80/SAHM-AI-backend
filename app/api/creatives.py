"""Creative endpoints: analyze-reference and generate."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import (
    LLMDep,
    SessionDep,
    WorkspaceContext,
    WorkspaceDep,
    assert_campaign_owned,
)
from app.core.tools.base import ToolContext
from app.core.tools.media import _handle_analyze_reference, _handle_generate_image
from app.db.models import BrandMemoryItem, Creative
from app.schemas.creatives import (
    AnalyzeReferenceRequest,
    GenerateCreativeRequest,
    GenerateCreativeResponse,
    ReferenceAnalysis,
)
from app.schemas.tools import AnalyzeReferenceInput, GenerateImageInput


router = APIRouter(prefix="/creatives", tags=["creatives"])


def _ctx(db, ws: WorkspaceContext, llm) -> ToolContext:
    return ToolContext(
        workspace_id=ws.workspace_id,
        business_id=ws.business_id,
        db=db,
        llm=llm,
        request_id="",
    )


@router.post("/analyze-reference", response_model=ReferenceAnalysis)
async def analyze_reference(
    payload: AnalyzeReferenceRequest, ws: WorkspaceDep, db: SessionDep, llm: LLMDep
) -> ReferenceAnalysis:
    """Analyse a reference creative into a structured StyleSpec."""
    # The campaign id arrives in the body, so it is checked against the
    # workspace before it is used -- otherwise a caller could file another
    # tenant's creative into their own analysis.
    await assert_campaign_owned(db, payload.campaign_id, ws)
    analysis = await _handle_analyze_reference(
        AnalyzeReferenceInput(
            image_url=payload.image_url,
            upload_id=payload.upload_id,
            campaign_id=payload.campaign_id,
            focus=payload.focus,
        ),
        _ctx(db, ws, llm),
    )
    await db.commit()
    return analysis


@router.post("/generate", response_model=GenerateCreativeResponse)
async def generate_creative(
    payload: GenerateCreativeRequest, ws: WorkspaceDep, db: SessionDep, llm: LLMDep
) -> GenerateCreativeResponse:
    """Generate image creatives from a prompt and an optional style spec."""
    await assert_campaign_owned(db, payload.campaign_id, ws)

    analysis: ReferenceAnalysis | None = None
    if payload.reference_analysis_id and not payload.style_spec:
        item = await db.get(BrandMemoryItem, payload.reference_analysis_id)
        # Another workspace's reference analysis is simply not found here:
        # degrading to "no style spec" rather than erroring keeps generation
        # working without ever reading across the boundary.
        if (
            item is not None
            and item.business_id == ws.business_id
            and (item.metadata_ or {}).get("style")
        ):
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
        _ctx(db, ws, llm),
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
async def list_creatives(ws: WorkspaceDep, db: SessionDep) -> list[dict]:
    rows = (
        await db.execute(
            select(Creative)
            .where(Creative.business_id == ws.business_id)
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
