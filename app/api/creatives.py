"""Creative endpoints: analyze-reference and generate."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import ValidationError
from sqlalchemy import select

from app.ai.errors import ProviderError
from app.api.deps import (
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    assert_campaign_owned,
    assert_owned,
)
from app.core.errors import NotFoundError, TaskNotAllowedError, ToolUnavailable
from app.core.logging import get_logger
from app.core.tools.media import (
    analyze_reference_for_workspace,
    aspect_ratio_label,
    encode_uploaded_creative_image,
    generate_image_for_workspace,
    persist_inline_reference_image,
)
from app.db.models import BrandMemoryItem, BrandProfile, Creative
from app.schemas.creatives import (
    AnalyzeReferenceRequest,
    CreativeReferenceItem,
    CreativeReferenceUpdate,
    GenerateCreativeRequest,
    GenerateCreativeResponse,
    GenerateLaylaTextRequest,
    GenerateLaylaTextResponse,
    GeneratedAsset,
    ReferenceAnalysis,
    StyleSpec,
)
from app.schemas.tools import AnalyzeReferenceInput, GenerateImageInput

router = APIRouter(prefix="/creatives", tags=["creatives"])

_logger = get_logger(__name__)


def _style_spec_payload(value: object) -> dict[str, Any]:
    """A ``StyleSpec`` as the plain dict the generation input expects.

    Accepts both shapes on purpose. The request field is declared
    ``StyleSpec | None``, but the style of a saved reference analysis lives in
    the database as raw JSON, and it used to be pushed onto the request with
    ``model_copy(update=...)`` -- which skips validation. ``payload.style_spec``
    was then a ``dict``, and the first thing to notice was a ``.model_dump()``
    further down: ``AttributeError: 'dict' object has no attribute 'model_dump'``
    and a 500 on every request that carried a ``referenceAnalysisId``.

    Validating here keeps the field what its type says it is, and a stored style
    that no longer parses degrades to "no style" instead of failing the request:
    the prompt alone still generates an image, and the operator gets a log line
    naming the reference instead of a stack trace.
    """
    if isinstance(value, StyleSpec):
        return value.model_dump()
    if isinstance(value, dict):
        try:
            return StyleSpec.model_validate(value).model_dump()
        except ValidationError:
            _logger.warning(
                "reference_style_ignored",
                extra={"reason": "stored style is not a valid StyleSpec"},
            )
            return {}
    return {}


@router.post("/analyze-reference", response_model=ReferenceAnalysis)
async def analyze_reference(
    payload: AnalyzeReferenceRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    settings: SettingsDep,
) -> ReferenceAnalysis:
    """Analyse a reference creative into a structured StyleSpec."""
    # The campaign id arrives in the body, so it is checked against the
    # workspace before it is used -- otherwise a caller could file another
    # tenant's creative into their own analysis.
    await assert_campaign_owned(db, payload.campaign_id, ws)
    upload_id = payload.upload_id
    if payload.image_data:
        try:
            upload_id = persist_inline_reference_image(payload.image_data, settings)
        except ToolUnavailable as exc:
            return ReferenceAnalysis.unavailable(
                source="analyze_reference_creative:upload", reason=exc.message
            )
    analysis = await analyze_reference_for_workspace(
        AnalyzeReferenceInput(
            image_url=payload.image_url,
            upload_id=upload_id,
            campaign_id=payload.campaign_id,
            source_name=payload.source_name,
            focus=payload.focus,
        ),
        db=db,
        business_id=ws.business_id,
        settings=settings,
    )
    await db.commit()
    return analysis


@router.get("/references", response_model=list[CreativeReferenceItem])
async def list_creative_references(
    ws: WorkspaceDep, db: SessionDep
) -> list[CreativeReferenceItem]:
    """List the current workspace's saved reference-image analyses."""
    rows = (
        await db.execute(
            select(BrandMemoryItem)
            .where(
                BrandMemoryItem.business_id == ws.business_id,
                BrandMemoryItem.kind == "reference_analysis",
            )
            .order_by(BrandMemoryItem.created_at.desc())
            .limit(100)
        )
    ).scalars().all()
    references: list[CreativeReferenceItem] = []
    for row in rows:
        metadata = row.metadata_ or {}
        style = metadata.get("style")
        if not isinstance(style, dict):
            continue
        references.append(
            CreativeReferenceItem(
                id=row.id,
                source_name=str(metadata.get("source_name") or "Saved reference")[:160],
                summary=str(metadata.get("summary") or "")[:2000],
                style=StyleSpec.model_validate(style),
                favorite=bool(metadata.get("favorite", False)),
                created_at=row.created_at,
            )
        )
    return references


@router.patch("/references/{reference_id}", response_model=CreativeReferenceItem)
async def update_creative_reference(
    reference_id: str,
    payload: CreativeReferenceUpdate,
    ws: WorkspaceDep,
    db: SessionDep,
) -> CreativeReferenceItem:
    """Update the favorite flag without crossing workspace boundaries."""
    row = await db.get(BrandMemoryItem, reference_id)
    if row is None or row.business_id != ws.business_id or row.kind != "reference_analysis":
        raise NotFoundError("creative reference not found")
    metadata = dict(row.metadata_ or {})
    metadata["favorite"] = payload.favorite
    row.metadata_ = metadata
    await db.commit()
    await db.refresh(row)
    return CreativeReferenceItem(
        id=row.id,
        source_name=str(metadata.get("source_name") or "Saved reference")[:160],
        summary=str(metadata.get("summary") or "")[:2000],
        style=StyleSpec.model_validate(metadata.get("style") or {}),
        favorite=bool(metadata.get("favorite", False)),
        created_at=row.created_at,
    )


@router.post("/generate", response_model=GenerateCreativeResponse)
async def generate_creative(
    payload: GenerateCreativeRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    settings: SettingsDep,
) -> GenerateCreativeResponse:
    """Generate image creatives from a prompt and an optional style spec."""
    await assert_campaign_owned(db, payload.campaign_id, ws)

    # Resolve the style before the request object is used: ``style_spec`` is a
    # dict from here on, never a half-validated model.
    style_spec = _style_spec_payload(payload.style_spec)

    analysis: ReferenceAnalysis | None = None
    if payload.reference_analysis_id and not style_spec:
        item = await db.get(BrandMemoryItem, payload.reference_analysis_id)
        # Another workspace's reference analysis is simply not found here:
        # degrading to "no style spec" rather than erroring keeps generation
        # working without ever reading across the boundary.
        if item is not None and item.business_id == ws.business_id:
            style_spec = _style_spec_payload((item.metadata_ or {}).get("style"))

    result = await generate_image_for_workspace(
        GenerateImageInput(
            prompt=payload.prompt,
            style_spec=style_spec,
            aspect_ratio=payload.aspect_ratio,
            campaign_id=payload.campaign_id,
            num_images=payload.num_images,
            reference_analysis_id=payload.reference_analysis_id,
            reference_image_url=payload.reference_image_url,
        ),
        db=db,
        business_id=ws.business_id,
        settings=settings,
        campaign_id=payload.campaign_id,
        persona_id=payload.persona_id,
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
            # ``result.assets`` are ``GeneratedAssetRecord``s: status and source
            # describe the *generation*, not the single asset, and live on the
            # output. Reading them off the asset raised AttributeError -- a 500
            # on the success path, after the image had already been generated
            # and persisted, so the creative existed but the caller was told the
            # request had failed.
            status=result.status,
            source=result.source,
            created_at=None,
        )
        for a in result.assets
    ]
    return GenerateCreativeResponse(
        assets=assets,
        reference_analysis=analysis,
        status=str(getattr(result.status, "value", result.status)),
        unavailable_reason=result.unavailable_reason,
    )


@router.post("/generate-text", response_model=GenerateLaylaTextResponse)
async def generate_layla_text(
    payload: GenerateLaylaTextRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> GenerateLaylaTextResponse:
    """Generate supported Layla text deliverables through app.ai, not /tasks."""
    if payload.tool in {"generate_image_prompt", "generate_video_prompt"}:
        raise TaskNotAllowedError(
            "Image and video prompts must use Layla's guided conversation workflow.",
            detail={"endpoint": "/assistant/message", "workflow": "creative_prompt"},
        )
    await assert_campaign_owned(db, payload.campaign_id, ws)
    if payload.creative_id:
        creative = await db.get(Creative, payload.creative_id)
        assert_owned(creative, ws, entity="creative")

    profile = await db.scalar(
        select(BrandProfile).where(BrandProfile.business_id == ws.business_id).limit(1)
    )
    # visual_style is intentionally excluded: the frontend only appends it to
    # the brief when the user explicitly enables the saved-style toggle.
    brand_context = {
        "brand_voice": profile.brand_voice if profile else None,
        "target_audience": profile.target_audience if profile else None,
        "key_offers": profile.key_offers if profile else [],
        "goals": profile.goals if profile else [],
    }

    from app.ai.agents.layla.service import generate_layla_text as run_layla_text

    try:
        output = await run_layla_text(
            tool=payload.tool,
            brief=payload.brief,
            language=payload.language,
            brand_context=brand_context,
        )
    except ProviderError as exc:
        # Keep missing keys, missing optional packages and provider HTTP errors
        # actionable without leaking credentials, provider URLs or raw payloads.
        raise ToolUnavailable(
            "Layla's text provider is unavailable. Check the existing provider configuration; no creative was saved.",
            detail={"service": "layla_text", "failure": "provider_unavailable"},
        ) from exc
    return GenerateLaylaTextResponse(
        id=f"layla_{uuid.uuid4().hex[:20]}",
        tool=payload.tool,
        output=output,
        created_at=datetime.now(UTC),
    )


@router.post("/upload", response_model=GeneratedAsset)
async def upload_creative(
    file: Annotated[UploadFile, File()],
    ws: WorkspaceDep,
    db: SessionDep,
    campaign_id: Annotated[str | None, Form(alias="campaignId")] = None,
    note: Annotated[str, Form()] = "",
) -> GeneratedAsset:
    """Store a user-uploaded image as a creative.

    Nothing is written to the public filesystem and no URL is handed out: the
    image is validated and kept inline, and the only place it ever leaves the
    app is the ``bytes`` field Meta's adimages edge requires.
    """
    if campaign_id:
        await assert_campaign_owned(db, campaign_id, ws)

    raw = await file.read()
    data_url, width, height = encode_uploaded_creative_image(raw)

    creative = Creative(
        id=f"cr_{uuid.uuid4().hex}",
        business_id=ws.business_id,
        campaign_id=campaign_id or None,
        task_name="upload_creative",
        kind="image",
        url=data_url,
        prompt=note.strip(),
        aspect_ratio=aspect_ratio_label(width, height),
        provider="upload",
        model=None,
        status="ok",
    )
    db.add(creative)
    await db.commit()
    # commit() expires row attributes, so refresh before reading them back.
    await db.refresh(creative)
    return GeneratedAsset(
        id=creative.id,
        url=creative.url,
        prompt=creative.prompt,
        aspect_ratio=creative.aspect_ratio,
        provider=creative.provider or "",
        model=creative.model or "",
        created_at=creative.created_at,
    )


@router.get("", response_model=list[dict])
async def list_creatives(
    ws: WorkspaceDep,
    db: SessionDep,
    campaign_id: str | None = Query(default=None, alias="campaignId"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    """List this workspace's creatives, optionally scoped to one campaign.

    The filter is server-side on purpose: the client used to fetch everything and
    filter locally, so any workspace past the row cap silently lost its older
    creatives from the campaign wizard's picker.
    """
    stmt = select(Creative).where(Creative.business_id == ws.business_id)
    if campaign_id:
        stmt = stmt.where(Creative.campaign_id == campaign_id)
    rows = (
        await db.execute(stmt.order_by(Creative.created_at.desc()).limit(limit))
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


__all__ = [
    "analyze_reference",
    "generate_creative",
    "generate_layla_text",
    "list_creative_references",
    "list_creatives",
    "router",
    "update_creative_reference",
]
