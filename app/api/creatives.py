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
    persist_inline_reference_image_async,
)
from app.db.models import BrandMemoryItem, BrandProfile, Creative
from app.schemas.common import ToolStatus
from app.schemas.creatives import (
    AnalyzeReferenceRequest,
    SaveTextCreativeRequest,
    UpdateTextCreativeRequest,
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


def _style_spec_model(value: object) -> StyleSpec | None:
    """A validated ``StyleSpec`` from either shape, or ``None`` if it cannot be read.

    Accepts both shapes on purpose. The request field is declared
    ``StyleSpec | None``, but the style of a saved reference analysis lives in
    the database as raw JSON, and it used to be pushed onto the request with
    ``model_copy(update=...)`` -- which skips validation. ``payload.style_spec``
    was then a ``dict``, and the first thing to notice was a ``.model_dump()``
    further down: ``AttributeError: 'dict' object has no attribute 'model_dump'``
    and a 500 on every request that carried a ``referenceAnalysisId``.

    Everything that reads a style -- from the request or from a row -- goes
    through here, so a stored style that no longer parses is a logged "no style"
    instead of a failed request.
    """
    if isinstance(value, StyleSpec):
        return value
    if isinstance(value, dict):
        try:
            return StyleSpec.model_validate(value)
        except ValidationError:
            _logger.warning(
                "reference_style_ignored",
                extra={"reason": "stored style is not a valid StyleSpec"},
            )
            return None
    return None


def _style_spec_payload(value: object) -> dict[str, Any]:
    """The generation input's ``style_spec``: a validated model as a plain dict."""
    model = _style_spec_model(value)
    return model.model_dump() if model is not None else {}


def _reference_analysis_from_item(item: BrandMemoryItem) -> ReferenceAnalysis:
    """A stored reference analysis, in the shape the analyse endpoint returns.

    The row keeps the analysis as JSON (``style``) plus the labels it was saved
    with. Reading it back means a caller of ``/creatives/generate`` can show
    *which* reference a creative came from and what was read off it, instead of
    receiving ``referenceAnalysis: null`` for a reference that was used.

    A style that can no longer be read is reported, not hidden: the analysis is
    ``partial`` with a reason, which is exactly what happened.
    """
    metadata = item.metadata_ or {}
    style = _style_spec_model(metadata.get("style"))
    return ReferenceAnalysis(
        id=item.id,
        source=str(metadata.get("source") or ""),
        source_name=str(metadata.get("source_name") or "Saved reference")[:160],
        status=ToolStatus.OK if style is not None else ToolStatus.PARTIAL,
        unavailable_reason=(
            None
            if style is not None
            else "The saved reference's style could not be read; the image was generated from the prompt only."
        ),
        summary=str(metadata.get("summary") or "")[:2000],
        style=style or StyleSpec(),
        observed_offers=[str(offer) for offer in metadata.get("observed_offers") or []],
        observed_cta=[str(cta) for cta in metadata.get("observed_cta") or []],
        model=str(metadata.get("model") or ""),
        created_at=item.created_at,
    )


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
            upload_id = await persist_inline_reference_image_async(payload.image_data, settings)
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
        # A row whose style can no longer be read is not a usable reference. It
        # is skipped rather than raised: ``StyleSpec.model_validate`` here used to
        # take the whole list -- and so every workspace's picker -- down with it.
        style = _style_spec_model(metadata.get("style"))
        if style is None:
            continue
        references.append(
            CreativeReferenceItem(
                id=row.id,
                source_name=str(metadata.get("source_name") or "Saved reference")[:160],
                summary=str(metadata.get("summary") or "")[:2000],
                style=style,
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
    # Favouriting is a flag on the row, not a re-read of its style: a style that
    # no longer parses must not turn this PATCH into a 500 either.
    return CreativeReferenceItem(
        id=row.id,
        source_name=str(metadata.get("source_name") or "Saved reference")[:160],
        summary=str(metadata.get("summary") or "")[:2000],
        style=_style_spec_model(metadata.get("style")) or StyleSpec(),
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
            analysis = _reference_analysis_from_item(item)

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


def _brand_context_token_estimate(brand_context_text: str) -> int | None:
    """Rough size of the brand block that was sent, for the UI's debug view.

    Uses the same 4-chars-per-token estimate as the memory store; None when no
    context was attached at all.
    """
    text = (brand_context_text or "").strip()
    if not text:
        return None
    return max(1, len(text) // 4)


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
    # visual_style is intentionally excluded from the *field* fallback: the
    # frontend only appends it to the brief when the user explicitly enables the
    # saved-style toggle.
    brand_context = {
        "brand_voice": profile.brand_voice if profile else None,
        "target_audience": profile.target_audience if profile else None,
        "key_offers": profile.key_offers if profile else [],
        "goals": profile.goals if profile else [],
    }

    # The semantic brand memory, not just four profile columns. This is the
    # difference between generic strong-tier copy and copy that sounds like this
    # business: past approved examples, offers, audience notes, competitor
    # observations and campaign learnings, retrieved against the user's own
    # brief. A failure here degrades to the profile fields above -- a broken
    # embedding provider must not cost the user their ad copy.
    brand_context_text = ""
    try:
        from app.core.memory.store import (
            CREATIVE_BRAND_CONTEXT_TOKEN_CAP,
            BrandMemoryStore,
            format_context_for_prompt,
        )
        from app.schemas.tools import MemoryScope

        memory_context = await BrandMemoryStore(db).build_context(
            business_id=ws.business_id,
            scope=MemoryScope(
                query=(payload.brief or profile.brand_voice or "advertising creative")[:500],
                top_k=8,
                include_learnings=True,
                max_learnings=4,
            ),
            campaign_id=payload.campaign_id,
            token_cap=CREATIVE_BRAND_CONTEXT_TOKEN_CAP,
        )
        brand_context_text = format_context_for_prompt(
            memory_context, token_cap=CREATIVE_BRAND_CONTEXT_TOKEN_CAP
        )
    except Exception as exc:  # pragma: no cover - degraded, still useful
        _logger.warning(
            "Layla text generation is running without semantic brand memory",
            extra={"workspace_id": ws.business_id, "reason": type(exc).__name__},
        )

    from app.ai.agents.layla.service import (
        generate_layla_text as run_layla_text,
        role_for_text_tool,
    )
    from app.ai.providers.llm import get_workspace_chat_llm
    from app.ai.settings import get_ai_settings

    # Ad copy is creative work, not extraction: it gets the workspace's strong
    # tier at the creative role's temperature instead of the process-wide model
    # at `AI_LLM_TEMPERATURE`.
    role = role_for_text_tool(payload.tool)
    spec = get_ai_settings().role_spec(role)
    creative_llm: Any | None = None
    try:
        creative_llm = await get_workspace_chat_llm(
            db=db,
            business_id=ws.business_id,
            tier=spec.tier,
            default_temperature=spec.temperature,
            default_max_tokens=spec.max_tokens,
        )
    except Exception as exc:  # pragma: no cover - falls back to process settings
        _logger.warning(
            "Layla text generation fell back to process LLM settings",
            extra={"workspace_id": ws.business_id, "reason": type(exc).__name__},
        )

    # The review pass runs on its own handle: cold (see `critique` in app.ai.roles)
    # and, like the writer, resolved from the workspace's own provider settings.
    critique_llm: Any | None = None
    review_spec = get_ai_settings().role_spec("critique")
    try:
        critique_llm = await get_workspace_chat_llm(
            db=db,
            business_id=ws.business_id,
            tier=review_spec.tier,
            default_temperature=review_spec.temperature,
        )
    except Exception as exc:  # pragma: no cover - the service resolves its own
        _logger.warning(
            "Self-critique fell back to process LLM settings",
            extra={"workspace_id": ws.business_id, "reason": type(exc).__name__},
        )

    try:
        output = await run_layla_text(
            tool=payload.tool,
            brief=payload.brief,
            language=payload.language,
            brand_context=brand_context,
            brand_context_text=brand_context_text,
            llm=creative_llm,
            critique_llm=critique_llm,
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
        role=role,
        model=getattr(creative_llm, "model_name", None) if creative_llm else None,
        brand_context_tokens=_brand_context_token_estimate(brand_context_text),
    )


@router.post("/text", response_model=GeneratedAsset)
async def save_text_creative(
    payload: SaveTextCreativeRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> GeneratedAsset:
    """Save approved Layla copy as a creative, and learn the user's taste from it.

    Two signals arrive here, and they are different lessons:

    * the text is kept as Layla wrote it -> a **positive example** (this is what
      "on brand" looks like, in this workspace, today);
    * the user rewrote it before saving -> a **before/after pair**, which is worth
      more than any description of the brand voice.

    Both land in brand memory as ``voice_example`` items and both are retrieved
    into later prompts as few-shot examples.
    """
    await assert_campaign_owned(db, payload.campaign_id, ws)
    if payload.creative_id:
        assert_owned(await db.get(Creative, payload.creative_id), ws, entity="creative")

    from app.ai.marketing.assets import persist_prompt_creative

    final_text = payload.text.strip()
    record = await persist_prompt_creative(
        db,
        business_id=ws.business_id,
        campaign_id=payload.campaign_id,
        kind="copy",
        title=payload.title.strip(),
        prompt=final_text,
        aspect_ratio="1:1",
        provider="layla_text",
        model=payload.model,
        # The row's `prompt` is the deliverable itself, not a captioned one.
        include_title=False,
    )
    await _learn_from_saved_copy(
        db,
        business_id=ws.business_id,
        final_text=final_text,
        original_text=(payload.original_text or "").strip(),
        brief=payload.brief,
        campaign_id=payload.campaign_id,
    )
    await db.commit()
    return GeneratedAsset(
        id=record.id,
        url="",
        prompt=final_text,
        aspect_ratio="1:1",
        provider="layla_text",
        model=record.model or "",
        created_at=datetime.now(UTC),
    )


@router.patch("/{creative_id}/text", response_model=GeneratedAsset)
async def update_text_creative(
    creative_id: str,
    payload: UpdateTextCreativeRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> GeneratedAsset:
    """Edit a saved text creative's copy; the edit becomes brand-voice memory.

    This is the second half of the learning loop's manual-edit hook: the user
    fixes a line, and the difference between what Layla wrote and what they
    wanted is stored as an example for the next request.
    """
    creative = assert_owned(await db.get(Creative, creative_id), ws, entity="creative")

    previous = creative.prompt or ""
    revised = payload.text.strip()
    creative.prompt = revised
    creative.status = "ready"
    await db.flush()

    await _learn_from_saved_copy(
        db,
        business_id=ws.business_id,
        final_text=revised,
        original_text=previous,
        brief="",
        campaign_id=creative.campaign_id,
    )
    await db.commit()
    await db.refresh(creative)
    return GeneratedAsset(
        id=creative.id,
        url=creative.url or "",
        prompt=creative.prompt,
        aspect_ratio=creative.aspect_ratio,
        provider=creative.provider or "",
        model=creative.model or "",
        created_at=creative.created_at,
    )


async def _learn_from_saved_copy(
    db: Any,
    *,
    business_id: str,
    final_text: str,
    original_text: str,
    brief: str,
    campaign_id: str | None,
) -> None:
    """Record an approval or an edit as ``voice_example`` memory. Never raises."""
    from app.ai.agents.layla.learning import learn_from_approval, learn_from_edit
    from app.core.memory.store import BrandMemoryStore

    store = BrandMemoryStore(db)
    try:
        if not await learn_from_edit(
            store,
            business_id=business_id,
            original=original_text,
            revised=final_text,
            context=brief,
            campaign_id=campaign_id,
        ):
            # No previous text, or an edit too small to read as intent (a comma,
            # an exclamation mark): the copy itself is still a positive example.
            await learn_from_approval(
                store,
                business_id=business_id,
                text=final_text,
                brief=brief,
                campaign_id=campaign_id,
            )
    except Exception as exc:  # pragma: no cover - a memory write must not fail the save
        _logger.warning(
            "voice example could not be stored for a saved creative",
            extra={"workspace_id": business_id, "reason": type(exc).__name__},
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
            # Without `kind` the client had to assume every row was an image, so a
            # saved ad-copy creative rendered as a broken thumbnail.
            "kind": c.kind,
            "status": c.status,
            "model": c.model,
            "aspectRatio": c.aspect_ratio,
            "provider": c.provider,
            "campaignId": c.campaign_id,
            "createdAt": c.created_at.isoformat() if c.created_at else None,
        }
        for c in rows
    ]


__all__ = [
    "analyze_reference",
    "save_text_creative",
    "update_text_creative",
    "generate_creative",
    "generate_layla_text",
    "list_creative_references",
    "list_creatives",
    "router",
    "update_creative_reference",
]
