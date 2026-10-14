"""Reusable audience presets and plain-language audience assistance."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AuthDep, SessionDep, SettingsDep, WorkspaceDep, require_role
from app.core.errors import NotFoundError, ValidationFailed
from app.db.models import SavedAudience
from app.schemas.audiences import (
    AudienceSuggestionRequest,
    AudienceSuggestionResponse,
    SavedAudienceCreateRequest,
    SavedAudienceListResponse,
    SavedAudienceResponse,
    SavedAudienceUpdateRequest,
)

router = APIRouter(prefix="/audiences", tags=["audiences"])


def _to_schema(row: SavedAudience) -> SavedAudienceResponse:
    return SavedAudienceResponse(
        id=row.id,
        business_id=row.business_id,
        name=row.name,
        description=row.description,
        targeting=row.targeting or {},
        source=row.source or "manual",
        is_archived=row.is_archived,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def get_saved_audience(
    db: AsyncSession,
    *,
    audience_id: str,
    business_id: str,
    include_archived: bool = False,
) -> SavedAudience:
    """Look up an audience inside one business boundary; never trust an ID alone."""
    stmt = select(SavedAudience).where(
        SavedAudience.id == audience_id,
        SavedAudience.business_id == business_id,
    )
    if not include_archived:
        stmt = stmt.where(SavedAudience.is_archived.is_(False))
    row = (await db.execute(stmt)).scalars().first()
    if row is None:
        raise NotFoundError("saved audience not found")
    return row


@router.get("", response_model=SavedAudienceListResponse)
async def list_audiences(
    ws: WorkspaceDep,
    db: SessionDep,
    include_archived: bool = Query(default=False),
) -> SavedAudienceListResponse:
    stmt = select(SavedAudience).where(SavedAudience.business_id == ws.business_id)
    if not include_archived:
        stmt = stmt.where(SavedAudience.is_archived.is_(False))
    stmt = stmt.order_by(SavedAudience.updated_at.desc(), SavedAudience.name.asc())
    rows = (await db.execute(stmt)).scalars().all()
    return SavedAudienceListResponse(
        audiences=[_to_schema(row) for row in rows],
        total=len(rows),
    )


@router.post("", response_model=SavedAudienceResponse, status_code=201)
async def create_audience(
    payload: SavedAudienceCreateRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> SavedAudienceResponse:
    await require_role(principal, settings, ws, minimum="member")
    row = SavedAudience(
        id=f"aud_{uuid.uuid4().hex[:20]}",
        business_id=ws.business_id,
        name=payload.name,
        description=payload.description,
        targeting=payload.targeting.model_dump(mode="json", by_alias=True),
        source=payload.source,
        is_archived=False,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _to_schema(row)


@router.get("/{audience_id}", response_model=SavedAudienceResponse)
async def read_audience(
    audience_id: str, ws: WorkspaceDep, db: SessionDep
) -> SavedAudienceResponse:
    # Archived presets remain readable for campaign history and snapshot review.
    row = await get_saved_audience(
        db,
        audience_id=audience_id,
        business_id=ws.business_id,
        include_archived=True,
    )
    return _to_schema(row)


@router.patch("/{audience_id}", response_model=SavedAudienceResponse)
async def update_audience(
    audience_id: str,
    payload: SavedAudienceUpdateRequest,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> SavedAudienceResponse:
    await require_role(principal, settings, ws, minimum="member")
    row = await get_saved_audience(
        db, audience_id=audience_id, business_id=ws.business_id
    )
    fields_set = payload.model_fields_set
    if "name" in fields_set:
        if payload.name is None:
            raise ValidationFailed("Audience name cannot be cleared.")
        row.name = payload.name
    if "description" in fields_set:
        row.description = payload.description
    if "source" in fields_set:
        if payload.source is None:
            raise ValidationFailed("Audience source cannot be cleared.")
        row.source = payload.source
    if "targeting" in fields_set:
        if payload.targeting is None:
            raise ValidationFailed("Audience targeting cannot be cleared.")
        row.targeting = payload.targeting.model_dump(mode="json", by_alias=True)
    await db.commit()
    await db.refresh(row)
    return _to_schema(row)


@router.delete("/{audience_id}", response_model=SavedAudienceResponse)
async def archive_audience(
    audience_id: str,
    ws: WorkspaceDep,
    db: SessionDep,
    principal: AuthDep,
    settings: SettingsDep,
) -> SavedAudienceResponse:
    """Archive, rather than hard-delete, so linked campaign history remains safe."""
    await require_role(principal, settings, ws, minimum="member")
    row = await get_saved_audience(
        db, audience_id=audience_id, business_id=ws.business_id
    )
    row.is_archived = True
    await db.commit()
    await db.refresh(row)
    return _to_schema(row)


@router.post("/suggest", response_model=AudienceSuggestionResponse)
async def suggest_audience(
    payload: AudienceSuggestionRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> AudienceSuggestionResponse:
    """Suggest audience angles from plain language; never invent Meta IDs or write externally."""
    from app.core.audiences.suggestions import generate_audience_suggestions

    return await generate_audience_suggestions(
        db,
        business_id=ws.business_id,
        request=payload,
    )
