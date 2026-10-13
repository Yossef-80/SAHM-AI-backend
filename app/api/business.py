"""Business endpoints: POST /business, GET|PATCH /business/profile.

P2.1 made the brand profile actually writable. Before this, ``POST /business``
wrote exactly one of the six profile fields (``goals``) and the other five --
``brand_voice``, ``visual_style``, ``target_audience``, ``key_offers``,
``competitors`` -- had a read path and no write path anywhere in the codebase.

Why ``POST`` was kept
---------------------
``POST /business`` is the *create/replace* path: the frontend's onboarding calls
it with a whole business, and ``scripts/seed_demo_data.py`` depends on it. It is
not removed and not deprecated. What changed is that it is no longer the only
way to write a profile: ``PATCH /business/profile`` is the partial-update surface,
and both return the identical ``BusinessProfileResponse`` so the client needs no
adapter for either.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import (
    AuthDep,
    LLMDep,
    SessionDep,
    SettingsDep,
    WorkspaceDep,
    require_role,
)
from app.core.brand.identity_service import (
    get_brand_identity as get_identity_service,
    sync_brand_identity_from_chats as sync_identity_service,
    update_brand_identity as update_identity_service,
)
from app.core.errors import ValidationFailed
from app.core.memory.store import BrandMemoryStore
from app.db.models import BrandProfile as BrandProfileRow
from app.schemas.business import (
    BrandIdentityResponse,
    BrandIdentityUpdateRequest,
    BrandProfile,
    BusinessCreateRequest,
    BusinessProfileResponse,
    BusinessProfileUpdateRequest,
)

router = APIRouter(prefix="/business", tags=["business"])

#: Writes need ``member`` or above; a viewer gets 403 (P1.7 role matrix:
#: "Write brand profile + memory" is owner/member).
WRITE_PROFILE = "member"


@router.post("", response_model=BusinessProfileResponse)
async def create_business(
    payload: BusinessCreateRequest, ws: WorkspaceDep, db: SessionDep
) -> BusinessProfileResponse:
    """Create (or replace) the workspace's business and brand profile."""
    business = ws.business
    business.name = payload.name
    business.industry = payload.industry
    business.website_url = payload.website_url
    business.country = (payload.country or "").upper() or None
    business.language = payload.language or business.language
    business.currency = payload.currency
    business.monthly_budget = payload.monthly_budget

    stmt = select(BrandProfileRow).where(BrandProfileRow.business_id == business.id)
    profile = (await db.execute(stmt)).scalars().first()
    if profile is None:
        profile = BrandProfileRow(id=f"bp_{business.id}", business_id=business.id)
        db.add(profile)
    profile.goals = payload.goals

    await db.commit()
    # ``updated_at`` is an ``onupdate=func.now()`` column, so after the commit it
    # is expired-but-partially-loaded and reading it would issue a one-column
    # SELECT from a synchronous attribute access -- which raises
    # ``MissingGreenlet`` under the async session. Refresh explicitly.
    await db.refresh(business)
    return await _profile_response(db, business.id)


@router.get("/profile", response_model=BusinessProfileResponse)
async def get_business_profile(
    ws: WorkspaceDep, db: SessionDep
) -> BusinessProfileResponse:
    return await _profile_response(db, ws.business_id)


@router.patch("/profile", response_model=BusinessProfileResponse)
async def update_business_profile(
    payload: BusinessProfileUpdateRequest,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> BusinessProfileResponse:
    """Write every brand profile field. Partial: only what you send changes.

    Same response shape as ``GET /business/profile``, so a form can PATCH and
    then render the result without a second request or an adapter.

    ``null`` clears a field, omitting it leaves it alone, and ``[]`` empties a
    list -- the three instructions a UI needs and none of them are ambiguous.
    """
    await require_role(principal, settings, ws, minimum=WRITE_PROFILE)
    provided = payload.provided()
    if not provided:
        raise ValidationFailed(
            "no fields to update; send at least one field in the body",
            detail={"accepted_fields": sorted(provided | _WRITABLE_FIELDS)},
        )

    business = ws.business
    for field in _BUSINESS_FIELDS:
        if field in provided:
            value = getattr(payload, field)
            if field == "country" and isinstance(value, str):
                value = value.upper() or None
            setattr(business, field, value)

    profile_fields = _PROFILE_FIELDS & provided
    if profile_fields:
        profile = (
            await db.execute(
                select(BrandProfileRow).where(BrandProfileRow.business_id == business.id)
            )
        ).scalars().first()
        if profile is None:
            profile = BrandProfileRow(id=f"bp_{business.id}", business_id=business.id)
            db.add(profile)
        for field in _PROFILE_TEXT_FIELDS & provided:
            # ``brand_profiles`` keeps these in nullable Text columns, so a
            # client ``null`` maps straight to SQL NULL. Storing ``[]`` here
            # would bind a list into a Text column and blow up at flush time.
            setattr(profile, field, getattr(payload, field))
        for field in _PROFILE_LIST_FIELDS & provided:
            # These are JSON columns declared non-nullable with a list default,
            # so "cleared" means an empty list rather than NULL.
            value = getattr(payload, field)
            setattr(profile, field, [] if value is None else value)

    await db.commit()
    # Same reason as ``create_business``: see the comment there.
    await db.refresh(business)
    return await _profile_response(db, business.id)


#: Fields written on ``businesses`` by PATCH /business/profile.
_BUSINESS_FIELDS = frozenset(
    {
        "name",
        "industry",
        "website_url",
        "country",
        "language",
        "currency",
        "monthly_budget",
        "daily_token_budget",
    }
)

#: ``brand_profiles`` columns that are nullable Text. ``null`` -> SQL NULL.
_PROFILE_TEXT_FIELDS = frozenset({"brand_voice", "visual_style", "target_audience"})

#: ``brand_profiles`` columns that are non-nullable JSON lists. ``null`` -> ``[]``.
_PROFILE_LIST_FIELDS = frozenset({"key_offers", "goals", "competitors"})

#: Fields written on ``brand_profiles`` by PATCH /business/profile.
_PROFILE_FIELDS = _PROFILE_TEXT_FIELDS | _PROFILE_LIST_FIELDS

_WRITABLE_FIELDS = _BUSINESS_FIELDS | _PROFILE_FIELDS


async def _profile_response(db, business_id: str) -> BusinessProfileResponse:
    """Assemble the brand profile view for one business (already ownership-checked)."""
    store = BrandMemoryStore(db)
    business = await store.get_business(business_id)
    profile = await store.get_profile(business_id)
    count = await store.count_items(business_id)
    learning_count = sum(
        1 for _ in await store.recent_learnings(business_id=business_id, limit=1000)
    )
    schema = BrandProfile(
        business_id=business.id,
        name=business.name,
        industry=business.industry,
        website_url=business.website_url,
        country=business.country,
        language=business.language,
        currency=business.currency,
        monthly_budget=business.monthly_budget,
        daily_token_budget=business.daily_token_budget,
        brand_voice=profile.brand_voice if profile else None,
        visual_style=profile.visual_style if profile else None,
        target_audience=profile.target_audience if profile else None,
        key_offers=list(profile.key_offers) if profile else [],
        goals=list(profile.goals) if profile else [],
        competitors=list(profile.competitors) if profile else [],
        updated_at=business.updated_at,
    )
    return BusinessProfileResponse(
        business=schema, memory_item_count=count, learning_count=learning_count
    )


@router.get("/brand-identity", response_model=BrandIdentityResponse)
async def get_brand_identity(
    ws: WorkspaceDep, db: SessionDep
) -> BrandIdentityResponse:
    """Retrieve the unified Brand Identity profile (colors, voice, SWOT, audience)."""
    return await get_identity_service(db, ws.business_id)


@router.post("/brand-identity/sync", response_model=BrandIdentityResponse)
async def sync_brand_identity(
    ws: WorkspaceDep, db: SessionDep, llm: LLMDep
) -> BrandIdentityResponse:
    """Synthesize and update the Brand Identity using insights from all chats and memory."""
    return await sync_identity_service(db, ws.business_id, llm)


@router.patch("/brand-identity", response_model=BrandIdentityResponse)
async def update_brand_identity(
    payload: BrandIdentityUpdateRequest,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> BrandIdentityResponse:
    """Manually update specific fields of the Brand Identity."""
    await require_role(principal, settings, ws, minimum=WRITE_PROFILE)
    return await update_identity_service(db, ws.business_id, payload)

