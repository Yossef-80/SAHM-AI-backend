"""Complete profile endpoint for new OAuth users."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUserDep, SessionDep, SettingsDep
from app.core.errors import NotFoundError
from app.core.supabase_auth import get_or_create_profile
from app.db.models import Business, Profile
from app.schemas.auth import CompleteProfileRequest, CompleteProfileResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/complete-profile",
    response_model=CompleteProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def complete_profile(
    payload: CompleteProfileRequest,
    principal: CurrentUserDep,
    db: SessionDep,
    settings: SettingsDep,
) -> CompleteProfileResponse:
    """Complete profile for a new OAuth user.

    Creates a business and profile for a user who signed up via OAuth
    and doesn't have a workspace yet.
    """
    # Check if user already has a profile
    existing_profile = await get_or_create_profile(
        db, principal.user_id, principal.email, principal.full_name
    )

    # If user already has a workspace, return existing info
    if existing_profile:
        from app.api.auth import _workspaces_for, _active_workspace
        workspaces = await _workspaces_for(db, principal)
        active_workspace_id = _active_workspace(workspaces)
        return CompleteProfileResponse(
            workspaces=[
                {
                    "workspaceId": w.workspace_id,
                    "businessId": w.business_id,
                    "businessName": w.business_name,
                    "role": w.role,
                    "isPrimary": w.is_primary,
                }
                for w in workspaces
            ],
            activeWorkspaceId=active_workspace_id,
        )

    # Create new business and profile
    import uuid
    business_id = f"biz_{uuid.uuid4().hex[:20]}"
    workspace_id = f"ws_{uuid.uuid4().hex[:20]}"

    business = Business(
        id=business_id,
        workspace_id=workspace_id,
        name=payload.workspace_name,
    )
    db.add(business)

    profile = Profile(
        supabase_user_id=principal.user_id,
        business_id=business_id,
        role="owner",
    )
    db.add(profile)

    await db.flush()
    await db.commit()

    # Return workspace info
    workspaces = [
        {
            "workspaceId": workspace_id,
            "businessId": business_id,
            "businessName": payload.workspace_name,
            "role": "owner",
            "isPrimary": True,
        }
    ]

    return CompleteProfileResponse(
        workspaces=workspaces,
        activeWorkspaceId=business_id,
    )