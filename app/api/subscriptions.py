"""``/api/v1/subscriptions`` -- plan and subscription management.

Allows workspaces to view available plans, inspect current subscription status & budget,
and upgrade or switch plans seamlessly without any paid gateway requirement.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import SessionDep, WorkspaceDep
from app.ai.observability import get_observability_status
from app.core.subscriptions import (
    get_workspace_subscription,
    list_subscription_plans,
    switch_workspace_plan,
)
from app.schemas.subscriptions import (
    SubscriptionPlan,
    SwitchPlanRequest,
    SwitchPlanResponse,
    WorkspaceSubscriptionStatus,
)

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


@router.get("/plans", response_model=list[SubscriptionPlan])
async def get_plans() -> list[SubscriptionPlan]:
    """List all available subscription plans and their quotas."""
    return list_subscription_plans()


@router.get("/current", response_model=WorkspaceSubscriptionStatus)
async def get_current_subscription(
    ws: WorkspaceDep,
    db: SessionDep,
) -> WorkspaceSubscriptionStatus:
    """Get the active subscription plan, token budget, and usage for this workspace."""
    return await get_workspace_subscription(db, ws.workspace_id)


@router.post("/switch", response_model=SwitchPlanResponse)
async def switch_plan(
    request: SwitchPlanRequest,
    ws: WorkspaceDep,
    db: SessionDep,
) -> SwitchPlanResponse:
    """Switch subscription plan for this workspace and update the daily token budget."""
    return await switch_workspace_plan(db, ws.workspace_id, request.plan_id)


@router.get("/observability")
async def get_observability_info(
    ws: WorkspaceDep,
) -> dict[str, Any]:
    """Inspect active LangSmith, Langfuse, and local logging integrations."""
    return get_observability_status()


__all__ = ["router"]
