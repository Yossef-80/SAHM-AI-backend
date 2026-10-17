"""Pydantic schemas for the subscriptions and plan management system."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from app.schemas.common import ApiModel


class PlanId(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionPlan(ApiModel):
    id: PlanId = Field(..., description="Unique plan identifier")
    name: str = Field(..., description="Human-readable plan name")
    description: str = Field(..., description="Brief description of the plan")
    daily_token_budget: int = Field(..., description="Tokens allowed per UTC day")
    monthly_price_usd: float = Field(0.0, description="Monthly price in USD (0.0 for free)")
    is_free: bool = Field(True, description="Whether this plan is 100% free")
    features: list[str] = Field(default_factory=list, description="List of included capabilities")
    supported_models: list[str] = Field(default_factory=list, description="Model tiers supported")
    max_workspaces: int = Field(..., description="Allowed workspaces for this plan")


class WorkspaceSubscriptionStatus(ApiModel):
    workspace_id: str
    business_id: str
    business_name: str
    current_plan: SubscriptionPlan
    daily_token_budget: int
    tokens_used_today: int
    percent_consumed: float
    is_budget_exceeded: bool
    observability_active: bool
    observability_details: dict[str, Any] = Field(default_factory=dict)


class SwitchPlanRequest(ApiModel):
    plan_id: PlanId = Field(..., description="Target plan ID to switch to")


class SwitchPlanResponse(ApiModel):
    success: bool
    message: str
    previous_plan_id: PlanId
    new_plan: SubscriptionPlan
    updated_daily_budget: int


__all__ = [
    "PlanId",
    "SubscriptionPlan",
    "SwitchPlanRequest",
    "SwitchPlanResponse",
    "WorkspaceSubscriptionStatus",
]
