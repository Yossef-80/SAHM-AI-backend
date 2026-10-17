"""Core business logic for subscriptions and token plan management.

Designed to be 100% free and professional out-of-the-box:
- No paid credit card or Stripe gateway required to run.
- Supports Free, Pro (Free Access), and Enterprise (Self-Hosted) plans.
- Dynamically controls daily token budgets and maps seamlessly to the usage meter.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.observability import get_observability_status
from app.core.errors import NotFoundError, ValidationFailed
from app.core.logging import get_logger
from app.core.usage import workspace_usage_today
from app.db.models import Business
from app.schemas.subscriptions import (
    PlanId,
    SubscriptionPlan,
    SwitchPlanResponse,
    WorkspaceSubscriptionStatus,
)

_logger = get_logger(__name__)

PLANS: dict[PlanId, SubscriptionPlan] = {
    PlanId.FREE: SubscriptionPlan(
        id=PlanId.FREE,
        name="Community Free",
        description="100% free starter tier for solo marketers and small teams.",
        daily_token_budget=100_000,
        monthly_price_usd=0.0,
        is_free=True,
        features=[
            "All 5 Specialist AI Agents (Omar, Layla, Karim, Nour, Salma)",
            "Local Diagnostic File Logging (LLM & Agent Tracing)",
            "Free LangSmith Tracing (5,000 traces/mo included)",
            "Free Langfuse Monitoring (50,000 events/mo included)",
            "Real-time Token & Cost Estimation per Run",
            "Content Strategy & Calendar Generation",
        ],
        supported_models=["Ollama (Local/Free)", "Gemini 1.5 Flash", "GPT-4o-mini", "Mock"],
        max_workspaces=2,
    ),
    PlanId.PRO: SubscriptionPlan(
        id=PlanId.PRO,
        name="Pro Marketing OS",
        description="High-volume marketing operations with strong-tier reasoning (Free Community License).",
        daily_token_budget=1_000_000,
        monthly_price_usd=0.0,
        is_free=True,
        features=[
            "Everything in Community Free",
            "1,000,000 Daily Token Allowance",
            "Strong-Tier LLM Access (Claude 3.5 Sonnet, GPT-4o, DeepSeek)",
            "Automated Multi-Agent Creative Campaigns & Reference Vision",
            "Competitor Ad Library Analysis & Strategy Gaps",
            "Full Langfuse & LangSmith Cloud Tracing",
        ],
        supported_models=[
            "Claude 3.5 Sonnet",
            "GPT-4o",
            "Gemini 1.5 Pro",
            "DeepSeek R1",
            "Ollama / Local",
        ],
        max_workspaces=10,
    ),
    PlanId.ENTERPRISE: SubscriptionPlan(
        id=PlanId.ENTERPRISE,
        name="Enterprise Self-Hosted",
        description="Unlimited agency workloads with private deployment & dedicated VLM (Free Self-Hosted License).",
        daily_token_budget=10_000_000,
        monthly_price_usd=0.0,
        is_free=True,
        features=[
            "Everything in Pro",
            "10,000,000 Daily Token Allowance (Unlimited Ceiling)",
            "Dedicated Vision Language Model (VLM) Endpoints",
            "Self-Hosted Langfuse Docker Tracing Integration",
            "Full Tenant Audit Logging & Security Header Enforcement",
            "Unlimited Multi-Workspace Routing",
        ],
        supported_models=["All Cloud & Local Models", "Custom Gateway Endpoints", "Private Ollama VPC"],
        max_workspaces=100,
    ),
}


def list_subscription_plans() -> list[SubscriptionPlan]:
    """Return all available subscription plans."""
    return list(PLANS.values())


def get_plan(plan_id: PlanId | str) -> SubscriptionPlan:
    """Retrieve a subscription plan by ID."""
    if isinstance(plan_id, PlanId):
        clean_id = plan_id
    else:
        val = plan_id.value if hasattr(plan_id, "value") else str(plan_id)
        clean_id = PlanId(val.lower())
    if clean_id not in PLANS:
        raise NotFoundError(f"Subscription plan '{plan_id}' not found.")
    return PLANS[clean_id]


def resolve_plan_for_budget(daily_budget: int | None) -> SubscriptionPlan:
    """Determine the closest matching plan for a given daily token budget."""
    budget = daily_budget or 100_000
    if budget >= PLANS[PlanId.ENTERPRISE].daily_token_budget:
        return PLANS[PlanId.ENTERPRISE]
    if budget >= PLANS[PlanId.PRO].daily_token_budget:
        return PLANS[PlanId.PRO]
    return PLANS[PlanId.FREE]


async def get_workspace_subscription(
    db: AsyncSession, workspace_id: str
) -> WorkspaceSubscriptionStatus:
    """Fetch the active subscription status and live meter for a workspace."""
    stmt = select(Business).where(
        (Business.workspace_id == workspace_id) | (Business.id == workspace_id)
    )
    business = await db.scalar(stmt)
    if business is None:
        raise NotFoundError(f"Business for workspace '{workspace_id}' not found.")

    daily_budget = business.daily_token_budget or 100_000
    current_plan = resolve_plan_for_budget(daily_budget)

    tokens_used = await workspace_usage_today(db, business.workspace_id or workspace_id)
    percent = round((tokens_used * 100.0 / daily_budget), 2) if daily_budget > 0 else 0.0

    obs_status = get_observability_status()
    obs_active = obs_status["langsmith"]["enabled"] or obs_status["langfuse"]["enabled"]

    return WorkspaceSubscriptionStatus(
        workspace_id=business.workspace_id or workspace_id,
        business_id=business.id,
        business_name=business.name,
        current_plan=current_plan,
        daily_token_budget=daily_budget,
        tokens_used_today=tokens_used,
        percent_consumed=percent,
        is_budget_exceeded=tokens_used >= daily_budget,
        observability_active=obs_active,
        observability_details=obs_status,
    )


async def switch_workspace_plan(
    db: AsyncSession, workspace_id: str, new_plan_id: PlanId
) -> SwitchPlanResponse:
    """Switch a workspace's subscription tier and update its daily token budget."""
    stmt = select(Business).where(
        (Business.workspace_id == workspace_id) | (Business.id == workspace_id)
    )
    business = await db.scalar(stmt)
    if business is None:
        raise NotFoundError(f"Business for workspace '{workspace_id}' not found.")

    previous_plan = resolve_plan_for_budget(business.daily_token_budget)
    new_plan = get_plan(new_plan_id)

    # Update business token budget in the database
    business.daily_token_budget = new_plan.daily_token_budget
    await db.commit()

    _logger.info(
        "subscription_plan_switched",
        extra={
            "workspace_id": workspace_id,
            "business_id": business.id,
            "previous_plan": previous_plan.id.value,
            "new_plan": new_plan.id.value,
            "updated_budget": new_plan.daily_token_budget,
        },
    )

    return SwitchPlanResponse(
        success=True,
        message=f"Workspace successfully switched to {new_plan.name}. Daily token budget updated to {new_plan.daily_token_budget:,} tokens.",
        previous_plan_id=previous_plan.id,
        new_plan=new_plan,
        updated_daily_budget=new_plan.daily_token_budget,
    )


__all__ = [
    "PLANS",
    "get_plan",
    "get_workspace_subscription",
    "list_subscription_plans",
    "resolve_plan_for_budget",
    "switch_workspace_plan",
]
