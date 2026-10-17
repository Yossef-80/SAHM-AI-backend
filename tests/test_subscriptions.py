"""Tests for Subscriptions catalog, plan switching, and API endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.subscriptions import (
    PLANS,
    get_plan,
    list_subscription_plans,
    resolve_plan_for_budget,
    switch_workspace_plan,
)
from app.db.models import Business
from app.schemas.subscriptions import PlanId


def test_plans_catalog() -> None:
    plans = list_subscription_plans()
    assert len(plans) == 3
    plan_ids = [p.id for p in plans]
    assert PlanId.FREE in plan_ids
    assert PlanId.PRO in plan_ids
    assert PlanId.ENTERPRISE in plan_ids

    # All plans are fully free to use
    for plan in plans:
        assert plan.is_free is True
        assert plan.monthly_price_usd == 0.0
        assert plan.daily_token_budget > 0
        assert len(plan.features) > 0


def test_resolve_plan_for_budget() -> None:
    assert resolve_plan_for_budget(50_000).id == PlanId.FREE
    assert resolve_plan_for_budget(100_000).id == PlanId.FREE
    assert resolve_plan_for_budget(1_000_000).id == PlanId.PRO
    assert resolve_plan_for_budget(10_000_000).id == PlanId.ENTERPRISE


@pytest.mark.asyncio
async def test_switch_workspace_plan_updates_budget(seeded_db: AsyncSession) -> None:
    # Query seeded business
    stmt = select(Business)
    business = (await seeded_db.scalars(stmt)).first()
    assert business is not None

    workspace_id = business.workspace_id

    # Switch to PRO
    resp = await switch_workspace_plan(seeded_db, workspace_id, PlanId.PRO)
    assert resp.success is True
    assert resp.new_plan.id == PlanId.PRO
    assert resp.updated_daily_budget == 1_000_000

    # Verify updated in DB
    await seeded_db.refresh(business)
    assert business.daily_token_budget == 1_000_000

    # Switch to ENTERPRISE
    resp_ent = await switch_workspace_plan(seeded_db, workspace_id, PlanId.ENTERPRISE)
    assert resp_ent.success is True
    assert resp_ent.new_plan.id == PlanId.ENTERPRISE
    assert resp_ent.updated_daily_budget == 10_000_000

    await seeded_db.refresh(business)
    assert business.daily_token_budget == 10_000_000


@pytest.mark.asyncio
async def test_subscriptions_api_endpoints(api_client: AsyncClient) -> None:
    client = api_client

    # 1. Plans list
    res = await client.get("/api/v1/subscriptions/plans")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 3
    assert any(p["id"] == "free" for p in data)

    # 2. Current subscription (default workspace resolution)
    res_curr = await client.get("/api/v1/subscriptions/current")
    assert res_curr.status_code == 200
    curr_data = res_curr.json()
    assert "currentPlan" in curr_data
    assert "dailyTokenBudget" in curr_data
    assert "observabilityDetails" in curr_data

    # 3. Observability info
    res_obs = await client.get("/api/v1/subscriptions/observability")
    assert res_obs.status_code == 200
    obs_data = res_obs.json()
    assert "local_logging" in obs_data
    assert "langsmith" in obs_data
    assert "langfuse" in obs_data

    # 4. Switch plan (with explicit business id header)
    res_switch = await client.post(
        "/api/v1/subscriptions/switch",
        headers={"X-Workspace-Id": "biz_test"},
        json={"planId": "pro"},
    )
    assert res_switch.status_code == 200
    switch_data = res_switch.json()
    assert switch_data["success"] is True
    assert switch_data["newPlan"]["id"] == "pro"
    assert switch_data["updatedDailyBudget"] == 1_000_000


async def test_displayed_budget_matches_the_enforced_budget(
    seeded_db: AsyncSession, test_settings
) -> None:
    """The plan endpoint must report the ceiling the cost guard actually enforces.

    A workspace without an override used to be shown a hardcoded 100k while the
    guard fell back to ``Settings.daily_token_budget`` (2M by default) — the user
    would see "5% used" and then get cut off at a completely different number.
    """
    from app.core.subscriptions import get_workspace_subscription
    from app.core.usage import effective_daily_budget

    business = await seeded_db.scalar(select(Business))
    assert business is not None
    business.daily_token_budget = None
    await seeded_db.commit()

    status = await get_workspace_subscription(seeded_db, business.workspace_id)
    enforced = await effective_daily_budget(seeded_db, business.workspace_id)

    assert status.daily_token_budget == enforced
    assert status.daily_token_budget == test_settings.daily_token_budget
