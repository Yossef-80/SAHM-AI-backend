"""Offline contract tests for Meta ad planning, approvals, Insights, and references."""

from __future__ import annotations

import base64
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.marketing import meta_ads
from app.ai.marketing.creative_prompt import compose_reference_faithful_prompt
from app.core.approvals import service as approvals
from app.core.errors import ApprovalRequired, ApprovalStateError
from app.db.models import BrandMemoryItem, Campaign, Creative, Insight, Recommendation
from app.schemas.campaigns import CampaignCreateRequest
from app.schemas.recommendations import ApproveRecommendationRequest
from app.schemas.tools import GeneratedAssetRecord, GenerateImageInput

# A real generated creative stores its bytes inline as a data URL, so the
# fixtures do too. Meta now receives `bytes`; an https fixture would make these
# tests reach out to a host that does not exist.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGPwdv8EAAJlAYUUlRngAAAAAElFTkSuQmCC"
)
PUBLIC_A = f"data:image/png;base64,{PNG_B64}"
PUBLIC_B = f"data:image/png;base64,{PNG_B64}"
DESTINATION = "https://shop.example.com/offer"


def _plan(objective: str = "sales", creative_ids: list[str] | None = None) -> meta_ads.MetaAdPlanSettings:
    delivery = {
        "awareness": ("IMPRESSIONS", "REACH"),
        "traffic": ("LINK_CLICKS", "LINK_CLICKS"),
        "engagement": ("IMPRESSIONS", "POST_ENGAGEMENT"),
        "leads": ("OFFSITE_CONVERSIONS", "OFFSITE_CONVERSIONS"),
        "sales": ("OFFSITE_CONVERSIONS", "OFFSITE_CONVERSIONS"),
        "app_installs": ("APP_INSTALLS", "APP_INSTALLS"),
    }
    billing_event, optimization_goal = delivery[objective]
    return meta_ads.MetaAdPlanSettings(
        objective=objective,
        creative_ids=creative_ids or ["cr_a", "cr_b"],
        page_id="page_123",
        targeting={"geo_locations": {"countries": ["EG"]}},
        billing_event=billing_event,
        optimization_goal=optimization_goal,
        special_ad_categories=[],
        destination_type="WEBSITE" if objective in {"traffic", "engagement", "leads"} else None,
        promoted_object={"pixel_id": "pixel_42", "custom_event_type": "PURCHASE"}
        if objective in {"leads", "sales", "app_installs"}
        else None,
        destination_url=DESTINATION,
        primary_text="A user-supplied ad message.",
        headline="A user-supplied headline",
        call_to_action={"type": "LEARN_MORE", "value": {"link": DESTINATION}},
    )


async def _add_creatives(db: AsyncSession, business_id: str = "biz_test") -> None:
    db.add_all(
        [
            Creative(
                id="cr_a",
                business_id=business_id,
                kind="image",
                url=PUBLIC_A,
                prompt="Prompt for final image A",
                aspect_ratio="1:1",
                reference_analysis_id="ref_a",
                status="ok",
            ),
            Creative(
                id="cr_b",
                business_id=business_id,
                kind="image",
                url=PUBLIC_B,
                prompt="Prompt for final image B",
                aspect_ratio="4:5",
                reference_analysis_id="ref_b",
                status="ok",
            ),
        ]
    )
    await db.flush()


async def _create_plan(db: AsyncSession, objective: str = "sales") -> tuple[Campaign, Recommendation]:
    await _add_creatives(db)
    campaign_input = CampaignCreateRequest(
        name=f"Offline {objective} plan",
        objective=objective,
        channel="meta_ads",
        budget=250.50,
        currency="EGP",
        start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
        target_audience="Explicitly supplied by test",
    )
    return await meta_ads.create_meta_ad_plan_draft(
        db,
        business_id="biz_test",
        campaign_input=campaign_input,
        plan=_plan(objective),
    )


class FakeGraph:
    """In-memory Meta Graph boundary; tests never make a network/API call."""

    def __init__(
        self,
        *,
        fail_path: str | None = None,
        fail_status_code: int = 400,
        insight_rows: list[dict[str, Any]] | None = None,
    ):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_path = fail_path
        self.fail_status_code = fail_status_code
        self.insight_rows = insight_rows or []
        self.ids: dict[str, int] = {}

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((path, payload))
        if path == self.fail_path:
            raise meta_ads.MetaGraphError(
                "Meta rejected the request (Meta error 100).",
                retry_safe=400 <= self.fail_status_code < 500,
                status_code=self.fail_status_code,
            )
        if path.endswith("/adimages"):
            self.ids["image"] = self.ids.get("image", 0) + 1
            n = self.ids["image"]
            return {"images": {f"final-{n}.png": {"hash": f"img_hash_{n}"}}}
        kind = path.rsplit("/", 1)[-1]
        self.ids[kind] = self.ids.get(kind, 0) + 1
        prefixes = {
            "campaigns": "meta_campaign",
            "adsets": "meta_adset",
            "adcreatives": "meta_creative",
            "ads": "meta_ad",
        }
        return {"id": f"{prefixes.get(kind, kind)}_{self.ids[kind]}"}

    async def post_multipart(self, path: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Mirrors the real client: same recording, same behaviour.

        Ad images now travel as form fields, so the payload carries `bytes`
        rather than `url`; recording it the same way keeps the call-sequence
        assertions meaningful.
        """
        return await self.post(path, fields)

    async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((f"GET {path}", params))
        return {"data": self.insight_rows}


async def test_supported_objective_mapping_and_server_required_settings_are_explicit() -> None:
    assert meta_ads.OBJECTIVE_TO_META == {
        "awareness": "OUTCOME_AWARENESS",
        "traffic": "OUTCOME_TRAFFIC",
        "engagement": "OUTCOME_ENGAGEMENT",
        "leads": "OUTCOME_LEADS",
        "sales": "OUTCOME_SALES",
        "app_installs": "OUTCOME_APP_PROMOTION",
    }
    required_by_objective = {
        "awareness": set(),
        "traffic": {"destination_type"},
        "engagement": {"destination_type"},
        "leads": {"destination_type", "promoted_object"},
        "sales": {"promoted_object"},
        "app_installs": {"promoted_object"},
    }
    common = {
        "page_id",
        "targeting",
        "billing_event",
        "optimization_goal",
        "special_ad_categories",
        "destination_url",
        "primary_text",
        "headline",
        "call_to_action",
    }
    for objective in meta_ads.OBJECTIVE_TO_META:
        complete = _plan(objective)
        assert meta_ads.missing_meta_ad_settings(complete) == []
        incomplete = meta_ads.MetaAdPlanSettings(objective=objective, creative_ids=["cr_a"])
        assert common | required_by_objective[objective] <= set(meta_ads.missing_meta_ad_settings(incomplete))


def test_awareness_delivery_validation_rejects_a_stale_sales_pair() -> None:
    stale = _plan("awareness").model_copy(
        update={
            "billing_event": "OFFSITE_CONVERSIONS",
            "optimization_goal": "OFFSITE_CONVERSIONS",
        }
    )
    issues = meta_ads.missing_meta_ad_settings(stale)
    assert "billing_event (for awareness, use IMPRESSIONS)" in issues
    assert "optimization_goal (for awareness, use REACH)" in issues


async def test_new_awareness_draft_canonicalizes_stale_delivery_fields(
    seeded_db: AsyncSession,
) -> None:
    await _add_creatives(seeded_db)
    plan = _plan("awareness").model_copy(
        update={
            "billing_event": "OFFSITE_CONVERSIONS",
            "optimization_goal": "OFFSITE_CONVERSIONS",
        }
    )
    campaign_input = CampaignCreateRequest(
        name="Awareness delivery map",
        objective="awareness",
        channel="meta_ads",
        budget=250.50,
        currency="EGP",
        start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
        target_audience="Explicitly supplied by test",
    )
    _, recommendation = await meta_ads.create_meta_ad_plan_draft(
        seeded_db,
        business_id="biz_test",
        campaign_input=campaign_input,
        plan=plan,
    )
    saved_plan = recommendation.payload["plan"]
    assert saved_plan["billingEvent"] == "IMPRESSIONS"
    assert saved_plan["optimizationGoal"] == "REACH"


async def test_credentials_are_rejected_from_free_form_plan_json() -> None:
    candidate = _plan()
    candidate.targeting = {"geo_locations": {"countries": ["EG"]}, "access_token": "not-a-real-token"}
    missing = meta_ads.missing_meta_ad_settings(candidate)
    assert any("credential-like" in issue for issue in missing)
    assert "not-a-real-token" not in str(missing)


def test_targeting_genders_requires_a_meta_collection_and_omits_all() -> None:
    plan = _plan("traffic")
    plan.targeting = {"geo_locations": {"countries": ["EG"]}, "genders": 1}
    issues = meta_ads.missing_meta_ad_settings(plan)
    assert any("targeting.genders" in issue for issue in issues), issues

    plan.targeting["genders"] = [1]
    assert not any("targeting.genders" in issue for issue in meta_ads.missing_meta_ad_settings(plan))
    plan.targeting.pop("genders")
    assert not any("targeting.genders" in issue for issue in meta_ads.missing_meta_ad_settings(plan))


def test_meta_plan_canonicalization_drops_text_caption_and_aligns_cta_link() -> None:
    plan = _plan("awareness")
    plan.call_to_action = {
        "type": "LEARN_MORE",
        "value": {"link": "https://old.example.com/path", "link_caption": "Shop now"},
    }

    normalized = meta_ads.canonicalize_meta_ad_plan_settings(plan)

    assert normalized.call_to_action == {
        "type": "LEARN_MORE",
        "value": {"link": DESTINATION},
    }


def test_advantage_audience_flag_only_accepts_zero_or_one() -> None:
    plan = _plan("awareness")
    plan.targeting = {
        "geo_locations": {"countries": ["EG"]},
        "targeting_automation": {"advantage_audience": 2},
    }
    issues = meta_ads.missing_meta_ad_settings(plan)
    assert "targeting.targeting_automation.advantage_audience (use 0 or 1)" in issues


async def test_campaign_plan_api_saves_pending_draft_without_meta_writes(
    api_client: Any, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _add_creatives(seeded_db)
    await seeded_db.commit()

    def forbidden_graph(**kwargs: Any) -> Any:
        raise AssertionError("saving a draft must not construct or call a Meta client")

    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", forbidden_graph)
    campaign = CampaignCreateRequest(
        name="API plan smoke",
        objective="sales",
        channel="meta_ads",
        budget=250.50,
        currency="EGP",
        start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
        target_audience="Explicitly supplied",
    )
    response = await api_client.post(
        "/api/v1/campaigns/meta-ad-plan",
        json={
            "campaign": campaign.model_dump(mode="json", by_alias=True),
            "plan": _plan("sales").model_dump(mode="json", by_alias=True),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert "No Meta request has been made" in body["message"]
    saved = await seeded_db.get(Campaign, body["campaignId"])
    recommendation = await seeded_db.get(Recommendation, body["recommendationId"])
    assert saved is not None and saved.meta_campaign_id is None and saved.status == "draft"
    assert recommendation is not None and recommendation.status == "pending"


async def test_plan_creation_is_local_and_no_meta_call_is_possible(seeded_db: AsyncSession) -> None:
    campaign, recommendation = await _create_plan(seeded_db)
    assert campaign.status == "draft"
    assert campaign.meta_campaign_id is None
    assert recommendation.status == "pending"
    assert recommendation.requires_approval is True
    assert recommendation.action_type == meta_ads.META_AD_PLAN_ACTION
    assert recommendation.execution_result is None


async def test_plan_creation_links_selected_creatives_to_the_new_campaign(
    seeded_db: AsyncSession,
) -> None:
    """Selected creatives must belong to the campaign they were chosen for.

    The pairing used to live only inside ``Recommendation.execution_result``,
    which is written after a successful Meta execution -- so ``creatives.campaign_id``
    stayed NULL and the campaign's Creatives tab (and ``GET /creatives?campaignId=``)
    could never find them.
    """
    campaign, _ = await _create_plan(seeded_db)

    rows = (
        await seeded_db.execute(
            select(Creative).where(Creative.id.in_(["cr_a", "cr_b"]))
        )
    ).scalars().all()
    assert len(rows) == 2
    assert {row.campaign_id for row in rows} == {campaign.id}


async def test_plan_creation_does_not_steal_a_creative_owned_by_another_campaign(
    seeded_db: AsyncSession,
) -> None:
    """A creative already attached elsewhere is reused, never re-parented."""
    await _add_creatives(seeded_db)
    other = (
        await seeded_db.execute(select(Creative).where(Creative.id == "cr_a"))
    ).scalar_one()
    other.campaign_id = "cmp_other"
    await seeded_db.flush()

    # _create_plan calls _add_creatives itself, so build the plan directly here.
    campaign, _ = await meta_ads.create_meta_ad_plan_draft(
        seeded_db,
        business_id="biz_test",
        campaign_input=CampaignCreateRequest(
            name="Reuses an attached creative",
            objective="sales",
            channel="meta_ads",
            budget=250.50,
            currency="EGP",
            start_date=datetime(2026, 10, 12, tzinfo=timezone.utc),
            target_audience="Explicitly supplied by test",
        ),
        plan=_plan("sales"),
    )

    refreshed = (
        await seeded_db.execute(select(Creative).where(Creative.id == "cr_a"))
    ).scalar_one()
    assert refreshed.campaign_id == "cmp_other"
    # The unattached one is still linked to the new campaign.
    unattached = (
        await seeded_db.execute(select(Creative).where(Creative.id == "cr_b"))
    ).scalar_one()
    assert unattached.campaign_id == campaign.id


@pytest.mark.parametrize("objective", list(meta_ads.OBJECTIVE_TO_META))
async def test_two_phase_approval_creates_full_hierarchy_paused_and_maps_each_creative(
    objective: str,
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, recommendation = await _create_plan(seeded_db, objective)
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "act_987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    preview = await approvals.approve_recommendation(
        seeded_db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=False),
        actor="member-test",
        request_id="req-plan-preview",
        settings=test_settings,
    )
    assert preview.requires_confirmation is True
    assert preview.executed is False
    assert fake.calls == []
    assert preview.diff[0].after["meta_status"] == "PAUSED"

    result = await approvals.approve_recommendation(
        seeded_db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=True),
        actor="member-test",
        request_id="req-plan-confirm",
        settings=test_settings,
    )
    assert result.executed is True
    assert result.recommendation.status.value == "executed"
    assert campaign.status == "paused"
    assert campaign.meta_campaign_id == "meta_campaign_1"

    expected_paths = [
        "act_987/campaigns",
        "act_987/adsets",
        "act_987/adimages",
        "act_987/adcreatives",
        "act_987/ads",
        "act_987/adimages",
        "act_987/adcreatives",
        "act_987/ads",
    ]
    assert [path for path, _ in fake.calls] == expected_paths
    assert fake.calls[0][1]["objective"] == meta_ads.OBJECTIVE_TO_META[objective]
    assert fake.calls[0][1]["status"] == "PAUSED"
    assert fake.calls[0][1]["is_adset_budget_sharing_enabled"] is False
    assert "bid_strategy" not in fake.calls[0][1]
    assert fake.calls[1][1]["status"] == "PAUSED"
    assert fake.calls[1][1]["bid_strategy"] == "LOWEST_COST_WITHOUT_CAP"
    expected_delivery = {
        "awareness": ("IMPRESSIONS", "REACH"),
        "traffic": ("LINK_CLICKS", "LINK_CLICKS"),
        "engagement": ("IMPRESSIONS", "POST_ENGAGEMENT"),
        "leads": ("OFFSITE_CONVERSIONS", "OFFSITE_CONVERSIONS"),
        "sales": ("OFFSITE_CONVERSIONS", "OFFSITE_CONVERSIONS"),
        "app_installs": ("APP_INSTALLS", "APP_INSTALLS"),
    }[objective]
    assert (
        fake.calls[1][1]["billing_event"],
        fake.calls[1][1]["optimization_goal"],
    ) == expected_delivery
    assert fake.calls[1][1]["targeting"]["targeting_automation"]["advantage_audience"] == 0
    assert "bid_amount" not in fake.calls[1][1]
    assert "bid_constraints" not in fake.calls[1][1]
    assert all(fake.calls[i][1]["status"] == "PAUSED" for i in (4, 7))
    assert fake.calls[1][1]["daily_budget"] == "25050"

    execution = recommendation.execution_result
    assert execution["status"] == "ok"
    assert execution["bid_strategy"] == "LOWEST_COST_WITHOUT_CAP"
    assert execution["billing_event"] == expected_delivery[0]
    assert execution["optimization_goal"] == expected_delivery[1]
    assert execution["advantage_audience"] == 0
    assert execution["graph_api_version"] == test_settings.meta_graph_version
    assert [entry["creative_id"] for entry in execution["ads"]] == ["cr_a", "cr_b"]
    assert [entry["creative_prompt"] for entry in execution["ads"]] == [
        "Prompt for final image A",
        "Prompt for final image B",
    ]
    assert "offline-test-token" not in str(execution)


async def test_ad_level_insights_join_by_meta_ad_id_not_row_order_or_name(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, recommendation = await _create_plan(seeded_db)
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=False), settings=test_settings
    )
    await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=True), settings=test_settings
    )
    fake.insight_rows = [
        {
            "ad_id": "meta_ad_2",
            "ad_name": "misleading name that looks like A",
            "date_start": "2026-10-01",
            "date_stop": "2026-10-10",
            "spend": "7.25",
            "impressions": "120",
            "clicks": "6",
            "ctr": "5.0",
            "actions": [{"action_type": "purchase", "value": "2"}],
        },
        {
            "ad_id": "meta_ad_1",
            "ad_name": "misleading name that looks like B",
            "date_start": "2026-10-01",
            "date_stop": "2026-10-10",
            "spend": "3.00",
            "impressions": "90",
            "clicks": "3",
            "ctr": "3.3",
            "actions": [],
        },
        {
            "ad_id": "meta_ad_external",
            "ad_name": "No known local creative",
            "date_start": "2026-10-01",
            "date_stop": "2026-10-10",
            "spend": "1.00",
            "impressions": "10",
            "clicks": "1",
            "ctr": "10",
            "actions": [],
        },
    ]
    result = await meta_ads.pull_ad_level_insights(
        seeded_db,
        campaign,
        settings=test_settings,
        date_start=date(2026, 10, 1),
        date_end=date(2026, 10, 10),
    )
    by_id = {ad.meta_ad_id: ad for ad in result.ads}
    assert result.status == "partial"
    assert by_id["meta_ad_2"].creative_id == "cr_b"
    assert by_id["meta_ad_2"].prompt == "Prompt for final image B"
    assert by_id["meta_ad_2"].image_url == PUBLIC_B
    assert by_id["meta_ad_1"].creative_id == "cr_a"
    assert by_id["meta_ad_1"].prompt == "Prompt for final image A"
    assert by_id["meta_ad_external"].mapped is False
    assert by_id["meta_ad_external"].creative_id is None
    assert by_id["meta_ad_external"].prompt is None
    assert by_id["meta_ad_2"].metrics["spend"] == 7.25
    rows = (await seeded_db.execute(select(Insight).where(Insight.campaign_id == campaign.id))).scalars().all()
    assert len(rows) == 3
    assert all(row.payload["source"] == "meta_marketing_api" for row in rows)


async def test_phase_two_cannot_skip_review_and_missing_meta_config_blocks_phase_one(
    seeded_db: AsyncSession, test_settings: Any
) -> None:
    _, recommendation = await _create_plan(seeded_db)
    test_settings.meta_access_token = None
    test_settings.meta_ad_account_id = None
    with pytest.raises(ApprovalRequired, match="cannot be approved"):
        await approvals.approve_recommendation(
            seeded_db,
            recommendation.id,
            ApproveRecommendationRequest(confirm=False),
            settings=test_settings,
        )
    assert recommendation.status == "pending"
    with pytest.raises(ApprovalStateError, match="phase 1"):
        await approvals.approve_recommendation(
            seeded_db,
            recommendation.id,
            ApproveRecommendationRequest(confirm=True),
            settings=test_settings,
        )


async def test_failed_safe_campaign_creation_can_retry_saved_plan(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry the same reviewed recommendation only when no Meta object exists."""
    campaign, recommendation = await _create_plan(seeded_db)
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "987"
    fake = FakeGraph(fail_path="act_987/campaigns")
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=False),
        actor="member-test", request_id="req-safe-retry-preview", settings=test_settings,
    )
    first = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=True),
        actor="member-test", request_id="req-safe-retry-failed", settings=test_settings,
    )
    assert first.executed is False
    failed = recommendation.execution_result
    assert failed["status"] == "failed_safe"
    assert failed["retry_safe"] is True
    assert failed["created_ids"] == {}
    assert len(fake.calls) == 1
    campaign_payload = fake.calls[0][1]
    assert "bid_strategy" not in campaign_payload
    assert campaign_payload["is_adset_budget_sharing_enabled"] is False

    # A retry still repeats phase one; the failed Meta call is not replayed
    # until the operator reviews and confirms again.
    retry_preview = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=False),
        actor="member-test", request_id="req-safe-retry-review", settings=test_settings,
    )
    assert retry_preview.requires_confirmation is True
    fake.fail_path = None
    retry = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=True),
        actor="member-test", request_id="req-safe-retry-again", settings=test_settings,
    )
    assert retry.executed is True, retry.message
    assert recommendation.status == "executed"
    assert recommendation.execution_result["status"] == "ok"
    assert campaign.meta_campaign_id == "meta_campaign_1"
    assert [path for path, _ in fake.calls].count("act_987/campaigns") == 2


async def test_rejected_adset_request_resumes_campaign_without_creating_a_duplicate(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, recommendation = await _create_plan(seeded_db)
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "987"
    fake = FakeGraph(fail_path="act_987/adsets")
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=False), settings=test_settings
    )
    first = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=True), settings=test_settings
    )
    assert first.executed is False
    assert recommendation.status == "failed"
    partial = recommendation.execution_result
    assert partial["status"] == "partial"
    assert partial["attempted_step"] == "ad_set"
    assert partial["retry_safe"] is True
    assert partial["created_ids"] == {"campaign": "meta_campaign_1"}
    call_count = len(fake.calls)

    # Recovery still requires a fresh review and explicit confirmation.
    review = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=False), settings=test_settings
    )
    assert review.requires_confirmation is True
    assert len(fake.calls) == call_count
    fake.fail_path = None
    resumed = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=True), settings=test_settings
    )

    assert resumed.executed is True, resumed.message
    assert recommendation.status == "executed"
    assert campaign.meta_campaign_id == "meta_campaign_1"
    assert recommendation.execution_result["resumed_from_partial"] is True
    paths = [path for path, _ in fake.calls]
    assert paths.count("act_987/campaigns") == 1
    assert paths.count("meta_campaign_1") == 0
    assert paths.count("act_987/adsets") == 2
    # Resume only retries the failed ad set; the campaign is not re-posted or edited.
    assert fake.calls[2][0] == "act_987/adsets"
    assert fake.calls[2][1]["daily_budget"]
    assert fake.calls[2][1]["bid_strategy"] == "LOWEST_COST_WITHOUT_CAP"


async def test_legacy_awareness_partial_reuses_campaign_and_reviews_corrected_goal(
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover the exact partial shape without another campaign POST."""
    campaign, recommendation = await _create_plan(seeded_db, "awareness")
    existing_meta_campaign_id = "120252033253890323"
    campaign.meta_campaign_id = existing_meta_campaign_id
    campaign.status = "paused"

    # Simulate an older wizard draft that carried Sales delivery values into an
    # Awareness objective, plus the already-created paused campaign in Meta.
    legacy_payload = dict(recommendation.payload)
    legacy_plan = dict(legacy_payload["plan"])
    legacy_plan["billingEvent"] = "OFFSITE_CONVERSIONS"
    legacy_plan["optimizationGoal"] = "OFFSITE_CONVERSIONS"
    legacy_payload["plan"] = legacy_plan
    recommendation.payload = legacy_payload
    recommendation.status = "failed"
    recommendation.execution_result = {
        "kind": meta_ads.META_AD_PLAN_KIND,
        "status": "partial",
        "recommendation_id": recommendation.id,
        "campaign_id": campaign.id,
        "objective": "awareness",
        "meta_objective": "OUTCOME_AWARENESS",
        "created_ids": {"campaign": existing_meta_campaign_id},
        "ads": [],
        "attempted_step": "ad_set",
        "resumed_from_partial": False,
        "failure_status_code": 400,
        "retry_safe": True,
    }

    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    review = await approvals.approve_recommendation(
        seeded_db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=False),
        actor="member-test",
        request_id="req-awareness-resume-review",
        settings=test_settings,
    )
    assert review.requires_confirmation is True
    assert fake.calls == []
    reviewed_delivery = review.diff[0].after
    assert reviewed_delivery["billing_event"] == "IMPRESSIONS"
    assert reviewed_delivery["optimization_goal"] == "REACH"
    assert reviewed_delivery["advantage_audience"] == 0
    assert recommendation.payload["plan"]["billingEvent"] == "IMPRESSIONS"
    assert recommendation.payload["plan"]["optimizationGoal"] == "REACH"

    resumed = await approvals.approve_recommendation(
        seeded_db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=True),
        actor="member-test",
        request_id="req-awareness-resume-confirm",
        settings=test_settings,
    )
    assert resumed.executed is True, resumed.message
    assert recommendation.execution_result["resumed_from_partial"] is True
    assert recommendation.execution_result["created_ids"]["campaign"] == existing_meta_campaign_id
    assert recommendation.execution_result["billing_event"] == "IMPRESSIONS"
    assert recommendation.execution_result["optimization_goal"] == "REACH"
    assert recommendation.execution_result["advantage_audience"] == 0
    assert recommendation.execution_result["graph_api_version"] == test_settings.meta_graph_version
    paths = [path for path, _ in fake.calls]
    assert "act_987/campaigns" not in paths
    assert paths.count("act_987/adsets") == 1
    adset_payload = fake.calls[0][1]
    assert adset_payload["campaign_id"] == existing_meta_campaign_id
    assert adset_payload["billing_event"] == "IMPRESSIONS"
    assert adset_payload["optimization_goal"] == "REACH"
    assert adset_payload["targeting"]["targeting_automation"]["advantage_audience"] == 0


async def test_creative_cta_rejection_resumes_without_recreating_campaign_or_ad_set(
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The known creative 400 is repaired locally, then resumes at the creative.

    The previous attempt created a paused campaign and ad set, but Meta rejected
    the first creative. Phase one must remove the invalid text-valued
    ``link_caption``; phase two may then reuse the recorded IDs but must not POST
    another campaign or ad set. All Graph calls here go to FakeGraph.
    """
    campaign, recommendation = await _create_plan(seeded_db, "awareness")
    meta_campaign_id = "120252033253890323"
    meta_ad_set_id = "120252033617460323"
    campaign.meta_campaign_id = meta_campaign_id
    campaign.status = "paused"

    payload = dict(recommendation.payload)
    plan = dict(payload["plan"])
    plan["callToAction"] = {
        "type": "LEARN_MORE",
        "value": {
            "link": DESTINATION,
            "link_caption": plan["headline"],
        },
    }
    payload["plan"] = plan
    recommendation.payload = payload
    recommendation.status = "failed"
    recommendation.execution_result = {
        "kind": meta_ads.META_AD_PLAN_KIND,
        "status": "partial",
        "recommendation_id": recommendation.id,
        "campaign_id": campaign.id,
        "objective": "awareness",
        "meta_objective": "OUTCOME_AWARENESS",
        "advantage_audience": 0,
        "created_ids": {"campaign": meta_campaign_id, "ad_set": meta_ad_set_id},
        "ads": [],
        "attempted_step": f"creative:{plan['creativeIds'][0]}",
        "resumed_from_partial": True,
        "failure_status_code": 400,
        "retry_safe": True,
    }

    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    review = await approvals.approve_recommendation(
        seeded_db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=False),
        actor="member-test",
        request_id="req-creative-resume-review",
        settings=test_settings,
    )
    assert review.requires_confirmation is True
    assert fake.calls == []
    reviewed_diff_cta = review.diff[0].after["call_to_action"]
    assert reviewed_diff_cta["value"]["link"] == DESTINATION
    assert "link_caption" not in reviewed_diff_cta["value"]
    reviewed_cta = recommendation.payload["plan"]["callToAction"]
    assert reviewed_cta["value"]["link"] == DESTINATION
    assert "link_caption" not in reviewed_cta["value"]

    resumed = await approvals.approve_recommendation(
        seeded_db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=True),
        actor="member-test",
        request_id="req-creative-resume-confirm",
        settings=test_settings,
    )
    assert resumed.executed is True, resumed.message
    assert recommendation.execution_result["resumed_from_partial"] is True
    assert recommendation.execution_result["created_ids"]["campaign"] == meta_campaign_id
    assert recommendation.execution_result["created_ids"]["ad_set"] == meta_ad_set_id

    paths = [path for path, _ in fake.calls]
    assert "act_987/campaigns" not in paths
    assert "act_987/adsets" not in paths
    assert paths == [
        "act_987/adimages",
        "act_987/adcreatives",
        "act_987/ads",
        "act_987/adimages",
        "act_987/adcreatives",
        "act_987/ads",
    ]
    creative_payloads = [body for path, body in fake.calls if path.endswith("/adcreatives")]
    assert len(creative_payloads) == 2
    for body in creative_payloads:
        link_data = body["object_story_spec"]["link_data"]
        assert link_data["link"] == DESTINATION
        assert link_data["call_to_action"]["value"]["link"] == DESTINATION
        assert "link_caption" not in link_data["call_to_action"]["value"]
    assert recommendation.execution_result["advantage_audience"] == 0


async def test_uncertain_partial_meta_failure_is_recorded_and_never_replayed(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, recommendation = await _create_plan(seeded_db)
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "987"
    fake = FakeGraph(fail_path="act_987/adsets", fail_status_code=503)
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=False), settings=test_settings
    )
    first = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=True), settings=test_settings
    )
    assert first.executed is False
    assert recommendation.status == "failed"
    assert recommendation.execution_result["status"] == "unknown"
    assert recommendation.execution_result["retry_safe"] is False
    first_call_count = len(fake.calls)

    await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=False), settings=test_settings
    )
    second = await approvals.approve_recommendation(
        seeded_db, recommendation.id, ApproveRecommendationRequest(confirm=True), settings=test_settings
    )
    assert second.executed is False
    assert len(fake.calls) == first_call_count
    assert "blocked" in second.message.lower() or "partial" in second.message.lower()


async def test_reference_prompt_preserves_verbatim_text_and_layout_constraints() -> None:
    prompt = compose_reference_faithful_prompt(
        "Change only the product color to blue.",
        {
            "visible_text": ["SUMMER SALE", "Shop now"],
            "elements": [
                {
                    "element": "headline",
                    "relative_position": "upper-left",
                    "relative_size": "large",
                    "visual_attributes": "bold white sans-serif",
                    "exact_text": "SUMMER SALE",
                    "layer_order": 1,
                }
            ],
        },
    )
    assert "layout" in prompt.lower()
    assert "SUMMER SALE" in prompt
    assert "Shop now" in prompt
    assert "Only change items explicitly requested" in prompt
    assert "distinct generated final image" in prompt


async def test_base64_image_storage_is_file_checked_and_served_without_network(
    tmp_path: Any, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api import media_assets
    from app.core.storage import persist_image_asset

    test_settings.upload_dir = str(tmp_path)
    test_settings.public_asset_base_url = "https://assets.example.com"
    content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 12
    encoded = base64.b64encode(content).decode()
    url, _error = await persist_image_asset(
        test_settings, asset_url=f"data:image/png;base64,{encoded}"
    )
    assert url is not None and not _error
    assert url.startswith("https://assets.example.com/api/v1/media/public-assets/asset_")
    filename = url.rsplit("/", 1)[-1]
    monkeypatch.setattr(media_assets, "get_settings", lambda: test_settings)
    response = await media_assets.get_public_asset(filename)
    assert response.media_type == "image/png"
    assert Path(response.path).read_bytes() == content
    with pytest.raises(HTTPException) as caught:
        await media_assets.get_public_asset("../secret.png")
    assert caught.value.status_code == 404


async def test_reference_url_bytes_feed_image_edit_and_url_query_is_not_persisted(
    tmp_path: Any,
    seeded_db: AsyncSession,
    tool_ctx: Any,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.tools import media
    from app.core.tools import settings as connector_settings

    content = b"\x89PNG\r\n\x1a\n" + b"\x11" * 16
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    test_settings.upload_dir = str(upload_root)
    test_settings.website_max_bytes = 1024
    tool_ctx.settings = test_settings
    requested: list[tuple[str, str]] = []

    class FakeStream:
        headers = {"content-type": "image/png", "content-length": str(len(content))}

        async def __aenter__(self) -> "FakeStream":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield content

    class FakeHttpClient:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs.get("trust_env") is False

        async def __aenter__(self) -> "FakeHttpClient":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeStream:
            requested.append((method, url))
            return FakeStream()

    monkeypatch.setattr(media, "resolve_and_check", lambda url, **kwargs: SimpleNamespace(url=url))
    monkeypatch.setattr(media.httpx, "AsyncClient", FakeHttpClient)

    async def fake_load_connector_settings(_db: Any, _business_id: str, _settings: Any) -> Any:
        return SimpleNamespace(
            image_provider="openai",
            image_api_key="offline-image-key",
            image_model="offline-edit-model",
            image_base_url="https://image-provider.invalid/v1",
        )

    monkeypatch.setattr(connector_settings, "load_connector_settings", fake_load_connector_settings)
    calls: list[dict[str, Any]] = []

    async def fake_generate(prompt: str, aspect_ratio: str, **kwargs: Any) -> list[GeneratedAssetRecord]:
        calls.append({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
        return [
            GeneratedAssetRecord(
                id="asset_from_url",
                url="https://assets.example.com/generated-from-url.png",
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider="openai",
                model=str(kwargs["model"]),
            )
        ]

    monkeypatch.setattr(media, "_generate_openai_image", fake_generate)
    signed_url = "https://images.example.com/ref.png?signature=do-not-store"
    output = await media._handle_generate_image(
        GenerateImageInput(
            prompt="Preserve the supplied design and change only the requested detail.",
            reference_image_url=signed_url,
            aspect_ratio="4:5",
        ),
        tool_ctx,
    )
    assert output.status.value == "ok"
    assert requested == [("GET", signed_url)]
    assert len(calls) == 1
    assert calls[0]["reference_image"] == ("image/png", content)
    assert calls[0]["aspect_ratio"] == "4:5"
    assert "REFERENCE-MATCH MODE" in calls[0]["prompt"]
    assert "do-not-store" not in calls[0]["prompt"]
    private_files = list(upload_root.glob("reference_*.png"))
    assert len(private_files) == 1
    assert private_files[0].read_bytes() == content
    assert "signature" not in private_files[0].name
    stored = await seeded_db.get(Creative, "asset_from_url")
    assert stored is not None
    assert stored.url == "https://assets.example.com/generated-from-url.png"
    assert stored.reference_analysis_id is None


async def test_reference_analysis_id_restores_uploaded_image_and_fails_closed_when_missing(
    tmp_path: Any,
    seeded_db: AsyncSession,
    tool_ctx: Any,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.tools import media
    from app.core.tools import settings as connector_settings

    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    test_settings.upload_dir = str(upload_root)
    test_settings.website_max_bytes = 1024
    tool_ctx.settings = test_settings
    reference_bytes = b"\x89PNG\r\n\x1a\n" + b"\x22" * 16
    (upload_root / "reference-original.png").write_bytes(reference_bytes)
    style = {"visible_text": ["Exact original headline"], "layout_pattern": "centered product card"}
    seeded_db.add(
        BrandMemoryItem(
            id="ref_uploaded",
            business_id="biz_test",
            kind="reference_analysis",
            text="Uploaded image reference",
            metadata_={"source": "upload:reference-original.png", "style": style},
        )
    )
    await seeded_db.flush()

    async def fake_load_connector_settings(_db: Any, _business_id: str, _settings: Any) -> Any:
        return SimpleNamespace(
            image_provider="openai",
            image_api_key="offline-image-key",
            image_model="offline-edit-model",
            image_base_url="https://image-provider.invalid/v1",
        )

    monkeypatch.setattr(connector_settings, "load_connector_settings", fake_load_connector_settings)
    calls: list[dict[str, Any]] = []

    async def fake_generate(prompt: str, aspect_ratio: str, **kwargs: Any) -> list[GeneratedAssetRecord]:
        calls.append({"prompt": prompt, "aspect_ratio": aspect_ratio, **kwargs})
        return [
            GeneratedAssetRecord(
                id="asset_from_analysis",
                url="https://assets.example.com/generated-from-analysis.png",
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider="openai",
                model=str(kwargs["model"]),
            )
        ]

    monkeypatch.setattr(media, "_generate_openai_image", fake_generate)
    output = await media._handle_generate_image(
        GenerateImageInput(
            prompt="Change only the background color.",
            reference_analysis_id="ref_uploaded",
        ),
        tool_ctx,
    )
    assert output.status.value == "ok"
    assert len(calls) == 1
    assert calls[0]["reference_image"] == ("image/png", reference_bytes)
    assert "Exact original headline" in calls[0]["prompt"]
    assert "centered product card" in calls[0]["prompt"]
    stored = await seeded_db.get(Creative, "asset_from_analysis")
    assert stored is not None
    assert stored.reference_analysis_id == "ref_uploaded"
    assert stored.prompt == calls[0]["prompt"]
    assert stored.url == "https://assets.example.com/generated-from-analysis.png"

    missing = await media._handle_generate_image(
        GenerateImageInput(prompt="Do not fall back to text-only.", reference_analysis_id="ref_missing"),
        tool_ctx,
    )
    assert missing.status.value == "unavailable"
    assert "not available" in (missing.unavailable_reason or "")
    assert len(calls) == 1


async def test_public_asset_requires_https_origin_and_valid_raster_type_and_size(
    tmp_path: Any, test_settings: Any
) -> None:
    from app.core.storage import persist_image_asset

    test_settings.upload_dir = str(tmp_path)
    content = b"\x89PNG\r\n\x1a\n" + b"\x33" * 16
    encoded = base64.b64encode(content).decode()
    data_url = f"data:image/png;base64,{encoded}"

    test_settings.public_asset_base_url = None
    url, error = await persist_image_asset(test_settings, asset_url=data_url)
    assert url is None and error and "PUBLIC_ASSET_BASE_URL" in error
    assert not (tmp_path / "public_assets").exists()

    test_settings.public_asset_base_url = "http://assets.example.com"
    url, error = await persist_image_asset(test_settings, asset_url=data_url)
    assert url is None and error and "public HTTPS" in error

    test_settings.public_asset_base_url = "https://assets.example.com"
    url, error = await persist_image_asset(
        test_settings, asset_url="data:image/svg+xml;base64,PHN2Zz4="
    )
    # SVG only ever reaches storage as a dev-only mock placeholder; the public
    # asset route Meta reads accepts raster formats only.
    assert url is None and error and "PNG, JPEG, or WebP" in error

    test_settings.upload_max_bytes = len(content) - 1
    url, error = await persist_image_asset(test_settings, asset_url=data_url)
    assert url is None and error and "exceeds" in error
    assert not (tmp_path / "public_assets").exists()


# ================================================= activation (Phase 2.3) ==


async def _executed_plan(
    db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
    objective: str = "sales",
) -> tuple[Campaign, Recommendation, FakeGraph]:
    """Create + execute a plan so ``created_ids`` is populated, all PAUSED."""
    campaign, recommendation = await _create_plan(db, objective)
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "act_987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)
    # Phase 1 (diff preview) is mandatory before phase 2 can execute.
    await approvals.approve_recommendation(
        db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=False),
        actor="member-test",
        request_id="req-activation-preview",
        settings=test_settings,
    )
    await approvals.approve_recommendation(
        db,
        recommendation.id,
        ApproveRecommendationRequest(confirm=True),
        actor="member-test",
        request_id="req-activation-setup",
        settings=test_settings,
    )
    return campaign, recommendation, fake


@pytest.mark.anyio
async def test_activation_switches_campaign_adset_and_ads_active_parent_first(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Meta needs all three levels ACTIVE, in parent-first order."""
    campaign, recommendation, fake = await _executed_plan(
        seeded_db, test_settings, monkeypatch
    )
    assert campaign.status == "paused"  # creation is still PAUSED
    mark = len(fake.calls)

    out = await meta_ads.activate_meta_ad_plan(
        seeded_db, recommendation, actor="member-test", settings=test_settings, graph=fake
    )

    assert out.status == "ok", out.unavailable_reason
    activation_calls = fake.calls[mark:]
    assert [path for path, _ in activation_calls] == [
        "meta_campaign_1",
        "meta_adset_1",
        "meta_ad_1",
        "meta_ad_2",
    ]
    assert all(payload == {"status": "ACTIVE"} for _, payload in activation_calls)
    assert out.activated == ["campaign", "ad_set", "ad:meta_ad_1", "ad:meta_ad_2"]

    # A real DB write, so a CHECK-constraint violation would surface here.
    assert campaign.status == "active"
    await seeded_db.flush()
    assert recommendation.execution_result["status"] == "active"


@pytest.mark.anyio
async def test_activation_resumes_without_reposting_already_active_objects(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, recommendation, fake = await _executed_plan(
        seeded_db, test_settings, monkeypatch
    )
    first = await meta_ads.activate_meta_ad_plan(
        seeded_db, recommendation, actor="member-test", settings=test_settings, graph=fake
    )
    assert first.status == "ok"
    mark = len(fake.calls)

    second = await meta_ads.activate_meta_ad_plan(
        seeded_db, recommendation, actor="member-test", settings=test_settings, graph=fake
    )

    assert second.status == "ok"
    assert "no Meta write was needed" in second.detail
    assert fake.calls[mark:] == []  # idempotent: nothing re-POSTed
    assert campaign.status == "active"


@pytest.mark.anyio
async def test_activation_stops_on_meta_error_and_leaves_campaign_paused(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-activated campaign must not be reported as live."""
    campaign, recommendation, _ = await _executed_plan(
        seeded_db, test_settings, monkeypatch
    )
    broken = FakeGraph(fail_path="meta_adset_1")

    out = await meta_ads.activate_meta_ad_plan(
        seeded_db, recommendation, actor="member-test", settings=test_settings, graph=broken
    )

    assert out.status == "unavailable"
    assert "stopped at ad_set" in out.detail
    assert out.activated == ["campaign"]  # the campaign did go through
    assert campaign.status == "paused"  # ...but delivery is not live
    assert recommendation.execution_result["status"] == "partial_active"

    # Resuming with a healthy client finishes the remaining objects only.
    healthy = FakeGraph()
    resumed = await meta_ads.activate_meta_ad_plan(
        seeded_db, recommendation, actor="member-test", settings=test_settings, graph=healthy
    )
    assert resumed.status == "ok", resumed.unavailable_reason
    assert [path for path, _ in healthy.calls] == [
        "meta_adset_1",
        "meta_ad_1",
        "meta_ad_2",
    ]
    assert campaign.status == "active"


@pytest.mark.anyio
async def test_activation_requires_an_explicit_human_actor(
    seeded_db: AsyncSession, test_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, recommendation, fake = await _executed_plan(
        seeded_db, test_settings, monkeypatch
    )
    mark = len(fake.calls)

    for actor in ("", "system"):
        out = await meta_ads.activate_meta_ad_plan(
            seeded_db, recommendation, actor=actor, settings=test_settings, graph=fake
        )
        assert out.status == "unavailable"
        assert "explicit user confirmation" in (out.unavailable_reason or "")

    assert fake.calls[mark:] == []  # no Meta write without a human
    assert campaign.status == "paused"


@pytest.mark.anyio
async def test_activation_refuses_when_no_meta_objects_were_created(
    seeded_db: AsyncSession, test_settings: Any
) -> None:
    """A draft that never reached Meta has nothing to switch on."""
    campaign, recommendation = await _create_plan(seeded_db, "sales")
    recommendation.execution_result = {"created_ids": {}}
    await seeded_db.flush()

    out = await meta_ads.activate_meta_ad_plan(
        seeded_db, recommendation, actor="member-test", settings=test_settings
    )

    assert out.status == "unavailable"
    assert "No Meta objects were recorded" in out.detail
    assert campaign.status != "active"


@pytest.mark.anyio
async def test_activation_endpoint_is_two_phase_and_commits_active_status(
    api_client: Any,
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP surface: preview first, then execute, then a committed row."""
    from app.api.deps import WORKSPACE_HEADER

    campaign, recommendation, fake = await _executed_plan(
        seeded_db, test_settings, monkeypatch
    )
    headers = {WORKSPACE_HEADER: campaign.business_id}
    url = f"/api/v1/recommendations/{recommendation.id}/activate"

    preview = await api_client.post(url, json={"confirm": False}, headers=headers)
    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["executed"] is False
    assert body["requiresConfirmation"] is True
    assert body["targets"] == ["campaign", "ad_set", "ad:meta_ad_1", "ad:meta_ad_2"]
    assert fake.calls == fake.calls  # no new calls in phase 1
    await seeded_db.refresh(campaign)
    assert campaign.status == "paused", "phase 1 must not start delivery"

    mark = len(fake.calls)
    confirm = await api_client.post(url, json={"confirm": True}, headers=headers)
    assert confirm.status_code == 200, confirm.text
    out = confirm.json()
    assert out["executed"] is True, out
    assert out["activated"] == ["campaign", "ad_set", "ad:meta_ad_1", "ad:meta_ad_2"]
    assert [path for path, _ in fake.calls[mark:]] == [
        "meta_campaign_1",
        "meta_adset_1",
        "meta_ad_1",
        "meta_ad_2",
    ]

    # The route runs in its own session, so a refresh proves it really committed.
    await seeded_db.refresh(campaign)
    assert campaign.status == "active"


@pytest.mark.anyio
async def test_activation_endpoint_404s_for_another_tenants_recommendation(
    api_client: Any,
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace isolation: another tenant's plan must not be activatable."""
    from app.api.deps import WORKSPACE_HEADER

    _, recommendation, _ = await _executed_plan(seeded_db, test_settings, monkeypatch)
    url = f"/api/v1/recommendations/{recommendation.id}/activate"

    resp = await api_client.post(
        url, json={"confirm": True}, headers={WORKSPACE_HEADER: "biz_does_not_exist"}
    )
    assert resp.status_code in (403, 404), resp.text


async def test_audience_age_and_gender_reach_the_ad_set(
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Age and gender must survive the trip to Meta, not just the country list.

    The wizard's audience step collects all of them; only `geo_locations` used to
    be forwarded, so every campaign went out with Meta's default age/gender
    spread. This pins the passthrough so a future key whitelist cannot silently
    drop them again.
    """
    campaign, recommendation = await _create_plan(seeded_db, "traffic")

    audience = {
        "age_min": 25,
        "age_max": 44,
        # Meta requires a collection; all genders are represented by omitting the key.
        "genders": [1],
        "geo_locations": {"countries": ["EG"]},
        # What Targeting Search makes possible: interest IDs, not names.
        "flexible_spec": [
            {"interests": [{"id": "6003070872673", "name": "Golf"}]}
        ],
    }
    # Reassign a fresh dict: a plain JSON column is not mutation-tracked, so an
    # in-place edit would compare equal at flush time and never be written.
    payload = dict(recommendation.payload)
    plan = dict(payload["plan"])
    plan["targeting"] = audience
    payload["plan"] = plan
    recommendation.payload = payload
    await seeded_db.commit()

    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "act_987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    for confirm in (False, True):
        await approvals.approve_recommendation(
            seeded_db,
            recommendation.id,
            ApproveRecommendationRequest(confirm=confirm),
            actor="member-test",
            request_id=f"req-audience-{confirm}",
            settings=test_settings,
        )

    adset_calls = [(p, payload) for p, payload in fake.calls if p.endswith("/adsets")]
    assert len(adset_calls) == 1, [p for p, _ in fake.calls]
    sent = adset_calls[0][1]["targeting"]
    assert sent["age_min"] == 25, sent
    assert sent["age_max"] == 44, sent
    assert sent["targeting_automation"]["advantage_audience"] == 0, sent
    assert "age_range" not in sent, sent
    assert sent["genders"] == [1], sent
    assert sent["geo_locations"] == {"countries": ["EG"]}, sent
    assert sent["flexible_spec"] == [
        {"interests": [{"id": "6003070872673", "name": "Golf"}]}
    ], sent


async def test_ad_images_are_uploaded_as_bytes_never_as_url(
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta's adimages edge accepts `bytes` or `copy_from` — there is no `url`.

    Posting an undocumented field would be rejected live, and only after the
    campaign and ad set already existed. This pins both the parameter name and
    that the payload really carries the image.
    """
    campaign, recommendation = await _create_plan(seeded_db, "traffic")
    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "act_987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    for confirm in (False, True):
        result = await approvals.approve_recommendation(
            seeded_db,
            recommendation.id,
            ApproveRecommendationRequest(confirm=confirm),
            actor="member-test",
            request_id=f"req-bytes-{confirm}",
            settings=test_settings,
        )
    assert result.executed is True, result.execution_result

    uploads = [p for p, _ in fake.calls if p.endswith("/adimages")]
    assert len(uploads) == 2, [p for p, _ in fake.calls]

    for path, payload in fake.calls:
        if not path.endswith("/adimages"):
            continue
        assert "url" not in payload, payload
        raw = base64.b64decode(payload["bytes"])
        assert raw == base64.b64decode(PNG_B64), "uploaded bytes are not the creative"


async def test_an_unreadable_creative_image_stops_before_uploading(
    seeded_db: AsyncSession,
    test_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A creative we cannot read must fail cleanly, not upload nothing silently."""
    campaign, recommendation = await _create_plan(seeded_db, "traffic")
    # Swap one creative for a URL that cannot be fetched offline.
    payload = dict(recommendation.payload)
    plan = dict(payload["plan"])
    plan["creative_ids"] = ["cr_a"]
    payload["plan"] = plan
    recommendation.payload = payload

    row = await seeded_db.get(Creative, "cr_a")
    row.url = "https://unreachable.invalid/final.png"
    await seeded_db.commit()

    test_settings.meta_access_token = "offline-test-token"
    test_settings.meta_ad_account_id = "act_987"
    fake = FakeGraph()
    monkeypatch.setattr(meta_ads, "HttpMetaGraphClient", lambda **kwargs: fake)

    for confirm in (False, True):
        result = await approvals.approve_recommendation(
            seeded_db,
            recommendation.id,
            ApproveRecommendationRequest(confirm=confirm),
            actor="member-test",
            request_id=f"req-unreadable-{confirm}",
            settings=test_settings,
        )

    assert result.executed is False
    assert [p for p, _ in fake.calls if p.endswith("/adimages")] == []
    # Assert what was persisted, which is what an operator would inspect.
    await seeded_db.refresh(recommendation)
    stored = recommendation.execution_result or {}
    assert "could not be read" in str(stored.get("failure")), stored
