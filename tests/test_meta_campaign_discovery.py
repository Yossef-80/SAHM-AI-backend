"""Read-only discovery of Meta campaigns that were not created in Sahm."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from pydantic import ValidationError

from app.api import integrations as integrations_api
from app.api.deps import WORKSPACE_HEADER
from app.core.security import TokenCipher
from app.core.tools.base import ToolContext
from app.core.tools.meta_read import (
    _handle_get_campaign_insights,
    _handle_list_meta_campaigns,
)
from app.db.models import Integration
from app.schemas.common import ToolStatus
from app.schemas.tools import (
    GetCampaignInsightsInput,
    ListMetaCampaignsInput,
    ListMetaCampaignsOutput,
    MetaCampaignSummary,
)

pytestmark = pytest.mark.anyio


async def test_external_campaign_id_rejects_non_numeric_path_values() -> None:
    with pytest.raises(ValidationError):
        GetCampaignInsightsInput(campaign_id="238001", meta_campaign_id="238001/insights")


async def _connect_meta(ctx: ToolContext, *, account_id: str = "act_123") -> str:
    settings = ctx.require_settings()
    token = "offline-test-meta-grant"
    ctx.db.add(
        Integration(
            id="int_campaign_discovery_test",
            business_id=ctx.business_id,
            provider="meta",
            account_id=account_id,
            account_name="Read-only test account",
            status="connected",
            access_token_enc=TokenCipher(settings.token_encryption_key).encrypt(token),
            scopes=["ads_read"],
            last_synced_at=datetime.now(timezone.utc),
        )
    )
    await ctx.db.commit()
    return token


async def test_list_meta_campaigns_finds_external_campaigns_and_never_exposes_next_url(
    tool_ctx: ToolContext,
    respx_mock,
) -> None:
    token = await _connect_meta(tool_ctx)
    route = respx_mock.get(path__regex=r"/act_123/campaigns$").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "238001",
                        "name": "Summer coffee launch",
                        "objective": "OUTCOME_SALES",
                        "status": "PAUSED",
                        "effective_status": "CAMPAIGN_PAUSED",
                        "created_time": "2025-04-01T10:00:00+0000",
                        "start_time": "2025-04-05T00:00:00+0000",
                        "stop_time": "2025-05-01T00:00:00+0000",
                    },
                    {"id": "238002", "name": "Winter sale", "status": "ACTIVE"},
                    {"name": "Malformed row without an ID"},
                ],
                "paging": {
                    "cursors": {"after": "opaque-page-2"},
                    "next": f"https://graph.facebook.com/v20.0/act_123/campaigns?access_token={token}",
                },
            },
        )
    )

    output = await _handle_list_meta_campaigns(
        ListMetaCampaignsInput(search="summer coffee", limit=10),
        tool_ctx,
    )

    assert output.status == ToolStatus.PARTIAL
    assert output.account_id == "123"
    assert output.has_more is True
    assert output.next_cursor == "opaque-page-2"
    assert [
        (campaign.meta_campaign_id, campaign.name) for campaign in output.campaigns
    ] == [("238001", "Summer coffee launch")]
    assert output.campaigns[0].effective_status == "CAMPAIGN_PAUSED"
    assert output.campaigns[0].start_time == "2025-04-05T00:00:00+0000"
    assert route.called
    params = route.calls.last.request.url.params
    assert params["limit"] == "10"
    assert "effective_status" in params["fields"]
    # The paging URL contains the token; only the cursor may leave the tool.
    assert token not in repr(output.model_dump(mode="json"))
    assert "paging.next" not in repr(output.model_dump(mode="json"))


async def test_meta_campaigns_cursor_is_passed_without_creating_a_local_campaign(
    tool_ctx: ToolContext,
    respx_mock,
) -> None:
    await _connect_meta(tool_ctx)
    route = respx_mock.get(path__regex=r"/act_123/campaigns$").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "238003", "name": "Older campaign"}]},
        )
    )

    output = await _handle_list_meta_campaigns(
        ListMetaCampaignsInput(after="opaque-page-2", limit=25),
        tool_ctx,
    )

    assert output.status == ToolStatus.OK
    assert [campaign.meta_campaign_id for campaign in output.campaigns] == ["238003"]
    assert route.calls.last.request.url.params["after"] == "opaque-page-2"
    # Campaign discovery is not a sync/import operation.
    from sqlalchemy import select
    from app.db.models import Campaign

    campaigns = (await tool_ctx.db.execute(select(Campaign))).scalars().all()
    assert campaigns == []


async def test_list_meta_campaigns_is_unavailable_without_connection_and_does_not_call_meta(
    tool_ctx: ToolContext,
    respx_mock,
) -> None:
    route = respx_mock.get(path__regex=r"/act_.*/campaigns$").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    output = await _handle_list_meta_campaigns(ListMetaCampaignsInput(), tool_ctx)

    assert output.status == ToolStatus.UNAVAILABLE
    assert output.campaigns == []
    assert "not connected" in (output.unavailable_reason or "").lower()
    assert not route.called


async def test_meta_api_errors_are_sanitized_for_campaign_discovery(
    tool_ctx: ToolContext,
    respx_mock,
) -> None:
    token = await _connect_meta(tool_ctx)
    respx_mock.get(path__regex=r"/act_123/campaigns$").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"code": 190, "message": f"Bad token {token}"}},
        )
    )

    output = await _handle_list_meta_campaigns(ListMetaCampaignsInput(), tool_ctx)

    assert output.status == ToolStatus.UNAVAILABLE
    assert "Meta campaigns request failed" in (output.unavailable_reason or "")
    assert token not in repr(output.model_dump(mode="json"))


async def test_external_campaign_insights_error_never_exposes_access_token(
    tool_ctx: ToolContext,
    respx_mock,
) -> None:
    token = await _connect_meta(tool_ctx)
    respx_mock.get(path__regex=r"/238001/insights$").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"code": 190, "message": f"Bad token {token}"}},
        )
    )

    output = await _handle_get_campaign_insights(
        GetCampaignInsightsInput(
            campaign_id="238001",
            meta_campaign_id="238001",
        ),
        tool_ctx,
    )

    assert output.status == ToolStatus.UNAVAILABLE
    assert "Meta insights request failed" in (output.unavailable_reason or "")
    assert token not in repr(output.model_dump(mode="json"))


async def test_campaign_list_api_returns_meta_rows_as_read_only_page(
    api_client,
    monkeypatch,
) -> None:
    captured = {}

    async def fake_list_meta_campaigns(db, business_id, settings, payload):
        captured["business_id"] = business_id
        captured["search"] = payload.search
        captured["limit"] = payload.limit
        captured["after"] = payload.after
        return ListMetaCampaignsOutput(
            status=ToolStatus.PARTIAL,
            source="meta_marketing_api",
            account_id="123",
            campaigns=[
                MetaCampaignSummary(
                    meta_campaign_id="238001",
                    name="Winter coffee campaign",
                    objective="OUTCOME_SALES",
                    status="PAUSED",
                    effective_status="CAMPAIGN_PAUSED",
                )
            ],
            has_more=True,
            next_cursor="opaque-page-2",
        )

    monkeypatch.setattr(
        integrations_api, "list_meta_campaigns_for_workspace", fake_list_meta_campaigns
    )
    response = await api_client.get(
        "/api/v1/integrations/meta/campaigns",
        params={"q": "winter", "limit": 25, "after": "opaque-page-1"},
        headers={WORKSPACE_HEADER: "biz_test"},
    )

    assert response.status_code == 200, response.text
    assert captured == {
        "business_id": "biz_test",
        "search": "winter",
        "limit": 25,
        "after": "opaque-page-1",
    }
    assert response.json()["status"] == "partial"
    assert response.json()["accountId"] == "123"
    assert response.json()["campaigns"][0]["metaCampaignId"] == "238001"
    assert response.json()["campaigns"][0]["effectiveStatus"] == "CAMPAIGN_PAUSED"
    assert response.json()["nextCursor"] == "opaque-page-2"
