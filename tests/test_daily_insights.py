"""Daily performance series for campaign insights (Phase 1.1).

These exercise the real ``_handle_get_campaign_insights`` against a mocked Meta
response. They exist because the frontend contract (``MetricPoint``) needs one
row per day, while Meta was previously asked for a single aggregate row.

The ratio test is the important one: summing ctr/cpc/cpm across days is wrong,
so they must be recomputed from the summed additive base.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core.security import TokenCipher
from app.core.tools.base import ToolContext
from app.core.tools.meta_read import _derive_ratio_metrics, _handle_get_campaign_insights
from app.db.models import Integration
from app.schemas.tools import GetCampaignInsightsInput

pytestmark = pytest.mark.anyio

META_ID = "23800000000000"

#: Three days where the daily ctr differs, so a naive sum would be visibly wrong.
DAILY_ROWS = [
    {
        "date_start": "2026-10-01",
        "date_stop": "2026-10-01",
        "spend": "100.0",
        "impressions": "1000",
        "clicks": "10",
        "conversions": "1",
        "reach": "900",
        "ctr": "1.0",
        "cpc": "10.0",
        "cpm": "100.0",
        "cost_per_conversion": "100.0",
        "frequency": "1.1111",
    },
    {
        "date_start": "2026-10-02",
        "date_stop": "2026-10-02",
        "spend": "200.0",
        "impressions": "4000",
        "clicks": "80",
        "conversions": "4",
        "reach": "3500",
        "ctr": "2.0",
        "cpc": "2.5",
        "cpm": "50.0",
        "cost_per_conversion": "50.0",
        "frequency": "1.1428",
    },
    {
        "date_start": "2026-10-03",
        "date_stop": "2026-10-03",
        "spend": "300.0",
        "impressions": "5000",
        "clicks": "110",
        "conversions": "5",
        "reach": "4200",
        "ctr": "2.2",
        "cpc": "2.7273",
        "cpm": "60.0",
        "cost_per_conversion": "60.0",
        "frequency": "1.1905",
    },
]


async def _connect_meta(ctx: ToolContext) -> None:
    """Seed a connected Meta grant so ``_get_meta_token`` resolves."""
    settings = ctx.require_settings()
    ctx.db.add(
        Integration(
            id="int_insights_test",
            business_id=ctx.business_id,
            provider="meta",
            account_id="act_123",
            account_name="Test Ads",
            status="connected",
            access_token_enc=TokenCipher(settings.token_encryption_key).encrypt(
                "offline-test-meta-grant"
            ),
            scopes=["ads_read"],
            last_synced_at=datetime.now(timezone.utc),
        )
    )
    await ctx.db.commit()


def _mock_meta(respx_mock, rows):
    """Intercept the Meta insights call and record the outgoing params.

    The regex must tolerate the query string, so it anchors on the path only.
    """
    import httpx

    route = respx_mock.get(path__regex=r"/insights$").mock(
        return_value=httpx.Response(200, json={"data": rows})
    )
    return route


async def test_daily_series_has_one_row_per_day(tool_ctx: ToolContext, respx_mock) -> None:
    await _connect_meta(tool_ctx)
    _mock_meta(respx_mock, DAILY_ROWS)

    out = await _handle_get_campaign_insights(
        GetCampaignInsightsInput(campaign_id="cmp_x", meta_campaign_id=META_ID),
        tool_ctx,
    )

    assert out.status == "ok", out.reason
    assert [d.date.isoformat() for d in out.daily] == [
        "2026-10-01",
        "2026-10-02",
        "2026-10-03",
    ]
    assert [d.spend for d in out.daily] == [100.0, 200.0, 300.0]
    assert [d.impressions for d in out.daily] == [1000.0, 4000.0, 5000.0]
    assert [d.clicks for d in out.daily] == [10.0, 80.0, 110.0]
    assert [d.conversions for d in out.daily] == [1.0, 4.0, 5.0]


async def test_time_increment_is_requested(tool_ctx: ToolContext, respx_mock) -> None:
    """Without time_increment=1 Meta returns one aggregate row and no trend."""
    await _connect_meta(tool_ctx)
    route = _mock_meta(respx_mock, DAILY_ROWS)

    await _handle_get_campaign_insights(
        GetCampaignInsightsInput(campaign_id="cmp_x", meta_campaign_id=META_ID),
        tool_ctx,
    )

    sent = route.calls.last.request
    # GET params live in the URL, not the body.
    assert "time_increment=1" in str(sent.url)


async def test_ratio_metrics_are_derived_not_summed(
    tool_ctx: ToolContext, respx_mock
) -> None:
    """ctr/cpc/cpm/cost_per_conversion must not be summed across days."""
    await _connect_meta(tool_ctx)
    _mock_meta(respx_mock, DAILY_ROWS)

    out = await _handle_get_campaign_insights(
        GetCampaignInsightsInput(campaign_id="cmp_x", meta_campaign_id=META_ID),
        tool_ctx,
    )

    # Additive metrics really are summed.
    assert out.metrics["spend"] == 600.0
    assert out.metrics["impressions"] == 10000.0
    assert out.metrics["clicks"] == 200.0
    assert out.metrics["conversions"] == 10.0

    # A naive sum of the daily ratios would be 1.0 + 2.0 + 2.2 = 5.2 for ctr.
    assert out.metrics["ctr"] != pytest.approx(5.2)
    assert out.metrics["ctr"] == pytest.approx(200 / 10000 * 100, abs=1e-3)  # 2.0
    assert out.metrics["cpc"] == pytest.approx(600 / 200, abs=1e-3)  # 3.0
    assert out.metrics["cpm"] == pytest.approx(600 / 10000 * 1000, abs=1e-3)  # 60.0
    assert out.metrics["cost_per_conversion"] == pytest.approx(600 / 10, abs=1e-3)  # 60.0


async def test_no_division_by_zero_when_a_day_has_no_traffic(
    tool_ctx: ToolContext, respx_mock
) -> None:
    await _connect_meta(tool_ctx)
    _mock_meta(
        respx_mock,
        [{"date_start": "2026-10-01", "date_stop": "2026-10-01", "spend": "0"}],
    )

    out = await _handle_get_campaign_insights(
        GetCampaignInsightsInput(campaign_id="cmp_x", meta_campaign_id=META_ID),
        tool_ctx,
    )

    assert out.status == "ok"
    assert len(out.daily) == 1
    assert "ctr" not in out.metrics  # no impressions -> ratio withheld, not zero
    assert "cpc" not in out.metrics


def test_derive_ratio_metrics_is_pure() -> None:
    """Unit-level guard on the helper itself."""
    derived = _derive_ratio_metrics(
        {"spend": 100.0, "impressions": 0.0, "clicks": 0.0, "conversions": 0.0}
    )
    assert derived == {}

    derived = _derive_ratio_metrics(
        {"spend": 50.0, "impressions": 500.0, "clicks": 5.0, "conversions": 1.0, "reach": 400.0}
    )
    assert derived["ctr"] == pytest.approx(1.0)
    assert derived["cpc"] == pytest.approx(10.0)
    assert derived["cpm"] == pytest.approx(100.0)
    assert derived["cost_per_conversion"] == pytest.approx(50.0)
    assert derived["frequency"] == pytest.approx(1.25)
