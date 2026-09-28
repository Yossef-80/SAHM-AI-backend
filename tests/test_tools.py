"""Required test 5: an unavailable source returns status="unavailable" with a
reason, and the agent's output states the limitation instead of inventing data.
"""

from __future__ import annotations

import json

import pytest

from app.core.agents.personas import KARIM, PERSONAS, RESEARCHER, SALMA
from app.core.agents.runtime import run_agent
from app.core.llm.mock_client import MockClient, sample_from_schema
from app.core.errors import ToolUnavailable
from app.core.tools.base import ToolContext
from app.core.tools.meta_read import (
    _handle_get_breakdowns,
    _handle_get_campaign_insights,
)
from app.core.tools.registry import register_all
from app.schemas.tools import (
    GetBreakdownsInput,
    GetCampaignInsightsInput,
    GetReviewsInput,
    WebSearchInput,
)
from app.schemas.common import ToolStatus


@pytest.fixture(autouse=True)
def _registry():
    register_all()
    yield


async def test_campaign_insights_returns_unavailable_without_meta(tool_ctx: ToolContext) -> None:
    """No stored Meta token -> unavailable, never fabricated metrics."""
    from app.db.models import Campaign

    tool_ctx.db.add(
        Campaign(
            id="cmp_1",
            business_id="biz_test",
            name="Live campaign",
            status="active",
            meta_campaign_id="meta_cmp_1",
        )
    )
    await tool_ctx.db.commit()

    payload = GetCampaignInsightsInput(campaign_id="cmp_1")
    result = await _handle_get_campaign_insights(payload, tool_ctx)
    assert result.status == ToolStatus.UNAVAILABLE
    assert result.metrics == {}
    assert result.series == []
    assert result.unavailable_reason
    assert "not connected" in result.unavailable_reason.lower()
    assert result.campaign_id == "cmp_1"


async def test_breakdowns_returns_unavailable_without_meta(tool_ctx: ToolContext) -> None:
    result = await _handle_get_breakdowns(
        GetBreakdownsInput(campaign_id="cmp_1", by="audience"), tool_ctx
    )
    assert result.status == ToolStatus.UNAVAILABLE
    assert result.rows == []
    assert result.unavailable_reason


async def test_breakdowns_rejects_an_unsupported_dimension(tool_ctx: ToolContext) -> None:
    result = await _handle_get_breakdowns(
        GetBreakdownsInput(campaign_id="cmp_1", by="zodiac_sign"), tool_ctx
    )
    assert result.status == ToolStatus.UNAVAILABLE
    assert "not supported" in (result.unavailable_reason or "")


async def test_insights_reports_an_unlinked_campaign(tool_ctx: ToolContext) -> None:
    """A campaign that exists but has no Meta id cannot be read."""
    from app.db.models import Campaign

    tool_ctx.db.add(
        Campaign(
            id="cmp_unlinked",
            business_id="biz_test",
            name="Not on Meta",
            status="draft",
        )
    )
    await tool_ctx.db.commit()
    result = await _handle_get_campaign_insights(
        GetCampaignInsightsInput(campaign_id="cmp_unlinked"), tool_ctx
    )
    assert result.status == ToolStatus.UNAVAILABLE
    assert "not linked" in (result.unavailable_reason or "").lower()


async def test_web_search_never_fabricates_without_a_provider(tool_ctx: ToolContext) -> None:
    from app.core.tools.research import _handle_web_search

    result = await _handle_web_search(WebSearchInput(query="coffee"), tool_ctx)
    # Mock results are placeholders, so they are reported as partial with a reason.
    assert result.status == ToolStatus.PARTIAL
    assert result.unavailable_reason
    assert "placeholder" in result.unavailable_reason.lower()
    assert result.source.startswith("web_search:")


async def test_karim_reports_missing_data_instead_of_analysing_nothing(
    tool_ctx: ToolContext,
) -> None:
    """The headline guarantee: no metrics in, no analysis out."""
    result = await run_agent(KARIM, "performance_analysis", {}, tool_ctx, llm=MockClient())
    assert result.error is None
    assert result.output["headline_metrics"] == {}
    assert len(result.unavailable_sources) == 2  # insights + breakdowns
    assert any("get_campaign_insights" in s for s in result.unavailable_sources)


async def test_salma_declines_to_recommend_without_data(tool_ctx: ToolContext) -> None:
    """No performance data -> needs_data, not a made-up change."""
    # Scripted the way a well-behaved model responds when the tools come back
    # unavailable: it reports the gap instead of proposing a change.
    scripted = MockClient(
        model="mock-strong-1",
        script=[
            # Tools turn: the model asks for the insights it needs.
            {
                "text": "Let me check the campaign performance first.",
                "tool_calls": [{"id": "c1", "name": "get_campaign_insights",
                                "arguments": {"campaign_id": "cmp_1"}}],
            },
            # Final-text turn (no tool calls): the model gives its answer.
            {"text": "Meta is not connected, so I cannot read performance data."},
            # Structured-output turn: with the data unavailable, report the gap.
            {
                "text": json.dumps({
                    "action_type": "needs_data",
                    "title": "Performance data required",
                    "rationale": (
                        "Meta is not connected and the campaign is not linked, so "
                        "there is nothing to base a change on."
                    ),
                    "payload": {},
                    "estimated_impact": "not available",
                    "needs_creative_variations": False,
                    "creative_brief_for_layla": "",
                    "summary": (
                        "I cannot recommend a change without performance data."
                    ),
                    "evidence": [],
                    "unavailable_sources": ["get_campaign_insights"],
                    "notes": "Connect Meta or paste the campaign numbers.",
                })
            }
        ],
    )
    result = await run_agent(
        SALMA, "optimization_proposal", {}, tool_ctx, llm=scripted
    )
    assert result.error is None
    assert result.output["action_type"] == "needs_data"
    assert result.output["estimated_impact"] == "not available"
    assert result.output["payload"] == {}
    assert "get_campaign_insights" in result.output["unavailable_sources"]


async def test_researcher_declares_unreachable_sources(tool_ctx: ToolContext) -> None:
    result = await run_agent(
        RESEARCHER, "competitor_research", {}, tool_ctx, llm=MockClient()
    )
    assert result.error is None
    # Two research providers are unconfigured in the test settings.
    assert len(result.unavailable_sources) >= 1
    for entry in result.unavailable_sources:
        assert entry.strip()


async def test_every_unavailable_output_carries_a_reason(tool_ctx: ToolContext) -> None:
    """Any tool that returns unavailable must explain why."""
    from app.core.tools.registry import get_registry

    registry = get_registry()
    for name in ("web_search", "get_reviews", "get_keywords", "get_campaign_insights"):
        tool = registry.get(name)
        assert tool is not None
        schema = tool.input_model.model_json_schema()
        args = sample_from_schema(schema)
        payload = tool.input_model.model_validate(args)
        try:
            output = await tool.handler(payload, tool_ctx)
        except ToolUnavailable as exc:
            assert exc.message
            continue
        status = getattr(output, "status", None)
        if status == ToolStatus.UNAVAILABLE:
            assert getattr(output, "unavailable_reason", None), (
                f"{name} returned unavailable without a reason"
            )


async def test_generate_image_mock_mode_is_marked_as_mock(tool_ctx: ToolContext) -> None:
    """Placeholder assets are labelled, so they can never be mistaken for real."""
    from app.core.tools.media import _handle_generate_image
    from app.schemas.tools import GenerateImageInput

    result = await _handle_generate_image(
        GenerateImageInput(prompt="a coffee cup", num_images=2), tool_ctx
    )
    assert result.status == ToolStatus.OK
    assert len(result.assets) == 2
    for asset in result.assets:
        assert asset.provider == "mock"
        assert asset.url.startswith("data:image/svg+xml")
