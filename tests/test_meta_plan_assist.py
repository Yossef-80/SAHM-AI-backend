"""Offline tests for MCP-backed, editable campaign wizard assistance."""

from __future__ import annotations

from typing import Any

from app.ai.marketing import meta_plan_assist
from app.core.llm.base import LLMResult, ToolCall
from app.schemas.campaigns import GeneratedCampaignStrategy, MetaAdPlanAssistRequest


class FakeLLM:
    def __init__(self, *, output: dict[str, Any] | None = None) -> None:
        self.output = output or {}
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        if kwargs.get("tools"):
            return LLMResult(
                text="Looking up live Meta field context.",
                tool_calls=[
                    ToolCall(
                        id="field-context-1",
                        name="ads_get_field_context",
                        arguments={
                            # The service must add every validated field even when
                            # the LLM requests only a subset.
                            "fields": ["objective"]
                        },
                    )
                ],
            )
        return LLMResult(
            text="structured suggestion",
            parsed=meta_plan_assist._MetaPlanSuggestion.model_validate(self.output),
        )


class FakeMcp:
    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context
        self.arguments: list[dict[str, Any]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "ads_get_field_context",
                "description": "Read Meta Ads field metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"fields": {"type": "array", "items": {"type": "string"}}},
                    "required": ["fields"],
                },
            },
            # Write tools can be advertised by Meta but are never handed to the LLM.
            {"name": "ads_create_campaign", "inputSchema": {"type": "object"}},
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert name == "ads_get_field_context"
        self.arguments.append(arguments)
        return {"structuredContent": self.context, "isError": False}


def _request(*, billing_event: str = "LINK_CLICKS") -> MetaAdPlanAssistRequest:
    return MetaAdPlanAssistRequest(
        campaign_name="Summer offer",
        objective="traffic",
        daily_budget=500,
        currency="EGP",
        brief="Promote the summer offer.",
        language="en",
        strategy=GeneratedCampaignStrategy(
            summary="Drive qualified visits to the offer page.",
            objective="traffic",
            audience="Adults interested in coffee.",
            cta_direction="Shop now",
            key_messages=["Try the summer blend"],
        ),
        audience={
            "name": "Coffee fans",
            "ageMin": 18,
            "ageMax": 45,
            "gender": "all",
            "countries": ["EG"],
            "interests": ["Coffee"],
        },
        destination_url="https://example.com/offer",
        primary_text="",
        headline="",
        call_to_action_type=None,
        billing_event=billing_event,
        optimization_goal="LINK_CLICKS",
        destination_type="WEBSITE",
    )


def _context() -> dict[str, Any]:
    return {
        "fields": [
            {"fieldName": "billing_event", "enumValues": ["IMPRESSIONS", "LINK_CLICKS"]},
            {"fieldName": "optimization_goal", "enumValues": ["REACH", "LINK_CLICKS"]},
            {"fieldName": "destination_type", "enumValues": ["WEBSITE", "APP"]},
            {"fieldName": "call_to_action.type", "enumValues": ["LEARN_MORE", "SHOP_NOW"]},
            {"fieldName": "bid_strategy", "enumValues": ["LOWEST_COST_WITHOUT_CAP", "BID_CAP"]},
        ]
    }


async def test_assistant_fetches_mcp_context_and_returns_reviewable_suggestions() -> None:
    llm = FakeLLM(
        output={
            "primary_text": "Try the summer blend today.",
            "headline": "Shop the summer offer",
            "call_to_action_type": "SHOP_NOW",
            "billing_event": "LINK_CLICKS",
            "optimization_goal": "LINK_CLICKS",
            "destination_type": "WEBSITE",
            "assumptions": [],
            "missing_inputs": [],
        }
    )
    mcp = FakeMcp(_context())

    result = await meta_plan_assist.assist_meta_ad_plan(
        llm=llm,
        mcp=mcp,  # type: ignore[arg-type]
        request=_request(),
    )

    assert set(mcp.arguments[0]["fields"]) >= {
        "objective",
        "billing_event",
        "optimization_goal",
        "destination_type",
        "call_to_action.type",
        "special_ad_categories",
        "bid_strategy",
    }
    assert len(llm.calls) == 2
    assert result.primary_text == "Try the summer blend today."
    assert result.headline == "Shop the summer offer"
    assert result.call_to_action_type == "SHOP_NOW"
    assert result.field_validation["billingEvent"] == "valid"
    assert result.field_validation["optimizationGoal"] == "valid"
    assert result.field_validation["destinationType"] == "valid"
    assert result.field_validation["callToActionType"] == "valid"
    assert result.field_validation["bidStrategy"] == "valid"
    assert result.bid_strategy == "LOWEST_COST_WITHOUT_CAP"
    assert result.review_required is True
    assert result.writes_to_meta is False


async def test_existing_copy_is_never_overwritten_and_invalid_enum_is_dropped() -> None:
    llm = FakeLLM(
        output={
            "primary_text": "Model tries to replace the copy.",
            "headline": "Model headline",
            "call_to_action_type": "SHOP_NOW",
            "billing_event": "IMPRESSIONS",
            "optimization_goal": "LINK_CLICKS",
            "destination_type": "WEBSITE",
            "assumptions": [],
            "missing_inputs": [],
        }
    )
    request = _request(billing_event="CPC").model_copy(
        update={"primary_text": "Keep this copy", "headline": "Keep this headline"}
    )

    result = await meta_plan_assist.assist_meta_ad_plan(
        llm=llm,
        mcp=FakeMcp(_context()),  # type: ignore[arg-type]
        request=request,
    )

    assert result.primary_text is None
    assert result.headline is None
    assert result.billing_event is None
    assert result.field_validation["billingEvent"] == "invalid"
    assert any("billing event" in item for item in result.missing_inputs)


def test_enum_extraction_only_matches_field_context_not_unrelated_enums() -> None:
    context = {
        "fields": [
            {"fieldName": "optimization_goal", "enumValues": ["REACH", "LINK_CLICKS"]},
            {"fieldName": "unrelated_status", "enumValues": ["ACTIVE", "PAUSED"]},
        ]
    }

    assert meta_plan_assist._enum_values_for_field(context, {"optimizationgoal"}) == [
        "REACH",
        "LINK_CLICKS",
    ]
    assert meta_plan_assist._enum_values_for_field(context, {"destinationtype"}) == []


def test_mcp_text_json_and_enum_mappings_are_parsed() -> None:
    response = {
        "content": [
            {
                "type": "text",
                "text": (
                    "Field metadata follows:\\n```json\\n"
                    '{"fields":[{"fieldName":"billing_event",'
                    '"enumValues":{"IMPRESSIONS":"Impressions",'
                    '"LINK_CLICKS":"Link clicks"}}]}\\n```'
                ),
            }
        ]
    }

    context = meta_plan_assist._context_for_model(response)

    assert meta_plan_assist._enum_values_for_field(context, {"billingevent"}) == [
        "IMPRESSIONS",
        "LINK_CLICKS",
    ]


async def test_mock_llm_is_rejected_before_mcp_or_placeholder_draft() -> None:
    class MockLLM(FakeLLM):
        model = "mock-strong-llm"

    class UnusedMcp:
        async def list_tools(self) -> list[dict[str, Any]]:
            raise AssertionError("MCP must not be called when the LLM is a mock")

    llm = MockLLM(output={"primary_text": "sample"})
    try:
        await meta_plan_assist.assist_meta_ad_plan(
            llm=llm,
            mcp=UnusedMcp(),  # type: ignore[arg-type]
            request=_request(),
        )
    except meta_plan_assist.MetaPlanAssistError as exc:
        assert "mock/demo mode" in str(exc)
    else:
        raise AssertionError("mock output must not be returned as an ad draft")

    assert llm.calls == []


async def test_only_the_meta_field_context_tool_is_exposed_to_the_llm() -> None:
    llm = FakeLLM(output={})
    mcp = FakeMcp(_context())
    await meta_plan_assist.assist_meta_ad_plan(llm=llm, mcp=mcp, request=_request())  # type: ignore[arg-type]

    names = [tool.name for tool in llm.calls[0]["tools"]]
    assert names == ["ads_get_field_context"]
