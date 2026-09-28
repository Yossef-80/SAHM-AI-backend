"""records tool (13): create_recommendation.

Writes a recommendation with status="pending". It is a ``draft_write``: it only
touches our own database. Execution against Meta happens exclusively through
``approvals/service.py`` -> ``protected.apply_change`` / ``publish_campaign``.
"""

from __future__ import annotations

import uuid

from app.core.logging import get_logger
from app.core.tools.base import Tool, ToolContext
from app.db.models import Recommendation
from app.schemas.common import ToolSideEffect
from app.schemas.recommendations import RecommendationStatus
from app.schemas.tools import CreateRecommendationInput, CreateRecommendationOutput

_logger = get_logger(__name__)


async def _handle_create_recommendation(
    payload: CreateRecommendationInput, ctx: ToolContext
) -> CreateRecommendationOutput:
    recommendation_id = f"rec_{uuid.uuid4().hex[:24]}"
    ctx.db.add(
        Recommendation(
            id=recommendation_id,
            campaign_id=payload.campaign_id or ctx.campaign_id,
            insight_id=payload.insight_id or ctx.insight_id,
            action_type=payload.action_type,
            title=payload.title,
            rationale=payload.rationale,
            payload=payload.payload,
            evidence=[e.model_dump(mode="json") for e in payload.evidence],
            status=RecommendationStatus.PENDING.value,
            persona_id=ctx.persona_id,
            requires_approval=True,
            estimated_impact=payload.estimated_impact,
        )
    )
    await ctx.db.flush()
    _logger.info(
        "recommendation_created",
        extra={"recommendation_id": recommendation_id, "action_type": payload.action_type},
    )
    return CreateRecommendationOutput(
        recommendation_id=recommendation_id,
        recommendation_status=RecommendationStatus.PENDING.value,
        requires_approval=True,
        title=payload.title,
        action_type=payload.action_type,
        status="ok",
        source="records",
    )


create_recommendation_tool = Tool(
    name="create_recommendation",
    description=(
        "Create a pending recommendation for a campaign: an action type, the change "
        "payload, the rationale, and the evidence that supports it. The recommendation "
        "is NOT executed here -- a human must approve it first. Use this whenever you "
        "propose a change to a live campaign. Never propose a change you cannot back "
        "with evidence from tool results."
    ),
    input_model=CreateRecommendationInput,
    output_model=CreateRecommendationOutput,
    side_effect=ToolSideEffect.DRAFT_WRITE,
    handler=_handle_create_recommendation,
)

RECORDS_TOOLS: list[Tool] = [create_recommendation_tool]

__all__ = ["RECORDS_TOOLS", "create_recommendation_tool"]
