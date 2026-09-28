"""Protected tools: the ONLY code that writes to Meta.

``apply_change`` and ``publish_campaign`` have ``side_effect="external_write"``.
They are deliberately NOT registered in the agent tool registry -- the registry
refuses external-write tools outright -- and they are importable from exactly
one place: ``app/core/approvals/service.py``.

If you are reading this because you want to call one of these from an agent:
don't. Have the agent create a recommendation with ``create_recommendation``,
surface it as a ``recommendation`` block, and let the user approve it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.core.errors import ApprovalRequired, ToolUnavailable
from app.core.logging import get_logger
from app.core.security import TokenCipher
from app.core.tools.base import Tool, ToolContext
from app.db.models import Integration, Recommendation
from app.schemas.common import ToolSideEffect

_logger = get_logger(__name__)


# ------------------------------------------------------------------- models --


class ApplyChangeInput(BaseModel):
    recommendation_id: str = Field(min_length=1)
    #: Set by approvals/service.py once the human has confirmed the diff.
    confirmed_by: str = "system"
    note: str = ""


class ApplyChangeOutput(BaseModel):
    status: str = "ok"
    recommendation_id: str = ""
    action_type: str = ""
    external_id: str | None = None
    detail: str = ""
    unavailable_reason: str | None = None


class PublishCampaignInput(BaseModel):
    campaign_id: str = Field(min_length=1)
    confirmed_by: str = "system"
    note: str = ""


class PublishCampaignOutput(BaseModel):
    status: str = "ok"
    campaign_id: str = ""
    external_id: str | None = None
    detail: str = ""
    unavailable_reason: str | None = None


# ------------------------------------------------------------------ helpers --


async def _meta_token(ctx: ToolContext) -> str | None:
    from sqlalchemy import select

    stmt = select(Integration).where(
        Integration.business_id == ctx.business_id,
        Integration.provider == "meta",
        Integration.status == "connected",
    )
    integration = (await ctx.db.execute(stmt)).scalars().first()
    if integration is None or not integration.access_token_enc:
        return None
    cfg = ctx.require_settings()
    return TokenCipher(cfg.token_encryption_key).decrypt(integration.access_token_enc)


async def _graph_post_async(
    cfg: Any, token: str, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    url = f"{cfg.meta_graph_base_url}/{cfg.meta_graph_version}/{path}"
    async with httpx.AsyncClient(timeout=cfg.meta_api_timeout_seconds) as client:
        resp = await client.post(url, params={"access_token": token}, json=payload)
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------- apply_change handler --


async def _handle_apply_change(payload: ApplyChangeInput, ctx: ToolContext) -> ApplyChangeOutput:
    """Apply an approved recommendation's change to Meta.

    Preconditions enforced here, independently of the caller:
    * the recommendation must exist and be in status "approved"
    * the caller must have passed an explicit ``confirmed_by``
    """
    recommendation = await ctx.db.get(Recommendation, payload.recommendation_id)
    if recommendation is None:
        raise ApprovalRequired(f"recommendation '{payload.recommendation_id}' not found")
    if recommendation.status != "approved":
        raise ApprovalRequired(
            f"recommendation '{payload.recommendation_id}' is '{recommendation.status}', "
            "not 'approved'. It must be approved (and confirmed) before it can be "
            "applied to Meta."
        )
    if not payload.confirmed_by or payload.confirmed_by == "system":
        raise ApprovalRequired(
            "an explicit confirming actor is required before writing to Meta"
        )

    cfg = ctx.require_settings()
    token = await _meta_token(ctx)
    if not token:
        return ApplyChangeOutput(
            status="unavailable",
            recommendation_id=payload.recommendation_id,
            action_type=recommendation.action_type,
            detail="Meta is not connected for this workspace.",
            unavailable_reason=(
                "No Meta token is stored, so the change could not be applied. "
                "Connect Meta and retry."
            ),
        )

    change_payload = dict(recommendation.payload or {})
    try:
        result = await _graph_post_async(
            cfg, token, f"{recommendation.action_type}", change_payload
        )
    except httpx.HTTPError as exc:
        return ApplyChangeOutput(
            status="unavailable",
            recommendation_id=payload.recommendation_id,
            action_type=recommendation.action_type,
            detail=str(exc),
            unavailable_reason=f"Meta rejected the change: {exc}",
        )

    recommendation.executed_at = datetime.now(timezone.utc)
    recommendation.execution_result = result
    await ctx.db.flush()
    _logger.info(
        "apply_change_executed",
        extra={
            "recommendation_id": payload.recommendation_id,
            "action_type": recommendation.action_type,
        },
    )
    return ApplyChangeOutput(
        status="ok",
        recommendation_id=payload.recommendation_id,
        action_type=recommendation.action_type,
        external_id=str(result.get("id")) if isinstance(result, dict) else None,
        detail="change applied to Meta",
    )


# ------------------------------------------------- publish_campaign handler --


async def _handle_publish_campaign(
    payload: PublishCampaignInput, ctx: ToolContext
) -> PublishCampaignOutput:
    from app.db.models import Campaign

    campaign = await ctx.db.get(Campaign, payload.campaign_id)
    if campaign is None:
        raise ApprovalRequired(f"campaign '{payload.campaign_id}' not found")
    if not payload.confirmed_by or payload.confirmed_by == "system":
        raise ApprovalRequired(
            "an explicit confirming actor is required before publishing"
        )

    cfg = ctx.require_settings()
    token = await _meta_token(ctx)
    if not token:
        return PublishCampaignOutput(
            status="unavailable",
            campaign_id=payload.campaign_id,
            detail="Meta is not connected for this workspace.",
            unavailable_reason=(
                "No Meta token is stored, so the campaign could not be published."
            ),
        )

    try:
        result = await _graph_post_async(
            cfg,
            token,
            f"act_{cfg.meta_ad_account_id}/campaigns" if cfg.meta_ad_account_id else "campaigns",
            {"name": campaign.name, "objective": campaign.objective or "OUTCOME_AWARENESS",
             "status": "PAUSED"},
        )
    except httpx.HTTPError as exc:
        return PublishCampaignOutput(
            status="unavailable",
            campaign_id=payload.campaign_id,
            detail=str(exc),
            unavailable_reason=f"Meta rejected the publish request: {exc}",
        )

    external_id = str(result.get("id")) if isinstance(result, dict) else None
    campaign.meta_campaign_id = external_id
    campaign.status = "published"
    await ctx.db.flush()
    _logger.info("publish_campaign_executed", extra={"campaign_id": payload.campaign_id})
    return PublishCampaignOutput(
        status="ok", campaign_id=payload.campaign_id, external_id=external_id,
        detail="campaign created on Meta (PAUSED)",
    )


# --------------------------------------------------------------- tool objects --

#: These are NEVER passed to ``ToolRegistry.register`` -- it raises on
#: external_write. They exist so approvals/service.py has typed handlers.
apply_change = Tool(
    name="apply_change",
    description=(
        "PROTECTED: applies an approved recommendation's change to Meta. Only "
        "callable from approvals/service.py after a human approval and explicit "
        "confirmation. Never registered for agent use."
    ),
    input_model=ApplyChangeInput,
    output_model=ApplyChangeOutput,
    side_effect=ToolSideEffect.EXTERNAL_WRITE,
    handler=_handle_apply_change,
)

publish_campaign = Tool(
    name="publish_campaign",
    description=(
        "PROTECTED: creates/publishes a campaign on Meta. Only callable from "
        "approvals/service.py after a human approval and explicit confirmation. "
        "Never registered for agent use."
    ),
    input_model=PublishCampaignInput,
    output_model=PublishCampaignOutput,
    side_effect=ToolSideEffect.EXTERNAL_WRITE,
    handler=_handle_publish_campaign,
)

#: The complete protected set. approvals/service.py imports from here only.
PROTECTED_TOOLS: list[Tool] = [apply_change, publish_campaign]


__all__ = [
    "PROTECTED_TOOLS",
    "ApplyChangeInput",
    "ApplyChangeOutput",
    "PublishCampaignInput",
    "PublishCampaignOutput",
    "apply_change",
    "publish_campaign",
]
