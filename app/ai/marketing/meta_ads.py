"""Meta Ads plan validation, approved hierarchy creation, and ad-level insights.

This module owns the new marketing workflow. It never creates Meta objects while
building a plan: ``execute_meta_ad_plan`` is called only by the approval service
on the explicit second (confirm=True) phase. All created objects are PAUSED.

No database migrations are needed. The existing ``Recommendation.payload`` holds
the reviewed plan and ``Recommendation.execution_result`` records Meta IDs and
the exact ``Creative.id`` each ad was built from. ``Insight.payload`` stores the
Meta Ad ID / Creative ID mapping for real ad-level performance rows.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol
from urllib.parse import parse_qsl, urlparse

import httpx
from pydantic import Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.security import TokenCipher
from app.core.tools.settings import load_connector_settings
from app.db.models import Campaign, Creative, Insight, Integration, Recommendation
from app.schemas.common import ApiModel

META_AD_PLAN_ACTION = "create_meta_ad_plan"
META_AD_PLAN_KIND = "meta_ad_plan_v1"
META_AD_INSIGHT_KIND = "meta_ad_performance"
# The wizard does not collect bid caps or ROAS floors, so use Meta's no-cap
# strategy explicitly at both campaign and ad-set levels.
META_DEFAULT_BID_STRATEGY = "LOWEST_COST_WITHOUT_CAP"

# These are the six objectives exposed by the existing CampaignObjectiveSchema.
OBJECTIVE_TO_META: dict[str, str] = {
    "awareness": "OUTCOME_AWARENESS",
    "traffic": "OUTCOME_TRAFFIC",
    "engagement": "OUTCOME_ENGAGEMENT",
    "leads": "OUTCOME_LEADS",
    "sales": "OUTCOME_SALES",
    "app_installs": "OUTCOME_APP_PROMOTION",
}

_OBJECTIVE_REQUIRED: dict[str, tuple[str, ...]] = {
    "awareness": (),
    "traffic": ("destination_type",),
    "engagement": ("destination_type",),
    "leads": ("destination_type", "promoted_object"),
    "sales": ("promoted_object",),
    "app_installs": ("promoted_object",),
}


class MetaAdPlanSettings(ApiModel):
    """Non-secret values needed to create a single Ad Set and one Ad per image.

    Objective values are deliberately user-supplied. In particular there are no
    defaults for optimization, billing, targeting, special-ad-category status,
    page identity, destination, CTA, or promoted object.
    """

    objective: Literal[
        "awareness", "traffic", "engagement", "leads", "sales", "app_installs"
    ]
    creative_ids: list[str] = Field(min_length=1, max_length=10)
    page_id: str | None = None
    targeting: dict[str, Any] | None = None
    billing_event: str | None = None
    optimization_goal: str | None = None
    special_ad_categories: list[str] | None = None
    destination_type: str | None = None
    promoted_object: dict[str, Any] | None = None
    destination_url: str | None = None
    primary_text: str | None = None
    headline: str | None = None
    call_to_action: dict[str, Any] | None = None
    instagram_actor_id: str | None = None

    @field_validator(
        "page_id",
        "billing_event",
        "optimization_goal",
        "destination_type",
        "destination_url",
        "primary_text",
        "headline",
        "instagram_actor_id",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("creative_ids")
    @classmethod
    def unique_creatives(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("creativeIds must not contain duplicates")
        return value


def _contains_credential_key(value: Any) -> bool:
    sensitive_parts = ("access_token", "app_secret", "client_secret", "password", "api_key")
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).replace("-", "_").lower()
            if any(part in normalized for part in sensitive_parts) or _contains_credential_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_credential_key(item) for item in value)
    return False


def _url_contains_credential_query(value: str) -> bool:
    try:
        query_keys = [key.lower() for key, _ in parse_qsl(urlparse(value).query, keep_blank_values=True)]
    except ValueError:
        return True
    return any(
        any(part in key for part in ("access_token", "app_secret", "client_secret", "password", "api_key"))
        for key in query_keys
    )


def missing_meta_ad_settings(settings: MetaAdPlanSettings) -> list[str]:
    """Return required form keys not yet supplied; never fills them by inference."""
    missing: list[str] = []
    for key in (
        "page_id",
        "targeting",
        "billing_event",
        "optimization_goal",
        "special_ad_categories",
        "destination_url",
        "primary_text",
        "headline",
        "call_to_action",
    ):
        value = getattr(settings, key)
        if value is None or value == "" or value == {}:
            missing.append(key)
    for key in _OBJECTIVE_REQUIRED[settings.objective]:
        value = getattr(settings, key)
        if value is None or value == "" or value == {}:
            missing.append(key)

    if settings.targeting is not None:
        geo = settings.targeting.get("geo_locations")
        if not isinstance(geo, dict) or not geo:
            missing.append("targeting.geo_locations")
        if "genders" in settings.targeting:
            genders = settings.targeting["genders"]
            if (
                not isinstance(genders, list)
                or any(type(gender) is not int or gender not in {1, 2} for gender in genders)
                or len(set(genders)) != len(genders)
            ):
                missing.append(
                    "targeting.genders (use an array containing only 1 or 2; omit for all genders)"
                )
    if settings.call_to_action is not None:
        if not isinstance(settings.call_to_action.get("type"), str) or not settings.call_to_action["type"].strip():
            missing.append("call_to_action.type")
        if not isinstance(settings.call_to_action.get("value"), dict):
            missing.append("call_to_action.value")
    if settings.destination_url and not _is_public_https_url(settings.destination_url):
        missing.append("destination_url (must be a public HTTPS URL)")
    if settings.destination_url and _url_contains_credential_query(settings.destination_url):
        missing.append("destination_url (credential-like query parameters are not allowed)")
    if settings.promoted_object is not None and not settings.promoted_object:
        missing.append("promoted_object")
    if _contains_credential_key(
        {
            "targeting": settings.targeting,
            "promoted_object": settings.promoted_object,
            "call_to_action": settings.call_to_action,
        }
    ):
        missing.append("credential-like keys are not allowed in Meta plan JSON")
    return list(dict.fromkeys(missing))


def objective_to_meta(objective: str) -> str:
    """Map only the objectives supported by the existing frontend contract."""
    try:
        return OBJECTIVE_TO_META[objective]
    except KeyError as exc:
        raise ValueError(f"Unsupported campaign objective: {objective}") from exc


def _is_public_https_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
        if parsed.scheme.lower() != "https" or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        host = parsed.hostname.rstrip(".").lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal", ".test")):
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return "." in host and " " not in host
        return ip.is_global
    except (TypeError, ValueError):
        return False


# Meta accepts images up to 30 MB. Refuse larger ones here rather than after the
# campaign and ad set already exist.
MAX_AD_IMAGE_BYTES = 30 * 1024 * 1024

_DATA_IMAGE_URL = re.compile(
    r"data:image/(?:png|jpe?g|webp);base64,([A-Za-z0-9+/]+={0,2})", re.DOTALL
)


async def _creative_image_payload(url: str, *, timeout: float) -> str | None:
    """Base64 payload for Meta's /adimages `bytes` parameter, or None.

    Meta's adimages edge documents exactly two create parameters — `bytes` and
    `copy_from`. There is no `url`, and the `url` on the returned object is
    explicitly marked "do not use in ad creative creation", so the image has to
    travel with the request.

    Generated creatives already carry their bytes inline as a data URL, which
    needs no network call. A hosted URL is fetched here, and only if it passes
    the same public-HTTPS check used for destination URLs, so this introduces no
    new way to reach an internal address.
    """
    value = (url or "").strip()
    if not value:
        return None

    inline = _DATA_IMAGE_URL.fullmatch(value)
    if inline is not None:
        try:
            raw = base64.b64decode(inline.group(1), validate=True)
        except (ValueError, binascii.Error):
            return None
    elif _is_public_https_url(value):
        try:
            async with httpx.AsyncClient(
                timeout=timeout, trust_env=False, follow_redirects=False
            ) as client:
                response = await client.get(value)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        raw = response.content
    else:
        return None

    if not raw or len(raw) > MAX_AD_IMAGE_BYTES:
        return None
    return base64.b64encode(raw).decode()


def validate_creative_image_url(value: str) -> bool:
    """Accept the two shapes we can actually hand to Meta; never analyse pixels here.

    Meta's adimages edge takes `bytes`, so a public URL is no longer a
    requirement — an inline data URL works just as well, and that is what a
    generated creative stores. Requiring public HTTPS here silently disqualified
    every generated image, because those never have a hosted URL.
    """
    text = (value or "").strip()
    if not text:
        return False
    if _DATA_IMAGE_URL.fullmatch(text) is not None:
        return True
    return _is_public_https_url(text)


class MetaAdPlanValidationError(ValueError):
    def __init__(self, message: str, *, missing_fields: list[str] | None = None):
        super().__init__(message)
        self.missing_fields = missing_fields or []


async def create_meta_ad_plan_draft(
    db: AsyncSession,
    *,
    business_id: str,
    campaign_input: Any,
    plan: MetaAdPlanSettings,
) -> tuple[Campaign, Recommendation]:
    """Persist a local campaign draft and its pending Meta plan atomically.

    This function performs validation and DB writes only. It never resolves a
    credential or calls Meta.
    """
    if campaign_input.objective not in OBJECTIVE_TO_META:
        raise MetaAdPlanValidationError("Unsupported campaign objective.")
    if campaign_input.objective != plan.objective:
        raise MetaAdPlanValidationError(
            "The selected Meta objective must match the campaign objective."
        )
    if campaign_input.meta_campaign_id:
        raise MetaAdPlanValidationError(
            "A new Meta launch plan cannot be linked to an existing Meta Campaign ID."
        )
    missing = missing_meta_ad_settings(plan)
    if missing:
        raise MetaAdPlanValidationError(
            "Complete all required non-secret settings before preparing the plan.",
            missing_fields=missing,
        )
    try:
        _minor_units(campaign_input.budget)
    except ValueError as exc:
        raise MetaAdPlanValidationError(str(exc)) from exc
    if campaign_input.start_date and campaign_input.end_date:
        if campaign_input.end_date <= campaign_input.start_date:
            raise MetaAdPlanValidationError("End date must be after the start date.")

    creative_rows = (
        await db.execute(
            select(Creative).where(
                Creative.id.in_(plan.creative_ids),
                Creative.business_id == business_id,
            )
        )
    ).scalars().all()
    creatives = {row.id: row for row in creative_rows}
    missing_creatives = [cid for cid in plan.creative_ids if cid not in creatives]
    if missing_creatives:
        raise MetaAdPlanValidationError(
            "Selected creatives must exist in this workspace.",
            missing_fields=[f"creative_ids:{cid}" for cid in missing_creatives],
        )
    unusable = [
        row.id
        for row in creative_rows
        if row.kind != "image" or row.status != "ok" or not validate_creative_image_url(row.url)
    ]
    if unusable:
        raise MetaAdPlanValidationError(
            "Selected final assets must be public HTTPS image files. The final image is not AI-analysed.",
            missing_fields=[f"usable_image:{creative_id}" for creative_id in unusable],
        )

    campaign = Campaign(
        id=_new_id("cmp"),
        business_id=business_id,
        name=campaign_input.name,
        objective=campaign_input.objective,
        channel=campaign_input.channel or "meta_ads",
        budget=campaign_input.budget,
        currency=campaign_input.currency,
        start_date=campaign_input.start_date,
        end_date=campaign_input.end_date,
        target_audience=campaign_input.target_audience,
        meta_campaign_id=None,
        notes=campaign_input.notes,
        status="draft",
    )
    db.add(campaign)
    await db.flush()

    # Link the selected creatives to this campaign. Without this the campaign's
    # Creatives tab and GET /creatives?campaignId=... cannot find them: the only
    # other record of the pairing lives inside Recommendation.execution_result,
    # which is written after a successful Meta execution. A creative already
    # owned by a different campaign is deliberately left alone -- the same asset
    # may legitimately be reused, and silently re-parenting it would corrupt the
    # other campaign's view.
    for row in creative_rows:
        if row.campaign_id is None:
            row.campaign_id = campaign.id
    await db.flush()

    plan_payload = plan.model_dump(mode="json", by_alias=True)
    recommendation = Recommendation(
        id=_new_id("rec"),
        campaign_id=campaign.id,
        workspace_id=business_id,
        action_type=META_AD_PLAN_ACTION,
        title=f"Prepare paused Meta ads: {campaign.name}",
        rationale=(
            "Create one paused Meta Campaign and Ad Set, then one Creative and Ad "
            "for each selected final image. No object is created until the second "
            "explicit approval confirmation."
        ),
        payload={
            "kind": META_AD_PLAN_KIND,
            "campaign_summary": {
                "name": campaign.name,
                "objective": campaign.objective,
                "daily_budget": campaign.budget,
                "currency": campaign.currency,
                "status_after_execution": "PAUSED",
            },
            "plan": plan_payload,
        },
        evidence=[{"source": "user_input", "detail": "Campaign wizard launch settings"}],
        status="pending",
        requires_approval=True,
    )
    db.add(recommendation)
    await db.flush()
    return campaign, recommendation


async def preflight_meta_ad_plan(
    db: AsyncSession,
    recommendation: Recommendation,
    *,
    settings: Settings,
) -> list[str]:
    """Check local settings/ownership/config before recording phase-one approval.

    This is intentionally offline: it resolves encrypted/local configuration but
    does not call Meta or verify the token against a live API.
    """
    issues: list[str] = []
    if recommendation.action_type != META_AD_PLAN_ACTION:
        return ["This recommendation is not a Meta ad plan."]
    payload = recommendation.payload if isinstance(recommendation.payload, dict) else {}
    plan_raw = payload.get("plan") if isinstance(payload.get("plan"), dict) else None
    try:
        plan = MetaAdPlanSettings.model_validate(plan_raw or {})
    except ValidationError:
        return ["The saved Meta plan is invalid; prepare it again from the campaign wizard."]
    issues.extend(missing_meta_ad_settings(plan))
    campaign = await db.get(Campaign, recommendation.campaign_id) if recommendation.campaign_id else None
    if campaign is None:
        issues.append("The local campaign record is missing.")
        return issues
    if recommendation.workspace_id != campaign.business_id:
        issues.append("The campaign and recommendation workspace do not match.")
    if campaign.objective != plan.objective:
        issues.append("The saved objective no longer matches the campaign.")
    try:
        _minor_units(campaign.budget)
    except ValueError as exc:
        issues.append(str(exc))

    rows = (
        await db.execute(
            select(Creative).where(
                Creative.id.in_(plan.creative_ids),
                Creative.business_id == campaign.business_id,
            )
        )
    ).scalars().all() if plan.creative_ids else []
    by_id = {row.id: row for row in rows}
    for creative_id in plan.creative_ids:
        creative = by_id.get(creative_id)
        if creative is None:
            issues.append(f"Selected creative {creative_id} is unavailable in this workspace.")
        elif creative.kind != "image" or creative.status != "ok" or not validate_creative_image_url(creative.url):
            issues.append(f"Selected creative {creative_id} is not a public HTTPS final image.")

    connection = await resolve_meta_connection(db, campaign.business_id, settings)
    if not connection.access_token:
        issues.append("Meta access is not configured for this workspace; connect Meta first.")
    if not connection.ad_account_id:
        issues.append("Meta Ad Account ID is not configured in connector settings.")
    return list(dict.fromkeys(issues))


# Meta's error payload is the only thing that explains *why* a call failed.
# error_user_msg is written for humans; message is the developer-level reason.
_META_ERROR_TEXT_FIELDS = ("error_user_msg", "message")
# Defensive: never let a token-shaped run through, whatever Meta happens to echo.
_META_TOKEN_SHAPED = re.compile(r"[A-Za-z0-9_-]{24,}")
# This long value is a public Graph API field name, not a secret token.
_META_SAFE_ERROR_IDENTIFIERS = frozenset({"is_adset_budget_sharing_enabled"})


def _sanitize_meta_error_text(value: Any) -> str:
    """Collapse whitespace, redact token-shaped runs, and cap the length."""
    if not isinstance(value, str):
        return ""
    text = _META_TOKEN_SHAPED.sub(
        lambda match: (
            match.group(0)
            if match.group(0) in _META_SAFE_ERROR_IDENTIFIERS
            else "[redacted]"
        ),
        value,
    )
    return " ".join(text.split())[:300]


class MetaGraphError(RuntimeError):
    """Sanitized Meta error; never includes request URLs or credentials."""

    def __init__(self, message: str, *, retry_safe: bool, status_code: int | None = None):
        super().__init__(message)
        self.retry_safe = retry_safe
        self.status_code = status_code


class MetaGraphClient(Protocol):
    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def post_multipart(self, path: str, fields: dict[str, Any]) -> dict[str, Any]: ...

    async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]: ...


class HttpMetaGraphClient:
    """Small, sanitizing HTTP boundary. It never logs or returns access tokens."""

    def __init__(
        self,
        *,
        settings: Settings,
        access_token: str,
        base_url: str,
    ) -> None:
        self._settings = settings
        self._access_token = access_token
        self._base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        clean = path.lstrip("/")
        return f"{self._base_url}/{self._settings.meta_graph_version}/{clean}"

    @staticmethod
    def _error_message(data: Any, status: int) -> str:
        error = data.get("error") if isinstance(data, dict) else None
        if not isinstance(error, dict):
            return f"Meta rejected the request (HTTP {status})."
        code = error.get("code")
        suffix = f" (Meta error {code})" if isinstance(code, (int, str)) else ""

        reason = next(
            (
                cleaned
                for cleaned in (
                    _sanitize_meta_error_text(error.get(field))
                    for field in _META_ERROR_TEXT_FIELDS
                )
                if cleaned
            ),
            "",
        )
        hints: list[str] = []
        subcode = error.get("error_subcode")
        if isinstance(subcode, (int, str)):
            hints.append(f"subcode {subcode}")
        blame = error.get("error_data")
        if isinstance(blame, dict):
            fields = blame.get("blame_field_specs")
            if isinstance(fields, list) and fields:
                hints.append("field: " + ", ".join(str(f) for f in fields[:3]))
        joined = "·".join(hints)
        if hints:
            reason = f"{reason} [{joined}]" if reason else f"[{joined}]"

        # Still never the request URL or the raw body: the URL carries the token.
        if reason:
            return f"Meta rejected the request{suffix}: {reason}"
        return f"Meta rejected the request{suffix}. Check the submitted settings and permissions."

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
        *,
        multipart: bool = False,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self._settings.meta_api_timeout_seconds,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                if method == "POST":
                    response = await client.post(
                        self._url(path),
                        params={"access_token": self._access_token},
                        # /adimages only honours `bytes` as a form field, so that
                        # one call goes out as multipart; everything else stays
                        # a JSON body exactly as before.
                        data=payload if multipart else None,
                        json=None if multipart else payload,
                    )
                else:
                    response = await client.get(
                        self._url(path),
                        params={"access_token": self._access_token, **payload},
                    )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise MetaGraphError(
                "Meta connection failed; the outcome of the last request may be unknown.",
                retry_safe=False,
            ) from exc
        except httpx.HTTPError as exc:
            raise MetaGraphError(
                "Meta connection failed; the outcome of the last request may be unknown.",
                retry_safe=False,
            ) from exc

        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code >= 400:
            retry_safe = 400 <= response.status_code < 500
            raise MetaGraphError(
                self._error_message(data, response.status_code),
                retry_safe=retry_safe,
                status_code=response.status_code,
            )
        if not isinstance(data, dict):
            raise MetaGraphError("Meta returned an unexpected response.", retry_safe=False)
        return data

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", path, payload)

    async def post_multipart(self, path: str, fields: dict[str, Any]) -> dict[str, Any]:
        """POST as multipart/form-data.

        Required by /adimages: its `bytes` parameter is only honoured as a form
        field, so sending it in a JSON body would upload nothing.
        """
        return await self._request("POST", path, fields, multipart=True)

    async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._request("GET", path, params)


@dataclass(frozen=True)
class MetaConnectionConfig:
    access_token: str | None
    ad_account_id: str | None
    page_id: str | None
    graph_base_url: str


async def resolve_meta_connection(
    db: AsyncSession,
    business_id: str,
    settings: Settings,
) -> MetaConnectionConfig:
    """Use the existing Integration > workspace settings > environment precedence."""
    integration = await db.scalar(
        select(Integration).where(
            Integration.business_id == business_id,
            Integration.provider == "meta",
            Integration.status == "connected",
        )
    )
    tools = await load_connector_settings(db, business_id, settings)

    access_token: str | None = None
    account_id: str | None = None
    if integration is not None and integration.access_token_enc:
        try:
            access_token = TokenCipher(settings.token_encryption_key).decrypt(
                integration.access_token_enc
            )
        except Exception:
            # Corrupt/rotated encryption key is treated as a disconnected grant.
            access_token = None
        account_id = integration.account_id

    access_token = access_token or tools.meta_access_token or settings.meta_access_token or None
    account_id = account_id or tools.meta_ad_account_id
    page_id = tools.meta_page_id or settings.meta_page_id or None
    graph_base = tools.meta_graph_base_url or settings.meta_graph_base_url
    if account_id:
        account_id = account_id.removeprefix("act_")
    if page_id:
        page_id = page_id.strip()
    return MetaConnectionConfig(
        access_token=access_token,
        ad_account_id=account_id,
        page_id=page_id,
        graph_base_url=graph_base,
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _numeric_metrics(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("spend", "impressions", "reach", "frequency", "clicks", "ctr", "cpc", "cpm"):
        raw = row.get(key)
        if isinstance(raw, (int, float)):
            out[key] = float(raw)
        elif isinstance(raw, str):
            try:
                out[key] = float(raw)
            except ValueError:
                continue
    return out


def _to_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _minor_units(budget: float | None) -> str:
    if budget is None or budget <= 0:
        raise ValueError("A positive campaign daily budget is required before approval.")
    try:
        amount = Decimal(str(budget)) * Decimal(100)
    except InvalidOperation as exc:
        raise ValueError("The campaign daily budget is invalid.") from exc
    if amount != amount.to_integral_value():
        raise ValueError("The campaign daily budget must use at most two decimal places.")
    return str(int(amount))


def _graph_object_id(data: dict[str, Any]) -> str:
    value = data.get("id")
    if isinstance(value, (int, str)) and str(value).strip():
        return str(value)
    raise MetaGraphError("Meta returned no object ID; creation stopped safely.", retry_safe=False)


def _ad_image_hash(data: dict[str, Any]) -> str:
    images = data.get("images")
    if isinstance(images, dict):
        for image in images.values():
            if isinstance(image, dict) and image.get("hash"):
                return str(image["hash"])
    if data.get("hash"):
        return str(data["hash"])
    raise MetaGraphError("Meta returned no uploaded image hash; creation stopped safely.", retry_safe=False)


class MetaAdPlanExecutionResult(ApiModel):
    status: Literal["ok", "unavailable"] = "unavailable"
    recommendation_id: str = ""
    action_type: str = META_AD_PLAN_ACTION
    external_id: str | None = None
    detail: str = ""
    unavailable_reason: str | None = None
    execution_result: dict[str, Any] | None = None


async def execute_meta_ad_plan(
    db: AsyncSession,
    recommendation: Recommendation,
    *,
    actor: str,
    settings: Settings,
    graph: MetaGraphClient | None = None,
) -> MetaAdPlanExecutionResult:
    """Create Campaign → Ad Set → AdCreative → Ad after the approved second phase.

    Every write is PAUSED. Progress with external IDs is saved into the existing
    Recommendation JSON column after each successful Graph call. Uncertain partial
    attempts remain blocked. The sole resumable partial is a confirmed HTTP 400
    rejection while setting the strategy or creating the ad set, when the campaign
    ID is known; after explicit review, the existing paused campaign is reused
    rather than duplicated. A prior
    ``failed_safe`` result with no created IDs can be retried by explicit user action.
    """
    rec_id = recommendation.id
    if recommendation.action_type != META_AD_PLAN_ACTION:
        raise ValueError("This executor only accepts Meta ad plan recommendations.")
    if recommendation.status != "approved":
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="The plan has not completed the explicit approval phase.",
            unavailable_reason="Approve the reviewed plan before creating Meta objects.",
        )
    if not actor or actor == "system":
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="A human actor is required.",
            unavailable_reason="An explicit user confirmation is required before writing to Meta.",
        )

    prior = recommendation.execution_result if isinstance(recommendation.execution_result, dict) else {}
    campaign_id = recommendation.campaign_id
    campaign = await db.get(Campaign, campaign_id) if campaign_id else None
    if campaign is None:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="The local campaign for this plan was not found.",
            unavailable_reason="The campaign record is missing; no Meta write was attempted.",
        )

    prior_ids = prior.get("created_ids")
    resume_campaign_id: str | None = None
    if (
        prior.get("status") == "partial"
        and prior.get("attempted_step") in {"campaign_bid_strategy", "ad_set"}
        and prior.get("failure_status_code") == 400
        and prior.get("retry_safe") is True
        and prior.get("campaign_id") == campaign.id
        and isinstance(prior_ids, dict)
        and set(prior_ids) == {"campaign"}
        and isinstance(prior_ids.get("campaign"), str)
        and bool(prior_ids["campaign"])
        and campaign.meta_campaign_id == prior_ids["campaign"]
    ):
        resume_campaign_id = prior_ids["campaign"]

    if (
        prior.get("status") in {"in_progress", "partial", "unknown"}
        or prior.get("created_ids")
    ) and not resume_campaign_id:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="A partial/uncertain prior attempt exists; automatic replay was blocked.",
            unavailable_reason=(
                "Some paused Meta objects may already exist. Review the recorded IDs and "
                "resolve the partial attempt before creating another plan."
            ),
            execution_result=prior,
        )

    payload = recommendation.payload if isinstance(recommendation.payload, dict) else {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else None
    if not plan or payload.get("kind") != META_AD_PLAN_KIND:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="The saved plan payload is invalid.",
            unavailable_reason="Rebuild the plan from the campaign wizard; no Meta write was attempted.",
        )
    try:
        plan_model = MetaAdPlanSettings.model_validate(plan)
    except ValidationError:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="The saved plan settings are invalid.",
            unavailable_reason="Rebuild the plan with all required Meta settings; no Meta write was attempted.",
        )
    missing = missing_meta_ad_settings(plan_model)
    if missing:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="Required non-secret settings are missing.",
            unavailable_reason="Complete these fields before approval: " + ", ".join(missing),
        )
    if campaign.objective != plan_model.objective:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="The campaign objective no longer matches the reviewed plan.",
            unavailable_reason="Rebuild the plan so the objective and Meta settings match.",
        )

    creative_stmt = select(Creative).where(
        Creative.id.in_(plan_model.creative_ids),
        Creative.business_id == campaign.business_id,
    )
    creative_rows = (await db.execute(creative_stmt)).scalars().all()
    creative_by_id = {row.id: row for row in creative_rows}
    missing_creatives = [cid for cid in plan_model.creative_ids if cid not in creative_by_id]
    if missing_creatives:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="One or more selected final creatives are no longer available in this workspace.",
            unavailable_reason="Missing creative IDs: " + ", ".join(missing_creatives),
        )
    if any(
        row.kind != "image" or row.status != "ok" or not validate_creative_image_url(row.url)
        for row in creative_rows
    ):
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="At least one selected asset is not a usable public HTTPS image.",
            unavailable_reason=(
                "Select final image assets hosted at public HTTPS URLs. Data URLs, local URLs, "
                "videos, and unsupported assets are not sent to Meta. No AI analysis is run "
                "on the final image."
            ),
        )
    # Preserve user selection order; SQL IN() does not guarantee row order.
    ordered_creatives = [creative_by_id[cid] for cid in plan_model.creative_ids]

    connection = await resolve_meta_connection(db, campaign.business_id, settings)
    if not connection.access_token:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="Meta is not connected for this workspace.",
            unavailable_reason="Connect Meta through the existing Integrations page; no Meta write was attempted.",
        )
    if not connection.ad_account_id:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="The Meta ad account ID is not configured.",
            unavailable_reason=(
                "Set the non-secret Meta Ad Account ID in the existing connector settings; "
                "no Meta write was attempted."
            ),
        )
    try:
        daily_budget = _minor_units(campaign.budget)
        objective = objective_to_meta(plan_model.objective)
    except ValueError as exc:
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="The campaign cannot be translated to a Meta create request.",
            unavailable_reason=str(exc),
        )

    graph_client = graph or HttpMetaGraphClient(
        settings=settings,
        access_token=connection.access_token,
        base_url=connection.graph_base_url,
    )
    account_path = f"act_{connection.ad_account_id}"
    result: dict[str, Any] = {
        "kind": META_AD_PLAN_KIND,
        "status": "in_progress",
        "recommendation_id": rec_id,
        "campaign_id": campaign.id,
        "objective": plan_model.objective,
        "meta_objective": objective,
        "bid_strategy": META_DEFAULT_BID_STRATEGY,
        "created_ids": {"campaign": resume_campaign_id} if resume_campaign_id else {},
        "ads": [],
        "attempted_step": None,
        "resumed_from_partial": bool(resume_campaign_id),
    }

    async def checkpoint(step: str) -> None:
        result["attempted_step"] = step
        recommendation.execution_result = dict(result)
        await db.flush()

    async def save_created_id(step: str, object_id: str) -> None:
        ids = dict(result.get("created_ids") or {})
        ids[step] = object_id
        result["created_ids"] = ids
        recommendation.execution_result = dict(result)
        await db.flush()

    try:
        if resume_campaign_id:
            # Reuse the already-recorded paused campaign; explicitly repair its
            # strategy before retrying the rejected ad-set request.
            meta_campaign_id = resume_campaign_id
            campaign.status = "paused"
            await checkpoint("campaign_bid_strategy")
            await graph_client.post(
                meta_campaign_id, {"bid_strategy": META_DEFAULT_BID_STRATEGY}
            )
            result["campaign_bid_strategy_updated"] = True
        else:
            await checkpoint("campaign")
            campaign_response = await graph_client.post(
                f"{account_path}/campaigns",
                {
                    "name": campaign.name,
                    "objective": objective,
                    "status": "PAUSED",
                    "bid_strategy": META_DEFAULT_BID_STRATEGY,
                    "special_ad_categories": list(plan_model.special_ad_categories or []),
                    # The budget is assigned at the ad-set level below. Make the
                    # sharing policy explicit and preserve that allocation.
                    "is_adset_budget_sharing_enabled": False,
                },
            )
            meta_campaign_id = _graph_object_id(campaign_response)
            await save_created_id("campaign", meta_campaign_id)
            campaign.meta_campaign_id = meta_campaign_id
            campaign.status = "paused"
            await db.flush()

        adset_body: dict[str, Any] = {
            "name": f"{campaign.name} — Ad Set",
            "campaign_id": meta_campaign_id,
            "daily_budget": daily_budget,
            "bid_strategy": META_DEFAULT_BID_STRATEGY,
            "billing_event": plan_model.billing_event,
            "optimization_goal": plan_model.optimization_goal,
            "targeting": plan_model.targeting,
            "status": "PAUSED",
        }
        if campaign.start_date:
            adset_body["start_time"] = _to_utc(campaign.start_date)
        if campaign.end_date:
            adset_body["end_time"] = _to_utc(campaign.end_date)
        if plan_model.destination_type:
            adset_body["destination_type"] = plan_model.destination_type
        if plan_model.promoted_object:
            adset_body["promoted_object"] = plan_model.promoted_object

        await checkpoint("ad_set")
        adset_response = await graph_client.post(f"{account_path}/adsets", adset_body)
        meta_adset_id = _graph_object_id(adset_response)
        await save_created_id("ad_set", meta_adset_id)

        for index, creative in enumerate(ordered_creatives):
            await checkpoint(f"image_upload:{creative.id}")
            # Meta's adimages edge accepts `bytes` or `copy_from` — not `url`.
            # Sending an undocumented field would be rejected live, and only
            # after the campaign and ad set already existed.
            image_payload = await _creative_image_payload(
                creative.url, timeout=settings.meta_api_timeout_seconds
            )
            if image_payload is None:
                raise MetaGraphError(
                    "The creative image could not be read, so nothing was uploaded to Meta.",
                    retry_safe=True,
                )
            image_response = await graph_client.post_multipart(
                f"{account_path}/adimages", {"bytes": image_payload}
            )
            image_hash = _ad_image_hash(image_response)

            creative_name = f"{campaign.name} — {creative.id}"
            object_story_spec: dict[str, Any] = {
                "page_id": plan_model.page_id,
                "link_data": {
                    "image_hash": image_hash,
                    "link": plan_model.destination_url,
                    "message": plan_model.primary_text,
                    "name": plan_model.headline,
                    "call_to_action": plan_model.call_to_action,
                },
            }
            if plan_model.instagram_actor_id:
                object_story_spec["instagram_actor_id"] = plan_model.instagram_actor_id

            await checkpoint(f"creative:{creative.id}")
            meta_creative_response = await graph_client.post(
                f"{account_path}/adcreatives",
                {"name": creative_name, "object_story_spec": object_story_spec},
            )
            meta_creative_id = _graph_object_id(meta_creative_response)

            await checkpoint(f"ad:{creative.id}")
            ad_response = await graph_client.post(
                f"{account_path}/ads",
                {
                    "name": creative_name,
                    "adset_id": meta_adset_id,
                    "creative": {"creative_id": meta_creative_id},
                    "status": "PAUSED",
                },
            )
            meta_ad_id = _graph_object_id(ad_response)
            ad_mapping = {
                "meta_ad_id": meta_ad_id,
                "meta_adcreative_id": meta_creative_id,
                "meta_adset_id": meta_adset_id,
                "creative_id": creative.id,
                "creative_prompt": creative.prompt,
                "reference_analysis_id": creative.reference_analysis_id,
                "asset_url_sha256": hashlib.sha256(creative.url.encode("utf-8")).hexdigest(),
                "image_hash": image_hash,
                "name": creative_name,
                "selection_index": index,
            }
            result["ads"] = [*result["ads"], ad_mapping]
            ids = dict(result.get("created_ids") or {})
            ids.setdefault("ads", []).append(meta_ad_id)
            result["created_ids"] = ids
            recommendation.execution_result = dict(result)
            await db.flush()

        result.update(
            {
                "status": "ok",
                "attempted_step": None,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        recommendation.execution_result = dict(result)
        campaign.meta_campaign_id = meta_campaign_id
        campaign.status = "paused"
        await db.flush()
        return MetaAdPlanExecutionResult(
            status="ok",
            recommendation_id=rec_id,
            action_type=META_AD_PLAN_ACTION,
            external_id=meta_campaign_id,
            detail=f"Created {len(ordered_creatives)} Meta ad(s); Campaign, Ad Set, Creative(s), and Ad(s) are PAUSED.",
            execution_result=result,
        )
    except MetaGraphError as exc:
        created_ids = result.get("created_ids") or {}
        partial = bool(created_ids)
        result.update(
            {
                "status": "unknown" if not exc.retry_safe else ("partial" if partial else "failed_safe"),
                "failure": str(exc),
                "failure_status_code": exc.status_code,
                "retry_safe": exc.retry_safe,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        recommendation.execution_result = dict(result)
        if campaign.meta_campaign_id:
            campaign.status = "paused"
        await db.flush()
        detail = (
            "Meta returned an error after a create request; object creation stopped at the failed step."
        )
        reason = str(exc)
        if partial or not exc.retry_safe:
            reason += " Automatic retry is blocked because an earlier step may already have created paused objects."
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            action_type=META_AD_PLAN_ACTION,
            external_id=campaign.meta_campaign_id,
            detail=detail,
            unavailable_reason=reason,
            execution_result=result,
        )
    except (ValueError, ValidationError) as exc:
        # Input issues are caught before the first network call above. Keep this
        # guard for adapter/serialization drift and never emit a raw request body.
        result.update({"status": "failed_safe", "failure": "request payload invalid"})
        recommendation.execution_result = dict(result)
        await db.flush()
        return MetaAdPlanExecutionResult(
            recommendation_id=rec_id,
            detail="A Meta create request could not be prepared.",
            unavailable_reason="Review the non-secret settings in the plan and rebuild it.",
            execution_result=result,
        )


#: Meta requires the campaign, its ad set and the ad itself to all be ACTIVE
#: before anything delivers, so activation always touches all three levels.
_ACTIVATION_STEPS = ("campaign", "ad_set")


class MetaAdPlanActivationResult(ApiModel):
    status: Literal["ok", "unavailable"] = "unavailable"
    recommendation_id: str = ""
    detail: str = ""
    unavailable_reason: str | None = None
    #: Labels of the Meta objects confirmed ACTIVE, e.g. "campaign", "ad:123".
    activated: list[str] = Field(default_factory=list)
    execution_result: dict[str, Any] | None = None


async def activate_meta_ad_plan(
    db: AsyncSession,
    recommendation: Recommendation,
    *,
    actor: str,
    settings: Settings,
    graph: MetaGraphClient | None = None,
) -> MetaAdPlanActivationResult:
    """Switch an already-created PAUSED plan to ACTIVE.

    This is the only path in the product that starts delivery, so it spends real
    money. It is deliberately separate from ``execute_meta_ad_plan``: creating
    Meta objects is reversible, activating them is not.

    Objects an earlier partial attempt already switched are skipped, so a retry
    resumes where it stopped instead of re-POSTing to Meta.
    """
    rec_id = recommendation.id
    if recommendation.action_type != META_AD_PLAN_ACTION:
        raise ValueError("This activator only accepts Meta ad plan recommendations.")
    if not actor or actor == "system":
        return MetaAdPlanActivationResult(
            recommendation_id=rec_id,
            detail="A human actor is required.",
            unavailable_reason=(
                "An explicit user confirmation is required before starting delivery."
            ),
        )

    execution = (
        recommendation.execution_result
        if isinstance(recommendation.execution_result, dict)
        else {}
    )
    created_raw = execution.get("created_ids")
    created = created_raw if isinstance(created_raw, dict) else {}
    if not created:
        return MetaAdPlanActivationResult(
            recommendation_id=rec_id,
            detail="No Meta objects were recorded for this plan.",
            unavailable_reason="Create the plan on Meta before trying to activate it.",
            execution_result=execution or None,
        )

    campaign_id = execution.get("campaign_id") or recommendation.campaign_id
    campaign = await db.get(Campaign, campaign_id) if campaign_id else None
    if campaign is None:
        return MetaAdPlanActivationResult(
            recommendation_id=rec_id,
            detail="The campaign record is missing.",
            unavailable_reason="No Meta write was attempted.",
        )

    connection = await resolve_meta_connection(db, campaign.business_id, settings)
    if not connection.access_token:
        return MetaAdPlanActivationResult(
            recommendation_id=rec_id,
            detail="Meta is not connected for this workspace.",
            unavailable_reason="Connect Meta before starting delivery.",
        )

    graph_client = graph or HttpMetaGraphClient(
        settings=settings,
        access_token=connection.access_token,
        base_url=connection.graph_base_url,
    )

    # Parent first: an ACTIVE ad under a PAUSED campaign still will not deliver.
    targets: list[tuple[str, str]] = []
    for step in _ACTIVATION_STEPS:
        object_id = created.get(step)
        if isinstance(object_id, str) and object_id:
            targets.append((step, object_id))
    ads_raw = created.get("ads")
    for ad_id in ads_raw if isinstance(ads_raw, list) else []:
        if isinstance(ad_id, str) and ad_id:
            targets.append((f"ad:{ad_id}", ad_id))

    done_raw = execution.get("activated_ids")
    already = [x for x in done_raw if isinstance(x, str)] if isinstance(done_raw, list) else []
    remaining = [(label, oid) for label, oid in targets if label not in already]

    if not remaining:
        campaign.status = "active"
        await db.flush()
        return MetaAdPlanActivationResult(
            status="ok",
            recommendation_id=rec_id,
            detail="Already active; no Meta write was needed.",
            activated=already,
            execution_result=execution,
        )

    activated = list(already)
    for label, object_id in remaining:
        try:
            await graph_client.post(object_id, {"status": "ACTIVE"})
        except MetaGraphError as exc:
            execution["activated_ids"] = list(activated)
            execution["status"] = "partial_active"
            recommendation.execution_result = dict(execution)
            await db.flush()
            return MetaAdPlanActivationResult(
                recommendation_id=rec_id,
                detail=f"Activation stopped at {label}.",
                unavailable_reason=str(exc),
                activated=activated,
                execution_result=execution,
            )
        activated.append(label)
        execution["activated_ids"] = list(activated)
        recommendation.execution_result = dict(execution)
        await db.flush()

    execution["status"] = "active"
    recommendation.execution_result = dict(execution)
    campaign.status = "active"
    await db.flush()
    return MetaAdPlanActivationResult(
        status="ok",
        recommendation_id=rec_id,
        detail=f"Meta objects switched to ACTIVE: {', '.join(activated)}.",
        activated=activated,
        execution_result=execution,
    )


class AdLevelInsight(ApiModel):
    meta_ad_id: str
    meta_ad_name: str = ""
    creative_id: str | None = None
    prompt: str | None = None
    image_url: str | None = None
    reference_analysis_id: str | None = None
    mapped: bool = False
    date_start: date | None = None
    date_end: date | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_at: datetime


class AdLevelInsightsResponse(ApiModel):
    campaign_id: str
    status: Literal["ok", "partial", "unavailable"] = "unavailable"
    reason: str | None = None
    date_start: date
    date_end: date
    ads: list[AdLevelInsight] = Field(default_factory=list)


async def _load_ad_mappings(
    db: AsyncSession, campaign_id: str
) -> dict[str, dict[str, Any]]:
    rows = (
        await db.execute(
            select(Recommendation)
            .where(
                Recommendation.campaign_id == campaign_id,
                Recommendation.action_type == META_AD_PLAN_ACTION,
            )
            .order_by(Recommendation.created_at.desc())
            .limit(100)
        )
    ).scalars().all()
    mapping: dict[str, dict[str, Any]] = {}
    for recommendation in rows:
        execution = recommendation.execution_result or {}
        for ad in execution.get("ads", []) if isinstance(execution, dict) else []:
            if isinstance(ad, dict) and ad.get("meta_ad_id"):
                mapping.setdefault(str(ad["meta_ad_id"]), ad)
    return mapping


async def pull_ad_level_insights(
    db: AsyncSession,
    campaign: Campaign,
    *,
    settings: Settings,
    date_start: date,
    date_end: date,
    graph: MetaGraphClient | None = None,
) -> AdLevelInsightsResponse:
    """Fetch actual Meta ``level=ad`` rows and attach only exact ID matches.

    Unmatched rows are explicitly marked ``mapped=False`` and never inherit a
    prompt or image from another ad. Stored rows are upserted in ``Insight`` JSON
    fields, keyed by (campaign, Meta Ad ID, requested date range).
    """
    now = datetime.now(timezone.utc)
    if date_end < date_start:
        return AdLevelInsightsResponse(
            campaign_id=campaign.id,
            status="unavailable",
            reason="dateEnd must be on or after dateStart.",
            date_start=date_start,
            date_end=date_end,
        )
    if not campaign.meta_campaign_id:
        stored = await _stored_ad_insights(db, campaign.id, date_start, date_end)
        return AdLevelInsightsResponse(
            campaign_id=campaign.id,
            status="partial" if stored else "unavailable",
            reason="The local campaign is not linked to a Meta campaign.",
            date_start=date_start,
            date_end=date_end,
            ads=stored,
        )

    connection = await resolve_meta_connection(db, campaign.business_id, settings)
    if not connection.access_token:
        stored = await _stored_ad_insights(db, campaign.id, date_start, date_end)
        return AdLevelInsightsResponse(
            campaign_id=campaign.id,
            status="partial" if stored else "unavailable",
            reason="Meta is not connected; no new Insights were fetched.",
            date_start=date_start,
            date_end=date_end,
            ads=stored,
        )

    graph_client = graph or HttpMetaGraphClient(
        settings=settings,
        access_token=connection.access_token,
        base_url=connection.graph_base_url,
    )
    params = {
        "level": "ad",
        "fields": ",".join(
            [
                "ad_id",
                "ad_name",
                "adset_id",
                "campaign_id",
                "date_start",
                "date_stop",
                "spend",
                "impressions",
                "reach",
                "frequency",
                "clicks",
                "ctr",
                "cpc",
                "cpm",
                "actions",
            ]
        ),
        "time_range": json.dumps({"since": date_start.isoformat(), "until": date_end.isoformat()}),
        "limit": 500,
    }
    try:
        response = await graph_client.get(f"{campaign.meta_campaign_id}/insights", params)
        raw_rows = response.get("data", [])
        if not isinstance(raw_rows, list):
            raise MetaGraphError("Meta returned an unexpected Insights response.", retry_safe=True)
    except MetaGraphError as exc:
        stored = await _stored_ad_insights(db, campaign.id, date_start, date_end)
        return AdLevelInsightsResponse(
            campaign_id=campaign.id,
            status="partial" if stored else "unavailable",
            reason=(f"{exc} Previously stored Meta rows are shown when available." if stored else str(exc)),
            date_start=date_start,
            date_end=date_end,
            ads=stored,
        )

    mappings = await _load_ad_mappings(db, campaign.id)
    creative_ids = {
        str(item.get("creative_id"))
        for item in mappings.values()
        if item.get("creative_id")
    }
    creative_rows = (
        await db.execute(
            select(Creative).where(
                Creative.id.in_(creative_ids), Creative.business_id == campaign.business_id
            )
        )
    ).scalars().all() if creative_ids else []
    creatives = {row.id: row for row in creative_rows}

    existing_rows = (
        await db.execute(
            select(Insight)
            .where(Insight.campaign_id == campaign.id, Insight.kind == META_AD_INSIGHT_KIND)
            .order_by(Insight.created_at.desc())
            .limit(2000)
        )
    ).scalars().all()
    existing_by_key: dict[tuple[str, str, str], Insight] = {}
    for stored_row in existing_rows:
        payload = stored_row.payload or {}
        if isinstance(payload, dict) and payload.get("meta_ad_id"):
            key = (
                str(payload["meta_ad_id"]),
                str(payload.get("date_start", "")),
                str(payload.get("date_end", "")),
            )
            existing_by_key.setdefault(key, stored_row)

    out: list[AdLevelInsight] = []
    mapped_count = 0
    for raw in raw_rows:
        if not isinstance(raw, dict) or not raw.get("ad_id"):
            continue
        ad_id = str(raw["ad_id"])
        mapping = mappings.get(ad_id)
        creative_id = str(mapping.get("creative_id")) if mapping and mapping.get("creative_id") else None
        creative = creatives.get(creative_id) if creative_id else None
        mapped = mapping is not None and creative is not None
        if mapped:
            mapped_count += 1
        metrics = _numeric_metrics(raw)
        actions = raw.get("actions") if isinstance(raw.get("actions"), list) else []
        row_start = str(raw.get("date_start") or date_start.isoformat())
        row_end = str(raw.get("date_stop") or date_end.isoformat())
        prompt_snapshot = str(mapping.get("creative_prompt") or "") if mapped and mapping else None
        asset_hash = hashlib.sha256(creative.url.encode("utf-8")).hexdigest() if mapped and creative else None
        # The local creative must still match the exact image URL hash and prompt
        # snapshot captured when the Meta Ad was created. If it changed, keep the
        # Meta metrics but deliberately remove the creative association.
        if mapped and mapping and creative:
            mapped = (
                asset_hash == mapping.get("asset_url_sha256")
                and creative.prompt == mapping.get("creative_prompt")
            )
        insight_payload = {
            "source": "meta_marketing_api",
            "meta_ad_id": ad_id,
            "meta_ad_name": str(raw.get("ad_name") or ""),
            "creative_id": creative_id if mapped else None,
            "prompt": prompt_snapshot if mapped else None,
            "reference_analysis_id": mapping.get("reference_analysis_id") if mapped and mapping else None,
            "asset_url_sha256": mapping.get("asset_url_sha256") if mapped and mapping else None,
            "mapped": mapped,
            "date_start": row_start,
            "date_end": row_end,
            "actions": actions,
            "retrieved_at": now.isoformat(),
        }
        key = (ad_id, row_start, row_end)
        record = existing_by_key.get(key)
        if record is None:
            record = Insight(
                id=_new_id("ins"),
                campaign_id=campaign.id,
                kind=META_AD_INSIGHT_KIND,
                title=str(raw.get("ad_name") or f"Meta Ad {ad_id}"),
                detail=(
                    f"Meta Ad {ad_id} linked to Creative {creative_id}."
                    if mapped and creative_id
                    else f"Meta Ad {ad_id}; no exact local Creative ID mapping was found."
                ),
                severity="info",
                metrics=metrics,
                payload=insight_payload,
            )
            db.add(record)
            existing_by_key[key] = record
        else:
            record.title = str(raw.get("ad_name") or f"Meta Ad {ad_id}")
            record.metrics = metrics
            record.payload = insight_payload
            record.detail = (
                f"Meta Ad {ad_id} linked to Creative {creative_id}."
                if mapped and creative_id
                else f"Meta Ad {ad_id}; no exact local Creative ID mapping was found."
            )
        out.append(
            AdLevelInsight(
                meta_ad_id=ad_id,
                meta_ad_name=str(raw.get("ad_name") or ""),
                creative_id=creative_id if mapped else None,
                prompt=str(mapping.get("creative_prompt") or "") if mapped and mapping else None,
                image_url=creative.url if mapped and creative else None,
                reference_analysis_id=mapping.get("reference_analysis_id") if mapped and mapping else None,
                mapped=mapped,
                date_start=date.fromisoformat(row_start),
                date_end=date.fromisoformat(row_end),
                metrics=metrics,
                actions=actions,
                retrieved_at=now,
            )
        )
    await db.flush()

    status: Literal["ok", "partial", "unavailable"]
    if out and mapped_count == len(out):
        status = "ok"
        reason = None
    elif out:
        status = "partial"
        reason = "Meta returned ad-level rows; rows without exact local Creative IDs are marked unmapped."
    else:
        status = "partial"
        reason = "Meta returned no ad-level Insight rows for this date range."
    return AdLevelInsightsResponse(
        campaign_id=campaign.id,
        status=status,
        reason=reason,
        date_start=date_start,
        date_end=date_end,
        ads=out,
    )


async def _stored_ad_insights(
    db: AsyncSession,
    campaign_id: str,
    date_start: date,
    date_end: date,
) -> list[AdLevelInsight]:
    rows = (
        await db.execute(
            select(Insight)
            .where(Insight.campaign_id == campaign_id, Insight.kind == META_AD_INSIGHT_KIND)
            .order_by(Insight.created_at.desc())
            .limit(1000)
        )
    ).scalars().all()
    campaign = await db.get(Campaign, campaign_id)
    business_id = campaign.business_id if campaign is not None else ""
    creative_ids = {
        str((row.payload or {}).get("creative_id"))
        for row in rows
        if isinstance(row.payload, dict) and (row.payload or {}).get("creative_id")
    }
    creative_rows = (
        await db.execute(
            select(Creative).where(
                Creative.id.in_(creative_ids), Creative.business_id == business_id
            )
        )
    ).scalars().all() if creative_ids and business_id else []
    creatives = {creative.id: creative for creative in creative_rows}

    out: list[AdLevelInsight] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        payload = row.payload or {}
        if not isinstance(payload, dict):
            continue
        try:
            start = date.fromisoformat(str(payload.get("date_start", "")))
            end = date.fromisoformat(str(payload.get("date_end", "")))
        except ValueError:
            continue
        if start < date_start or end > date_end:
            continue
        ad_id = str(payload.get("meta_ad_id") or "")
        if not ad_id:
            continue
        key = (ad_id, start.isoformat(), end.isoformat())
        if key in seen:
            continue
        seen.add(key)
        creative_id = str(payload.get("creative_id")) if payload.get("creative_id") else None
        creative = creatives.get(creative_id) if creative_id else None
        exact_match = bool(
            payload.get("mapped")
            and creative is not None
            and hashlib.sha256(creative.url.encode("utf-8")).hexdigest()
            == payload.get("asset_url_sha256")
            and creative.prompt == payload.get("prompt")
        )
        out.append(
            AdLevelInsight(
                meta_ad_id=ad_id,
                meta_ad_name=str(payload.get("meta_ad_name") or row.title),
                creative_id=creative.id if exact_match and creative else None,
                prompt=str(payload.get("prompt")) if exact_match and payload.get("prompt") is not None else None,
                image_url=creative.url if exact_match and creative else None,
                reference_analysis_id=str(payload.get("reference_analysis_id")) if exact_match and payload.get("reference_analysis_id") else None,
                mapped=exact_match,
                date_start=start,
                date_end=end,
                metrics={str(k): float(v) for k, v in (row.metrics or {}).items() if _is_number(v)},
                actions=payload.get("actions") if isinstance(payload.get("actions"), list) else [],
                retrieved_at=_as_datetime(payload.get("retrieved_at"), row.created_at),
            )
        )
    return out


def _as_datetime(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "MetaAdPlanActivationResult",
    "activate_meta_ad_plan",
    "AdLevelInsight",
    "AdLevelInsightsResponse",
    "HttpMetaGraphClient",
    "META_AD_INSIGHT_KIND",
    "META_AD_PLAN_ACTION",
    "META_AD_PLAN_KIND",
    "MetaAdPlanExecutionResult",
    "MetaAdPlanSettings",
    "MetaAdPlanValidationError",
    "MetaGraphClient",
    "MetaGraphError",
    "execute_meta_ad_plan",
    "missing_meta_ad_settings",
    "objective_to_meta",
    "preflight_meta_ad_plan",
    "pull_ad_level_insights",
    "resolve_meta_connection",
    "validate_creative_image_url",
]
