"""Workspace settings: ``/api/v1/settings/connectors`` (P2.4).

This is the "fill in all the fields the tools need" surface. Before P2.4 every
provider credential was a process-wide env var, so two customers could never use
two different Meta ad accounts: whoever set ``META_APP_ID`` last set it for
everybody. Three endpoints make the configuration *data*:

* ``GET``    -- what is effective right now, and which layer it came from.
* ``PATCH``  -- set this workspace's override (owner only).
* ``DELETE`` -- drop the override and fall back to the env (owner only).

The secrets rule
----------------
A secret is write-only. It is encrypted with ``TokenCipher`` on the way in and is
never present in a response body, a log line, or an audit row -- ``GET`` reports
``hasValue`` and a mask. :mod:`app.schemas.settings` declares secrets as
``SecretStr`` so even an accidental repr cannot print one.

Layering
--------
Resolution order is workspace row -> process env -> built-in default, and it is
implemented once in :mod:`app.core.tools.settings`. This router only renders what
that module resolved; it does not re-implement the merge, which is what keeps
``GET /settings/connectors`` and the tool catalogue telling the same story.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import AuthDep, SessionDep, SettingsDep, WorkspaceDep, require_role
from app.core.errors import NotFoundError, ValidationFailed
from app.core.logging import get_logger
from app.core.security import TokenCipher
from app.core.tools.settings import (
    CONNECTOR_ORDER,
    CONNECTOR_SPECS,
    SECRET_MASK,
    ConnectorSpec,
    ResolvedConnector,
    connector_spec,
    load_connector_settings,
)
from app.core.llm.model_compat import AVAILABLE_REASONING_EFFORTS
from app.core.llm.router import invalidate_workspace_llm_cache, resolve_workspace_tier
from app.db.models import LlmProviderSettings, ToolSettings as ToolSettingsRow
from app.schemas.llm_settings import LlmSettingsOut, LlmSettingsUpdate, LlmTierOut
from app.schemas.settings import (
    ConnectorDeleteResponse,
    ConnectorFieldOut,
    ConnectorOut,
    ConnectorSettingsResponse,
    ConnectorUpdateRequest,
)

#: The four values this build accepts, imported from the compat layer so the
#: validated set and the forwarded set cannot drift. Anything else is rejected
#: here rather than sent and refused as a provider-side "unsupported value".
REASONING_EFFORT_VALUES = AVAILABLE_REASONING_EFFORTS


router = APIRouter(prefix="/settings", tags=["settings"])

_logger = get_logger(__name__)

#: Connector settings are workspace-wide configuration: owner only, matching the
#: P1.7 role matrix ("Connector settings (Meta, Places, keys): owner").
MANAGE_CONNECTORS = "owner"


def _require_spec(connector: str) -> ConnectorSpec:
    """Unknown connector -> 404.

    A 422 would suggest the body was wrong; the truth is that this build has no
    such connector at all, which is a missing resource.
    """
    spec = connector_spec(connector)
    if spec is None:
        raise NotFoundError(
            f"connector '{connector}' does not exist",
            detail={"known": list(CONNECTOR_ORDER)},
        )
    return spec


def _render(resolved: ResolvedConnector, *, has_override: bool) -> ConnectorOut:
    """Serialise one resolved connector. Secrets never reach the wire."""
    fields: list[ConnectorFieldOut] = []
    for item in resolved.fields:
        secret = item.spec.secret
        fields.append(
            ConnectorFieldOut(
                key=item.key,
                label=item.spec.label,
                secret=secret,
                required=item.spec.required,
                max_length=item.spec.max_length,
                description=item.spec.description,
                choices=list(item.spec.choices),
                value=None if secret else item.value,
                has_value=item.has_value,
                masked=SECRET_MASK if (secret and item.has_value) else None,
                source=item.source,
            )
        )
    return ConnectorOut(
        connector=resolved.spec.name,
        label=resolved.spec.label,
        source=resolved.source,
        has_override=has_override,
        configured=resolved.configured,
        provider=resolved.provider or None,
        providers=list(resolved.spec.allowed_providers),
        missing_required=list(resolved.missing_required),
        placeholder=resolved.is_placeholder,
        fields=fields,
        notes=resolved.spec.notes,
    )


def _validate_provider_selection(
    spec: ConnectorSpec, resolved: ResolvedConnector, sent: dict[str, Any]
) -> None:
    """Refuse a save that stores credentials the selected provider will never read.

    The failure this prevents: a user fills in the image API key and model, never
    touches the provider box (which defaults to ``mock``), gets a 200 with a green
    toast, and then watches image generation stay disabled -- because
    ``generate_image`` only reads ``image_api_key``/``image_model`` when the
    provider is a real one. Nothing in the old response said so.

    Only the *credential* fields are checked, so clearing a provider on its own
    (or saving an unrelated field) is still allowed.
    """
    if spec.provider_field is None:
        return
    credentials = sorted(
        key for key in sent if key != spec.provider_field
    )
    if not credentials or not resolved.is_placeholder:
        return
    provider = resolved.provider or "(unset)"
    allowed = ", ".join(spec.allowed_providers) or "n/a"
    raise ValidationFailed(
        (
            f"{', '.join(repr(key) for key in credentials)} "
            f"{'was' if len(credentials) == 1 else 'were'} set for connector "
            f"'{spec.name}', but its provider resolves to '{provider}', which only "
            f"returns placeholder data and never reads them. Send "
            f"'{spec.provider_field}' with a real provider in the same request "
            f"(accepted: {allowed}) -- nothing was saved."
        ),
        detail={
            "code": "provider_not_selected",
            "connector": spec.name,
            "provider_field": spec.provider_field,
            "provider": provider,
            "accepted": list(spec.allowed_providers),
            "fields": credentials,
        },
    )


def _validate_provider_value(spec: ConnectorSpec, sent: dict[str, Any]) -> None:
    """Reject a provider this build does not implement, and normalise the case.

    ``"OpenAI"`` is stored as ``"openai"`` because every comparison against the
    provider happens on the stored string (``media.py`` compares it directly, and
    ``models.py``-backed rows are never validated by a Literal). A name that is
    simply wrong -- the model id typed into the provider box, say -- is a 422
    listing what is accepted, rather than a save that silently does nothing.
    """
    if spec.provider_field is None or not spec.allowed_providers:
        return
    raw = sent.get(spec.provider_field)
    if raw is None:
        return
    value = str(raw).strip().lower()
    if not value:
        sent[spec.provider_field] = None
        return
    if value not in spec.allowed_providers:
        raise ValidationFailed(
            (
                f"'{raw}' is not a supported '{spec.provider_field}' for connector "
                f"'{spec.name}'. Accepted: {', '.join(spec.allowed_providers)}."
            ),
            detail={
                "code": "unknown_provider",
                "connector": spec.name,
                "provider_field": spec.provider_field,
                "accepted": list(spec.allowed_providers),
            },
        )
    sent[spec.provider_field] = value


def _validate_field_choices(spec: ConnectorSpec, sent: dict[str, Any]) -> None:
    """Reject a value outside a field's declared set, and normalise its case.

    Mirrors ``_validate_provider_value`` for every field that declares
    ``choices`` -- today the image connector's ``quality``. A typo there used to
    be indistinguishable from a working setting, because the call site falls back
    to the provider default; a 422 that names the accepted values is the only
    outcome a user can act on.
    """
    for field_spec in spec.fields:
        if not field_spec.choices or field_spec.key not in sent:
            continue
        raw = sent[field_spec.key]
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if not value:
            sent[field_spec.key] = None
            continue
        if value not in field_spec.choices:
            raise ValidationFailed(
                (
                    f"'{raw}' is not a valid '{field_spec.key}' for connector "
                    f"'{spec.name}'. Accepted: {', '.join(field_spec.choices)}."
                ),
                detail={
                    "code": "invalid_field_value",
                    "connector": spec.name,
                    "field": field_spec.key,
                    "accepted": list(field_spec.choices),
                },
            )
        sent[field_spec.key] = value


@router.get("/connectors", response_model=ConnectorSettingsResponse)
async def get_connectors(
    ws: WorkspaceDep, db: SessionDep, settings: SettingsDep
) -> ConnectorSettingsResponse:
    """Effective configuration per connector, with per-field provenance.

    Readable by any role: a form has to render the current state before it can
    decide what to ask for. Secrets appear as ``hasValue`` only.
    """
    resolved = await load_connector_settings(db, ws.business_id, settings)
    rows = {
        row.connector: row
        for row in (
            await db.execute(
                select(ToolSettingsRow).where(
                    ToolSettingsRow.business_id == ws.business_id
                )
            )
        )
        .scalars()
        .all()
    }
    rendered = [
        _render(
            resolved.connectors[name],
            has_override=name in rows,
        )
        for name in CONNECTOR_ORDER
        if name in resolved.connectors
    ]
    return ConnectorSettingsResponse(
        business_id=ws.business_id,
        environment=settings.environment,
        override_count=len(rows),
        connectors=rendered,
        encryption_key_configured=bool(settings.token_encryption_key),
    )


@router.patch("/connectors/{connector}", response_model=ConnectorOut)
async def update_connector(
    connector: str,
    payload: ConnectorUpdateRequest,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> ConnectorOut:
    """Set this workspace's override for one connector. Owner only.

    Secrets are encrypted with ``TokenCipher`` on the way in and are never echoed.
    Omitted keys keep their stored value; ``null`` or ``""`` clears one.
    """
    await require_role(principal, settings, ws, minimum=MANAGE_CONNECTORS)
    spec = _require_spec(connector)

    sent = payload.sent()
    allowed = {field.key for field in spec.fields}
    unknown = sorted(set(sent) - allowed)
    if unknown:
        raise ValidationFailed(
            f"field(s) {unknown} do not belong to connector '{connector}'",
            detail={"connector": connector, "accepted": sorted(allowed)},
        )
    if not sent:
        raise ValidationFailed(
            "no fields to update; send at least one field in the body",
            detail={"accepted": sorted(allowed)},
        )

    # Validate before any row is touched: the provider decides whether the
    # credential fields mean anything at all.
    _validate_provider_value(spec, sent)
    _validate_field_choices(spec, sent)

    stmt = select(ToolSettingsRow).where(
        ToolSettingsRow.business_id == ws.business_id,
        ToolSettingsRow.connector == connector,
    )
    row = (await db.execute(stmt)).scalars().first()
    if row is None:
        row = ToolSettingsRow(
            id=f"ts_{uuid.uuid4().hex[:24]}",
            business_id=ws.business_id,
            connector=connector,
        )
        db.add(row)

    cipher = TokenCipher(settings.token_encryption_key)
    for field_spec in spec.fields:
        if field_spec.key not in sent:
            continue
        value = sent[field_spec.key]
        text = value.strip() if isinstance(value, str) else value
        if field_spec.db_column is None:  # pragma: no cover - all fields have one
            continue
        if len(text or "") > field_spec.max_length:
            raise ValidationFailed(
                f"'{field_spec.key}' exceeds {field_spec.max_length} characters",
                detail={"field": field_spec.key, "max_length": field_spec.max_length},
            )
        if field_spec.secret:
            # Empty clears; anything else is stored as Fernet ciphertext. The
            # plaintext goes out of scope here and is never logged or returned.
            setattr(row, field_spec.db_column, cipher.encrypt(text) if text else None)
        else:
            setattr(row, field_spec.db_column, text or None)
    row.updated_by = principal.user_id if principal is not None else None

    # Resolve the row exactly as a tool call would, then refuse a save that left
    # credentials attached to a placeholder provider. Flushing first means the
    # check sees the post-patch values; rolling back keeps the 422 atomic.
    await db.flush()
    resolved_after = (
        await load_connector_settings(db, ws.business_id, settings)
    ).connectors[connector]
    try:
        _validate_provider_selection(spec, resolved_after, sent)
    except ValidationFailed:
        await db.rollback()
        raise

    await db.commit()
    # Never log the values -- not even a count of characters, which would be a
    # length oracle for a pasted token.
    _logger.info(
        "connector_settings_updated",
        extra={
            "connector": connector,
            "business_id": ws.business_id,
            "fields": sorted(sent),
        },
    )

    resolved = await load_connector_settings(db, ws.business_id, settings)
    return _render(resolved.connectors[connector], has_override=True)


@router.delete("/connectors/{connector}", response_model=ConnectorDeleteResponse)
async def delete_connector(
    connector: str,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> ConnectorDeleteResponse:
    """Remove this workspace's override so the connector falls back to env.

    Idempotent: deleting an override that does not exist is a 200 with
    ``removed=false`` rather than a 404, because the caller's intent -- "this
    workspace should not override it" -- is satisfied either way.
    """
    await require_role(principal, settings, ws, minimum=MANAGE_CONNECTORS)
    _require_spec(connector)

    stmt = select(ToolSettingsRow).where(
        ToolSettingsRow.business_id == ws.business_id,
        ToolSettingsRow.connector == connector,
    )
    row = (await db.execute(stmt)).scalars().first()
    if row is None:
        resolved_empty = await load_connector_settings(db, ws.business_id, settings)
        fallback = resolved_empty.connectors[connector]
        return ConnectorDeleteResponse(
            connector=connector,
            removed=False,
            source=fallback.source,
            configured=fallback.configured,
        )

    await db.delete(row)
    await db.commit()
    _logger.info(
        "connector_settings_deleted",
        extra={"connector": connector, "business_id": ws.business_id},
    )

    resolved = await load_connector_settings(db, ws.business_id, settings)
    fallback = resolved.connectors[connector]
    return ConnectorDeleteResponse(
        connector=connector,
        removed=True,
        source=fallback.source,
        configured=fallback.configured,
    )


def _llm_tier_out(values: dict[str, object]) -> LlmTierOut:
    return LlmTierOut(
        provider=values["provider"],
        model=values["model"],
        base_url=values["base_url"],
        # Effective, so the form can show what is actually in force: the
        # workspace value when set, otherwise None ("process default").
        max_tokens=values.get("max_tokens"),  # type: ignore[arg-type]
        temperature=values.get("temperature"),  # type: ignore[arg-type]
        reasoning_effort=values.get("reasoning_effort"),  # type: ignore[arg-type]
        key_configured=values["api_key"] is not None,
        key_source=values["key_source"],
        extra_headers_configured=bool(values["extra_headers"]),
        extra_headers_source=values["headers_source"],
    )


@router.get("/llm", response_model=LlmSettingsOut)
async def get_llm_settings(
    principal: AuthDep,
    ws: WorkspaceDep,
    db: SessionDep,
    settings: SettingsDep,
) -> LlmSettingsOut:
    """Get effective LLM routing per tier; API keys are never returned."""
    fast = await resolve_workspace_tier(db, ws.business_id, "fast", settings)
    strong = await resolve_workspace_tier(db, ws.business_id, "strong", settings)
    row = await db.scalar(
        select(LlmProviderSettings).where(
            LlmProviderSettings.business_id == ws.business_id
        )
    )
    return LlmSettingsOut(
        business_id=ws.business_id,
        fast=_llm_tier_out(fast),
        strong=_llm_tier_out(strong),
        has_workspace_override=row is not None,
        encryption_key_configured=bool(settings.token_encryption_key),
    )


@router.patch("/llm", response_model=LlmSettingsOut)
async def update_llm_settings(
    payload: LlmSettingsUpdate,
    principal: AuthDep,
    settings: SettingsDep,
    ws: WorkspaceDep,
    db: SessionDep,
) -> LlmSettingsOut:
    """Save per-workspace provider choices; encrypt keys and custom headers."""
    await require_role(principal, settings, ws, minimum="owner")
    row = await db.scalar(
        select(LlmProviderSettings).where(
            LlmProviderSettings.business_id == ws.business_id
        )
    )
    patches = {"fast": payload.fast, "strong": payload.strong}
    if row is None:
        row = LlmProviderSettings(
            id=f"llm_{uuid.uuid4().hex[:24]}",
            business_id=ws.business_id,
        )
        db.add(row)

    cipher = TokenCipher(settings.token_encryption_key)
    for tier, patch in patches.items():
        if patch is None:
            continue
        sent = patch.model_dump(exclude_unset=True)
        for field in (
            "provider",
            "model",
            "base_url",
            # Sampling controls. `null` is sent explicitly by the UI to mean
            # "remove my override", so they are written whenever the key is
            # present, not only when it carries a value.
            "max_tokens",
            "temperature",
            "reasoning_effort",
        ):
            if field not in sent:
                continue
            value = sent[field]
            if field == "reasoning_effort" and isinstance(value, str):
                value = value.strip().lower() or None
                if value is not None and value not in REASONING_EFFORT_VALUES:
                    raise ValidationFailed(
                        "'reasoning_effort' must be one of "
                        + ", ".join(REASONING_EFFORT_VALUES),
                        detail={
                            "code": "invalid_reasoning_effort",
                            "field": "reasoning_effort",
                            "value": value,
                            "accepted": list(REASONING_EFFORT_VALUES),
                        },
                    )
            setattr(row, f"{tier}_{field}", value.strip() or None if isinstance(value, str) else value)
        if "api_key" in sent:
            secret = sent["api_key"]
            value = secret.get_secret_value().strip() if secret else ""
            setattr(row, f"{tier}_api_key_enc", cipher.encrypt(value) if value else None)
        if "extra_headers" in sent:
            headers = sent["extra_headers"] or {}
            if len(headers) > 20 or any(len(k) > 100 or len(v) > 1000 for k, v in headers.items()):
                raise ValidationFailed("custom header values exceed the supported size")
            encrypted = cipher.encrypt(json.dumps(headers, ensure_ascii=False)) if headers else None
            setattr(row, f"{tier}_extra_headers_enc", encrypted)
    row.updated_by = principal.user_id if principal is not None else None
    await db.commit()
    invalidate_workspace_llm_cache(ws.business_id)

    fast = await resolve_workspace_tier(db, ws.business_id, "fast", settings)
    strong = await resolve_workspace_tier(db, ws.business_id, "strong", settings)
    return LlmSettingsOut(
        business_id=ws.business_id,
        fast=_llm_tier_out(fast),
        strong=_llm_tier_out(strong),
        has_workspace_override=True,
        encryption_key_configured=bool(settings.token_encryption_key),
    )


__all__ = ["CONNECTOR_SPECS", "router"]
