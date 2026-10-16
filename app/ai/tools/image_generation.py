"""Opt-in image-generation tool for Layla's LangGraph.

This adapter deliberately uses only server-side image settings and does not
construct a ToolContext. The tool is guarded by a server-set RunnableConfig
flag so it is callable only when the user explicitly asks for an in-app draft.

Single generation path
----------------------
Generation, durable storage and the ``Creative`` row all happen inside
``app.core.tools.media.generate_creative_assets`` -- the same function the
``/creatives`` screen uses. This module only resolves the workspace's image
configuration and shapes the tool result, so the chat and the creatives screen
can never drift apart again (they used to: chat returned the provider's
expiring URL and stored no creative at all).

Credential resolution
---------------------
When the run carries a ``tool_context`` (the normal chat path) the workspace
connector row is resolved first -- the same three layers ``generate_image`` uses
-- so Settings -> Connectors -> Image generation actually takes effect here too.
Reading ``Settings`` alone was a real bug: a workspace that had configured
``openai`` plus its key in the UI still got a placeholder SVG labelled "MOCK
ASSET" with ``status: ok`` from this tool, while the creatives screen generated
with the real provider. Without a context (direct tool invocation in tests or
scripts) the process env remains the only source, and a ``mock`` preview stays
allowed, which is the pre-existing behaviour.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.tools.media import (
    _generate_openai_image,
    generate_creative_assets,
    persist_creative_assets,
)

__all__ = ["LaylaImageGenerationInput", "generate_image_draft_tool", "_generate_openai_image"]


async def _provider_call(prompt: str, aspect_ratio: str, **kwargs: Any) -> list[Any]:
    """Indirection so monkeypatching ``_generate_openai_image`` on this module
    (tests, and anyone embedding the tool) still intercepts the real call."""
    return await _generate_openai_image(prompt, aspect_ratio, **kwargs)


class LaylaImageGenerationInput(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    style_spec: dict[str, Any] = Field(default_factory=dict)
    aspect_ratio: Literal["1:1", "4:5", "9:16", "16:9"] = "1:1"


def _result(status: str, *, reason: str | None = None, asset: Any = None) -> str:
    payload: dict[str, Any] = {"status": status, "source": "layla_image_generation"}
    if reason:
        payload["reason"] = reason
    if asset is not None:
        payload["asset"] = {
            "url": asset.url,
            "prompt": asset.prompt,
            "provider": asset.provider,
            "model": asset.model,
            "aspectRatio": asset.aspect_ratio,
        }
    return json.dumps(payload, ensure_ascii=False)


async def _resolve_image_configuration(
    configurable: dict[str, Any],
) -> tuple[str, str | None, str | None, str | None, str | None, bool]:
    """Return ``(provider, key, model, base_url, quality, allow_mock)`` for this run.

    A workspace connector row beats the env var exactly like it does for every
    other tool; the process env is the fallback, not the only source. ``quality``
    is part of the tuple for the same reason ``model`` is: this is the path chat
    takes, and until it was forwarded here the quality a workspace saved in
    Settings -> Connectors never reached the request built by Layla.
    """
    settings = get_settings()
    context = configurable.get("tool_context")
    if context is None:
        # No workspace context: keep the historical env-only behaviour, where an
        # explicit in-app request may legitimately produce a labelled mock.
        return (
            str(settings.image_provider).strip().lower(),
            settings.image_api_key,
            settings.image_model,
            settings.image_base_url,
            getattr(settings, "image_quality", None),
            True,
        )

    from app.core.tools.settings import resolve_tool_settings

    resolved = await resolve_tool_settings(context)
    supplied = getattr(context, "user_supplied", {}) or {}
    allow_mock = bool(supplied.get("allow_mock_assets")) if isinstance(supplied, dict) else False
    allow_mock = allow_mock or bool(getattr(settings, "image_allow_mock_assets", False))
    return (
        str(resolved.image_provider or settings.image_provider or "mock").strip().lower(),
        resolved.image_api_key or settings.image_api_key,
        resolved.image_model or settings.image_model,
        resolved.image_base_url or settings.image_base_url,
        getattr(resolved, "image_quality", None) or getattr(settings, "image_quality", None),
        allow_mock,
    )


@tool(args_schema=LaylaImageGenerationInput)
async def generate_image_draft_tool(
    prompt: str,
    config: RunnableConfig,
    style_spec: dict[str, Any] | None = None,
    aspect_ratio: Literal["1:1", "4:5", "9:16", "16:9"] = "1:1",
) -> str:
    """Create one original image draft after an explicit in-app generation request.

    Use only when the user directly asks Layla to generate/render an image inside
    Sahm. This creates a preview only; it never publishes or changes a campaign.
    """
    configurable = (config or {}).get("configurable") or {}
    if configurable.get("allow_image_generation") is not True:
        return _result("unavailable", reason="Image generation was not explicitly requested.")

    provider, api_key, model, base_url, quality, allow_mock = await _resolve_image_configuration(
        configurable
    )
    settings = get_settings()
    style = dict(style_spec or {})

    if provider == "mock" and not allow_mock:
        # Never hand back a placeholder as if it were the requested image:
        # a workspace with a real provider configured must fail loudly here
        # instead of quietly returning an SVG labelled "MOCK ASSET".
        return _result(
            "unavailable",
            reason=(
                "No real image provider is configured for this workspace. Select an "
                "image provider in Settings -> Connectors; no placeholder was created."
            ),
        )
    if provider not in {"openai", "mock"}:
        return _result(
            "unavailable",
            reason="The configured image provider is not supported by this Layla adapter.",
        )
    if provider == "openai" and (not api_key or not model):
        return _result(
            "unavailable",
            reason="Configure IMAGE_API_KEY and IMAGE_MODEL for the selected image provider.",
        )

    assets, reason = await generate_creative_assets(
        cfg=settings,
        prompt=prompt[:4000],
        aspect_ratio=aspect_ratio,
        num_images=1,
        style=style,
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        quality=quality,
        allow_mock_assets=allow_mock,
        image_call=_provider_call,
    )
    if reason or not assets:
        return _result("unavailable", reason=reason or "The image provider returned no preview URL.")
    asset = assets[0]

    # The same durable creative the /creatives screen would create, so the chat
    # draft shows up in the gallery and can be attached to a campaign later.
    context = configurable.get("tool_context")
    db = configurable.get("tool_db") or getattr(context, "db", None)
    business_id = str(
        configurable.get("business_id") or getattr(context, "business_id", "") or ""
    )
    if db is not None and business_id:
        await persist_creative_assets(
            db,
            business_id=business_id,
            assets=[asset],
            campaign_id=str(configurable.get("campaign_id") or "") or getattr(context, "campaign_id", None),
            persona_id=str(configurable.get("agent") or "layla"),
            style=style,
            task_name="generate_image_draft",
        )

    return _result("ok", asset=asset)


__all__ = ["LaylaImageGenerationInput", "generate_image_draft_tool"]
