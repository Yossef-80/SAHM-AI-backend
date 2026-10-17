"""Opt-in image-generation tool for Layla's LangGraph.

This adapter deliberately uses only server-side image settings and does not
construct a ToolContext. The tool is guarded by a server-set RunnableConfig
flag so it is callable only when the user explicitly asks for an in-app draft.

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
from app.core.tools.media import _compose_image_prompt, _generate_openai_image, _mock_asset

#: The placeholder model id ``config.py`` ships as a default. Sending it to a real
#: provider is always a configuration mistake, never an intent.
_MOCK_MODEL = "mock-image-1"


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
) -> tuple[str, str | None, str | None, str | None, bool]:
    """Return ``(provider, api_key, model, base_url, allow_mock)`` for this run.

    A workspace connector row beats the env var exactly like it does for every
    other tool; the process env is the fallback, not the only source.
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

    provider, api_key, model, base_url, allow_mock = await _resolve_image_configuration(configurable)
    style = dict(style_spec or {})
    if not style and isinstance(configurable.get("reference_analysis_context"), dict):
        ref_style = configurable["reference_analysis_context"].get("style")
        if isinstance(ref_style, dict):
            style = dict(ref_style)
    effective_prompt = _compose_image_prompt(prompt[:4000], style)

    try:
        if provider == "mock":
            if not allow_mock:
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
            asset = _mock_asset(effective_prompt, aspect_ratio, 0)
        elif provider == "openai":
            if not api_key or not model or model == _MOCK_MODEL:
                return _result(
                    "unavailable",
                    reason="Configure IMAGE_API_KEY and IMAGE_MODEL for the selected image provider.",
                )
            assets = await _generate_openai_image(
                effective_prompt,
                aspect_ratio,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
            asset = next((item for item in assets if item.url), None)
            if asset is None:
                return _result("unavailable", reason="The image provider returned no preview URL.")
        else:
            return _result(
                "unavailable",
                reason="The configured image provider is not supported by this Layla adapter.",
            )
    except Exception as exc:  # noqa: BLE001 - never include provider URLs or secrets in the tool result.
        detail = ""
        if hasattr(exc, "response") and getattr(exc.response, "text", None):
            try:
                err_data = exc.response.json()
                msg = err_data.get("error", {}).get("message") or exc.response.text[:200]
                detail = f": {msg}"
            except Exception:
                detail = f": {exc.response.text[:200]}"
        return _result("unavailable", reason=f"Image generation failed ({type(exc).__name__}{detail}).")

    if asset is not None:
        db = configurable.get("tool_db")
        biz_id = configurable.get("business_id")
        campaign_id = configurable.get("campaign_id")
        ref_ctx = configurable.get("reference_analysis_context")
        ref_id = ref_ctx.get("id") if isinstance(ref_ctx, dict) else configurable.get("reference_analysis_id")
        if db is not None and biz_id:
            try:
                import uuid
                from app.db.models import Creative

                creative = Creative(
                    id=f"cr_{uuid.uuid4().hex[:16]}",
                    business_id=biz_id,
                    campaign_id=campaign_id or None,
                    reference_analysis_id=ref_id or None,
                    persona_id="layla",
                    task_name="generate_image_draft",
                    kind="image",
                    url=asset.url,
                    prompt=getattr(asset, "prompt", None) or effective_prompt,
                    aspect_ratio=getattr(asset, "aspect_ratio", None) or aspect_ratio,
                    provider=getattr(asset, "provider", None) or provider,
                    model=getattr(asset, "model", None) or model,
                    style_spec=style or None,
                    status="ok",
                )
                db.add(creative)
                await db.flush()
            except Exception:  # noqa: BLE001
                pass

    return _result("ok", asset=asset)


__all__ = ["LaylaImageGenerationInput", "generate_image_draft_tool"]
