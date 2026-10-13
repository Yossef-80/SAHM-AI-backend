"""Opt-in image-generation tool for Layla's LangGraph.

This adapter deliberately uses only server-side image settings and does not
construct a ToolContext. The tool is guarded by a server-set RunnableConfig
flag so it is callable only when the user explicitly asks for an in-app draft.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.tools.media import _compose_image_prompt, _generate_openai_image, _mock_asset


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

    settings = get_settings()
    provider = str(settings.image_provider)
    style = dict(style_spec or {})
    effective_prompt = _compose_image_prompt(prompt[:4000], style)

    try:
        if provider == "mock":
            asset = _mock_asset(effective_prompt, aspect_ratio, 0)
        elif provider == "openai":
            api_key = settings.image_api_key
            model = settings.image_model
            if not api_key or not model or model == "mock-image-1":
                return _result(
                    "unavailable",
                    reason="Configure IMAGE_API_KEY and IMAGE_MODEL for the selected image provider.",
                )
            assets = await _generate_openai_image(
                effective_prompt,
                aspect_ratio,
                api_key=api_key,
                model=model,
                base_url=settings.image_base_url,
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
        return _result("unavailable", reason=f"Image generation failed ({type(exc).__name__}).")

    return _result("ok", asset=asset)


__all__ = ["LaylaImageGenerationInput", "generate_image_draft_tool"]
