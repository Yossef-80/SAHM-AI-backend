"""Image-provider bridge and prompt-only creative persistence."""

from __future__ import annotations

import base64
import hashlib
import uuid
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.tools.settings import load_connector_settings
from app.db.models import Creative
from app.schemas.tools import GeneratedAssetRecord


class AssetGenerationUnavailable(RuntimeError):
    """Image rendering is not configured or the provider is unavailable."""


async def generate_image_assets(
    db: AsyncSession,
    *,
    business_id: str,
    campaign_id: str | None,
    prompt: str,
    aspect_ratio: str,
    num_images: int,
    style_spec: dict[str, Any] | None = None,
    reference_analysis_id: str | None = None,
    settings: Settings | None = None,
) -> list[GeneratedAssetRecord]:
    cfg = settings or get_settings()
    connectors = await load_connector_settings(db, business_id, cfg)
    provider = connectors.image_provider or cfg.image_provider
    api_key = connectors.image_api_key or cfg.image_api_key
    model = connectors.image_model or cfg.image_model
    base_url = connectors.image_base_url or cfg.image_base_url

    if provider == "mock":
        if cfg.environment not in {"dev", "test"}:
            raise AssetGenerationUnavailable("Mock image generation is disabled outside development and tests.")
        return [_mock_asset(prompt, aspect_ratio, index) for index in range(num_images)]
    if provider == "stability":
        raise AssetGenerationUnavailable("The configured Stability image provider is not wired in this build.")
    if provider != "openai" or not api_key:
        raise AssetGenerationUnavailable("No usable image renderer is configured for this workspace.")

    size = {"1:1": "1024x1024", "4:5": "1024x1280", "9:16": "1024x1792", "16:9": "1792x1024"}.get(aspect_ratio, "1024x1024")
    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/") + "/images/generations"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                endpoint,
                headers={"authorization": f"Bearer {api_key}"},
                json={"model": model, "prompt": prompt, "size": size, "n": num_images},
            )
        if response.is_error:
            raise AssetGenerationUnavailable(f"Image renderer returned HTTP {response.status_code}.")
        rows = response.json().get("data", [])
        assets = [
            GeneratedAssetRecord(
                id=f"asset_{uuid.uuid4().hex[:20]}",
                url=str(item.get("url") or ""),
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                provider="openai",
                model=model,
                style_spec=style_spec or {},
                reference_analysis_id=reference_analysis_id,
            )
            for item in rows
            if item.get("url")
        ]
        if len(assets) != num_images:
            raise AssetGenerationUnavailable("Image renderer returned an incomplete asset set.")
        return assets
    except AssetGenerationUnavailable:
        raise
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        raise AssetGenerationUnavailable("Image renderer request failed or returned an unreadable response.") from exc


async def persist_prompt_creative(
    db: AsyncSession,
    *,
    business_id: str,
    campaign_id: str | None,
    kind: str,
    title: str,
    prompt: str,
    aspect_ratio: str,
    style_spec: dict[str, Any] | None = None,
    reference_analysis_id: str | None = None,
    provider: str = "prompt_only",
    model: str | None = None,
) -> GeneratedAssetRecord:
    """Save copy or a prompt draft without pretending it is a rendered asset."""
    record = GeneratedAssetRecord(
        id=f"draft_{uuid.uuid4().hex[:20]}",
        url="",
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        provider=provider,
        model=model or "text-only",
        style_spec=style_spec or {},
        reference_analysis_id=reference_analysis_id,
    )
    db.add(
        Creative(
            id=record.id,
            business_id=business_id,
            campaign_id=campaign_id,
            kind=kind,
            url="",
            prompt=f"{title}\n\n{prompt}"[:12000],
            aspect_ratio=aspect_ratio,
            provider=provider,
            model=record.model,
            style_spec=style_spec or None,
            reference_analysis_id=reference_analysis_id,
            status="prompt_only" if kind != "copy" else "ready",
        )
    )
    await db.flush()
    return record


async def persist_image_assets(
    db: AsyncSession,
    *,
    business_id: str,
    campaign_id: str | None,
    assets: list[GeneratedAssetRecord],
    style_spec: dict[str, Any] | None = None,
    reference_analysis_id: str | None = None,
) -> None:
    for asset in assets:
        asset.style_spec = style_spec or {}
        asset.reference_analysis_id = reference_analysis_id
        db.add(
            Creative(
                id=asset.id,
                business_id=business_id,
                campaign_id=campaign_id,
                task_name="generate_image",
                kind="image",
                url=asset.url,
                prompt=asset.prompt,
                aspect_ratio=asset.aspect_ratio,
                provider=asset.provider,
                model=asset.model,
                style_spec=style_spec or None,
                reference_analysis_id=reference_analysis_id,
                status="ready",
            )
        )
    await db.flush()


def _mock_asset(prompt: str, aspect_ratio: str, index: int) -> GeneratedAssetRecord:
    """Clearly labelled deterministic asset, permitted only in dev/test."""
    digest = hashlib.sha256(f"{prompt}|{index}".encode()).hexdigest()[:10]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1080">'
        '<rect width="100%" height="100%" fill="#111827"/>'
        '<text x="50%" y="48%" text-anchor="middle" fill="#f9fafb" '
        'font-family="sans-serif" font-size="48">MOCK PREVIEW</text>'
        f'<text x="50%" y="56%" text-anchor="middle" fill="#9ca3af" '
        f'font-family="sans-serif" font-size="24">{digest}</text></svg>'
    )
    encoded = base64.b64encode(svg.encode()).decode()
    return GeneratedAssetRecord(
        id=f"asset_{uuid.uuid4().hex[:20]}",
        url=f"data:image/svg+xml;base64,{encoded}",
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        provider="mock",
        model="mock-preview",
    )


__all__ = [
    "AssetGenerationUnavailable",
    "generate_image_assets",
    "persist_image_assets",
    "persist_prompt_creative",
]
