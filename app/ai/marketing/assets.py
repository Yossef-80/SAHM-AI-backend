"""Compatibility shim over the single image-generation path.

Image generation, durable storage and ``Creative`` rows all live in
``app.core.tools.media`` (``generate_creative_assets`` / ``persist_creative_assets``).
This module used to carry its own provider call with a DALL-E-only size map
(``1024x1280``, ``1024x1792``, ``1792x1024``) that GPT Image models reject, and
its own ``_mock_asset`` helper. Both are gone: keeping one implementation is the
whole point of the merge.

``persist_prompt_creative`` stays here because it is not image generation -- it
stores copy or a prompt draft and never pretends to be a rendered asset.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.tools.media import generate_creative_assets, persist_creative_assets
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
    """Generate assets through the shared path; raise when unavailable."""
    cfg = settings or get_settings()
    connectors = await load_connector_settings(db, business_id, cfg)
    provider = str(connectors.image_provider or cfg.image_provider or "mock").strip().lower()

    assets, reason = await generate_creative_assets(
        cfg=cfg,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        num_images=num_images,
        style=dict(style_spec or {}),
        provider=provider,
        api_key=connectors.image_api_key or cfg.image_api_key,
        model=connectors.image_model or cfg.image_model,
        base_url=connectors.image_base_url or cfg.image_base_url,
        quality=getattr(connectors, "image_quality", None),
    )
    if reason or not assets:
        raise AssetGenerationUnavailable(reason or "The image provider returned no asset.")
    for asset in assets:
        asset.reference_analysis_id = reference_analysis_id
    return assets


async def persist_image_assets(
    db: AsyncSession,
    *,
    business_id: str,
    campaign_id: str | None,
    assets: list[GeneratedAssetRecord],
    style_spec: dict[str, Any] | None = None,
    reference_analysis_id: str | None = None,
) -> None:
    await persist_creative_assets(
        db,
        business_id=business_id,
        assets=assets,
        campaign_id=campaign_id,
        style=style_spec,
        reference_analysis_id=reference_analysis_id,
        status="ready",
    )


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
    include_title: bool = True,
) -> GeneratedAssetRecord:
    """Save copy or a prompt draft without pretending it is a rendered asset.

    ``include_title`` is False for approved ad copy, where the row's ``prompt``
    column *is* the deliverable: prefixing it with a title would pollute the text
    the user reads back, the text a later edit is diffed against, and the example
    the learning loop stores.
    """
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
            prompt=(f"{title}\n\n{prompt}" if include_title and title.strip() else prompt)[
                :12000
            ],
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


__all__ = [
    "AssetGenerationUnavailable",
    "generate_image_assets",
    "persist_image_assets",
    "persist_prompt_creative",
]
