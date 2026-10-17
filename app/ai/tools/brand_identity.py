"""Direct Brand Identity & Visual Style update tool for conversational agents."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.memory.brand_memory import business_id_from
from app.ai.security import redact_sensitive_data
from app.core.logging import get_logger

logger = get_logger(__name__)

_COLOR_NAME_MAP = {
    "yellow": "#FACC15",
    "gold": "#F59E0B",
    "orange": "#EA580C",
    "amber": "#D97706",
    "blue": "#2563EB",
    "navy": "#1E3A8A",
    "sky": "#0EA5E9",
    "green": "#16A34A",
    "emerald": "#059669",
    "red": "#DC2626",
    "crimson": "#991B1B",
    "purple": "#9333EA",
    "violet": "#7C3AED",
    "pink": "#EC4899",
    "black": "#0F172A",
    "dark": "#1E293B",
    "white": "#FFFFFF",
    "gray": "#64748B",
    "grey": "#64748B",
    "أصفر": "#FACC15",
    "اصفر": "#FACC15",
    "ذهبي": "#F59E0B",
    "برتقالي": "#EA580C",
    "أزرق": "#2563EB",
    "ازرق": "#2563EB",
    "كحلي": "#1E3A8A",
    "سماوي": "#0EA5E9",
    "أخضر": "#16A34A",
    "اخضر": "#16A34A",
    "أحمر": "#DC2626",
    "احمر": "#DC2626",
    "بنفسجي": "#9333EA",
    "موف": "#7C3AED",
    "وردي": "#EC4899",
    "بينك": "#EC4899",
    "أسود": "#0F172A",
    "اسود": "#0F172A",
    "أبيض": "#FFFFFF",
    "ابيض": "#FFFFFF",
    "رمادي": "#64748B",
    "رصاصي": "#64748B",
}


def _normalize_hex(color: str | None) -> str | None:
    if not color:
        return None
    val = str(color).strip().lower()
    if val in _COLOR_NAME_MAP:
        return _COLOR_NAME_MAP[val]
    for name, hex_code in _COLOR_NAME_MAP.items():
        if name in val:
            return hex_code
    if not val.startswith("#") and len(val) in (6, 8):
        val = f"#{val}"
    if re.match(r"^#[0-9a-fA-F]{6}$", val) or re.match(r"^#[0-9a-fA-F]{8}$", val):
        return val.upper()
    return None


class UpdateBrandIdentityInput(BaseModel):
    primary_color: str | None = Field(
        default=None,
        description="Primary brand hex color code, e.g. '#FACC15' or '#F59E0B'. If the user mentions a color name in Arabic or English like 'yellow' or 'أصفر', provide the hex or name.",
    )
    secondary_color: str | None = Field(
        default=None,
        description="Secondary brand hex color code, e.g. '#1E293B'.",
    )
    accent_color: str | None = Field(
        default=None,
        description="Accent brand hex color code, e.g. '#EA580C' or '#F97316'.",
    )
    palette_colors: list[str] | None = Field(
        default=None,
        description="List of 3-5 cohesive hex color codes for the brand palette.",
    )
    brand_voice: str | None = Field(
        default=None,
        description="Tone of voice or communication personality, e.g. 'Warm, expert, transparent'.",
    )
    tagline: str | None = Field(
        default=None,
        description="Brand slogan or tagline.",
    )
    target_audience: str | None = Field(
        default=None,
        description="Summary of the target customer/audience.",
    )


@tool(args_schema=UpdateBrandIdentityInput)
async def update_brand_identity_tool(
    primary_color: str | None = None,
    secondary_color: str | None = None,
    accent_color: str | None = None,
    palette_colors: list[str] | None = None,
    brand_voice: str | None = None,
    tagline: str | None = None,
    target_audience: str | None = None,
    config: RunnableConfig = None,
) -> str:
    """Update and persist brand identity attributes (colors, voice, tagline, audience) directly to the database.

    Call this immediately whenever the user asks to change, update, or set the brand color,
    visual identity, tone of voice, or tagline. This updates the Brand Identity profile instantly.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    business_id = configurable.get("business_id") or business_id_from(config)
    db = configurable.get("tool_db") or configurable.get("db")

    norm_primary = _normalize_hex(primary_color)
    norm_secondary = _normalize_hex(secondary_color)
    norm_accent = _normalize_hex(accent_color)
    norm_palette = (
        [_normalize_hex(c) for c in palette_colors if _normalize_hex(c)]
        if palette_colors
        else None
    )

    if db is not None and business_id:
        try:
            from app.core.brand.identity_service import get_brand_identity, update_brand_identity
            from app.schemas.business import (
                BrandIdentityUpdateRequest,
                ColorPalette,
                IdentityEssence,
                TargetAudienceProfile,
            )

            current = await get_brand_identity(db, business_id)

            updated_colors = None
            if norm_primary or norm_secondary or norm_accent or norm_palette:
                p_col = norm_primary or current.colors.primary
                s_col = norm_secondary or current.colors.secondary
                a_col = norm_accent or current.colors.accent
                palette = norm_palette or [p_col, a_col, s_col, "#FFF7ED", "#0F172A"]
                updated_colors = ColorPalette(
                    primary=p_col,
                    secondary=s_col,
                    accent=a_col,
                    background=current.colors.background or "#F8FAFC",
                    text=current.colors.text or "#0F172A",
                    palette=palette,
                )

            updated_essence = None
            if brand_voice or tagline:
                updated_essence = IdentityEssence(
                    tagline=tagline or current.essence.tagline,
                    mission=current.essence.mission,
                    vision=current.essence.vision,
                    brand_voice=brand_voice or current.essence.brand_voice,
                    tone_keywords=current.essence.tone_keywords,
                )

            updated_audience = None
            if target_audience:
                updated_audience = TargetAudienceProfile(
                    summary=target_audience,
                    demographics=current.audience.demographics,
                    psychographics=current.audience.psychographics,
                    pain_points=current.audience.pain_points,
                    core_segments=current.audience.core_segments,
                )

            patch = BrandIdentityUpdateRequest(
                colors=updated_colors,
                essence=updated_essence,
                audience=updated_audience,
            )
            await update_brand_identity(db, business_id, patch)
            await db.commit()

            logger.info(
                "brand_identity_updated_from_chat",
                extra={
                    "business_id": business_id,
                    "primary_color": norm_primary,
                    "brand_voice_changed": bool(brand_voice),
                },
            )
            return json.dumps(
                {
                    "status": "ok",
                    "message": "Brand identity updated successfully in database.",
                    "primary_color": norm_primary,
                    "accent_color": norm_accent,
                    "brand_voice": brand_voice,
                    "tagline": tagline,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.warning("brand_identity_update_tool_failed", extra={"error": str(exc)})
            return json.dumps({"status": "error", "message": f"Failed to update brand identity: {exc}"})

    return json.dumps(
        {
            "status": "ok",
            "message": "Brand identity update recorded (offline).",
            "primary_color": norm_primary,
        },
        ensure_ascii=False,
    )
