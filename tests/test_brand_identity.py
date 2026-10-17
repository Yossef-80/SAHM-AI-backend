"""Unit and integration tests for Brand Identity service, schemas, and API endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.brand.identity_service import (
    get_brand_identity,
    record_competitors,
    sync_brand_identity_from_chats,
    update_brand_identity,
)
from app.schemas.business import (
    BrandIdentityResponse,
    BrandIdentityUpdateRequest,
    ColorPalette,
    IdentityEssence,
    SWOTAnalysis,
    TargetAudienceProfile,
)


@pytest.mark.asyncio
async def test_get_brand_identity_service(seeded_db: AsyncSession) -> None:
    """Verify get_brand_identity constructs response from seeded business and profile."""
    identity = await get_brand_identity(seeded_db, "biz_test")
    assert isinstance(identity, BrandIdentityResponse)
    assert identity.business_id == "biz_test"
    assert identity.name == "Cairo Coffee Co"
    assert identity.industry == "F&B"
    assert identity.colors is not None
    assert identity.colors.primary.startswith("#")
    assert "Bean There" in identity.competitors
    assert "Kaffa Roasters" in identity.competitors


@pytest.mark.asyncio
async def test_record_competitors(seeded_db: AsyncSession) -> None:
    """Verify record_competitors cleanly dedupes and stores competitors."""
    updated = await record_competitors(
        seeded_db,
        "biz_test",
        ["Bean There", "Specialty Beans EG", "Specialty Beans EG", ""],
    )
    assert "Bean There" in updated
    assert "Specialty Beans EG" in updated
    assert len(updated) >= 3


@pytest.mark.asyncio
async def test_update_brand_identity_service(seeded_db: AsyncSession) -> None:
    """Verify manual update of brand identity persists across database and memory."""
    patch = BrandIdentityUpdateRequest(
        name="Cairo Artisan Roasters",
        colors=ColorPalette(
            primary="#F59E0B",
            secondary="#1E293B",
            accent="#EF4444",
            palette=["#F59E0B", "#1E293B", "#EF4444", "#FEF3C7", "#10B981"],
        ),
        essence=IdentityEssence(
            tagline="Artisanal Roasts for Discerning Tastes",
            brand_voice="Friendly, knowledgeable and direct",
            tone_keywords=["Artisanal", "Direct", "Reliable"],
        ),
        audience=TargetAudienceProfile(
            summary="Specialty coffee connoisseurs across Cairo",
            pain_points=["Overpriced stale commercial coffee"],
        ),
        swot=SWOTAnalysis(
            strengths=["Fresh on-demand roasting"],
            weaknesses=["Single micro-roastery facility"],
            opportunities=["Direct B2B office subscription packages"],
            threats=["Green bean import tariff volatility"],
        ),
        competitors=["Bean There", "Brew Brothers"],
    )

    updated = await update_brand_identity(seeded_db, "biz_test", patch)
    assert updated.name == "Cairo Artisan Roasters"
    assert updated.colors.primary == "#F59E0B"
    assert updated.essence.tagline == "Artisanal Roasts for Discerning Tastes"
    assert "Bean There" in updated.competitors
    assert "Brew Brothers" in updated.competitors

    # Verify re-fetching retrieves updated values
    refetched = await get_brand_identity(seeded_db, "biz_test")
    assert refetched.name == "Cairo Artisan Roasters"
    assert refetched.colors.primary == "#F59E0B"
    assert "Artisanal Roasts for Discerning Tastes" in (refetched.essence.tagline or "")


async def _auth_headers(client: AsyncClient, email: str = "brand_test@example.com") -> dict[str, str]:
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "SecurePassword123!",
            "fullName": "Brand Tester",
            "workspaceName": "Cairo Coffee Co",
        },
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_api_get_brand_identity(api_client: AsyncClient) -> None:
    """Verify GET /api/v1/business/brand-identity endpoint."""
    headers = await _auth_headers(api_client, "get_brand@example.com")
    resp = await api_client.get(
        "/api/v1/business/brand-identity",
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "businessId" in data
    assert "name" in data
    assert "colors" in data
    assert "palette" in data["colors"]
    assert "essence" in data
    assert "audience" in data
    assert "swot" in data
    assert "competitors" in data


@pytest.mark.asyncio
async def test_api_patch_brand_identity(api_client: AsyncClient) -> None:
    """Verify PATCH /api/v1/business/brand-identity endpoint with camelCase payload."""
    headers = await _auth_headers(api_client, "patch_brand@example.com")
    payload = {
        "name": "Nile Roasters Co.",
        "colors": {
            "primary": "#DC2626",
            "secondary": "#0F172A",
            "accent": "#F59E0B",
            "background": "#FFFBEB",
            "text": "#0F172A",
            "palette": ["#DC2626", "#F59E0B", "#0F172A"],
        },
        "essence": {
            "tagline": "Crafted Daily in Cairo",
            "brandVoice": "Energetic and authentic",
        },
        "competitors": ["AbuAuf", "Bean There"],
    }
    resp = await api_client.patch(
        "/api/v1/business/brand-identity",
        json=payload,
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["name"] == "Nile Roasters Co."
    assert data["colors"]["primary"] == "#DC2626"
    assert "AbuAuf" in data["competitors"]


@pytest.mark.asyncio
async def test_api_sync_brand_identity(api_client: AsyncClient) -> None:
    """Verify POST /api/v1/business/brand-identity/sync endpoint synthesizes profile."""
    headers = await _auth_headers(api_client, "sync_brand@example.com")
    resp = await api_client.post(
        "/api/v1/business/brand-identity/sync",
        json={},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "businessId" in data
    assert "colors" in data
    assert "audience" in data
    assert "swot" in data
    assert isinstance(data["competitors"], list)


@pytest.mark.asyncio
async def test_update_brand_identity_tool(seeded_db: AsyncSession) -> None:
    import json
    from app.ai.tools.brand_identity import update_brand_identity_tool

    res_str = await update_brand_identity_tool.ainvoke(
        {
            "primary_color": "yellow",
            "accent_color": "orange",
            "brand_voice": "Energetic and warm",
        },
        config={"configurable": {"tool_db": seeded_db, "business_id": "biz_test"}},
    )
    res = json.loads(res_str)
    assert res.get("status") == "ok", res
    assert res["primary_color"] == "#FACC15"
    assert res["accent_color"] == "#EA580C"
    assert res["brand_voice"] == "Energetic and warm"

    # Verify database was updated immediately
    identity = await get_brand_identity(seeded_db, "biz_test")
    assert identity.colors.primary == "#FACC15"
    assert identity.colors.accent == "#EA580C"
    assert identity.essence.brand_voice == "Energetic and warm"


