from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.ai.agents.nour.graph import NOUR_TOOLS
from app.ai.settings import get_ai_settings
from app.ai.tools import fetch_competitor_ads, fetch_competitor_ads_tool
from app.ai.tools.fetch_competitor_ads import (
    FetchCompetitorAdsInput,
    clean_page_url,
    extract_ad,
    top_keywords,
    _save_to_memory,
)


def test_clean_page_url_success():
    """Valid Facebook Page URLs are cleansed and normalized."""
    url = "https://www.facebook.com/KFCEgypt/?ref=page_internal#about"
    cleaned = clean_page_url(url)
    assert cleaned == "https://www.facebook.com/KFCEgypt"

    url_fb = "http://fb.com/brandpage"
    assert clean_page_url(url_fb) == "https://www.facebook.com/brandpage"


def test_clean_page_url_invalid_host():
    """Non-Facebook hosts are rejected."""
    with pytest.raises(ValueError, match="must be a Facebook Page URL"):
        clean_page_url("https://instagram.com/mybrand")


def test_clean_page_url_no_path():
    """Root Facebook URL without a page path is rejected."""
    with pytest.raises(ValueError, match="point to a specific Facebook Page"):
        clean_page_url("https://www.facebook.com/")


def test_input_validation():
    """Pydantic input schema validation."""
    inp = FetchCompetitorAdsInput(
        competitor_name="KFC Egypt",
        competitor_url="https://www.facebook.com/KFCEgypt",
    )
    assert inp.competitor_name == "KFC Egypt"

    with pytest.raises(ValidationError):
        FetchCompetitorAdsInput(competitor_name="KFC Egypt")


def test_tool_registered_in_nour_tools():
    """Verifies fetch_competitor_ads is registered in Nour's toolset."""
    tool_names = [t.name for t in NOUR_TOOLS]
    assert "fetch_competitor_ads" in tool_names


def test_extract_ad():
    """Extracts caption, cards, and media URLs correctly."""
    raw = {
        "adArchiveID": "12345678",
        "pageName": "KFC Egypt",
        "startDate": "2026-01-01",
        "isActive": True,
        "snapshot": {
            "body": "Special Ramadan Family Meal Offer!",
            "title": "Order Online Now",
            "images": [
                {"imageUrl": "https://cdn.example.com/ad1.jpg"},
                {"imageUrl": "https://cdn.example.com/ad2.jpg"},
            ],
            "videos": [
                {"videoUrl": "https://cdn.example.com/ad_video.mp4"},
            ],
            "cards": [
                {
                    "title": "Combo 1",
                    "originalImageUrl": "https://cdn.example.com/card1.jpg",
                }
            ],
        },
    }

    ad = extract_ad(raw, 0)
    assert ad["ad_id"] == "12345678"
    assert ad["page_name"] == "KFC Egypt"
    assert "Special Ramadan Family Meal Offer!" in ad["caption"]
    assert "Order Online Now" in ad["caption"]
    assert "https://cdn.example.com/ad1.jpg" in ad["image_urls"]
    assert "https://cdn.example.com/card1.jpg" in ad["image_urls"]
    assert "https://cdn.example.com/ad_video.mp4" in ad["video_urls"]
    assert ad["is_active"] is True


def test_top_keywords():
    """Counts English and Arabic keywords while ignoring stopwords."""
    ads = [
        {"caption": "Delicious burger and crispy chicken offer", "card_captions": []},
        {"caption": "عرض دجاج مقرمش لا يفوتك chicken", "card_captions": []},
    ]
    keywords = top_keywords(ads, n=5)
    kw_dict = {item["keyword"]: item["ad_count"] for item in keywords}
    assert "chicken" in kw_dict
    assert kw_dict["chicken"] == 2
    assert "and" not in kw_dict  # stopword


@pytest.mark.asyncio
async def test_fetch_competitor_ads_scrape_error():
    """Handles scraper exception gracefully and returns structured error payload."""
    with patch("app.ai.tools.fetch_competitor_ads._scrape", side_effect=RuntimeError("Apify connection error")):
        raw = await fetch_competitor_ads.ainvoke({
            "competitor_name": "KFC",
            "competitor_url": "https://facebook.com/KFCEgypt",
        })
        data = json.loads(raw)
        assert data["competitor"] == "KFC"
        assert "Apify connection error" in data["error"]
        assert data["ads"] == []


@pytest.mark.asyncio
async def test_fetch_competitor_ads_success_with_db():
    """Successful scraping parses ads and upserts into database memory."""
    mock_raw_ads = [
        {
            "adArchiveID": "999888",
            "pageName": "KFC Egypt",
            "snapshot": {
                "body": "Best fried chicken in town",
                "images": [{"imageUrl": "https://cdn.example.com/kfc.jpg"}],
            },
        }
    ]

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    # Mock db.execute returning None for existing item
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None
    mock_db.execute.return_value = mock_result
    mock_db.scalar.return_value = "biz_123"

    with patch("app.ai.tools.fetch_competitor_ads._scrape", return_value=mock_raw_ads):
        raw = await fetch_competitor_ads.ainvoke({
            "competitor_name": "KFC",
            "competitor_url": "https://facebook.com/KFCEgypt",
            "db": mock_db,
        })

    data = json.loads(raw)
    assert data["competitor"] == "KFC"
    assert data["ads_fetched"] == 1
    assert data["ads_with_caption"] == 1
    assert data["ads_with_images"] == 1
    assert len(data["ads"]) == 1
    assert data["ads"][0]["ad_id"] == "999888"

    # Verify db.add and db.commit were called
    assert mock_db.add.called
    assert mock_db.commit.called
