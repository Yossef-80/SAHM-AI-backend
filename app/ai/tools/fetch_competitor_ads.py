"""Fetch competitor ads from a Facebook Ads Library scraper Actor.

Input: competitor name and Facebook Page URL.
Output: one structured record per ad, including caption and its own media URLs.
The result is also saved to BrandMemoryItem for later research by the agent.

Requirements/configuration:
- APIFY_API_TOKEN must be configured in app.ai.settings.get_ai_settings().
- The configured Apify Actor must support the input fields used in _scrape().
- The calling agent/system must inject an AsyncSession for the hidden `db` argument.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from typing import Annotated, Any
from urllib.parse import urlparse

from apify_client import ApifyClientAsync
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.settings import get_ai_settings
from app.core.memory.lifecycle import memory_defaults
from app.db.models import BrandMemoryItem

logger = logging.getLogger(__name__)

ACTOR_ID = "apify/facebook-ads-scraper"
MAX_ADS = 30
TOP_KEYWORDS = 15
MAX_IMAGES_PER_AD = 10

STOPWORDS = set("""
a an the and or of to in on at for with from by is are was be this that it its
 your you our we i as not no and the for with from this that have has are was
""".split())


class FetchCompetitorAdsInput(BaseModel):
    competitor_name: str = Field(description="Competitor brand name, e.g. 'KFC Egypt'.")
    competitor_url: str = Field(
        description="The competitor's Facebook PAGE URL, e.g. https://www.facebook.com/KFCEgypt."
    )


def clean_page_url(url: str) -> str:
    """Validate a Facebook Page URL and remove query strings/fragments."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if not (host == "facebook.com" or host.endswith(".facebook.com") or host == "fb.com" or host.endswith(".fb.com")):
        raise ValueError("competitor_url must be a Facebook Page URL, not another website.")
    if not parsed.path.strip("/"):
        raise ValueError("competitor_url must point to a specific Facebook Page.")
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP/HTTPS Facebook Page URLs are supported.")
    return f"https://www.facebook.com{parsed.path.rstrip('/')}"


def _as_text(value: Any) -> str | None:
    """Normalize common text shapes returned by scraping Actors."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, dict):
        for key in ("text", "caption", "message", "content"):
            result = _as_text(value.get(key))
            if result:
                return result
    return None


def _media_url(item: Any) -> str | None:
    if isinstance(item, str):
        return item if item.startswith(("https://", "http://")) else None
    if isinstance(item, dict):
        for key in (
            "originalImageUrl", "resizedImageUrl", "image_url", "imageUrl",
            "thumbnailUrl", "thumbnail_url", "url", "src", "videoUrl", "video_url",
        ):
            value = item.get(key)
            if isinstance(value, str) and value.startswith(("https://", "http://")):
                return value
    return None


def extract_ad(raw: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Extract one ad while keeping its caption and media together."""
    snapshot = raw.get("snapshot") or raw.get("adSnapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}

    cards = snapshot.get("cards") or raw.get("cards") or []
    if not isinstance(cards, list):
        cards = []

    caption_parts: list[str] = []
    # Keep the main ad text first; do not merge all ads into one caption.
    for value in (
        _as_text(snapshot.get("body")),
        _as_text(raw.get("body")),
        _as_text(snapshot.get("title")),
        _as_text(raw.get("title")),
        _as_text(snapshot.get("caption")),
        _as_text(raw.get("caption")),
        _as_text(snapshot.get("linkDescription")),
        _as_text(raw.get("description")),
    ):
        if value and value not in caption_parts:
            caption_parts.append(value)

    image_urls: list[str] = []
    video_urls: list[str] = []
    for collection in (snapshot.get("images"), raw.get("images"), snapshot.get("videos"), raw.get("videos")):
        if not isinstance(collection, list):
            continue
        for media in collection:
            url = _media_url(media)
            if not url:
                continue
            is_video = any(k in str(media).lower() for k in ("video",)) if isinstance(media, dict) else False
            target = video_urls if is_video else image_urls
            if url not in target:
                target.append(url)

    card_captions: list[str] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        text = _as_text(card.get("body")) or _as_text(card.get("title")) or _as_text(card.get("caption"))
        if text and text not in card_captions:
            card_captions.append(text)
        url = _media_url(card)
        if url and url not in image_urls:
            image_urls.append(url)
        for key in ("originalImageUrl", "resizedImageUrl", "image_url", "imageUrl", "thumbnailUrl"):
            url = _media_url(card.get(key))
            if url and url not in image_urls:
                image_urls.append(url)
        for key in ("videoUrl", "video_url"):
            url = _media_url(card.get(key))
            if url and url not in video_urls:
                video_urls.append(url)

    caption = "\n\n".join(caption_parts)
    ad_id = raw.get("adArchiveID") or raw.get("ad_archive_id") or raw.get("id") or raw.get("adId")
    library_url = raw.get("adLibraryUrl") or raw.get("ad_library_url") or raw.get("url")
    if not library_url and ad_id:
        library_url = f"https://www.facebook.com/ads/library/?id={ad_id}"

    return {
        "ad_id": str(ad_id) if ad_id is not None else f"unknown_{index + 1}",
        "ad_library_url": library_url if isinstance(library_url, str) else None,
        "page_name": raw.get("pageName") or raw.get("page_name") or raw.get("advertiserName"),
        "caption": caption,
        "card_captions": card_captions,
        "image_urls": image_urls[:MAX_IMAGES_PER_AD],
        "video_urls": video_urls,
        "started_running": raw.get("startDate") or raw.get("start_date") or raw.get("startedRunning"),
        "is_active": raw.get("isActive") if "isActive" in raw else raw.get("is_active"),
    }


def top_keywords(ads: list[dict[str, Any]], n: int = TOP_KEYWORDS) -> list[dict[str, Any]]:
    """Count each keyword at most once per ad; supports Arabic and English words."""
    counter: Counter[str] = Counter()
    for ad in ads:
        text = " ".join([ad.get("caption") or "", *ad.get("card_captions", [])]).lower()
        words = set(re.findall(r"[A-Za-z؀-ۿ]{3,}", text))
        counter.update(word for word in words if word not in STOPWORDS)
    return [{"keyword": word, "ad_count": count} for word, count in counter.most_common(n)]


async def _scrape(page_url: str) -> list[dict[str, Any]]:
    token = get_ai_settings().apify_api_token
    if not token:
        raise RuntimeError("APIFY_API_TOKEN is not configured.")

    client = ApifyClientAsync(token)
    # Confirm these input keys against the selected Actor's current Apify Store docs.
    run = await client.actor(ACTOR_ID).call(
        run_input={"startUrls": [{"url": page_url}], "resultsLimit": MAX_ADS}
    )
    if not run:
        return []

    dataset_id = run.get("defaultDatasetId") if isinstance(run, dict) else getattr(run, "default_dataset_id", None)
    if not dataset_id:
        raise RuntimeError("Apify Actor finished without returning a dataset ID.")

    return [item async for item in client.dataset(dataset_id).iterate_items(limit=MAX_ADS)]


async def _save_to_memory(db: AsyncSession, name: str, payload: dict[str, Any]) -> None:
    """Upsert one memory record per competitor, including each ad's caption and media URLs."""
    from app.db.models import Business

    safe_name = re.sub(r"\W+", "_", name.lower()).strip("_") or "competitor"
    ref = f"competitor_ads:{safe_name}"
    content = json.dumps(payload, ensure_ascii=False, indent=2)

    # Resolve business_id. Only a real string counts: a MagicMock (tests) or a
    # session object must not become the tenant key of a memory row.
    business_id = getattr(db, "business_id", None)
    if not isinstance(business_id, str) or not business_id:
        business_id = None
    if not business_id and hasattr(db, "info") and isinstance(db.info, dict):
        candidate = db.info.get("business_id")
        business_id = candidate if isinstance(candidate, str) and candidate else None
    if not business_id:
        try:
            biz_id = await db.scalar(select(Business.id).order_by(Business.created_at).limit(1))
            business_id = biz_id or "default"
        except Exception:
            business_id = "default"

    # Query existing memory item
    item = None
    if hasattr(BrandMemoryItem, "source_ref"):
        result = await db.execute(select(BrandMemoryItem).where(BrandMemoryItem.source_ref == ref))
        item = result.scalars().first()
    else:
        item_id = f"mem_comp_{safe_name}"
        result = await db.execute(select(BrandMemoryItem).where(BrandMemoryItem.id == item_id))
        item = result.scalars().first()

    if item:
        if hasattr(item, "content"):
            item.content = content
        if hasattr(item, "text"):
            item.text = f"Competitor ads for {name} ({payload.get('ads_fetched', 0)} ads). Top keywords: {', '.join(k['keyword'] for k in payload.get('top_keywords', [])[:5])}."
        if hasattr(item, "metadata_"):
            meta = dict(item.metadata_ or {})
            meta["payload"] = payload
            meta["source_ref"] = ref
            item.metadata_ = meta
    else:
        item_id = f"mem_comp_{safe_name}"
        kwargs: dict[str, Any] = {
            "id": item_id,
            "kind": "competitor_note",
            "business_id": business_id,
        }
        if hasattr(BrandMemoryItem, "text"):
            kwargs["text"] = f"Competitor ads for {name} ({payload.get('ads_fetched', 0)} ads). Top keywords: {', '.join(k['keyword'] for k in payload.get('top_keywords', [])[:5])}."
        if hasattr(BrandMemoryItem, "content"):
            kwargs["content"] = content
        if hasattr(BrandMemoryItem, "title"):
            kwargs["title"] = f"Competitor ads: {name}"
        if hasattr(BrandMemoryItem, "source_ref"):
            kwargs["source_ref"] = ref
        if hasattr(BrandMemoryItem, "metadata_"):
            kwargs["metadata_"] = {
                "source_ref": ref,
                "title": f"Competitor ads: {name}",
                "full_payload": payload,
            }
        kwargs.update(
            memory_defaults(
                "competitor_note",
                business_id=business_id,
                text=str(kwargs.get("text") or ""),
                source="fetch_competitor_ads",
            )
        )
        db.add(BrandMemoryItem(**kwargs))

    await db.commit()


@tool(args_schema=FetchCompetitorAdsInput)
async def fetch_competitor_ads(
    competitor_name: str,
    competitor_url: str,
    db: Annotated[AsyncSession | None, InjectedToolArg] = None,
) -> str:
    """Fetch up to 30 competitor ads. Returns a JSON report with one record per ad,
    including its caption, image/video URLs, Ad Library URL, dates/status when available,
    plus keyword counts. Saves the report to BrandMemoryItem.
    """
    try:
        page_url = clean_page_url(competitor_url)
        raw_ads = await _scrape(page_url)
    except Exception as exc:
        logger.exception("fetch_competitor_ads failed")
        return json.dumps(
            {"competitor": competitor_name, "error": str(exc)[:300], "ads": []},
            ensure_ascii=False,
        )

    ads = [extract_ad(raw, i) for i, raw in enumerate(raw_ads) if isinstance(raw, dict)]
    # Keep records even when a scraper item has no caption; it may still have media/metadata.
    payload = {
        "competitor": competitor_name,
        "facebook_page_url": page_url,
        "ads_fetched": len(ads),
        "ads_with_caption": sum(bool(ad["caption"] or ad["card_captions"]) for ad in ads),
        "ads_with_images": sum(bool(ad["image_urls"]) for ad in ads),
        "top_keywords": top_keywords(ads),
        "ads": ads,
    }

    if not ads:
        payload["message"] = "No ads were returned. Check the Page URL, Actor configuration, and whether ads are available."
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if db is not None:
        try:
            await _save_to_memory(db, competitor_name, payload)
        except Exception as exc:
            logger.exception("Could not save competitor ads to BrandMemoryItem")
            payload["memory_save_warning"] = str(exc)[:200]

    return json.dumps(payload, ensure_ascii=False, indent=2)


fetch_competitor_ads_tool = fetch_competitor_ads
