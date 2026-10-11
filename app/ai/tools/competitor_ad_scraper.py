"""Competitor Ad Library Scraper  (agent: nour / Researcher)

Input : competitor name + competitor Facebook page url.
Output: a plain-text summary (top keywords used in their ads + image links),
        also saved as a BrandMemoryItem (long-term memory).
Tool type: Action + Read  (needs `db: AsyncSession`, injected by the system).
"""
import logging
import re
import uuid
from collections import Counter
from urllib.parse import urlparse

from apify_client import ApifyClientAsync
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.settings import get_ai_settings

from app.db.models import BrandMemoryItem

logger = logging.getLogger(__name__)

ACTOR_ID = "apify/facebook-ads-scraper"
MAX_ADS = 30          # keep small: Apify charges per ad
TOP_KEYWORDS = 15
MAX_IMAGES = 10

STOPWORDS = set("""
a an the and or of to in on at for with from by is are was be this that it its your you our we i as not no
""".split())


# --------------------------------------------------------------------------- input schema
class FetchCompetitorAdsInput(BaseModel):
    competitor_name: str = Field(description="Name of the competitor brand, e.g. 'KFC Egypt'.")
    competitor_url: str = Field(
        description="The competitor's Facebook PAGE url, e.g. 'https://www.facebook.com/KFCEgypt'."
    )


# --------------------------------------------------------------------------- helpers
def clean_page_url(url: str) -> str:
    """Drop tracking params (?utm_source=...) and validate it is a page url."""
    parsed = urlparse(url.strip())
    if "facebook.com" not in parsed.netloc and "fb.com" not in parsed.netloc:
        raise ValueError("competitor_url must be a Facebook page url, e.g. https://www.facebook.com/KFCEgypt")
    if not parsed.path.strip("/"):
        raise ValueError("competitor_url must point to a page, not to facebook.com itself.")
    return f"https://www.facebook.com{parsed.path}"


def _first(*values):
    return next((v for v in values if v not in (None, "", [])), None)


def extract_ad(raw: dict) -> tuple[str, list[str]]:
    """Return (ad text, image urls) from one raw Apify ad."""
    snap = raw.get("snapshot") or {}
    cards = snap.get("cards") or []
    texts = [(snap.get("body") or {}).get("text"), snap.get("title")]
    texts += [c.get("body") or c.get("title") for c in cards]
    images = [i.get("originalImageUrl") or i.get("resizedImageUrl") for i in snap.get("images") or []]
    images += [c.get("originalImageUrl") or c.get("resizedImageUrl") for c in cards]
    return " ".join(t for t in texts if t), [i for i in images if i]


def top_keywords(texts: list[str], n: int = TOP_KEYWORDS) -> list[tuple[str, int]]:
    """Most frequent meaningful words (Arabic + English); counted once per ad."""
    counter: Counter = Counter()
    for text in texts:
        words = set(re.findall(r"[A-Za-z؀-ۿ]{3,}", text.lower()))
        counter.update(w for w in words if w not in STOPWORDS)
    return counter.most_common(n)


def build_summary(name: str, n_ads: int, keywords: list[tuple[str, int]], images: list[str]) -> str:
    kw = "\n".join(f"- {w} ({c} ads)" for w, c in keywords) or "- (no text found)"
    imgs = "\n".join(f"- {u}" for u in images) or "- (no images found)"
    return (
        f"Competitor ads summary: {name}\n"
        f"Ads analysed: {n_ads}\n\n"
        f"Top keywords used in their ads:\n{kw}\n\n"
        f"Images they use:\n{imgs}"
    )


async def _scrape(page_url: str) -> list[dict]:
    token = get_ai_settings().apify_api_token
    if not token:
        raise RuntimeError("APIFY_API_TOKEN is not configured.")
    client = ApifyClientAsync(token)
    run = await client.actor(ACTOR_ID).call(
        run_input={"startUrls": [{"url": page_url}], "resultsLimit": MAX_ADS},
        timeout_secs=300,
    )
    if not run:
        return []
    dataset_id = run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id
    return [item async for item in client.dataset(dataset_id).iterate_items(limit=MAX_ADS)]


async def _save_to_memory(db: AsyncSession, name: str, summary: str) -> None:
    """One BrandMemoryItem per competitor: update it if it already exists."""
    ref = "competitor_ads:" + re.sub(r"\W+", "_", name.lower()).strip("_")
    item = (await db.execute(select(BrandMemoryItem).where(BrandMemoryItem.source_ref == ref))).scalars().first()
    if item:
        item.content = summary
    else:
        db.add(
            BrandMemoryItem(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                kind="competitor_ads",
                title=f"Competitor ads: {name}",
                content=summary,
                source_ref=ref,
            )
        )
    await db.commit()


# --------------------------------------------------------------------------- the tool
@tool(args_schema=FetchCompetitorAdsInput)
async def fetch_competitor_ads(competitor_name: str, competitor_url: str, db: AsyncSession) -> str:
    """Scrapes a competitor's ads from the Facebook Ad Library and returns a summary of the most-used
    keywords and the image links they use. The summary is also saved to brand memory."""
    try:
        page_url = clean_page_url(competitor_url)
        raw_ads = await _scrape(page_url)
    except Exception as e:  # return the error as text so the agent can recover
        logger.exception("fetch_competitor_ads failed")
        return f"Could not fetch ads for {competitor_name}: {str(e)[:300]}"

    if not raw_ads:
        return (
            f"No ads found for {competitor_name} ({page_url}). Check that the url is the competitor's Facebook "
            f"PAGE url and that the page is running ads."
        )

    parsed = [extract_ad(r) for r in raw_ads]
    texts = [t for t, _ in parsed if t]
    images = list(dict.fromkeys(i for _, imgs in parsed for i in imgs))[:MAX_IMAGES]

    summary = build_summary(competitor_name, len(raw_ads), top_keywords(texts), images)
    await _save_to_memory(db, competitor_name, summary)
    return summary
