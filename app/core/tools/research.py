"""Research tools (1-6).

Each tool is real where an API is configured and reachable, and returns
``status="unavailable"`` with a human-readable reason when it is not. No tool
ever fabricates data. Several support a user-supplied fallback so the agent can
ask the user to paste the data instead of guessing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app.config import Settings
from app.core.errors import BlockedTargetError, ToolUnavailable
from app.core.llm.base import Msg
from app.core.logging import get_logger
from app.core.security import check_redirect, resolve_and_check
from app.core.tools.base import Tool, ToolContext
from app.schemas.common import ToolSideEffect
from app.schemas.tools import (
    AdRecord,
    GetCompetitorAdsInput,
    GetCompetitorAdsOutput,
    GetKeywordsInput,
    GetKeywordsOutput,
    GetReviewsInput,
    GetReviewsOutput,
    KeywordMetric,
    Review,
    SearchResult,
    SocialProfile,
    SocialProfileInput,
    WebSearchInput,
    WebSearchOutput,
    WebsiteProfile,
    WebsiteProfileInput,
)

_logger = get_logger(__name__)


def _settings(ctx: ToolContext) -> Settings:
    return ctx.require_settings()


def _unavailable(output_cls: type, *, source: str, reason: str) -> Any:
    return output_cls.unavailable(source=source, reason=reason)


# ============================================================ 1. web_search ==


async def _search_tavily(payload: WebSearchInput, key: str, timeout: float) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": payload.query,
                "max_results": payload.max_results,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
            published_date=r.get("published_date"),
            score=r.get("score"),
        )
        for r in data.get("results", [])
    ]


async def _search_serper(payload: WebSearchInput, key: str, timeout: float) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key},
            json={"q": payload.query, "num": payload.max_results, "gl": payload.country or "us"},
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("link", ""),
            snippet=r.get("snippet", ""),
            published_date=r.get("date"),
        )
        for r in data.get("organic", [])
    ]


async def _search_brave(payload: WebSearchInput, key: str, timeout: float) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": key},
            params={"q": payload.query, "count": payload.max_results},
        )
        resp.raise_for_status()
        data = resp.json()
    results = (data.get("web") or {}).get("results", [])
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("description", ""),
            published_date=r.get("age"),
        )
        for r in results
    ]


def _search_mock(payload: WebSearchInput) -> list[SearchResult]:
    """Deterministic offline results, clearly labelled as demo data."""
    digest = hashlib.sha256(payload.query.encode()).hexdigest()
    return [
        SearchResult(
            title=f"Demo result {i + 1} for '{payload.query}'",
            url=f"https://example.com/demo/{digest[:8]}/{i + 1}",
            snippet=(
                f"Offline demo snippet {i + 1}. No live search provider is configured, "
                "so this is placeholder data, not a real source."
            ),
        )
        for i in range(min(payload.max_results, 3))
    ]


async def _handle_web_search(payload: WebSearchInput, ctx: ToolContext) -> WebSearchOutput:
    cfg = _settings(ctx)
    provider = cfg.search_provider
    source = f"web_search:{provider}"
    try:
        if provider == "tavily" and cfg.tavily_api_key:
            results = await _search_tavily(payload, cfg.tavily_api_key, cfg.llm_timeout_seconds)
        elif provider == "serper" and cfg.serper_api_key:
            results = await _search_serper(payload, cfg.serper_api_key, cfg.llm_timeout_seconds)
        elif provider == "brave" and cfg.brave_api_key:
            results = await _search_brave(payload, cfg.brave_api_key, cfg.llm_timeout_seconds)
        elif provider == "mock":
            results = _search_mock(payload)
        else:
            return _unavailable(
                WebSearchOutput,
                source=source,
                reason=(
                    f"No web search provider is configured (SEARCH_PROVIDER='{provider}' "
                    "and the matching API key is missing). Ask the user to paste the "
                    "search results they want analysed."
                ),
            )
    except httpx.HTTPStatusError as exc:
        return _unavailable(
            WebSearchOutput,
            source=source,
            reason=f"search provider returned HTTP {exc.response.status_code}.",
        )
    except httpx.HTTPError as exc:
        return _unavailable(WebSearchOutput, source=source, reason=f"search request failed: {exc}")

    if not results:
        return WebSearchOutput.partial(
            source=source, reason="The search provider returned no results for this query."
        ).model_copy(update={"results": [], "query": payload.query})
    # Mock results are placeholders, so they are reported as partial with a
    # reason. Claiming status="ok" would let a caller treat them as real hits.
    if provider == "mock":
        return WebSearchOutput.partial(
            source=source,
            reason=(
                "No search API key is configured. The results below are "
                "placeholders, not real search hits."
            ),
        ).model_copy(update={"results": results, "query": payload.query})
    return WebSearchOutput(results=results, query=payload.query, status="ok", source=source)


web_search_tool = Tool(
    name="web_search",
    description=(
        "Search the public web for a query and return titles, URLs and snippets. "
        "Use it for market context, competitor mentions, pricing and trends. "
        "Do NOT use it for the brand's own campaign numbers (use "
        "get_campaign_insights) or for the brand's own memory (use get_brand_context). "
        "If it returns status='unavailable', say so and ask the user to paste results."
    ),
    input_model=WebSearchInput,
    output_model=WebSearchOutput,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_web_search,
)


# ======================================================= 2. analyze_website ==


class _WebsiteExtraction(BaseModel):
    """What the LLM extracts from a page's text."""

    title: str = ""
    description: str = ""
    summary: str = ""
    offers: list[str] = []
    ctas: list[str] = []
    language: str | None = None


async def _fetch_website(url: str, cfg: Settings) -> tuple[str, str]:
    """Fetch a public URL with SSRF protection, size cap and timeout."""
    safe = resolve_and_check(url, allowed_schemes=cfg.website_allowed_schemes)
    async with httpx.AsyncClient(
        timeout=cfg.website_fetch_timeout_seconds,
        follow_redirects=False,
        headers={"user-agent": "SahmBot/1.0 (+https://sahm.example/bot)"},
    ) as client:
        current = safe.url
        for _ in range(cfg.website_max_redirects):
            resp = await client.get(current)
            if resp.is_redirect:
                location = resp.headers.get("location", "")
                if not location:
                    raise ToolUnavailable("website redirected without a location header")
                # Validate every hop; redirects are the classic SSRF bypass.
                current = check_redirect(
                    str(resp.url.join(location)), allowed_schemes=cfg.website_allowed_schemes
                ).url
                continue
            resp.raise_for_status()
            break
        else:
            raise ToolUnavailable("too many redirects while fetching the website")

        content_length = resp.headers.get("content-length")
        if content_length and int(content_length) > cfg.website_max_bytes:
            raise ToolUnavailable(
                f"website response is larger than the {cfg.website_max_bytes} byte cap"
            )
        raw = resp.content[: cfg.website_max_bytes]
        if len(raw) >= cfg.website_max_bytes:
            raise ToolUnavailable(
                f"website response exceeded the {cfg.website_max_bytes} byte cap"
            )
    return raw.decode(resp.encoding or "utf-8", errors="replace"), str(resp.url)


def _parse_html(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    def _meta(name: str) -> str:
        node = soup.find("meta", attrs={"name": name}) or soup.find(
            "meta", attrs={"property": f"og:{name}"}
        )
        return (node.get("content") or "").strip() if node else ""

    headings = [
        h.get_text(" ", strip=True)
        for h in soup.find_all(["h1", "h2", "h3"])[:20]
        if h.get_text(strip=True)
    ]
    text = soup.get_text(" ", strip=True)
    links = [
        a.get("href", "")
        for a in soup.find_all("a", href=True)
        if a.get("href", "").startswith("http")
    ]
    social_hosts = ("instagram.com", "facebook.com", "tiktok.com", "x.com", "twitter.com", "linkedin.com")
    socials = []
    for href in links:
        if any(host in href for host in social_hosts) and href not in socials:
            socials.append(href)

    return {
        "title": (soup.title.get_text(strip=True) if soup.title else "") or _meta("title"),
        "description": _meta("description"),
        "headings": headings,
        "text": text[:12000],
        "socials": socials[:10],
        "links": links[:60],
    }


async def _handle_analyze_website(
    payload: WebsiteProfileInput, ctx: ToolContext
) -> WebsiteProfile:
    cfg = _settings(ctx)
    source = "analyze_website"
    try:
        html, final_url = await _fetch_website(payload.url, cfg)
    except BlockedTargetError as exc:
        return _unavailable(WebsiteProfile, source=source, reason=str(exc.message))
    except ToolUnavailable as exc:
        return _unavailable(WebsiteProfile, source=source, reason=exc.message)
    except httpx.HTTPError as exc:
        return _unavailable(WebsiteProfile, source=source, reason=f"fetch failed: {exc}")

    parsed = _parse_html(html)
    offers: list[dict[str, str]] = []
    contact: dict[str, str] = {}

    try:
        result = await ctx.llm.complete(
            system=(
                "You extract structured facts from a website's text. Report only what "
                "is literally present. If something is absent, leave it empty. Never "
                "infer prices, claims or statistics."
            ),
            messages=[
                Msg(
                    role="user",
                    content=(
                        f"URL: {final_url}\n"
                        f"TITLE: {parsed['title']}\n"
                        f"DESCRIPTION: {parsed['description']}\n"
                        f"HEADINGS: {json.dumps(parsed['headings'])}\n\n"
                        f"TEXT:\n{parsed['text']}"
                    ),
                )
            ],
            response_schema=_WebsiteExtraction,
            tier="fast",
            max_tokens=1200,
            temperature=0.0,
        )
        extraction = result.parsed if isinstance(result.parsed, _WebsiteExtraction) else None
    except Exception as exc:  # LLM failure -> still return the parsed page
        _logger.warning("website_llm_structuring_failed", extra={"error": str(exc)})
        extraction = None

    for link in parsed["links"]:
        host = urlparse(link).netloc.lower()
        if "mailto:" in link or "@" in link:
            contact.setdefault("email", link)
        elif "wa.me" in link or "whatsapp" in link:
            contact.setdefault("whatsapp", link)
        elif "tel:" in link:
            contact.setdefault("phone", link)
        elif host and "email" not in contact:
            continue

    return WebsiteProfile(
        url=final_url,
        title=extraction.title if extraction else parsed["title"],
        description=extraction.description if extraction else parsed["description"],
        language=extraction.language if extraction else None,
        headings=parsed["headings"],
        offers=[
            {"name": o, "detail": ""} for o in (extraction.offers if extraction else [])
        ],
        ctas=extraction.ctas if extraction else [],
        social_links=parsed["socials"],
        contact=contact,
        summary=extraction.summary if extraction else "",
        word_count=len(parsed["text"].split()),
        status="ok",
        source=source,
    )


analyze_website_tool = Tool(
    name="analyze_website",
    description=(
        "Fetch a public URL and return a structured profile: title, description, "
        "headings, offers, calls-to-action, social links and a summary. Use it when "
        "the user gives a website or asks about a competitor's positioning. It only "
        "reads public pages and blocks private/internal addresses. Do NOT use it for "
        "the brand's own campaign data."
    ),
    input_model=WebsiteProfileInput,
    output_model=WebsiteProfile,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_analyze_website,
)


# ==================================================== 3. get_social_profile ==

_SUPPORTED_SOCIAL = {"instagram", "facebook", "tiktok", "x", "twitter", "linkedin"}


async def _handle_get_social_profile(
    payload: SocialProfileInput, ctx: ToolContext
) -> SocialProfile:
    cfg = _settings(ctx)
    platform = payload.platform.lower()
    source = f"get_social_profile:{platform}"
    if platform not in _SUPPORTED_SOCIAL:
        return _unavailable(
            SocialProfile,
            source=source,
            reason=f"Platform '{payload.platform}' is not supported.",
        )
    if not cfg.meta_access_token:
        return _unavailable(
            SocialProfile,
            source=source,
            reason=(
                "No Meta access token is configured, so social profiles cannot be "
                "read. Connect Meta in Integrations, or paste the profile details "
                "you want analysed."
            ),
        )
    # Business Discovery is only available for a small set of markets and
    # requires additional permissions. Rather than pretending, be explicit.
    return _unavailable(
        SocialProfile,
        source=source,
        reason=(
            "Social profile lookup is not enabled for this workspace. The Meta Graph "
            "API does not expose public profile metrics for this market/permission "
            "set. Paste the profile bio, follower count or recent posts you want "
            "analysed instead."
        ),
    )


get_social_profile_tool = Tool(
    name="get_social_profile",
    description=(
        "Look up a public social profile (instagram, facebook, tiktok, x, linkedin) "
        "and return bio, follower counts and recent posts. Use it to understand a "
        "competitor's or the brand's own social presence. It frequently returns "
        "status='unavailable' -- when it does, say so and ask the user to paste the "
        "profile details."
    ),
    input_model=SocialProfileInput,
    output_model=SocialProfile,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_social_profile,
)


# =========================================================== 4. get_reviews ==


def _reviews_mock(payload: GetReviewsInput) -> list[Review]:
    digest = hashlib.sha256(payload.place_query.encode()).hexdigest()
    return [
        Review(
            author=f"Demo reviewer {i + 1}",
            rating=5.0 - (i % 3),
            text=(
                f"Offline demo review {i + 1} for '{payload.place_query}'. No Google "
                "Places key is configured, so this is placeholder data."
            ),
            relative_time="demo",
        )
        for i in range(min(payload.max_results, 3))
    ]


async def _handle_get_reviews(payload: GetReviewsInput, ctx: ToolContext) -> GetReviewsOutput:
    cfg = _settings(ctx)
    source = "get_reviews:google_places"
    if not cfg.google_places_api_key:
        if cfg.environment in ("dev", "test"):
            reviews = _reviews_mock(payload)
            return GetReviewsOutput(
                reviews=reviews,
                place_name=payload.place_query,
                status="partial",
                source="mock_google_places",
                unavailable_reason=(
                    "No Google Places API key is configured. These are offline demo "
                    "reviews, not real customer feedback."
                ),
            )
        return _unavailable(
            GetReviewsOutput,
            source=source,
            reason=(
                "No Google Places API key is configured, so reviews cannot be read. "
                "Paste the reviews you want analysed, or connect a reviews source."
            ),
        )

    async with httpx.AsyncClient(timeout=cfg.llm_timeout_seconds) as client:
        try:
            find = await client.get(
                f"{cfg.places_base_url}/findplacefromtext/json",
                params={
                    "input": payload.place_query,
                    "inputtype": "textquery",
                    "fields": "place_id,name,rating,user_ratings_total",
                    "key": cfg.google_places_api_key,
                },
            )
            find.raise_for_status()
            candidates = find.json().get("candidates", [])
            if not candidates:
                return GetReviewsOutput.partial(
                    source=source,
                    reason=f"No place matched '{payload.place_query}'.",
                )
            place = candidates[0]
            details = await client.get(
                f"{cfg.places_base_url}/details/json",
                params={
                    "place_id": place["place_id"],
                    "fields": "name,rating,user_ratings_total,reviews",
                    "reviews_sort": "most_relevant",
                    "key": cfg.google_places_api_key,
                },
            )
            details.raise_for_status()
            data = details.json().get("result", {})
        except httpx.HTTPError as exc:
            return _unavailable(GetReviewsOutput, source=source, reason=f"Places API failed: {exc}")

    reviews = [
        Review(
            author=r.get("author_name", ""),
            rating=r.get("rating"),
            text=r.get("text", ""),
            relative_time=r.get("relative_time_description"),
        )
        for r in data.get("reviews", [])[: payload.max_results]
    ]
    return GetReviewsOutput(
        reviews=reviews,
        place_name=data.get("name", payload.place_query),
        average_rating=data.get("rating"),
        total_ratings=data.get("user_ratings_total"),
        status="ok",
        source=source,
    )


get_reviews_tool = Tool(
    name="get_reviews",
    description=(
        "Fetch Google reviews for a place or business name and return individual "
        "reviews plus the aggregate rating. Use it to understand customer sentiment "
        "and recurring complaints. When no Places key is configured it returns "
        "demo data with status='partial' -- never present that as real feedback."
    ),
    input_model=GetReviewsInput,
    output_model=GetReviewsOutput,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_reviews,
)


# ========================================================== 5. get_keywords ==


def _keywords_mock(payload: GetKeywordsInput) -> list[KeywordMetric]:
    out: list[KeywordMetric] = []
    for term in payload.seed_terms:
        digest = hashlib.sha256(f"{term}|{payload.country}".encode()).hexdigest()
        volume = 200 + (int(digest[:6], 16) % 9800)
        out.append(
            KeywordMetric(
                term=term,
                volume=volume,
                competition=round((int(digest[6:10], 16) % 100) / 100.0, 2),
                cpc=round((int(digest[10:14], 16) % 400) / 100.0, 2),
                currency="USD",
                trend="flat",
            )
        )
    return out


async def _handle_get_keywords(payload: GetKeywordsInput, ctx: ToolContext) -> GetKeywordsOutput:
    cfg = _settings(ctx)
    source = f"get_keywords:{cfg.keyword_provider}"
    if cfg.keyword_provider == "dataforseo" and cfg.dataforseo_login:
        try:
            async with httpx.AsyncClient(timeout=cfg.llm_timeout_seconds) as client:
                resp = await client.post(
                    "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live",
                    auth=(cfg.dataforseo_login, cfg.dataforseo_password or ""),
                    json=[
                        {
                            "keywords": payload.seed_terms,
                            "location_name": payload.country,
                            "language_code": payload.language or "en",
                        }
                    ],
                )
                resp.raise_for_status()
                tasks = resp.json().get("tasks", [])
                rows = []
                for task in tasks:
                    for item in task.get("result") or []:
                        rows.append(
                            KeywordMetric(
                                term=item.get("keyword", ""),
                                volume=item.get("search_volume"),
                                competition=(item.get("competition") or 0) / 100.0
                                if item.get("competition") is not None
                                else None,
                                cpc=item.get("cpc"),
                                currency="USD",
                            )
                        )
            return GetKeywordsOutput(
                keywords=rows, country=payload.country, seed_terms=payload.seed_terms,
                status="ok", source=source,
            )
        except httpx.HTTPError as exc:
            return _unavailable(
                GetKeywordsOutput,
                source=source,
                reason=f"DataForSEO request failed: {exc}",
            )

    if cfg.keyword_provider == "mock" or cfg.environment in ("dev", "test"):
        return GetKeywordsOutput(
            keywords=_keywords_mock(payload),
            country=payload.country,
            seed_terms=payload.seed_terms,
            status="partial",
            source="mock_keyword_provider",
            unavailable_reason=(
                "No keyword-volume provider is configured. These volumes are "
                "deterministic placeholders, not real search-volume data."
            ),
        )
    return _unavailable(
        GetKeywordsOutput,
        source=source,
        reason=(
            f"Keyword provider '{cfg.keyword_provider}' is not configured. Set "
            "DATAFORSEO_LOGIN/PASSWORD, or ask the user to supply keyword volumes."
        ),
    )


get_keywords_tool = Tool(
    name="get_keywords",
    description=(
        "Return search volume, competition and CPC for seed keyword terms in a "
        "country. Use it to prioritise messaging and SEO topics. Without a keyword "
        "provider it returns placeholder volumes with status='partial' -- label them "
        "as estimates, never as measured data."
    ),
    input_model=GetKeywordsInput,
    output_model=GetKeywordsOutput,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_keywords,
)


# ====================================================== 6. get_competitor_ads


#: The Meta Ad Library API only covers a subset of countries. Anything outside
#: this list returns "unavailable" rather than an empty (and therefore
#: misleading) result.
_AD_LIBRARY_COUNTRIES = {
    "US", "GB", "CA", "AU", "NZ", "IE", "FR", "DE", "ES", "IT", "NL", "BE",
    "SE", "NO", "DK", "FI", "PL", "PT", "AT", "CH", "BR", "MX", "AR", "CL",
    "CO", "PE", "IN", "ID", "MY", "PH", "SG", "TH", "VN", "JP", "KR", "TW",
    "HK", "IL", "AE", "SA", "EG", "ZA", "TR", "UA", "RU",
}


async def _handle_get_competitor_ads(
    payload: GetCompetitorAdsInput, ctx: ToolContext
) -> GetCompetitorAdsOutput:
    cfg = _settings(ctx)
    country = payload.country.upper()
    source = f"get_competitor_ads:meta_ad_library:{country}"

    if payload.user_supplied_ads:
        return GetCompetitorAdsOutput(
            ads=payload.user_supplied_ads[: payload.max_results],
            page_or_query=payload.page_or_query,
            country=country,
            used_user_supplied=True,
            status="ok",
            source="user_supplied_ads",
        )

    if country not in _AD_LIBRARY_COUNTRIES:
        return _unavailable(
            GetCompetitorAdsOutput,
            source=source,
            reason=(
                f"The Meta Ad Library API does not return ads for '{country}'. "
                "Paste or upload the competitor ads you want analysed and pass them "
                "as user_supplied_ads."
            ),
        )
    if not cfg.meta_access_token:
        return _unavailable(
            GetCompetitorAdsOutput,
            source=source,
            reason=(
                "No Meta access token is configured, so the Ad Library cannot be "
                "queried. Connect Meta in Integrations, or paste the ads as "
                "user_supplied_ads."
            ),
        )

    try:
        async with httpx.AsyncClient(timeout=cfg.meta_api_timeout_seconds) as client:
            resp = await client.get(
                f"{cfg.meta_graph_base_url}/{cfg.meta_graph_version}/ads_archive",
                params={
                    "access_token": cfg.meta_access_token,
                    "search_terms": payload.page_or_query,
                    "ad_reached_countries": f'["{country}"]',
                    "limit": payload.max_results,
                    "fields": "id,page_name,ad_creative_bodies,ad_creative_link_titles,"
                    "ad_snapshot_url,publisher_platforms",
                },
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
    except httpx.HTTPError as exc:
        return _unavailable(
            GetCompetitorAdsOutput, source=source, reason=f"Ad Library request failed: {exc}"
        )

    ads = [
        AdRecord(
            page_name=r.get("page_name", ""),
            ad_id=r.get("id"),
            body=(r.get("ad_creative_bodies") or [""])[0],
            headline=(r.get("ad_creative_link_titles") or [None])[0],
            media_type=None,
            started_running=None,
            platforms=r.get("publisher_platforms") or [],
            snapshot_url=r.get("ad_snapshot_url"),
        )
        for r in rows
    ]
    if not ads:
        return GetCompetitorAdsOutput.partial(
            source=source,
            reason=f"The Ad Library returned no active ads matching '{payload.page_or_query}'.",
        ).model_copy(
            update={"ads": [], "page_or_query": payload.page_or_query, "country": country}
        )
    return GetCompetitorAdsOutput(
        ads=ads,
        page_or_query=payload.page_or_query,
        country=country,
        status="ok",
        source=source,
    )


get_competitor_ads_tool = Tool(
    name="get_competitor_ads",
    description=(
        "Return competitor ads from the Meta Ad Library for a country. Use it to see "
        "what messaging competitors are running. The API does not cover every "
        "country: when it returns status='unavailable', ask the user to paste or "
        "upload the ads and pass them as user_supplied_ads instead of guessing."
    ),
    input_model=GetCompetitorAdsInput,
    output_model=GetCompetitorAdsOutput,
    side_effect=ToolSideEffect.NONE,
    handler=_handle_get_competitor_ads,
)


RESEARCH_TOOLS: list[Tool] = [
    web_search_tool,
    analyze_website_tool,
    get_social_profile_tool,
    get_reviews_tool,
    get_keywords_tool,
    get_competitor_ads_tool,
]

__all__ = [
    "RESEARCH_TOOLS",
    "analyze_website_tool",
    "get_competitor_ads_tool",
    "get_keywords_tool",
    "get_reviews_tool",
    "get_social_profile_tool",
    "web_search_tool",
]
