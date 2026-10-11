"""Website Analyzer Tool for SAHM OS.

Bilingual (Arabic & English) deep website analysis tool.
Extracts metadata, OpenGraph tags, JSON-LD Schema.org entities (Products, Offers,
Ratings, LocalBusiness), prices & currencies (EGP, LE, USD, Arabic numerals),
bilingual CTAs, trust signals (COD, guarantees, shipping), and clean body content.
Fully protected with SSRF checks, token budget boundaries, and graceful error handling.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.settings import get_ai_settings
from app.core.errors import BlockedTargetError
from app.core.security import resolve_and_check

MAX_HTML_BYTES = 4 * 1024 * 1024  # 4 MB
MAX_BODY_CHARS = 5000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en;q=0.9,en-US;q=0.8",
}

# Bilingual Trust Signals
TRUST_KEYWORDS = [
    # English
    "guarantee", "warranty", "secure payment", "verified", "authentic",
    "free shipping", "cash on delivery", "cod", "return policy", "refund",
    "installment", "money back",
    # Arabic
    "ضمان", "شحن مجاني", "الدفع عند الاستلام", "استرجاع", "استبدال",
    "سياسة الاسترجاع", "تقسيط", "أصلي", "معتمد", "دفع آمن", "استرداد الأموال",
]

# Bilingual CTAs
CTA_KEYWORDS = [
    # English
    "buy now", "shop now", "add to cart", "checkout", "order now",
    "contact us", "whatsapp", "learn more", "get started", "book now",
    # Arabic
    "اطلب الآن", "اطلب الان", "اطلب", "اشتري الآن", "اشتري الان", "اشتري",
    "تسوق الآن", "تسوق الان", "تسوق", "تواصل معنا", "تواصل", "واتساب",
    "أضف للسلة", "اضف للسلة", "أضف إلى السلة", "اضف الى السلة",
    "اعرف المزيد", "ابدأ الآن", "ابدأ الان", "احجز الآن", "احجز الان",
]

# Regex for Egyptian Pounds (EGP, LE, L.E., جنيه, ج.م), USD, and Arabic numerals
PRICE_REGEX = re.compile(
    r"(?:EGP|LE|L\.E\.|جنيه(?:اً)?|ج\.م)\s*"
    r"[\d٠-٩0-9,٬]+(?:[.٫][\d٠-٩0-9]{1,2})?"
    r"|[\d٠-٩0-9,٬]+(?:[.٫][\d٠-٩0-9]{1,2})?\s*"
    r"(?:EGP|LE|L\.E\.|جنيه(?:اً)?|ج\.م)"
    r"|(?:USD\s*)?[\$]\s*[\d0-9,]+(?:\.\d{1,2})?"
    r"|USD\s*[\d0-9,]+(?:\.\d{1,2})?",
    re.IGNORECASE,
)


class AnalyzeWebsiteInput(BaseModel):
    url: str = Field(description="The full public HTTP or HTTPS URL of the website or competitor page to analyze.")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _first_meta(soup: BeautifulSoup, selectors: list[dict[str, str]]) -> str:
    for attrs in selectors:
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            val = _clean_text(tag["content"])
            if val:
                return val[:500]
    return ""


def _extract_open_graph(soup: BeautifulSoup) -> dict[str, str]:
    result: dict[str, str] = {}
    keys = [
        "og:title", "og:description", "og:image", "og:type",
        "og:url", "og:site_name", "og:price:amount", "og:price:currency",
    ]
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            val = _clean_text(tag["content"])
            if val:
                result[key] = val[:500]
    return result


def _schema_nodes(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _schema_nodes(item)
    elif isinstance(value, dict):
        if "@graph" in value:
            yield from _schema_nodes(value["@graph"])
        else:
            yield value


def _extract_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    accepted = {
        "Product", "Organization", "LocalBusiness", "Store",
        "WebSite", "WebPage", "Offer", "AggregateRating",
    }

    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw.strip())
        except (json.JSONDecodeError, TypeError):
            continue

        for node in _schema_nodes(payload):
            types = node.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if not accepted.intersection(set(types)):
                continue

            item: dict[str, Any] = {}
            for key in ("@type", "name", "description", "url", "sku", "priceRange", "telephone"):
                val = node.get(key)
                if val is not None:
                    item[key] = val if isinstance(val, (dict, list)) else str(val)[:300]

            brand = node.get("brand")
            if brand:
                item["brand"] = brand.get("name", brand) if isinstance(brand, dict) else str(brand)[:100]

            offers = node.get("offers")
            if offers:
                offer_list = offers if isinstance(offers, list) else [offers]
                item["offers"] = [
                    {k: offer.get(k) for k in ("price", "priceCurrency", "availability", "url") if offer.get(k) is not None}
                    for offer in offer_list[:5]
                    if isinstance(offer, dict)
                ]

            rating = node.get("aggregateRating")
            if isinstance(rating, dict):
                item["aggregateRating"] = {
                    k: rating.get(k) for k in ("ratingValue", "reviewCount", "bestRating") if rating.get(k) is not None
                }

            if item:
                output.append(item)
            if len(output) >= 8:
                return output

    return output


def _extract_headings(soup: BeautifulSoup) -> dict[str, list[str]]:
    headings: dict[str, list[str]] = {}
    for level in ("h1", "h2", "h3"):
        headings[level] = [
            _clean_text(tag.get_text(" ", strip=True))[:150]
            for tag in soup.find_all(level)
            if tag.get_text(strip=True)
        ][:8]
    return headings


def _extract_prices(text: str, json_ld: list[dict[str, Any]]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()

    for match in PRICE_REGEX.finditer(text or ""):
        start, end = match.span()
        price = match.group(0).strip()
        key = price.casefold()
        if key in seen:
            continue
        seen.add(key)

        context = _clean_text((text or "")[max(0, start - 45):min(len(text or ""), end + 45)])
        found.append({"price": price, "context": context[:120]})
        if len(found) >= 12:
            break

    for item in json_ld:
        for offer in item.get("offers", []):
            if offer.get("price") is None:
                continue
            curr = str(offer.get("priceCurrency") or "").strip()
            price_val = str(offer.get("price") or "").strip()
            full_price = f"{curr} {price_val}".strip()
            key = full_price.casefold()
            if key not in seen:
                seen.add(key)
                found.append({"price": full_price, "source": "Schema Offer"})
        if len(found) >= 15:
            break

    return found[:15]


def _extract_ctas(soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for tag in soup.find_all(["a", "button", "input"]):
        text = _clean_text(tag.get_text(" ", strip=True) or tag.get("aria-label", "") or tag.get("value", ""))
        classes = " ".join(tag.get("class", []))
        identity = f"{classes} {tag.get('id', '')}".lower()

        if not text or not (2 <= len(text) <= 80):
            continue

        class_match = any(word in identity for word in ("btn", "button", "cta", "buy", "shop", "cart", "order", "checkout"))
        text_match = any(word in text.casefold() for word in CTA_KEYWORDS)

        if not (class_match or text_match):
            continue

        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)

        href = urljoin(base_url, tag.get("href", "")) if tag.get("href") else ""
        item = {"text": text}
        if href and href.startswith("http"):
            item["url"] = href[:250]
        output.append(item)
        if len(output) >= 15:
            break

    return output


def _extract_important_links(soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
    base_host = (urlparse(base_url).hostname or "").lower().removeprefix("www.")
    keywords = (
        "product", "products", "shop", "collection", "category",
        "service", "offer", "contact", "about", "cart",
        "منتج", "منتجات", "تسوق", "خدمة", "عروض", "تواصل", "من-نحن",
    )
    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for tag in soup.find_all("a", href=True):
        raw_href = tag["href"].strip()
        href = urljoin(base_url, raw_href)
        parsed = urlparse(href)
        host = (parsed.hostname or "").lower().removeprefix("www.")

        if parsed.scheme not in {"http", "https"} or host != base_host:
            continue

        text = _clean_text(tag.get_text(" ", strip=True) or tag.get("aria-label", ""))
        relevant = any(k in href.lower() for k in keywords) or any(k in text.casefold() for k in keywords)

        if not text or not relevant:
            continue

        clean_url = href.split("#", 1)[0]
        if clean_url in seen:
            continue
        seen.add(clean_url)

        output.append({"text": text[:80], "url": clean_url})
        if len(output) >= 10:
            break

    return output


@tool(args_schema=AnalyzeWebsiteInput)
def analyze_website_tool(url: str, config: RunnableConfig | None = None) -> str:
    """Fetch and deeply analyze a website URL in both Arabic and English.
    
    Extracts brand positioning, core offers, Egyptian & USD price anchors,
    CTAs, trust signals (free shipping, cash on delivery, warranties),
    and structured JSON-LD schemas.
    
    Args:
        url: The website URL to inspect (e.g. https://brand.com/products).
    """
    settings = get_ai_settings()
    url = (url or "").strip()
    if not url:
        return "Error: URL is empty."

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    # 1. SSRF & Inbound Security Check
    try:
        safe = resolve_and_check(url)
        target_url = safe.url
    except BlockedTargetError as exc:
        return f"Error: Target URL blocked for security reasons: {exc.message}"
    except Exception as exc:
        return f"Error validating URL {url}: {str(exc)}"

    timeout = float(getattr(settings, "website_timeout_seconds", 25.0) or 25.0)

    # 2. Fetch Website with Content-Length bounds
    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout, connect=8.0),
            follow_redirects=True,
            max_redirects=5,
            headers=HEADERS,
        ) as client:
            resp = client.get(target_url)
            if resp.status_code >= 400:
                return f"Error: Website returned HTTP {resp.status_code}."

            content_type = resp.headers.get("content-type", "").lower()
            if content_type and "html" not in content_type and "xhtml" not in content_type:
                return f"Notice: URL returned non-HTML content ({content_type})."

            length = resp.headers.get("content-length")
            if length and length.isdigit() and int(length) > MAX_HTML_BYTES:
                return "Error: HTML document exceeds the 4 MB limit."

            html = resp.text
            final_url = str(resp.url)
    except httpx.TimeoutException:
        return f"Error: Connection to {url} timed out."
    except httpx.HTTPError as exc:
        return f"Error: Could not reach {url} ({exc.__class__.__name__})."
    except Exception as exc:
        return f"Unexpected error fetching {url}: {str(exc)}"

    # 3. Parse HTML
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = _clean_text(title_tag.get_text(" ", strip=True))[:250] if title_tag else ""
    meta_description = _first_meta(soup, [{"name": "description"}, {"property": "og:description"}])
    og_tags = _extract_open_graph(soup)
    json_ld = _extract_json_ld(soup)
    headings = _extract_headings(soup)
    ctas = _extract_ctas(soup, final_url)
    important_links = _extract_important_links(soup, final_url)

    # 4. Extract Main Content Text
    content_soup = BeautifulSoup(html, "html.parser")
    for tag in content_soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "svg", "form"]):
        tag.decompose()

    main_nodes = content_soup.select("main, article, [role='main']")
    content_node = (
        max(main_nodes, key=lambda node: len(node.get_text(" ", strip=True)))
        if main_nodes
        else (content_soup.body or content_soup)
    )

    full_text = _clean_text(content_node.get_text(" ", strip=True))
    if len(full_text) < 200:
        fallback_soup = BeautifulSoup(html, "html.parser")
        for tag in fallback_soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "svg"]):
            tag.decompose()
        full_text = _clean_text(fallback_soup.get_text(" ", strip=True))

    prices = _extract_prices(full_text, json_ld)
    lowered = full_text.casefold()
    trust_signals = [w for w in TRUST_KEYWORDS if w.casefold() in lowered][:15]

    max_chars = int(getattr(settings, "website_text_chars", MAX_BODY_CHARS) or MAX_BODY_CHARS)
    body_snippet = full_text[:max_chars]

    extracted_report = {
        "url": final_url,
        "title": title or og_tags.get("og:title", ""),
        "meta_description": meta_description or og_tags.get("og:description", ""),
        "brand_name": og_tags.get("og:site_name", ""),
        "trust_signals_found": trust_signals,
        "pricing_cues": prices,
        "calls_to_action": ctas,
        "headings": headings,
        "structured_data_entities": json_ld,
        "important_links": important_links,
        "body_text_excerpt": body_snippet,
    }

    return (
        f"### Website Analysis for {final_url}\n\n"
        f"**Title:** {extracted_report['title']}\n"
        f"**Description:** {extracted_report['meta_description']}\n\n"
        f"```json\n{json.dumps(extracted_report, ensure_ascii=False, indent=2)}\n```"
    )
