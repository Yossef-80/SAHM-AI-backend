"""Evidence-first website analyzer used by Nour.

The extraction is deliberately deterministic: every reported offer, price, CTA,
trust phrase, and important link points back to page text or structured JSON-LD.
It fetches only the supplied public HTTP(S) page, validates every redirect hop,
and never follows discovered links automatically.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.ai.settings import get_ai_settings
from app.core.errors import BlockedTargetError, ToolUnavailable
from app.core.security import check_redirect, resolve_and_check


class ObservedFact(BaseModel):
    category: str
    label: str
    value: str
    evidence: str
    source_url: str


class WebsiteAudit(BaseModel):
    status: str
    source: str = "user_supplied_public_website"
    requested_url: str
    final_url: str | None = None
    reason: str | None = None
    title: str | None = None
    description: str | None = None
    language: str | None = None
    headings: list[str] = Field(default_factory=list)
    offers_and_prices: list[ObservedFact] = Field(default_factory=list)
    calls_to_action: list[ObservedFact] = Field(default_factory=list)
    trust_signals: list[ObservedFact] = Field(default_factory=list)
    important_links: list[ObservedFact] = Field(default_factory=list)
    observations: list[ObservedFact] = Field(default_factory=list)
    visible_text_excerpt: str = ""
    not_observed: list[str] = Field(default_factory=list)
    analysis_boundaries: list[str] = Field(default_factory=list)


def normalize_url(url: str) -> str:
    """Normalize a user-supplied URL without allowing credentials or odd schemes."""
    value = str(url or "").strip()
    if not value:
        raise ValueError("a website URL is required")
    if not re.match(r"^[a-z][a-z0-9+.-]*://", value, re.IGNORECASE):
        value = "https://" + value
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"}:
        raise ValueError("only public HTTP or HTTPS pages can be analyzed")
    if not parts.hostname:
        raise ValueError("the URL must include a hostname")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URLs containing usernames or passwords are not accepted")
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path or "/", parts.query, ""))


async def _fetch_public_html(url: str) -> tuple[str, str]:
    """Fetch one public HTML page with SSRF, redirect, timeout, and size guards."""
    settings = get_ai_settings()
    current = normalize_url(url)
    current = resolve_and_check(current, allowed_schemes=("http", "https")).url

    async with httpx.AsyncClient(
        timeout=settings.website_timeout_seconds,
        follow_redirects=False,
        trust_env=False,
        headers={
            "User-Agent": "SahmResearchBot/1.0 (+public-page analysis)",
            "Accept": "text/html,application/xhtml+xml;q=0.9",
        },
    ) as client:
        for redirect_count in range(settings.website_max_redirects + 1):
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location", "").strip()
                    if not location:
                        raise ToolUnavailable("website redirected without a location header")
                    if redirect_count >= settings.website_max_redirects:
                        raise ToolUnavailable("too many redirects while fetching the website")
                    redirected = str(response.url.join(location))
                    current = check_redirect(
                        redirected, allowed_schemes=("http", "https")
                    ).url
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
                    raise ToolUnavailable("the supplied URL did not return an HTML page")
                content_length = response.headers.get("content-length")
                try:
                    declared_size = int(content_length) if content_length else None
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > settings.website_max_bytes:
                    raise ToolUnavailable(
                        f"website response is larger than the {settings.website_max_bytes} byte cap"
                    )

                # Stream into a bounded buffer; post-hoc checks on response.content
                # would already have allowed a chunked response to exhaust memory.
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > settings.website_max_bytes:
                        raise ToolUnavailable(
                            f"website response exceeded the {settings.website_max_bytes} byte cap"
                        )
                    raw.extend(chunk)
                return bytes(raw).decode(response.encoding or "utf-8", errors="replace"), str(response.url)

    raise ToolUnavailable("too many redirects while fetching the website")


def _clean_text(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _json_ld_nodes(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Parse only explicit JSON-LD data; malformed scripts are ignored."""
    nodes: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            nodes.append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        raw = script.string or script.get_text(" ", strip=True)
        try:
            walk(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return nodes


def _types(node: dict[str, Any]) -> set[str]:
    raw = node.get("@type", [])
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).rsplit("/", 1)[-1].lower() for item in raw if item}


def _schema_offers(nodes: list[dict[str, Any]], source_url: str) -> list[ObservedFact]:
    facts: list[ObservedFact] = []
    for node in nodes:
        node_types = _types(node)
        if "product" not in node_types and "offer" not in node_types and "aggregateoffer" not in node_types:
            continue
        name = _clean_text(node.get("name"), 120)
        raw_offers = node.get("offers") if "product" in node_types else node
        offer_nodes = raw_offers if isinstance(raw_offers, list) else [raw_offers]
        for offer in offer_nodes:
            if not isinstance(offer, dict):
                continue
            currency = _clean_text(offer.get("priceCurrency"), 12)
            price = _clean_text(offer.get("price") or offer.get("lowPrice"), 40)
            availability = _clean_text(offer.get("availability"), 80)
            pieces = [piece for piece in (f"{price} {currency}".strip(), availability) if piece]
            if not pieces:
                continue
            label = name or _clean_text(offer.get("name"), 120) or "Structured offer"
            evidence_obj = {key: offer[key] for key in ("price", "lowPrice", "highPrice", "priceCurrency", "availability", "url") if key in offer}
            facts.append(
                ObservedFact(
                    category="structured_offer",
                    label=label,
                    value="; ".join(pieces),
                    evidence="JSON-LD Offer/Product data: " + _clean_text(evidence_obj, 360),
                    source_url=source_url,
                )
            )
            if len(facts) >= 20:
                return facts
    return facts


_PRICE_RE = re.compile(
    r"(?:\b(?:EGP|LE|L\.E\.|USD|EUR|SAR|AED)\s*\d[\d,]*(?:\.\d+)?|"
    r"[$€£]\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s*"
    r"(?:EGP|LE|L\.E\.|USD|EUR|SAR|AED|جنيه|ج\.م))",
    re.IGNORECASE,
)
_CTA_RE = re.compile(
    r"buy|shop|order|subscribe|book|contact|call|whats.?app|sign.?up|"
    r"get started|learn more|add to cart|ابدأ|اشتر|اطلب|تسوق|احجز|تواصل|اتصل|اشترك|اعرف المزيد",
    re.IGNORECASE,
)
_TRUST_RE = re.compile(
    r"return|refund|warranty|guarantee|secure payment|verified|certified|"
    r"customer review|rating|delivery|shipping|privacy policy|استرجاع|استبدال|"
    r"ضمان|توصيل|شحن|تقييم|آراء العملاء|الدفع الآمن",
    re.IGNORECASE,
)
_IMPORTANT_PATH_RE = re.compile(
    r"price|pricing|product|shop|store|contact|about|delivery|shipping|faq|"
    r"return|refund|policy|service|menu|offers|pricing|أسعار|منتج|تواصل|توصيل",
    re.IGNORECASE,
)


def _extract_audit(html: str, final_url: str, requested_url: str, max_chars: int) -> WebsiteAudit:
    soup = BeautifulSoup(html, "html.parser")
    json_ld = _json_ld_nodes(soup)
    title_tag = soup.title
    title = _clean_text(title_tag.get_text(" ", strip=True) if title_tag else "", 240) or None
    description_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    description = _clean_text(description_tag.get("content") if description_tag else "", 500) or None
    html_tag = soup.find("html")
    language = _clean_text(html_tag.get("lang") if html_tag else "", 16) or None

    headings = [
        _clean_text(node.get_text(" ", strip=True), 180)
        for node in soup.find_all(["h1", "h2", "h3"])
        if node.get_text(" ", strip=True)
    ][:24]

    offers = _schema_offers(json_ld, final_url)
    offer_nodes = soup.find_all("meta", attrs={"property": re.compile(r"(?:product|og):price:(?:amount|currency)", re.I)})
    for node in offer_nodes[:12]:
        value = _clean_text(node.get("content"), 80)
        if value:
            offers.append(
                ObservedFact(
                    category="visible_price_metadata",
                    label=_clean_text(node.get("property"), 80) or "Page price metadata",
                    value=value,
                    evidence=f'<meta property="{node.get("property", "")}" content="{value}">',
                    source_url=final_url,
                )
            )

    # Extract display text only after JSON-LD, metadata, and headings have been read.
    for element in soup(["script", "style", "noscript", "svg", "template"]):
        element.decompose()
    visible_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()

    for match in list(_PRICE_RE.finditer(visible_text))[:20]:
        start, end = max(0, match.start() - 65), min(len(visible_text), match.end() + 65)
        quote = visible_text[start:end]
        if not any(f.evidence == quote for f in offers):
            offers.append(
                ObservedFact(
                    category="visible_price_text",
                    label="Price-like text observed",
                    value=match.group(0),
                    evidence=quote,
                    source_url=final_url,
                )
            )

    ctas: list[ObservedFact] = []
    trust: list[ObservedFact] = []
    important_links: list[ObservedFact] = []
    seen_ctas: set[tuple[str, str]] = set()
    seen_links: set[str] = set()

    for element in soup.find_all(["a", "button", "input"]):
        label = _clean_text(element.get("aria-label") or element.get("value") or element.get_text(" ", strip=True), 120)
        if not label:
            continue
        href = str(element.get("href") or "").strip()
        target = urljoin(final_url, href) if href else ""
        parsed_target = urlsplit(target) if target else None
        if _CTA_RE.search(label):
            key = (label.lower(), target)
            if key not in seen_ctas:
                seen_ctas.add(key)
                ctas.append(
                    ObservedFact(
                        category="call_to_action",
                        label="Visible CTA",
                        value=f"{label}" + (f" → {target}" if target else ""),
                        evidence=f'Visible control text: "{label}"',
                        source_url=final_url,
                    )
                )
        if target and parsed_target and parsed_target.scheme in {"http", "https"}:
            if _IMPORTANT_PATH_RE.search(parsed_target.path + " " + label) and target not in seen_links:
                seen_links.add(target)
                important_links.append(
                    ObservedFact(
                        category="important_link",
                        label=label or parsed_target.path or "Important link",
                        value=target,
                        evidence=f'Visible anchor text: "{label}"; href: {target}',
                        source_url=final_url,
                    )
                )

    for sentence in re.split(r"(?<=[.!?؟])\s+|[\n|]", visible_text):
        snippet = _clean_text(sentence, 220)
        if snippet and _TRUST_RE.search(snippet):
            trust.append(
                ObservedFact(
                    category="trust_or_service_claim",
                    label="Page text mentioning trust, service, or policy",
                    value=snippet,
                    evidence=f'Visible page text: "{snippet}"',
                    source_url=final_url,
                )
            )
        if len(trust) >= 15:
            break

    observations: list[ObservedFact] = []
    if title:
        observations.append(ObservedFact(category="page_metadata", label="Title", value=title, evidence=f"HTML title: {title}", source_url=final_url))
    if description:
        observations.append(ObservedFact(category="page_metadata", label="Meta description", value=description, evidence=f"HTML meta description: {description}", source_url=final_url))
    observations.extend(
        ObservedFact(category="heading", label=f"Heading {index + 1}", value=value, evidence=f"Visible heading: {value}", source_url=final_url)
        for index, value in enumerate(headings)
    )
    observations.extend(offers)
    observations.extend(ctas)
    observations.extend(trust)
    observations.extend(important_links)

    not_observed: list[str] = []
    if not offers:
        not_observed.append("No explicit price or offer was observed in the page text or structured data.")
    if not trust:
        not_observed.append("No matching trust, review, delivery, or policy wording was observed in the extracted page text.")
    if not ctas:
        not_observed.append("No recognizable call-to-action control was observed.")
    return WebsiteAudit(
        status="ok",
        requested_url=requested_url,
        final_url=final_url,
        title=title,
        description=description,
        language=language,
        headings=headings,
        offers_and_prices=offers,
        calls_to_action=ctas,
        trust_signals=trust,
        important_links=important_links,
        observations=observations,
        visible_text_excerpt=visible_text[:max_chars],
        not_observed=not_observed,
        analysis_boundaries=[
            "These are observations from one fetched public page, not a full-site audit.",
            "Discovered links are listed but are not fetched automatically.",
            "Do not infer audience size, market share, conversion rate, campaign performance, or budget from page appearance.",
            "Keep any downstream inference or recommendation separate from these quoted observations and name its evidence.",
        ],
    )


def _error_audit(requested_url: str, reason: str) -> str:
    return WebsiteAudit(
        status="unavailable",
        requested_url=requested_url,
        reason=reason,
        analysis_boundaries=["No website facts were extracted; do not replace missing data with guesses."],
    ).model_dump_json(indent=2)


@tool
async def analyze_website_tool(url: str, config: RunnableConfig) -> str:
    """Analyze one supplied public website page for evidence-backed marketing signals.

    Extracts metadata, headings, JSON-LD/visible prices, offers, CTAs, trust or
    service wording, and important links. It does not fetch discovered links,
    access private networks, or claim anything not observed on the supplied page.

    Args:
        url: The public HTTP(S) page the user or a verified search result supplied.
    """
    requested_url = str(url or "").strip()
    try:
        normalized = normalize_url(requested_url)
        html, final_url = await _fetch_public_html(normalized)
        audit = _extract_audit(
            html,
            final_url,
            requested_url=normalized,
            max_chars=get_ai_settings().website_text_chars,
        )
        return audit.model_dump_json(indent=2)
    except (BlockedTargetError, ToolUnavailable, ValueError) as exc:
        return _error_audit(requested_url, str(exc))
    except httpx.HTTPStatusError as exc:
        return _error_audit(requested_url, f"website returned HTTP {exc.response.status_code}")
    except httpx.HTTPError as exc:
        return _error_audit(requested_url, f"website fetch failed: {type(exc).__name__}")
    except Exception as exc:  # defensive boundary for a tool called by the LLM
        return _error_audit(requested_url, f"website analysis failed: {type(exc).__name__}")
