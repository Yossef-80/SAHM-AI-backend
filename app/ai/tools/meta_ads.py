"""AI-side adapters for read-only Meta Ad Library research.

The existing Meta connector and typed handler stay in ``app.core.tools``. This
module exposes only a narrow AI-facing adapter and safely inspects the public
snapshot returned by that handler. Raw snapshot URLs are never sent to an LLM
or logged because Meta may include temporary credentials in them.
"""

from __future__ import annotations

import base64
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.core.security import check_redirect, resolve_and_check
from app.core.tools.base import ToolContext
from app.schemas.tools import GetCompetitorAdsInput, GetCompetitorAdsOutput

MAX_SNAPSHOT_HTML_BYTES = 1_500_000
MAX_SNAPSHOT_IMAGE_BYTES = 2_000_000
SNAPSHOT_TIMEOUT_SECONDS = 12.0
MAX_SNAPSHOT_REDIRECTS = 3
_TOKEN_QUERY_VALUE = re.compile(
    r"(?i)([?&](?:access_token|token|api[_-]?key|client_secret|app_secret)=)[^&#\s\"']+"
)


class _HttpxCredentialRedactionFilter(logging.Filter):
    """Redact query-string credentials from HTTPX's request-summary log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _TOKEN_QUERY_VALUE.sub(r"\1***REDACTED***", message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


@contextmanager
def _redact_httpx_credentials():
    credential_filter = _HttpxCredentialRedactionFilter()
    loggers = (logging.getLogger("httpx"), logging.getLogger("httpx._client"))
    for logger in loggers:
        logger.addFilter(credential_filter)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(credential_filter)


@dataclass(frozen=True)
class SnapshotImage:
    """An image extracted from a Meta snapshot.

    ``base64_data`` is present only while fetching/processing the snapshot;
    assistant blocks contain the short-lived opaque ``preview_id`` instead.
    """

    ad_id: str
    mime_type: str
    base64_data: str = ""
    preview_id: str | None = None


def public_ad_library_url(ad_id: str | None) -> str | None:
    """Build a clean citation URL; never expose Meta's token-bearing snapshot URL."""
    if not ad_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(ad_id)):
        return None
    return f"https://www.facebook.com/ads/library/?id={ad_id}"


async def search_competitor_ads(
    payload: GetCompetitorAdsInput, context: ToolContext
) -> GetCompetitorAdsOutput:
    """Call the existing core Meta Ad Library handler without duplicating it."""
    # Lazy import keeps non-ad assistant turns from loading the Meta research tools.
    from app.core.tools.research import get_competitor_ads_tool

    # The existing Meta handler puts the access token in the Graph API query.
    # HTTPX logs request URLs at INFO, so redact sensitive query parameters only
    # while this route is active; concurrent HTTPX records are safely redacted too.
    with _redact_httpx_credentials():
        result = await get_competitor_ads_tool.handler(payload, context)

    if isinstance(result, GetCompetitorAdsOutput):
        return result
    return GetCompetitorAdsOutput.model_validate(result)


def _host_allowed(host: str, *, initial_snapshot: bool = False) -> bool:
    host = host.lower().rstrip(".")
    if initial_snapshot:
        return host == "facebook.com" or host.endswith(".facebook.com")
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in ("facebook.com", "fbcdn.net", "fbsbx.com")
    )


def _validated_url(url: str, *, initial_snapshot: bool = False) -> str | None:
    """Reject non-HTTPS, non-Meta, credentialed, unusual-port, or private URLs."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.scheme.lower() != "https" or not _host_allowed(
            host, initial_snapshot=initial_snapshot
        ):
            return None
        if parsed.username or parsed.password or parsed.port not in (None, 443):
            return None
        if initial_snapshot and not any(
            marker in parsed.path.lower()
            for marker in ("/ads/archive/render_ad", "/ads/library")
        ):
            return None
        safe = resolve_and_check(url, allowed_schemes=("https",))
        if not _host_allowed(safe.host, initial_snapshot=initial_snapshot):
            return None
        return safe.url
    except Exception:  # noqa: BLE001 - security parsing and DNS failures are a safe rejection.
        # Never include the URL in an exception/log message; it may contain a token.
        return None


async def _fetch_limited(
    url: str,
    *,
    max_bytes: int,
    expected_prefixes: tuple[str, ...],
    initial_snapshot: bool = False,
) -> tuple[bytes, str] | None:
    current = _validated_url(url, initial_snapshot=initial_snapshot)
    if current is None:
        return None

    with _redact_httpx_credentials():
        try:
            async with httpx.AsyncClient(
                timeout=SNAPSHOT_TIMEOUT_SECONDS,
                follow_redirects=False,
                headers={
                    "user-agent": "Mozilla/5.0 (compatible; SahmAdResearch/1.0)",
                    "accept": "text/html,image/avif,image/webp,image/png,image/jpeg,*/*;q=0.7",
                },
            ) as client:
                for hop in range(MAX_SNAPSHOT_REDIRECTS + 1):
                    safe_current = _validated_url(
                        current, initial_snapshot=initial_snapshot and hop == 0
                    )
                    if safe_current is None:
                        return None
                    async with client.stream("GET", safe_current) as response:
                        if response.is_redirect:
                            if hop >= MAX_SNAPSHOT_REDIRECTS:
                                return None
                            location = response.headers.get("location")
                            if not location:
                                return None
                            next_url = str(response.url.join(location))
                            safe_next = check_redirect(next_url, allowed_schemes=("https",))
                            if not _host_allowed(safe_next.host):
                                return None
                            current = safe_next.url
                            continue

                        if response.status_code < 200 or response.status_code >= 300:
                            return None
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                        if not any(content_type.startswith(prefix) for prefix in expected_prefixes):
                            return None
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > max_bytes:
                            return None
                        chunks = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(chunks) + len(chunk) > max_bytes:
                                return None
                            chunks.extend(chunk)
                        if not chunks:
                            return None
                        return bytes(chunks), content_type
        except Exception:  # noqa: BLE001 - snapshot network errors must not fail the ad query.
            # A snapshot is optional evidence; access failures must not fail the ad search.
            return None
    return None


def _image_candidates(html: bytes, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []
    for attrs in (
        {"property": "og:image"},
        {"name": "twitter:image"},
        {"property": "og:image:secure_url"},
    ):
        node = soup.find("meta", attrs=attrs)
        if node and node.get("content"):
            candidates.append(str(node["content"]))

    for node in soup.find_all(["img", "video", "source"]):
        for attr in ("src", "data-src", "data-original", "poster"):
            value = node.get(attr)
            if value:
                candidates.append(str(value))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        absolute = urljoin(base_url, candidate.strip())
        if absolute not in seen:
            seen.add(absolute)
            unique.append(absolute)
    return unique


async def inspect_ad_snapshot(ad_id: str | None, snapshot_url: str | None) -> SnapshotImage | None:
    """Fetch one public Meta snapshot and its first safe image, in memory only."""
    if not snapshot_url:
        return None
    snapshot = _validated_url(snapshot_url, initial_snapshot=True)
    if snapshot is None:
        return None

    page = await _fetch_limited(
        snapshot,
        max_bytes=MAX_SNAPSHOT_HTML_BYTES,
        expected_prefixes=("text/html", "application/xhtml+xml", "image/"),
        initial_snapshot=True,
    )
    if page is None:
        return None
    content, content_type = page

    if content_type.startswith("image/"):
        return SnapshotImage(
            ad_id=ad_id or "unknown",
            mime_type=content_type,
            base64_data=base64.b64encode(content).decode("ascii"),
        )

    for candidate in _image_candidates(content, snapshot):
        image_url = _validated_url(candidate)
        if image_url is None:
            continue
        fetched = await _fetch_limited(
            image_url,
            max_bytes=MAX_SNAPSHOT_IMAGE_BYTES,
            expected_prefixes=("image/",),
        )
        if fetched is None:
            continue
        image_bytes, mime_type = fetched
        if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            continue
        return SnapshotImage(
            ad_id=ad_id or "unknown",
            mime_type=mime_type,
            base64_data=base64.b64encode(image_bytes).decode("ascii"),
        )
    return None


__all__ = [
    "SnapshotImage",
    "inspect_ad_snapshot",
    "public_ad_library_url",
    "search_competitor_ads",
]
