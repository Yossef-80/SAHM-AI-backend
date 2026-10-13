"""Offline extraction and fetch-boundary tests for the evidence-first site audit."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from app.ai.tools import analyze_website as website
from app.core.errors import BlockedTargetError


FIXTURE_HTML = """<!doctype html>
<html lang="en">
<head>
  <title>River Roast Coffee | Fresh beans</title>
  <meta name="description" content="Small-batch coffee delivered across Cairo">
  <meta property="product:price:amount" content="275">
  <meta property="product:price:currency" content="EGP">
  <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Product","name":"House blend",
     "offers":{"@type":"Offer","price":"275","priceCurrency":"EGP","availability":"https://schema.org/InStock"}}
  </script>
  <script>const privateText = "EGP 9999 - fake price must not appear";</script>
</head>
<body>
  <h1>Fresh coffee, roasted in Cairo</h1>
  <h2>House blend — EGP 275</h2>
  <p>Free delivery across Cairo. Every bag is roasted to order.</p>
  <p>14-day returns are available for unopened bags.</p>
  <button>Order now</button>
  <a href="/delivery-policy">Delivery and returns</a>
  <a href="https://other.example/pricing">Pricing details</a>
</body>
</html>"""


class _FakeResponse:
    def __init__(self, *, status_code: int, headers: dict[str, str], content: bytes, url: str):
        self.status_code = status_code
        self.headers = headers
        self.content = content
        self.url = httpx.URL(url)
        self.encoding = "utf-8"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "fake status", request=httpx.Request("GET", str(self.url)), response=httpx.Response(self.status_code)
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def aiter_bytes(self):
        yield self.content


class _FakeAsyncClient:
    response: _FakeResponse

    def __init__(self, *args, **kwargs):
        self.calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, method: str, url: str):
        assert method == "GET"
        self.calls.append(str(url))
        return self.response


def _safe_url(url: str, **_kwargs):
    return SimpleNamespace(url=url, host="example.com", port=443, scheme="https")


def test_extract_audit_keeps_page_facts_and_evidence_separate() -> None:
    final_url = "https://shop.example.com/coffee"
    audit = website._extract_audit(
        FIXTURE_HTML,
        final_url=final_url,
        requested_url=final_url,
        max_chars=500,
    )

    assert audit.status == "ok"
    assert audit.title == "River Roast Coffee | Fresh beans"
    assert audit.description == "Small-batch coffee delivered across Cairo"
    assert audit.language == "en"
    assert "Fresh coffee, roasted in Cairo" in audit.headings
    assert any(f.value == "275 EGP; https://schema.org/InStock" for f in audit.offers_and_prices)
    assert any(f.value == "EGP 275" for f in audit.offers_and_prices)
    assert any("Order now" in f.value for f in audit.calls_to_action)
    assert any("14-day returns" in f.value for f in audit.trust_signals)
    assert any(f.value == "https://other.example/pricing" for f in audit.important_links)
    assert all(f.source_url == final_url for f in audit.observations)
    assert all(f.evidence for f in audit.observations)
    assert "EGP 9999" not in audit.visible_text_excerpt
    assert audit.analysis_boundaries


def test_extract_audit_reports_missing_price_instead_of_guessing() -> None:
    html = "<html><body><h1>Welcome</h1><p>Contact our team.</p></body></html>"
    audit = website._extract_audit(
        html,
        final_url="https://example.com/",
        requested_url="https://example.com/",
        max_chars=100,
    )
    assert audit.offers_and_prices == []
    assert any("No explicit price" in note for note in audit.not_observed)


def test_normalize_url_rejects_credentials_and_unsupported_schemes() -> None:
    with pytest.raises(ValueError):
        website.normalize_url("https://user:password@example.com")
    with pytest.raises(ValueError):
        website.normalize_url("file:///etc/passwd")
    assert website.normalize_url("example.com/path") == "https://example.com/path"


@pytest.mark.asyncio
async def test_tool_extracts_static_html_without_a_live_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAsyncClient
    fake_client.response = _FakeResponse(
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8", "content-length": str(len(FIXTURE_HTML))},
        content=FIXTURE_HTML.encode(),
        url="https://example.com/coffee",
    )
    monkeypatch.setattr(website, "resolve_and_check", _safe_url)
    monkeypatch.setattr(website.httpx, "AsyncClient", fake_client)

    raw = await website.analyze_website_tool.ainvoke({"url": "https://example.com/coffee"})
    audit = json.loads(raw)
    assert audit["status"] == "ok"
    assert audit["final_url"] == "https://example.com/coffee"
    assert audit["offers_and_prices"]


@pytest.mark.asyncio
async def test_fetch_stops_when_a_chunked_response_exceeds_the_size_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.errors import ToolUnavailable

    class LargeResponse(_FakeResponse):
        async def aiter_bytes(self):
            yield b"12345678"
            yield b"abcdefgh"

    _FakeAsyncClient.response = LargeResponse(
        status_code=200,
        headers={"content-type": "text/html"},
        content=b"",
        url="https://example.com/large",
    )
    monkeypatch.setattr(website, "get_ai_settings", lambda: SimpleNamespace(
        website_timeout_seconds=1.0, website_max_redirects=1, website_max_bytes=10,
    ))
    monkeypatch.setattr(website, "resolve_and_check", _safe_url)
    monkeypatch.setattr(website.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(ToolUnavailable, match="byte cap"):
        await website._fetch_public_html("https://example.com/large")


@pytest.mark.asyncio
async def test_fetch_validates_each_redirect_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    class RedirectClient(_FakeAsyncClient):
        def stream(self, method: str, url: str):
            assert method == "GET"
            self.calls.append(str(url))
            return _FakeResponse(
                status_code=302,
                headers={"location": "http://169.254.169.254/latest/meta-data/"},
                content=b"",
                url=str(url),
            )

    monkeypatch.setattr(website, "resolve_and_check", _safe_url)
    monkeypatch.setattr(website.httpx, "AsyncClient", RedirectClient)

    # Keep the production redirect validator: the private IP literal is rejected
    # without DNS or any outbound request.
    with pytest.raises(BlockedTargetError, match="private/reserved range"):
        await website._fetch_public_html("https://example.com/start")
