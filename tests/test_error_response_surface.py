"""A 500 has to reach the browser, not look like a dead backend.

An exception that escapes the request-id middleware is turned into an HTTP
response by Starlette's ``ServerErrorMiddleware`` -- the outermost layer, outside
both the CORS middleware and the security-header middleware. The response
therefore carried neither, and a browser refuses to hand such a response to
``fetch`` at all: the client had no status, no body and nothing to report, so the
UI showed "could not reach the backend" while the server was up and had in fact
answered. These tests pin the headers that make a real 5xx readable *and* keep
CORS from being widened for it.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

ALLOWED_ORIGIN = "http://localhost:3000"
FOREIGN_ORIGIN = "http://attacker.example"


async def _crash_inside_a_request(
    api_client: object, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch, origin: str
) -> object:
    """POST to a route that raises, the way a real bug would.

    The ASGI transport re-raises an app exception after the response has been
    sent, so it has to be told not to: a browser only ever sees the response.
    """
    import app.api.creatives as creatives_module

    async def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("image provider exploded")

    monkeypatch.setattr(creatives_module, "generate_image_for_workspace", _boom)
    transport = api_client._transport  # type: ignore[attr-defined]
    transport.raise_app_exceptions = False
    return await api_client.post(  # type: ignore[attr-defined]
        "/api/v1/creatives/generate",
        headers={"X-Workspace-Id": "biz_test", "Origin": origin},
        json={"prompt": "صمم لي إعلان", "aspectRatio": "1:1", "numImages": 1},
    )


async def test_unhandled_500_is_readable_from_the_browser(
    api_client: object, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    resp = await _crash_inside_a_request(api_client, seeded_db, monkeypatch, ALLOWED_ORIGIN)

    assert resp.status_code == 500  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    assert body["error"]["code"] == "internal_error"

    headers = resp.headers  # type: ignore[attr-defined]
    assert headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert headers["access-control-allow-credentials"] == "true"
    # The security headers a 4xx would have carried: a 5xx body is still JSON
    # served to a browser and must not be sniffable or framable.
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    # The request id in the log line has to travel with the response, otherwise
    # the user can quote an error the operator cannot find.
    assert headers["x-request-id"]
    assert headers["x-request-id"] == body["error"]["requestId"]


async def test_unhandled_500_does_not_widen_cors(
    api_client: object, seeded_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an allowlisted origin is echoed; everyone else gets no CORS grant."""
    resp = await _crash_inside_a_request(api_client, seeded_db, monkeypatch, FOREIGN_ORIGIN)

    assert resp.status_code == 500  # type: ignore[attr-defined]
    assert "access-control-allow-origin" not in resp.headers  # type: ignore[attr-defined]
