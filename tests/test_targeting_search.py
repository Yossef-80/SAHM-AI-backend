"""Meta Targeting Search — interests must resolve to IDs before they can be sent.

Meta rejects interest *names* in targeting; it needs the numeric IDs from
`GET /search?type=adinterest`. This endpoint is what lets the wizard's interests
field reach Meta at all, so the important behaviours are: it asks Meta for the
`adinterest` type, it maps the response into the shape `flexible_spec` wants, and
it degrades to `available: false` instead of a 5xx when Meta is absent or
unhappy — a missing grant is an expected state, not a server fault.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from app.ai.marketing.meta_ads import MetaConnectionConfig
from app.api import integrations as integrations_api
from app.api.deps import WORKSPACE_HEADER

URL = "/api/v1/integrations/meta/targeting-search"
HEADERS = {WORKSPACE_HEADER: "biz_test"}

GOLF = {
    "id": "6003070872673",
    "name": "Golf",
    "audience_size": 41_200_000,
    "path": ["Golf"],
    "type": "interest",
}
CARS = {
    "id": "6003139266461",
    "name": "Cars",
    "audience_size": None,
    "path": ["Vehicles", "Cars"],
}


def _stub_connection(monkeypatch: pytest.MonkeyPatch, token: str | None) -> None:
    async def fake_resolve(db: Any, business_id: str, settings: Any) -> MetaConnectionConfig:
        return MetaConnectionConfig(
            access_token=token,
            ad_account_id="987",
            page_id="page_123",
            graph_base_url="https://graph.facebook.com",
        )

    monkeypatch.setattr(integrations_api, "resolve_meta_connection", fake_resolve)


async def test_interests_are_looked_up_as_adinterest_and_mapped(
    api_client, monkeypatch, respx_mock
) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    route = respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(200, json={"data": [GOLF, CARS]})
    )

    resp = await api_client.get(URL, params={"q": "golf"}, headers=HEADERS)

    assert resp.status_code == 200, resp.text
    assert route.called
    # The whole point: Meta only returns usable IDs for this type.
    assert route.calls.last.request.url.params["type"] == "adinterest"
    assert route.calls.last.request.url.params["q"] == "golf"

    body = resp.json()
    assert body["available"] is True
    # ApiModel serialises camelCase, so the client reads audienceSize.
    assert body["results"][0] == {
        "id": "6003070872673",
        "name": "Golf",
        "audienceSize": 41_200_000,
        "path": ["Golf"],
    }
    assert body["results"][1]["audienceSize"] is None
    assert body["results"][1]["path"] == ["Vehicles", "Cars"]


async def test_no_meta_grant_is_unavailable_and_never_calls_meta(
    api_client, monkeypatch, respx_mock
) -> None:
    _stub_connection(monkeypatch, None)
    route = respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    resp = await api_client.get(URL, params={"q": "golf"}, headers=HEADERS)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is False
    assert body["unavailableReason"] == "meta_not_connected"
    assert body["results"] == []
    assert not route.called


async def test_meta_rejection_degrades_instead_of_raising(
    api_client, monkeypatch, respx_mock
) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(400, json={"error": {"code": 100, "message": "bad"}})
    )

    resp = await api_client.get(URL, params={"q": "golf"}, headers=HEADERS)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is False
    assert body["unavailableReason"] == "meta_lookup_failed"
    # The raw Meta message must not be echoed back to the client.
    assert "bad" not in resp.text


async def test_malformed_rows_are_skipped_not_forwarded(
    api_client, monkeypatch, respx_mock
) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    GOLF,
                    {"id": "", "name": "No id"},
                    {"id": "123", "name": "   "},
                    "not-a-dict",
                    {"name": "No id key"},
                ]
            },
        )
    )

    resp = await api_client.get(URL, params={"q": "golf"}, headers=HEADERS)

    assert resp.status_code == 200, resp.text
    assert [r["id"] for r in resp.json()["results"]] == ["6003070872673"]


async def test_an_empty_query_is_rejected(api_client, monkeypatch) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    resp = await api_client.get(URL, params={"q": ""}, headers=HEADERS)
    assert resp.status_code == 422, resp.text


async def test_a_whitespace_query_is_reported_not_sent(
    api_client, monkeypatch, respx_mock
) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    route = respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    resp = await api_client.get(URL, params={"q": "   "}, headers=HEADERS)

    assert resp.status_code == 200, resp.text
    assert resp.json()["unavailableReason"] == "empty_query"
    assert not route.called
