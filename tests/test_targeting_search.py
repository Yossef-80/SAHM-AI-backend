"""Meta detailed targeting resolves category-scoped phrases to Meta results.

The wizard must not send free text as a targeting value. Searchable categories
use Meta's typed autocomplete endpoints; browse-only classes are filtered from
Meta's catalog and cached briefly. Missing grants and Meta errors are a normal
unavailable state rather than a server fault.
"""

from __future__ import annotations

import logging
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
        "category": "interests",
        "audienceSize": 41_200_000,
        "description": None,
        "path": ["Golf"],
    }
    assert body["results"][1]["category"] == "interests"
    assert body["results"][1]["audienceSize"] is None
    assert body["results"][1]["description"] is None
    assert body["results"][1]["path"] == ["Vehicles", "Cars"]
    assert body["hasMore"] is False


async def test_comma_separated_input_searches_only_the_active_last_term(
    api_client, monkeypatch, respx_mock
) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    route = respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(200, json={"data": [CARS]})
    )

    resp = await api_client.get(
        URL, params={"q": "قطع غيار، الصيانة, MG"}, headers=HEADERS
    )

    assert resp.status_code == 200, resp.text
    assert route.called
    assert route.calls.last.request.url.params["q"] == "MG"
    assert [item["name"] for item in resp.json()["results"]] == ["Cars"]


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


async def test_meta_rejection_degrades_and_logs_only_sanitized_diagnostics(
    api_client, monkeypatch, respx_mock
) -> None:
    secret = "offline-test-token-should-never-be-logged-123456789"
    _stub_connection(monkeypatch, secret)
    respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(400, json={"error": {"code": 100, "message": "bad"}})
    )

    class CaptureLogger:
        fields: dict[str, Any] | None = None

        def warning(self, _message: str, *, extra: dict[str, Any]) -> None:
            self.fields = extra

    capture = CaptureLogger()
    monkeypatch.setattr(integrations_api, "_logger", capture)
    assert logging.getLogger("httpx").level >= logging.WARNING

    resp = await api_client.get(URL, params={"q": "golf"}, headers=HEADERS)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is False
    assert body["unavailableReason"] == "meta_lookup_failed"
    # The sanitized Meta reason is kept server-side, never echoed to UI or mixed
    # with the configured access token.
    assert capture.fields is not None
    assert "Meta error 100" in capture.fields["reason"]
    assert secret not in repr(capture.fields)
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


@pytest.mark.parametrize(
    ("category", "meta_type"),
    [
        ("interests", "adinterest"),
        ("job_titles", "adworkposition"),
        ("employers", "adworkemployer"),
        ("education_majors", "adeducationmajor"),
        ("education_schools", "adeducationschool"),
    ],
)
async def test_autocomplete_category_uses_its_meta_search_type(
    api_client, monkeypatch, respx_mock, category: str, meta_type: str
) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    route = respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": f"meta-{category}", "name": "Verified result", "coverage": 1234}]},
        )
    )

    resp = await api_client.get(
        URL, params={"q": "verified", "category": category}, headers=HEADERS
    )

    assert resp.status_code == 200, resp.text
    assert route.called
    request_params = route.calls.last.request.url.params
    assert request_params["type"] == meta_type
    assert request_params["q"] == "verified"
    result = resp.json()["results"][0]
    assert result["id"] == f"meta-{category}"
    assert result["category"] == category
    assert result["name"] == "Verified result"


@pytest.mark.parametrize(
    ("category", "meta_class", "query", "name"),
    [
        ("behaviors", "behaviors", "travel", "Frequent travelers"),
        ("life_events", "life_events", "moved", "Recently moved"),
        ("industries", "industries", "food", "Food and beverage"),
        ("income", "income", "household", "Household income segment"),
        ("family_statuses", "family_statuses", "parent", "Parent"),
        ("devices", "user_device", "iphone", "iPhone"),
        ("operating_systems", "user_os", "ios", "iOS"),
    ],
)
async def test_browse_category_filters_meta_catalog_and_preserves_id(
    api_client, monkeypatch, respx_mock, category: str, meta_class: str, query: str, name: str
) -> None:
    _stub_connection(monkeypatch, "offline-test-token")
    integrations_api._TARGETING_CATALOG_CACHE.clear()
    route = respx_mock.get(path__regex=r"/search$").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": f"meta-{category}",
                        "name": name,
                        "audience_size_upper_bound": 4567,
                        "description": "Returned by Meta's category catalogue",
                        "path": ["Category", name],
                    },
                    {"id": "other-id", "name": "Does not match", "path": []},
                ],
                "paging": {},
            },
        )
    )

    resp = await api_client.get(
        URL, params={"q": query, "category": category}, headers=HEADERS
    )

    assert resp.status_code == 200, resp.text
    assert route.called
    request_params = route.calls.last.request.url.params
    assert request_params["type"] == "adTargetingCategory"
    assert request_params["class"] == meta_class
    # Meta ignores q for this browse endpoint; the API filters locally.
    assert "q" not in request_params
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["id"] == f"meta-{category}"
    assert body["results"][0]["category"] == category
    assert body["results"][0]["name"] == name
    assert body["results"][0]["audienceSize"] == 4567
