"""Meta's error text must reach the operator.

A live campaign creation that dies at its first step used to report only
"Meta rejected the request (Meta error 100). Check the submitted settings and
permissions." — identical for a typo'd objective, an ad account the token cannot
reach, and a missing permission. Error 100 is Meta's catch-all for "invalid
parameter", so the numeric code alone carries no diagnostic value at all.

What still must NOT leak is the request URL: the access token is a query
parameter on it. These tests pin both halves.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.ai.marketing.meta_ads import HttpMetaGraphClient

TOKEN = "EAABsecrettokenvalue1234567890ABCDEF"


def _client() -> HttpMetaGraphClient:
    class _S:
        meta_graph_version = "v21.0"
        meta_api_timeout_seconds = 5.0

    return HttpMetaGraphClient(
        settings=_S(), access_token=TOKEN, base_url="https://graph.facebook.com"
    )


def test_a_realistic_error_100_surfaces_the_reason() -> None:
    payload = {
        "error": {
            "message": "Invalid parameter",
            "type": "OAuthException",
            "code": 100,
            "error_subcode": 33,
            "error_user_title": "Invalid Objective",
            "error_user_msg": "The objective you selected is not supported for this ad account.",
            "error_data": {"blame_field_specs": ["objective"]},
            "fbtrace_id": "A1B2C3",
        }
    }
    message = HttpMetaGraphClient._error_message(payload, 400)

    assert "Meta error 100" in message
    # The human-readable reason is the whole point.
    assert "not supported for this ad account" in message
    assert "subcode 33" in message
    assert "field: objective" in message


def test_the_unsupported_post_request_variant_is_distinguishable() -> None:
    """The single most common real-world error 100 at campaign creation."""
    payload = {
        "error": {
            "message": (
                "Unsupported post request. Object with ID 'act_1234567890' does not "
                "exist, cannot be loaded due to missing permissions, or does not "
                "support this operation."
            ),
            "type": "GraphMethodException",
            "code": 100,
        }
    }
    message = HttpMetaGraphClient._error_message(payload, 400)
    assert "Unsupported post request" in message
    assert "missing permissions" in message


def test_token_shaped_strings_are_redacted() -> None:
    payload = {"error": {"code": 100, "message": f"bad token {TOKEN} was supplied"}}
    message = HttpMetaGraphClient._error_message(payload, 400)
    assert TOKEN not in message
    assert "[redacted]" in message


def test_the_budget_sharing_field_name_is_not_redacted_as_a_token() -> None:
    payload = {
        "error": {
            "code": 100,
            "error_subcode": 4834011,
            "message": (
                "You must specify True or False in the field "
                "is_adset_budget_sharing_enabled if you are not using campaign budget."
            ),
        }
    }
    message = HttpMetaGraphClient._error_message(payload, 400)
    assert "is_adset_budget_sharing_enabled" in message
    assert "subcode 4834011" in message


def test_an_error_without_a_body_still_reports_the_status() -> None:
    assert HttpMetaGraphClient._error_message({}, 500) == (
        "Meta rejected the request (HTTP 500)."
    )


def test_a_long_message_is_capped() -> None:
    payload = {"error": {"code": 100, "message": "x " * 400}}
    assert len(HttpMetaGraphClient._error_message(payload, 400)) < 400


@pytest.mark.anyio
async def test_the_url_and_token_never_reach_the_raised_error(respx_mock) -> None:
    """End to end: what the caller sees must be informative and token-free."""
    route = respx_mock.post(path__regex=r"/campaigns$").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": 100,
                    "message": "Invalid parameter",
                    "error_user_msg": "Daily budget is too low for this ad account.",
                    "error_subcode": 1487,
                }
            },
        )
    )
    assert route is not None

    with pytest.raises(Exception) as caught:
        await _client().post("/act_123/campaigns", {"name": "x", "objective": "y"})

    text = str(caught.value)
    assert "Daily budget is too low" in text
    assert "subcode 1487" in text
    assert TOKEN not in text
    assert "access_token" not in text
    assert "graph.facebook.com" not in text
