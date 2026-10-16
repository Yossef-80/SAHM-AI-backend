"""The image generator's stability contract, asserted rather than hoped for.

Four things went wrong in this path and each one is pinned here:

1. ``n`` was hardcoded to 1 while the schema and the UI offered up to four, so
   "generate 4" quietly produced one image. The product decision is now a single
   image, and *the number sent to the provider is the number the app resolves*.
2. A provider 400/429/503 reached the user as ``Client error '400 Bad Request'``
   with the provider's own explanation (safety filter? bad size? key?) discarded.
3. A fixed 120s ceiling cut off renders the provider itself says take longer --
   and those renders were still billed.
4. Anything unexpected (HTML instead of JSON, a library error) escaped as a 500
   after a minute of waiting.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.core.errors import ToolUnavailable
from app.core.tools.media import (
    _DEFAULT_IMAGE_TIMEOUT_SECONDS,
    _generate_openai_image,
    _image_timeout_seconds,
    _provider_error_reason,
    _usage_summary,
    generate_creative_assets,
)

pytestmark = pytest.mark.anyio

BASE = "https://image-provider.invalid/v1"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x22" * 16


def _cfg(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = dict(
        supabase_url=None,
        supabase_secret_key=None,
        supabase_service_role_key=None,
        supabase_publishable_key=None,
        public_asset_base_url=None,
        upload_dir=None,
        upload_max_bytes=10_000_000,
        image_timeout_seconds=300.0,
        image_allow_mock_assets=False,
        image_provider="openai",
        image_api_key="sk-test",
        image_model="gpt-image-1",
        image_base_url=BASE,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _no_cloud(monkeypatch) -> None:
    """No ambient Supabase credentials: persistence must not leave the process."""
    for name in (
        "SUPABASE_URL",
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_PUBLISHABLE_KEY",
        "SUPABASE_ANON_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _route(respx_mock, responses: list[httpx.Response], captured: list[dict] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(
                {
                    "path": request.url.path,
                    "headers": dict(request.headers),
                    "body": request.content,
                }
            )
        index = min(len(captured) - 1 if captured else 0, len(responses) - 1)
        return responses[index]

    respx_mock.route(method="POST", host="image-provider.invalid").mock(side_effect=handler)


def _ok(url: str = "https://assets.example.com/out.png") -> httpx.Response:
    return httpx.Response(200, json={"data": [{"url": url}]})


# --------------------------------------------------------------- one image ---


async def test_the_app_path_asks_for_one_image_however_many_were_requested(
    monkeypatch, respx_mock
) -> None:
    """The count the app promises is the count that reaches the provider.

    The real provider call runs here (only the HTTP layer is mocked), so this
    pins the whole decision: asking for four renders one image, the payload says
    ``n=1``, and the caller gets one asset.
    """
    _no_cloud(monkeypatch)
    captured: list[dict] = []
    _route(respx_mock, [_ok()], captured)

    assets, reason = await generate_creative_assets(
        cfg=_cfg(),
        prompt="a coffee bag",
        aspect_ratio="1:1",
        num_images=4,
        provider="openai",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
    )

    assert reason is None
    assert len(captured) == 1
    payload = json.loads(captured[0]["body"])
    assert payload["n"] == 1
    assert "input_fidelity" not in payload
    assert len(assets) == 1


async def test_a_gateway_returning_extra_images_still_yields_one_creative(
    monkeypatch, respx_mock
) -> None:
    """``n`` is a request, not a promise: some gateways ignore it."""
    _no_cloud(monkeypatch)
    captured: list[dict] = []
    _route(
        respx_mock,
        [
            httpx.Response(
                200,
                json={
                    "data": [
                        {"url": "https://assets.example.com/a.png"},
                        {"url": "https://assets.example.com/b.png"},
                        {"url": "https://assets.example.com/c.png"},
                    ]
                },
            )
        ],
        captured,
    )

    assets, reason = await generate_creative_assets(
        cfg=_cfg(),
        prompt="a coffee bag",
        aspect_ratio="1:1",
        num_images=1,
        provider="openai",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
    )

    assert reason is None
    assert len(assets) == 1


async def test_the_provider_layer_sends_the_count_it_was_given(respx_mock) -> None:
    """``_generate_openai_image`` is honest about ``n``; the *app* caps it to one.

    Keeping the two roles apart is what makes the cap testable: the provider
    function reports what it asked for, and the money-hating decision lives in
    one place instead of being smeared across both.
    """
    captured: list[dict] = []
    _route(respx_mock, [_ok()], captured)

    await _generate_openai_image(
        "A flat lay of coffee beans.",
        "1:1",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
        reference_image=None,
        count=4,
    )

    payload = json.loads(captured[0]["body"])
    assert payload["n"] == 4
    # Never sent: the provider default is `low`, and `high` costs extra per image.
    assert "input_fidelity" not in payload
    assert b"input_fidelity" not in captured[0]["body"]


async def test_the_multipart_edit_payload_also_omits_input_fidelity(respx_mock) -> None:
    captured: list[dict] = []
    _route(respx_mock, [_ok()], captured)

    await _generate_openai_image(
        "Keep the layout, change the product.",
        "4:5",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
        reference_image=("image/png", PNG),
        count=3,
    )

    body = captured[0]["body"]
    assert b'name="n"' in body and b"3" in body
    assert b'name="input_fidelity"' not in body
    assert b"1024x1536" in body


async def test_dall_e_3_never_receives_more_than_one_image(respx_mock) -> None:
    """``n`` > 1 is a hard 400 on DALL-E 3, so the clamp is per model."""
    captured: list[dict] = []
    _route(respx_mock, [_ok()], captured)

    await _generate_openai_image(
        "p", "1:1", api_key="k", model="dall-e-3", base_url=BASE, count=4
    )

    assert json.loads(captured[0]["body"])["n"] == 1


# ------------------------------------------------------------------ retries ---


async def test_a_503_is_retried_and_then_succeeds(respx_mock) -> None:
    """A busy endpoint is the normal case at peak; the retry costs nothing extra."""
    captured: list[dict] = []
    _route(
        respx_mock,
        [
            httpx.Response(503, json={"error": {"message": "model is currently experiencing high demand"}}),
            _ok(),
        ],
        captured,
    )
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    assets = await _generate_openai_image(
        "p", "1:1", api_key="k", model="gpt-image-1", base_url=BASE, sleep=pause
    )

    assert len(captured) == 2
    assert delays == [1.0]
    assert assets[0].url == "https://assets.example.com/out.png"


async def test_the_provider_retry_after_hint_wins_over_the_backoff(respx_mock) -> None:
    captured: list[dict] = []
    _route(
        respx_mock,
        [
            httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"code": "rate_limit_exceeded"}}),
            _ok(),
        ],
        captured,
    )
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    await _generate_openai_image(
        "p", "1:1", api_key="k", model="gpt-image-1", base_url=BASE, sleep=pause
    )

    assert delays == [2.0]


async def test_a_400_is_never_retried_and_keeps_the_provider_explanation(respx_mock) -> None:
    """Retrying a validation error would just burn the same 400 three times."""
    captured: list[dict] = []
    _route(
        respx_mock,
        [httpx.Response(400, json={"error": {"message": "Invalid size '1024x1280'", "param": "size"}})],
        captured,
    )

    with pytest.raises(ToolUnavailable) as raised:
        await _generate_openai_image(
            "p", "1:1", api_key="k", model="gpt-image-1", base_url=BASE
        )

    assert len(captured) == 1
    assert "Invalid size" in raised.value.message


async def test_exhausted_retries_still_explain_themselves(respx_mock) -> None:
    captured: list[dict] = []
    _route(
        respx_mock,
        [httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})],
        captured,
    )
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    with pytest.raises(ToolUnavailable) as raised:
        await _generate_openai_image(
            "p",
            "1:1",
            api_key="k",
            model="gpt-image-1",
            base_url=BASE,
            attempts=3,
            sleep=pause,
        )

    assert len(captured) == 3
    assert delays == [1.0, 2.0]
    assert "rate-limiting" in raised.value.message


async def test_a_timeout_is_not_retried_because_it_may_still_be_billing(respx_mock) -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"path": request.url.path})
        raise httpx.ReadTimeout("timed out", request=request)

    respx_mock.route(method="POST", host="image-provider.invalid").mock(side_effect=handler)

    with pytest.raises(ToolUnavailable) as raised:
        await _generate_openai_image("p", "1:1", api_key="k", model="gpt-image-1", base_url=BASE)

    assert len(captured) == 1
    assert "did not answer in time" in raised.value.message


async def test_a_refused_connection_is_retried(respx_mock) -> None:
    """The request demonstrably never ran, so another attempt is free."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return _ok()

    respx_mock.route(method="POST", host="image-provider.invalid").mock(side_effect=handler)
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    assets = await _generate_openai_image(
        "p", "1:1", api_key="k", model="gpt-image-1", base_url=BASE, sleep=pause
    )

    assert calls["n"] == 2
    assert delays == [1.0]
    assert len(assets) == 1


# ------------------------------------------------------- actionable messages ---


def test_error_messages_name_the_actual_problem() -> None:
    def reason(status: int, body: dict) -> str:
        return _provider_error_reason(httpx.Response(status, json=body))

    assert "API key" in reason(401, {"error": {"message": "Incorrect API key provided"}})
    assert "rate-limiting" in reason(429, {"error": {"code": "rate_limit_exceeded"}})
    assert "safety filter" in reason(
        400, {"error": {"code": "moderation_blocked", "message": "Your request was rejected as a result of our safety system."}}
    )
    assert "IMAGE_MODEL" in reason(404, {"error": {"message": "The model does not exist"}})
    assert "temporary failure" in reason(503, {"error": {"message": "overloaded"}})
    # A 400 that is not moderation still forwards the provider's own words.
    assert "Invalid size" in reason(400, {"error": {"message": "Invalid size '1024x1280'"}})
    # A non-JSON body must not raise while building a message.
    assert _provider_error_reason(httpx.Response(502, text="<html>bad gateway</html>"))


# --------------------------------------------------------------- timeouts ---


def test_the_render_ceiling_is_configurable_and_sane() -> None:
    assert _image_timeout_seconds(_cfg()) == 300.0
    assert _image_timeout_seconds(_cfg(image_timeout_seconds=45)) == 45.0
    # Below the floor the default wins: a 5s ceiling would fail every real render.
    assert _image_timeout_seconds(_cfg(image_timeout_seconds=5)) == _DEFAULT_IMAGE_TIMEOUT_SECONDS
    assert _image_timeout_seconds(_cfg(image_timeout_seconds="nonsense")) == _DEFAULT_IMAGE_TIMEOUT_SECONDS
    assert _image_timeout_seconds(SimpleNamespace()) == _DEFAULT_IMAGE_TIMEOUT_SECONDS


def test_the_client_timeout_separates_connect_from_read() -> None:
    """A dead host fails fast; only the render itself gets the long ceiling."""
    from app.core.tools.media import _DEFAULT_IMAGE_TIMEOUT_SECONDS as default

    timeout = httpx.Timeout(float(240), connect=15.0)
    assert timeout.read == 240
    assert timeout.connect == 15.0
    assert default == 300.0


# ------------------------------------------------- provider surprises = no 500 ---


async def test_a_non_json_response_becomes_a_reason_not_an_exception(respx_mock) -> None:
    respx_mock.route(method="POST", host="image-provider.invalid").mock(
        return_value=httpx.Response(200, text="<html>proxy</html>")
    )

    with pytest.raises(ToolUnavailable) as raised:
        await _generate_openai_image("p", "1:1", api_key="k", model="gpt-image-1", base_url=BASE)

    assert "not JSON" in raised.value.message


async def test_an_unexpected_provider_error_returns_a_reason_to_the_caller(monkeypatch) -> None:
    """Whatever the provider layer does, the request is answered, not 500'd."""
    _no_cloud(monkeypatch)

    async def exploding_call(prompt: str, ratio: str, **kwargs: Any) -> list[Any]:
        raise RuntimeError("provider library exploded")

    cfg = _cfg()
    assets, reason = await generate_creative_assets(
        cfg=cfg,
        prompt="p",
        aspect_ratio="1:1",
        provider="openai",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
        image_call=exploding_call,
    )

    assert assets == []
    assert reason and "RuntimeError" in reason


async def test_a_provider_error_message_reaches_the_caller_unwrapped(monkeypatch) -> None:
    _no_cloud(monkeypatch)

    async def refused_call(prompt: str, ratio: str, **kwargs: Any) -> list[Any]:
        raise ToolUnavailable(_provider_error_reason(
            httpx.Response(400, json={"error": {"code": "moderation_blocked"}})
        ))

    assets, reason = await generate_creative_assets(
        cfg=_cfg(),
        prompt="p",
        aspect_ratio="1:1",
        provider="openai",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
        image_call=refused_call,
    )

    assert assets == []
    assert reason and "safety filter" in reason


# -------------------------------------------------------------------- usage ---


def test_image_usage_is_flattened_for_the_log_line() -> None:
    summary = _usage_summary(
        {
            "data": [{"b64_json": "x"}],
            "usage": {
                "total_tokens": 4_200,
                "input_tokens": 30,
                "output_tokens": 4_170,
                "input_tokens_details": {"text_tokens": 12, "image_tokens": 18},
                "flags": True,
                "note": "n/a",
            },
        }
    )

    assert summary["total_tokens"] == 4_200
    assert summary["input_tokens_details_text_tokens"] == 12
    assert summary["input_tokens_details_image_tokens"] == 18
    # Booleans and strings are not counters and must not be logged as if they were.
    assert "flags" not in summary and "note" not in summary
    assert _usage_summary({"data": []}) == {}
    assert _usage_summary({"usage": "tokens"}) == {}
