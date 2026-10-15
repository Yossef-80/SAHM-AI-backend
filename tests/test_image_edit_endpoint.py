"""Phase 3.4 — prove which OpenAI endpoint the reference path actually hits.

The existing suite proves the reference *bytes* reach `_generate_openai_image`
and that REFERENCE-MATCH MODE is used, but it monkeypatches that function, so
the endpoint choice itself was never asserted. That choice is the whole
difference between an image that looks like the reference and one that does not:

    /images/edits        <- reference attached as multipart input
    /images/generations  <- text only; the reference is ignored entirely
"""

from __future__ import annotations

import json

import pytest

from app.core.tools.media import _generate_openai_image

pytestmark = pytest.mark.anyio

BASE = "https://image-provider.invalid/v1"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x22" * 16


def _mock(respx_mock, captured: list[dict]) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {
                "path": request.url.path,
                "content_type": request.headers.get("content-type", ""),
                "body": request.content,
            }
        )
        return httpx.Response(
            200, json={"data": [{"url": "https://assets.example.com/out.png"}]}
        )

    respx_mock.route(method="POST", host="image-provider.invalid").mock(
        side_effect=handler
    )


async def test_reference_image_selects_the_edits_endpoint_and_sends_bytes(
    respx_mock,
) -> None:
    captured: list[dict] = []
    _mock(respx_mock, captured)

    assets = await _generate_openai_image(
        "Keep the reference layout, change only the product.",
        "4:5",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
        reference_image=("image/png", PNG),
    )

    assert len(captured) == 1
    call = captured[0]
    # The whole point: edits, not generations.
    assert call["path"] == "/v1/images/edits"
    # Multipart, because the reference has to travel as a file part.
    assert call["content_type"].startswith("multipart/form-data")
    assert b"reference.png" in call["body"]
    assert PNG in call["body"]
    # Multipart encodes fields as parts, not as urlencoded key=value pairs.
    assert b'name="model"' in call["body"]
    assert b"gpt-image-1" in call["body"]
    # 4:5 maps to 1024x1536, not the square default.
    #
    # This used to assert 1024x1280, which is DALL-E's portrait size. A GPT Image
    # model rejects it, so the assertion was pinning a 400 in place. The value is
    # now chosen per model family by ``model_compat.image_size_for_model``.
    assert b'name="size"' in call["body"]
    assert b"1024x1536" in call["body"]
    # The quality is sent explicitly instead of leaving the provider default
    # (``high`` on GPT Image, the expensive one) to be bought silently.
    assert b'name="quality"' in call["body"]
    assert b"auto" in call["body"]
    # The prompt must travel with the edit request.
    assert b"name=\"prompt\"" in call["body"]
    assert assets and assets[0].url.startswith("https://")


async def test_no_reference_selects_the_generations_endpoint(respx_mock) -> None:
    captured: list[dict] = []
    _mock(respx_mock, captured)

    await _generate_openai_image(
        "A flat lay of coffee beans.",
        "1:1",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
        reference_image=None,
    )

    assert len(captured) == 1
    call = captured[0]
    assert call["path"] == "/v1/images/generations"
    assert call["content_type"].startswith("application/json")
    payload = json.loads(call["body"])
    assert payload["model"] == "gpt-image-1"
    assert payload["size"] == "1024x1024"
    assert payload["n"] == 1
    # No quality setting configured -> "auto", never the provider's silent high.
    assert payload["quality"] == "auto"


async def test_base64_responses_are_persisted_as_data_urls(respx_mock) -> None:
    """Edit models often return b64_json rather than a hosted URL."""
    import httpx

    respx_mock.route(method="POST", host="image-provider.invalid").mock(
        return_value=httpx.Response(
            200, json={"data": [{"b64_json": "aGVsbG8=", "mime_type": "image/png"}]}
        )
    )

    assets = await _generate_openai_image(
        "prompt",
        "1:1",
        api_key="sk-test",
        model="gpt-image-1",
        base_url=BASE,
        reference_image=("image/png", PNG),
    )

    assert len(assets) == 1
    assert assets[0].url == "data:image/png;base64,aGVsbG8="


async def test_every_supported_aspect_ratio_maps_to_a_real_size(respx_mock) -> None:
    """Each model family gets the sizes *it* accepts, per aspect ratio."""
    captured: list[dict] = []
    _mock(respx_mock, captured)

    gpt_image = {
        "1:1": "1024x1024",
        "4:5": "1024x1536",
        "9:16": "1024x1536",
        "16:9": "1536x1024",
    }
    for ratio in gpt_image:
        await _generate_openai_image(
            "p", ratio, api_key="k", model="gpt-image-1", base_url=BASE, reference_image=None
        )
    assert [json.loads(c["body"])["size"] for c in captured] == list(gpt_image.values())

    captured.clear()
    dall_e = {
        "1:1": "1024x1024",
        "4:5": "1024x1792",
        "9:16": "1024x1792",
        "16:9": "1792x1024",
    }
    for ratio in dall_e:
        await _generate_openai_image(
            "p", ratio, api_key="k", model="dall-e-3", base_url=BASE, reference_image=None
        )
    assert [json.loads(c["body"])["size"] for c in captured] == list(dall_e.values())
    # DALL-E 3 has no "auto": it must be translated, not forwarded verbatim.
    assert [json.loads(c["body"])["quality"] for c in captured] == ["standard"] * len(dall_e)


async def test_configured_quality_reaches_the_provider(respx_mock) -> None:
    """A saved quality is the one on the wire -- and is refused when unsupported."""
    captured: list[dict] = []
    _mock(respx_mock, captured)

    await _generate_openai_image(
        "p",
        "1:1",
        api_key="k",
        model="gpt-image-1",
        base_url=BASE,
        reference_image=None,
        quality="low",
    )
    assert json.loads(captured[0]["body"])["quality"] == "low"

    captured.clear()
    # DALL-E 3 spells the same idea standard/hd.
    await _generate_openai_image(
        "p",
        "1:1",
        api_key="k",
        model="dall-e-3",
        base_url=BASE,
        reference_image=None,
        quality="high",
    )
    assert json.loads(captured[0]["body"])["quality"] == "hd"
