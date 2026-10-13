"""Ad images must travel to Meta as `bytes`.

Meta's Ad Image reference documents exactly two Creating parameters for
POST /act_{id}/adimages — `bytes` and `copy_from`. There is no `url`; it exists
only on the returned object and is explicitly marked "do not use this URL in ad
creative creation". The previous code posted `{"url": creative.url}` in a JSON
body, which a live account would reject *after* the campaign and ad set already
existed.

These tests pin the replacement, including the two shapes a creative URL can take
and the refusal cases that keep this from becoming a way to reach an internal
address.
"""

from __future__ import annotations

import base64

import httpx
import pytest
import respx

from app.ai.marketing import meta_ads

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGPwdv8EAAJlAYUUlRngAAAAAElFTkSuQmCC"
)
PNG_DATA_URL = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"
TIMEOUT = 5.0


async def test_inline_data_url_yields_the_original_bytes() -> None:
    payload = await meta_ads._creative_image_payload(PNG_DATA_URL, timeout=TIMEOUT)
    assert payload is not None
    # Round-trips back to the exact bytes Meta will receive.
    assert base64.b64decode(payload) == PNG_BYTES


async def test_public_https_url_is_fetched_and_encoded(respx_mock) -> None:
    route = respx_mock.get("https://assets.example.com/final.png").mock(
        return_value=httpx.Response(200, content=PNG_BYTES)
    )

    payload = await meta_ads._creative_image_payload(
        "https://assets.example.com/final.png", timeout=TIMEOUT
    )

    assert route.called
    assert payload is not None
    assert base64.b64decode(payload) == PNG_BYTES


async def test_a_non_200_response_is_refused(respx_mock) -> None:
    respx_mock.get("https://assets.example.com/gone.png").mock(
        return_value=httpx.Response(404)
    )
    assert (
        await meta_ads._creative_image_payload(
            "https://assets.example.com/gone.png", timeout=TIMEOUT
        )
        is None
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://assets.example.com/final.png",  # not https
        "https://localhost/final.png",
        "https://127.0.0.1/final.png",
        "https://192.168.1.5/final.png",  # private range
        "https://assets.internal/final.png",  # reserved .internal suffix
        "https://user:secret@assets.example.com/final.png",  # credential leak
        "file:///etc/passwd",
        "",
        "   ",
    ],
)
async def test_unsafe_or_empty_urls_are_refused_without_a_request(
    url: str, respx_mock
) -> None:
    route = respx_mock.get(url__regex=r".*").mock(return_value=httpx.Response(200, content=PNG_BYTES))
    assert await meta_ads._creative_image_payload(url, timeout=TIMEOUT) is None
    assert not route.called


async def test_malformed_base64_is_refused() -> None:
    assert (
        await meta_ads._creative_image_payload(
            "data:image/png;base64,not!valid!base64!", timeout=TIMEOUT
        )
        is None
    )


async def test_an_oversized_image_is_refused_before_upload(respx_mock) -> None:
    respx_mock.get("https://assets.example.com/huge.png").mock(
        return_value=httpx.Response(200, content=b"\x00" * (meta_ads.MAX_AD_IMAGE_BYTES + 1))
    )
    assert (
        await meta_ads._creative_image_payload(
            "https://assets.example.com/huge.png", timeout=TIMEOUT
        )
        is None
    )


def test_generated_data_urls_are_now_valid_creatives() -> None:
    """The filter that gates which creatives reach Meta must accept them."""
    assert meta_ads.validate_creative_image_url(PNG_DATA_URL) is True
    assert meta_ads.validate_creative_image_url("https://assets.example.com/a.png") is True


def test_junk_urls_are_still_rejected_by_the_creative_filter() -> None:
    for bad in ["", "not a url", "http://assets.example.com/a.png", "https://localhost/a.png"]:
        assert meta_ads.validate_creative_image_url(bad) is False, bad
