"""Offline regression tests for Layla's reference-image path."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.ai.enums import LLMProvider
from app.ai.marketing import reference_vision
from app.ai.marketing.reference_vision import (
    ReferenceVisualExtraction,
    VisionAnalysisUnavailable,
    _prepare_reference_image_for_vlm,
    analyze_reference_image,
)
from app.ai.settings import AISettings
from app.core.errors import ToolUnavailable
from app.core.tools.media import persist_inline_reference_image
from app.schemas.creatives import AnalyzeReferenceRequest


class CaptureRunnable:
    def __init__(self, schema: type[ReferenceVisualExtraction]) -> None:
        self.schema = schema
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> ReferenceVisualExtraction:
        self.messages = messages
        return self.schema(
            canvas_ratio="4:5",
            visible_text=["Hello"],
            elements=[],
        )


class CaptureVisionHandle:
    provider = LLMProvider.OLLAMA
    model_name = "vision-capable-test-model"

    def __init__(self) -> None:
        self.runnable: CaptureRunnable | None = None

    def structured(self, schema: type[ReferenceVisualExtraction]) -> CaptureRunnable:
        self.runnable = CaptureRunnable(schema)
        return self.runnable


class MockVisionHandle(CaptureVisionHandle):
    provider = LLMProvider.MOCK
    model_name = "mock-strong-1"


class TextOnlyVisionHandle(CaptureVisionHandle):
    model_name = "openai/gpt-oss-20b"


def _encoded_image(size: tuple[int, int], image_format: str = "JPEG") -> str:
    output = BytesIO()
    Image.new("RGB", size, color=(42, 88, 132)).save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_vlm_pixel_budget_is_configurable_and_does_not_replace_general_ollama_model() -> None:
    settings = AISettings(
        _env_file=None,
        AI_ADS_VLM_MAX_PIXELS="1200000",
        OLLAMA_MODEL="general-llm",
        VLM_OLLAMA_MODEL="qwen3-vl:8b",
    )

    assert settings.ads_vlm_max_pixels == 1_200_000
    assert settings.ollama_model == "general-llm"
    assert settings.vlm_ollama_model == "qwen3-vl:8b"


def test_reference_preprocessor_downscales_to_pixel_budget_and_keeps_ratio() -> None:
    original = _encoded_image((1080, 1350))

    prepared, mime = _prepare_reference_image_for_vlm(
        original,
        "image/jpeg",
        max_pixels=1_000_000,
    )

    with Image.open(BytesIO(base64.b64decode(prepared))) as resized:
        assert resized.width * resized.height <= 1_000_000
        assert resized.width < 1080
        assert resized.height < 1350
        assert abs(resized.width / resized.height - 1080 / 1350) < 0.002
    assert mime == "image/jpeg"
    assert original != prepared


def test_reference_preprocessor_passes_small_image_through_byte_for_byte() -> None:
    original = _encoded_image((400, 500), image_format="PNG")

    prepared, mime = _prepare_reference_image_for_vlm(
        original,
        "image/png",
        max_pixels=1_000_000,
    )

    assert prepared == original
    assert mime == "image/png"


async def test_analysis_sends_resized_copy_to_vlm_without_mutating_source(
    monkeypatch: Any,
) -> None:
    handle = CaptureVisionHandle()
    original = _encoded_image((1080, 1350))
    settings = AISettings(_env_file=None, ads_vlm_max_pixels=1_000_000)
    monkeypatch.setattr(reference_vision, "get_ai_settings", lambda: settings)
    monkeypatch.setattr(reference_vision, "get_vlm_llm", lambda **_kwargs: handle)

    await analyze_reference_image(
        image_b64=original,
        mime="image/jpeg",
        focus=None,
    )

    with Image.open(BytesIO(base64.b64decode(original))) as source:
        assert (source.width, source.height) == (1080, 1350)
    assert handle.runnable is not None
    data_url = handle.runnable.messages[1].content[1]["image_url"]
    sent_image = data_url.partition(",")[2]
    with Image.open(BytesIO(base64.b64decode(sent_image))) as resized:
        assert resized.width * resized.height <= settings.ads_vlm_max_pixels
        assert (resized.width, resized.height) != (1080, 1350)


async def test_layla_reference_analysis_attaches_actual_image_bytes(monkeypatch: Any) -> None:
    handle = CaptureVisionHandle()
    monkeypatch.setattr(reference_vision, "get_vlm_llm", lambda **_kwargs: handle)

    result = await analyze_reference_image(
        image_b64="aGVsbG8=",
        mime="image/png",
        focus="focus on the product",
    )

    assert result.canvas_ratio == "4:5"
    assert handle.runnable is not None
    system = handle.runnable.messages[0].content
    human = handle.runnable.messages[1]
    assert "focus on the product" in system
    assert human.content[1] == {
        "type": "image_url",
        "image_url": "data:image/png;base64,aGVsbG8=",
    }


async def test_mock_or_known_text_only_models_are_not_said_to_analyse_images(
    monkeypatch: Any,
) -> None:
    mock = MockVisionHandle()
    monkeypatch.setattr(reference_vision, "get_vlm_llm", lambda **_kwargs: mock)
    with pytest.raises(VisionAnalysisUnavailable, match="mock LLM does not inspect image pixels"):
        await analyze_reference_image(image_b64="aGVsbG8=", mime="image/png", focus=None)
    assert mock.runnable is None

    text_only = TextOnlyVisionHandle()
    monkeypatch.setattr(reference_vision, "get_vlm_llm", lambda **_kwargs: text_only)
    with pytest.raises(VisionAnalysisUnavailable, match="text-only"):
        await analyze_reference_image(image_b64="aGVsbG8=", mime="image/png", focus=None)
    assert text_only.runnable is None


async def test_api_accepts_inline_image_and_returns_mock_provider_as_unavailable(
    api_client: Any, tmp_path: Path, test_settings: Any, monkeypatch: Any
) -> None:
    from app.config import get_settings

    settings = test_settings.model_copy(
        update={
            "upload_dir": str(tmp_path),
            "llm_provider_strong": "mock",
            "llm_model_strong": "mock-strong-1",
        }
    )
    app = api_client._transport.app
    app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr(reference_vision, "get_vlm_llm", lambda **_kwargs: MockVisionHandle())
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

    response = await api_client.post(
        "/api/v1/creatives/analyze-reference", json={"imageData": data_url}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert "mock llm does not inspect image pixels" in response.json()["unavailable_reason"].lower()
    assert len(list(tmp_path.glob("reference_*.png"))) == 1


def test_api_request_accepts_browser_image_data_url_alias() -> None:
    data_url = "data:image/png;base64,aGVsbG8="
    request = AnalyzeReferenceRequest.model_validate({"imageData": data_url})
    assert request.image_data == data_url

    with pytest.raises(ValueError, match="exactly one"):
        AnalyzeReferenceRequest.model_validate({})


def test_browser_data_url_is_signature_checked_and_saved_privately(tmp_path: Path, test_settings: Any) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    settings = test_settings.model_copy(
        update={"upload_dir": str(tmp_path), "website_max_bytes": 1024}
    )
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

    file_id = persist_inline_reference_image(data_url, settings)

    assert file_id.startswith("reference_")
    assert file_id.endswith(".png")
    assert (tmp_path / file_id).read_bytes() == image_bytes


def test_browser_data_url_rejects_mismatched_mime_and_oversized_payload(tmp_path: Path, test_settings: Any) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    wrong_mime = "data:image/jpeg;base64," + base64.b64encode(png).decode("ascii")
    too_large = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    normal_settings = test_settings.model_copy(
        update={"upload_dir": str(tmp_path), "website_max_bytes": 1024}
    )
    small_limit_settings = test_settings.model_copy(
        update={"upload_dir": str(tmp_path), "website_max_bytes": 8}
    )

    with pytest.raises(ToolUnavailable, match="file type"):
        persist_inline_reference_image(wrong_mime, normal_settings)
    with pytest.raises(ToolUnavailable, match="size cap"):
        persist_inline_reference_image(too_large, small_limit_settings)
