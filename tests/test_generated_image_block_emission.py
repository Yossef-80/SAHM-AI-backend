"""A generated image reaches the UI as an image block, whatever the tool name.

Regression guard for the reported bug: the assistant answered "the image was
sent" and pasted the asset URL as text instead of showing the picture. Two
independent causes were removed -- the orchestrator only looked for one
hardcoded tool name / tool-call id, and the frontend had no ``image_preview``
block at all -- so the orchestrator half is pinned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.agents.orchestrator.brain import _generated_image_blocks  # noqa: E402
from app.schemas.blocks import AdPromptBlock, ImagePreviewBlock, TextBlock  # noqa: E402

ASSET_URL = "http://localhost:8000/api/v1/media/public-assets/asset_0123456789abcdef0123456789abcdef.png"


class _Message:
    """Minimal stand-in for a LangChain tool/AI message."""

    def __init__(self, content: object, name: str | None = None) -> None:
        self.content = content
        self.name = name


def _payload(**overrides: object) -> str:
    asset = {"url": ASSET_URL, "prompt": "بوستر لمشروب", "provider": "openai", "model": "gpt-image-2.5-flare"}
    asset.update(overrides)
    return json.dumps({"status": "ok", "source": "layla_image_generation", "asset": asset}, ensure_ascii=False)


def _image_blocks(blocks: list[object]) -> list[ImagePreviewBlock]:
    return [block for block in blocks if isinstance(block, ImagePreviewBlock)]


def test_a_tool_result_with_a_different_name_still_shows_the_image() -> None:
    blocks = _generated_image_blocks([_Message(_payload(), name="some_other_tool")])

    images = _image_blocks(blocks)
    assert len(images) == 1
    assert images[0].url == ASSET_URL
    assert images[0].prompt == "بوستر لمشروب"
    assert images[0].model == "gpt-image-2.5-flare"


def test_a_tool_result_without_a_name_still_shows_the_image() -> None:
    assert len(_image_blocks(_generated_image_blocks([_Message(_payload())]))) == 1


def test_earlier_prose_does_not_hide_the_tool_result() -> None:
    messages = [_Message("تمام، هجهزلك التصميم حالًا"), _Message(_payload(), name="generate_image_draft_tool")]

    assert len(_image_blocks(_generated_image_blocks(messages))) == 1


def test_content_parts_are_unwrapped() -> None:
    parts = [{"type": "text", "text": _payload()}]

    assert len(_image_blocks(_generated_image_blocks([_Message(parts)]))) == 1


def test_the_prompt_block_comes_before_the_image() -> None:
    blocks = _generated_image_blocks([_Message(_payload())])

    assert isinstance(blocks[0], AdPromptBlock)
    assert blocks[0].prompt == "بوستر لمشروب"
    assert isinstance(blocks[-1], ImagePreviewBlock)


def test_a_mock_provider_is_labelled() -> None:
    blocks = _generated_image_blocks([_Message(_payload(provider="mock"))])

    assert any(isinstance(block, TextBlock) and "mock" in block.text for block in blocks)


def test_everything_that_is_not_a_generated_image_yields_nothing() -> None:
    cases: list[list[_Message]] = [
        [_Message("أهلاً بيك، أقدر أساعدك في إيه؟")],
        [_Message(json.dumps({"status": "error", "reason": "no provider"}))],
        [_Message(json.dumps({"status": "ok", "asset": {"url": "", "prompt": "p"}}))],
        [_Message("the asset is ready")],  # mentions the word, is not JSON
        [_Message("{not json at all")],
        [],
    ]

    for messages in cases:
        assert _generated_image_blocks(messages) == [], messages


# ------------------------------- the workspace tool's own shape (the live miss) ----


def _core_payload(**asset_overrides: object) -> str:
    """The exact JSON the core bridge serialises for ``generate_image``.

    Built from the real output model so a schema change breaks this test instead of
    silently dropping the picture: the live log had ``assets: 1`` and
    ``block_types: ["text"]`` because only the singular ``asset`` key was read.
    """
    from app.schemas.tools import GenerateImageOutput, GeneratedAssetRecord

    asset = {
        "id": "cr_dd4384d7a21b4be08bc93d0872e786f8",
        "url": ASSET_URL,
        "prompt": "اعلان عمرة موتور",
        "aspect_ratio": "1:1",
        "provider": "openai",
        "model": "gpt-image-2.5-flare",
        "durable": True,
    }
    asset.update(asset_overrides)
    record = GeneratedAssetRecord.model_validate(asset)
    return json.dumps(
        GenerateImageOutput(assets=[record], status="ok", source="generate_image:openai").model_dump(
            mode="json"
        ),
        ensure_ascii=False,
    )


def test_the_workspace_tool_assets_list_is_shown_as_an_image() -> None:
    blocks = _generated_image_blocks([_Message(_core_payload(), name="generate_image")])

    images = _image_blocks(blocks)
    assert len(images) == 1
    assert images[0].url == ASSET_URL
    assert images[0].model == "gpt-image-2.5-flare"
    assert images[0].provider == "openai"


def test_the_prompt_block_comes_from_the_assets_list_too() -> None:
    blocks = _generated_image_blocks([_Message(_core_payload())])

    prompts = [block for block in blocks if isinstance(block, AdPromptBlock)]
    assert prompts and "عمرة موتور" in prompts[0].prompt


def test_an_assets_entry_without_a_url_is_not_a_picture() -> None:
    from app.schemas.tools import GenerateImageOutput

    raw = GenerateImageOutput(assets=[], status="ok", source="generate_image:openai").model_dump(mode="json")
    raw["assets"] = [{"id": "cr_1", "url": "", "prompt": "p"}]

    assert _generated_image_blocks([_Message(json.dumps(raw, ensure_ascii=False))]) == []


def test_a_single_response_can_carry_more_than_one_preview() -> None:
    from app.schemas.tools import GenerateImageOutput, GeneratedAssetRecord

    records = [
        GeneratedAssetRecord.model_validate(
            {"id": f"cr_{n}", "url": f"{ASSET_URL[: -6]}{n}.png", "prompt": "p", "provider": "openai"}
        )
        for n in (1, 2)
    ]
    raw = json.dumps(
        GenerateImageOutput(assets=records, status="ok", source="generate_image:openai").model_dump(
            mode="json"
        ),
        ensure_ascii=False,
    )

    images = _image_blocks(_generated_image_blocks([_Message(raw)]))

    assert len(images) == 2


def test_a_huge_assets_list_is_bounded() -> None:
    from app.schemas.tools import GenerateImageOutput, GeneratedAssetRecord

    records = [
        GeneratedAssetRecord.model_validate({"id": f"cr_{n}", "url": f"{ASSET_URL[: -6]}{n}.png", "prompt": "p"})
        for n in range(12)
    ]
    raw = json.dumps(
        GenerateImageOutput(assets=records, status="ok", source="generate_image:openai").model_dump(
            mode="json"
        ),
        ensure_ascii=False,
    )

    from app.ai.agents.orchestrator.brain import MAX_PREVIEWED_ASSETS

    assert len(_image_blocks(_generated_image_blocks([_Message(raw)]))) == MAX_PREVIEWED_ASSETS


def test_both_keys_in_one_result_still_show_everything() -> None:
    payload = json.loads(_core_payload())
    payload["asset"] = {"url": ASSET_URL.replace("asset_0", "asset_9"), "prompt": "p"}

    images = _image_blocks(_generated_image_blocks([_Message(json.dumps(payload, ensure_ascii=False))]))

    assert len(images) == 2
