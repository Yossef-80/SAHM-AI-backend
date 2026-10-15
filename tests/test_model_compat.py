"""The model-family compatibility rules, tested directly and offline.

Every branch here is a hard ``400`` at the provider when it is wrong -- there is
no degraded mode -- and the previous code had exactly two of these wrong:
``max_tokens`` was always sent (rejected by every reasoning model) and the image
``size`` was always DALL-E's (rejected by every GPT Image model). These tests are
what makes the compat layer verifiable instead of discovered in production.

Source of truth for the rules: the OpenAI API reference for images and the
"latest model" migration guide, as recorded in
``MODEL_PARAMETERS_AND_IMAGE_QUALITY.md``.
"""

from __future__ import annotations

import pytest

from app.core.llm.model_compat import (
    AVAILABLE_REASONING_EFFORTS,
    chat_completions_tools_unsupported,
    image_quality_for_model,
    image_size_for_model,
    is_reasoning_model,
    recommended_reasoning_effort,
    supports_temperature,
    token_limit_param,
)


# ------------------------------------------------------------- bare naming ----


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("  Gpt-5.6-Terra  ", "gpt-5.6-terra"),
        ("", ""),
        (None, ""),
    ],
)
def test_bare_model_name_strips_the_gateway_prefix_and_case(
    raw: str | None, expected: str
) -> None:
    from app.core.llm.model_compat import bare_model_name

    assert bare_model_name(raw) == expected


# ------------------------------------------------------------ token field ----


@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo"],
)
def test_classic_chat_models_keep_max_tokens(model: str) -> None:
    assert token_limit_param(model) == "max_tokens"


@pytest.mark.parametrize(
    "model",
    [
        "o1",
        "o1-mini",
        "o3-mini",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.6-terra",
        "gpt-6",
        "gpt-6-astra",
        "gpt-6.1-sol",
        "openai/o3-mini",
    ],
)
def test_reasoning_models_use_max_completion_tokens(model: str) -> None:
    """Reasoning models reject `max_tokens` outright."""
    assert token_limit_param(model) == "max_completion_tokens"
    assert is_reasoning_model(model) is True


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1", "some-local-llama"])
def test_non_reasoning_models_are_not_flagged(model: str) -> None:
    assert is_reasoning_model(model) is False
    assert supports_temperature(model) is True


@pytest.mark.parametrize("model", ["o3-mini", "gpt-5.6-terra", "gpt-6"])
def test_reasoning_models_refuse_a_custom_temperature(model: str) -> None:
    assert supports_temperature(model) is False


# ----------------------------------------------------- effort + tools on chat ----


def test_chat_completions_tools_are_unsupported_where_openai_says_so() -> None:
    """These two can only bind tools through the Responses API."""
    assert chat_completions_tools_unsupported("gpt-6-astra") is True
    assert chat_completions_tools_unsupported("openai/gpt-6.1-sol") is True
    assert chat_completions_tools_unsupported("gpt-5.6-terra") is False


def test_reasoning_models_default_to_no_reasoning_so_tools_keep_working() -> None:
    """From GPT-5.4 on, a chat request with tools needs ``reasoning_effort: none``."""
    assert recommended_reasoning_effort("gpt-5.6-terra", None) == "none"
    assert recommended_reasoning_effort("o3-mini", None) == "none"


def test_a_configured_effort_is_honoured() -> None:
    assert recommended_reasoning_effort("gpt-5.6-terra", "high") == "high"
    assert recommended_reasoning_effort("gpt-5.6-terra", "LOW") == "low"


def test_non_reasoning_models_get_no_effort_field() -> None:
    """Sending it to gpt-4o-mini is itself a 400."""
    assert recommended_reasoning_effort("gpt-4o-mini", None) is None
    assert recommended_reasoning_effort("gpt-4o-mini", "high") is None


def test_an_effort_outside_the_supported_set_is_ignored_not_forwarded() -> None:
    """``minimal``/``xhigh`` exist on some models only; forwarding blindly 400s."""
    assert "minimal" not in AVAILABLE_REASONING_EFFORTS
    assert recommended_reasoning_effort("gpt-5.6-terra", "minimal") == "none"
    assert recommended_reasoning_effort("gpt-5.6-terra", "xhigh") == "none"


# ------------------------------------------------------------------ images ----


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [("1:1", "1024x1024"), ("4:5", "1024x1536"), ("9:16", "1024x1536"), ("16:9", "1536x1024")],
)
def test_gpt_image_sizes(ratio: str, expected: str) -> None:
    assert image_size_for_model(ratio, "gpt-image-1") == expected
    assert image_size_for_model(ratio, "gpt-image-1.5") == expected


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [("1:1", "1024x1024"), ("4:5", "1024x1792"), ("16:9", "1792x1024")],
)
def test_dalle_3_sizes(ratio: str, expected: str) -> None:
    assert image_size_for_model(ratio, "dall-e-3") == expected


def test_an_unknown_ratio_falls_back_to_the_square_the_model_accepts() -> None:
    assert image_size_for_model("7:3", "gpt-image-1") == "1024x1024"
    assert image_size_for_model(None, "dall-e-3") == "1024x1024"


def test_gpt_image_quality_is_passed_through() -> None:
    for quality in ("low", "medium", "high", "auto"):
        assert image_quality_for_model(quality, "gpt-image-1") == quality


def test_dalle_3_quality_is_translated_to_its_own_two_names() -> None:
    assert image_quality_for_model("high", "dall-e-3") == "hd"
    assert image_quality_for_model("low", "dall-e-3") == "standard"
    assert image_quality_for_model("medium", "dall-e-3") == "standard"
    # `auto` is not a DALL-E word at all; "let the provider choose" is `standard`.
    assert image_quality_for_model("auto", "dall-e-3") == "standard"
    assert image_quality_for_model(None, "dall-e-3") == "standard"


def test_an_unknown_quality_falls_back_to_auto_instead_of_erroring() -> None:
    """Never a 400 from a settings typo: the API rejects those, this is the net."""
    assert image_quality_for_model("HGIH", "gpt-image-1") == "auto"
    assert image_quality_for_model(None, "gpt-image-1") == "auto"
