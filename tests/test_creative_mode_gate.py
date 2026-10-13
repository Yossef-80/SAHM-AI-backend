"""The Layla mode picker must beat the keyword guess.

Before this, `core_bridge._validate_explicit_intent` decided whether Layla could
call `generate_image` purely by regex-matching her latest message
(`core_bridge.py:74`). A design request phrased as "عايز تصميم للبوتاجاز ده"
matched `_GENERATE_INTENT` nowhere, so the tool was refused and Layla only
talked — which is exactly the "confusing flow" complaint.

These tests pin the three modes and, just as importantly, pin that the old
regex path still works when no mode is sent, so no existing client regresses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.ai.tools.core_bridge import _validate_explicit_intent

# Deliberately free of every keyword in _GENERATE_INTENT, _PROMPT_ONLY and
# _REFERENCE_INTENT: this is how a real user asks for a design.
NEUTRAL_DESIGN_REQUEST = "عايز تصميم للبوتاجاز ده"
EXPLICIT_GENERATE = "اعمل صورة للبوتاجاز"
EXPLICIT_PROMPT_ONLY = "اكتب برومبت للبوتاجاز"


def _ctx(message: str, mode: str | None = None) -> Any:
    supplied: dict[str, Any] = {"latest_user_message": message}
    if mode is not None:
        supplied["creative_mode"] = mode
    return SimpleNamespace(user_supplied=supplied)


def test_design_mode_allows_generation_without_magic_words() -> None:
    assert _validate_explicit_intent("generate_image", _ctx(NEUTRAL_DESIGN_REQUEST, "design")) is None


def test_without_a_mode_the_keyword_guess_still_refuses() -> None:
    result = _validate_explicit_intent("generate_image", _ctx(NEUTRAL_DESIGN_REQUEST))
    assert result is not None
    assert "not explicitly requested" in result


def test_regex_path_is_unchanged_for_clients_that_send_no_mode() -> None:
    assert _validate_explicit_intent("generate_image", _ctx(EXPLICIT_GENERATE)) is None


def test_chat_mode_does_not_force_generation() -> None:
    assert _validate_explicit_intent("generate_image", _ctx(NEUTRAL_DESIGN_REQUEST, "chat")) is not None


def test_prompt_mode_refuses_pixels_even_when_the_words_ask_for_them() -> None:
    # The user asked for wording, not an asset; a matching phrase must not win.
    assert _validate_explicit_intent("generate_image", _ctx(EXPLICIT_GENERATE, "prompt")) is not None


def test_prompt_mode_still_refuses_a_neutral_message() -> None:
    assert _validate_explicit_intent("generate_image", _ctx(NEUTRAL_DESIGN_REQUEST, "prompt")) is not None


@pytest.mark.parametrize("mode", ["prompt", "design"])
def test_reference_analysis_is_allowed_in_the_creative_modes(mode: str) -> None:
    assert _validate_explicit_intent("analyze_reference_creative", _ctx(NEUTRAL_DESIGN_REQUEST, mode)) is None


def test_reference_analysis_still_needs_intent_without_a_mode() -> None:
    result = _validate_explicit_intent(
        "analyze_reference_creative", _ctx(NEUTRAL_DESIGN_REQUEST)
    )
    assert result is not None
    assert "reference" in result


def test_chat_mode_does_not_unlock_reference_analysis() -> None:
    assert (
        _validate_explicit_intent("analyze_reference_creative", _ctx(NEUTRAL_DESIGN_REQUEST, "chat"))
        is not None
    )


def test_mode_is_case_insensitive_and_trimmed() -> None:
    ctx = SimpleNamespace(user_supplied={"latest_user_message": NEUTRAL_DESIGN_REQUEST, "creative_mode": "  DESIGN "})
    assert _validate_explicit_intent("generate_image", ctx) is None


def test_missing_user_supplied_fails_closed() -> None:
    assert _validate_explicit_intent("generate_image", SimpleNamespace(user_supplied=None)) is not None


def test_unrelated_tools_are_never_gated() -> None:
    ctx = _ctx("أي رسالة", "prompt")
    assert _validate_explicit_intent("get_campaign_insights", ctx) is None
    assert _validate_explicit_intent("create_recommendation", ctx) is None
