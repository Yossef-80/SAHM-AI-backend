"""Leaving the guided prompt interview when the user asks for the image itself.

Live failure this pins: the interview reached ``prompt_ready``, the user moved the
composer to «تصميم» and wrote «صمم الصورة», and the conversation answered with the
same canned "the approved prompt is still in the previous message" on every turn --
no image, forever. The interview is prompt-only, so it must end when the user asks
for pixels; it must NOT end while they are still writing the prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.agents.orchestrator.brain import (  # noqa: E402
    _exit_creative_prompt_workflow_for_generation,
)

_INTERVIEW_KEYS = {
    "workflow",
    "prompt_type",
    "creative_prompt_stage",
    "creative_prompt_user_facts",
    "creative_prompt_confirmed_facts",
    "creative_prompt_reference_shown_key",
    "creative_prompt_unresolved_fields",
}


def _interview(**overrides: object) -> dict[str, object]:
    context: dict[str, object] = {
        "workflow": "creative_prompt",
        "prompt_type": "image",
        "creative_prompt_stage": "prompt_ready",
        "creative_prompt_user_facts": ["MG Corner", "عمرة الموتور"],
        "creative_prompt_confirmed_facts": {"headline": {"value": "x", "status": "confirmed"}},
        "creative_prompt_reference_shown_key": "ref-1",
        "creative_prompt_unresolved_fields": [],
        "creative_mode": "design",
        "persona_id": "layla",
    }
    context.update(overrides)
    return context


def test_design_mode_request_for_the_image_ends_the_interview() -> None:
    result = _exit_creative_prompt_workflow_for_generation(_interview(), "صمم الصورة")

    assert _INTERVIEW_KEYS.isdisjoint(result)
    assert result["persona_id"] == "layla"
    assert result["_creative_prompt_workflow_exited"] is True


def test_the_exact_turn_from_the_log_ends_the_interview() -> None:
    """`context.creativeMode = "design"`, message «صمم الصورة», stage prompt_ready."""
    context = _interview(creative_mode="design", creative_prompt_stage="prompt_ready")

    assert _exit_creative_prompt_workflow_for_generation(context, "صمم الصورة") != context


def test_prompt_mode_keeps_the_interview_running() -> None:
    """Mid-interview «اعمل صورة» typed in prompt mode must still not produce pixels."""
    context = _interview(creative_mode="prompt", creative_prompt_stage="collecting_details")

    result = _exit_creative_prompt_workflow_for_generation(context, "اعمل صورة")

    assert result == context
    assert result["workflow"] == "creative_prompt"


def test_design_mode_before_the_prompt_exists_does_not_end_it() -> None:
    """No deliverable yet and no generation request: the interview continues."""
    context = _interview(creative_mode="design", creative_prompt_stage="awaiting_approval")

    assert _exit_creative_prompt_workflow_for_generation(context, "تمام") == context


def test_design_mode_after_a_ready_prompt_ends_it_even_with_a_short_message() -> None:
    context = _interview(creative_mode="design", creative_prompt_stage="prompt_ready")

    assert _INTERVIEW_KEYS.isdisjoint(_exit_creative_prompt_workflow_for_generation(context, "يلا"))


def test_a_chat_turn_inside_the_interview_changes_nothing() -> None:
    context = _interview(creative_mode="chat", creative_prompt_stage="prompt_ready")

    assert _exit_creative_prompt_workflow_for_generation(context, "ايه هما؟") == context


def test_a_conversation_without_the_interview_is_untouched() -> None:
    plain = {"creative_mode": "design", "persona_id": "layla"}

    assert _exit_creative_prompt_workflow_for_generation(plain, "صمميها") == plain


def test_the_marker_is_not_left_on_the_interview_context() -> None:
    """The exit marker must be one turn only, and never persisted as state."""
    result = _exit_creative_prompt_workflow_for_generation(_interview(), "نفذيها")

    assert "_creative_prompt_workflow_exited" in result  # popped by the caller
    assert "workflow" not in result
