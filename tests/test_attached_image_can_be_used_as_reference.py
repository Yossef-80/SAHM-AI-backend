"""An image the user attaches in chat must be usable as the design reference.

Live failure this pins ("مش عارف يشوف الصورة اللي بديهالو في الشات"): the user
attached the picture they wanted to match, wrote "اعمل تصميم زي الصورة دي", and the
assistant answered "أحتاج معرّف التحميل (upload_id)". Three separate breaks did it,
and each one is covered here:

1. the attachment payload never carried the stored file id, so the model had
   nothing to pass to ``analyze_reference_creative``;
2. the prompt-only interview hid that tool along with the generation tools, so the
   reference could not be read at all while the prompt was being written;
3. the id-validation allow-list rejected the id this codebase itself generates
   (``reference_<hex>.png``) because of the dot -- so even a correct id failed.
"""

from __future__ import annotations

import ast
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.agents.layla.graph import LAYLA_TOOLS  # noqa: E402
from app.ai.agents.orchestrator.brain import _attachment_messages, _tool_user_supplied  # noqa: E402
from app.ai.tools.core_bridge import _validate_tool_action  # noqa: E402
from app.schemas.tools import AnalyzeReferenceInput  # noqa: E402

BACK = Path(__file__).resolve().parents[1]
STORED_ID = "reference_4f1a9c0d2b7e4f8a9c1d2e3f4a5b6c7d.png"


class _Context:
    """Minimal ToolContext stand-in."""

    def __init__(self, supplied: dict[str, object]) -> None:
        self.user_supplied = supplied


# --------------------------------------------------------------- 1. the payload


def test_the_attachment_payload_carries_the_stored_file_id() -> None:
    source = (BACK / "app" / "core" / "chat_attachments.py").read_text(encoding="utf-8")

    assert '"upload_id": str(metadata.get("file_id") or "")' in source


# -------------------------------------------------- 2. the model is given the id


def _attachment(handle: object, **overrides: object) -> list[object]:
    attachment = {
        "id": "att_abc123",
        "name": "عمرة الموتور.png",
        "mime": "image/png",
        "data_url": "data:image/png;base64,AAAA",
        "upload_id": STORED_ID,
    }
    attachment.update(overrides)
    return _attachment_messages({"chat_attachments": [attachment]}, handle)


class _VisionHandle:
    provider = "gemini"
    model_name = "gemini-3.6-flash"


def test_a_vision_model_is_given_the_upload_id() -> None:
    messages = _attachment(_VisionHandle())

    text = str(getattr(messages[0], "content"))
    assert STORED_ID in text
    assert "analyze_reference_creative" in text
    assert "NEVER ask the user for an upload id" in text


def test_the_instruction_is_absent_when_there_is_no_id() -> None:
    """No id (an unreadable attachment) must not invite a tool call with a blank id."""
    messages = _attachment(_VisionHandle(), upload_id="")

    text = str(getattr(messages[0], "content"))
    assert "Their stored upload ids are" not in text
    assert "NEVER ask the user for an upload id" not in text


def test_a_text_only_model_still_receives_the_id_it_cannot_look_at() -> None:
    class _TextOnly:
        provider = "groq"
        model_name = "openai/gpt-oss-120b"

    text = str(getattr(_attachment(_TextOnly())[0], "content"))
    assert STORED_ID in text


# ------------------------------------------- 3. the tool stays usable in an interview


def test_laylas_toolbox_contains_the_analysis_and_generation_tools() -> None:
    names = {tool.name for tool in LAYLA_TOOLS}

    assert {"analyze_reference_creative", "generate_image"} <= names


def test_the_prompt_only_filter_only_blocks_the_analysis_tool_without_an_attachment() -> None:
    source = (BACK / "app" / "ai" / "agents" / "layla" / "graph.py").read_text(encoding="utf-8")

    assert "has_reference_attachment" in source
    assert 'blocked_in_prompt_only = {"generate_image", "generate_image_draft_tool"}' in source
    assert "if not reference_attached:" in source


def test_the_turn_flags_that_a_reference_was_attached() -> None:
    with_attachment = _tool_user_supplied({"chat_attachments": [{"id": "att_1"}]}, "صمميها")
    without = _tool_user_supplied({}, "صمميها")

    assert with_attachment["has_reference_attachment"] is True
    assert without["has_reference_attachment"] is False


# ------------------------------------------------- 4. the stored id passes the gate


def test_a_stored_reference_id_passes_validation() -> None:
    """`reference_<hex>.png` is what this codebase writes; it must be acceptable."""
    context = _Context({"latest_user_message": "اعمل تصميم زي الصورة دي", "creative_mode": "design"})

    reason = asyncio.run(
        _validate_tool_action(
            "analyze_reference_creative",
            AnalyzeReferenceInput(upload_id=STORED_ID),
            context,
        )
    )

    assert reason is None


def test_traversal_is_still_refused() -> None:
    context = _Context({"latest_user_message": "x", "creative_mode": "design"})

    for bad in ("../../etc/passwd", "..", "reference_1.png/../../x"):
        reason = asyncio.run(
            _validate_tool_action(
                "analyze_reference_creative", AnalyzeReferenceInput(upload_id=bad), context
            )
        )
        assert reason == "The reference upload id is invalid; no file was opened.", bad


def test_the_allow_list_shape_is_a_dot_permissive_but_traversal_free_pattern() -> None:
    source = (BACK / "app" / "ai" / "tools" / "core_bridge.py").read_text(encoding="utf-8")
    ast.parse(source)

    assert re.search(r'fullmatch\(r"\[A-Za-z0-9_\.\-\]\{1,128\}"', source)
    assert '".." in candidate' in source
