"""A reference analysed from a chat attachment stays usable on later turns.

The picture a user attaches lives for exactly one turn -- that is deliberate, it
keeps the token bill flat. What must survive is the *analysis*: its memory id is
what `generate_image` takes as `reference_analysis_id`. Nothing persisted it, so
"خليها أحمر" on the next turn had no reference to apply and the assistant asked
for the picture again. These tests pin the lift-out (AI layer) and the persistence
(service layer) that close that loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.agents.orchestrator.brain import _reference_analysis_id_from_messages  # noqa: E402

BACK = Path(__file__).resolve().parents[1]
REFERENCE_ID = "mem_a7ee806157fc4c5a931b4b42"


class _Message:
    def __init__(self, content: object, name: str | None = None) -> None:
        self.content = content
        self.name = name


def _analysis(**overrides: object) -> str:
    payload = {
        "id": REFERENCE_ID,
        "source": "upload:reference_4f1a9c0d.png",
        "source_name": "عمرة الموتور.png",
        "status": "ok",
        "summary": "إعلان ورشة سيارات بألوان العلامة",
        "style": {"palette": ["#8B0000", "#1E293B"], "composition": "hero product"},
        "observed_offers": ["عمرة شاملة الضمان"],
        "observed_cta": ["احجز الآن"],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_the_analysis_id_is_lifted_out_of_the_tool_result() -> None:
    assert _reference_analysis_id_from_messages([_Message(_analysis())]) == REFERENCE_ID


def test_a_renamed_tool_still_counts() -> None:
    """The payload shape is the contract, not the function name."""
    assert (
        _reference_analysis_id_from_messages(
            [_Message(_analysis(), name="some_future_analysis_tool")]
        )
        == REFERENCE_ID
    )


def test_the_newest_analysis_wins() -> None:
    older = _analysis(id="mem_older")
    messages = [_Message(older), _Message("تمام"), _Message(_analysis(id="mem_newest"))]

    assert _reference_analysis_id_from_messages(messages) == "mem_newest"


def test_an_image_asset_result_is_not_a_reference_analysis() -> None:
    rendered = json.dumps({"status": "ok", "asset": {"url": "http://x/a.png", "prompt": "p"}})

    assert _reference_analysis_id_from_messages([_Message(rendered)]) is None


def test_a_style_key_without_an_id_is_ignored() -> None:
    assert _reference_analysis_id_from_messages([_Message(_analysis(id=""))]) is None


def test_plain_prose_and_broken_json_are_ignored() -> None:
    for content in ["ما استخرجته من الصورة: ألوان العلامة", "{ broken", "", None]:
        assert _reference_analysis_id_from_messages([_Message(content)]) is None


def test_content_parts_are_unwrapped() -> None:
    parts = [{"type": "text", "text": _analysis()}]

    assert _reference_analysis_id_from_messages([_Message(parts)]) == REFERENCE_ID


def test_the_service_persists_the_id_on_the_conversation() -> None:
    source = (BACK / "app" / "core" / "orchestrator" / "service.py").read_text(encoding="utf-8")

    assert 'getattr(final_gui, "reference_analysis_id", None)' in source
    assert 'context["reference_analysis_id"]' in source
    # Persisted as soon as the turn comes back, so the prompt-workflow block right
    # below it sees the reference and writes the finished context to the row once.
    assert source.index('context["reference_analysis_id"]') < source.index(
        "next_prompt_stage = getattr(final_gui"
    )
