"""«لما بضيفلو سياق في الشات برضو مش بيشوفها» -- the reference must reach the model.

The chain, as it was:

* the user picks a reference in chat, its id is persisted on the conversation
  (``context["reference_analysis_id"]``) and rehydrated by ``service.py`` into
  ``ai_context["reference_analysis_context"]``;
* ``generate_image`` takes ``reference_analysis_id`` as a **tool argument** --
  the model has to pass it;
* nothing ever told the model the id. ``reference_analysis_context`` was read only
  inside the prompt-interview path, so a plain "اعمل تصميم زي ده بالظبط" turn left
  the model with no id, no observations, and one honest-sounding option: ask the
  user to send the picture again.

Two independent fixes are pinned here:

1. ``_reference_context_message`` -- the active reference, its id and the exact
   instruction travel with every turn;
2. ``_context_reference_default`` -- a deterministic net for the turn where the
   model still drops the id, limited to the user's own deictic wording.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.agents.orchestrator.brain import _reference_context_message  # noqa: E402
from app.ai.tools.core_bridge import _context_reference_default  # noqa: E402
from app.schemas.tools import GenerateImageInput  # noqa: E402

REFERENCE_ID = "mem_a7ee806157fc4c5a931b4b42"

CONTEXT = {
    "reference_analysis_id": REFERENCE_ID,
    "reference_analysis_context": {
        "source_name": "reference_8f2c.png",
        "summary": "إعلان خدمة عمرة موتور بخلفية داكنة وسعر بارز",
        "style": {"palette": ["#111111", "#FFD400"], "composition": "يمين نص ويسار صورة"},
    },
}


# ------------------------------------------------------- the message the model sees ----


def test_the_reference_message_names_the_id_the_model_must_pass() -> None:
    message = _reference_context_message(CONTEXT)

    assert isinstance(message, SystemMessage)
    text = str(message.content)
    assert REFERENCE_ID in text
    assert "generate_image" in text


def test_the_reference_message_carries_the_persisted_observations() -> None:
    text = str(_reference_context_message(CONTEXT).content)

    assert "reference_8f2c.png" in text
    assert "عمرة موتور" in text


def test_the_reference_message_forbids_asking_for_the_picture_again() -> None:
    text = str(_reference_context_message(CONTEXT).content)

    assert "never ask" in text.lower()
    assert "id" in text.lower()


def test_an_analysis_without_an_id_asks_for_pixels_instead_of_guessing() -> None:
    message = _reference_context_message(
        {"reference_analysis_context": {"summary": "بوستر داكن", "style": {}}}
    )

    text = str(message.content)
    assert "بوستر داكن" in text
    assert "re-attach" in text
    assert "never invent" not in text


@pytest.mark.parametrize("context", [{}, {"reference_analysis_id": ""}, {"other": 1}])
def test_no_reference_means_no_extra_message(context: dict[str, Any]) -> None:
    assert _reference_context_message(context) is None


def test_the_turn_actually_appends_that_message_to_the_history() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "ai" / "agents" / "orchestrator" / "brain.py"
    ).read_text(encoding="utf-8")

    assert "reference_message = _reference_context_message(context)" in source
    # It has to land in the history every agent receives, not in a branch.
    assert source.index("chat_history.append(reference_message)") < source.index(
        "agent_history = chat_history + _attachment_messages("
    )


# ------------------------------------------------------- the deterministic net ----


def _context(latest: str, reference_id: str | None = REFERENCE_ID) -> Any:
    return SimpleNamespace(
        user_supplied={
            "reference_analysis_id": reference_id,
            "latest_user_message": latest,
        }
    )


@pytest.mark.parametrize(
    "latest",
    [
        "عايز اعمل تصميم زي ده بالظبط",
        "اعمل تصميم زي الصورة دي",
        "صمم على نفس الشكل",
        "خليها زي التصميم اللي فوق",
        "make one like this please",
        "same as the reference",
    ],
)
def test_deictic_wording_inherits_the_active_reference(latest: str) -> None:
    payload = GenerateImageInput(prompt="عمرة موتور")

    filled = _context_reference_default("generate_image", payload, _context(latest))

    assert filled.reference_analysis_id == REFERENCE_ID
    assert filled.prompt == "عمرة موتور"  # nothing else is touched


@pytest.mark.parametrize(
    "latest",
    ["اعمل تصميم من الأول", "فكرة جديدة خالص", "صمم صورة لحملة رمضان", "اعمل تصميم للبوتاجاز ده"],
)
def test_a_request_for_something_new_does_not_inherit(latest: str) -> None:
    """A wrong reference produces a wrong picture -- worse than a missing one."""
    payload = GenerateImageInput(prompt="فكرة جديدة")

    assert _context_reference_default("generate_image", payload, _context(latest)).reference_analysis_id is None


def test_an_explicit_choice_by_the_model_always_wins() -> None:
    payload = GenerateImageInput(prompt="x", reference_analysis_id="mem_other")

    kept = _context_reference_default("generate_image", payload, _context("اعمل تصميم زي ده"))

    assert kept.reference_analysis_id == "mem_other"


def test_no_reference_on_the_conversation_means_nothing_to_fill() -> None:
    payload = GenerateImageInput(prompt="x")

    assert (
        _context_reference_default("generate_image", payload, _context("زي ده", None)).reference_analysis_id
        is None
    )


def test_other_tools_are_never_touched() -> None:
    payload = GenerateImageInput(prompt="x")

    unchanged = _context_reference_default("analyze_reference_creative", payload, _context("زي ده"))

    assert unchanged.reference_analysis_id is None


def test_a_payload_without_the_field_survives() -> None:
    assert _context_reference_default("generate_image", None, _context("زي ده")) is None
    assert _context_reference_default("generate_image", object(), _context("زي ده")) is not None


# ------------------------------------------- through the tool wrapper itself ----


class _FakeTool:
    name = "generate_image"
    description = "Create one image draft (test double)."

    def __init__(self) -> None:
        from pydantic import BaseModel

        class _Out(BaseModel):
            status: str = "ok"

            @classmethod
            def unavailable(cls, **_kwargs: Any) -> "_Out":
                return cls(status="unavailable")

            def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
                return {"status": self.status}

        self.input_model = GenerateImageInput
        self.output_model = _Out
        self.seen: list[Any] = []

    async def handler(self, payload: Any, _context: Any) -> Any:
        self.seen.append(payload)
        return self.output_model()


def test_the_wrapper_hands_the_filled_id_to_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through ``_make_langchain_tool``: the handler sees the id."""
    from app.ai.tools import core_bridge

    monkeypatch.setattr(core_bridge, "_validate_tool_action", _allow)
    monkeypatch.setattr(core_bridge, "log_tool_call", _noop, raising=False)

    tool = _FakeTool()
    wrapped = core_bridge._make_langchain_tool(tool)
    config = {
        "configurable": {
            "tool_db": object(),
            "business_id": "biz_test",
            "workspace_id": "ws_test",
            "user_supplied": {
                "reference_analysis_id": REFERENCE_ID,
                "latest_user_message": "اعمل تصميم زي ده بالظبط على المساعدين",
                "creative_mode": "design",
            },
        }
    }

    asyncio.run(wrapped.ainvoke({"prompt": "عمرة موتور", "aspect_ratio": "1:1"}, config=config))

    assert tool.seen and tool.seen[0].reference_analysis_id == REFERENCE_ID


async def _allow(*_args: Any, **_kwargs: Any) -> None:
    return None


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def test_the_tool_result_is_still_serialised_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.ai.tools import core_bridge

    monkeypatch.setattr(core_bridge, "_validate_tool_action", _allow)

    tool = _FakeTool()
    wrapped = core_bridge._make_langchain_tool(tool)
    config = {
        "configurable": {
            "tool_db": object(),
            "business_id": "biz_test",
            "user_supplied": {
                "reference_analysis_id": REFERENCE_ID,
                "latest_user_message": "اعمل تصميم زي ده",
                "creative_mode": "design",
            },
        }
    }

    raw = asyncio.run(wrapped.ainvoke({"prompt": "x"}, config=config))

    assert json.loads(raw)["status"] == "ok"
