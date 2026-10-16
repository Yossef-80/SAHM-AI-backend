"""«هل كدة هيولد الصورة مرتين؟» -- one request must never buy two pictures.

Three different ways a single turn could generate a second image, all closed here:

1. the model calls the image tool again after seeing the first result
   (``graph.py`` drops both image tools once one has run);
2. the model asks for two in one step, or a fresh call slips through anyway
   (``core_bridge`` refuses any second call in the run *before* the provider is
   reached, so nothing is billed);
3. a provider refusal triggers the whole-graph retry after a picture already
   exists (``_invoke_graph`` no longer retries once a side-effecting tool ran).

Read-only tools stay retryable: that recovery path is the reason the retry exists.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.agents.orchestrator.brain import (  # noqa: E402
    _graph_tool_side_effects,
    _tool_is_side_effecting,
)
from app.ai.loggers.callbacks import AIToolAuditCallback  # noqa: E402
from app.ai.tools import core_bridge  # noqa: E402
from app.ai.tools.core_bridge import _image_already_produced  # noqa: E402


# ------------------------------------------------------- the audit answers safely ----


def _audit_with(tool_names: list[str]) -> AIToolAuditCallback:
    audit = AIToolAuditCallback(agent="layla")
    for name in tool_names:
        run_id = uuid4()
        audit.on_tool_start({"name": name}, "", run_id=run_id)
        audit.on_tool_end(json.dumps({"status": "ok"}), run_id=run_id)
    return audit


def test_a_completed_image_is_visible_to_the_guard() -> None:
    audit = _audit_with(["generate_image"])

    assert _image_already_produced({"callbacks": [audit]}) is True


def test_a_call_that_has_only_just_started_is_not_counted() -> None:
    """The call asking the question has already started; it must not refuse itself."""
    audit = AIToolAuditCallback(agent="layla")
    audit.on_tool_start({"name": "generate_image"}, "", run_id=uuid4())

    assert _image_already_produced({"callbacks": [audit]}) is False


def test_reading_the_guard_does_not_consume_the_telemetry() -> None:
    audit = _audit_with(["generate_image"])

    _image_already_produced({"callbacks": [audit]})

    assert [record["tool"] for record in audit.drain()] == ["generate_image"]


def test_a_run_without_an_audit_is_never_refused() -> None:
    assert _image_already_produced({}) is False
    assert _image_already_produced(None) is False
    assert _image_already_produced({"callbacks": object()}) is False


def test_only_images_count_for_the_image_guard() -> None:
    audit = _audit_with(["get_campaign_insights", "web_search"])

    assert _image_already_produced({"callbacks": [audit]}) is False


# ------------------------------------------------------------ the retry gate ----


def test_generating_an_image_blocks_the_whole_graph_retry() -> None:
    assert _tool_is_side_effecting("generate_image") is True
    assert _tool_is_side_effecting("generate_image_draft_tool") is True
    assert _tool_is_side_effecting("create_recommendation") is True


@pytest.mark.parametrize("name", ["get_campaign_insights", "web_search", "get_breakdowns"])
def test_read_only_tools_stay_retryable(name: str) -> None:
    assert _tool_is_side_effecting(name) is False


def test_the_registry_is_asked_about_tools_it_owns() -> None:
    """The side effect is declared with the tool, so a new tool needs no edit here."""
    from app.core.tools.registry import register_all

    register_all()  # the app does this at startup; the tool's own metadata decides

    assert _tool_is_side_effecting("analyze_reference_creative") is True  # draft_write


def test_an_unknown_tool_does_not_disable_recovery() -> None:
    assert _tool_is_side_effecting("a_tool_that_does_not_exist") is False
    assert _tool_is_side_effecting("") is False


def test_the_run_reports_exactly_the_side_effecting_tools_it_ran() -> None:
    audit = _audit_with(["get_campaign_insights", "generate_image", "web_search"])

    assert _graph_tool_side_effects(audit) == ["generate_image"]


def test_a_clean_run_reports_nothing() -> None:
    assert _graph_tool_side_effects(_audit_with(["web_search"])) == []


def test_the_retry_gate_raises_instead_of_replaying_the_turn() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "ai" / "agents" / "orchestrator" / "brain.py"
    ).read_text(encoding="utf-8")

    assert "agent_graph_retry_skipped_after_side_effects" in source
    # It must run *before* the retry actually happens.
    assert source.index("_graph_tool_side_effects(audit)") < source.index(
        'print(f"[ORCHESTRATOR] Retrying {agent_name} on'
    )


# -------------------------------------------------- the second call, refused ----


class _ImageTool:
    name = "generate_image"
    description = "Create one image draft (test double)."
    side_effect = None

    def __init__(self) -> None:
        from pydantic import BaseModel

        class _Out(BaseModel):
            status: str = "ok"

            @classmethod
            def unavailable(cls, **_kwargs: Any) -> "_Out":
                return cls(status="unavailable")

            def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
                return {"status": self.status}

        from app.schemas.tools import GenerateImageInput

        self.input_model = GenerateImageInput
        self.output_model = _Out
        self.calls = 0

    async def handler(self, payload: Any, _context: Any) -> Any:
        self.calls += 1
        return self.output_model()


async def _allow(*_args: Any, **_kwargs: Any) -> None:
    return None


def _config(audit: AIToolAuditCallback | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "configurable": {
            "tool_db": object(),
            "business_id": "biz_test",
            "user_supplied": {"latest_user_message": "صمم الصورة", "creative_mode": "design"},
        }
    }
    if audit is not None:
        config["callbacks"] = [audit]
    return config


def test_a_second_image_call_in_one_run_never_reaches_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_bridge, "_validate_tool_action", _allow)

    tool = _ImageTool()
    wrapped = core_bridge._make_langchain_tool(tool)
    audit = _audit_with(["generate_image"])  # the first image of this run

    raw = asyncio.run(wrapped.ainvoke({"prompt": "عمرة موتور"}, config=_config(audit)))

    assert json.loads(raw)["status"] == "unavailable"
    assert tool.calls == 0, "the provider must not be reached for a duplicate"


def test_the_first_image_call_still_goes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core_bridge, "_validate_tool_action", _allow)

    tool = _ImageTool()
    wrapped = core_bridge._make_langchain_tool(tool)

    raw = asyncio.run(wrapped.ainvoke({"prompt": "عمرة موتور"}, config=_config(_audit_with([]))))

    assert json.loads(raw)["status"] == "ok"
    assert tool.calls == 1


def test_a_non_image_tool_is_never_treated_as_a_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core_bridge, "_validate_tool_action", _allow)

    tool = _ImageTool()
    tool.name = "get_campaign_insights"
    wrapped = core_bridge._make_langchain_tool(tool)

    raw = asyncio.run(wrapped.ainvoke({"prompt": "x"}, config=_config(_audit_with(["generate_image"]))))

    assert tool.calls == 1


# ------------------------------------------------------ the graph hides the tool ----


def test_the_graph_drops_both_image_tools_after_one_picture() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "ai" / "agents" / "layla" / "graph.py"
    ).read_text(encoding="utf-8")

    assert '_IMAGE_TOOL_NAMES = frozenset({"generate_image", "generate_image_draft_tool"})' in source
    assert 'getattr(m, "name", None) in _IMAGE_TOOL_NAMES' in source
    # The second guard covers both names...
    assert "already_attempted_image and tool.name in _IMAGE_TOOL_NAMES" in source
    # ...without touching the original intent switch for the draft tool, which the
    # explicit UI mode still needs.
    assert 'tool.name == "generate_image_draft_tool" and not should_allow_image' in source


def test_a_second_picture_would_need_a_second_request() -> None:
    """Sanity check on the product rule the guards implement."""
    assert SimpleNamespace  # a placeholder keeps the intent readable above
    assert core_bridge._IMAGE_TOOLS == {"generate_image", "generate_image_draft_tool"}


# ------------------------------------------- quota, but only on a clean run ----


def test_an_exhausted_quota_is_recognised() -> None:
    from app.ai.agents.orchestrator.brain import _provider_quota_exhausted

    class _Quota(Exception):
        status_code = 429

    assert _provider_quota_exhausted(_Quota("too many requests")) is True
    assert (
        _provider_quota_exhausted(
            RuntimeError(
                "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
                "generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash"
            )
        )
        is True
    )


@pytest.mark.parametrize("message", ["500 internal error", "connection reset", "invalid argument"])
def test_other_failures_are_not_quota(message: str) -> None:
    from app.ai.agents.orchestrator.brain import _provider_quota_exhausted

    assert _provider_quota_exhausted(RuntimeError(message)) is False


def test_the_quota_retry_is_still_gated_by_side_effects() -> None:
    """A spent quota may be retried on another model -- but never after a picture."""
    source = (
        Path(__file__).resolve().parents[1] / "app" / "ai" / "agents" / "orchestrator" / "brain.py"
    ).read_text(encoding="utf-8")

    gate = source.index("_provider_quota_exhausted(exc)")
    assert gate < source.index("_graph_tool_side_effects(audit)")
    assert source.index("_graph_tool_side_effects(audit)") < source.index("Retrying {agent_name} on")


# ------------------------------ the picture survives a failure after it ran ----


def test_the_audit_rescues_the_picture_a_failed_turn_generated() -> None:
    from app.ai.agents.orchestrator.brain import _generated_image_blocks_from_audit
    from app.schemas.blocks import ImagePreviewBlock
    from app.schemas.tools import GenerateImageOutput, GeneratedAssetRecord

    audit = AIToolAuditCallback(agent="layla")
    run_id = uuid4()
    audit.on_tool_start({"name": "generate_image"}, "", run_id=run_id)
    audit.on_tool_end(
        json.dumps(
            GenerateImageOutput(
                assets=[
                    GeneratedAssetRecord.model_validate(
                        {
                            "id": "cr_dd4384d7a21b4be08bc93d0872e786f8",
                            "url": "http://localhost:8000/api/v1/media/public-assets/asset_abc.png",
                            "prompt": "اعلان",
                            "provider": "openai",
                            "model": "gpt-image-2.5-flare",
                        }
                    )
                ],
                status="ok",
                source="generate_image:openai",
            ).model_dump(mode="json"),
            ensure_ascii=False,
        ),
        run_id=run_id,
    )

    blocks = _generated_image_blocks_from_audit(audit)

    assert [type(block).__name__ for block in blocks] == ["AdPromptBlock", "ImagePreviewBlock"]
    assert isinstance(blocks[1], ImagePreviewBlock)
    # The telemetry is untouched: drain() must still see the record.
    assert len(audit.drain()) == 1


def test_a_run_with_no_image_rescues_nothing() -> None:
    from app.ai.agents.orchestrator.brain import _generated_image_blocks_from_audit

    assert _generated_image_blocks_from_audit(None) == []
    assert _generated_image_blocks_from_audit(_audit_with(["web_search"])) == []


def test_the_failed_run_hands_the_blocks_back_to_the_turn() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "ai" / "agents" / "orchestrator" / "brain.py"
    ).read_text(encoding="utf-8")

    assert "rescue_blocks=image_blocks" in source
    assert source.count("rescue_blocks=image_blocks") >= 2  # the plain turn and the handoff
    assert "agent_graph_failed_after_media" in source
    assert "rescued_now = bool(image_blocks)" in source


def test_the_failure_message_does_not_claim_nothing_was_saved_when_it_was() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "ai" / "agents" / "orchestrator" / "brain.py"
    ).read_text(encoding="utf-8")

    assert "التصميم نفسه اتعمل واتحفظ" in source
    # The sentence is built only when a picture was rescued.
    assert source.index("rescued_now = bool(image_blocks)") < source.index(
        "التصميم نفسه اتعمل واتحفظ"
    )
