"""The answer language follows the language the user picked, not the language
they happened to type.

The bug this pins: a workspace whose UI was set to Arabic could get an English
answer the moment somebody pasted a link, and a workspace set to English got
Arabic reports because the tenant's first message was Arabic. The reply language
was being *inferred from the prompt* on every branch, independently, so one turn
could be half Arabic and half English.

The rule now: an explicit choice always wins, a remembered choice comes next,
detection only fills the gap, and the decision is made once per turn in
``app/core/i18n.resolve_reply_language`` and threaded everywhere.

Three layers are covered:

1. the resolver itself (pure functions, no I/O);
2. the orchestrator wiring -- the specialist agents actually receive a
   ``[RESPONSE_LANGUAGE]`` block, and the creative-prompt labels follow it;
3. the transport -- the UI's language reaches the orchestrator, survives into the
   next turn, and a mid-conversation switch takes effect on the very next answer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text

from app.ai.agents.orchestrator import brain
from app.core.i18n import (
    detect_language,
    normalise_language,
    reply_language_directive,
    resolve_reply_language,
)

ARABIC_PROMPT = "إيه أخبار حملة رمضان؟"
ENGLISH_PROMPT = "How is the Ramadan campaign doing?"


# --------------------------------------------------------------- 1. resolver --


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ar", "ar"),
        ("AR", "ar"),
        ("ar-EG", "ar"),
        ("ar_SA", "ar"),
        ("Arabic", "ar"),
        ("العربية", "ar"),
        ("en", "en"),
        ("en-US", "en"),
        ("English", "en"),
        ("auto", None),
        ("", None),
        (None, None),
        ("klingon", None),
    ],
)
def test_normalise_language_reads_every_spelling_a_client_may_send(value, expected) -> None:
    assert normalise_language(value) == expected


def test_detection_reports_none_for_text_with_no_letters() -> None:
    """A lone link or a number is not a language; detection must not guess."""
    assert detect_language("https://example.com/2026") == "en"  # Latin letters present
    assert detect_language("12345 - 678") is None
    assert detect_language("مرحبا hello") == "ar"


def test_english_ui_still_answers_in_english_when_the_user_writes_arabic() -> None:
    """This is the turn that used to flip the whole conversation."""
    assert resolve_reply_language(requested="en", prompt=ARABIC_PROMPT) == "en"


def test_arabic_ui_still_answers_in_arabic_when_the_user_pastes_english() -> None:
    assert resolve_reply_language(requested="ar", prompt="look at https://acme.com") == "ar"


def test_remembered_choice_outranks_the_message_script() -> None:
    """Turn 2 with no header must not silently switch language."""
    assert resolve_reply_language(conversation="ar", prompt=ENGLISH_PROMPT) == "ar"
    assert resolve_reply_language(conversation="en", prompt=ARABIC_PROMPT) == "en"


def test_detection_is_only_a_fallback() -> None:
    assert resolve_reply_language(prompt=ARABIC_PROMPT) == "ar"
    assert resolve_reply_language(prompt=ENGLISH_PROMPT) == "en"


def test_an_arabic_project_language_beats_detection() -> None:
    """Arabic is never a default column value, so it can only have been chosen."""
    assert resolve_reply_language(project="ar", prompt=ENGLISH_PROMPT) == "ar"


def test_an_english_project_language_does_not_beat_detection() -> None:
    """``businesses.language`` defaults to 'en'; treating that as a choice would
    give Arabic-speaking tenants English answers whenever the client is silent."""
    assert resolve_reply_language(project="en", prompt=ARABIC_PROMPT) == "ar"


def test_the_directive_names_both_the_language_and_the_exception() -> None:
    english = reply_language_directive("en")
    assert "English" in english and "even if the user writes in Arabic" in english
    arabic = reply_language_directive("ar")
    assert "Arabic" in arabic and "even if the user writes in English" in arabic


# ------------------------------------------------------- 2. brain wiring ------


class _FakeRouter:
    def __init__(self, decision: brain.RoutingDecision) -> None:
        self.decision = decision

    async def ainvoke(self, _messages: Any) -> brain.RoutingDecision:
        return self.decision


class _FakeChat:
    def __init__(self, decision: brain.RoutingDecision) -> None:
        self.router = _FakeRouter(decision)

    def with_structured_output(self, schema: type[Any]) -> Any:
        assert schema is brain.RoutingDecision
        return self.router


class _RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((payload, config))
        return {
            "messages": [
                SimpleNamespace(
                    name="submit_strategy_report",
                    content={"summary": "ok", "missing_inputs": []},
                    tool_calls=[],
                )
            ]
        }


def _system_texts(payload: dict[str, Any]) -> str:
    return "\n".join(
        str(message.content)
        for message in payload["messages"]
        if message.__class__.__name__ == "SystemMessage"
    )


async def _run_turn(monkeypatch, *, prompt: str, language: str, agent: str = "omar") -> _RecordingGraph:
    decision = brain.RoutingDecision(agent=agent, reason="test routing")
    chat = _FakeChat(decision)
    handle = SimpleNamespace(chat=chat, structured=chat.with_structured_output)
    graph = _RecordingGraph()

    monkeypatch.setattr(brain, "get_ai_settings", lambda: SimpleNamespace(llm_provider="mock"))
    monkeypatch.setattr(brain, "get_llm", lambda **_kwargs: handle)
    monkeypatch.setattr(brain, f"{agent}_graph", graph)
    monkeypatch.setattr(brain, "compress_history_messages", lambda _history: [])

    await brain.orchestrate(prompt, history=[], language=language)
    return graph


@pytest.mark.asyncio
async def test_specialist_gets_an_english_block_even_for_an_arabic_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = await _run_turn(monkeypatch, prompt=ARABIC_PROMPT, language="en")

    joined = _system_texts(graph.calls[0][0])
    assert "[RESPONSE_LANGUAGE]" in joined
    assert "The user chose English" in joined


@pytest.mark.asyncio
async def test_specialist_gets_an_arabic_block_even_for_an_english_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = await _run_turn(monkeypatch, prompt=ENGLISH_PROMPT, language="ar")

    joined = _system_texts(graph.calls[0][0])
    assert "The user chose Arabic" in joined


@pytest.mark.asyncio
async def test_karim_keeps_his_role_prompt_when_context_is_prepended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Karim/Nour tested ``messages[0]`` for their own prompt, so any
    prepended system message dropped their entire role prompt."""
    from app.ai.agents.karim.graph import KARIM_PROMPT, call_model

    state = {
        "messages": [
            brain.SystemMessage(content="[RESPONSE_LANGUAGE]\nspeak English"),
            brain.HumanMessage(content=ARABIC_PROMPT),
        ]
    }
    captured: dict[str, Any] = {}

    class _Llm:
        class chat:  # noqa: N801 - mirrors the provider facade
            @staticmethod
            def bind_tools(_tools: Any) -> Any:
                class _Bound:
                    @staticmethod
                    def invoke(messages: Any, _config: Any) -> Any:
                        captured["messages"] = messages
                        return SimpleNamespace(content="ok", tool_calls=[])

                return _Bound()

    monkeypatch.setattr(
        "app.ai.agents.karim.graph.get_llm", lambda **_kwargs: _Llm
    )
    call_model(state, {})

    assert captured["messages"][0].content == KARIM_PROMPT


def test_creative_prompt_labels_follow_the_ui_language_not_the_prompt() -> None:
    """An English UI with an Arabic brief must not produce Arabic labels."""
    output = SimpleNamespace(
        summary="",
        details=[],
        missing_inputs=["subject", "objective"],
        questions_for_user=[],
        preserved_elements=[],
        proposed_changes=[],
        ready_for_approval=False,
    )
    english_blocks, _ = brain._creative_prompt_blocks(
        output,
        context={"prompt_type": "image"},
        prompt="عايز برومبت صورة للبوتاجاز",
        stage="collecting_details",
        history=[],
        language="en",
    )
    arabic_blocks, _ = brain._creative_prompt_blocks(
        output,
        context={"prompt_type": "image"},
        prompt="عايز برومبت صورة للبوتاجاز",
        stage="collecting_details",
        history=[],
        language="ar",
    )

    english_text = " ".join(block.text for block in english_blocks)
    arabic_text = " ".join(block.text for block in arabic_blocks)
    assert "image" in english_text.lower()
    assert "صورة" not in english_text
    assert "صورة" in arabic_text


def test_report_formatter_renders_labels_in_the_reply_language() -> None:
    from app.ai.agents.orchestrator.presentation import format_agent_report

    output = {
        "summary": "Steady spend, flat CTR.",
        "missing_inputs": [],
        "evidence": [],
    }
    english = format_agent_report("karim", output, arabic=False)
    arabic = format_agent_report("karim", output, arabic=True)

    assert english != arabic
    assert any("\u0600" <= char <= "\u06ff" for char in arabic)
    assert not any("\u0600" <= char <= "\u06ff" for char in english)


# ------------------------------------------------------- 3. transport --------


async def _stored_context(db_session, conversation_id: str) -> dict:
    row = (
        await db_session.execute(
            text("SELECT context FROM conversations WHERE id = :i"),
            {"i": conversation_id},
        )
    ).scalar_one()
    return json.loads(row) if isinstance(row, str) else (row or {})


async def _post(api_client, monkeypatch, payload: dict) -> tuple[int, list[dict]]:
    seen: list[dict] = []

    async def fake_orchestrate(*_args, **kwargs):
        seen.append(
            {
                "language": kwargs.get("language"),
                "context": dict(kwargs.get("context") or {}),
            }
        )
        return brain.OrchestratorResult(blocks=[brain.TextBlock(text="ok")])

    monkeypatch.setattr(brain, "orchestrate", fake_orchestrate)
    response = await api_client.post("/api/v1/assistant/message", json=payload)
    return response, seen


@pytest.mark.asyncio
async def test_the_ui_language_reaches_the_orchestrator(api_client, monkeypatch) -> None:
    response, seen = await _post(
        api_client,
        monkeypatch,
        {"message": ARABIC_PROMPT, "language": "en"},
    )

    assert response.status_code == 200, response.text
    assert seen[0]["language"] == "en", seen
    assert seen[0]["context"]["language"] == "en"


@pytest.mark.asyncio
async def test_switching_the_ui_language_switches_the_next_answer(
    api_client, monkeypatch, db_session
) -> None:
    first, seen = await _post(
        api_client,
        monkeypatch,
        {"message": ARABIC_PROMPT, "language": "ar"},
    )
    assert first.status_code == 200, first.text
    conversation_id = first.json()["conversationId"]
    assert seen[0]["language"] == "ar"

    # The user flips the UI to English mid-conversation and keeps typing Arabic.
    second, seen = await _post(
        api_client,
        monkeypatch,
        {"conversationId": conversation_id, "message": ARABIC_PROMPT, "language": "en"},
    )
    assert second.status_code == 200, second.text
    assert seen[0]["language"] == "en", seen
    # ...and the new choice is what the conversation remembers.
    stored = await _stored_context(db_session, conversation_id)
    assert stored.get("language") == "en", stored


@pytest.mark.asyncio
async def test_the_last_explicit_choice_survives_a_client_that_stops_sending_it(
    api_client, monkeypatch, db_session
) -> None:
    first, _ = await _post(
        api_client,
        monkeypatch,
        {"message": ARABIC_PROMPT, "language": "ar"},
    )
    conversation_id = first.json()["conversationId"]

    second, seen = await _post(
        api_client,
        monkeypatch,
        {"conversationId": conversation_id, "message": ENGLISH_PROMPT},
    )
    assert second.status_code == 200, second.text
    assert seen[0]["language"] == "ar", seen


@pytest.mark.asyncio
async def test_a_detected_language_is_not_persisted_as_a_choice(
    api_client, monkeypatch, db_session
) -> None:
    """Detection is per-turn. Remembering it would trap an English-speaking user
    in Arabic after a single pasted message."""
    first, seen = await _post(api_client, monkeypatch, {"message": ARABIC_PROMPT})
    conversation_id = first.json()["conversationId"]
    assert seen[0]["language"] == "ar"  # detected for this turn only

    stored = await _stored_context(db_session, conversation_id)
    assert "language" not in stored, stored

    second, seen = await _post(
        api_client,
        monkeypatch,
        {"conversationId": conversation_id, "message": ENGLISH_PROMPT},
    )
    assert second.status_code == 200, second.text
    assert seen[0]["language"] == "en", seen
