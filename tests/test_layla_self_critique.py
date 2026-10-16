"""The one review pass Layla runs before handing over a creative deliverable.

A critic that cannot fail safely is worse than no critic: if the review pass
crashes, times out, or returns a malformed rewrite, the user must still receive
the draft they asked for. These tests cover both halves -- the repair when the
review works, and the unchanged draft when it does not.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.ai.agents.layla.critique import (
    CritiqueVerdict,
    apply_verdict,
    build_critique_request,
    critique_deliverable,
    critique_enabled,
    critique_schema,
)
from app.ai.agents.layla.schema import CreativeBrief, CreativeVariant
from app.ai.settings import AISettings

DRAFT = CreativeBrief(
    deliverable_type="ad_copy",
    summary="Coffee subscription for offices.",
    variants=[
        CreativeVariant(
            angle="Convenience",
            headline="Free delivery everywhere, guaranteed cheapest",
            primary_text="Free delivery everywhere and the lowest price in Egypt.",
        )
    ],
)

OK_VERDICT = CritiqueVerdict(verdict="ok")

FIXED = CreativeBrief(
    deliverable_type="ad_copy",
    summary="Coffee subscription for offices.",
    variants=[
        CreativeVariant(
            angle="Convenience",
            headline="Beans at the office before the first meeting",
            primary_text="Twelve percent off for offices of ten to fifty, delivered in Cairo.",
        )
    ],
)


class _Runnable:
    def __init__(self, result: Any, *, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.messages = messages
        if self.raises is not None:
            raise self.raises
        return self.result


class _Handle:
    def __init__(self, result: Any = None, *, raises: Exception | None = None) -> None:
        self.runnable = _Runnable(result, raises=raises)

    def structured(self, schema: type[Any]) -> _Runnable:
        return self.runnable


# --------------------------------------------------------------------------- #
# The repair                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_revise_verdict_replaces_the_draft() -> None:
    # The verdict schema is generated per deliverable, so the repair inside it is
    # validated against exactly the draft's contract.
    verdict_schema = critique_schema(CreativeBrief)
    verdict = verdict_schema(
        verdict="revise",
        defects=["UNSUPPORTED"],
        notes="Removed a delivery promise and a price claim the brief never supplied.",
        # The repair travels as JSON text: a nested model reference becomes a
        # `$defs` entry some providers drop from the tool schema.
        revised_json=FIXED.model_dump_json(),
    )

    handle = _Handle(verdict)
    result = await critique_deliverable(
        handle,
        brief="Office coffee subscription, 12% off, Cairo only",
        draft=DRAFT.model_dump(),
        schema=critique_schema(CreativeBrief),
    )
    assert result is not None and result.verdict == "revise"

    reviewed, notes = apply_verdict(DRAFT, result, schema=CreativeBrief)
    assert reviewed.variants[0].headline == "Beans at the office before the first meeting"
    assert notes and "Removed a delivery promise" in notes[0]


@pytest.mark.asyncio
async def test_an_ok_verdict_leaves_the_draft_untouched() -> None:
    handle = _Handle(CritiqueVerdict(verdict="ok"))
    result = await critique_deliverable(
        handle, brief="brief", draft=DRAFT.model_dump(), schema=critique_schema(CreativeBrief)
    )
    reviewed, notes = apply_verdict(DRAFT, result, schema=CreativeBrief)
    assert reviewed is DRAFT
    assert notes == []


def test_a_malformed_repair_is_discarded_not_shipped() -> None:
    """The failure that matters: a repair that does not validate."""

    verdict = CritiqueVerdict(verdict="revise", revised={"variants": "not a list"})
    reviewed, notes = apply_verdict(DRAFT, verdict, schema=CreativeBrief)
    assert reviewed is DRAFT
    assert notes == []


def test_a_repair_that_is_not_json_is_discarded() -> None:
    verdict = CritiqueVerdict(verdict="revise", revised_json="Sure! I rewrote it for you.")
    reviewed, notes = apply_verdict(DRAFT, verdict, schema=CreativeBrief)
    assert reviewed is DRAFT and notes == []


def test_a_markdown_fenced_repair_is_still_parsed() -> None:
    """Models fence JSON even when the schema says string."""

    fenced = "```json\n" + FIXED.model_dump_json() + "\n```"
    verdict = CritiqueVerdict(verdict="revise", revised_json=fenced)
    reviewed, notes = apply_verdict(DRAFT, verdict, schema=CreativeBrief)
    assert reviewed.variants[0].headline.startswith("Beans")


def test_a_single_line_fenced_repair_is_still_parsed() -> None:
    """No newline to split on -- "```json{...}```"."""

    verdict = CritiqueVerdict(
        verdict="revise", revised_json="```json" + FIXED.model_dump_json() + "```"
    )
    reviewed, _ = apply_verdict(DRAFT, verdict, schema=CreativeBrief)
    assert reviewed.variants[0].headline.startswith("Beans")


def test_a_revise_verdict_with_no_repair_changes_nothing() -> None:
    """Saying "revise" without supplying the fix is not a licence to invent one."""

    reviewed, notes = apply_verdict(DRAFT, CritiqueVerdict(verdict="revise"), schema=CreativeBrief)
    assert reviewed is DRAFT and notes == []


def test_a_repair_with_no_notes_still_reports_what_it_fixed() -> None:
    verdict = CritiqueVerdict(
        verdict="revise", defects=["DUPLICATE", "LANGUAGE"], revised=FIXED.model_dump()
    )  # internal shape: an already-parsed repair
    reviewed, notes = apply_verdict(DRAFT, verdict, schema=CreativeBrief)
    assert reviewed.variants[0].headline.startswith("Beans")
    assert notes and "DUPLICATE" in notes[0]


# --------------------------------------------------------------------------- #
# Failing safely                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_provider_error_during_review_keeps_the_draft() -> None:
    handle = _Handle(raises=RuntimeError("provider exploded"))
    assert (
        await critique_deliverable(
            handle, brief="brief", draft=DRAFT.model_dump(), schema=critique_schema(CreativeBrief)
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_verdict_that_does_not_validate_is_dropped() -> None:
    class _Wrong:
        pass

    handle = _Handle(_Wrong())
    assert (
        await critique_deliverable(
            handle, brief="brief", draft=DRAFT.model_dump(), schema=critique_schema(CreativeBrief)
        )
        is None
    )


def test_applying_none_is_a_no_op() -> None:
    reviewed, notes = apply_verdict(DRAFT, None, schema=CreativeBrief)
    assert reviewed is DRAFT and notes == []


def test_the_review_can_be_switched_off() -> None:
    assert critique_enabled(AISettings()) is True
    assert critique_enabled(AISettings(layla_self_critique=False)) is False


# --------------------------------------------------------------------------- #
# The prompt                                                                  #
# --------------------------------------------------------------------------- #


def test_the_review_prompt_carries_the_brief_the_facts_and_the_draft() -> None:
    request = build_critique_request(
        brief="Office coffee subscription, 12% off",
        draft=DRAFT.model_dump(),
        brand_context_text="Key offers: 12% off for teams",
        language="en",
        extra_rules="Never invent an offer.",
    )
    assert "12% off" in request
    assert "BRAND FACTS THAT MAY APPEAR" in request
    assert "Never invent an offer." in request
    assert "Free delivery everywhere" in request


def test_an_empty_brief_is_explicit_that_no_claim_is_allowed() -> None:
    request = build_critique_request(brief="", draft={}, language="ar")
    assert "none supplied" in request
    assert "Arabic" in request


def test_the_verdict_schema_carries_the_repair_as_a_string() -> None:
    """No nested object reference: providers that drop `$defs` would drop the fix."""

    schema = critique_schema(CreativeBrief)
    assert schema.model_fields["revised_json"].annotation is str
    assert "revised" not in schema.model_json_schema().get("$defs", {})
    assert "$ref" not in json.dumps(schema.model_json_schema())

    parsed = schema.model_validate(
        {"verdict": "revise", "revised_json": FIXED.model_dump_json()}
    )
    reviewed, _ = apply_verdict(DRAFT, parsed, schema=CreativeBrief)
    assert reviewed.variants[0].headline.startswith("Beans")


# --------------------------------------------------------------------------- #
# A busy provider is not a broken review                                      #
# --------------------------------------------------------------------------- #


class _FlakyRunnable:
    """Fails the first ``failures`` calls with a provider-shaped error."""

    def __init__(self, result: Any, *, error: Exception, failures: int) -> None:
        self.result = result
        self.error = error
        self.failures = failures
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return self.result


class _FlakyHandle:
    def __init__(self, runnable: _FlakyRunnable) -> None:
        self.runnable = runnable

    def structured(self, schema: type[Any]) -> _FlakyRunnable:
        return self.runnable


class _GoogleLikeError(Exception):
    """Stands in for GoogleAPIError: the status travels on the exception."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.asyncio
async def test_a_503_is_retried_and_the_review_still_lands() -> None:
    """The exact failure that shipped an unreviewed draft to the user."""
    runnable = _FlakyRunnable(
        OK_VERDICT, error=_GoogleLikeError("model is overloaded", status_code=503), failures=1
    )
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    verdict = await critique_deliverable(
        _FlakyHandle(runnable),
        brief="brief",
        draft=DRAFT.model_dump(),
        schema=critique_schema(CreativeBrief),
        sleep=pause,
    )

    assert runnable.calls == 2
    assert delays == [2.0]
    assert verdict is not None


@pytest.mark.asyncio
async def test_a_rate_limit_is_retried_by_its_message_when_no_status_is_exposed() -> None:
    """Not every client surfaces a status code; the wording still identifies it."""
    runnable = _FlakyRunnable(
        OK_VERDICT, error=RuntimeError("429 RESOURCE_EXHAUSTED: quota"), failures=1
    )
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    verdict = await critique_deliverable(
        _FlakyHandle(runnable),
        brief="brief",
        draft=DRAFT.model_dump(),
        schema=critique_schema(CreativeBrief),
        sleep=pause,
    )

    assert runnable.calls == 2 and verdict is not None


@pytest.mark.asyncio
async def test_retries_are_bounded_and_the_draft_is_kept() -> None:
    """A provider that stays down must not hold the deliverable hostage."""
    runnable = _FlakyRunnable(
        OK_VERDICT, error=_GoogleLikeError("still overloaded", status_code=503), failures=99
    )
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    verdict = await critique_deliverable(
        _FlakyHandle(runnable),
        brief="brief",
        draft=DRAFT.model_dump(),
        schema=critique_schema(CreativeBrief),
        sleep=pause,
    )

    assert verdict is None
    assert runnable.calls == 3  # three attempts, then the draft is delivered as written
    assert len(delays) == 2


@pytest.mark.asyncio
async def test_a_deterministic_error_is_not_retried() -> None:
    """Repeating a schema/validation failure only burns tokens."""
    runnable = _FlakyRunnable(
        OK_VERDICT, error=ValueError("verdict did not validate"), failures=99
    )
    delays: list[float] = []

    async def pause(seconds: float) -> None:
        delays.append(seconds)

    verdict = await critique_deliverable(
        _FlakyHandle(runnable),
        brief="brief",
        draft=DRAFT.model_dump(),
        schema=critique_schema(CreativeBrief),
        sleep=pause,
    )

    assert runnable.calls == 1
    assert delays == []
    assert verdict is None


def test_only_transient_conditions_are_treated_as_retryable() -> None:
    from app.ai.agents.layla.critique import _retryable_review_error

    assert _retryable_review_error(_GoogleLikeError("overloaded", status_code=503)) is True
    assert _retryable_review_error(RuntimeError("429 too many requests")) is True
    assert _retryable_review_error(RuntimeError("UNAVAILABLE: high demand")) is True

    # A wrapped cause is still the real reason.
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = _GoogleLikeError("overloaded", status_code=503)
    assert _retryable_review_error(wrapped) is True

    assert _retryable_review_error(ValueError("verdict did not validate")) is False
    assert _retryable_review_error(RuntimeError("provider exploded")) is False
    # 400s are our own bug: retrying sends the same bad request.
    assert _retryable_review_error(_GoogleLikeError("bad request", status_code=400)) is False
