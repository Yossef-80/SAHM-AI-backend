"""The creative eval harness must be trustworthy before it is used to judge Layla.

An evaluation that is easy to fool is worse than none: it turns "the copy is
clean" into a number nobody can act on. These tests check the scorer against the
dataset's own planted flaws, and against the false positives that would make a
real run useless (timecodes read as discounts, "keep the bay free" read as a
promotional claim).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.creative_eval_rules import (
    SEVERITY_FAIL,
    SEVERITY_WARN,
    score_case,
    summarize,
)

DATASET = Path(__file__).parent / "creative_eval.jsonl"


def load_cases() -> list[dict]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def fails(case: dict, payload: dict) -> list:
    return [f for f in score_case(case, payload) if f.severity == SEVERITY_FAIL]


CASES = load_cases()
BY_ID = {case["id"]: case for case in CASES}


# --------------------------------------------------------------------------- #
# The dataset itself                                                          #
# --------------------------------------------------------------------------- #


def test_the_dataset_covers_every_text_tool_and_both_languages() -> None:
    tools = {case["tool"] for case in CASES}
    languages = {case["language"] for case in CASES}
    assert tools == {
        "generate_ad_copy",
        "generate_ad_variations",
        "generate_commercial_script",
        "generate_image_prompt",
    }
    assert languages == {"ar", "en"}


def test_every_case_has_a_recorded_deliverable_and_a_planted_flaw() -> None:
    for case in CASES:
        assert case.get("recorded"), case["id"]
        assert case.get("flawed"), case["id"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_recorded_deliverables_are_clean(case: dict) -> None:
    """If a recorded payload fails, the rules -- not the copy -- are wrong."""

    assert fails(case, case["recorded"]) == []


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_planted_flaws_are_detected(case: dict) -> None:
    detected = fails(case, case["flawed"])
    assert detected, f"{case['id']}: the harness did not notice the planted flaw"


# --------------------------------------------------------------------------- #
# Individual checks                                                           #
# --------------------------------------------------------------------------- #


def test_an_unsupported_number_is_caught() -> None:
    case = BY_ID["en_adcopy_gym_no_claims"]
    findings = fails(case, {"primary_texts": ["Save 40% this month."], "headlines": ["Join"]})
    assert any("40" in f.detail for f in findings)


def test_timecodes_are_not_treated_as_claims() -> None:
    """A shot list is full of numbers and none of them is a promise."""

    case = BY_ID["en_script_laundry"]
    payload = {
        "hook": "The bag by the door.",
        "scenes": [
            {"order": 1, "timecode": "0-3s", "visual": "Bag by a door.", "voiceover": "That bag."},
            {"order": 2, "timecode": "3-9s", "visual": "Courier.", "voiceover": "We collect it."},
            {"order": 3, "timecode": "9-15s", "visual": "Folded.", "voiceover": "And return it."},
        ],
        "cta": "",
        "duration_seconds": 15,
    }
    assert fails(case, payload) == []


def test_repeated_variations_are_caught() -> None:
    case = BY_ID["en_variations_car_parts"]
    payload = {
        "variations": [
            {"angle": "a", "headline": "Genuine parts to Giza in 48 hours", "primary_text": "Same words here."},
            {"angle": "b", "headline": "Genuine parts to Giza within 48 hours", "primary_text": "Same words here."},
        ],
        "based_on": "",
    }
    findings = fails(case, payload)
    assert any(f.check == "duplicate_variants" for f in findings)


def test_arabic_copy_for_an_english_request_is_caught() -> None:
    case = BY_ID["en_adcopy_gym_no_claims"]
    findings = fails(
        case,
        {"primary_texts": ["جيم صغير في المعادي مع مدربين يعرفونك."], "headlines": ["درّب قريب منك"]},
    )
    assert any(f.check == "language" for f in findings)


def test_english_copy_for_an_arabic_request_is_caught() -> None:
    case = BY_ID["ar_adcopy_office_coffee"]
    findings = fails(
        case,
        {"primary_texts": ["Coffee delivered to your office."], "headlines": ["Office coffee"]},
    )
    assert any(f.check == "language" for f in findings)


def test_a_placeholder_is_caught() -> None:
    case = BY_ID["ar_image_prompt_abaya"]
    findings = fails(case, {"prompt": "[PRODUCT DESCRIPTION HERE]", "aspect_ratio": "4:5"})
    assert any(f.check == "placeholder" for f in findings)


def test_an_empty_required_field_is_caught() -> None:
    case = BY_ID["en_adcopy_gym_no_claims"]
    findings = fails(case, {"primary_texts": ["A gym in Maadi."], "headlines": []})
    assert any(f.check == "missing_field" for f in findings)


def test_ambiguous_language_warns_rather_than_fails() -> None:
    """"keep the bay free" is not a promotional claim; a harness that says it is gets ignored."""

    case = BY_ID["en_variations_car_parts"]
    payload = {
        "variations": [
            {
                "angle": "a",
                "headline": "Genuine parts, 48 hours to Giza",
                "primary_text": "Keep the bay free while the part travels.",
            },
            {"angle": "b", "headline": "Keep the workshop moving", "primary_text": "Plan the fit ahead."},
        ],
        "based_on": "",
    }
    findings = score_case(case, payload)
    assert [f for f in findings if f.severity == SEVERITY_FAIL] == []
    assert [f for f in findings if f.severity == SEVERITY_WARN]


def test_summarize_counts_cases_not_findings() -> None:
    case = BY_ID["en_adcopy_gym_no_claims"]
    results = [
        (case, score_case(case, case["recorded"])),
        (case, score_case(case, case["flawed"])),
    ]
    summary = summarize(results)
    assert summary["cases"] == 2
    assert summary["failed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["failures_by_check"].get("unsupported_claim")


# --------------------------------------------------------------------------- #
# The CLI                                                                     #
# --------------------------------------------------------------------------- #


def test_the_offline_self_check_passes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """`python tests/run_creative_eval.py --mock` is the harness's own gate."""

    from tests import run_creative_eval

    monkeypatch.setattr(sys, "argv", ["run_creative_eval.py", "--mock"])
    assert run_creative_eval.main() == 0
    out = capsys.readouterr().out
    assert "harness self-check" in out


def test_the_offline_self_check_fails_when_a_rule_is_blind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Blinding a rule must make the run fail, not quietly pass."""

    from tests import run_creative_eval, creative_eval_rules

    monkeypatch.setattr(creative_eval_rules, "CHECKS", ())
    monkeypatch.setattr(run_creative_eval, "score_case", lambda case, payload: [])
    monkeypatch.setattr(sys, "argv", ["run_creative_eval.py", "--mock"])
    assert run_creative_eval.main() == 1
