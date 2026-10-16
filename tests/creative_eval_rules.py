"""Scoring rules for the creative evaluation harness.

`tests/run_routing_eval.py` scores *decisions*; this scores *writing*. The checks
are deliberately rule-based rather than model-judged: they must be deterministic,
runnable without a provider key, and cheap enough to run on every case in CI.

What is worth checking automatically, and why:

* **unsupported claims** -- the failure that has real consequences (a guarantee
  the business never made). Any number token or claim word must be traceable to
  the brief or the brand facts.
* **language** -- a bilingual product ships Arabic copy into English campaigns by
  accident far more often than anyone expects.
* **distinctness** -- "three variations" that are one sentence three times is the
  defect the per-role temperature work exists to remove, so it has to be measured.
* **required fields** -- an empty headline is not a style question.
* **placeholders** -- a leaked template fragment means the draft was never read.

Deliberately *not* checked: tone, punchiness, brand-fit. Those need judgement, and
a rule that pretends to judge them would be noise in a report people have to read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

SEVERITY_FAIL = "fail"
SEVERITY_WARN = "warn"

#: Numbers, percentages and money are what an invented claim usually turns into.
#: The lookbehind keeps identifiers out: "offer3" and "CTA2" are labels, not
#: quantities, and flagging them would make the number check useless noise.
_NUMBER_RE = re.compile(r"(?<![\w\d])\d+(?:[.,]\d+)?\s*%?")

#: Claim vocabulary in both product languages, with the severity a hit deserves.
#:
#: A hit must be traceable to the brief/facts. The severity split matters: an
#: explicit "free delivery" or "ضمان" is a commercial promise and a fail, while a
#: bare "free" (as in "keep the bay free") or "مضمون" (used for "solid work") is
#: ambiguous language that a reviewer should look at, not a defect to fail a run
#: on. A harness that cries wolf gets ignored, so ambiguous words warn.
_CLAIM_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\bfree\s+(?:delivery|shipping|fitting|installation|trial|gift|sample|consultation)\b|\bfor\s+free\b|\bfree\s+of\s+charge\b", "free", SEVERITY_FAIL),
    (r"\bfree\b", "free", SEVERITY_WARN),
    (r"\bguarantee(?:d)?\b|\bwarranty\b|\bmoney[- ]back\b", "guarantee", SEVERITY_FAIL),
    (r"\bcheapest\b|\blowest\s+price\b|\bbest\s+price\b|\brisk[- ]free\b", "superlative_price", SEVERITY_FAIL),
    (r"\blifetime\b|\b24/7\b|\bsame[- ]day\b|\binstant(?:ly)?\b", "absolute_service", SEVERITY_FAIL),
    (r"\baward[- ]winning\b|\bnumber\s*one\b|\b#1\b", "award", SEVERITY_FAIL),
    (r"مجان(?:ا|ًا)?", "free", SEVERITY_FAIL),
    (r"ضمان|استرداد\s+المبلغ", "guarantee", SEVERITY_FAIL),
    (r"مضمون", "guarantee", SEVERITY_WARN),
    (r"أرخص|الرخيص\s+الأول|أفضل\s+سعر|بدون\s+مخاطرة", "superlative_price", SEVERITY_FAIL),
    (r"مدى\s+الحياة|فوري(?:ا|ًا)?|على\s+مدار\s+الساعة", "absolute_service", SEVERITY_FAIL),
    (r"حاصل\s+على\s+جائزة|الأول\s+في|رقم\s+واحد", "award", SEVERITY_FAIL),
)

#: Timecodes are numbers, but they are not claims. `0-3s`, `9–15s`, `00:03`.
_TIMECODE_RE = re.compile(
    r"\b\d{1,2}\s*[:\-–]\s*\d{1,2}\s*(?:s|sec|secs|seconds|ث|ثانية|ثواني)?\b",
    re.IGNORECASE,
)

#: CTAs that carry no claim and are always acceptable.
_GENERIC_CTAS = {
    "learn more",
    "shop now",
    "discover more",
    "see more",
    "get started",
    "contact us",
    "اعرف المزيد",
    "اكتشف المزيد",
    "تواصل معنا",
    "ابدأ الآن",
    "تسوق الآن",
}

_PLACEHOLDER_RE = re.compile(
    r"\[[^\]]{0,40}\]|\{\{|\btbd\b|\btodo\b|lorem ipsum|\bxxx+\b|<[a-z_]+>",
    re.IGNORECASE,
)

#: Fields that must be present and non-empty, per tool. Mirrors the shapes
#: `_format_output` publishes for each UI surface.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "generate_creative_brief": ("concept", "headline"),
    "generate_commercial_script": ("hook", "scenes"),
    "generate_image_prompt": ("prompt",),
    "generate_video_prompt": ("prompt", "scenes"),
    "generate_ad_copy": ("primary_texts", "headlines"),
    "generate_ad_variations": ("variations",),
}


@dataclass(frozen=True)
class Finding:
    """One problem, or one thing the harness could not confirm."""

    case_id: str
    check: str
    severity: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "check": self.check,
            "severity": self.severity,
            "detail": self.detail,
        }


def _strings(value: Any) -> list[str]:
    """Every user-visible string inside a payload, flattened."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _strings(item)]
    return []


def _copy_text(payload: dict[str, Any]) -> str:
    """The deliverable's words, without the bookkeeping fields."""
    text_parts: list[str] = []
    for key, value in (payload or {}).items():
        if key in {"notes", "style_spec", "reference_notes", "negative_prompt", "aspect_ratio"}:
            continue
        text_parts.extend(_strings(value))
    return "\n".join(part for part in text_parts if part.strip())


def _allowed_text(case: dict[str, Any]) -> str:
    parts = [str(case.get("brief") or "")]
    brand = case.get("brand_context")
    if isinstance(brand, str):
        parts.append(brand)
    elif isinstance(brand, dict):
        parts.extend(_strings(brand))
    parts.extend(_strings(case.get("allowed_facts") or []))
    return "\n".join(parts)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def check_unsupported_claims(case_id: str, payload: dict[str, Any], case: dict[str, Any]) -> list[Finding]:
    """Every number and claim word must be traceable to the brief or the facts."""

    findings: list[Finding] = []
    copy = _copy_text(payload)
    # Timecodes come out before numbers are judged: "0-3s" is a shot list, not a
    # discount, and flagging it would make every script case fail.
    scannable = _TIMECODE_RE.sub(" ", copy)
    allowed = _normalize(_allowed_text(case))

    for token in _NUMBER_RE.findall(scannable):
        if _normalize(token) not in allowed:
            findings.append(
                Finding(case_id, "unsupported_claim", SEVERITY_FAIL, f"number '{token}' is not in the brief or brand facts")
            )

    for pattern, label, severity in _CLAIM_PATTERNS:
        match = re.search(pattern, copy, re.IGNORECASE)
        if match is None:
            continue
        if re.search(pattern, allowed, re.IGNORECASE):
            continue
        findings.append(
            Finding(
                case_id,
                "unsupported_claim",
                severity,
                f"claim '{match.group(0)}' ({label}) is not supported by the brief or brand facts",
            )
        )
    return findings


def check_language(case_id: str, payload: dict[str, Any], case: dict[str, Any]) -> list[Finding]:
    """The deliverable must be written in the requested language."""

    expected = str(case.get("language") or "en")
    copy = _copy_text(payload)
    if not copy.strip():
        return [Finding(case_id, "language", SEVERITY_FAIL, "deliverable is empty")]
    arabic = len(re.findall(r"[\u0600-\u06FF]", copy))
    letters = len(re.findall(r"[^\W\d_]", copy, re.UNICODE)) or 1
    ratio = arabic / letters
    if expected == "ar" and ratio < 0.5:
        return [Finding(case_id, "language", SEVERITY_FAIL, f"Arabic requested but only {ratio:.0%} of letters are Arabic")]
    if expected == "en" and ratio > 0.2:
        return [Finding(case_id, "language", SEVERITY_FAIL, f"English requested but {ratio:.0%} of letters are Arabic")]
    return []


def check_distinctness(case_id: str, payload: dict[str, Any], case: dict[str, Any]) -> list[Finding]:
    """Variations must differ in more than word order."""

    variants = payload.get("variations") or []
    if not isinstance(variants, list) or len(variants) < 2:
        return []
    texts = []
    for variant in variants:
        if isinstance(variant, dict):
            texts.append(_normalize(" ".join(_strings(variant))))
        elif isinstance(variant, str):
            texts.append(_normalize(variant))
    findings: list[Finding] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if not texts[i] or not texts[j]:
                continue
            ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
            if ratio > 0.75:
                findings.append(
                    Finding(
                        case_id,
                        "duplicate_variants",
                        SEVERITY_FAIL,
                        f"variations {i + 1} and {j + 1} are {ratio:.0%} identical",
                    )
                )
    return findings


def check_required_fields(case_id: str, payload: dict[str, Any], case: dict[str, Any]) -> list[Finding]:
    tool = str(case.get("tool") or "")
    findings: list[Finding] = []
    for field in REQUIRED_FIELDS.get(tool, ()):
        value = payload.get(field)
        empty = value is None or (isinstance(value, (str, list, dict)) and len(value) == 0)
        if empty:
            findings.append(Finding(case_id, "missing_field", SEVERITY_FAIL, f"'{field}' is empty"))
    return findings


def check_placeholders(case_id: str, payload: dict[str, Any], case: dict[str, Any]) -> list[Finding]:
    copy = _copy_text(payload)
    match = _PLACEHOLDER_RE.search(copy)
    if match:
        return [Finding(case_id, "placeholder", SEVERITY_FAIL, f"template fragment left in the copy: '{match.group(0)}'")]
    return []


def check_cta(case_id: str, payload: dict[str, Any], case: dict[str, Any]) -> list[Finding]:
    """A CTA is fine when it is generic or supported; flagged when invented."""

    ctas: list[str] = []
    for key in ("cta", "ctas", "call_to_action"):
        value = payload.get(key)
        ctas.extend(_strings(value))
    for variant in payload.get("variations") or []:
        if isinstance(variant, dict):
            ctas.extend(_strings(variant.get("cta") or variant.get("call_to_action") or ""))

    allowed = _normalize(_allowed_text(case))
    findings: list[Finding] = []
    for cta in ctas:
        normalized = _normalize(cta)
        if not normalized or normalized in _GENERIC_CTAS:
            continue
        # Anything not in the brief/facts and not generic is only a warning: a CTA
        # like "Get 12% off" is fine when the offer is real but phrased differently.
        if normalized not in allowed and any(
            re.search(pattern, cta, re.IGNORECASE) for pattern, _, _ in _CLAIM_PATTERNS
        ):
            findings.append(Finding(case_id, "cta_support", SEVERITY_WARN, f"CTA '{cta}' repeats a claim; verify it is supported"))
    return findings


CHECKS = (
    check_unsupported_claims,
    check_language,
    check_distinctness,
    check_required_fields,
    check_placeholders,
    check_cta,
)


def score_case(case: dict[str, Any], payload: dict[str, Any]) -> list[Finding]:
    """Run every check for one case against one deliverable."""
    case_id = str(case.get("id") or case.get("brief", ""))[:48]
    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(case_id, payload or {}, case))
    return findings


def summarize(results: list[tuple[dict[str, Any], list[Finding]]]) -> dict[str, Any]:
    """Aggregate the per-case findings into the numbers a report needs."""

    total = len(results)
    failed_cases = [case for case, findings in results if any(f.severity == SEVERITY_FAIL for f in findings)]
    by_check: dict[str, int] = {}
    for _, findings in results:
        for finding in findings:
            if finding.severity == SEVERITY_FAIL:
                by_check[finding.check] = by_check.get(finding.check, 0) + 1
    return {
        "cases": total,
        "passed": total - len(failed_cases),
        "failed": len(failed_cases),
        "pass_rate": (total - len(failed_cases)) / total if total else 0.0,
        "failures_by_check": by_check,
        "failed_case_ids": [str(case.get("id")) for case in failed_cases],
    }


__all__ = [
    "CHECKS",
    "Finding",
    "REQUIRED_FIELDS",
    "SEVERITY_FAIL",
    "SEVERITY_WARN",
    "score_case",
    "summarize",
]
