#!/usr/bin/env python3
"""Creative evaluation harness.

`run_routing_eval.py` answers "did Layla pick the right agent?"; this answers "is
the copy she wrote any good?" -- against `tests/creative_eval.jsonl`, using the
deterministic rules in `tests/creative_eval_rules.py`.

Two modes:

* ``--mock`` (offline, no provider key): scores the *recorded* deliverables
  shipped in the dataset. Every recorded payload must score clean and every
  planted flaw must be caught -- this is the harness checking itself, so a
  threshold change cannot silently neuter it.
* live: calls `generate_layla_text` for each case (including the self-critique
  pass, unless ``--no-critique`` is passed) and scores what comes back. Use this
  after changing a prompt, a temperature, or the model, and compare the two runs.

Exit code is 1 when the pass rate is below ``--min-pass-rate`` (default 0.9), so
it can gate a prompt change the way the routing harness gates a router change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - console dependent
        pass

from tests.creative_eval_rules import (  # noqa: E402
    Finding,
    SEVERITY_FAIL,
    SEVERITY_WARN,
    score_case,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the creative evaluation harness.")
    parser.add_argument(
        "--dataset",
        default=str(BACKEND_DIR / "tests" / "creative_eval.jsonl"),
        help="Path to the creative eval dataset",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Write a JSON report to this path (per-case findings included)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Offline: score the recorded deliverables in the dataset (harness self-check)",
    )
    parser.add_argument(
        "--no-critique",
        action="store_true",
        help="Skip Layla's self-critique pass in live mode (A/B the review step)",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--only", default="", help="Comma-separated case ids to run")
    return parser.parse_args()


def load_cases(path: str) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def label_of(case: dict[str, Any]) -> str:
    """Recorded-good cases have their own label: `flawed` must fail, `recorded` must pass."""
    return "flawed" if "flawed" in case else "recorded"


async def run_live(case: dict[str, Any], *, critique: bool, semaphore: asyncio.Semaphore, delay: float) -> dict[str, Any]:
    from app.ai.agents.layla.service import generate_layla_text

    async with semaphore:
        if delay:
            await asyncio.sleep(delay)
        try:
            payload = await generate_layla_text(
                tool=case["tool"],  # type: ignore[arg-type]
                brief=case["brief"],
                language=case.get("language", "en"),
                brand_context_text=str(case.get("brand_context") or ""),
                critique=critique,
            )
            return {"payload": payload, "error": None}
        except Exception as exc:  # a failed call is a failed case, not a crashed run
            return {"payload": {}, "error": f"{type(exc).__name__}: {exc}"}


async def main_async() -> int:
    args = parse_args()
    cases = load_cases(args.dataset)
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        cases = [case for case in cases if str(case.get("id")) in wanted]
    if not cases:
        print("No cases selected.")
        return 1

    # Each entry: the case, which payload was scored, and the findings for it.
    # Self-check mode scores two payloads per case (recorded + planted flaw), so
    # the label has to travel with the result rather than be re-derived.
    entries: list[dict[str, Any]] = []

    if args.mock:
        print(f"Offline self-check over {len(cases)} cases (recorded + planted flaws).\n")
        for case in cases:
            entries.append(
                {"case": case, "label": "recorded", "findings": score_case(case, case["recorded"])}
            )
            if "flawed" in case:
                entries.append(
                    {"case": case, "label": "flawed", "findings": score_case(case, case["flawed"])}
                )
    else:
        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        print(
            "Live evaluation over "
            f"{len(cases)} cases (self-critique: {'off' if args.no_critique else 'on'}).\n"
        )
        for case in cases:
            outcome = await run_live(
                case, critique=not args.no_critique, semaphore=semaphore, delay=args.delay
            )
            findings = score_case(case, outcome["payload"])
            if outcome["error"]:
                findings.append(
                    Finding(str(case.get("id")), "provider_error", SEVERITY_FAIL, outcome["error"])
                )
            entries.append({"case": case, "label": "live", "findings": findings})

    unexpected = 0
    for entry in entries:
        case = entry["case"]
        failures = [f for f in entry["findings"] if f.severity == SEVERITY_FAIL]
        warnings = [f for f in entry["findings"] if f.severity == SEVERITY_WARN]
        status = "FAIL" if failures else "PASS"
        note = ""
        if args.mock:
            # A recorded payload that fails, or a planted flaw that slips through,
            # is a harness bug -- either way the thresholds cannot be trusted.
            should_fail = entry["label"] == "flawed"
            if should_fail:
                note = " (planted flaw detected)" if failures else " (UNEXPECTED: flaw not detected)"
            elif failures:
                note = " (UNEXPECTED: recorded payload failed)"
            if bool(failures) != should_fail:
                unexpected += 1
        print(f"[{status}] {case.get('id')} [{entry['label']}]{note}")
        for finding in failures:
            print(f"    fail: {finding.check} -- {finding.detail}")
        for finding in warnings:
            print(f"    warn: {finding.check} -- {finding.detail}")

    summary = summarize([(entry["case"], entry["findings"]) for entry in entries])
    print("\n--- summary ---")
    print(f"scored deliverables : {summary['cases']}")
    print(f"clean               : {summary['passed']}")
    print(f"with failures       : {summary['failed']}")
    print(f"pass rate           : {summary['pass_rate']:.0%}")
    if summary["failures_by_check"]:
        print(
            "failures by check   : "
            + ", ".join(f"{k}={v}" for k, v in summary["failures_by_check"].items())
        )
    if args.mock:
        print(f"harness self-check  : {len(entries) - unexpected}/{len(entries)} as expected")

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "mode": "mock" if args.mock else "live",
                    "summary": summary,
                    "cases": [
                        {
                            "id": entry["case"].get("id"),
                            "label": entry["label"],
                            "findings": [f.as_dict() for f in entry["findings"]],
                        }
                        for entry in entries
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"report written to {args.report}")

    if args.mock:
        return 1 if unexpected else 0
    if summary["pass_rate"] < args.min_pass_rate:
        print(f"\nBelow the required pass rate of {args.min_pass_rate:.0%}.")
        return 1
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
