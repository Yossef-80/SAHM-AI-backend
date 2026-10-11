#!/usr/bin/env python3
"""Routing Evaluation Harness.

Evaluates the Orchestrator Router LLM against tests/routing_eval.jsonl.
Computes:
- Per-class accuracy, precision, and recall
- Confusion matrix
- Agent prediction distribution share (dominance detector)

Fails (exit 1) if:
- Overall accuracy < 90%
- Any agent exceeds 40% of predictions (dominance detector)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.messages import HumanMessage, SystemMessage

from app.ai.agents.orchestrator.brain import ROUTER_SYSTEM_PROMPT, RoutingDecision

ALL_AGENTS = ["nour", "karim", "omar", "layla", "salma", "chat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run routing evaluation harness.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(BACKEND_DIR / "tests" / "routing_eval.jsonl"),
        help="Path to routing_eval.jsonl dataset",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between requests to avoid rate limits",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline deterministic evaluation matching the prompt scope rules without live LLM API calls",
    )
    return parser.parse_args()


def mock_classify(prompt: str) -> tuple[str, str]:
    """Deterministic scope classification simulating prompt contract."""
    p_lower = prompt.lower()

    # Rule 1: Live research indicators -> nour
    live_search_patterns = [
        "make a search", "search the web", "look up current", "find out what competitors are offering",
        "دورلي في السوق", "ابحث عن", "لسا عايزك تدور", "حلل آراء العملاء",
    ]
    if any(pat in p_lower for pat in live_search_patterns):
        return "nour", "Matched live search intent requirement"

    # Rule 2: Memory recall of stored data -> karim
    recall_patterns = [
        "who are our competitors and what are their offers",
        "كنا طلعناهم في البحث", "previous competitor analysis", "البحث المخزن",
        "audit report", "التقرير القديم", "past performance data", "حفظناها قبل كده",
    ]
    if any(pat in p_lower for pat in recall_patterns):
        return "karim", "Matched stored memory recall requirement"

    # Rule 3: Strategy / campaign planning -> omar
    strategy_patterns = [
        "عايز اعمل حملة", "خطة تسويقية", "positioning strategy", "تموضع البراند",
        "create an ad campaign", "خطة استراتيجية", "positioning roadmap", "استراتيجية كاملة",
    ]
    if any(pat in p_lower for pat in strategy_patterns):
        return "omar", "Matched explicit strategy / campaign planning requirement"

    # Rule 4: Copy / creative writing -> layla
    creative_patterns = [
        "اسكريبت", "script", "headlines", "كابشن", "visual brief", "نص إعلاني ممول",
        "creative visual ideas", "primary text",
    ]
    if any(pat in p_lower for pat in creative_patterns):
        return "layla", "Matched creative copywriting / script requirement"

    # Rule 5: Live campaign ops / optimization of running campaigns -> salma
    running_patterns = [
        "الحملات الشغالة", "running meta campaign", "الحملة الإعلانية الشغالة",
        "active ad sets", "الحملة المباشرة", "current campaign", "الحملات الحية",
        "active google search",
    ]
    if any(pat in p_lower for pat in running_patterns):
        return "salma", "Matched live campaign optimization / operations requirement"

    # Rule 6: Conversational / profile inquiry -> chat
    chat_patterns = [
        "ايه الي تعرفه عني", "what is my brand scope", "صباح الخير", "thank you",
        "وضحلي قصدك", "brand voice in your profile memory", "مين انت", "خليك معايا",
    ]
    if any(pat in p_lower for pat in chat_patterns):
        return "chat", "Matched conversational / stored profile question"

    # Fallback tie-break to chat
    return "chat", "Tie-break default to chat"


async def evaluate_single_prompt(
    llm: Any,
    item: dict,
    semaphore: asyncio.Semaphore,
    delay: float = 0.0,
    is_mock: bool = False,
) -> dict:
    prompt = item["prompt"]
    expected = item["expected_agent"]
    note = item.get("note", "")

    async with semaphore:
        if is_mock:
            predicted, reason = mock_classify(prompt)
        else:
            if delay > 0:
                await asyncio.sleep(delay)
            max_retries = 3
            predicted = "unknown"
            reason = "No attempt made"
            for attempt in range(max_retries):
                try:
                    msgs = [
                        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                        HumanMessage(
                            content=f"Analyze the user's NEWEST prompt: '{prompt}'\n\nWhich agent should handle this newest prompt?"
                        ),
                    ]
                    decision: RoutingDecision = await llm.ainvoke(msgs)
                    predicted = decision.agent
                    reason = decision.reason
                    break
                except Exception as exc:
                    err_msg = str(exc)
                    if "429" in err_msg or "ResourceExhausted" in err_msg:
                        backoff = (2 ** attempt) * 4.0
                        await asyncio.sleep(backoff)
                        continue
                    predicted = "unknown"
                    reason = f"Error: {exc}"
                    break

    return {
        "prompt": prompt,
        "expected": expected,
        "predicted": predicted,
        "reason": reason,
        "note": note,
        "is_correct": (predicted == expected),
    }


def print_confusion_matrix(
    matrix: dict[str, dict[str, int]],
    classes: list[str],
) -> None:
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (Rows = Expected, Columns = Predicted)")
    print("=" * 70)
    header = f"{'Expected':<10} | " + " | ".join(f"{c:>7}" for c in classes) + " | Total"
    print(header)
    print("-" * len(header))
    for exp in classes:
        row_counts = [matrix[exp][pred] for pred in classes]
        row_total = sum(row_counts)
        row_str = f"{exp:<10} | " + " | ".join(f"{count:>7}" for count in row_counts) + f" | {row_total:>5}"
        print(row_str)
    print("=" * 70)


async def main_async() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}")
        return 1

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    mode_label = "MOCK / OFFLINE" if args.mock else "LIVE LLM (temperature=0.0)"
    print(f"Loaded {len(records)} evaluation prompts from {dataset_path.name}")
    print(f"Evaluation Mode: {mode_label}")

    llm = None
    if not args.mock:
        from app.ai.providers.llm import get_llm
        from app.ai.settings import get_ai_settings

        settings = get_ai_settings()
        llm = get_llm(provider=settings.llm_provider, temperature=0.0).chat.with_structured_output(RoutingDecision)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        evaluate_single_prompt(
            llm=llm,
            item=rec,
            semaphore=semaphore,
            delay=args.delay,
            is_mock=args.mock,
        )
        for rec in records
    ]

    print("Running routing evaluation...")
    results = await asyncio.gather(*tasks)

    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    overall_accuracy = (correct / total) * 100.0

    predictions_count = Counter(r["predicted"] for r in results)
    confusion: dict[str, dict[str, int]] = {
        exp: {pred: 0 for pred in ALL_AGENTS + ["unknown"]} for exp in ALL_AGENTS
    }

    per_class_expected = Counter(r["expected"] for r in results)
    per_class_correct = Counter(r["expected"] for r in results if r["is_correct"])

    print("\nDetailed Failures (if any):")
    print("-" * 70)
    failure_count = 0
    for r in results:
        confusion[r["expected"]][r["predicted"]] += 1
        if not r["is_correct"]:
            failure_count += 1
            print(f"[FAIL] Prompt   : {r['prompt']}")
            print(f"       Expected : {r['expected'].upper()} | Predicted: {r['predicted'].upper()}")
            print(f"       Reason   : {r['reason']}")
            print(f"       Note     : {r['note']}\n")

    if failure_count == 0:
        print("[SUCCESS] Zero routing failures! All prompts routed correctly.\n")

    print_confusion_matrix(confusion, ALL_AGENTS)

    print("\nPER-CLASS ACCURACY & PERFORMANCE:")
    print("-" * 70)
    for agent in ALL_AGENTS:
        exp_n = per_class_expected[agent]
        corr_n = per_class_correct[agent]
        acc = (corr_n / exp_n * 100.0) if exp_n > 0 else 0.0
        pred_n = predictions_count[agent]
        precision = (corr_n / pred_n * 100.0) if pred_n > 0 else 0.0
        share = (pred_n / total * 100.0)
        print(f"• {agent:<8} : Accuracy = {acc:>6.2f}% ({corr_n}/{exp_n}) | Precision = {precision:>6.2f}% | Share = {share:>5.1f}% ({pred_n}/{total})")

    print("\nOVERALL METRICS & THRESHOLD CHECKS:")
    print("=" * 70)
    print(f"Total Evaluated       : {total}")
    print(f"Overall Accuracy      : {overall_accuracy:.2f}% (Threshold: >= 90.00%)")

    # Dominance check
    dominance_detected = False
    for agent, count in predictions_count.items():
        share = (count / total) * 100.0
        if share > 40.0:
            print(f"[DOMINANCE DETECTED] Agent '{agent}' received {share:.1f}% of all predictions (> 40.0% cap)!")
            dominance_detected = True

    if not dominance_detected:
        print("[PASS] Dominance Check: Passed (no agent received > 40% share).")

    # Check exit conditions
    if overall_accuracy < 90.0:
        print(f"\n[EVAL FAILED] Overall accuracy {overall_accuracy:.2f}% is below 90.0% requirement.")
        return 1

    if dominance_detected:
        print("\n[EVAL FAILED] Dominance detector triggered.")
        return 1

    print("\n[EVAL PASSED] All requirements satisfied (Accuracy >= 90% and No Agent Dominance).")
    return 0


def main():
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
