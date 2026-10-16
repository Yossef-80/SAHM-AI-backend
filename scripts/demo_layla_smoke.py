#!/usr/bin/env python3
"""Live smoke test for the Layla upgrades.

Answers one question end to end, against the real provider: does the creative
path now use the strong tier at ad-copy temperature, with brand memory and past
examples in the prompt, a review pass before delivery, and a scorer to prove it?

    python scripts/demo_layla_smoke.py            # real provider calls
    python scripts/demo_layla_smoke.py --dry-run  # show routing only, no calls

Costs three provider calls at most (write, review, and nothing else).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

from app.ai.agents.layla.learning import format_examples_block  # noqa: E402
from app.ai.providers.llm.agent_llm import AGENT_ROLE, process_handle  # noqa: E402
from app.ai.roles import DEFAULTS  # noqa: E402
from app.ai.settings import get_ai_settings  # noqa: E402

BRIEF = "اكتب نسخة إعلانية لحملة قهوة مختصة للمكاتب، العرض: خصم 12% للفرق من 10 لـ 50 موظف، توصيل داخل القاهرة."

BRAND_FACTS = """Brand voice: دافي وبسيط، بدون مبالغة
Target audience: مديري مكاتب بيطلبوا لفريق من 10 لـ 50 شخص
Key offers: خصم 12% للفرق، توصيل أيام العمل داخل القاهرة
Goals: زيادة الاشتراكات الشهرية
Business: محمصة قهوة، القاهرة"""

PAST_EXAMPLES = [
    "Brief: إعلان اشتراك قهوة للمكاتب\n"
    "Rejected: أفضل قهوة في مصر بضمان الجودة\n"
    "Accepted: بنوصل بن أسبوعي لمكتبك، وخصم 12% للفرق من 10 أفراد",
]


def show_routing() -> None:
    settings = get_ai_settings()
    print("role routing")
    print(f"  fast   tier: {settings.provider_for_tier('fast')} / {settings.model_for_tier('fast')}")
    print(f"  strong tier: {settings.provider_for_tier('strong')} / {settings.model_for_tier('strong')}")
    print("  agent -> role -> tier -> temperature -> model")
    for agent, role in AGENT_ROLE.items():
        spec = settings.role_spec(role)
        print(
            f"    {agent:7} {role:11} {spec.tier:7} {spec.temperature:<5} "
            f"{settings.model_for_tier(spec.tier)}"
        )
    print("  writing roles")
    for role in ("creative_copy", "creative_script", "creative_brief", "critique", "routing"):
        spec = DEFAULTS[role]
        print(f"    {role:16} {spec.tier:7} temp={spec.temperature}")
    print()


#: A draft with the two defects the review exists to catch: claims the brief never
#: supplied, and three variations that are one sentence re-cut.
FLAWED_DRAFT = {
    "deliverable_type": "ad_variations",
    "summary": "نسخ إعلانية لحملة قهوة المكاتب.",
    "variants": [
        {
            "angle": "offer",
            "headline": "أرخص قهوة مختصة في مصر مع ضمان استرداد المبلغ",
            "primary_text": "توصيل مجاني في كل مصر وأرخص سعر، والدفع بعد التجربة.",
            "call_to_action": "اشترك الآن",
        },
        {
            "angle": "offer2",
            "headline": "أرخص قهوة مختصة في مصر، ضمان استرداد المبلغ",
            "primary_text": "توصيل مجاني في كل مصر وبأرخص سعر، وتقدر تدفع بعد التجربة.",
            "call_to_action": "اشترك الآن",
        },
        {
            "angle": "offer3",
            "headline": "أرخص قهوة مختصة في مصر",
            "primary_text": "توصيل مجاني لكل مصر، الدفع بعد التجربة.",
            "call_to_action": "اشترك الآن",
        },
    ],
}


async def run_flawed_review() -> int:
    """Prove the review catches invented claims and duplicate angles, live."""

    from app.ai.agents.layla.critique import (
        apply_verdict,
        critique_deliverable,
        critique_schema,
    )
    from app.ai.agents.layla.schema import CreativeBrief

    from tests.creative_eval_rules import SEVERITY_FAIL, score_case

    from app.ai.agents.layla.service import _format_output

    draft = CreativeBrief.model_validate(FLAWED_DRAFT)
    # Scored in the shape the UI is served, not the internal schema: the rules
    # judge what the user sees.
    as_output = lambda model: _format_output("generate_ad_variations", model)  # noqa: E731
    case = {
        "id": "smoke-flawed",
        "tool": "generate_ad_variations",
        "language": "ar",
        "brief": BRIEF,
        "brand_context": BRAND_FACTS,
    }
    before = [f for f in score_case(case, as_output(draft)) if f.severity == SEVERITY_FAIL]
    print(f"scorer on the planted draft: {len(before)} failures")
    for finding in before:
        print(f"    {finding.check}: {finding.detail}")

    print("\nrunning the review pass (critique role, cold)...")
    verdict = None
    for attempt in range(1, 4):
        verdict = await critique_deliverable(
            process_handle("critique"),
            brief=BRIEF,
            draft=draft.model_dump(),
            brand_context_text=BRAND_FACTS,
            language="ar",
            schema=critique_schema(CreativeBrief),
        )
        if verdict is not None:
            break
        print(f"  attempt {attempt}: no verdict (provider capacity or unusable output)")
        await asyncio.sleep(5 * attempt)
    if verdict is None:
        print("The review pass could not run. If this is a 503, switch LLM_MODEL_STRONG in .env.")
        return 3
    print(f"verdict: {verdict.verdict} | defects: {verdict.defects or '-'}")
    if verdict.notes:
        print(f"notes: {verdict.notes}")

    reviewed, notes = apply_verdict(draft, verdict, schema=CreativeBrief)
    after = [f for f in score_case(case, as_output(reviewed)) if f.severity == SEVERITY_FAIL]
    print("\n--- reviewed deliverable ---")
    for variant in reviewed.variants:
        print(f"  • {variant.headline} | {variant.primary_text}")
    print(f"\nscorer after review: {len(after)} failures (was {len(before)})")
    for finding in after:
        print(f"    still failing {finding.check}: {finding.detail}")
    for note in notes:
        print(f"  note: {note}")
    return 0 if len(after) < len(before) or not before else 1


async def main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print routing only")
    parser.add_argument("--no-critique", action="store_true", help="skip the review pass")
    parser.add_argument(
        "--flawed",
        action="store_true",
        help="hand the review pass a draft with invented claims and duplicates, and show the repair",
    )
    args = parser.parse_args()

    show_routing()
    if args.dry_run:
        return 0
    if args.flawed:
        return await run_flawed_review()

    from app.ai.agents.layla.service import generate_layla_text

    context = BRAND_FACTS + "\n\n" + format_examples_block(PAST_EXAMPLES)
    print("generating ad copy (strong tier, creative_copy temperature)...")
    # The free tier of a hosted model returns 503 under load; that is a capacity
    # fact, not a bug in this pipeline. Retry a couple of times before giving up,
    # and say which model answered so a 503 run is never mistaken for a bad draft.
    output = None
    for attempt in range(1, 4):
        try:
            output = await generate_layla_text(
                tool="generate_ad_copy",
                brief=BRIEF,
                language="ar",
                brand_context_text=context,
                llm=process_handle("creative_copy"),
                critique_llm=None if args.no_critique else process_handle("critique"),
                critique=not args.no_critique,
            )
            break
        except Exception as exc:
            print(f"  attempt {attempt} failed ({type(exc).__name__}: {str(exc)[:90]})")
            if attempt == 3:
                print(
                    "\nThe strong-tier model is overloaded. Switch LLM_MODEL_STRONG in .env "
                    "(gemini-3.8-flash or gemini-3.5-flash) and re-run."
                )
                return 3
            await asyncio.sleep(5 * attempt)

    print("\n--- primary texts ---")
    for line in output.get("primary_texts") or []:
        print(f"  • {line}")
    print("--- headlines ---")
    for line in output.get("headlines") or []:
        print(f"  • {line}")
    if output.get("ctas"):
        print("--- ctas ---")
        for line in output["ctas"]:
            print(f"  • {line}")
    if output.get("notes"):
        print("--- notes (includes anything the review pass corrected) ---")
        print(f"  {output['notes']}")

    # The same rules the eval harness uses, so this run is measured, not admired.
    from tests.creative_eval_rules import SEVERITY_FAIL, score_case

    case = {
        "id": "smoke",
        "tool": "generate_ad_copy",
        "language": "ar",
        "brief": BRIEF,
        "brand_context": BRAND_FACTS,
    }
    findings = score_case(case, output)
    failures = [f for f in findings if f.severity == SEVERITY_FAIL]
    warnings = [f for f in findings if f.severity != SEVERITY_FAIL]
    print("\n--- eval ---")
    print(f"  failures: {len(failures)} | warnings: {len(warnings)}")
    for finding in failures:
        print(f"  FAIL {finding.check}: {finding.detail}")
    for finding in warnings:
        print(f"  warn {finding.check}: {finding.detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
