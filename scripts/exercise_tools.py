"""Exercise every agent tool in the registry and print what each one returns.

Why this exists
---------------
Only two tools have a direct HTTP endpoint (``POST /creatives/analyze-reference``
and ``POST /creatives/generate``). The other eleven can only be reached through a
model tool-call inside a task run, which makes them impossible to test
deterministically over HTTP. This script calls the handlers directly against a
throwaway SQLite database, so you get a repeatable pass/fail for all thirteen.

Usage (from the ``sahm-ai backend`` folder)::

    python scripts/exercise_tools.py            # run everything
    python scripts/exercise_tools.py web_search # run one tool by name
    python scripts/exercise_tools.py --list     # list tools and their inputs

It never touches your configured ``DATABASE_URL`` -- a temporary file-backed
SQLite database is used so this is always safe to run.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow `python scripts/exercise_tools.py` from the backend root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Never let a real provider or a real database be picked up by this script.
os.environ.setdefault("LLM_PROVIDER_FAST", "mock")
os.environ.setdefault("LLM_PROVIDER_STRONG", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("SEARCH_PROVIDER", "mock")
os.environ.setdefault("KEYWORD_PROVIDER", "mock")
os.environ.setdefault("IMAGE_PROVIDER", "mock")

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from app.core.llm.base import json_schema_for  # noqa: E402
from app.core.llm.mock_client import MockClient, sample_from_schema  # noqa: E402
from app.core.tools.base import ToolContext  # noqa: E402
from app.core.tools.registry import ALL_TOOLS, register_all  # noqa: E402
from app.db.models import Base, BrandProfile, Business, Campaign  # noqa: E402

BUSINESS_ID = "biz_exercise"
CAMPAIGN_ID = "cmp_exercise"

#: Sensible overrides so the generated payloads point at our seeded records.
INPUT_OVERRIDES: dict[str, dict[str, Any]] = {
    "get_campaign_insights": {"campaign_id": CAMPAIGN_ID},
    "get_breakdowns": {"campaign_id": CAMPAIGN_ID, "by": "age"},
    "analyze_reference_creative": {"image_url": "https://example.com/reference.jpg"},
    "generate_image": {"prompt": "A flat-lay of two iced coffees on a marble counter"},
    "web_search": {"query": "speciality coffee Cairo"},
    "get_competitor_ads": {"page_or_query": "Blue Bottle Coffee"},
    "analyze_website": {"url": "https://example.com"},
    "get_reviews": {"place_query": "Cairo Coffee Roasters"},
    "get_social_profile": {"platform": "instagram", "handle": "example"},
    "get_keywords": {"seed_terms": ["speciality coffee", "cold brew"]},
    "save_learning": {"what_worked": "A 12% discount on the first order lifted signups 18%."},
    "create_recommendation": {
        "action_type": "pause_ad",
        "title": "Pause the under-performing Ramadan reel",
    },
}


def build_payload(tool: Any) -> Any:
    """Synthesise a schema-valid input for ``tool``."""
    schema = json_schema_for(tool.input_model)
    raw = sample_from_schema(schema)
    overrides = INPUT_OVERRIDES.get(tool.name)
    if overrides:
        raw.update(overrides)
    return tool.input_model.model_validate(raw)


def _status_value(output: Any) -> str:
    """``status`` is a ToolStatus enum; compare on its value."""
    status = getattr(output, "status", None)
    if status is None:
        return "n/a"
    return str(getattr(status, "value", status))


def summarise(output: Any) -> str:
    """One-line description of a tool result."""
    status = _status_value(output)
    source = getattr(output, "source", None)
    reason = getattr(output, "unavailable_reason", None)
    if status == "unavailable":
        return f"UNAVAILABLE ({source}) - {reason}"
    parts = [f"status={status}"]
    if source:
        parts.append(f"source={source}")
    payload = output.model_dump(exclude_none=True)
    for key in ("results", "reviews", "keywords", "ads", "items", "assets", "rows", "series"):
        value = payload.get(key)
        if isinstance(value, list):
            parts.append(f"{key}={len(value)}")
    return "  ".join(parts)


async def seed(db: AsyncSession) -> None:
    """Insert the business, brand profile and campaign the tools expect."""
    business = Business(
        id=BUSINESS_ID,
        workspace_id="ws_exercise",
        name="Cairo Coffee Co",
        industry="F&B",
        website_url="https://example.com",
        country="EG",
        language="en",
        currency="EGP",
        monthly_budget=50_000.0,
    )
    db.add(business)
    db.add(
        BrandProfile(
            id=f"bp_{BUSINESS_ID}",
            business_id=BUSINESS_ID,
            brand_voice="Warm, direct, a little playful",
            visual_style="Warm neutrals, #1f2937, high-contrast product shots",
            target_audience="Urban professionals aged 25-40",
            key_offers=["Subscription coffee", "Same-day delivery"],
            goals=["Grow subscriptions"],
            competitors=["Blue Bottle", "Bonobo"],
        )
    )
    db.add(
        Campaign(
            id=CAMPAIGN_ID,
            business_id=BUSINESS_ID,
            name="Ramadan Push",
            status="active",
            objective="conversions",
            budget=30_000.0,
            currency="EGP",
        )
    )
    await db.commit()


async def main(argv: list[str]) -> int:
    registry = register_all()
    tools = ALL_TOOLS

    if "--list" in argv:
        print(f"{'tool':<30}{'side effect':<14}input fields")
        print("-" * 80)
        for tool in tools:
            fields = ", ".join(tool.input_model.model_fields)
            print(f"{tool.name:<30}{tool.side_effect.value:<14}{fields}")
        return 0

    wanted = [a for a in argv if not a.startswith("-")]
    if wanted:
        tools = [t for t in tools if t.name in wanted]
        if not tools:
            print(f"No tool matched {wanted}. Use --list to see the registry.")
            return 1

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp.name}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    ok = partial = unavailable = failed = 0
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as db:
            await seed(db)
            ctx = ToolContext(
                workspace_id="ws_exercise",
                business_id=BUSINESS_ID,
                db=db,
                llm=MockClient(model="mock-strong-1"),
                request_id="req_exercise",
                settings=None,
            )

            print(f"{'tool':<30}{'result'}")
            print("=" * 78)
            for tool in tools:
                try:
                    payload = build_payload(tool)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"{tool.name:<30}BAD PAYLOAD  {type(exc).__name__}: {exc}")
                    continue
                try:
                    output = await tool.handler(payload, ctx)
                    await db.commit()
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"{tool.name:<30}RAISED       {type(exc).__name__}: {exc}")
                    continue

                status = _status_value(output)
                if status == "unavailable":
                    unavailable += 1
                elif status == "partial":
                    partial += 1
                else:
                    ok += 1
                print(f"{tool.name:<30}{summarise(output)}")
    finally:
        await engine.dispose()
        os.unlink(tmp.name)

    print("=" * 78)
    print(f"ok={ok}  partial={partial}  unavailable={unavailable}  failed={failed}")
    print(
        "unavailable = the tool ran correctly and reported it has no data "
        "(missing key / no Meta connection). That is a pass, not a bug."
    )
    print(f"registered tools: {len(registry.names())}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))