"""One-off: move existing base64 creatives out of the database and into storage.

Why this exists
---------------
``creatives.url`` used to hold the image itself: a ``data:image/...;base64,...``
string of one to several megabytes. Every row that carries one is slow to read,
slow to transfer, and slow to back up -- and the column is selected by the
creatives list endpoint, so the whole list paid for it on every page load.

This script walks rows whose ``url`` starts with ``data:``, uploads the bytes to
Supabase Storage and rewrites the row to the public URL. Idempotent: rows that
are already URLs are skipped, so it is safe to run more than once, and safe to
run after the app has been redeployed (new generations never write base64).

Usage (from the backend folder)::

    python scripts/migrate_creatives_to_storage.py            # dry run, reports only
    python scripts/migrate_creatives_to_storage.py --apply    # performs the migration
    python scripts/migrate_creatives_to_storage.py --apply --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.storage import (  # noqa: E402
    is_supabase_configured,
    persist_image_asset,
    sniff_image_mime,
)
from app.db.models import Creative  # noqa: E402
from app.db.session import dispose_engine, session_scope  # noqa: E402


def _describe(data_url: str) -> tuple[int, str]:
    header, _, encoded = data_url.partition(",")
    mime = header[5:].split(";", 1)[0] or "unknown"
    approx_bytes = (len(encoded) * 3) // 4
    return approx_bytes, mime


async def run(*, apply: bool, limit: int | None) -> int:
    settings = get_settings()
    if not is_supabase_configured(settings):
        print("[!] Supabase Storage is not configured (SUPABASE_URL + SUPABASE_SECRET_KEY).")
        print("    Nothing can be uploaded; set the credentials and run again.")
        return 2

    migrated = 0
    skipped = 0
    failed = 0
    saved_bytes = 0

    async with session_scope() as db:
        stmt = select(Creative).where(Creative.url.like("data:image/%"))
        if limit:
            stmt = stmt.limit(limit)
        rows = list((await db.execute(stmt)).scalars().all())

        print(f"[i] {len(rows)} creative row(s) still carry an inline image.")
        for row in rows:
            approx_bytes, mime = _describe(row.url or "")
            if sniff_image_mime(base64.b64decode((row.url or "").partition(",")[2] or b"")) is None:
                # Not a real image: leave it alone rather than half-migrating it.
                print(f"    [skip] {row.id}: unrecognised payload ({mime}, ~{approx_bytes:,} B)")
                skipped += 1
                continue
            if not apply:
                print(f"    [dry]  {row.id}: {mime}, ~{approx_bytes:,} B -> storage")
                saved_bytes += approx_bytes
                continue

            url, error = await persist_image_asset(
                settings,
                asset_url=row.url,
                path_prefix=f"migrated/{row.business_id}",
                bucket=str(getattr(settings, "storage_bucket", "creatives") or "creatives"),
            )
            if not url:
                print(f"    [fail] {row.id}: {error}")
                failed += 1
                continue
            row.url = url
            migrated += 1
            saved_bytes += approx_bytes
            print(f"    [ok]   {row.id}: {approx_bytes:,} B -> {url}")

        if apply and migrated:
            await db.commit()

    verb = "would free" if not apply else "freed"
    print(
        f"\n[i] done: migrated={migrated} skipped={skipped} failed={failed}; "
        f"{verb} ~{saved_bytes / (1024 * 1024):.1f} MB from the database"
    )
    if not apply and rows:
        print("[i] dry run only. Re-run with --apply to perform the migration.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform the migration")
    parser.add_argument("--limit", type=int, default=None, help="only process N rows")
    args = parser.parse_args(argv)

    async def _main() -> int:
        try:
            return await run(apply=args.apply, limit=args.limit)
        finally:
            await dispose_engine()

    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
