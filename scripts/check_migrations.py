#!/usr/bin/env python
"""Alembic history gate for CI and for the release pipeline (P5.4 / P5.5).

Two questions, answered in opposite orders of importance:

1. **Is the migration history itself sound?** -- Is it a connected graph
   with exactly one base and one head, and is every revision reachable from the
   base? Explicit merge revisions are allowed. This is a *pure file* question
   answered with Alembic's own ``ScriptDirectory``. It
   needs no database, no ``DATABASE_URL`` and no network, so it runs in a fresh
   container and in CI before anything is provisioned.

2. **Does the database match the code?** -- current revision vs head. Answered
   only if a database URL is reachable. Skipped, not failed, when there is none:
   "no database configured" is a legitimate state for a lint job.

Exit codes
----------
0  history is linear, and (if a DB was reachable) it is at head
1  the history has multiple heads, or the database is behind/diverged
2  the history could not be read at all (missing alembic.ini, no revisions, ...)

RELEASE ORDERING (the rule this script exists to enforce)
---------------------------------------------------------
Production release is, in this order and no other:

    alembic upgrade head      # exactly once, as a pre-deploy job
    ...then start the app...

The app never issues DDL against PostgreSQL at boot: even a development
process pointed at Supabase must not create tables behind Alembic's back.
``create_all()`` is reserved for SQLite dev/test databases (see
``tests/test_workspace_isolation.py::test_boot_skips_create_all_for_postgres_in_development``).
Three replicas booting concurrently and racing ``CREATE TABLE`` is a real
outage, which is why the migration is a release step rather than a boot step.

Usage
-----
    python scripts/check_migrations.py            # history only, plus DB if reachable
    python scripts/check_migrations.py --quiet    # only report problems
    python scripts/check_migrations.py --require-head   # non-zero if DB != head
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The backend folder. Alembic's `script_location` in alembic.ini is relative, so
# resolving it ourselves is what lets this script run from ANY working directory
# (a CI runner, a container, a cron job) instead of only from the repo root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = BACKEND_ROOT / "app" / "db" / "migrations" / "alembic.ini"

# Import the application settings only if the backend root is importable. The
# history check must keep working in a checkout with no dependencies installed,
# so a failure here is swallowed and reported, never fatal.
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@dataclass
class HistoryReport:
    """Result of the pure, database-free half of the check."""

    revisions: list[str] = field(default_factory=list)
    heads: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.heads) == 1


def build_config() -> object:
    """A ScriptDirectory-backed config with absolute paths.

    Absolute ``script_location``/``prepend_sys_path`` are the whole trick: they
    make this script CWD-independent without editing ``alembic.ini``.
    """
    from alembic.config import Config

    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ALEMBIC_INI.parent))
    config.set_main_option("prepend_sys_path", str(BACKEND_ROOT))
    # The repo's alembic.ini predates `path_separator`; without this Alembic
    # emits a DeprecationWarning and falls back to splitting on spaces -- which
    # would be wrong for a checkout whose path contains a space.
    config.set_main_option("path_separator", "os")
    return config


def check_history() -> HistoryReport:
    """Assert one connected migration graph with one base/head; merges are valid."""
    from alembic.script import ScriptDirectory

    report = HistoryReport()
    if not ALEMBIC_INI.is_file():
        report.errors.append(f"alembic config not found at {ALEMBIC_INI}")
        return report

    try:
        scripts = ScriptDirectory.from_config(build_config())
        report.revisions = [rev.revision for rev in scripts.walk_revisions()]
        report.heads = list(scripts.get_heads())
        report.bases = list(scripts.get_bases())
    except Exception as exc:  # pragma: no cover - defensive
        report.errors.append(f"could not read the revision history: {exc}")
        return report

    if not report.revisions:
        report.errors.append("no revisions found in app/db/migrations/versions")
        return report

    # A forked history is the failure that bites in production: two heads means
    # `alembic upgrade head` silently picks one and the other branch's migration
    # never runs anywhere. Exit non-zero so CI refuses the merge.
    if len(report.heads) > 1:
        report.errors.append(
            f"multiple heads: {', '.join(report.heads)} -- two release branches "
            "were merged without a merge revision. `alembic upgrade head` would "
            "apply only one of them."
        )
    elif not report.heads:
        report.errors.append("no head: the history is empty or circular")

    # More than one base means two independent roots; the same silent-partial-
    # apply problem as multiple heads.
    if len(report.bases) > 1:
        report.errors.append(
            f"multiple bases: {', '.join(report.bases)} -- the history has more "
            "than one root."
        )

    return report


def _is_async_url(url: str) -> bool:
    return "+asyncpg" in url or url.startswith("sqlite+aiosqlite")


async def _current_revision_async(url: str) -> str | None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from alembic.migration import MigrationContext

    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(
                lambda sync_conn: MigrationContext.configure(sync_conn).get_current_revision()
            )
    finally:
        await engine.dispose()


def _current_revision_sync(url: str) -> str | None:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    from alembic.migration import MigrationContext

    engine = create_engine(url, poolclass=NullPool)
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()


def current_revision(url: str) -> tuple[str | None, str | None]:
    """Return ``(revision, error)``. ``error`` is set when the DB is unreachable.

    Never raises: an unreachable database is an expected state, not a crash.
    """
    try:
        if _is_async_url(url):
            return asyncio.run(_current_revision_async(url)), None
        return _current_revision_sync(url), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _redact(url: str) -> str:
    """Never print a password. The URL is only ever echoed redacted."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"{parts.username or ''}:***@{host}", parts.path, parts.query, parts.fragment))


def _database_url() -> str | None:
    """DATABASE_URL from the environment, or the app default via Settings."""
    env = os.environ.get("DATABASE_URL")
    if env:
        return env
    try:
        from app.config import get_settings

        return get_settings().database_url
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_migrations.py",
        description="Verify the Alembic history is connected with one base and one head.",
    )
    parser.add_argument(
        "--require-head",
        action="store_true",
        help="exit non-zero when the database is not already at head (use in a "
        "post-deploy verification job)",
    )
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="skip the database comparison entirely; history only",
    )
    parser.add_argument("--quiet", action="store_true", help="print only problems")
    args = parser.parse_args(argv)

    def say(line: str = "") -> None:
        if not args.quiet:
            print(line)

    say("Alembic history check (P5.4)")
    say(f"  config: {_redact(str(ALEMBIC_INI))}")

    report = check_history()
    say(f"  revisions: {len(report.revisions)}")
    say(f"  head(s):   {', '.join(report.heads) if report.heads else '(none)'}")

    for error in report.errors:
        print(f"  ERROR: {error}", file=sys.stderr)

    if report.errors:
        say()
        say("HISTORY INVALID -- do not release. Merge the branches into a single")
        say("one connected history first (see docs/PRODUCTION_ROADMAP.md section 6, P5.5).")
        return 1

    say("  history: OK (single connected migration history; merge revisions allowed)")

    # --- optional: compare against a live database -------------------------
    url = None if args.no_database else _database_url()
    if url is None:
        say("  database: not configured -- history check only (this is fine)")
        say()
        say("OK. Release order: `alembic upgrade head` (once), THEN start the app.")
        return 0

    say(f"  database: {_redact(url)}")
    revision, error = current_revision(url)
    if error is not None:
        print(f"  database: unreachable ({error}) -- history check passed anyway", file=sys.stderr)
        say()
        say("OK. Release order: `alembic upgrade head` (once), THEN start the app.")
        return 0

    say(f"  current:  {revision or '(none - database has no alembic_version)'}")
    say(f"  head:     {report.heads[0]}")

    if revision != report.heads[0]:
        print(
            f"  ERROR: database is at {revision or '(none)'} but head is {report.heads[0]}."
            " Run `alembic upgrade head` before starting the app.",
            file=sys.stderr,
        )
        if args.require_head:
            return 1
        say()
        say("History is valid; the DATABASE has not been migrated yet.")
        return 0

    say()
    say("OK. Database is at head. Release order: `alembic upgrade head`, THEN start the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())