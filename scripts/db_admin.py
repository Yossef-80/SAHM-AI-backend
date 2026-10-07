#!/usr/bin/env python
r"""Backup and restore for the Sahm Postgres database (P5.7).

One place, testable, no shell scripts. Everything customer-visible -- brand
memory, campaigns, encrypted Meta tokens, users, sessions -- lives in this one
database, and an untested backup is not a backup.

Usage
-----
    python scripts/db_admin.py backup
    python scripts/db_admin.py backup --schema-only
    python scripts/db_admin.py backup --output var/backups/sahm.dump
    python scripts/db_admin.py backup --dry-run

    python scripts/db_admin.py restore var/backups/sahm-20261005-120000.dump
    python scripts/db_admin.py restore --database-url postgresql://... FILE
    python scripts/db_admin.py restore --force FILE     # non-empty database

CRON EXAMPLE
------------
Daily dump at 03:17, 14 days of retention, stderr mailed to the on-call::

    17 3 * * *  cd "/opt/sahm/sahm-ai backend" && \
        PGPASSWORD="$(cat /run/secrets/db_password)" \
        /opt/venv/bin/python scripts/db_admin.py backup \
            --output /var/backups/sahm/sahm-$(date +\%Y\%m\%d-\%H\%M\%S).dump \
        >> /var/log/sahm/backup.log 2>&1

Nightly `pg_dump --schema-only` for the drift diff::

    41 4 * * 0  cd "/opt/sahm/sahm-ai backend" && \
        /opt/venv/bin/python scripts/db_admin.py backup --schema-only \
            --output /var/backups/sahm/schema-$(date +\%Y\%m\%d).sql

After a restore, finish the job::

    python -m alembic -c app/db/migrations/alembic.ini upgrade head
    curl -fsS http://localhost:8000/health/ready

The `vector` extension is included in a custom-format dump (``pg_dump`` emits
``CREATE EXTENSION`` for it), so a restored database already has pgvector.

SECURITY
--------
The database password is **never** placed on a command line -- argv is visible to
every user on the host via `ps`. It is read from the parsed ``DATABASE_URL`` and
exported to the child process as ``PGPASSWORD``, which is the libpq-sanctioned
mechanism. Every URL this script prints has the password replaced with ``***``,
including in ``--dry-run`` output. ``DATABASE_URL`` is read from the environment
(``.env`` is never parsed here: this script must never read a secret file).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

BACKEND_ROOT = Path(__file__).resolve().parents[1]

#: Bumped whenever the dump/restore flags change shape, so a dump written by one
#: version can never be silently restored by an incompatible one.
DUMP_SUFFIX = ".dump"
SCHEMA_SUFFIX = ".sql"

#: "Is this database empty?" is answered with a real query, never by parsing
#: pg_restore's output. `public` is where every Sahm table lands, and
#: `alembic_version` is included in the count so a database that has been
#: migrated is never mistaken for a blank slate.
EMPTINESS_SQL = (
    "SELECT count(*) FROM information_schema.tables "
    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
)


class DbAdminError(RuntimeError):
    """Anything the operator has to act on, with the action already spelled out."""


# --------------------------------------------------------------------- utils --


def redact_url(url: str) -> str:
    """A printable URL with the password replaced by ``***``. Never log the real one."""
    parts = urlsplit(url)
    if not parts.password:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user = f"{parts.username}:" if parts.username else ""
    return urlunsplit((parts.scheme, f"{user}***@{host}", parts.path, parts.query, parts.fragment))


def split_credentials(url: str) -> tuple[str, str | None]:
    """Return ``(pg_url_without_password, password)``.

    The returned URL is safe to place on a command line: the password has been
    stripped, and the caller is expected to hand it to the child via ``PGPASSWORD``.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    netloc = f"{user}@{host}" if user else host
    return (
        urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)),
        parts.password,
    )


def to_pg_url(url: str) -> str:
    """Strip the SQLAlchemy driver suffix.

    ``postgresql+asyncpg://...`` is meaningless to ``pg_dump``: it only knows the
    ``postgresql`` scheme (and uses libpq, which is why PGPASSWORD works). This
    is also the check that the operator pointed us at Postgres and not SQLite.
    """
    if url.startswith("sqlite"):
        raise DbAdminError(
            "DATABASE_URL points at SQLite. Alembic and this script are "
            "Postgres-only -- there is no SQLite production database to back up. "
            "Set DATABASE_URL to a postgresql:// URL."
        )
    parts = urlsplit(url)
    scheme = parts.scheme
    if "+" in scheme:
        scheme = scheme.split("+", 1)[0]
    if scheme not in {"postgres", "postgresql"}:
        raise DbAdminError(f"Unsupported DATABASE_URL scheme {parts.scheme!r}; expected postgresql.")
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def require_binary(name: str) -> str:
    """Resolve an executable, or fail with the fix in the message.

    ``pg_dump`` / ``pg_restore`` ship with **PostgreSQL**, not with Python. On a
    machine that only ever ran `pip install`, they are simply absent, and the
    resulting ``FileNotFoundError`` from ``subprocess`` says nothing useful.
    """
    path = shutil.which(name)
    if path is None:
        raise DbAdminError(
            f"'{name}' is not on PATH.\n"
            f"  It ships with PostgreSQL (the client tools), NOT with Python.\n"
            f"  Windows : install PostgreSQL and add <install>\\bin to PATH.\n"
            f"  Debian  : apt-get install -y postgresql-client\n"
            f"  Alpine  : apk add --no-cache postgresql-client\n"
            f"  macOS   : brew install libpq  (or `brew install postgresql@16`)\n"
            f"  Docker  : run this script inside a postgresql-client container,\n"
            f"            e.g. docker run --rm -v \"$PWD:/w\" -w /w postgres:16 \\\n"
            f"              python scripts/db_admin.py backup --dry-run\n"
            f"  Verify  : pg_dump --version"
        )
    return path


def resolve_database_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise DbAdminError(
            "No database URL.\n"
            "  Pass --database-url, or set DATABASE_URL in the environment.\n"
            "  This script never reads .env -- it must not be the thing that\n"
            "  pulls a secret file into memory."
        )
    return url


def default_output(schema_only: bool, directory: str | None) -> Path:
    """A timestamped filename so two dumps in one day never collide."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = SCHEMA_SUFFIX if schema_only else DUMP_SUFFIX
    folder = Path(directory) if directory else BACKEND_ROOT / "var" / "backups"
    return folder / f"sahm-{stamp}{suffix}"


# ------------------------------------------------------------ command build --


@dataclass(frozen=True)
class Invocation:
    """A subprocess to run: argv (password-free) plus the env it needs."""

    argv: list[str]
    env: dict[str, str]
    printable: str


def _child_env(password: str | None) -> dict[str, str]:
    env = dict(os.environ)
    # The password goes here and ONLY here. Never in argv: `ps` shows the world
    # every argument of every process on the box.
    if password:
        env["PGPASSWORD"] = password
    return env


def build_backup(url: str, output: Path, schema_only: bool = False) -> Invocation:
    pg_url, password = split_credentials(to_pg_url(url))
    argv = [require_binary("pg_dump")]
    if schema_only:
        # Plain SQL, not custom format: the point of a schema-only dump is that a
        # human (or a diff tool) can read it. pg_restore cannot consume it --
        # restore expects the custom format produced below.
        argv += ["--schema-only", "--no-owner", "--no-privileges", "-f", str(output), pg_url]
    else:
        # -Fc custom format: compressed, and restorable with pg_restore.
        argv += ["--format=custom", "--no-owner", "--no-privileges", "-f", str(output), pg_url]
    return Invocation(argv=argv, env=_child_env(password), printable=render_command(argv))


def build_restore(url: str, dump_path: Path) -> Invocation:
    pg_url, password = split_credentials(to_pg_url(url))
    argv = [
        require_binary("pg_restore"),
        "--dbname",
        pg_url,
        # Fail on the first error instead of plowing on and leaving a half-loaded
        # database that looks fine until someone queries a missing table.
        "--exit-on-error",
        "--no-owner",
        "--no-privileges",
        str(dump_path),
    ]
    return Invocation(argv=argv, env=_child_env(password), printable=render_command(argv))


def render_command(argv: list[str]) -> str:
    """Render an argv for humans. Safe: the password is never in it."""
    return " ".join(argv)


# ------------------------------------------------------- emptiness checking --


async def _user_table_count_async(url: str) -> int:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    pg_url = to_pg_url(url)
    engine = create_async_engine(_as_async_url(pg_url), poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(EMPTINESS_SQL))
            return int(result.scalar_one())
    finally:
        await engine.dispose()


def user_table_count(url: str) -> int:
    """How many base tables live in ``public``. 0 means "safe to restore into".

    Synchronous on purpose for the caller-facing contract: the emptiness question
    must be answered by a real query against the catalogue, never by parsing
    ``pg_restore`` output, which is only available *after* the damage is done.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool

    pg_url = to_pg_url(url)
    # asyncpg has no synchronous entry point, so for an async URL the async engine
    # is used under asyncio.run(). A synchronous driver (psycopg2) is used as-is.
    if "+asyncpg" in url:
        return asyncio.run(_user_table_count_async(url))
    engine = create_engine(pg_url, poolclass=NullPool)
    try:
        with engine.connect() as conn:
            return int(conn.execute(text(EMPTINESS_SQL)).scalar_one())
    finally:
        engine.dispose()


def _as_async_url(url: str) -> str:
    return url if "+asyncpg" in url else url.replace("postgresql://", "postgresql+asyncpg://", 1)


def ensure_restore_target_is_empty(url: str, force: bool, dump_path: Path) -> int:
    """Refuse to restore over a populated database unless ``--force``.

    ``pg_restore`` would happily overwrite rows. Restoring a production-shaped
    dump into a live database is not a mistake this script should allow silently.
    Returns the table count (0 when it proceeded).
    """
    require_binary("pg_restore")
    count = user_table_count(url)
    if count and not force:
        raise DbAdminError(
            f"REFUSING to restore: {redact_url(url)} already contains {count} "
            f"base table(s) in schema 'public'.\n"
            f"  A restore would overwrite live rows.\n"
            f"  Options:\n"
            f"    - point --database-url at an EMPTY database (the drill), or\n"
            f"    - drop the existing schema first, or\n"
            f"    - re-run with --force if you are certain ({dump_path.name})."
        )
    return count


# ----------------------------------------------------------------- commands --


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DbAdminError(f"Cannot create output directory {path.parent}: {exc}") from exc


def run_backup(args: argparse.Namespace) -> int:
    url = resolve_database_url(args.database_url)
    pg_url = to_pg_url(url)  # validates the scheme before anything is created
    output = Path(args.output) if args.output else default_output(args.schema_only, args.directory)
    inv = build_backup(pg_url, output, schema_only=args.schema_only)

    print(f"backup -> {output}")
    print(f"  source: {redact_url(pg_url)}")
    print(f"  mode:   {'schema only' if args.schema_only else 'full (custom format)'}")
    print(f"  command: {inv.printable}")
    print("  PGPASSWORD: set from DATABASE_URL, never on the command line")

    if args.dry_run:
        print("  (dry run -- nothing executed)")
        return 0

    _ensure_parent(output)
    result = subprocess.run(inv.argv, env=inv.env, capture_output=True, text=True)
    if result.returncode != 0:
        raise DbAdminError(
            f"pg_dump failed (exit {result.returncode}).\n"
            f"  {result.stderr.strip() or result.stdout.strip()}\n"
            f"  Check the host, port and credentials in DATABASE_URL."
        )
    size = output.stat().st_size if output.exists() else 0
    print(f"  wrote {output} ({size:,} bytes)")
    if not args.schema_only:
        print("  rehearse it: python scripts/db_admin.py restore --database-url <empty-db> " + str(output))
    return 0


def run_restore(args: argparse.Namespace) -> int:
    url = resolve_database_url(args.database_url)
    pg_url = to_pg_url(url)
    dump_path = Path(args.dump)
    if not args.dry_run and not dump_path.is_file():
        raise DbAdminError(f"Dump file not found: {dump_path}")

    print(f"restore <- {dump_path}")
    print(f"  target: {redact_url(pg_url)}")

    if args.dry_run:
        print("  (dry run -- emptiness check skipped, nothing executed)")
        print(f"  command: {build_restore(pg_url, dump_path).printable}")
        return 0

    count = ensure_restore_target_is_empty(pg_url, args.force, dump_path)
    if count:
        print(f"  target has {count} table(s); continuing because --force was passed")

    inv = build_restore(pg_url, dump_path)
    print(f"  command: {inv.printable}")
    result = subprocess.run(inv.argv, env=inv.env, capture_output=True, text=True)
    if result.returncode != 0:
        raise DbAdminError(
            f"pg_restore failed (exit {result.returncode}).\n"
            f"  {result.stderr.strip() or result.stdout.strip()}\n"
            f"  The database may be partially loaded -- inspect before retrying."
        )
    print("  restore complete")
    print("  next: python -m alembic -c app/db/migrations/alembic.ini upgrade head")
    print("        curl -fsS http://localhost:8000/health/ready")
    return 0


# --------------------------------------------------------------------- cli --


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="db_admin.py",
        description="Back up and restore the Sahm Postgres database (pg_dump / pg_restore).",
        epilog="Run `db_admin.py <command> --help` for per-command options.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--database-url",
            default=None,
            help="Postgres URL. Defaults to the DATABASE_URL environment variable.",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="print the plan (and the exact command) without executing anything",
        )

    backup = sub.add_parser("backup", help="write a timestamped dump")
    add_common(backup)
    backup.add_argument(
        "--schema-only",
        action="store_true",
        help="schema only (plain SQL) -- for drift diffs, not for restorable backups",
    )
    backup.add_argument("--output", default=None, help="output path (default: var/backups/sahm-<UTC>.dump)")
    backup.add_argument("--directory", default=None, help="output directory (default: var/backups)")
    backup.set_defaults(func=run_backup)

    restore = sub.add_parser("restore", help="restore a dump into an EMPTY database")
    add_common(restore)
    restore.add_argument("dump", help="path to the dump produced by `backup`")
    restore.add_argument(
        "--force",
        action="store_true",
        help="allow restoring into a database that already has tables",
    )
    restore.set_defaults(func=run_restore)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DbAdminError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())