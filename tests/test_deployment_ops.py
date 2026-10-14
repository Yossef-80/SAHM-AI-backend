"""P5 -- deployment and operations: pooler compatibility, migration history,
secret redaction and the backup/restore CLI.

Everything here runs offline against SQLite or against pure functions: no
Postgres, no network, no `pg_dump`. The point is that the *decisions* these
modules encode are assertable without the infrastructure they exist to talk to.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# C (P5.3) -- pooler compatibility
# ---------------------------------------------------------------------------

# The real Supabase pooler from docs/PRODUCTION_ROADMAP.md §3.1, and the
# transaction-mode host the app is expected to be pointed at.
POOLER_URL = (
    "postgresql+asyncpg://postgres.fmvwhrduiwloqravbeqi:s3cr3t"
    "@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"
)
DIRECT_URL = "postgresql+asyncpg://sahm:s3cr3t@db.example.internal:5432/sahm"


def test_pooler_host_is_detected() -> None:
    from app.db.session import is_pooler_url

    assert is_pooler_url(POOLER_URL) is True
    assert is_pooler_url(DIRECT_URL) is False


def test_pooler_detection_only_inspects_the_host() -> None:
    """A password containing a marker must not flip the branch.

    Matching against the raw URL would let a credential decide how we connect.
    """
    from app.db.session import is_pooler_url

    tricky = "postgresql+asyncpg://u:pgbouncer@db.example.internal:5432/sahm"
    assert is_pooler_url(tricky) is False


def test_self_hosted_pgbouncer_is_detected() -> None:
    from app.db.session import is_pooler_url

    assert is_pooler_url("postgresql://u:p@pgbouncer.internal:5432/sahm") is True
    assert is_pooler_url("postgresql://u:p@10.0.0.7:5432/sahm") is False


def test_pooler_disables_asyncpg_statement_cache() -> None:
    """P5.3: transaction-mode pooling + asyncpg's named statement cache = breakage.

    Asserted on the kwargs the helper builds, not on a real connection -- a real
    connection would need a live pooler, which is the thing under test.
    """
    from app.db.session import _engine_kwargs

    kwargs = _engine_kwargs(POOLER_URL, echo=False)

    assert kwargs["connect_args"]["statement_cache_size"] == 0
    # P5.3 also asks for no local overflow on top of a pooler that already
    # multiplexes.
    assert kwargs["max_overflow"] == 0


def test_direct_connection_keeps_default_behaviour() -> None:
    """No pooler, no statement_cache_size override, default pool sizing."""
    from app.db.session import _engine_kwargs

    kwargs = _engine_kwargs(DIRECT_URL, echo=False)

    assert "connect_args" not in kwargs, "a direct connection must be left alone"
    assert kwargs["pool_size"] == 10
    assert kwargs["max_overflow"] == 20


def test_pre_ping_stays_on_for_both_branches() -> None:
    """A pooler restart or a rotated DB password kills pooled connections."""
    from app.db.session import _engine_kwargs

    assert _engine_kwargs(POOLER_URL, echo=False)["pool_pre_ping"] is True
    assert _engine_kwargs(DIRECT_URL, echo=False)["pool_pre_ping"] is True


def test_sqlite_branch_is_untouched() -> None:
    from app.db.session import _engine_kwargs

    kwargs = _engine_kwargs("sqlite+aiosqlite:///./sahm.db", echo=False)
    assert kwargs["connect_args"] == {"check_same_thread": False}
    assert "pool_size" not in kwargs


def test_pgvector_path_is_a_pooled_pooler_in_compose() -> None:
    """The local stack must exercise the pgvector path, not SQLite (P5.1)."""
    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "pgvector/pgvector:pg16" in compose
    assert "postgresql+asyncpg://" in compose
    assert "sqlite" not in compose


# ---------------------------------------------------------------------------
# D (P5.4 / P5.5) -- migration history gate
# ---------------------------------------------------------------------------


def test_migration_history_is_a_single_linear_chain() -> None:
    """Exactly one head. Two heads means `upgrade head` silently skips a branch."""
    sys.path.insert(0, str(BACKEND_ROOT / "scripts"))
    try:
        import check_migrations
    finally:
        sys.path.pop(0)

    report = check_migrations.check_history()

    assert report.errors == [], report.errors
    assert len(report.heads) == 1, f"expected one head, got {report.heads}"
    assert len(report.bases) == 1
    assert report.revisions, "no revisions found"


def test_migrations_ship_as_a_package() -> None:
    """The P5.1 packaging trap, asserted rather than assumed.

    `pyproject.toml` uses `[tool.setuptools.packages.find] include = ["app*"]`,
    which only finds real packages. Without these markers the migration tree is
    dropped from any built distribution.
    """
    migrations = BACKEND_ROOT / "app" / "db" / "migrations"
    assert (migrations / "__init__.py").is_file()
    assert (migrations / "versions" / "__init__.py").is_file()
    assert (migrations / "alembic.ini").is_file()
    assert (migrations / "env.py").is_file()
    assert (migrations / "script.py.mako").is_file()
    assert list((migrations / "versions").glob("*_*.py")), "no revision scripts"


def test_dockerfile_ships_the_migration_tree_from_the_source_tree() -> None:
    dockerfile = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "WORKDIR /srv/app" in dockerfile
    assert "app/db/migrations/" in dockerfile
    assert "USER app" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile
    # Pinned base tag, never :latest -- checked on the FROM lines only, so the
    # `docker build -t sahm-backend:latest .` example in the header is fine.
    from_lines = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert len(from_lines) == 2, from_lines
    assert all("latest" not in line for line in from_lines), from_lines
    assert all("python:3.11-slim-bookworm" in line for line in from_lines), from_lines
    # --reload must never be reachable from the image command.
    cmd_start = dockerfile.index("CMD [")
    cmd_block = dockerfile[cmd_start:]
    assert "--reload" not in cmd_block.split("\n\n")[0]
    # Production server flags (P5.4).
    for flag in ("--proxy-headers", "--forwarded-allow-ips", "--workers"):
        assert flag in cmd_block


def test_dockerignore_excludes_the_vendor_and_the_secrets() -> None:
    ignore = (BACKEND_ROOT / ".dockerignore").read_text(encoding="utf-8")
    entries = {
        line.strip()
        for line in ignore.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for required in (
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "*.egg-info",
        "sahm.db*",
        ".env",
        "var/",
        ".git",
    ):
        assert required in entries, f"{required} missing from .dockerignore"


def test_production_boot_never_issues_ddl() -> None:
    """The ordering rule, stated as a test.

    Release order is `alembic upgrade head` THEN start the app; `create_all()`
    is a dev convenience and must be unreachable when ENVIRONMENT=prod.
    """
    import app.main as main
    from app.config import get_settings
    from tests.test_workspace_isolation import _TEST_FERNET_KEY  # noqa: PLC0415

    issued: list[str] = []

    async def _spy() -> None:  # pragma: no cover - must not run in prod
        issued.append("create_all")

    settings = get_settings()
    original_env, original_key = settings.environment, settings.token_encryption_key
    original_create_all = main.create_all
    try:
        main.create_all = _spy  # type: ignore[assignment]
        settings.environment = "prod"
        settings.token_encryption_key = _TEST_FERNET_KEY

        import asyncio

        async def _boot() -> None:
            async with main.lifespan(main.app):
                pass

        asyncio.run(_boot())
    finally:
        main.create_all = original_create_all  # type: ignore[assignment]
        settings.environment = original_env
        settings.token_encryption_key = original_key

    assert issued == [], "production boot must not issue DDL"


# ---------------------------------------------------------------------------
# G -- health endpoint alignment
# ---------------------------------------------------------------------------


def _dockerfile_liveness_endpoint() -> str:
    import re

    dockerfile = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"urlopen\('(http://[^']+/health[^']*)'", dockerfile)
    assert match, "Dockerfile HEALTHCHECK does not probe a /health URL"
    return match.group(1)


def _compose_liveness_endpoint() -> str:
    import re

    compose = (BACKEND_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    match = re.search(r"urlopen\('(http://[^']+/health[^']*)'", compose)
    assert match, "compose api.healthcheck does not probe a /health URL"
    return match.group(1)


def test_all_three_probes_agree_on_one_endpoint() -> None:
    """Dockerfile HEALTHCHECK == compose healthcheck == the documented endpoint.

    P5 brief G: they must agree, and ``/health`` won -- it is the dependency-free
    liveness endpoint. ``/health/ready`` is the readiness gate and deliberately is
    NOT a restart-on-failure probe.
    """
    docs_file = BACKEND_ROOT.parent / "docs" / "RUNNING_BACKEND.md"
    if not docs_file.exists():
        docs_file = BACKEND_ROOT / "docs" / "RUNNING_BACKEND.md"
    if not docs_file.exists():
        pytest.skip("RUNNING_BACKEND.md does not exist in workspace")
    docs = docs_file.read_text(encoding="utf-8")

    docker_endpoint = _dockerfile_liveness_endpoint()
    compose_endpoint = _compose_liveness_endpoint()

    assert docker_endpoint == compose_endpoint
    assert docker_endpoint.endswith("/health"), "liveness must be /health, not /health/ready"
    # Section 2 of RUNNING_BACKEND.md documents the exact liveness curl probe.
    assert "curl http://127.0.0.1:8000/health" in docs
    # ...and the doc states which endpoint the container probes.
    assert "`Dockerfile` `HEALTHCHECK`" in docs
    assert "`/health/ready`" in docs, "readiness must be documented too"


# ---------------------------------------------------------------------------
# E (P5.5) -- redaction audit
# ---------------------------------------------------------------------------

#: The secret-bearing columns introduced across P0-P4. Listed explicitly as well
#: as being discovered by name, so DELETING one of these is a test failure rather
#: than a silent pass -- a rename that quietly drops a column from the audit.
EXPECTED_SECRET_COLUMNS: set[tuple[str, str]] = {
    ("integrations", "access_token_enc"),
    ("integrations", "refresh_token_enc"),
    ("users", "password_hash"),
    ("sessions", "token_hash"),
    # tool_settings: the six encrypted connector secrets (P2.4)
    ("tool_settings", "meta_app_secret_enc"),
    ("tool_settings", "meta_access_token_enc"),
    ("tool_settings", "google_places_api_key_enc"),
    ("tool_settings", "dataforseo_password_enc"),
    ("tool_settings", "image_api_key_enc"),
    ("tool_settings", "search_api_key_enc"),
    # llm_provider_settings: encrypted provider secrets
    ("llm_provider_settings", "fast_api_key_enc"),
    ("llm_provider_settings", "strong_api_key_enc"),
    ("llm_provider_settings", "fast_extra_headers_enc"),
    ("llm_provider_settings", "strong_extra_headers_enc"),
}

#: Naming rule that decides whether a column is secret-bearing.
SECRET_NAME_SUFFIXES = ("_enc", "_hash", "_secret", "_password", "_key")
SECRET_NAME_EXACT = frozenset({"token", "api_key", "secret", "password"})


def _is_secret_bearing(name: str) -> bool:
    lowered = name.lower()
    return lowered in SECRET_NAME_EXACT or lowered.endswith(SECRET_NAME_SUFFIXES)


def _walk_columns():
    from app.db.models import Base

    for table in Base.metadata.tables.values():
        for column in table.columns:
            yield table.name, column.name


def test_every_secret_bearing_column_is_redacted() -> None:
    """Walk every mapped table: no secret-bearing column may be loggable.

    This is the audit the P5.5 brief asks for, encoded so it cannot rot: adding a
    ``*_enc`` column without adding its name to ``_SENSITIVE_KEYS`` fails here.
    """
    from app.core.logging import redact

    found: set[tuple[str, str]] = {
        (table, name) for table, name in _walk_columns() if _is_secret_bearing(name)
    }

    assert found == EXPECTED_SECRET_COLUMNS, (
        "the secret-bearing column set changed; "
        f"added={sorted(found - EXPECTED_SECRET_COLUMNS)} "
        f"removed={sorted(EXPECTED_SECRET_COLUMNS - found)}"
    )

    for table, name in sorted(found):
        rendered = redact({name: "super-secret-value"})
        assert rendered[name] == "***REDACTED***", f"{table}.{name} would reach a log line"


def test_redaction_is_case_insensitive_and_recurses() -> None:
    from app.core.logging import redact

    payload = {
        "Password_Hash": "x",
        "tools": [{"meta_access_token_enc": "y"}, {"safe": "keep"}],
        "tuple_value": ({"token_hash": "z"},),
    }
    out = redact(payload)

    assert out["Password_Hash"] == "***REDACTED***"
    assert out["tools"][0]["meta_access_token_enc"] == "***REDACTED***"
    assert out["tools"][1]["safe"] == "keep"
    assert out["tuple_value"][0]["token_hash"] == "***REDACTED***"


#: Tight naming rule: matches credential-bearing settings fields and nothing
#: else. Deliberately narrow -- a loose rule ("contains 'token'") also catches
#: `daily_token_budget` and `cors_allow_headers`, which are not secrets and must
#: stay readable in a log line.
SECRET_SETTING_SUFFIXES = (
    "_api_key",
    "_secret",
    "_secret_key",
    "_password",
    "_access_token",
    "_encryption_key",
    "_extra_headers",
)

#: Secrets the naming rule above cannot see, listed explicitly.
EXPECTED_SECRET_SETTINGS = {"supabase_publishable_key"}


def test_plaintext_provider_keys_are_redacted() -> None:
    """P5.5: every secret-bearing env var in `app/config.py` must be covered."""
    from app.config import Settings
    from app.core.logging import redact

    discovered = {
        name
        for name in Settings.model_fields
        if name.endswith(SECRET_SETTING_SUFFIXES) or name in EXPECTED_SECRET_SETTINGS
    }
    assert discovered, "the naming heuristic found nothing -- update this test"

    # The audit itself. Adding a provider key without a redaction entry fails here.
    unredacted = {
        field for field in discovered if redact({field: "v"})[field] != "***REDACTED***"
    }
    assert unredacted == set(), f"these settings are not redacted: {sorted(unredacted)}"

    # The rule still finds the well-known ones.
    assert {"anthropic_api_key", "openai_api_key", "token_encryption_key"} <= discovered


# ---------------------------------------------------------------------------
# F (P5.7) -- backup / restore CLI
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_admin():
    """Import ``scripts/db_admin.py`` as a module (it is not a package)."""
    import importlib.util

    path = BACKEND_ROOT / "scripts" / "db_admin.py"
    spec = importlib.util.spec_from_file_location("db_admin_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves its own annotations through
    # sys.modules[cls.__module__], which is None until it is registered.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


PG_URL_WITH_PASSWORD = "postgresql+asyncpg://sahm:sup3rs3cret@db.internal:5432/sahm"


def test_backup_argument_parsing(db_admin) -> None:
    args = db_admin.build_parser().parse_args(["backup"])
    assert args.command == "backup"
    assert args.schema_only is False
    assert args.dry_run is False
    assert args.output is None
    assert args.directory is None
    # --force belongs to restore only: overwriting is a restore-time decision.
    assert not hasattr(args, "force")

    args = db_admin.build_parser().parse_args(
        ["backup", "--schema-only", "--dry-run", "--output", "x.dump"]
    )
    assert args.schema_only is True
    assert args.dry_run is True
    assert args.output == "x.dump"


def test_restore_argument_parsing(db_admin) -> None:
    args = db_admin.build_parser().parse_args(["restore", "sahm-20261005.dump"])
    assert args.command == "restore"
    assert args.dump == "sahm-20261005.dump"
    assert args.force is False

    args = db_admin.build_parser().parse_args(
        ["restore", "--force", "--dry-run", "sahm-20261005.dump"]
    )
    assert args.force is True
    assert args.dry_run is True


def test_backup_output_is_timestamped(db_admin, tmp_path: Path) -> None:
    first = db_admin.default_output(schema_only=False, directory=str(tmp_path))
    assert first.parent == tmp_path
    assert first.suffix == ".dump"
    assert first.name.startswith("sahm-")
    assert first.name.endswith(".dump")

    schema = db_admin.default_output(schema_only=True, directory=str(tmp_path))
    assert schema.suffix == ".sql"


def test_password_never_reaches_argv(db_admin, monkeypatch) -> None:
    """argv is world-readable through `ps`; PGPASSWORD is not."""
    monkeypatch.setattr(db_admin, "require_binary", lambda name: f"/usr/bin/{name}")

    inv = db_admin.build_backup(PG_URL_WITH_PASSWORD, Path("out.dump"))

    assert "sup3rs3cret" not in " ".join(inv.argv), "password leaked into argv"
    assert "sup3rs3cret" not in inv.printable
    assert inv.env["PGPASSWORD"] == "sup3rs3cret"


def test_restore_command_strips_the_password(db_admin, monkeypatch) -> None:
    monkeypatch.setattr(db_admin, "require_binary", lambda name: f"/usr/bin/{name}")

    inv = db_admin.build_restore(PG_URL_WITH_PASSWORD, Path("out.dump"))

    assert "sup3rs3cret" not in " ".join(inv.argv)
    assert inv.env["PGPASSWORD"] == "sup3rs3cret"
    assert "--exit-on-error" in inv.argv


def test_printable_urls_are_redacted(db_admin) -> None:
    rendered = db_admin.redact_url(PG_URL_WITH_PASSWORD)
    assert "sup3rs3cret" not in rendered
    assert rendered == "postgresql+asyncpg://sahm:***@db.internal:5432/sahm"


def test_sqlite_url_is_refused_with_an_actionable_message(db_admin) -> None:
    with pytest.raises(db_admin.DbAdminError) as excinfo:
        db_admin.to_pg_url("sqlite+aiosqlite:///./sahm.db")
    assert "SQLite" in str(excinfo.value)


def test_asyncpg_scheme_is_stripped_for_pg_tools(db_admin) -> None:
    """pg_dump knows `postgresql://` only; `+asyncpg` is a SQLAlchemy dialect."""
    assert db_admin.to_pg_url(PG_URL_WITH_PASSWORD) == (
        "postgresql://sahm:sup3rs3cret@db.internal:5432/sahm"
    )


def test_missing_pg_dump_fails_loudly(db_admin, monkeypatch) -> None:
    """pg_dump ships with PostgreSQL, not with Python -- say so, with the fix."""
    monkeypatch.setattr(db_admin.shutil, "which", lambda name: None)

    with pytest.raises(db_admin.DbAdminError) as excinfo:
        db_admin.require_binary("pg_dump")

    message = str(excinfo.value)
    assert "not on PATH" in message
    assert "postgresql-client" in message
    assert "PostgreSQL" in message


def test_restore_refuses_a_non_empty_database(db_admin, monkeypatch, tmp_path: Path) -> None:
    """The refusal path, with no live database."""
    monkeypatch.setattr(db_admin, "require_binary", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(db_admin, "user_table_count", lambda url: 17)

    with pytest.raises(db_admin.DbAdminError) as excinfo:
        db_admin.ensure_restore_target_is_empty(
            PG_URL_WITH_PASSWORD, force=False, dump_path=tmp_path / "d.dump"
        )

    message = str(excinfo.value)
    assert "REFUSING to restore" in message
    assert "17" in message
    assert "--force" in message
    assert "sup3rs3cret" not in message, "the refusal message leaked the password"


def test_force_overrides_the_emptiness_refusal(db_admin, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(db_admin, "require_binary", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(db_admin, "user_table_count", lambda url: 17)

    count = db_admin.ensure_restore_target_is_empty(
        PG_URL_WITH_PASSWORD, force=True, dump_path=tmp_path / "d.dump"
    )
    assert count == 17


def test_emptiness_is_a_query_not_output_parsing(db_admin) -> None:
    assert "information_schema" in db_admin.EMPTINESS_SQL
    assert "public" in db_admin.EMPTINESS_SQL


def test_dry_run_executes_nothing(db_admin, monkeypatch, capsys, tmp_path: Path) -> None:
    """--dry-run must not shell out at all -- not even for a harmless command."""
    monkeypatch.setattr(db_admin, "require_binary", lambda name: f"/usr/bin/{name}")

    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("dry run executed a subprocess")

    monkeypatch.setattr(db_admin.subprocess, "run", _boom)
    monkeypatch.setenv("DATABASE_URL", PG_URL_WITH_PASSWORD)

    assert db_admin.main(["backup", "--dry-run"]) == 0
    assert db_admin.main(["restore", "--dry-run", str(tmp_path / "d.dump")]) == 0

    out = capsys.readouterr().out
    assert "dry run" in out
    assert "pg_dump" in out and "pg_restore" in out
    assert "sup3rs3cret" not in out, "dry-run output leaked the password"


def test_check_migrations_runs_standalone() -> None:
    """It must work with no database configured; the history check is pure files."""
    import os

    env = dict(os.environ)
    env.pop("DATABASE_URL", None)  # "no database configured" is a valid state
    result = subprocess.run(
        [sys.executable, str(BACKEND_ROOT / "scripts" / "check_migrations.py"), "--no-database"],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "single connected migration history" in result.stdout