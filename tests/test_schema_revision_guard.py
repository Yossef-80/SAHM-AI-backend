"""A database behind the code must say so, not drop rows quietly.

A live deployment ran code that writes ``usage_records.action`` against a
database that had never had migration ``c1f4a9d27e35`` applied: every meter
flush failed with ``column "action" of relation "usage_records" does not exist``
and the row was dropped, so the token accounting behind the daily budget guard
was silently incomplete. The schema comparison is now part of boot and of
``/health/ready``, and ``behind`` refuses to take traffic -- the documented
release order is ``alembic upgrade head`` *then* the app.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db.schema_state import (
    check_schema_state,
    current_revision,
    known_revisions,
    migration_head,
)

#: The revision that added ``usage_records.action`` and ``estimated``.
USAGE_COLUMNS_REVISION = "c1f4a9d27e35"


async def _stamp(db_engine: object, revision: str | None) -> None:
    """Give the test database an ``alembic_version`` row, as a real upgrade would."""
    async with db_engine.begin() as conn:  # type: ignore[attr-defined]
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS alembic_version "
                "(version_num VARCHAR(32) NOT NULL)"
            )
        )
        await conn.execute(text("DELETE FROM alembic_version"))
        if revision is not None:
            await conn.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                {"rev": revision},
            )


def test_the_migration_that_adds_the_action_column_is_in_the_chain() -> None:
    """The fix for the dropped usage rows is a migration that already exists."""
    assert USAGE_COLUMNS_REVISION in known_revisions()


def test_head_is_readable_without_a_database() -> None:
    head = migration_head()
    assert head, "the migration scripts must be readable from the installed package"
    assert head == "b3d51c7a9f02"


async def test_a_fresh_sqlite_schema_is_reported_as_unmanaged(db_engine: object) -> None:
    """Dev/test databases are built by ``create_all``; that is not drift."""
    state = await check_schema_state(db_engine)
    assert state.state == "unmanaged"
    assert state.current is None
    assert state.head == migration_head()


async def test_a_database_behind_head_is_reported_as_behind(
    db_engine: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _stamp(db_engine, USAGE_COLUMNS_REVISION)
    state = await check_schema_state(db_engine)

    assert state.state == "behind"
    assert state.current == USAGE_COLUMNS_REVISION
    assert state.head == migration_head()
    # The message has to carry the command, because it is the whole remedy.
    assert "alembic upgrade head" in state.describe()


async def test_a_database_at_head_is_ok(
    db_engine: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _stamp(db_engine, migration_head())
    state = await check_schema_state(db_engine)

    assert state.state == "ok"
    assert state.is_ok
    assert state.describe() == "ok"


async def test_a_revision_this_code_does_not_know_is_ahead(db_engine: object) -> None:
    await _stamp(db_engine, "ffffffffffff")
    state = await check_schema_state(db_engine)

    assert state.state == "ahead"
    assert not state.is_behind


async def test_readiness_refuses_traffic_while_the_schema_is_behind(
    api_client: object, db_engine: object
) -> None:
    """The outage-in-waiting state is a 503 in the standard envelope."""
    await _stamp(db_engine, USAGE_COLUMNS_REVISION)

    resp = await api_client.get("/health/ready")  # type: ignore[attr-defined]

    assert resp.status_code == 503, resp.text  # type: ignore[attr-defined]
    error = resp.json()["error"]  # type: ignore[attr-defined]
    assert error["code"] == "not_ready"
    assert "schema" in error["detail"]["degraded"]
    assert "alembic upgrade head" in error["detail"]["checks"]["schema"]
    # The database itself is fine; the *schema* is the reason for the refusal.
    assert error["detail"]["checks"]["database"] == "ok"


async def test_readiness_is_ready_when_the_database_is_at_head(
    api_client: object, db_engine: object
) -> None:
    await _stamp(db_engine, migration_head())

    resp = await api_client.get("/health/ready")  # type: ignore[attr-defined]

    assert resp.status_code == 200, resp.text  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    assert body["status"] == "ready"
    assert body["checks"]["schema"] == "ok"


async def test_readiness_does_not_blame_the_schema_when_the_database_is_down(
    api_client: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cause, one message: an unreachable DB must not also report drift."""
    import app.main as main

    async def _down() -> tuple[bool, str]:
        return False, "OperationalError: connection refused"

    monkeypatch.setattr(main, "_check_database", _down)
    resp = await api_client.get("/health/ready")  # type: ignore[attr-defined]

    assert resp.status_code == 503  # type: ignore[attr-defined]
    checks = resp.json()["error"]["detail"]["checks"]  # type: ignore[attr-defined]
    assert checks["schema"] == "not checked: database unavailable"
    assert resp.json()["error"]["detail"]["degraded"] == []  # type: ignore[attr-defined]


async def test_current_revision_reads_alembic_version(db_engine: object) -> None:
    await _stamp(db_engine, USAGE_COLUMNS_REVISION)
    revision, error = await current_revision(db_engine)
    assert error is None
    assert revision == USAGE_COLUMNS_REVISION
