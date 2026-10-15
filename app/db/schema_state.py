"""Is the database's schema the one this code was written against?

The release order this project documents is ``alembic upgrade head`` **then**
start the app, and ``scripts/check_migrations.py --require-head`` is the gate
that enforces it. Nothing enforced it *at runtime*, and that gap has a silent
failure mode: a deployment running code that expects a column the database does
not have keeps serving traffic and fails only at the moment it writes.

That is not hypothetical. A live deployment was missing
``usage_records.action`` -- the migration exists and is in the chain, it had
simply never been applied -- so every usage-meter flush died with

    column "action" of relation "usage_records" does not exist

and the meter *dropped the row* (``usage_meter_flush_failed``, ``dropped: 1``).
The product kept working, which is exactly the problem: the token accounting
that the daily budget guard reads was quietly losing data, and the only trace
was one warning line.

This module answers the question without importing the app's settings, so it can
be called at boot and from the readiness probe. States
--------------------------------------------------------------
``ok``         the database is at the revision this code expects.
``behind``     the database is at an older revision -- the app is reading and
               writing a schema the code does not match.
``ahead``      the database has a revision this code does not know about (a
               newer migration, or a diverged history).
``unmanaged``  there is no ``alembic_version`` table at all. Normal for the
               SQLite dev/test databases ``create_all()`` builds; informational
               everywhere else.
``unknown``    the migration scripts could not be read, or the database could
               not be reached to ask.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_ALEMBIC_INI = _MIGRATIONS_DIR / "alembic.ini"

#: Alembic logs "Context impl ..." and "Will assume non-transactional DDL." at
#: INFO every time a context is configured -- once per readiness probe otherwise.
_ALEMBIC_MIGRATION_LOGGER = "alembic.runtime.migration"


@contextmanager
def _quiet_alembic() -> Iterator[None]:
    """Keep Alembic's per-connection chatter out of the request log."""
    logger = logging.getLogger(_ALEMBIC_MIGRATION_LOGGER)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


@dataclass(frozen=True)
class SchemaState:
    """The comparison, in one value a boot log or a readiness body can carry."""

    state: str
    current: str | None
    head: str | None
    detail: str = ""

    @property
    def is_ok(self) -> bool:
        return self.state == "ok"

    @property
    def is_behind(self) -> bool:
        return self.state == "behind"

    def describe(self) -> str:
        if self.state == "ok":
            return "ok"
        if self.state == "behind":
            return (
                f"behind: database is at {self.current}, code expects {self.head} "
                "-- run `alembic upgrade head`"
            )
        if self.state == "ahead":
            return (
                f"ahead: database is at {self.current}, this code knows only up to "
                f"{self.head} -- deploy the matching code, or roll the database back"
            )
        return self.detail or self.state


def migration_head() -> str | None:
    """The revision the migration scripts end at, or ``None`` if unreadable."""
    try:
        from alembic.config import Config as AlembicConfig
        from alembic.script import ScriptDirectory

        config = AlembicConfig(str(_ALEMBIC_INI))
        # Absolute, so the answer does not depend on the process's cwd.
        config.set_main_option("script_location", str(_MIGRATIONS_DIR))
        # Silences Alembic's own deprecation warning about the legacy
        # ``prepend_sys_path`` splitting we never use.
        config.set_main_option("path_separator", "os")
        return ScriptDirectory.from_config(config).get_current_head()
    except Exception:
        return None


def known_revisions() -> set[str]:
    """Every revision reachable from the heads of this code's history."""
    try:
        from alembic.config import Config as AlembicConfig
        from alembic.script import ScriptDirectory

        config = AlembicConfig(str(_ALEMBIC_INI))
        config.set_main_option("script_location", str(_MIGRATIONS_DIR))
        config.set_main_option("path_separator", "os")
        return {
            revision.revision
            for revision in ScriptDirectory.from_config(config).walk_revisions()
        }
    except Exception:
        return set()


def _classify(current: str | None, head: str | None) -> SchemaState:
    if head is None:
        return SchemaState("unknown", current, None, "migration scripts could not be read")
    if current is None:
        return SchemaState(
            "unmanaged",
            None,
            head,
            "no alembic_version table: the schema was created outside Alembic "
            "(SQLite dev/test), so drift cannot be detected",
        )
    if current == head:
        return SchemaState("ok", current, head)
    if current in known_revisions():
        return SchemaState("behind", current, head)
    return SchemaState("ahead", current, head)


async def current_revision(engine: Any | None = None) -> tuple[str | None, str | None]:
    """``(revision, error)`` straight from ``alembic_version``.

    ``MigrationContext`` is Alembic's own way of asking, which keeps this
    identical to what ``alembic upgrade``/``downgrade`` see. Never raises: an
    unreachable database is a state to report, not a crash. ``engine`` defaults
    to the app's own engine; tests pass theirs so the answer comes from the
    database under test.
    """
    try:
        from alembic.migration import MigrationContext

        if engine is None:
            from app.db.session import get_engine

            engine = get_engine()
        async with engine.connect() as conn:
            with _quiet_alembic():
                revision = await conn.run_sync(
                    lambda sync_conn: MigrationContext.configure(sync_conn).get_current_revision()
                )
        return revision, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def check_schema_state(engine: Any | None = None) -> SchemaState:
    """Compare the database's revision against the one this code ships."""
    head = migration_head()
    if head is None:
        return _classify(None, None)
    revision, error = await current_revision(engine)
    if error is not None and revision is None:
        # An unreachable database is already reported by the database check;
        # do not pretend to know anything about its schema.
        return SchemaState("unknown", None, head, f"database not reachable ({error})")
    return _classify(revision, head)
