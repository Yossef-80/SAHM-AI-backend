"""The semantic search must reach pgvector, not quietly fall back to Python.

``BrandMemoryStore._search_backend`` tries Postgres first and falls back to
scanning rows in Python. That fallback is right for SQLite, but it also swallowed
a *type* bug: ``embedding <=> :vec`` was compiled without an explicit type, this
stack inferred JSON for it, and the driver then fed a float to the JSON
deserializer ("'float' object is not subscriptable"). Every workspace on Postgres
was therefore getting its memory ordered by recency and scored 0.0, while the log
line read like a cosmetic warning.

The fix is a single ``cast(..., Float)``. This test asserts the emitted SQL so
the cast cannot be dropped again without something failing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from app.core.memory.store import BrandMemoryStore

pytestmark = pytest.mark.anyio


class _Result:
    def mappings(self) -> "_Result":
        return self

    def all(self) -> list[Any]:
        return []


class _PostgresDialect:
    name = "postgresql"


class _CapturingSession:
    """Records the statements handed to ``execute`` and returns no rows."""

    def __init__(self) -> None:
        self.statements: list[Any] = []
        # ``_search_backend`` chooses its strategy from the dialect name.
        self.bind = SimpleNamespace(dialect=_PostgresDialect())

    async def execute(self, statement: Any) -> _Result:
        self.statements.append(statement)
        return _Result()


async def _compiled_search_sql(*, literal: bool = False, **kwargs: Any) -> str:
    session = _CapturingSession()
    store = BrandMemoryStore(session)  # type: ignore[arg-type]
    await store._search_pgvector(
        business_id="biz_1",
        vector=[0.0] * 8,
        kinds=kwargs.get("kinds", []),
        top_k=kwargs.get("top_k", 3),
    )
    assert len(session.statements) == 1, "pgvector search must issue exactly one statement"
    compile_kwargs = {"literal_binds": True} if literal else {}
    return str(
        session.statements[0].compile(
            dialect=postgresql.dialect(), compile_kwargs=compile_kwargs
        )
    )


async def test_the_distance_is_cast_to_float() -> None:
    """Without the cast the driver deserialises the distance as JSON."""
    sql = await _compiled_search_sql()
    upper = sql.upper()
    # The operator itself must survive -- this is still pgvector's cosine distance.
    assert "<=>" in sql
    assert "CAST(" in upper
    assert "FLOAT" in upper
    assert " AS DISTANCE" in upper


async def test_the_search_is_ordered_and_limited_in_sql() -> None:
    """Ordering belongs in the database: it is the whole point of the index."""
    sql = await _compiled_search_sql(top_k=7)
    upper = sql.upper()
    assert "ORDER BY" in upper
    assert "LIMIT" in upper
    assert "CAST(" in upper


async def test_kind_filters_are_applied_in_sql() -> None:
    """``voice_example`` reads are scoped in SQL, not filtered after the fact."""
    sql = await _compiled_search_sql(kinds=["voice_example", "learning"], literal=True)
    assert "KIND IN" in sql.upper()
    assert "voice_example" in sql and "learning" in sql


async def test_the_fallback_still_exists_for_sqlite() -> None:
    """The Postgres path is an optimisation; non-Postgres installs must keep working."""
    store = BrandMemoryStore(_CapturingSession())  # type: ignore[arg-type]
    assert hasattr(store, "_search_python")
    assert store.PYTHON_SEARCH_CANDIDATES >= 100


async def test_nan_distances_fall_back_instead_of_reaching_the_wire() -> None:
    """A degenerate query must not produce null scores and arbitrary ordering.

    pgvector returns NaN for a cosine distance against a zero-magnitude vector
    (the offline mock embedding provider returns exactly that). NaN orders
    arbitrarily in SQL and serialises as a null score, so the store detects it and
    serves the request from the Python scan, which handles zero vectors.
    """
    from app.db.models import BrandMemoryItem

    item = BrandMemoryItem(
        id="mem_1", business_id="biz_1", kind="offer", text="twelve percent off", metadata_={}
    )
    session = _CapturingSession()
    store = BrandMemoryStore(session)  # type: ignore[arg-type]

    nan_pairs = [(item, float("nan"))] * 3
    python_pairs = [(item, 0.0)]

    async def _pgvector(**_kwargs: Any) -> list[tuple[BrandMemoryItem, float]]:
        return nan_pairs

    async def _python(**_kwargs: Any) -> list[tuple[BrandMemoryItem, float]]:
        return python_pairs

    store._search_pgvector = _pgvector  # type: ignore[method-assign]
    store._search_python = _python  # type: ignore[method-assign]

    hits = await store._search_backend(
        business_id="biz_1", vector=[0.0] * 8, kinds=[], top_k=3
    )

    assert hits == python_pairs
    assert all(score == score for _, score in hits)  # never NaN


async def test_finite_distances_are_used_as_they_are() -> None:
    """The guard must not divert a healthy search away from Postgres."""
    from app.db.models import BrandMemoryItem

    best = BrandMemoryItem(id="mem_1", business_id="biz_1", kind="offer", text="a", metadata_={})
    worse = BrandMemoryItem(id="mem_2", business_id="biz_1", kind="offer", text="b", metadata_={})
    session = _CapturingSession()
    store = BrandMemoryStore(session)  # type: ignore[arg-type]

    async def _pgvector(**_kwargs: Any) -> list[tuple[BrandMemoryItem, float]]:
        return [(best, 0.92), (worse, 0.31)]

    async def _python(**_kwargs: Any) -> list[tuple[BrandMemoryItem, float]]:
        raise AssertionError("the Python scan must not run when Postgres answered")

    store._search_pgvector = _pgvector  # type: ignore[method-assign]
    store._search_python = _python  # type: ignore[method-assign]

    hits = await store._search_backend(
        business_id="biz_1", vector=[0.1] * 8, kinds=[], top_k=2
    )

    assert [score for _, score in hits] == [0.92, 0.31]
