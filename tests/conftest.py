"""Shared pytest fixtures.

The whole suite runs on SQLite with the mock LLM by default: no API keys, no
Postgres, no network. Set ``SAHM_TEST_DATABASE_URL`` to point at Postgres.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

os.environ.setdefault("SAHM_ENVIRONMENT", "test")
os.environ.setdefault("LLM_PROVIDER_FAST", "mock")
os.environ.setdefault("LLM_PROVIDER_STRONG", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("SEARCH_PROVIDER", "mock")
os.environ.setdefault("KEYWORD_PROVIDER", "mock")
os.environ.setdefault("IMAGE_PROVIDER", "mock")

from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db import models  # noqa: E402,F401  (registers mappers)
from app.core.tools.base import ToolContext  # noqa: E402
from app.db.models import Base  # noqa: E402


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def test_settings() -> Iterator[Any]:
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    yield settings
    get_settings.cache_clear()


@pytest.fixture()
async def db_engine(tmp_path: Any) -> AsyncIterator[Any]:
    """A fresh SQLite database per test."""
    from sqlalchemy import event

    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    engine = create_async_engine(url, future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture()
async def db_session(db_engine: Any) -> AsyncIterator[AsyncSession]:
    maker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with maker() as session:
        yield session
        await session.rollback()


@pytest.fixture()
def request_id() -> str:
    return "req-test-0001"


@pytest.fixture()
async def seeded_db(db_session: AsyncSession) -> AsyncSession:
    """A business with a brand profile and a few memory items."""
    from app.db.models import BrandMemoryItem, BrandProfile, Business

    business = Business(
        id="biz_test",
        workspace_id="ws_test",
        name="Cairo Coffee Co",
        industry="F&B",
        country="EG",
        language="en",
        currency="EGP",
        monthly_budget=50000.0,
    )
    db_session.add(business)
    db_session.add(
        BrandProfile(
            id="bp_test",
            business_id="biz_test",
            brand_voice="warm, direct, a bit cheeky",
            visual_style="earthy tones, natural light",
            target_audience="25-40 Cairo professionals",
            key_offers=["subscription beans", "office catering"],
            goals=["grow subscriptions"],
            competitors=["Bean There", "Kaffa Roasters"],
        )
    )
    for kind, text in [
        ("voice_example", "We talk like a friend who knows coffee, not a barista snob."),
        ("offer", "Subscription beans: 12% off, free delivery in Cairo, cancel anytime."),
        ("audience", "Office managers ordering for teams of 10-50."),
        ("competitor_note", "Bean There leads on price; Kaffa leads on Instagram aesthetics."),
    ]:
        db_session.add(
            BrandMemoryItem(
                id=f"mem_{kind}",
                business_id="biz_test",
                kind=kind,
                text=text,
                metadata_={},
            )
        )
    await db_session.commit()
    return db_session


@pytest.fixture()
def tool_ctx(seeded_db: AsyncSession, request_id: str) -> ToolContext:
    """A ToolContext wired to the seeded business and the mock LLM."""
    from app.core.llm.mock_client import MockClient

    return ToolContext(
        workspace_id="ws_test",
        business_id="biz_test",
        db=seeded_db,
        llm=MockClient(model="mock-strong-1"),
        request_id=request_id,
    )


@pytest.fixture()
async def api_client(db_engine: Any, seeded_db: AsyncSession):
    """An httpx client whose app shares this test's database."""
    import httpx

    from app.db.session import dispose_engine, set_engine_for_tests
    from app.main import create_app

    set_engine_for_tests(db_engine)
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client
    finally:
        await dispose_engine()
