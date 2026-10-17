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
os.environ["AI_LLM_PROVIDER"] = "mock"
os.environ["AI_SEARCH_PROVIDER"] = "mock"
os.environ["AI_DB_PROVIDER"] = "memory"
os.environ["AI_ALLOW_PAID_SEARCH_CALLS"] = "false"
os.environ["AI_ALLOW_PAID_REVIEW_CALLS"] = "false"
os.environ["LLM_PROVIDER_FAST"] = "mock"
os.environ["LLM_PROVIDER_STRONG"] = "mock"
os.environ["SEARCH_PROVIDER"] = "mock"
os.environ["KEYWORD_PROVIDER"] = "mock"
os.environ["IMAGE_PROVIDER"] = "mock"
os.environ["IMAGE_ALLOW_MOCK_ASSETS"] = "false"
# A test process must not inherit ambient provider credentials from a developer's
# shell or .env file. Tests that exercise an authenticated adapter set their own
# fake value explicitly with monkeypatch.
for _credential_env in (
    "META_ACCESS_TOKEN",
    "META_APP_ID",
    "META_APP_SECRET",
    "META_AD_ACCOUNT_ID",
    "TOKEN_ENCRYPTION_KEY",
    "GOOGLE_PLACES_API_KEY",
    "APIFY_TOKEN",
    "TAVILY_API_KEY",
    "SERPER_API_KEY",
    "BRAVE_API_KEY",
    "DATAFORSEO_LOGIN",
    "DATAFORSEO_PASSWORD",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "APIFY_API_TOKEN",
    "AI_POSTGRES_URL",
):
    os.environ[_credential_env] = ""

from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db import models  # noqa: E402,F401  (registers mappers)
from app.core.tools.base import ToolContext  # noqa: E402
from app.db.models import Base  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_legacy_mcp_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the test suite genuinely offline, including legacy MCP wrappers.

    The generic mock LLM deliberately exercises every offered tool. The old
    ``brand_report`` / ``full_analysis`` MCP implementation performs its own
    DuckDuckGo/APify requests, bypassing the app's mock-provider settings. Stub
    only those legacy boundaries; safe tool registration and output contracts
    remain under test.
    """
    from app.core.tools import research
    from app.schemas.common import ToolStatus

    monkeypatch.setattr(
        research,
        "_brand_report",
        lambda **kwargs: {"status": "unavailable", "source": "offline_test"},
    )
    monkeypatch.setattr(
        research,
        "_full_analysis",
        lambda **kwargs: {"status": "unavailable", "source": "offline_test"},
    )
    monkeypatch.setattr(
        research,
        "_route_file_to_tool",
        lambda **kwargs: {"status": "unavailable", "source": "offline_test"},
    )

    async def _website_unavailable(payload: Any, ctx: Any) -> Any:
        return research.WebsiteProfile(
            url=payload.url,
            status=ToolStatus.UNAVAILABLE,
            source="offline_test",
            unavailable_reason="website requests are disabled in offline tests",
        )

    monkeypatch.setattr(research.analyze_website_tool, "handler", _website_unavailable)


@pytest.fixture(autouse=True)
def _isolate_ai_provider_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    """Reset app.ai caches/log destinations so the suite cannot use live providers."""
    monkeypatch.setenv("AI_LLM_PROVIDER", "mock")
    monkeypatch.setenv("AI_SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("AI_DB_PROVIDER", "memory")
    monkeypatch.setenv("AI_ALLOW_PAID_SEARCH_CALLS", "false")
    monkeypatch.setenv("AI_ALLOW_PAID_REVIEW_CALLS", "false")
    monkeypatch.setenv("AI_LOG_DIR", str(tmp_path / "ai-logs"))

    from app.ai.loggers.base import reset_loggers
    from app.ai.providers.db import reset_db_providers
    from app.ai.providers.search import reset_search_providers
    from app.ai.settings import get_ai_settings

    get_ai_settings.cache_clear()
    reset_search_providers()
    reset_db_providers()
    reset_loggers()
    yield
    get_ai_settings.cache_clear()
    reset_search_providers()
    reset_db_providers()
    reset_loggers()


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
        # P4.2: buckets are per app. A test that makes many requests in one
        # second (the auth suite does) must not inherit the previous test's
        # spend, so every test starts from a full set of buckets. The limiter's
        # own behaviour is asserted directly in tests/test_usage_rate_limit.py.
        app.state.rate_limiter.reset()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                yield client
    finally:
        await dispose_engine()


@pytest.fixture()
def rate_limiter(api_client: Any) -> Any:
    """The limiter instance the running app is actually enforcing.

    ``create_app`` builds its limiter once, from the settings at boot, so a test
    that wants a small capacity cannot simply patch ``Settings`` afterwards -- the
    buckets would still be the ones the app captured. This hands the test the live
    object so it can reshape the limits (or switch the limiter off) exactly the way
    an operator would through configuration.
    """
    app = getattr(api_client._transport, "app", None)
    assert app is not None, "api_client must use an ASGI transport"
    return app.state.rate_limiter
