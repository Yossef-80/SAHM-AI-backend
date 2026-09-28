"""Sahm backend: FastAPI app, router registration, middleware, error handling.

Boots with ``LLM_PROVIDER=mock`` and no external keys, running the whole flow on
canned data.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (
    assistant,
    business,
    campaigns,
    creatives,
    insights,
    integrations,
    recommendations,
    tasks,
)
from app.api import personas as personas_router
from app.config import get_settings
from app.core.errors import SahmError
from app.core.llm.router import close_clients
from app.core.logging import bind_request_id, configure_logging, get_logger
from app.core.tools.registry import register_all
from app.db.session import create_all, dispose_engine
from app.schemas.common import ErrorResponse

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    register_all()
    await create_all()
    _logger.info(
        "app_started",
        extra={
            "environment": settings.environment,
            "llm_provider_fast": settings.llm_provider_fast,
            "llm_provider_strong": settings.llm_provider_strong,
            "database": settings.database_url.split("://")[0],
        },
    )
    yield
    await close_clients()
    await dispose_engine()
    _logger.info("app_stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="AI marketing operating system - multi-agent backend",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        bind_request_id(request_id)
        request.state.request_id = request_id
        started = time.monotonic()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        _logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": int((time.monotonic() - started) * 1000),
            },
        )
        return response

    # ------------------------------------------------------------ routers --
    prefix = settings.api_prefix
    app.include_router(assistant.router, prefix=prefix)
    app.include_router(personas_router.router, prefix=prefix)
    app.include_router(business.router, prefix=prefix)
    app.include_router(campaigns.router, prefix=prefix)
    app.include_router(creatives.router, prefix=prefix)
    app.include_router(insights.router, prefix=prefix)
    app.include_router(recommendations.router, prefix=prefix)
    app.include_router(integrations.router, prefix=prefix)
    app.include_router(tasks.router, prefix=prefix)

    # ------------------------------------------------------------- errors --
    @app.exception_handler(SahmError)
    async def sahm_error_handler(request: Request, exc: SahmError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        _logger.warning(
            "request_failed",
            extra={"code": exc.code, "message": exc.message, "path": request.url.path},
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(
                error={
                    "code": exc.code,
                    "message": exc.message,
                    "detail": exc.detail,
                    "requestId": request_id,
                }
            ).model_dump(mode="json", by_alias=True),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        _logger.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error={
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                    "detail": {},
                    "requestId": request_id,
                }
            ).model_dump(mode="json", by_alias=True),
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        settings = get_settings()
        return {
            "status": "ok",
            "environment": settings.environment,
            "llm": {
                "fast": settings.llm_provider_fast,
                "strong": settings.llm_provider_strong,
                "mock": settings.is_mock_llm(),
            },
        }

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": settings.app_name, "docs": "/docs"}

    return app


app = create_app()
