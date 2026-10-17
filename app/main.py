"""Sahm backend: FastAPI app, router registration, middleware, error handling.

Boots with ``LLM_PROVIDER=mock`` and no external keys, running the whole flow on
canned data.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import (
    assistant,
    audiences,
    audit,
    auth,
    business,
    campaigns,
    content,
    creatives,
    insights,
    integrations,
    media_assets,
    memory,
    recommendations,
    research,
    research_compare,
    subscriptions,
    tasks,
    tools,
    usage,
)
from app.api import personas as personas_router
from app.api import (
    settings as settings_router,
)
from app.config import get_settings
from app.core.errors import SahmError
from app.core.llm.router import close_clients
from app.core.logging import bind_request_id, configure_logging, get_logger
from app.core.ratelimit import RateLimitMiddleware, build_limiter
from app.core.security import verify_runtime_security
from app.core.security_headers import (
    SecurityHeadersMiddleware,
    build_headers as build_security_headers,
    validate_cors,
)
from app.core.tools.registry import register_all
from app.db.session import create_all, dispose_engine
from app.schemas.common import ErrorResponse

_logger = get_logger(__name__)

#: HTTP status -> envelope code, for the framework's own ``HTTPException``.
_HTTP_ERROR_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    415: "unsupported_media_type",
    422: "validation_failed",
    429: "rate_limited",
    503: "service_unavailable",
}


def _http_error_code(status_code: int) -> str:
    """A stable envelope code for a status FastAPI raised itself."""
    return _HTTP_ERROR_CODES.get(status_code, "http_error")


def _jsonable_errors(exc: RequestValidationError) -> list[dict[str, object]]:
    """Pydantic errors rendered JSON-safe, in the same shape as ``app/api/tools.py``."""
    out: list[dict[str, object]] = []
    for err in exc.errors():
        out.append(
            {
                "loc": [str(part) for part in err.get("loc", ())],
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
        )
    return out


async def _check_database() -> tuple[bool, str]:
    """Open a connection and run ``SELECT 1``. Returns (ok, detail)."""
    from sqlalchemy import text

    from app.db.session import get_engine

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    # P4/D5: refuse to serve with the published dev encryption key. Raises in
    # prod before a single request is accepted; a warning line everywhere else.
    for message in verify_runtime_security(settings):
        _logger.warning("insecure_runtime_configuration", extra={"detail": message})
    register_all()
    # Token accounting writer. Every LLM call anywhere in the process queues a
    # usage delta; this task is what turns those deltas into ``usage_records``
    # rows, on its own session, so the request transaction is never involved.
    from app.core.ai_usage import start_usage_writer

    start_usage_writer()
    if settings.environment == "prod":
        # Production never issues DDL at boot, even if it is misconfigured to
        # use SQLite. The schema must already have been installed by release.
        _logger.info(
            "schema_creation_skipped",
            extra={"reason": "production", "schema_source": "alembic"},
        )
    elif settings.database_url.lower().startswith("sqlite"):
        # create_all is a convenience for disposable local/test SQLite stores.
        # Never point it at PostgreSQL: it can create future tables without
        # applying their Alembic revision or alter existing columns, leaving a
        # live database between schema versions.
        await create_all()
        _logger.info(
            "schema_creation_performed",
            extra={"reason": settings.environment, "schema_source": "create_all"},
        )
    else:
        # PostgreSQL (including Supabase) is migration-managed in every
        # environment. Apply ``alembic upgrade head`` before starting the app.
        _logger.info(
            "schema_creation_skipped",
            extra={
                "reason": "alembic_managed_database",
                "schema_source": "alembic",
                "environment": settings.environment,
            },
        )
    _logger.info(
        "app_started",
        extra={
            "environment": settings.environment,
            "llm_provider_fast": settings.llm_provider_fast,
            "llm_provider_strong": settings.llm_provider_strong,
            "database": settings.database_url.split("://")[0],
            # False means the sensitive endpoints still answer an anonymous
            # caller, exactly as they did before authentication existed. Logged
            # so an operator can see at a glance whether a deployed instance is
            # actually locked.
            "require_auth": settings.require_auth,
        },
    )
    yield
    from app.core.ai_usage import stop_usage_writer

    await stop_usage_writer()
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

    # --- CORS (P5.9): explicit methods/headers, wildcard refused at startup ---
    for message in validate_cors(settings):
        _logger.warning("cors_configuration", extra={"detail": message})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=list(settings.cors_allow_methods),
        allow_headers=list(settings.cors_allow_headers),
        expose_headers=["X-Request-Id", "X-RateLimit-Limit", "X-RateLimit-Remaining",
                        "Retry-After"],
    )

    # --- Security headers + rate limiting (P4.2/P4.8) ---
    # Order matters and is subtle: ``add_middleware`` *prepends*, so the
    # middleware added last runs **outermost**. The limiter is therefore added
    # first, which puts the security-header middleware outside it -- so a 429
    # carries the same headers as a 200. A throttled response without
    # ``X-Content-Type-Options`` is still a response a browser may mishandle.
    limiter = build_limiter(settings)
    app.state.rate_limiter = limiter
    app.add_middleware(RateLimitMiddleware, limiter=limiter)
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"
        bind_request_id(request_id)
        request.state.request_id = request_id

        from app.db.session import get_request_db_metrics, init_request_db_metrics
        init_request_db_metrics()

        query_params = dict(request.query_params)
        client_host = request.client.host if request.client else "unknown"
        body_preview = None

        content_type = request.headers.get("content-type", "")
        if (
            request.method in {"POST", "PUT", "PATCH"}
            and not content_type.startswith("multipart/")
        ):
            try:
                raw_body = await request.body()
                if raw_body and len(raw_body) < 100_000:
                    try:
                        from app.core.logging import redact
                        parsed = json.loads(raw_body.decode("utf-8", errors="replace"))
                        body_preview = redact(parsed)
                    except Exception:
                        text_val = raw_body.decode("utf-8", errors="replace")
                        body_preview = text_val[:300] + ("..." if len(text_val) > 300 else "")
            except Exception:
                pass

        _logger.info(
            f"[ROUTING - INCOMING] {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query": query_params,
                "client": client_host,
                "body": body_preview,
            },
        )

        started = time.monotonic()
        response = await call_next(request)
        duration_ms = int((time.monotonic() - started) * 1000)
        response.headers["x-request-id"] = request_id

        db_metrics = get_request_db_metrics()
        if db_metrics:
            response.headers["X-DB-Queries"] = str(db_metrics["query_count"])
            response.headers["X-DB-Time"] = f"{db_metrics['total_duration_ms']}ms"

        db_q_count = db_metrics.get("query_count", 0) if db_metrics else 0
        db_time_ms = db_metrics.get("total_duration_ms", 0) if db_metrics else 0

        _logger.info(
            f"[ROUTING - OUTGOING] {response.status_code} {request.method} {request.url.path} ({duration_ms}ms, DB: {db_time_ms}ms/{db_q_count}q)",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "db_queries": db_q_count,
                "db_time_ms": db_time_ms,
            },
        )
        return response

    # ------------------------------------------------------------ routers --
    prefix = settings.api_prefix
    app.include_router(auth.router, prefix=prefix)
    app.include_router(assistant.router, prefix=prefix)
    app.include_router(audiences.router, prefix=prefix)
    app.include_router(personas_router.router, prefix=prefix)
    app.include_router(business.router, prefix=prefix)
    app.include_router(campaigns.router, prefix=prefix)
    app.include_router(content.router, prefix=prefix)
    app.include_router(creatives.router, prefix=prefix)
    app.include_router(insights.router, prefix=prefix)
    app.include_router(recommendations.router, prefix=prefix)
    app.include_router(integrations.router, prefix=prefix)
    app.include_router(media_assets.router, prefix=prefix)
    app.include_router(tasks.router, prefix=prefix)
    app.include_router(tools.router, prefix=prefix)
    app.include_router(research.router, prefix=prefix)
    # Competitor-vs-competitor gap analysis lives in its own module so the
    # 400-line research runner stays readable; it mounts under /research.
    app.include_router(research_compare.router, prefix=prefix)
    # --- P2 configuration surface ---
    app.include_router(memory.router, prefix=prefix)
    app.include_router(settings_router.router, prefix=prefix)
    # --- P4 usage / audit / subscriptions ---
    app.include_router(usage.router, prefix=prefix)
    app.include_router(audit.router, prefix=prefix)
    app.include_router(subscriptions.router, prefix=prefix)

    # ------------------------------------------------------------- errors --
    def browser_visible_headers(request: Request) -> dict[str, str]:
        """The headers a crashed request would otherwise reach the browser without.

        ``ServerErrorMiddleware`` -- the layer that turns the bare-``Exception``
        handler's response into an HTTP response -- sits *outside* both the CORS
        middleware and the security-header middleware. When an exception escapes
        the request-id middleware (any exception raised anywhere inside it), the
        response it builds is therefore sent with neither, and the browser
        refuses to hand it to ``fetch`` at all. The frontend then has no status
        and no body to report, so it says "could not reach the backend" -- a
        live server crash presented to the user (and to support) as a
        connectivity problem, while the 500 never appears in the UI.

        Re-attaching the same headers here keeps a 5xx as diagnosable as a 4xx:
        the browser can read the envelope, and the request id the log line names
        travels with it.
        """
        headers: dict[str, str] = (
            dict(build_security_headers(settings))
            if settings.security_headers_enabled
            else {}
        )

        origin = request.headers.get("origin")
        allowed_origins = list(settings.cors_origins)
        if origin and origin in allowed_origins:
            # Echo the exact origin: with credentials allowed, ``*`` is not a
            # substitute and every browser would discard the response.
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Access-Control-Expose-Headers"] = "X-Request-Id"
            headers["Vary"] = "Origin"

        request_id = getattr(request.state, "request_id", None)
        if request_id:
            headers["X-Request-Id"] = request_id
        return headers

    def envelope(
        status_code: int,
        *,
        code: str,
        message: str,
        detail: dict[str, object],
        request_id: str | None,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        """Render the one error shape every failure path must use (P0.6).

        ``SahmError``, FastAPI's own ``RequestValidationError``/``HTTPException``
        and the unhandled-exception path all come through here, so a client can
        parse one shape without special-casing 422.
        """
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error={
                    "code": code,
                    "message": message,
                    "detail": detail,
                    "requestId": request_id,
                }
            ).model_dump(mode="json", by_alias=True),
            headers=headers,
        )

    @app.exception_handler(SahmError)
    async def sahm_error_handler(request: Request, exc: SahmError) -> JSONResponse:
        _logger.warning(
            "request_failed",
            extra={"code": exc.code, "message": exc.message, "path": request.url.path},
        )
        return envelope(
            exc.http_status,
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """FastAPI's own 422, reshaped into the standard envelope.

        Without this the body is FastAPI's default ``{"detail": [...]}``,
        which is the one failure shape a client cannot parse generically.
        """
        return envelope(
            422,
            code="validation_failed",
            message="The request body or parameters failed validation.",
            detail={"issues": _jsonable_errors(exc)},
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Any ``HTTPException`` raised anywhere, in the standard envelope."""
        return envelope(
            exc.status_code,
            code=_http_error_code(exc.status_code),
            message=str(exc.detail),
            detail={},
            request_id=getattr(request.state, "request_id", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        _logger.exception("unhandled_error", extra={"path": request.url.path})
        return envelope(
            500,
            code="internal_error",
            message="An unexpected error occurred.",
            detail={},
            request_id=getattr(request.state, "request_id", None),
            headers=browser_visible_headers(request),
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Liveness only: no DB, no I/O, so a database blip never restarts a pod."""
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

    @app.get("/health/ready")
    async def readiness(request: Request) -> JSONResponse:
        """Readiness with a per-component breakdown.

        Two components, both of which have actually broken real deployments:

        * ``database`` -- opens a connection and runs ``SELECT 1`` (D4). The old
          liveness probe could report ``ok`` with a dead database, so
          orchestrators kept routing traffic to a process that could not serve.
        * ``encryption_key`` -- a production deployment with no
          ``TOKEN_ENCRYPTION_KEY`` is *not* ready, because every credential it
          stores would be encrypted with a constant published in the repository
          (D5). Failing here catches it at deploy time instead of at incident
          time.

        ``/health`` stays liveness-only and dependency-free, so a database blip
        never restarts a healthy pod.
        """
        settings = get_settings()
        checks: dict[str, str] = {}

        db_ok, db_message = await _check_database()
        checks["database"] = "ok" if db_ok else db_message

        try:
            verify_runtime_security(settings)
            checks["encryption_key"] = "ok"
        except SahmError as exc:
            checks["encryption_key"] = exc.code

        degraded = [
            name
            for name, value in checks.items()
            if name != "database" and value != "ok"
        ]

        if db_ok and not degraded:
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "checks": checks},
            )
        _logger.error(
            "readiness_failed",
            extra={"checks": checks, "degraded": degraded},
        )
        return envelope(
            503,
            code="not_ready",
            message="The service is not ready to receive traffic.",
            detail={"checks": checks, "degraded": degraded},
            request_id=getattr(request.state, "request_id", None),
        )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": settings.app_name, "docs": "/docs"}

    return app


app = create_app()
