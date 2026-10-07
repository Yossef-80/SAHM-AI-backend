"""P4.2: dependency-free in-process rate limiting.

Design
------
A token bucket per key, refilled at ``rate`` tokens/second up to ``capacity``.
The bucket is chosen over a fixed window because an agent product has a very
specific failure mode -- one client firing 200 tool invocations in a second --
and a fixed window either lets that through at the boundary or throttles a
legitimate burst for no reason. A bucket permits a burst of ``capacity`` and then
settles at ``rate``, which is the behaviour a human and an API client both
expect.

Keying
------
``user_id`` when the request carries a valid session, otherwise the client IP.
One shared key would mean one noisy anonymous client throttling every other
anonymous client on the same egress NAT, which is a real deployment shape for
this product (an agency behind one router).

The honest limitation
---------------------
**This bucket is per worker process.** With ``uvicorn --workers 4`` the effective
limit is four times the configured one, because there are four independent
dictionaries. That is stated here, in the config comments and in the response
headers rather than buried, because a rate limit that silently multiplies with
replica count is worse than no rate limit -- it looks configured and is not.

The fix, when this matters, is a shared store (Redis ``INCR``/``EXPIRE``); the
interface below (:class:`RateLimiter`) is already shaped like that -- one
``consume`` call returning an allow/deny plus a retry hint -- so swapping the
dictionary for Redis is a change to one method, not to the middleware, the
headers, or the 429 shape.

``RATE_LIMIT_ENABLED=false`` turns the whole thing off, which is the escape
hatch for a load test and for the case where a shared store exists upstream and
this would just double-count.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings, get_settings
from app.core.errors import RateLimitExceeded
from app.core.logging import get_logger

_logger = get_logger(__name__)

#: Route classes. Ordered strictest-first because ``classify`` returns the first
#: match, so ordering is load-bearing rather than cosmetic.
ROUTE_CLASS_AUTH = "auth"
ROUTE_CLASS_AUTH_EXCHANGE = "auth_exchange"
ROUTE_CLASS_TOOL = "tool"
ROUTE_CLASS_WRITE = "write"
ROUTE_CLASS_READ = "read"


@dataclass(frozen=True)
class Limit:
    """One route class's bucket shape."""

    capacity: float
    refill_per_second: float

    @property
    def refill_per_minute(self) -> float:
        return round(self.refill_per_second * 60.0, 2)


def limits_from_settings(settings: Settings) -> dict[str, Limit]:
    """The four route-class buckets, from configuration."""
    return {
        ROUTE_CLASS_AUTH: Limit(
            capacity=float(settings.rate_limit_auth_capacity),
            refill_per_second=settings.rate_limit_auth_refill_per_second,
        ),
        # Supabase emits TOKEN_REFRESHED events during normal session renewal.
        # Each event performs one exchange, so keep this route bounded while
        # allowing routine refreshes and retries without consuming the login bucket.
        ROUTE_CLASS_AUTH_EXCHANGE: Limit(capacity=60.0, refill_per_second=1.0),
        ROUTE_CLASS_TOOL: Limit(
            capacity=float(settings.rate_limit_tool_capacity),
            refill_per_second=settings.rate_limit_tool_refill_per_second,
        ),
        ROUTE_CLASS_WRITE: Limit(
            capacity=float(settings.rate_limit_write_capacity),
            refill_per_second=settings.rate_limit_write_refill_per_second,
        ),
        ROUTE_CLASS_READ: Limit(
            capacity=float(settings.rate_limit_read_capacity),
            refill_per_second=settings.rate_limit_read_refill_per_second,
        ),
    }


def classify(method: str, path: str) -> str:
    """Which bucket a request belongs to.

    Path suffixes are matched rather than enumerated so a new endpoint inherits a
    sane class instead of silently landing in the permissive one.
    """
    if path.endswith("/auth/supabase-exchange"):
        return ROUTE_CLASS_AUTH_EXCHANGE
    if "/auth/" in path or path.endswith("/auth"):
        return ROUTE_CLASS_AUTH
    if "/tools/" in path and method.upper() in {"POST", "PUT", "PATCH"}:
        return ROUTE_CLASS_TOOL
    if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        return ROUTE_CLASS_WRITE
    return ROUTE_CLASS_READ


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


@dataclass
class Verdict:
    """The result of one ``consume`` call."""

    allowed: bool
    remaining: int
    #: Whole seconds until the bucket holds one token again. Always >= 1.
    retry_after: int
    limit: int

    def headers(self) -> dict[str, str]:
        out = {
            "x-ratelimit-limit": str(self.limit),
            "x-ratelimit-remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            out["retry-after"] = str(self.retry_after)
        return out


@dataclass
class RateLimiter:
    """``dict[key, bucket]`` behind a lock. See the module docstring."""

    limits: dict[str, Limit]
    enabled: bool = True
    _buckets: dict[tuple[str, str], _Bucket] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: Bounded so a flood of distinct IPs cannot grow the dict without limit.
    max_keys: int = 20_000

    async def consume(self, route_class: str, key: str) -> Verdict:
        """Take one token, or report how long to wait."""
        limit = self.limits.get(route_class, self.limits[ROUTE_CLASS_READ])
        capacity = limit.capacity
        rate = limit.refill_per_second

        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get((route_class, key))
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    self._evict()
                bucket = _Bucket(tokens=capacity, updated_at=now)
                self._buckets[(route_class, key)] = bucket

            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(capacity, bucket.tokens + elapsed * rate)
            bucket.updated_at = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return Verdict(
                    allowed=True,
                    remaining=int(bucket.tokens),
                    retry_after=0,
                    limit=int(capacity),
                )

            # Need (1 - tokens) more tokens at `rate` tokens/second.
            deficit = 1.0 - bucket.tokens
            retry_after = max(1, int(deficit / rate) + 1) if rate > 0 else 60
            return Verdict(
                allowed=False,
                remaining=0,
                retry_after=retry_after,
                limit=int(capacity),
            )

    def _evict(self) -> None:
        """Drop full buckets.

        A full bucket is indistinguishable from a brand new one, so dropping the
        full ones is a perfect LRU of everything that has not been throttled and
        bounds the dictionary without ever evicting an actual abuser. Called with
        the lock held.
        """
        for key in [
            k
            for k, b in self._buckets.items()
            if b.tokens
            >= self.limits.get(k[0], self.limits[ROUTE_CLASS_READ]).capacity
        ]:
            del self._buckets[key]

    def reset(self) -> None:
        """Drop every bucket. Test-only; also used by ``RATE_LIMIT_ENABLED``."""
        self._buckets.clear()


def _client_key(request: Any, workspace_header: str | None) -> str:
    """The rate-limit key for one request.

    ``user:<id>`` when the caller authenticated (the id was resolved by this
    middleware before the token was consumed -- see the class docstring), then
    the workspace header, then the peer IP. The ladder matters because one
    shared key would let a single noisy anonymous client throttle every other
    client behind the same egress NAT, which is a real shape for this product.
    """
    user_id = getattr(getattr(request, "state", None), "auth_user_id", None)
    if user_id:
        return f"user:{user_id}"
    if workspace_header:
        return f"ws:{workspace_header}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def build_limiter(settings: Settings | None = None) -> RateLimiter:
    """Construct the process-wide limiter from configuration."""
    cfg = settings or get_settings()
    return RateLimiter(
        limits=limits_from_settings(cfg), enabled=bool(cfg.rate_limit_enabled)
    )


#: One limiter per process. A module global because the bucket state is
#: deliberately per-process -- see the module docstring.
_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = build_limiter()
    return _limiter


def reset_limiter() -> None:
    """Drop the process limiter. Test helper."""
    global _limiter
    _limiter = None


class RateLimitMiddleware:
    """ASGI middleware enforcing :class:`RateLimiter`.

    Returns 429 with the standard error envelope, ``Retry-After`` and the
    ``X-RateLimit-*`` family. Headers are attached to *successful* responses too,
    because a client that can see its remaining budget can back off before it is
    throttled rather than after.

    Identity before token
    ---------------------
    The key must be the *user id* for an authenticated caller, but a middleware
    runs before FastAPI dependencies, so ``get_principal`` has not run yet. This
    middleware therefore resolves the bearer token itself -- but only when an
    ``Authorization`` header is actually present, so an anonymous request costs
    nothing -- and publishes the result on ``request.state.auth_user_id``.

    ``app/api/deps.py::get_principal`` reuses that resolution instead of
    repeating the session lookup, so the total cost is one session query per
    authenticated request, the same as before this middleware existed.
    """

    def __init__(self, app: ASGIApp, limiter: RateLimiter | None = None) -> None:
        self.app = app
        self.limiter = limiter or get_limiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.limiter.enabled:
            await self.app(scope, receive, send)
            return

        from starlette.datastructures import Headers
        from starlette.requests import Request

        request = Request(scope, receive=receive)
        headers = Headers(scope=scope)
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        settings = get_settings()
        if path in settings.rate_limit_exempt_paths:
            await self.app(scope, receive, send)
            return

        # Skip rate limiting for OPTIONS (CORS preflight) requests
        if method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        await self._resolve_identity(request, headers)

        route_class = classify(method, path)
        key = _client_key(request, headers.get("x-workspace-id"))
        verdict = await self.limiter.consume(route_class, key)

        if not verdict.allowed:
            error = RateLimitExceeded(
                "Too many requests. Slow down and retry after the interval in "
                "Retry-After.",
                detail={
                    "route_class": route_class,
                    "limit": verdict.limit,
                    "retry_after": verdict.retry_after,
                },
                retry_after=verdict.retry_after,
            )
            _logger.warning(
                "rate_limited",
                extra={"route_class": route_class, "path": path, "status": 429},
            )
            await self._send_429(scope, receive, send, error, verdict)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = message.setdefault("headers", [])
                for name, value in verdict.headers().items():
                    response_headers.append(
                        (name.encode("latin-1"), value.encode("latin-1"))
                    )
            await send(message)

        await self.app(scope, receive, send_wrapper)

    async def _resolve_identity(self, request: Any, headers: Any) -> None:
        """Bind ``request.state.auth_user_id`` when a bearer token is present.

        Cheap by construction: no header, no work. On any failure the request
        simply stays anonymous and the limiter falls back to the workspace header
        or the peer IP -- a limiter must never be the thing that turns a working
        request into a 500.
        """
        token = headers.get("authorization")
        if not token or not token.lower().startswith("bearer "):
            return
        from app.core.auth import resolve_session
        from app.db.session import session_scope

        try:
            async with session_scope() as db:
                user = await resolve_session(db, token[len("bearer ") :].strip())
            if user is not None:
                request.state.auth_user_id = user.id
        except Exception as exc:  # pragma: no cover - limiter must never 500
            _logger.warning(
                "rate_limit_identity_unavailable", extra={"error": str(exc)}
            )

    async def _send_429(
        self, scope: Scope, receive: Receive, send: Send, error: RateLimitExceeded, verdict: Verdict
    ) -> None:
        """Render 429 in the one envelope every failure uses.

        ``requestId`` is echoed from the request when the request-id middleware
        already bound one, so a throttled response is still traceable in logs.
        """
        from starlette.datastructures import Headers
        from starlette.responses import JSONResponse

        from app.schemas.common import ErrorResponse

        headers = Headers(scope=scope)
        request_id = headers.get("x-request-id")
        response = JSONResponse(
            status_code=error.http_status,
            content=ErrorResponse(
                error={
                    "code": error.code,
                    "message": error.message,
                    "detail": error.detail,
                    "requestId": request_id,
                }
            ).model_dump(mode="json", by_alias=True),
            headers=verdict.headers(),
        )
        await response(scope, receive, send)


__all__ = [
    "ROUTE_CLASS_AUTH",
    "ROUTE_CLASS_READ",
    "ROUTE_CLASS_TOOL",
    "ROUTE_CLASS_WRITE",
    "Limit",
    "RateLimitExceeded",
    "RateLimitMiddleware",
    "RateLimiter",
    "Verdict",
    "build_limiter",
    "classify",
    "get_limiter",
    "limits_from_settings",
    "reset_limiter",
]
