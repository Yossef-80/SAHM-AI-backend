"""P4.8/P5.9: standard security headers on every response.

What is set and why
-------------------
``X-Content-Type-Options: nosniff``
    The API returns JSON only. Without ``nosniff`` a browser may be persuaded
    by a reflected string to treat a response as HTML, which turns a data field
    into script execution.
``X-Frame-Options: DENY``
    There is nothing to frame. ``CSP: frame-ancestors 'none'`` says the same
    thing to modern browsers; the legacy header is kept because it costs nothing
    and old browsers honour only it.
``Referrer-Policy: strict-origin-when-cross-origin``
    The browser default is older than this and leaks full URLs to third parties.
``Strict-Transport-Security``
    **Only meaningful over TLS.** A browser ignores the header on ``http://``,
    and HSTS on a plain-http origin is a no-op rather than a hazard -- but a pod
    that terminates TLS at a proxy must still forward the header, so it is set
    here and can be switched off with ``SECURITY_HSTS_ENABLED=false`` for a
    local dev server where claiming "this host is HTTPS-only" would be wrong.
``Content-Security-Policy``
    ``default-src 'none'`` for a JSON API: no scripts, no connections, no
    frames, no form posts. It costs nothing because nothing legitimate is
    loaded, and it is the header that would actually contain a reflected-value
    injection if one ever appeared.

Added extras that cost nothing: ``X-Permitted-Cross-Domain-Policies: none``,
``Cross-Origin-Resource-Policy: same-site`` and
``Permissions-Policy`` denying the powerful browser APIs the API never uses.
"""

from __future__ import annotations

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings, get_settings
from app.core.logging import get_logger

_logger = get_logger(__name__)


def build_headers(settings: Settings) -> dict[str, str]:
    """The header set for this deployment."""
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-Permitted-Cross-Domain-Policies": "none",
        "Cross-Origin-Resource-Policy": "same-site",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
        "Content-Security-Policy": settings.content_security_policy,
    }
    if settings.security_hsts_enabled:
        headers["Strict-Transport-Security"] = (
            f"max-age={settings.security_hsts_max_age}; includeSubDomains"
        )
    return headers


class SecurityHeadersMiddleware:
    """Attach the header set to every response, including errors."""

    def __init__(self, app: ASGIApp, settings: Settings | None = None) -> None:
        self.app = app
        cfg = settings or get_settings()
        self.enabled = bool(cfg.security_headers_enabled)
        self.headers = build_headers(cfg)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                for name, value in self.headers.items():
                    if name not in response_headers:
                        response_headers[name] = value
            await send(message)

        await self.app(scope, receive, send_wrapper)


def validate_cors(settings: Settings) -> list[str]:
    """Reject a CORS configuration a browser would refuse anyway. Returns warnings.

    ``allow_origins=["*"]`` with ``allow_credentials=True`` is the dangerous pair:
    Starlette will happily send ``Access-Control-Allow-Origin: *`` alongside
    ``Access-Control-Allow-Credentials: true``, and every browser then discards
    the response. The configuration looks permissive in review and is silently
    broken in production, so it is refused **at startup** rather than discovered
    as "the frontend cannot call the API".

    This app always sends ``allow_credentials=True``, so the rule is simply "the
    wildcard origin is never valid here" and it raises in every environment. A
    browser already enforces it; failing at boot turns a silent, intermittent
    frontend failure into one clear log line.
    """
    warnings: list[str] = []
    origins = {origin.strip() for origin in settings.cors_origins}
    if "*" in origins:
        raise ValueError(
            "CORS_ORIGINS contains '*', which cannot be combined with the "
            "credentialed CORS this API always enables -- every browser would "
            "reject the response. List the frontend origins explicitly "
            "(e.g. 'http://localhost:3000')."
        )
    if not origins:
        warnings.append(
            "CORS_ORIGINS is empty: browser clients on another origin will be "
            "blocked. This is fine for a server-to-server deployment."
        )
    return warnings


def request_origin(scope: Scope) -> str | None:
    """The ``Origin`` header of a request, or None."""
    return Headers(scope=scope).get("origin")


__all__ = [
    "SecurityHeadersMiddleware",
    "build_headers",
    "request_origin",
    "validate_cors",
]