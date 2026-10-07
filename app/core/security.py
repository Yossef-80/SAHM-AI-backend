"""Security helpers: SSRF protection, token encryption, upload validation.

Three separate concerns, all driven by config:
1. ``analyze_website`` must not be able to reach private networks (SSRF).
2. Integration tokens are encrypted at rest with Fernet.
3. Uploaded files are type- and size-checked before anything touches them.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from app.core.errors import (
    BlockedTargetError,
    EncryptionKeyMissingError,
    UploadRejectedError,
)
from app.core.logging import get_logger

_logger = get_logger(__name__)

#: Schemes we will ever fetch. Anything else is rejected.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Extensions we accept, mapped to the MIME type we insist on.
_ALLOWED_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".csv": "text/csv",
    ".pdf": "application/pdf",
}

#: Magic-number prefixes for the image types we accept.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


@dataclass(frozen=True)
class SafeUrl:
    url: str
    host: str
    port: int
    scheme: str


def _is_private_ip(ip_text: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_text)
    except ValueError:
        return True  # unparseable -> treat as unsafe
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def resolve_and_check(url: str, *, allowed_schemes: tuple[str, ...] = ("http", "https")) -> SafeUrl:
    """Parse a URL, resolve its host, and refuse private/reserved targets.

    Raises ``BlockedTargetError`` for anything that is not a public HTTP(S)
    host. Called before every outbound fetch from a tool.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in allowed_schemes:
        raise BlockedTargetError(f"scheme '{scheme}' is not allowed")
    host = parsed.hostname
    if not host:
        raise BlockedTargetError("url has no host")

    # A literal IP is checked directly.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if _is_private_ip(host):
            raise BlockedTargetError(f"host '{host}' is in a private/reserved range")
        return SafeUrl(url=url, host=host, port=parsed.port or (443 if scheme == "https" else 80), scheme=scheme)

    # Otherwise resolve and check every answer.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise BlockedTargetError(f"could not resolve host '{host}'") from exc
    for info in infos:
        ip_text = str(info[4][0])
        if _is_private_ip(ip_text):
            raise BlockedTargetError(
                f"host '{host}' resolves to a private/reserved address ({ip_text})"
            )
    port = parsed.port or (443 if scheme == "https" else 80)
    return SafeUrl(url=url, host=host, port=port, scheme=scheme)


def check_redirect(target: str, *, allowed_schemes: tuple[str, ...]) -> SafeUrl:
    """Validate each redirect hop. Redirects are the classic SSRF bypass."""
    return resolve_and_check(target, allowed_schemes=allowed_schemes)


# ------------------------------------------------------------- encryption ----

#: The deterministic development key. Published in this file on purpose: it is
#: not a secret and must never be trusted. It exists so ``python -m uvicorn``
#: works with zero configuration, and ``verify_runtime_security`` refuses to let
#: it reach production.
DEV_FERNET_KEY = b"c2FobS1kZXYtb25seS10b2tlbi1rZXktMzItYnl0ZXM="


def verify_runtime_security(settings: Any | None = None) -> list[str]:
    """P4 startup validator: refuse to serve with dev-grade security.

    Called once from the FastAPI lifespan, *before* the app reports itself ready,
    so a deploy with no ``TOKEN_ENCRYPTION_KEY`` fails at boot instead of
    "encrypting" customer Meta tokens with a key that is in git.

    Returns the list of warnings raised (empty when the deployment is clean).
    Raises ``EncryptionKeyMissingError`` in production when the key is absent --
    the one condition here that is fatal rather than noisy.

    D5 in the roadmap. The roadmap files this under P5.6 but it is a one-line
    guard, and shipping the configuration surface (P2.4) without it would mean
    a new "set my connector credentials in the UI" feature writes secrets under
    a published key.
    """
    from app.config import get_settings

    cfg = settings or get_settings()
    warnings: list[str] = []
    if not cfg.token_encryption_key:
        if cfg.environment == "prod":
            raise EncryptionKeyMissingError(
                "TOKEN_ENCRYPTION_KEY is not set and ENVIRONMENT=prod. Every "
                "stored integration token would be encrypted with a key that is "
                "published in the source. Generate one with "
                "`python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\"` and set it before starting.",
                detail={"environment": cfg.environment, "setting": "TOKEN_ENCRYPTION_KEY"},
            )
        warnings.append(
            "TOKEN_ENCRYPTION_KEY is unset: falling back to the published dev key. "
            "Stored tokens are only as private as this repository."
        )
    return warnings


class TokenCipher:
    """Fernet-based encryption for integration tokens at rest.

    When no key is configured in dev/test we derive a deterministic dev key so
    the app still boots with zero configuration -- and warn, loudly, once. In
    production the same situation raises ``EncryptionKeyMissingError``: the
    alternative is silently protecting customer credentials with a constant
    from this file.
    """

    def __init__(self, key: str | None, *, environment: str | None = None) -> None:
        from app.config import get_settings

        env = environment or get_settings().environment
        if not key and env == "prod":
            # Same failure as the startup validator, reached from the request
            # path. Raised here rather than delegated: ``verify_runtime_security``
            # reads the *cached* settings, which in a test (or any process where
            # the two disagree) would answer for a different environment.
            raise EncryptionKeyMissingError(
                "TOKEN_ENCRYPTION_KEY is not set and the deployment is running as "
                "production, so there is no key to encrypt or decrypt with.",
                detail={"environment": env, "setting": "TOKEN_ENCRYPTION_KEY"},
            )
        self._fernet = None
        self._dev_mode = key is None
        if key:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        elif self._dev_mode:
            _logger.warning(
                "token_cipher_dev_key_in_use",
                extra={"environment": env},
            )

    @property
    def is_dev_mode(self) -> bool:
        return self._dev_mode

    def _dev_fernet(self):
        from cryptography.fernet import Fernet

        return Fernet(DEV_FERNET_KEY)

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        fernet = self._fernet or self._dev_fernet()
        return fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        if not ciphertext:
            return ""
        fernet = self._fernet or self._dev_fernet()
        return fernet.decrypt(ciphertext.encode()).decode()


# ---------------------------------------------------------------- uploads ----


def sniff_mime(head: bytes) -> str | None:
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            return mime
    return None


def validate_upload(
    filename: str,
    content_type: str | None,
    data: bytes,
    *,
    max_bytes: int,
    allowed_types: tuple[str, ...],
) -> str:
    """Validate an uploaded file and return its canonical MIME type.

    Checks, in order: size, extension, declared content type, and magic bytes
    for images. Raises ``UploadRejectedError`` on any mismatch.
    """
    import os

    if len(data) > max_bytes:
        raise UploadRejectedError(
            f"file exceeds {max_bytes} bytes", detail={"size": len(data)}
        )
    if not data:
        raise UploadRejectedError("empty file")

    ext = os.path.splitext(filename.lower())[1]
    if ext not in _ALLOWED_EXTENSIONS:
        raise UploadRejectedError(
            f"extension '{ext}' is not allowed", detail={"allowed": sorted(_ALLOWED_EXTENSIONS)}
        )
    expected = _ALLOWED_EXTENSIONS[ext]
    if content_type and content_type != expected and content_type != "application/octet-stream":
        raise UploadRejectedError(
            f"declared content type '{content_type}' does not match extension '{ext}'"
        )
    if allowed_types and expected not in allowed_types:
        raise UploadRejectedError(f"type '{expected}' is not enabled on this deployment")

    sniffed = sniff_mime(data[:16])
    if sniffed is not None and sniffed != expected:
        raise UploadRejectedError(
            f"file contents ({sniffed}) do not match the declared type ({expected})"
        )
    if expected.startswith("image/") and sniffed is None:
        # An image whose bytes match no known signature is not that image.
        raise UploadRejectedError(
            f"file contents do not look like {expected} (no matching signature)"
        )
    if expected == "text/csv":
        try:
            data[:4096].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadRejectedError("csv upload is not valid utf-8") from exc
    return expected
