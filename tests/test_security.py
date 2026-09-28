"""Cross-cutting: SSRF protection for analyze_website, token encryption at rest,
upload validation, and log redaction.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from app.core.errors import BlockedTargetError, UploadRejectedError
from app.core.logging import ContextFilter, configure_logging, get_logger, redact
from app.core.security import (
    TokenCipher,
    check_redirect,
    resolve_and_check,
    validate_upload,
)
from app.config import Settings


# ------------------------------------------------------------------ SSRF guard --


def test_https_url_is_allowed() -> None:
    result = resolve_and_check("https://example.com/pricing")
    assert result.url.startswith("https://")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8080/admin",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/internal",
        "http://192.168.1.1/router",
        "http://172.16.0.9/private",
        "http://[::1]/admin",
        "http://0.0.0.0/",
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/x",
        "http://2130706433/",  # decimal-encoded 127.0.0.1
        "http://0x7f000001/",  # hex-encoded 127.0.0.1
        "http://127.1/",  # short-form 127.0.0.1
    ],
)
def test_private_and_non_http_targets_are_blocked(url: str) -> None:
    with pytest.raises(BlockedTargetError):
        resolve_and_check(url)


def test_redirect_to_a_private_host_is_blocked() -> None:
    with pytest.raises(BlockedTargetError):
        check_redirect("http://169.254.169.254/latest/meta-data/",
                      allowed_schemes=("http", "https"))


def test_redirect_to_https_is_allowed() -> None:
    assert check_redirect("https://example.com/next",
                      allowed_schemes=("http", "https")).url.startswith("https://")


# ------------------------------------------------------------ token encryption --


def test_token_round_trip() -> None:
    cipher = TokenCipher(None)
    secret = "EAAG1234567890-real-looking-meta-token"
    encrypted = cipher.encrypt(secret)
    assert encrypted != secret
    assert secret not in encrypted
    assert cipher.decrypt(encrypted) == secret


def test_two_ciphers_with_the_same_key_interoperate() -> None:
    """A deterministic dev key means the stored token survives a restart."""
    cipher_a = TokenCipher(None)
    cipher_b = TokenCipher(None)
    assert cipher_b.decrypt(cipher_a.encrypt("token-value")) == "token-value"


def test_ciphertext_is_not_the_plaintext() -> None:
    cipher = TokenCipher(None)
    for secret in ("meta-token", "EAAG1234567890", "x" * 200):
        encrypted = cipher.encrypt(secret)
        assert secret not in encrypted
        # Fernet output is base64 with the standard version prefix.
        assert encrypted.startswith("gAAAAA")


# ----------------------------------------------------------- upload validation --


def test_valid_png_upload_is_accepted(tmp_path: Path) -> None:
    # A minimal 1x1 PNG.
    data = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
                         "890000000a49444154789c6300010000050001")
    path = tmp_path / "reference.png"
    path.write_bytes(data)
    mime = validate_upload(
        "reference.png", "image/png", data,
        max_bytes=10 * 1024 * 1024, allowed_types=("image/png", "image/jpeg"),
    )
    assert mime == "image/png"


def test_oversized_upload_is_rejected(tmp_path: Path) -> None:
    data = b"x" * (11 * 1024 * 1024)
    path = tmp_path / "big.png"
    path.write_bytes(data)
    with pytest.raises(UploadRejectedError):
        validate_upload(
            "big.png", "image/png", data,
            max_bytes=10 * 1024 * 1024, allowed_types=("image/png",),
        )


def test_extension_mime_mismatch_is_rejected(tmp_path: Path) -> None:
    data = b"MZ\x90\x00notreallyanimage"
    path = tmp_path / "malicious.png"
    path.write_bytes(data)
    with pytest.raises(UploadRejectedError):
        validate_upload(
            "malicious.png", "image/png", data,
            max_bytes=1024 * 1024, allowed_types=("image/png",),
        )


def test_utf8_csv_is_accepted(tmp_path: Path) -> None:
    content = "name,price\nقهوة,50\n".encode("utf-8")
    path = tmp_path / "products.csv"
    path.write_bytes(content)
    mime = validate_upload(
        "products.csv", "text/csv", content,
        max_bytes=1024 * 1024, allowed_types=("text/csv",),
    )
    assert mime == "text/csv"


# --------------------------------------------------------------- log redaction --


def test_redact_removes_sensitive_keys() -> None:
    payload = {
        "access_token": "EAAG-secret",
        "authorization": "Bearer sk-live-123",
        "api_key": "key-123",
        "password": "hunter2",
        "nested": {"token": "t", "safe": "visible"},
        "list": [{"secret": "s", "ok": 1}],
    }
    redacted = redact(payload)
    assert redacted["access_token"] == "***REDACTED***"
    assert redacted["authorization"] == "***REDACTED***"
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["password"] == "***REDACTED***"
    assert redacted["nested"]["token"] == "***REDACTED***"
    assert redacted["nested"]["safe"] == "visible"
    assert redacted["list"][0]["secret"] == "***REDACTED***"
    assert redacted["list"][0]["ok"] == 1


def test_redact_handles_non_dicts() -> None:
    assert redact("plain") == "plain"
    assert redact(None) is None
    assert redact([1, 2]) == [1, 2]


def test_request_id_is_injected_into_log_records() -> None:
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    ContextFilter().filter(record)
    assert hasattr(record, "request_id")


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO", as_json=True)
    configure_logging("INFO", as_json=True)
    logger = get_logger("app.test")
    logger.info("structured log", extra={"request_id": "req-1", "custom": "value"})
    assert callable(logger.info)


def test_settings_expose_provider_maps(test_settings) -> None:
    assert test_settings.provider_for_tier["fast"] == "mock"
    assert test_settings.provider_for_tier["strong"] == "mock"
    assert test_settings.model_for_tier["fast"]
    assert test_settings.model_for_tier["strong"]
    assert test_settings.is_mock_llm() is True


def test_settings_validate_csv_fields() -> None:
    from app.config import Settings

    settings = Settings(
        cors_origins="http://a.com,http://b.com",
        upload_allowed_types="image/png,text/csv",
    )
    assert settings.cors_origins == ("http://a.com", "http://b.com")
    assert "image/png" in settings.upload_allowed_types
