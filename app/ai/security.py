"""Credential-safe redaction shared by AI memory, logs, and chat adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_REDACTED = "[REDACTED]"

_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:api[_-]?(?:key|token)|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"auth(?:orization)?[_-]?token|authorization|cookie|password|passwd|"
    r"client[_-]?secret|secret|bearer)"
)
# The bare key ``token`` is sensitive, but substrings such as ``input_tokens``
# and ``token_count`` are usage metadata, not credentials. Keep these exact
# names separate from the broader credential-shaped pattern above.
_SENSITIVE_EXACT_KEYS = frozenset(
    {"token", "token_hash", "token_encryption_key", "session_token", "bearer_token"}
)
_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_ -]?(?:key|token)|access[_ -]?token|refresh[_ -]?token|id[_ -]?token|"
    r"auth(?:orization)?|auth[_ -]?token|session[_ -]?token|bearer[_ -]?token|"
    r"token|token[_ -]?hash|token[_ -]?encryption[_ -]?key|client[_ -]?secret|secret|"
    r"password|passwd|cookie)\b\s*[:=]\s*)"
    r"(\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;&\]}]+)"
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]+=*")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?(?:key|token)|access[_-]?token|refresh[_-]?token|"
    r"id[_-]?token|auth[_-]?token|session[_-]?token|token|secret|password)=)"
    r"[^&#\s]+"
)
_PROVIDER_TOKEN_RE = re.compile(
    r"(?i)\b(?:gsk_[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,}|"
    r"AIza[A-Za-z0-9_-]{20,}|ya29\.[A-Za-z0-9._-]{20,}|"
    r"eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}(?:\.[A-Za-z0-9_-]{8,})?)\b"
)


def is_sensitive_key(key: object) -> bool:
    """Return true for credential-shaped field names, including nested variants."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return normalized in _SENSITIVE_EXACT_KEYS or bool(_SENSITIVE_KEY_RE.search(str(key)))


def redact_credential_text(value: str) -> str:
    """Mask recognizable credentials in free text without changing ordinary prose."""
    text = str(value)
    text = _ASSIGNMENT_RE.sub(lambda match: match.group(1) + _REDACTED, text)
    text = _BEARER_RE.sub(lambda match: match.group(1) + _REDACTED, text)
    text = _QUERY_SECRET_RE.sub(lambda match: match.group(1) + _REDACTED, text)
    text = _PROVIDER_TOKEN_RE.sub(_REDACTED, text)
    return text


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact secret fields and credential values in arbitrary data."""
    if isinstance(value, Mapping):
        return {
            key: _REDACTED if is_sensitive_key(key) else redact_sensitive_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, str):
        text = redact_credential_text(value)
        # Tool outputs are often JSON strings. Redact nested credential-shaped
        # keys as structured data too, while preserving valid JSON for auditing.
        stripped = value.lstrip()
        if len(value) <= 200_000 and stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                pass
            else:
                return json.dumps(
                    redact_sensitive_data(parsed), ensure_ascii=False, default=str
                )
        return text
    if hasattr(value, "model_dump"):
        return redact_sensitive_data(value.model_dump(mode="json"))
    return value


__all__ = ["is_sensitive_key", "redact_credential_text", "redact_sensitive_data"]
