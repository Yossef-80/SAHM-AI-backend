"""Print the models the configured API keys can actually use.

Why this exists
---------------
Choosing a model by typing its name is a guessing game: ``gemini-3-flash`` looks
right and returns ``404 NOT_FOUND``, a free OpenRouter model returns ``403``
because it is gated to coding agents, and the settings screen only finds out when
the next chat turn dies. This asks each provider for its own list and marks the
models that are safe choices for an agent that calls tools.

Usage (from the backend folder)::

    python scripts\\list_llm_models.py                 # every provider with a key
    python scripts\\list_llm_models.py --provider gemini
    python scripts\\list_llm_models.py --all           # include non-chat models

It never prints a key, and it changes nothing: every call is a read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Substrings that mean "not a chat model an agent can call tools on".
_NON_CHAT_HINTS = (
    "embedding",
    "embed",
    "aqa",
    "imagen",
    "image",
    "tts",
    "audio",
    "native-audio",
    "live",
    "veo",
    "vision-latest",
    "learnlm",
    "gemma",
    "whisper",
    "moderation",
)

_ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
}

_ENV_KEY = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _read_env_key(name: str) -> str:
    """Read a key from the environment, then from ``back/.env``."""
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        if key.strip() == name:
            return raw.strip().strip('"').strip("'")
    return ""


def _get(url: str, key: str, header: str = "x-goog-api-key") -> dict:
    request = urllib.request.Request(url, headers={header: key})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def _gemini(key: str, show_all: bool) -> list[str]:
    payload = _get(f"{_ENDPOINTS['gemini']}?key={key}&pageSize=200", key, "x-api-key")
    names = [str(item.get("name") or "").replace("models/", "") for item in payload.get("models", [])]
    methods = {
        str(item.get("name") or "").replace("models/", ""): item.get("supportedGenerationMethods") or []
        for item in payload.get("models", [])
    }
    chat = [n for n in names if "generateContent" in methods.get(n, [])]
    if show_all:
        return sorted(chat)
    return sorted(n for n in chat if not any(hint in n for hint in _NON_CHAT_HINTS))


def _openrouter(key: str, show_all: bool) -> list[str]:
    payload = _get(f"{_ENDPOINTS['openrouter']}/models", key, "Authorization")
    models = payload.get("data", [])
    names = [str(item.get("id") or "") for item in models]
    if show_all:
        return sorted(names)
    # A paid model is not gated the way the free tier is; mark the free ones so the
    # "403: only available on agentic harnesses" trap is visible before it is hit.
    return sorted(names)


def _openai_like(provider: str, key: str, base: str) -> list[str]:
    payload = _get(f"{base.rstrip('/')}/models", key, "Authorization")
    return sorted(str(item.get("id") or "") for item in payload.get("data", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(_ENV_KEY), default=None)
    parser.add_argument("--all", action="store_true", help="include non-chat models")
    args = parser.parse_args()

    providers = [args.provider] if args.provider else ["gemini", "openrouter", "openai", "groq"]
    printed = 0
    for provider in providers:
        key = _read_env_key(_ENV_KEY[provider])
        if not key:
            if args.provider:
                print(f"[{provider}] no {_ENV_KEY[provider]} found in the environment or back/.env")
            continue
        print(f"\n=== {provider} (key: {'*' * 6}{key[-4:]}) ===")
        try:
            if provider == "gemini":
                names = _gemini(key, args.all)
            elif provider == "openrouter":
                names = _openrouter(key, args.all)
            elif provider == "openai":
                names = _openai_like(provider, key, "https://api.openai.com/v1")
            else:
                names = _openai_like(provider, key, "https://api.groq.com/openai/v1")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            print(f"  the provider refused the listing ({exc.code}): {detail}")
            continue
        except Exception as exc:  # noqa: BLE001 - a script must not traceback at a user
            print(f"  could not reach the provider: {type(exc).__name__}: {exc}")
            continue
        if not names:
            print("  no usable model found")
            continue
        for name in names:
            marker = "  (free tier: may be gated to coding agents)" if name.endswith(":free") else ""
            print(f"  - {name}{marker}")
        printed += len(names)
        if provider == "gemini":
            print(
                "\n  Put the one you want in back/.env as:\n"
                "    LLM_PROVIDER_STRONG=gemini\n"
                "    LLM_MODEL_STRONG=<model>\n"
                "  ...or pick it in the app under Settings → AI provider."
            )

    if not printed:
        print(
            "\nNo models listed. Check that the key is in back/.env "
            "(GEMINI_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY / GROQ_API_KEY)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
