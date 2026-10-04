"""Errors raised by the AI layer's providers.

Tools catch ``ProviderError`` and turn it into a ``status="unavailable"``
result, so a missing key or a dead tunnel never crashes an agent run.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for every provider failure."""


class ProviderNotConfiguredError(ProviderError):
    """A provider was selected but its URL / key / model is missing in ``.env``."""


class ProviderNotInstalledError(ProviderError):
    """A provider was selected but its Python package is not installed."""

    def __init__(self, provider: str, package: str) -> None:
        super().__init__(
            f"Provider '{provider}' needs the '{package}' package. "
            f"Install it with:  uv add {package}"
        )
        self.provider = provider
        self.package = package


class ProviderRequestError(ProviderError):
    """The provider was reachable but the request failed (HTTP error, timeout...)."""


__all__ = [
    "ProviderError",
    "ProviderNotConfiguredError",
    "ProviderNotInstalledError",
    "ProviderRequestError",
]
