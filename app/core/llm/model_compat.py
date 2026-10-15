"""What the OpenAI family expects *today* -- one place, tested directly.

OpenAI changed three things that this codebase had to be told about, and each of
them is a hard ``400`` rather than a degradation:

``max_tokens`` -> ``max_completion_tokens``
    Reasoning models (o-series, GPT-5, GPT-6) reject ``max_tokens`` outright:
    "Unsupported parameter: 'max_tokens' is not supported with this model."
    ``max_completion_tokens`` counts reasoning tokens *and* visible output, so
    the same number buys less than it used to.
``temperature``
    The same models accept only the default. Sending the product's 0.2/0.4 is
    "Unsupported value: 'temperature' does not support 0.4 with this model."
``reasoning_effort`` and tools on Chat Completions
    From GPT-5.4 on, a Chat Completions request that carries *function tools*
    is refused unless ``reasoning_effort`` is ``none`` -- "To use function
    tools, use /v1/responses or set reasoning_effort to 'none'". GPT-6 Astra
    and GPT-6.1 Sol cannot call tools on Chat Completions at all. Every agent in
    this product binds tools, so the default here is the one that keeps tools
    working, and the override is explicit.

Image models have their own two: the GPT Image family accepts only
``1024x1024``, ``1536x1024``, ``1024x1536`` (or ``auto``) -- the DALL-E sizes
this code used to send (``1024x1280``, ``1024x1792``, ``1792x1024``) are
rejected -- and it takes a ``quality`` of ``low``/``medium``/``high``/``auto``
where DALL-E 3 takes ``standard``/``hd``.

Nothing in this module performs I/O, so every rule below is unit-tested rather
than discovered in production.
"""

from __future__ import annotations

from typing import Final

from app.core.logging import get_logger

#: Model families whose Chat Completions contract differs from GPT-4-era ones.
#: Matched on the *bare* name, so ``openai/gpt-5.6-terra`` (OpenRouter style) and
#: ``gpt-5.6-terra`` behave the same.
_REASONING_PREFIXES: Final[tuple[str, ...]] = ("o1", "o3", "o4", "gpt-5", "gpt-6")

#: GPT-6 Astra added a hard rule: no function tools on Chat Completions, at any
#: reasoning effort. These models require the Responses API for tool use.
_TOOLS_UNSUPPORTED_ON_CHAT: Final[tuple[str, ...]] = ("gpt-6-astra", "gpt-6.1-sol")

#: The reasoning values every reasoning model understands. ``minimal`` belongs
#: to the original GPT-5 series only and ``xhigh``/``max`` to the newest ones, so
#: neither is ever defaulted to here. Public, ordered: the settings API validates
#: user input against this exact list, so the accepted values and the forwarded
#: ones cannot drift apart.
AVAILABLE_REASONING_EFFORTS: Final[tuple[str, ...]] = ("none", "low", "medium", "high")

_SAFE_REASONING_EFFORTS: Final[frozenset[str]] = frozenset(AVAILABLE_REASONING_EFFORTS)

#: ``quality`` as the GPT Image family documents it.
_IMAGE_QUALITIES: Final[frozenset[str]] = frozenset({"auto", "low", "medium", "high"})

#: DALL-E 3's own two values. Anything else is a 400 on that model.
_DALLE_QUALITY: Final[dict[str, str]] = {"low": "standard", "medium": "standard", "high": "hd"}

_logger = get_logger(__name__)


def bare_model_name(model: str | None) -> str:
    """``openai/gpt-5.6-terra`` -> ``gpt-5.6-terra``; lowercased, stripped."""
    name = (model or "").strip().lower()
    if "/" in name:
        # Aggregators prefix the vendor ("openai/...", "google/..."). Only strip
        # when what follows looks like a model id, so a local path survives.
        head, _, tail = name.partition("/")
        if head in {"openai", "azure", "azure-openai", "openrouter", "groq", "together", "fireworks"} and tail:
            return tail
    return name


def is_reasoning_model(model: str | None) -> bool:
    """True for the OpenAI families with the divergent Chat Completions contract."""
    name = bare_model_name(model)
    return any(name.startswith(prefix) for prefix in _REASONING_PREFIXES)


def token_limit_param(model: str | None) -> str:
    """The output-token field name this model accepts."""
    return "max_completion_tokens" if is_reasoning_model(model) else "max_tokens"


def supports_temperature(model: str | None) -> bool:
    """Reasoning models accept only the default temperature."""
    return not is_reasoning_model(model)


def chat_completions_tools_unsupported(model: str | None) -> bool:
    """Models that cannot call tools on Chat Completions, whatever we send."""
    name = bare_model_name(model)
    return any(name.startswith(prefix) for prefix in _TOOLS_UNSUPPORTED_ON_CHAT)


def recommended_reasoning_effort(model: str | None, configured: str | None = None) -> str | None:
    """The ``reasoning_effort`` to send, or ``None`` to send the field not at all.

    A configured value wins (with a warning when it is one this build does not
    recognise), because raising the effort is a deliberate quality/cost choice.
    Otherwise reasoning models default to ``none``: every agent here binds
    tools, and from GPT-5.4 on that combination is refused on Chat Completions
    unless the effort is ``none``. Non-reasoning models never carry the field --
    it is not a parameter they know.
    """
    if not is_reasoning_model(model):
        return None
    if configured:
        value = configured.strip().lower()
        if value in _SAFE_REASONING_EFFORTS:
            return value
        _logger.warning(
            "reasoning_effort_ignored",
            extra={"configured": configured, "model": bare_model_name(model)},
        )
    return "none"


def image_size_for_model(aspect_ratio: str | None, model: str | None) -> str:
    """The ``size`` string this model accepts for this aspect ratio.

    The GPT Image family takes two portrait/landscape sizes; DALL-E 3 takes its
    own pair; DALL-E 2 is square-only. Sending a DALL-E size to a GPT Image model
    (or the reverse) is a 400, which is exactly what the previous single mapping
    did for every provider.
    """
    name = bare_model_name(model)
    ratio = (aspect_ratio or "1:1").strip()
    if name.startswith("dall-e-3"):
        return {
            "1:1": "1024x1024",
            "4:5": "1024x1792",
            "2:3": "1024x1792",
            "9:16": "1024x1792",
            "3:2": "1792x1024",
            "16:9": "1792x1024",
        }.get(ratio, "1024x1024")
    if name.startswith("dall-e-2"):
        return "1024x1024"
    # GPT Image (current) and every other OpenAI-compatible image endpoint.
    return {
        "1:1": "1024x1024",
        "4:5": "1024x1536",
        "2:3": "1024x1536",
        "9:16": "1024x1536",
        "3:2": "1536x1024",
        "16:9": "1536x1024",
    }.get(ratio, "1024x1024")


def image_quality_for_model(quality: str | None, model: str | None) -> str | None:
    """The ``quality`` value this model accepts, or ``None`` to send nothing.

    ``auto`` is the honest default: it is what the API documents as "let the
    model choose", so a workspace that never touches the setting is not silently
    paying for ``high``.
    """
    name = bare_model_name(model)
    wanted = (quality or "auto").strip().lower()
    if name.startswith("dall-e-3"):
        # DALL-E 3 has two names for what GPT Image calls three levels.
        return _DALLE_QUALITY.get(wanted, "standard")
    if name.startswith("dall-e-2"):
        return "standard"
    if wanted not in _IMAGE_QUALITIES:
        _logger.warning(
            "image_quality_ignored",
            extra={"configured": quality, "model": name},
        )
        return "auto"
    return wanted


__all__ = [
    "AVAILABLE_REASONING_EFFORTS",
    "bare_model_name",
    "chat_completions_tools_unsupported",
    "image_quality_for_model",
    "image_size_for_model",
    "is_reasoning_model",
    "recommended_reasoning_effort",
    "supports_temperature",
    "token_limit_param",
]
