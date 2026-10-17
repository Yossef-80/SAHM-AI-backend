"""Real-time token cost estimator for LLM calls across providers.

Computes estimated USD cost based on published provider pricing per 1M tokens.
Supported models:
- OpenAI (GPT-4o, GPT-4o-mini, o1, o3-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)
- Google Gemini (Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash)
- Groq (Llama 3.3 70B, Llama 3.1 8B, Mixtral)
- DeepSeek (DeepSeek V3, DeepSeek R1)
- Ollama & Mock (100% Free: $0.00)
"""

from __future__ import annotations

from typing import NamedTuple

from app.config import MODEL_PRICING


class ModelPricing(NamedTuple):
    input_usd_per_m: float
    output_usd_per_m: float


#: Backwards-compatible alias. Tariffs are declared in ``app/config.py`` so the
#: "model ids live in config only" invariant stays true.
_PRICING_TABLE: dict[str, ModelPricing] = {
    model: ModelPricing(*rates) for model, rates in MODEL_PRICING.items()
}

def _match_pricing(model_name: str | None) -> ModelPricing:
    """Find the best pricing entry matching the given model name."""
    if not model_name:
        return ModelPricing(0.0, 0.0)

    clean = model_name.strip().lower()

    # Exact match first
    if clean in _PRICING_TABLE:
        return _PRICING_TABLE[clean]

    # Prefix/substring heuristics
    if "mock" in clean:
        return ModelPricing(0.0, 0.0)
    if "ollama" in clean or "qwen" in clean or "llama" in clean and "/" not in clean:
        # Local models running on Ollama are free
        return ModelPricing(0.0, 0.0)

    # Longest key first, so a dated variant of a mini model cannot match the
    # pricier base row that happens to be a prefix of it.
    for key in sorted(_PRICING_TABLE, key=len, reverse=True):
        if key in clean:
            return _PRICING_TABLE[key]

    # Fallback to an economical estimate ($0.50 / $1.50 per 1M)
    return ModelPricing(0.50, 1.50)


def estimate_token_cost(
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float:
    """Calculate the estimated cost in USD for the given token usage.

    Returns:
        float: Estimated cost in USD, rounded to 6 decimal places.
    """
    in_tok = max(0, input_tokens or 0)
    out_tok = max(0, output_tokens or 0)

    pricing = _match_pricing(model)
    in_cost = (in_tok / 1_000_000.0) * pricing.input_usd_per_m
    out_cost = (out_tok / 1_000_000.0) * pricing.output_usd_per_m

    return round(in_cost + out_cost, 6)


def format_cost(cost_usd: float) -> str:
    """Format cost nicely for logs and user interfaces.

    Examples:
        format_cost(0.0) -> "Free ($0.00)"
        format_cost(0.000125) -> "$0.000125"
        format_cost(0.042) -> "$0.0420"
    """
    if cost_usd <= 0.0:
        return "Free ($0.00)"
    if cost_usd < 0.01:
        return f"${cost_usd:.6f}"
    return f"${cost_usd:.4f}"


__all__ = ["ModelPricing", "estimate_token_cost", "format_cost"]
