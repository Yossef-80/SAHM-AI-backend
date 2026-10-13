"""Defaults for the isolated ``app.ai`` provider layer.

Keep model identifiers here so provider modules and agent graphs do not pin a
model independently. Environment overrides are declared in ``settings.py``.
Changing one of these values is a provider/model decision; this completion keeps
all existing defaults unchanged.
"""

DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OPENROUTER_MODEL = "google/gemini-flash-1.5-exp"
TEXT_ONLY_VISION_MODEL_PREFIXES = ("openai/gpt-oss", "gpt-oss")

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENROUTER_MODEL",
    "TEXT_ONLY_VISION_MODEL_PREFIXES",
]
