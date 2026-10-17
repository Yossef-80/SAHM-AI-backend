"""Token measurement shared by the core memory layer and the AI layer.

``app.core`` must not import ``app.ai`` (the dependency only points one way),
so the language-aware counter lives here and ``app.ai.memory.compressor``
imports it from here instead of keeping a second copy.

The fallback matters in this product: the default chat model is a Gemini
flash-lite reached through an OpenAI-compatible endpoint configured *for
English*, and roughly half the traffic is Arabic. A flat ``len(text) / 4``
under-counts Arabic by ~2x, which is how a "1 500 token" session budget turned
into a 3 000-token prompt on the real API. tiktoken is used when installed; the
fallback is script-aware.
"""

from __future__ import annotations

import re

#: Arabic block (0600-06FF), Arabic Supplement (0750-077F), Presentation Forms.
_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+")


def count_tokens(text: str) -> int:
    """Measure ``text`` in tokens.

    Uses ``tiktoken`` (cl100k_base) when available, else a script-aware
    estimate: ~1.8 chars/token for Arabic script, ~3.5 for everything else.
    Never returns 0 for non-empty text.
    """
    if not text:
        return 0
    try:  # optional dependency; the AI layer ships it, the core layer may not
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        pass

    arabic_chars = len(_ARABIC_RE.findall(text))
    other_chars = len(text) - arabic_chars
    return max(1, int((arabic_chars / 1.8) + (other_chars / 3.5)))


def estimate_chars_for_tokens(tokens: int) -> int:
    """Conservative character budget for ``tokens`` (used for hard slices)."""
    return max(1, int(tokens * 3.5))


def fits(text: str, max_tokens: int) -> bool:
    return count_tokens(text) <= max_tokens


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` to ``max_tokens`` without splitting mid-sentence.

    Tries line boundaries first, then sentence boundaries, then words. This is
    what keeps a truncated memory item readable in a prompt instead of ending
    on half a clause.
    """
    if not text or max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text

    kept_lines: list[str] = []
    for line in text.split("\n"):
        candidate = "\n".join([*kept_lines, line])
        if count_tokens(candidate) > max_tokens:
            if not kept_lines:
                return _truncate_first_block(line, max_tokens)
            break
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _truncate_first_block(block: str, max_tokens: int) -> str:
    sentences = _SENTENCE_SPLIT_RE.split(block)
    kept: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*kept, sentence])
        if count_tokens(candidate) <= max_tokens:
            kept.append(sentence)
        else:
            break
    if kept:
        return " ".join(kept)

    words: list[str] = []
    for word in block.split():
        candidate = " ".join([*words, word])
        if count_tokens(candidate) <= max_tokens:
            words.append(word)
        else:
            break
    return " ".join(words)


__all__ = [
    "count_tokens",
    "estimate_chars_for_tokens",
    "fits",
    "truncate_to_tokens",
]
