"""Deterministic lexical retrieval -- the replacement for embedding search.

Why not embeddings
------------------
The previous implementation embedded every memory row and ranked the search
box by cosine distance. With the shipped defaults (``EMBEDDING_PROVIDER=mock``,
``EMBEDDING_DIM=64``) that "embedding" was a hashed bag of words: it carried no
semantics, cost a column, a provider, a reindex endpoint and a pgvector
extension, and its ordering was strictly worse than BM25 on a corpus of a few
hundred short rows. At this scale -- one workspace, tens to low thousands of
items, each ~100 tokens -- lexical scoring is more accurate, explainable, has
no cold start, no reindex step, and no per-write API call.

What this module does
---------------------
* :func:`normalize` -- Unicode/case folding tuned for Arabic: strip harakat and
  tatweel, unify alef/ya/ta-marbuta forms. ``بن البيت`` and ``بِنْ البَيتِ``
  must collide; ``قهوه`` and ``قهوة`` must collide.
* :func:`tokenize` -- unigrams, minus a small bilingual stopword list.
* :func:`bm25_scores` -- Okapi BM25 over the *candidate set only* (the store
  feeds it a recency-bounded window). IDF is computed from that set, so the
  scorer adapts per workspace instead of needing a global corpus.
* :class:`LexicalIndex` -- a tiny in-memory index the caller builds once per
  read and can query repeatedly.

Nothing here is a vector, a model call, or a background job.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

#: Harakat, Quranic annotation, tatweel.
_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
_NON_WORD_RE = re.compile(r"[^\w\u0600-\u06ff]+", re.UNICODE)
_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

#: Arabic letter folding. Order matters: alef variants first.
_LETTER_FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
    }
)

#: Deliberately small: enough to stop "في / من / the / and" dominating BM25,
#: not a linguistic stopword project. Arabic clitics are handled by prefix
#: stripping below instead of a huge list.
STOPWORDS: frozenset[str] = frozenset(
    {
        # Arabic
        "في", "من", "علي", "على", "الي", "عن", "مع", "هو", "هي", "هم", "انا",
        "انت", "كان", "كانت", "يكون", "هذا", "هذه", "ذلك", "التي", "الذي",
        "و", "او", "ثم", "لا", "ما", "كل", "بعد", "قبل", "بين", "حول", "ايضا",
        # English
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "will", "have", "has", "had", "but", "not", "you", "your",
        "our", "their", "its", "about", "into", "than", "then", "them", "they",
        "what", "which", "who", "how", "when", "where", "why", "can", "could",
        "should", "would", "may", "might", "must", "of", "to", "in", "on",
        "at", "by", "as", "is", "be", "been", "being", "or", "an", "a",
    }
)

#: Single-letter Arabic prefixes that attach to words ("بال", "وال", "لل").
_PREFIXES: tuple[str, ...] = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ك", "ف")


def normalize(text: str) -> str:
    """Fold ``text`` into a comparable form (case, Arabic letters, spacing)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _DIACRITICS_RE.sub("", text)
    text = text.translate(_LETTER_FOLD)
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _strip_prefix(token: str) -> str:
    """Drop one common Arabic article/prefix when a stem remains readable."""
    for prefix in _PREFIXES:
        if len(token) > len(prefix) + 2 and token.startswith(prefix):
            return token[len(prefix) :]
    return token


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Normalised unigrams, stopwords removed unless asked to keep them."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        token = _strip_prefix(raw)
        if len(token) < 2 and not token.isdigit():
            continue
        if not keep_stopwords and token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class ScoredText:
    """One scored document. ``score`` is normalised to 0..1."""

    index: int
    score: float
    matched: tuple[str, ...] = ()

    @property
    def matched_text(self) -> str:
        return ", ".join(self.matched)


@dataclass
class LexicalIndex:
    """BM25 over an in-memory candidate set.

    Build it once per read from the rows the store already fetched; query it
    for the ranking and for a human-readable "why did this match".
    """

    documents: Sequence[str]
    k1: float = 1.5
    b: float = 0.75
    _tokens: list[list[str]] = field(default_factory=list, init=False, repr=False)
    _df: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _avg_len: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = [tokenize(doc) for doc in self.documents]
        df: dict[str, int] = {}
        for tokens in self._tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._df = df
        lengths = [len(tokens) for tokens in self._tokens]
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._tokens)
        df = self._df.get(term, 0)
        if n == 0:
            return 0.0
        # BM25+ style floor keeps a term present in every document from
        # contributing a negative score.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[ScoredText]:
        """Score every document against ``query``; normalised to 0..1."""
        query_tokens = tokenize(query)
        if not query_tokens or not self._tokens:
            return []

        raw: list[tuple[int, float, tuple[str, ...]]] = []
        for index, tokens in enumerate(self._tokens):
            if not tokens:
                continue
            length = len(tokens)
            freq: dict[str, int] = {}
            for term in tokens:
                freq[term] = freq.get(term, 0) + 1
            total = 0.0
            matched: list[str] = []
            for term in dict.fromkeys(query_tokens):
                tf = freq.get(term, 0)
                if not tf:
                    continue
                matched.append(term)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (length / (self._avg_len or 1.0))
                )
                total += self._idf(term) * (tf * (self.k1 + 1.0) / denominator)
            if total > 0:
                raw.append((index, total, tuple(matched)))

        if not raw:
            return []
        best = max(score for _, score, _ in raw) or 1.0
        return [
            ScoredText(index=index, score=round(score / best, 4), matched=matched)
            for index, score, matched in sorted(raw, key=lambda row: row[1], reverse=True)
        ]

    def score_map(self, query: str) -> dict[int, ScoredText]:
        return {hit.index: hit for hit in self.score(query)}


def phrase_bonus(query: str, document: str) -> float:
    """Small bonus when the normalised query appears verbatim in the document.

    BM25 on unigrams cannot tell ``"بن البيت"`` (a brand) from the two words
    appearing apart; the phrase check restores that signal cheaply.
    """
    q = normalize(query)
    if len(q) < 4:
        return 0.0
    return 1.0 if q in normalize(document) else 0.0


def contains_any(text: str, needles: Iterable[str]) -> bool:
    folded = normalize(text)
    return any(normalize(needle) in folded for needle in needles)


__all__ = [
    "LexicalIndex",
    "STOPWORDS",
    "ScoredText",
    "contains_any",
    "normalize",
    "phrase_bonus",
    "tokenize",
]
