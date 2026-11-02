"""Score_ctx: semantic alignment between current context and tool description.

The paper uses SimCSE. We provide:
  * EmbeddingScorer - sentence-transformers (closest to the paper)
  * LexicalScorer   - dependency-free fallback for CI / quick demos
"""
from __future__ import annotations

import re
from typing import Any

from .config import SelectorConfig

_STOP = set("""a an and are as at be by can could did do does for from has have how i if
in into is it its of on or our so that the this to was what when where which who will
with you your""".split())

_SYNONYMS = {
    "toxic": {"toxicity", "abusive", "harmful"},
    "toxicity": {"toxic", "abuse", "harmful"},
    "hate": {"hateful", "toxicity"},
    "harassment": {"harass", "bully", "abuse"},
    "moderation": {"moderate", "safety"},
    "explain": {"explanation", "reason"},
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in _STOP]


def _expand(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for t in tokens:
        out |= _SYNONYMS.get(t, set())
    return out


class LexicalScorer:
    name = "lexical"

    def similarity(self, a: str, b: str) -> float:
        ta, tb = set(tokenize(a)), set(tokenize(b))
        if not ta or not tb:
            return 0.0
        ea, eb = _expand(ta), _expand(tb)
        inter = ea & eb
        if not inter:
            return 0.0
        return min(1.0, 0.75 * len(inter) / len(ea) + 0.25 * len(inter) / len(eb))


class EmbeddingScorer:
    """Cosine similarity of sentence embeddings (SimCSE stand-in)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        import numpy as np

        self._np = np
        self._model = SentenceTransformer(model_name)
        self._cache: dict[str, Any] = {}

    def _vec(self, text: str):
        if text not in self._cache:
            self._cache[text] = self._model.encode(text, normalize_embeddings=True)
        return self._cache[text]

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        return float(max(0.0, min(1.0, self._np.dot(self._vec(a), self._vec(b)))))


def build_scorer(config: SelectorConfig):
    if config.embedding_backend != "lexical":
        try:
            return EmbeddingScorer(config.embedding_model)
        except ImportError:
            if config.embedding_backend == "sentence-transformers":
                raise
    return LexicalScorer()