"""Embedders: an offline deterministic one and an optional HTTP one.

HashEmbedder needs no network and no key. It hashes words and adjacent word
pairs into a fixed width vector (the "hashing trick"), so texts that share
vocabulary land close together under cosine similarity. It has no notion of
meaning beyond shared words. It exists so the demo and tests are repeatable.

HttpEmbedder calls any OpenAI compatible /embeddings endpoint using only the
standard library. It is used only when EMBEDDINGS_API_KEY is set.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from typing import Callable, Mapping, Optional, Protocol, Sequence

from .textsearch import tokenize

EMBEDDING_DIM = 384


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Zero if either vector is all zeros."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HashEmbedder:
    """Deterministic bag of words embedder. Offline stand in for a real model."""

    def __init__(self, dim: int = EMBEDDING_DIM, bigram_weight: float = 0.5):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.bigram_weight = bigram_weight

    def _bump(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign * weight

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        for tok in tokens:
            self._bump(vector, "w:" + tok, 1.0)
        for left, right in zip(tokens, tokens[1:]):
            self._bump(vector, "b:" + left + "_" + right, self.bigram_weight)
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]


Transport = Callable[[str, str, Mapping[str, str], Optional[bytes]], tuple[int, bytes]]


def urllib_transport(method: str, url: str, headers: Mapping[str, str],
                     body: Optional[bytes]) -> tuple[int, bytes]:
    """Default HTTP transport. Returns (status, body) and never raises on 4xx or 5xx."""
    request = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


class HttpEmbedder:
    """Client for an OpenAI compatible embeddings endpoint."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "text-embedding-3-small", dim: int = EMBEDDING_DIM,
                 transport: Transport = urllib_transport):
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self._transport = transport

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": text, "dimensions": self.dim})
        status, body = self._transport(
            "POST",
            self.base_url + "/embeddings",
            {"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"},
            payload.encode("utf-8"),
        )
        if status >= 300:
            raise RuntimeError(f"embeddings request failed with HTTP {status}")
        vector = json.loads(body)["data"][0]["embedding"]
        if len(vector) != self.dim:
            raise RuntimeError(f"expected {self.dim} dimensions, got {len(vector)}")
        return [float(x) for x in vector]


def embedder_from_env(env: Optional[Mapping[str, str]] = None) -> Embedder:
    """HttpEmbedder when EMBEDDINGS_API_KEY is set, else the offline HashEmbedder."""
    env = os.environ if env is None else env
    key = env.get("EMBEDDINGS_API_KEY", "").strip()
    if not key:
        return HashEmbedder()
    return HttpEmbedder(
        api_key=key,
        base_url=env.get("EMBEDDINGS_BASE_URL", "https://api.openai.com/v1"),
        model=env.get("EMBEDDINGS_MODEL", "text-embedding-3-small"),
    )
