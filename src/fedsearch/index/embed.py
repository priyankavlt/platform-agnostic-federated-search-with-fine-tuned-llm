"""Dense embedders. ``HashingEmbedder`` is dependency-free and deterministic (good for CI and laptops);
``FastEmbedEmbedder`` runs a real ONNX sentence-embedding model on CPU."""

from __future__ import annotations

import hashlib
import itertools
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 array of shape (len(texts), dim)."""


class HashingEmbedder(Embedder):
    """Feature hashing over word unigrams/bigrams and character trigrams (typo-tolerant)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _feats(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        feats = [f"w:{w}" for w in words] + [f"b:{a}_{b}" for a, b in itertools.pairwise(words)]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._feats(t):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self._m = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._m.embed(["probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.asarray(list(self._m.embed(texts)), dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)


def make_embedder(settings) -> Embedder:
    if settings.embedder == "fastembed":
        return FastEmbedEmbedder(settings.fastembed_model)
    return HashingEmbedder()
