"""Hybrid (BM25 + dense) index with an entity inverted index, persisted to a directory."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from fedsearch.index.embed import Embedder
from fedsearch.ingest.entities import EntityRegistry
from fedsearch.schemas import Document

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOP = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has", "have", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "s", "show", "that", "the", "their", "them", "there", "this", "to", "us", "was", "we", "what", "when", "which", "who", "with", "you"])


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for t in _TOKEN.findall(text.lower()):
        if "-" in t:
            toks.append(t)  # keep "e-sso-112" whole ...
            toks.extend(p for p in t.split("-") if p not in STOP)  # ... and its parts
        elif t not in STOP:
            toks.append(t)
    return toks


def rrf(rankings: list[tuple[list[int], float]], k: int = 60) -> dict[int, float]:
    """Weighted reciprocal-rank fusion of several ranked lists of doc indices."""
    scores: dict[int, float] = defaultdict(float)
    for ranked, weight in rankings:
        for rank, idx in enumerate(ranked):
            scores[idx] += weight / (k + rank + 1)
    return scores


class HybridIndex:
    def __init__(self, embedder: Embedder, registry: EntityRegistry | None = None):
        self.embedder = embedder
        self.registry = registry or EntityRegistry()
        self.docs: list[Document] = []
        self._pos: dict[str, int] = {}
        self._vecs = np.zeros((0, embedder.dim), dtype=np.float32)
        self._bm25: BM25Okapi | None = None
        self.by_entity: dict[str, set[int]] = defaultdict(set)
        self.by_source: dict[str, set[int]] = defaultdict(set)

    # ---- building -------------------------------------------------------------------------------
    def upsert(self, docs: list[Document]) -> None:
        new_vecs = self.embedder.embed([f"{d.title}\n{d.text}" for d in docs]) if docs else None
        rows: list[np.ndarray] = []
        for d, v in zip(docs, new_vecs if new_vecs is not None else []):
            if d.id in self._pos:
                i = self._pos[d.id]
                for e in self.docs[i].entities:
                    self.by_entity[e].discard(i)
                self.docs[i] = d
                self._vecs[i] = v
            else:
                i = len(self.docs)
                self._pos[d.id] = i
                self.docs.append(d)
                rows.append(v)
            for e in d.entities:
                self.by_entity[e].add(i)
            self.by_source[d.source].add(i)
        if rows:
            self._vecs = np.vstack([self._vecs, np.stack(rows)])
        self._bm25 = None

    def _ensure_bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(f"{d.title} {d.text}") or ["_"] for d in self.docs])
        return self._bm25

    # ---- querying -------------------------------------------------------------------------------
    def get(self, doc_id: str) -> Document | None:
        i = self._pos.get(doc_id)
        return self.docs[i] if i is not None else None

    def candidates(self, sources: list[str] | None) -> np.ndarray:
        if not sources:
            return np.arange(len(self.docs))
        return np.array(sorted(set().union(*(self.by_source.get(s, set()) for s in sources))), dtype=int)

    def lexical_scores(self, query: str) -> np.ndarray:
        if not self.docs:
            return np.zeros(0)
        return np.asarray(self._ensure_bm25().get_scores(tokenize(query)))

    def dense_scores(self, query: str) -> np.ndarray:
        if not self.docs:
            return np.zeros(0)
        return self._vecs @ self.embedder.embed([query])[0]

    def hybrid(self, query: str, sources: list[str] | None, k: int, depth: int = 100) -> list[tuple[int, float, list[str]]]:
        cand = self.candidates(sources)
        if len(cand) == 0:
            return []
        lex = self.lexical_scores(query)[cand]
        den = self.dense_scores(query)[cand]
        lex_rank = [int(cand[j]) for j in np.argsort(-lex)[:depth] if lex[j] > 0]
        den_rank = [int(cand[j]) for j in np.argsort(-den)[:depth]]
        fused = rrf([(lex_rank, 1.0), (den_rank, 1.0)])
        lex_set, den_set = set(lex_rank[:k]), set(den_rank[:k])
        out = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(i, s, [r for r, ok in (("lexical", i in lex_set), ("dense", i in den_set)) if ok]) for i, s in out]

    # ---- persistence ----------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "docs.jsonl").open("w") as f:
            for d in self.docs:
                f.write(d.model_dump_json() + "\n")
        np.save(path / "vectors.npy", self._vecs)
        (path / "registry.json").write_text(json.dumps(self.registry.to_dict()))
        (path / "meta.json").write_text(json.dumps({"embedder": type(self.embedder).__name__, "dim": self.embedder.dim}))

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> HybridIndex:
        meta = json.loads((path / "meta.json").read_text())
        if meta["embedder"] != type(embedder).__name__ or meta["dim"] != embedder.dim:
            raise ValueError(f"index built with {meta}, but current embedder is {type(embedder).__name__}/{embedder.dim}")
        idx = cls(embedder, EntityRegistry.from_dict(json.loads((path / "registry.json").read_text())))
        with (path / "docs.jsonl").open() as f:
            idx.docs = [Document.model_validate_json(line) for line in f if line.strip()]
        idx._vecs = np.load(path / "vectors.npy")
        for i, d in enumerate(idx.docs):
            idx._pos[d.id] = i
            idx.by_source[d.source].add(i)
            for e in d.entities:
                idx.by_entity[e].add(i)
        return idx
