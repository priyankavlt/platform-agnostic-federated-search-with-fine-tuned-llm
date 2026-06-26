"""End-to-end search engine: plan -> federated retrieve -> grounded answer."""

from __future__ import annotations

from pathlib import Path

from fedsearch.config import Settings
from fedsearch.index.embed import make_embedder
from fedsearch.index.store import HybridIndex
from fedsearch.schemas import SearchResponse
from fedsearch.search.answer import Answerer, make_answerer
from fedsearch.search.federated import FederatedSearcher
from fedsearch.search.planner import Planner, make_planner


class SearchEngine:
    def __init__(self, index: HybridIndex, planner: Planner, answerer: Answerer, top_k: int = 10):
        self.index = index
        self.planner = planner
        self.answerer = answerer
        self.searcher = FederatedSearcher(index)
        self.top_k = top_k

    @classmethod
    def from_settings(cls, settings: Settings) -> SearchEngine:
        index = HybridIndex.load(Path(settings.index_dir), make_embedder(settings))
        return cls(index, make_planner(settings, index.registry), make_answerer(settings), settings.top_k)

    async def search(self, query: str, k: int | None = None, answer: bool = True) -> SearchResponse:
        plan = await self.planner.plan(query)
        hits = self.searcher.search(plan, k or self.top_k)
        ans = await self.answerer.answer(query, hits) if answer else None
        return SearchResponse(query=query, plan=plan, hits=hits, answer=ans)
