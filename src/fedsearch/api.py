"""HTTP API. Run: ``fedsearch serve`` (or ``uvicorn fedsearch.api:app``)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fedsearch.config import get_settings
from fedsearch.schemas import SearchResponse
from fedsearch.search.engine import SearchEngine

_engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine
    _engine = SearchEngine.from_settings(get_settings())
    yield


app = FastAPI(title="Federated Search", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(10, ge=1, le=50)
    answer: bool = True


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    assert _engine is not None
    return await _engine.search(req.query, req.k, req.answer)


@app.get("/health")
def health() -> dict:
    return {"ok": _engine is not None, "documents": len(_engine.index.docs) if _engine else 0}
