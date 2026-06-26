"""HTTP API. Run: ``fedsearch serve`` (or ``uvicorn fedsearch.api:app``).

- ``POST /agui``   AG-UI stream for the chat UI (CopilotKit), backed by the PydanticAI agent
- ``POST /search`` the original federated text search
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydantic_ai.ui.ag_ui import AGUIAdapter

from fedsearch.agent.core import AgentDeps, build_agent, build_deps, run_settings
from fedsearch.config import get_settings
from fedsearch.schemas import SearchResponse

_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    deps: AgentDeps = build_deps(s)
    _state.update(settings=s, deps=deps, agent=build_agent(s, deps.federation))
    yield
    _state.clear()


app = FastAPI(title="Federated Search & Aggregation", lifespan=lifespan)


@app.post("/agui")
async def agui(request: Request) -> Response:
    try:
        thread_id = json.loads(await request.body()).get("threadId") or str(uuid.uuid4())
    except ValueError:
        thread_id = str(uuid.uuid4())
    return await AGUIAdapter.dispatch_request(
        request, agent=_state["agent"], deps=_state["deps"],
        model_settings=run_settings(_state["settings"], thread_id),
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(10, ge=1, le=50)
    answer: bool = True


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    engine = _state["deps"].search_engine
    assert engine is not None, "text index not built; run `fedsearch ingest`"
    return await engine.search(req.query, req.k, req.answer)


@app.get("/health")
def health() -> dict:
    deps: AgentDeps | None = _state.get("deps")
    return {
        "ok": deps is not None,
        "tables": sorted(deps.federation.table_system) if deps else [],
        "documents": len(deps.search_engine.index.docs) if deps and deps.search_engine else 0,
        "model": _state["settings"].agent_model if deps else None,
    }
