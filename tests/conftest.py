from __future__ import annotations

import json

import httpx
import pytest

from fedsearch.config import Settings
from fedsearch.connectors import base as connectors_base
from fedsearch.connectors.http import HTTPConnector
from fedsearch.ingest.pipeline import ingest
from fedsearch.search.answer import ExtractiveAnswerer
from fedsearch.search.engine import SearchEngine
from fedsearch.search.planner import RulePlanner
from fedsearch.seed.write import write_all
from fedsearch.streaming.bus import MemoryBus


@pytest.fixture(scope="session")
def world_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    write_all(d / "data")
    return d


@pytest.fixture(scope="session")
def settings(world_dir, monkeypatch_session):
    monkeypatch_session.setenv("FEDSEARCH_DATA_DIR", str(world_dir / "data"))
    monkeypatch_session.setenv("FEDSEARCH_CRM_URL", f"sqlite:///{world_dir / 'data' / 'crm.db'}")
    return Settings(data_dir=world_dir / "data", index_dir=world_dir / "index", bus="memory")


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def index(settings, world_dir, monkeypatch_session):
    """Full pipeline: connectors -> memory bus -> normalise -> index. The support REST API is mocked."""
    tickets = json.loads((world_dir / "data" / "support" / "tickets.json").read_text())

    def handler(req: httpx.Request) -> httpx.Response:
        page, per = int(req.url.params["page"]), int(req.url.params["per_page"])
        chunk = tickets[(page - 1) * per: page * per]
        return httpx.Response(200, json={"tickets": chunk, "next_page": page + 1 if page * per < len(tickets) else None})

    real = connectors_base.build_connector

    def build(source, cfg):
        if cfg["connector"] == "http":
            return HTTPConnector(source, cfg, client=httpx.Client(base_url="http://mock", transport=httpx.MockTransport(handler)))
        return real(source, cfg)

    monkeypatch_session.setattr("fedsearch.ingest.pipeline.build_connector", build)
    _, idx = ingest(settings, MemoryBus("t"))
    return idx


@pytest.fixture(scope="session")
def engine(index):
    return SearchEngine(index, RulePlanner(index.registry), ExtractiveAnswerer())
