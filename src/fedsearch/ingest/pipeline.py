"""Extract (connectors -> bus) and index (bus -> normalise -> link -> hybrid index)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fedsearch.connectors.base import build_connector, load_sources_config
from fedsearch.index.embed import make_embedder
from fedsearch.index.store import HybridIndex
from fedsearch.ingest.entities import EntityRegistry
from fedsearch.ingest.normalize import Normalizer
from fedsearch.schemas import Record
from fedsearch.streaming.bus import Bus, encode_key

log = logging.getLogger(__name__)


def extract(bus: Bus, sources_cfg: dict[str, Any], only: list[str] | None = None,
            state_path: Path | None = None) -> dict[str, int]:
    """Run every configured connector and publish its records to ``<prefix>.<source>.<stream>`` topics."""
    state: dict[str, Any] = json.loads(state_path.read_text()) if state_path and state_path.exists() else {}
    counts: dict[str, int] = {}
    for name, cfg in sources_cfg["sources"].items():
        if only and name not in only:
            continue
        conn = build_connector(name, cfg)
        n = 0
        for rec in conn.read(state.setdefault(name, {})):
            bus.publish_record(rec, encode_key(rec, conn.streams[rec.stream].get("primary_key")))
            n += 1
        counts[name] = n
        log.info("extracted %d records from %s", n, name)
    bus.flush()
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=1))
    return counts


def index_from_bus(bus: Bus, sources_cfg: dict[str, Any], settings, idle_timeout: float = 5.0) -> HybridIndex:
    """Consume all pending records and upsert them into the (possibly pre-existing) index."""
    embedder = make_embedder(settings)
    path = Path(settings.index_dir)
    idx = HybridIndex.load(path, embedder) if (path / "meta.json").exists() else HybridIndex(embedder, EntityRegistry())
    norm = Normalizer(sources_cfg, idx.registry)

    records: list[Record] = [r for r in bus.consume_records(idle_timeout) if r.source in sources_cfg["sources"]]
    # Pass 1: learn aliases from every source first, so that a chat message consumed before the CRM
    # topic can still be linked to the right customer.
    for r in records:
        norm.learn_aliases(r)
    docs = [norm.link(norm.to_document(r)) for r in records]
    idx.upsert(docs)
    idx.save(path)
    log.info("indexed %d documents (total %d)", len(docs), len(idx.docs))
    return idx


def ingest(settings, bus: Bus, only: list[str] | None = None) -> tuple[dict[str, int], HybridIndex]:
    cfg = load_sources_config(Path(settings.sources_config))
    counts = extract(bus, cfg, only, Path(settings.index_dir) / "connector_state.json")
    return counts, index_from_bus(bus, cfg, settings)
