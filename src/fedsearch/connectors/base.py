"""Connector abstraction.

Every connector emits :class:`Record` objects that mirror Airbyte protocol RECORD messages
(``stream``, ``data``, ``emitted_at``). That makes native connectors and real Airbyte connectors
interchangeable: downstream code (Kafka topics, normalisation, indexing) never knows which produced a record.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from fedsearch.schemas import Record


class Connector(ABC):
    kind: str

    def __init__(self, source: str, config: dict[str, Any]):
        self.source = source
        self.config = config
        self.streams: dict[str, dict[str, Any]] = config.get("streams", {})

    @abstractmethod
    def read_stream(self, stream: str, spec: dict[str, Any], state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield raw rows for one stream. May update ``state`` (incremental cursor)."""

    def read(self, state: dict[str, dict[str, Any]] | None = None) -> Iterator[Record]:
        state = state if state is not None else {}
        for stream, spec in self.streams.items():
            st = state.setdefault(stream, {})
            cursor_field = spec.get("cursor_field")
            for row in self.read_stream(stream, spec, st):
                if cursor_field and row.get(cursor_field) is not None:
                    st["cursor"] = max(str(row[cursor_field]), st.get("cursor", ""))
                yield Record(source=self.source, stream=stream, data=row, emitted_at=int(time.time() * 1000))


_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _interpolate(obj: Any) -> Any:
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    return obj


def load_sources_config(path: Path) -> dict[str, Any]:
    return _interpolate(yaml.safe_load(path.read_text()))


def build_connector(source: str, config: dict[str, Any]) -> Connector:
    kind = config["connector"]
    if kind == "sql":
        from fedsearch.connectors.sql import SQLConnector
        return SQLConnector(source, config)
    if kind == "http":
        from fedsearch.connectors.http import HTTPConnector
        return HTTPConnector(source, config)
    if kind == "files":
        from fedsearch.connectors.files import FilesConnector
        return FilesConnector(source, config)
    if kind == "airbyte":
        from fedsearch.connectors.airbyte import AirbyteConnector
        return AirbyteConnector(source, config)
    raise ValueError(f"unknown connector kind {kind!r} for source {source!r}")
