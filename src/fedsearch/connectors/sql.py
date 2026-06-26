"""Any SQLAlchemy-reachable database (Postgres, MySQL, SQLite, Snowflake, ...)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine, text

from fedsearch.connectors.base import Connector


class SQLConnector(Connector):
    kind = "sql"

    def __init__(self, source: str, config: dict[str, Any]):
        super().__init__(source, config)
        self.engine = create_engine(config["url"])

    def read_stream(self, stream: str, spec: dict[str, Any], state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        query = spec.get("query") or f"SELECT * FROM {spec.get('table', stream)}"
        params: dict[str, Any] = {}
        if (cf := spec.get("cursor_field")) and state.get("cursor"):
            query = f"SELECT * FROM ({query}) AS q WHERE {cf} > :cursor ORDER BY {cf}"
            params["cursor"] = state["cursor"]
        with self.engine.connect() as conn:
            for row in conn.execute(text(query), params).mappings():
                yield dict(row)
