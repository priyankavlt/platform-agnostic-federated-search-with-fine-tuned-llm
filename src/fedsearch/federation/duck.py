"""DuckDB federation core: attach every system read-only and expose catalog tables as views.

Queries run live against the source systems (DuckDB pushes filters and projections down to Postgres and
SQLite), and joins across systems happen inside DuckDB. The agent never sees connection details.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import yaml

from fedsearch.federation.guard import check_sql

_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(value: str) -> str:
    return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)


def _duckdb_attach(alias: str, url: str) -> str:
    """Translate an SQLAlchemy-style URL into a read-only DuckDB ATTACH statement."""
    if url.startswith("sqlite:///"):
        return f"ATTACH '{Path(url.removeprefix('sqlite:///')).resolve()}' AS {alias} (TYPE sqlite, READ_ONLY)"
    if url.startswith(("postgresql", "postgres")):
        dsn = re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql://", url)
        return f"ATTACH '{dsn}' AS {alias} (TYPE postgres, READ_ONLY)"
    raise ValueError(f"Unsupported database URL for {alias}: {url}")


@dataclass
class TableResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    systems: list[str]
    elapsed_ms: int


@dataclass
class Federation:
    catalog: dict
    con: duckdb.DuckDBPyConnection
    max_rows: int = 200
    timeout_s: float = 15.0
    table_system: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_catalog(cls, path: str | Path = "configs/catalog.yaml", **kw) -> Federation:
        catalog = yaml.safe_load(Path(path).read_text())
        con = duckdb.connect(":memory:")
        systems = catalog["systems"]
        for alias, sysdef in systems.items():
            if sysdef["kind"] == "postgres_or_sqlite":
                url = _expand(sysdef["url"])
                con.execute("INSTALL postgres; LOAD postgres;" if url.startswith("postgres") else
                            "INSTALL sqlite; LOAD sqlite;")
                con.execute(_duckdb_attach(alias, url))
        for name, t in catalog["tables"].items():
            sysdef = systems[t["system"]]
            if sysdef["kind"] == "json":
                # Snapshot API exports into a local table (this is the "sync" half of hybrid federation).
                path = Path(_expand(sysdef["path"])).resolve()
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_json_auto('{path}')")
            else:
                con.execute(f"CREATE VIEW {name} AS SELECT * FROM {t['system']}.{t['source']}")
        # From here on the agent's SQL cannot touch local files or change settings.
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        fed = cls(catalog, con, **kw)
        fed.table_system = {n: t["system"] for n, t in catalog["tables"].items()}
        return fed

    def describe(self) -> dict:
        """What the agent sees: systems, tables, columns with meanings, and the legal join keys."""
        systems = self.catalog["systems"]
        return {
            "tables": {
                name: {"system": systems[t["system"]]["label"], "description": t["description"],
                       "columns": t["columns"]}
                for name, t in self.catalog["tables"].items()
            },
            "joins": self.catalog.get("joins", []),
            "dialect": "DuckDB SQL",
        }

    def run(self, sql: str) -> TableResult:
        safe_sql, used = check_sql(sql, set(self.table_system), self.max_rows)
        cur = self.con.cursor()
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        start = time.perf_counter()
        timer.start()
        try:
            rel = cur.execute(safe_sql)
            columns = [d[0] for d in rel.description]
            rows = rel.fetchmany(self.max_rows + 1)
        finally:
            timer.cancel()
        truncated = len(rows) > self.max_rows
        rows = [[_jsonable(v) for v in r] for r in rows[: self.max_rows]]
        labels = sorted({self.catalog["systems"][self.table_system[t]]["label"] for t in used})
        return TableResult(safe_sql, columns, rows, len(rows), truncated, labels,
                           int((time.perf_counter() - start) * 1000))


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)  # dates, decimals, timestamps
