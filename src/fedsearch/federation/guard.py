"""Guardrails for agent-written SQL: one read-only SELECT over catalog tables, with a row limit."""

from __future__ import annotations

import sqlglot
from sqlglot import exp


class SQLRejected(ValueError):
    """The statement is not allowed; the message is shown to the model so it can fix the query."""


def check_sql(sql: str, allowed_tables: set[str], max_rows: int) -> tuple[str, set[str]]:
    """Validate ``sql`` and return (safe SQL with a LIMIT, catalog tables it reads)."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLRejected(f"SQL does not parse: {e}") from e
    if len(statements) != 1:
        raise SQLRejected("Send exactly one statement.")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise SQLRejected(f"Only SELECT queries are allowed, got {tree.key.upper()}.")

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    used: set[str] = set()
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):  # table functions: read_csv(...), glob(...), ...
            raise SQLRejected(f"Table functions are not allowed: {t.sql('duckdb')}")
        if t.args.get("db") or t.args.get("catalog"):
            raise SQLRejected(f"Use the catalog table names without a schema prefix: {t.sql('duckdb')}")
        name = t.name.lower()
        if name in ctes:
            continue
        if name not in allowed_tables:
            raise SQLRejected(f"Unknown table {t.name!r}. Available tables: {', '.join(sorted(allowed_tables))}.")
        used.add(name)

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_rows)
    return tree.sql("duckdb"), used
