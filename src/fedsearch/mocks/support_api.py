"""A Zendesk-style ticketing REST API over the seeded tickets, so the HTTP connector hits a real service.

Run: ``uvicorn fedsearch.mocks.support_api:app --port 8081``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Query

app = FastAPI(title="Mock Support API")


def _tickets() -> list[dict]:
    path = Path(os.environ.get("FEDSEARCH_DATA_DIR", "data")) / "support" / "tickets.json"
    return sorted(json.loads(path.read_text()), key=lambda t: t["updated_at"])


@app.get("/api/v2/tickets")
def list_tickets(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    updated_since: str | None = None,
) -> dict:
    rows = _tickets()
    if updated_since:
        rows = [t for t in rows if t["updated_at"] > updated_since]
    start = (page - 1) * per_page
    chunk = rows[start:start + per_page]
    return {"tickets": chunk, "next_page": page + 1 if start + per_page < len(rows) else None, "count": len(rows)}
