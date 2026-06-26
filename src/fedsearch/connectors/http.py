"""Paginated JSON REST APIs (page-number pagination with an optional ``updated_since`` cursor)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from fedsearch.connectors.base import Connector


class HTTPConnector(Connector):
    kind = "http"

    def __init__(self, source: str, config: dict[str, Any], client: httpx.Client | None = None):
        super().__init__(source, config)
        self.client = client or httpx.Client(
            base_url=config["base_url"], headers=config.get("headers", {}), timeout=config.get("timeout", 30)
        )

    def read_stream(self, stream: str, spec: dict[str, Any], state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        page: int | None = 1
        params: dict[str, Any] = {"per_page": spec.get("page_size", 100)}
        if spec.get("cursor_param") and state.get("cursor"):
            params[spec["cursor_param"]] = state["cursor"]
        while page:
            resp = self.client.get(spec["path"], params={**params, "page": page})
            resp.raise_for_status()
            body = resp.json()
            yield from body[spec.get("records_field", stream)]
            page = body.get(spec.get("next_page_field", "next_page"))
