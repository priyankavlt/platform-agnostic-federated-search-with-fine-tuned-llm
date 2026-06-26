"""The federated search & aggregation agent.

One PydanticAI agent with tools; the model only decides *what* to ask. Numbers come from SQL that DuckDB
executes across the attached systems, and text answers come from the hybrid index, so the model never
does arithmetic or invents records.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from fedsearch.config import Settings
from fedsearch.federation.duck import Federation
from fedsearch.federation.guard import SQLRejected

log = logging.getLogger(__name__)

INSTRUCTIONS = """\
You answer questions for sales, support and engineering staff by querying the company's systems.
The data lives in separate systems (CRM, billing, support desk, docs, chat); you federate across them.

Tools:
- run_sql: DuckDB SQL over the catalog tables below. Use it for anything countable, summable, listable or
  filterable, and for joining systems. Join only on the declared join keys.
- search_text: keyword + semantic search over unstructured text (support tickets, runbooks, product docs,
  engineering chat). Use it for "why", "how", "what happened", error codes and symptoms.

Rules:
- Never compute or guess numbers yourself; every number you state must come from a tool result.
- Prefer one well-built SQL query over many small ones. If a query is rejected or errors, fix it and retry.
- Dates are ISO strings in CRM tables (compare as strings or CAST to DATE); billing dates are DATEs.
  Today is {today}.
- Keep the answer short: lead with the result, then 1-3 bullets of context. The user sees result tables
  from run_sql directly, so do not repeat large tables in your reply; summarise them.
- Cite text sources by id in square brackets, e.g. [ZD-1042] or [runbook-e-sso-112].
- End with a line "Sources: ..." naming the systems you used.

Catalog:
{catalog}
"""


@dataclass
class AgentDeps:
    federation: Federation
    search_engine: Any | None = None  # fedsearch.search.engine.SearchEngine, if the index has been built


def make_agent_model(settings: Settings) -> Model | str:
    if ":" in settings.agent_model:  # "anthropic:...", "openai:...", "google-gla:...", "ollama:..."
        return settings.agent_model
    provider = OpenAIProvider(base_url=settings.agent_base_url, api_key=settings.agent_api_key or "EMPTY")
    return OpenAIChatModel(settings.agent_model, provider=provider)


def run_settings(settings: Settings, session_id: str) -> ModelSettings:
    # OpenCode Go routes and caches per conversation via x-opencode-session; other providers ignore it.
    return ModelSettings(temperature=0.0, extra_headers={"x-opencode-session": session_id,
                                                         "User-Agent": settings.agent_user_agent})


def build_agent(settings: Settings, federation: Federation, today: str = "2026-11-26") -> Agent[AgentDeps, str]:
    agent = Agent(
        make_agent_model(settings),
        deps_type=AgentDeps,
        instructions=INSTRUCTIONS.format(today=today, catalog=json.dumps(federation.describe(), indent=1)),
        retries=3,
        name="fedsearch",
    )

    @agent.tool
    def run_sql(ctx: RunContext[AgentDeps], sql: str, title: str) -> dict:
        """Run one read-only DuckDB SELECT across the federated catalog tables.

        Args:
            sql: a single SELECT (CTEs allowed) over catalog table names only.
            title: a short human title for the result table, e.g. "Overdue invoices by customer".
        """
        try:
            r = ctx.deps.federation.run(sql)
        except SQLRejected as e:
            raise ModelRetry(f"Query rejected by guardrails: {e}") from e
        except Exception as e:  # DuckDB binder/catalog errors go back to the model
            raise ModelRetry(f"Query failed: {e}") from e
        log.info("run_sql %s rows=%d %dms", r.sql, r.row_count, r.elapsed_ms)
        return {"title": title, "sql": r.sql, "columns": r.columns, "rows": r.rows, "row_count": r.row_count,
                "truncated": r.truncated, "systems": r.systems, "elapsed_ms": r.elapsed_ms}

    @agent.tool
    async def search_text(ctx: RunContext[AgentDeps], query: str, k: int = 6) -> dict:
        """Search tickets, runbooks, product docs and engineering chat. Returns ranked snippets with ids.

        Args:
            query: natural-language query; include company names, products and error codes if known.
            k: number of results (1-10).
        """
        engine = ctx.deps.search_engine
        if engine is None:
            return {"hits": [], "note": "Text index not built yet (run `fedsearch ingest`)."}
        resp = await engine.search(query, max(1, min(k, 10)), answer=False)
        return {"hits": [
            {"id": h.doc.id, "system": h.doc.source, "title": h.doc.title, "url": h.doc.url,
             "snippet": h.doc.text[:400]}
            for h in resp.hits
        ]}

    return agent


def build_deps(settings: Settings) -> AgentDeps:
    federation = Federation.from_catalog(settings.catalog_config)
    engine = None
    if (Path(settings.index_dir) / "docs.jsonl").exists():
        try:
            from fedsearch.search.engine import SearchEngine

            engine = SearchEngine.from_settings(settings)
        except Exception as e:  # noqa: BLE001
            log.warning("text search disabled: %s", e)
    return AgentDeps(federation, engine)
