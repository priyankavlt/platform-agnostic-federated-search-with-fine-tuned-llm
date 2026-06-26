# Product plan: federated search & aggregation over heterogeneous sources

## 0. Decisions (2026-06-26) and POC status

- **Agent framework:** PydanticAI, an existing model-agnostic framework (no custom LLM layer). The model is one
  setting: a bare id uses any OpenAI-compatible endpoint (OpenCode Go by default), a `provider:model` id
  (`anthropic:...`, `openai:...`, `google-gla:...`, `ollama:...`) uses PydanticAI's provider registry.
- **Default model:** OpenCode Go `deepseek-v4.1-flash` (fast, cheap, reliable tool calls); `glm-5.3` and `kimi-k3`
  also work. Pick the final default by running the gold questions.
- **Databases:** Postgres (CRM), Supabase (billing), Cassandra (usage events). **Quickwit dropped.**
- **SaaS sources:** real Gmail and Notion in the public demo, using a dedicated demo account and workspace.
- **Repo:** to be renamed.

**POC (done):** DuckDB federation over CRM (Postgres) + billing (SQLite locally, Supabase via
`FEDSEARCH_BILLING_URL`) + support tickets; SQL guardrails; PydanticAI agent with `run_sql` and `search_text`;
`fedsearch ask`; AG-UI endpoint; CopilotKit chat UI with result-table, SQL and CSV rendering.
**Next:** Cassandra, Gmail and Notion connectors, numeric grounding, native queries with approval, gold-set evals.

## 1. Problem

Company knowledge is spread across systems that no single tool can join. Today a person does the joining:
they ask another team, get a spreadsheet back, and VLOOKUP by hand. This product replaces that person with a
**conversational join engine**: one chat box that

- (a) finds things anywhere,
- (b) computes numbers across systems correctly,
- (c) lets power users write queries in each system's own query language,
- (d) never asks the user to trust an LLM's arithmetic.

Users: sales, support, engineers.

### Example questions

| # | Job | Example | Type |
|---|---|---|---|
| 1 | Find | "What has Globex complained about regarding SSO?" | cross-source search (CRM → support → chat) |
| 2 | Find | "Find the email thread where Initech asked for a renewal discount" | Gmail search, entity-resolved |
| 3 | Aggregate | "ARR of customers with more than 3 open urgent tickets this quarter" | join CRM × support, filter, sum |
| 4 | Aggregate | "Open tickets per account manager, enterprise tier only, as a bar chart" | join + group-by + chart |
| 5 | Aggregate | "Which customers had an error spike in usage events last week and what is their renewal date?" | Cassandra rollup × CRM |
| 6 | Native | "Run in Cassandra: `SELECT * FROM usage_events WHERE customer_id = ? AND day = ?`" | passthrough with approval |
| 7 | Native | "Gmail: `from:*@globex.com newer_than:30d has:attachment`" | passthrough |
| 8 | Native | "`SELECT tier, count(*) FROM customers GROUP BY 1` on the CRM" | passthrough SQL |

### What makes it hard

- **Schema heterogeneity.** "Customer" is `customers.id` in Postgres, `customer_id` in tickets, an email domain
  in Gmail, a relation property in Notion, and `tags.customer` in logs.
- **Entity resolution.** Joins are only correct if "Globex" = `globex.com` = `CUST-0001` everywhere.
- **Aggregation semantics.** "This quarter", "open" and "urgent" must be pinned down in a semantic layer.
- **Pushdown vs. ingestion.** Postgres and Supabase can compute aggregates themselves; Cassandra only per partition. Gmail and Notion can't
  (no group-by API, and roughly 3 requests per second).
- **Freshness.** SaaS syncs lag behind; database queries are live. Users must see the "as of" time.
- **Permissions.** Gmail is per user, so user A must never aggregate user B's inbox.
- **LLM failure modes.** Invented joins, hallucinated numbers, the wrong table, unbounded queries. The fix is
  structural (tools compute, the LLM narrates), not more prompting.

## 2. Architecture: hybrid, with DuckDB as the federation core

The design is hybrid:

- **Push the work down** to sources that can compute (Postgres, Supabase; Cassandra per partition).
- **Sync data locally** from sources that can't (Gmail, Notion).
- **Join everything locally** in DuckDB.
- **The LLM never computes.** It writes SQL; a tool runs it.

| Option | Verdict |
|---|---|
| Ingest everything into one index (the current code) | Keep, but only for unstructured text search |
| Pure query-time federation | Reject: Gmail and Notion have no aggregation APIs, so joins still need a local engine |
| **Hybrid + DuckDB** | **Adopt** |

```
Chat (CopilotKit)  <-- AG-UI -->  PydanticAI Agent
                                       |
         +-----------------------------+-----------------------------+
         | search(q, sources)          | run_sql(duckdb_sql)         | native_query(source, q)  [requires approval]
         v                             v                             v
  Hybrid text index              DuckDB federation core         Per-source adapters
  (existing BM25+dense)          - ATTACH postgres/mysql        - Gmail API (q= syntax)
                                 - synced tables: gmail,        - Notion filter API
                                   notion, cassandra rollups    - Cassandra CQL (per partition)
                                 - entity_links table           - SQL passthrough (read-only)
                                 - semantic views (catalog.yaml)
```

**Key decisions**

1. **DuckDB is the aggregation layer.** It can attach Postgres and MySQL with pushdown, reads Parquet, JSONL and
   CSV, returns Arrow, runs in one process, and fits in 8 GB. Trino needs too much RAM; Steampipe and DataFusion
   are a worse fit.
2. **A semantic layer** lives in `configs/catalog.yaml`. It holds the view definitions, join keys, business
   definitions and freshness mode for each table. Only declared join keys are legal.
   `entity_links(source, local_id, canonical_key)` is the bridge table between systems.
3. **The agent writes DuckDB SQL over the semantic views, and a tool executes it.** `run_sql` has these
   guardrails:
   - sqlglot allowlist: only SELECT/WITH
   - only tables in the catalog
   - LIMIT added automatically
   - timeout
   - row caps
   - every statement logged

   A numeric-grounding check requires every number in the answer to appear in a returned result.
4. **Native queries are tools that need user approval** (PydanticAI `requires_approval`, which maps to an AG-UI
   interrupt). The user can edit or approve the query in the UI before it runs.
5. **MCP** exposes one server with the same toolset in Phase 4, so the tool also works from Claude Desktop.
   There will be no per-source MCP servers.
6. **Text search stays ingest-based.** Cassandra is a *source* (high-volume usage events), not the product's search engine.

**Keep, change or delete in the existing code**

| Area | Action |
|---|---|
| `connectors/` | Keep. Also land records in DuckDB as Parquet tables. |
| `ingest/entities.py` | Keep. Also materialize it as the `entity_links` table. |
| `index/` | Keep for text search. Make fastembed the default. |
| `schemas.py` | Remove the hard-coded `SourceName`/`ALL_SOURCES` lists. Add `TableResult`, `NativeQueryRequest`, `ChartSpec`, `AnswerWithEvidence`. |
| `search/planner.py` | Demote. The agent with tools becomes the main path; the planner stays as the `search` tool's internal plan and as an eval baseline. |
| `search/federated.py` | Keep, as the implementation behind the `search` tool. |
| `search/answer.py` | Add numeric grounding. |
| Kafka (`streaming/`, compose) | Keep, but make it an optional compose profile. |
| `eval/` | Add an aggregation execution-accuracy eval and a native-query eval. |
| `seed/world.py` | Add emails, Notion pages, log events, and realistic dates. |
| `training/`, notebooks | Keep. Retarget to "question → native query" in Phase 5. |
| `api.py` | Add an AG-UI endpoint. Keep `/search`. |

## 3. Toolset critique

| Tool | Verdict | Why |
|---|---|---|
| Airbyte platform | Cut | Won't fit in Docker's 2 GB. The Airbyte *protocol* is already modeled. |
| PyAirbyte | Keep, for Notion | Backs up the "350+ sources" claim. Can't send Gmail search syntax to Gmail. |
| Kafka | Optional profile | Uses ~600 MB. One "live update" demo moment; be ready to explain why it isn't needed at this scale. |
| PydanticAI | Core | Typed tools, `requires_approval`, `StateDeps`, built-in AG-UI adapter, MCP, works with any model. Fix the stale `>=1.0` version pin; add the `[ag-ui,anthropic,mcp]` extras. |
| AG-UI | Keep | `pydantic_ai.ui.ag_ui.AGUIAdapter` is already installed. It needs `ag-ui-protocol>=0.1.19`. |
| CopilotKit | Keep | Provides the chat shell, streaming, tool-call rendering and human-in-the-loop. You still write about 4 React components (ResultTable, Chart, CitationList, ApproveQueryCard), and those *are* the demo. Stopgap UI: `agent.to_web()`. |
| vLLM + Qwen QLoRA | Defer to Phase 5 | A measured experiment on turning questions into native queries. |
| Quickwit | Dropped | Replaced by more widely used databases: Postgres, Supabase, Cassandra. |
| Supabase | Add | Hosted Postgres with its own REST API (PostgREST); the billing system. Free tier. |
| Cassandra | Add | Wide-column store for usage/error events. CQL has no joins, so query per partition and sync daily rollups into DuckDB. Use DataStax Astra (free hosted) to stay within the 2 GB Docker budget. |
| **DuckDB** | Add (core) | The federation engine. |
| Google API client, notion-client | Add | Gmail and Notion adapters that run queries directly in those systems. |
| Logfire / OTel | Add (small) | Traces every tool call and SQL statement. |

**Docker memory budget (2 GB):**

| Service | Approx. memory | When it runs |
|---|---|---|
| Postgres | ~100 MB | always |
| SQLite (second database) | 0 MB | always (file, not a container) |
| MySQL | ~400 MB | only if there's room |
| Cassandra (local) | ~1 GB | only if not using Astra; Kafka off |
| Kafka | ~600 MB | `streaming` profile only |
| Airbyte platform | 4+ GB | never |

## 4. Phases

Each phase ends with something you can demo. The core (Phases 0–4) is about 21–29 days for one developer;
Phase 5 adds 3–5 days.

### Phase 0 — Re-scope, extend the data, write the gold set (1.5–2 days)

- **Build:**
  - Rewrite the top of the README around the new positioning.
  - Write 3 ADRs in `docs/adr/`: hybrid federation, DuckDB, LLM-never-computes.
  - Extend `seed/world.py` with emails (JSONL), Notion-like pages with a customer relation, gateway log events
    (NDJSON), and realistic ARR, renewal and created dates.
  - Write `data/gold/questions.yaml`: about 30 search, aggregation and native questions, each with a
    machine-checkable expected result.
- **Demo:** "I defined the evals before building the system."
- **Done when:** `fedsearch seed` produces all the new data, and every gold question can be checked automatically.

### Phase 1 — Federation core + agent in the CLI (4–5 days)

- **Build:**
  - `federation/duck.py`: attach the databases read-only, register synced tables, build `entity_links`, and
    create views from `catalog.yaml`.
  - `agent/`: one PydanticAI agent with four tools: `describe_catalog`, `search`, `run_sql` (with the
    guardrails) and `propose_chart`.
  - Numeric grounding in the answerer.
  - A `fedsearch ask` CLI command.
  - `eval/aggregation_eval.py`: execution accuracy, SQL validity, tool-call count.
- **Demo:** a cross-source ARR question answered in the terminal, showing the SQL, the result table and which
  sources it touched. A bad query gets refused by the guardrails.
- **Done when:** execution accuracy on the gold aggregation questions is at least 80% with a hosted model, and
  every SQL statement either passes the guardrails or is rejected.

### Phase 2 — Web UI via AG-UI + CopilotKit (4–6 days)

- **Build:**
  - `POST /agui` through `AGUIAdapter.dispatch_request`, with `StateDeps[SessionState]`.
  - A `web/` app in Next.js + CopilotKit with these components:
    - `ResultTable` (sorting, CSV download)
    - `Chart` (Recharts)
    - `SourceBadges` (which systems, live vs. synced, as-of time)
    - `SqlDisclosure` (collapsible generated SQL)
    - `CitationList`
  - Human-in-the-loop:
    - `ApproveQueryCard`, for approving native queries
    - a disambiguation card ("which Globex did you mean?")
  - A sidebar showing source status.
  - Deploy: Vercel for the web app, Fly or Render for the API.
- **Demo:** watch tool calls stream in, a table and chart appear, open the SQL, click a citation, approve a
  drafted native query.
- **Done when:** all 8 example questions run end to end in the UI, and the edit-and-approve round trip works.

### Phase 3 — Real heterogeneous sources (5–7 days)

- **Build:**
  - Gmail: your own Google Cloud OAuth client, header and snippet sync for joins, and a `search_gmail(q)` native
    tool.
  - Notion: sync via PyAirbyte, plus a `notion_query(filter_json)` tool.
  - Cassandra (Astra): seed usage events, a `cassandra_query` native tool (per-partition CQL), and daily rollups synced into DuckDB.
  - A SQLite or MySQL "billing" database.
  - Per-user source allowlist; Gmail restricted to its owner.
  - As-of badges and a `fedsearch sync` command.
  - Optional: a Kafka live-update moment.
  - A `--synthetic` fallback so the demo never depends on live credentials.
- **Demo:** the 5xx-spike × renewal-date question, a Gmail search with approval, and a cited Notion page.
- **Done when:**
  - each native tool has a contract test against recorded fixtures, plus one live smoke test
  - there are 10 new native-query gold questions
  - a test proves a user is denied a source they're not allowed

### Phase 4 — Trust, evals, observability, MCP (3–4 days)

- **Build:**
  - A numeric-grounding eval.
  - A check that every join uses a key declared in the catalog.
  - A static eval report.
  - Logfire tracing.
  - Result caching.
  - `fedsearch mcp` (FastMCP).
  - A model switcher in the UI, compared on the same evals.
- **Demo:** the eval table, a trace of one question, and the same question asked from Claude Desktop.
- **Done when:** grounding is at least 0.95, and the full eval report regenerates with one command.

### Phase 5 (optional) — Fine-tune: question → native query (3–5 days + Colab time)

- **Task:** given the catalog/schema and a question, produce the native query for a named source:
  - DuckDB SQL
  - Cassandra CQL
  - a Gmail `q` string
  - a Notion filter
- **Data:**
  - About 3–5k examples, generated from the gold templates and the synthetic world.
  - Every gold query is validated by actually running it.
  - Unseen templates are held out for testing.
- **Models:** Qwen3.8-27B with 4-bit QLoRA (existing notebook), plus an 8B-class profile.
- **Measure:**
  - Execution accuracy per source.
  - Validity rate.
  - How often users edit the proposed query before approving.
  - Latency and cost, comparing base vs. fine-tuned vs. hosted model.

## 5. Three-minute demo script

- **0:00 — Problem.** "Answering 'which big customers are unhappy' meant messaging three teams and merging
  spreadsheets. This is one chat box over all those systems, and it isn't allowed to do math in its head."
- **0:20 — Search.** Ask about Globex and SSO. Point out entity resolution, the hits across systems, and the
  citations.
- **0:50 — Aggregation.** Ask for ARR by account manager as a bar chart. Show the SQL, the semantic views, the
  source badges, the table and the chart, and mention the numeric-grounding check.
- **1:50 — Native query with approval.** A Cassandra CQL query for a customer's error events. Edit it, approve it, then follow up by
  joining to renewal dates.
- **2:30 — Trust and extensibility.** The eval report, a trace, `sources.yaml` and `catalog.yaml` ("adding a
  source is config"), and MCP.

**Talking points:**

- Hybrid rather than pure pushdown or pure ingestion.
- DuckDB rather than Trino or Steampipe.
- The LLM never computes, enforced by the structure of the system.
- Why Kafka is optional at this scale.
- AG-UI vs. MCP: both are just clients of the same agent.
- Honest weaknesses: rules-based entity resolution, coarse permissions, a hand-written semantic layer.
- Fine-tuning as a trade-off between cost, latency and schema fluency.

## 6. Open questions (recommended defaults)

| Question | Default |
|---|---|
| Hosted LLM | Anthropic first, with a model switcher (OpenAI-compatible endpoints, Ollama, vLLM) |
| Real Gmail/Notion in the public demo? | Synthetic/mocked for the public demo and video; live for local walkthroughs |
| Second database kind | SQLite first, MySQL if there's room |
| Repo positioning | "Federated search & aggregation over heterogeneous sources (optional fine-tuned planner)" |
| Frontend start | `agent.to_web()` in Phase 1, CopilotKit in Phase 2 |
| Kafka | Optional profile, one live-update moment |
| Business definitions | Calendar quarters, defined in `catalog.yaml` |
| Permissions | Per-user source allowlist, Gmail owner-only, no row-level security in v1 |
| Deploy | Vercel + Fly or Render, using synthetic data |
| Fine-tune size | 27B notebook as is, plus an 8B profile |

Sources: [PydanticAI AG-UI](https://pydantic.dev/docs/ai/integrations/ui/ag-ui/),
[CopilotKit × Pydantic AI](https://docs.copilotkit.ai/pydantic-ai),
[CopilotKit generative UI](https://docs.copilotkit.ai/pydantic-ai/generative-ui),
[CopilotKit human-in-the-loop](https://docs.copilotkit.ai/pydantic-ai/human-in-the-loop),
[Supabase docs](https://supabase.com/docs), [DataStax Astra](https://docs.datastax.com/en/astra-db-serverless/),
[Airbyte Gmail](https://docs.airbyte.com/integrations/sources/gmail),
[Airbyte Notion](https://docs.airbyte.com/integrations/sources/notion).
