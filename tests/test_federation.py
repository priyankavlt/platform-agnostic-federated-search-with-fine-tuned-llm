"""Federation core (DuckDB over CRM + billing + support), SQL guardrails, and the agent's tool loop."""

from __future__ import annotations

import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from fedsearch.agent.core import AgentDeps, build_agent
from fedsearch.federation.duck import Federation
from fedsearch.federation.guard import SQLRejected, check_sql

TABLES = {"customers", "invoices", "opportunities", "subscriptions", "tickets"}


@pytest.fixture(scope="module")
def fed(world_dir, monkeypatch_session):
    data = world_dir / "data"
    monkeypatch_session.setenv("FEDSEARCH_DATA_DIR", str(data))
    monkeypatch_session.setenv("FEDSEARCH_CRM_URL", f"sqlite:///{data / 'crm.db'}")
    monkeypatch_session.setenv("FEDSEARCH_BILLING_URL", f"sqlite:///{data / 'billing.db'}")
    return Federation.from_catalog("configs/catalog.yaml")


def test_cross_system_join(fed):
    r = fed.run("""
        SELECT c.name, count(*) AS overdue
        FROM customers c JOIN invoices i ON i.customer_domain = c.domain
        WHERE i.status = 'overdue' GROUP BY c.name ORDER BY overdue DESC""")
    assert r.columns == ["name", "overdue"] and r.rows
    assert r.systems == ["Billing (Supabase / Postgres)", "CRM (Postgres)"]
    assert r.sql.endswith("LIMIT 200")


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers",
    "SELECT 1; SELECT 2",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM crm.customers",
    "SELECT * FROM pg_settings",
    "INSERT INTO customers VALUES (1)",
    "COPY customers TO '/tmp/x.csv'",
])
def test_guardrails_reject(sql):
    with pytest.raises(SQLRejected):
        check_sql(sql, TABLES, 200)


def test_guardrails_allow_ctes_and_keep_limit():
    safe, used = check_sql("WITH t AS (SELECT * FROM tickets) SELECT * FROM t LIMIT 5", TABLES, 200)
    assert used == {"tickets"} and safe.endswith("LIMIT 5")


def test_no_file_access_after_setup(fed):
    with pytest.raises(Exception, match="(?i)permission|disabled"):
        fed.con.execute("SELECT * FROM read_text('/etc/hosts')")


async def test_agent_runs_sql_then_answers(fed, settings):
    """Tool loop with a scripted model: rejected SQL is retried, then the table result reaches the answer."""
    calls: list[str] = []

    def model(messages, info: AgentInfo) -> ModelResponse:
        returns = [p.content for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not calls:
            calls.append("bad")
            return ModelResponse(parts=[ToolCallPart("run_sql", {"sql": "DELETE FROM customers", "title": "x"})])
        if len(calls) == 1:
            assert "rejected by guardrails" in str(returns[-1])  # the model sees why and can fix it
            calls.append("good")
            return ModelResponse(parts=[ToolCallPart("run_sql", {
                "sql": "SELECT region, sum(arr_usd) AS arr FROM customers GROUP BY region ORDER BY region",
                "title": "ARR by region"})])
        content = returns[-1]
        rows = (json.loads(content) if isinstance(content, str) else content)["rows"]
        return ModelResponse(parts=[TextPart(f"{len(rows)} regions. Sources: CRM")])

    agent = build_agent(settings, fed)
    with agent.override(model=FunctionModel(model)):
        result = await agent.run("ARR by region?", deps=AgentDeps(fed))
    assert calls == ["bad", "good"]
    assert result.output == "4 regions. Sources: CRM"
    assert json.dumps(fed.describe())  # catalog is serialisable for the instructions
