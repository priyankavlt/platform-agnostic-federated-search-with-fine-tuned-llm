"""End-to-end over the synthetic world: ingestion, cross-referencing, and retrieval quality floors."""

import json

from fedsearch.eval.run import evaluate
from fedsearch.schemas import Entity, QueryPlan


def test_all_sources_indexed_and_linked(index):
    assert {d.source for d in index.docs} == {"crm", "support", "docs", "chat"}
    ticket = next(d for d in index.docs if d.source == "support")
    assert any(e.startswith("customer:") for e in ticket.entities)
    runbook = index.get("docs:pages:runbook-e-sso-112")
    assert runbook and "product:sso" in runbook.entities and "error_code:E-SSO-112" in runbook.entities


def test_cross_reference_company_name_to_chat(engine, index):
    """Chat never names the company; it must be reached via CRM alias -> ticket -> chat."""
    def chat_count(d):
        tkey = next(e for e in d.entities if e.startswith("ticket:"))
        return sum(index.docs[i].source == "chat" for i in index.by_entity[tkey])

    t = next(d for d in index.docs if d.source == "support" and chat_count(d) > 0)
    cust_key = next(e for e in t.entities if e.startswith("customer:"))
    prod_key = next(e for e in t.entities if e.startswith("product:"))
    company = index.registry.names[cust_key]
    product = index.registry.names[prod_key]
    plan = QueryPlan(intent="customer_issues", sources=["support", "chat"],
                     entities=[Entity(type="customer", value=company), Entity(type="product", value=product)],
                     keywords=[], rewritten_query=f"{company} {product} tickets")
    hits = engine.searcher.search(plan, k=15)
    ids = [h.doc.id for h in hits]
    assert t.id in ids[:5]
    chat = [h for h in hits if h.doc.source == "chat"]
    assert chat and any(r.startswith("xref:ticket:") for r in chat[0].reasons)
    assert all(company.lower() not in h.doc.text.lower() for h in chat)


async def test_eval_quality_floor(engine, settings):
    rows = [json.loads(line) for line in (settings.data_dir / "queries" / "test.jsonl").read_text().splitlines()]
    report = await evaluate(engine, rows[:150], k=10)
    o = report["overall"]
    assert o["retrieval_oracle_plan"]["mrr"] > 0.9
    assert o["retrieval_oracle_plan"]["recall@10"] > 0.85
    assert o["answers"]["citation_validity"] == 1.0
    assert o["answers"]["quote_support"] == 1.0
    assert o["planner"]["valid"] == 1.0
