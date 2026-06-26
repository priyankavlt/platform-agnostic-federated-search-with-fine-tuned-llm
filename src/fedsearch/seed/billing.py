"""The billing system (Stripe-like app database; Supabase in the hosted demo).

It knows customers only by email domain, not by CRM id, so joining it to the CRM goes through the
entity link (``crm.customers.domain`` = ``billing.subscriptions.customer_domain``), as in real companies.
"""

from __future__ import annotations

import random
from datetime import date

from sqlalchemy import Column, Date, Integer, MetaData, String, Table, create_engine

from fedsearch.seed.world import World

PLANS = {"enterprise": "Scale", "growth": "Team", "starter": "Starter"}
INVOICE_MONTHS = [date(2026, m, 1) for m in range(6, 12)]  # Jun..Nov 2026


def build_billing(world: World, seed: int = 11) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    subs, invoices = [], []
    for c in world.customers:
        monthly = round(c["arr_usd"] / 12)
        subs.append({
            "id": f"sub_{c['id'].lower()}",
            "customer_domain": c["domain"],
            "plan": PLANS[c["tier"]],
            "seats": rng.randrange(5, 400),
            "mrr_usd": monthly,
            "status": "past_due" if c["health"] == "churn risk" else "active",
            "started_on": date(2024, rng.randint(1, 12), 1),
        })
        late_from = {"healthy": 99, "at risk": 4, "churn risk": 2}[c["health"]]  # months before invoices go unpaid
        for i, month in enumerate(INVOICE_MONTHS):
            unpaid = i >= late_from or (i == len(INVOICE_MONTHS) - 1 and rng.random() < 0.3)
            due = date(month.year, month.month, 28)
            invoices.append({
                "id": f"in_{len(invoices) + 1:05d}",
                "customer_domain": c["domain"],
                "amount_usd": monthly,
                "issued_on": month,
                "due_on": due,
                "status": ("open" if month == INVOICE_MONTHS[-1] else "overdue") if unpaid else "paid",
            })
    return {"subscriptions": subs, "invoices": invoices}


def write_billing(world: World, url: str, seed: int = 11) -> dict[str, int]:
    data = build_billing(world, seed)
    engine = create_engine(url)
    md = MetaData()
    subscriptions = Table(
        "subscriptions", md,
        Column("id", String, primary_key=True), Column("customer_domain", String), Column("plan", String),
        Column("seats", Integer), Column("mrr_usd", Integer), Column("status", String), Column("started_on", Date),
    )
    invoices = Table(
        "invoices", md,
        Column("id", String, primary_key=True), Column("customer_domain", String), Column("amount_usd", Integer),
        Column("issued_on", Date), Column("due_on", Date), Column("status", String),
    )
    md.drop_all(engine)
    md.create_all(engine)
    with engine.begin() as conn:
        conn.execute(subscriptions.insert(), data["subscriptions"])
        conn.execute(invoices.insert(), data["invoices"])
    return {k: len(v) for k, v in data.items()}
