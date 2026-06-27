"""Materialise the synthetic world into real, separate systems."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from fedsearch.seed.billing import write_billing
from fedsearch.seed.queries import generate_queries, split
from fedsearch.seed.world import World, build_world


def write_crm(world: World, sql_url: str) -> None:
    from fedsearch.seed.billing import sqlalchemy_url

    engine = create_engine(sqlalchemy_url(sql_url))
    md = MetaData()
    customers = Table(
        "customers", md,
        Column("id", String, primary_key=True), Column("name", String), Column("domain", String),
        Column("tier", String), Column("region", String), Column("account_manager", String),
        Column("health", String), Column("arr_usd", Integer), Column("products", String),
        Column("renewal_date", String),
    )
    opportunities = Table(
        "opportunities", md,
        Column("id", String, primary_key=True), Column("customer_id", String), Column("product", String),
        Column("stage", String), Column("amount_usd", Integer), Column("close_date", String), Column("notes", String),
    )
    md.drop_all(engine)
    md.create_all(engine)
    with engine.begin() as conn:
        conn.execute(customers.insert(), world.customers)
        conn.execute(opportunities.insert(), world.opportunities)


def write_all(data_dir: Path, sql_url: str | None = None, seed: int = 7,
              billing_url: str | None = None) -> dict[str, int]:
    world = build_world(seed=seed)
    data_dir.mkdir(parents=True, exist_ok=True)

    write_crm(world, sql_url or f"sqlite:///{data_dir / 'crm.db'}")
    billing = write_billing(world, billing_url or f"sqlite:///{data_dir / 'billing.db'}")

    (data_dir / "support").mkdir(exist_ok=True)
    (data_dir / "support" / "tickets.json").write_text(json.dumps(world.tickets, indent=1))

    for page in world.pages:
        p = data_dir / "docs" / page["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(page["body"])

    (data_dir / "chat").mkdir(exist_ok=True)
    with (data_dir / "chat" / "messages.jsonl").open("w") as f:
        for m in world.messages:
            f.write(json.dumps(m) + "\n")

    splits = split(generate_queries(world))
    (data_dir / "queries").mkdir(exist_ok=True)
    for name, rows in splits.items():
        with (data_dir / "queries" / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return {
        "customers": len(world.customers), "opportunities": len(world.opportunities),
        **{f"billing_{k}": v for k, v in billing.items()},
        "tickets": len(world.tickets), "pages": len(world.pages), "messages": len(world.messages),
        **{f"queries_{k}": len(v) for k, v in splits.items()},
    }
