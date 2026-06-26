"""Labelled natural-language queries generated from the synthetic world.

Each example carries the gold :class:`QueryPlan` (fine-tuning target) and the gold relevant document ids
(retrieval eval). Some phrasings are reserved for the test split so evaluation measures generalisation to
unseen wording rather than template memorisation.
"""

from __future__ import annotations

import random
from collections import defaultdict

from fedsearch.schemas import Entity, QueryPlan
from fedsearch.seed.world import ERRORS, PRODUCT_NICKNAMES, PRODUCTS, World

# (template, is_test_only). Placeholders: {company} {product} {code} {ticket} {symptom}
TEMPLATES: dict[str, list[tuple[str, bool]]] = {
    "account_overview": [
        ("How is the {company} account doing?", False),
        ("Give me an overview of {company}.", False),
        ("Who manages {company} and when do they renew?", False),
        ("What's the health and ARR of {company}?", False),
        ("Any open deals with {company}?", False),
        ("Is {company} a churn risk?", True),
        ("quick status on {company} pls", True),
    ],
    "customer_issues": [
        ("What problems has {company} had with {product}?", False),
        ("Show me {company}'s support tickets about {product}.", False),
        ("Has {company} reported any {product} issues?", False),
        ("List {product} complaints from {company}.", False),
        ("Is {company} still having trouble with {product}?", True),
        ("{company} {product} escalations", True),
    ],
    "troubleshoot": [
        ("How do I fix {code}?", False),
        ("What causes error {code}?", False),
        ("Customer is seeing {code}, what's the workaround?", False),
        ("runbook for {code}", False),
        ("We keep getting {code} - what should support do?", True),
        ("{code} root cause and fix", True),
    ],
    "discussion": [
        ("What did the team say about {ticket}?", False),
        ("Any chat discussion on {ticket}?", False),
        ("Who has been working on {ticket} and what's the latest?", False),
        ("Summarize the conversation around {ticket}.", True),
        ("is anyone on {ticket}", True),
    ],
    "general": [
        ("What is {product}?", False),
        ("Which error codes are known for {product}?", False),
        ("Tell me about the {product} feature.", False),
        ("Who owns {product}?", True),
    ],
}

INTENT_SOURCES = {
    "account_overview": ["crm"],
    "customer_issues": ["support", "chat"],
    "troubleshoot": ["docs", "support"],
    "discussion": ["chat", "support"],
    "general": ["docs"],
}


def doc_id(source: str, stream: str, pk: str) -> str:
    return f"{source}:{stream}:{pk}"


def _keywords(*phrases: str) -> list[str]:
    out: list[str] = []
    for p in phrases:
        for tok in p.lower().replace("-", " ").split():
            if len(tok) > 2 and tok not in out:
                out.append(tok)
    return out


def generate_queries(world: World, n_per_intent: int = 400, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    cust_by_id = {c["id"]: c for c in world.customers}
    opps_by_cust: dict[str, list[dict]] = defaultdict(list)
    for o in world.opportunities:
        opps_by_cust[o["customer_id"]].append(o)
    tickets_by_cust_prod: dict[tuple[str, str], list[dict]] = defaultdict(list)
    tickets_by_code: dict[str, list[dict]] = defaultdict(list)
    for t in world.tickets:
        tickets_by_cust_prod[(t["customer_id"], t["product"])].append(t)
        tickets_by_code[t["error_code"]].append(t)
    msgs_by_ticket: dict[str, list[dict]] = defaultdict(list)
    for m in world.messages:
        for t in world.tickets:  # small world; fine
            if t["id"] in m["text"]:
                msgs_by_ticket[t["id"]].append(m)

    examples: list[dict] = []

    def add(intent: str, template: str, test_only: bool, fill: dict, entities: list[Entity],
            relevant: list[str], facts: list[str], kw: list[str]) -> None:
        query = _perturb(rng, template, fill)
        plan = QueryPlan(
            intent=intent,  # type: ignore[arg-type]
            sources=INTENT_SOURCES[intent],  # type: ignore[arg-type]
            entities=entities,
            keywords=kw,
            rewritten_query=_rewrite(intent, fill),
        )
        examples.append({
            "query": query,
            "intent": intent,
            "template": template,
            "test_only": test_only,
            "plan": plan.model_dump(),
            "relevant_doc_ids": relevant,
            "answer_facts": facts,
        })

    for _ in range(n_per_intent):
        # account_overview
        c = rng.choice(world.customers)
        tpl, test_only = rng.choice(TEMPLATES["account_overview"])
        rel = [doc_id("crm", "customers", c["id"])] + [doc_id("crm", "opportunities", o["id"]) for o in opps_by_cust[c["id"]]]
        add("account_overview", tpl, test_only, {"company": c["name"]}, [Entity(type="customer", value=c["name"])],
            rel, [c["health"], c["account_manager"]], _keywords(c["name"], "account"))

        # customer_issues: only pairs that actually have tickets
        (cid, prod), ts = rng.choice(sorted(tickets_by_cust_prod.items()))
        c = cust_by_id[cid]
        tpl, test_only = rng.choice(TEMPLATES["customer_issues"])
        rel = [doc_id("support", "tickets", t["id"]) for t in ts]
        rel += [doc_id("chat", "messages", m["id"]) for t in ts for m in msgs_by_ticket[t["id"]]]
        add("customer_issues", tpl, test_only, {"company": c["name"], "product": PRODUCTS[prod]},
            [Entity(type="customer", value=c["name"]), Entity(type="product", value=PRODUCTS[prod])],
            rel, [t["id"] for t in ts], _keywords(c["name"], PRODUCTS[prod]))

        # troubleshoot
        code = rng.choice(sorted(ERRORS))
        tpl, test_only = rng.choice(TEMPLATES["troubleshoot"])
        rel = [doc_id("docs", "pages", f"runbook-{code.lower()}")]
        rel += [doc_id("support", "tickets", t["id"]) for t in tickets_by_code[code]]
        add("troubleshoot", tpl, test_only, {"code": code}, [Entity(type="error_code", value=code)],
            rel, [ERRORS[code][3]], _keywords(code, "fix"))

        # discussion: tickets that have chat
        tid = rng.choice(sorted(msgs_by_ticket))
        tpl, test_only = rng.choice(TEMPLATES["discussion"])
        rel = [doc_id("chat", "messages", m["id"]) for m in msgs_by_ticket[tid]] + [doc_id("support", "tickets", tid)]
        add("discussion", tpl, test_only, {"ticket": tid}, [Entity(type="ticket", value=tid)],
            rel, [tid], _keywords(tid))

        # general
        slug = rng.choice(sorted(PRODUCTS))
        tpl, test_only = rng.choice(TEMPLATES["general"])
        add("general", tpl, test_only, {"product": PRODUCTS[slug]}, [Entity(type="product", value=PRODUCTS[slug])],
            [doc_id("docs", "pages", f"product-{slug}")], [PRODUCTS[slug]], _keywords(PRODUCTS[slug]))

    # Dedupe identical queries, keep first.
    seen: set[str] = set()
    uniq = []
    for ex in examples:
        if ex["query"] not in seen:
            seen.add(ex["query"])
            uniq.append(ex)
    return uniq


def _typo(rng: random.Random, word: str) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _perturb(rng: random.Random, template: str, fill: dict, p: float = 0.35) -> str:
    """Make queries look like real user input: nicknames, typos, lowercase, no punctuation.

    Gold plans keep canonical entity names, so the model must learn to normalise surface forms.
    """
    surface = dict(fill)
    if "product" in surface and rng.random() < p:
        slug = next(s for s, name in PRODUCTS.items() if name == surface["product"])
        surface["product"] = rng.choice(PRODUCT_NICKNAMES[slug])
    if "company" in surface and rng.random() < p * 0.6:
        words = surface["company"].split()
        j = rng.randrange(len(words))
        words[j] = _typo(rng, words[j])
        surface["company"] = " ".join(words)
    q = template.format(**surface)
    if rng.random() < p:
        q = q.lower()
    if rng.random() < p:
        q = q.rstrip("?.!")
    return q


def _rewrite(intent: str, fill: dict) -> str:
    return {
        "account_overview": "account overview for {company}: health, tier, account manager, renewal, opportunities",
        "customer_issues": "support tickets and discussion from {company} about {product}",
        "troubleshoot": "root cause and fix for error {code}",
        "discussion": "chat discussion and status of ticket {ticket}",
        "general": "{product} overview and known error codes",
    }[intent].format(**fill)


def split(examples: list[dict], test_frac: float = 0.1, val_frac: float = 0.07, seed: int = 3) -> dict[str, list[dict]]:
    """Test = every held-out-phrasing example + a random ``test_frac`` of the seen-phrasing ones, so the eval
    reports both generalisation to new wording and accuracy on familiar wording."""
    rng = random.Random(seed)
    held = [e for e in examples if e["test_only"]]
    rest = [e for e in examples if not e["test_only"]]
    rng.shuffle(rest)
    n_test = int(len(rest) * test_frac)
    n_val = int(len(rest) * val_frac)
    return {"test": held + rest[:n_test], "val": rest[n_test:n_test + n_val], "train": rest[n_test + n_val:]}
