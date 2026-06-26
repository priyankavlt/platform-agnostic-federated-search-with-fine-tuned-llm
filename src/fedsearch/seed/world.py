"""Deterministic synthetic "company" spread across four heterogeneous systems.

The data is deliberately fragmented the way real enterprise data is:

* CRM (SQL)            knows company names, tiers, account managers, opportunities.
* Support (REST API)   tickets reference ``customer_id`` and ``requester_email`` but never the company name.
* Docs (Markdown)      product pages and runbooks keyed by error code.
* Chat (JSONL export)  engineers talk about ticket IDs, rarely naming the customer or product.

Answering "what problems has Globex had with SSO?" therefore requires resolving Globex -> CUST id in
the CRM, finding that customer's tickets in Support, and following ticket IDs into Chat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

COMPANIES = [
    "Globex", "Initech", "Umbrella Health", "Hooli", "Stark Logistics", "Wayne Analytics", "Soylent Foods",
    "Cyberdyne Systems", "Tyrell Biotech", "Wonka Retail", "Vandelay Imports", "Pied Piper", "Massive Dynamic",
    "Oscorp Energy", "Gringotts Finance", "Nakatomi Trading", "Duff Beverages", "Monarch Solutions",
    "Aperture Labs", "Black Mesa Research", "Virtucon", "Prestige Worldwide", "Dunder Paper", "Bluth Homes",
    "Sterling Cooper", "Los Pollos Group", "Krusty Brands", "Paper Street Co", "Octan Fuels", "Ollivander Tools",
    "Zorg Industries", "Weyland Freight", "Rekall Travel", "Omni Consumer", "Spacely Sprockets", "Cogswell Cogs",
    "Gekko Capital", "Hanso Foundation", "Abstergo Media", "Blue Sun Mining",
]

PRODUCTS = {
    "billing-export": "Billing Export",
    "sso": "Single Sign-On",
    "webhooks": "Webhooks",
    "analytics-dashboard": "Analytics Dashboard",
    "mobile-sync": "Mobile Sync",
    "api-gateway": "API Gateway",
}

# How people actually refer to products in queries and chat.
PRODUCT_NICKNAMES = {
    "billing-export": ["billing exports", "invoice export", "the export"],
    "sso": ["SSO", "SAML login", "okta login"],
    "webhooks": ["webhook", "callbacks"],
    "analytics-dashboard": ["dashboards", "analytics", "reporting"],
    "mobile-sync": ["mobile app sync", "offline sync"],
    "api-gateway": ["gateway", "the API"],
}

# error code -> (product, symptom, root cause, fix)
ERRORS = {
    "E-BX-401": ("billing-export", "export job fails with 401 Unauthorized",
                 "the export service account token expired", "rotate the export token under Settings > Billing > Tokens"),
    "E-BX-509": ("billing-export", "CSV export truncates after 10,000 rows",
                 "pagination cursor is dropped on large ledgers", "enable streamed exports or split by month"),
    "E-SSO-112": ("sso", "SAML assertion rejected as expired",
                  "clock skew between the IdP and our servers exceeds 3 minutes", "sync the IdP clock via NTP and raise allowed skew to 5 minutes"),
    "E-SSO-207": ("sso", "users bounced back to the login page after Okta redirect",
                  "the ACS URL still points at the legacy domain", "update the ACS URL to the new tenant domain"),
    "E-WH-300": ("webhooks", "webhook deliveries stuck in retry",
                 "the receiving endpoint returns 3xx redirects", "point the webhook at the final URL and disable redirects"),
    "E-WH-429": ("webhooks", "webhook deliveries throttled",
                 "customer endpoint rate-limits our dispatcher", "lower max concurrency to 4 in webhook settings"),
    "E-AD-504": ("analytics-dashboard", "dashboard widgets time out",
                 "queries over 90 days scan unpartitioned tables", "enable the partitioned warehouse flag for the workspace"),
    "E-AD-418": ("analytics-dashboard", "charts render blank after timezone change",
                 "cached aggregates are keyed on the old timezone", "clear the aggregate cache from the admin console"),
    "E-MS-610": ("mobile-sync", "offline edits lost after reconnect",
                 "conflict resolver prefers server copy on clock drift", "upgrade the mobile SDK to 4.2 or later"),
    "E-MS-622": ("mobile-sync", "sync stalls at 99 percent",
                 "attachments over 25 MB block the queue", "raise the attachment limit or enable chunked uploads"),
    "E-GW-502": ("api-gateway", "intermittent 502 Bad Gateway responses",
                 "upstream keep-alive is shorter than the gateway idle timeout", "set upstream keep-alive above 75 seconds"),
    "E-GW-431": ("api-gateway", "requests rejected with 431 header too large",
                 "oversized JWTs in the Authorization header", "trim custom claims or enable token compression"),
}

ACCOUNT_MANAGERS = ["Priya Shah", "Marcus Lee", "Elena Petrova", "Tom Okafor", "Sara Lindqvist"]
ENGINEERS = ["dana", "ravi", "mei", "jonas", "aisha", "luca", "noor"]
TIERS = ["enterprise", "growth", "starter"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
HEALTH = ["healthy", "at risk", "churn risk"]
STAGES = ["prospecting", "negotiation", "closed won", "closed lost"]
TICKET_STATUS = ["open", "pending", "solved"]
PRIORITY = ["low", "normal", "high", "urgent"]
CHANNELS = ["#support-escalations", "#eng-oncall", "#customer-success"]


@dataclass
class World:
    customers: list[dict] = field(default_factory=list)
    opportunities: list[dict] = field(default_factory=list)
    tickets: list[dict] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


def _slug(name: str) -> str:
    return "".join(c for c in name.lower().replace(" ", "") if c.isalnum())


def build_world(seed: int = 7, tickets_per_customer: tuple[int, int] = (1, 5)) -> World:
    rng = random.Random(seed)
    w = World()
    base = datetime(2026, 1, 5, tzinfo=UTC)

    for i, name in enumerate(COMPANIES, start=1):
        cid = f"CUST-{i:04d}"
        domain = f"{_slug(name)}.com"
        products = rng.sample(sorted(PRODUCTS), k=rng.randint(2, 4))
        w.customers.append({
            "id": cid,
            "name": name,
            "domain": domain,
            "tier": rng.choice(TIERS),
            "region": rng.choice(REGIONS),
            "account_manager": rng.choice(ACCOUNT_MANAGERS),
            "health": rng.choices(HEALTH, weights=[6, 3, 1])[0],
            "arr_usd": rng.randrange(12, 900) * 1000,
            "products": ",".join(products),
            "renewal_date": (date(2026, 1, 1) + timedelta(days=rng.randint(30, 400))).isoformat(),
        })
        for _ in range(rng.randint(1, 2)):
            product = rng.choice(products)
            w.opportunities.append({
                "id": f"OPP-{len(w.opportunities) + 1:04d}",
                "customer_id": cid,
                "product": product,
                "stage": rng.choice(STAGES),
                "amount_usd": rng.randrange(5, 250) * 1000,
                "close_date": (date(2026, 2, 1) + timedelta(days=rng.randint(0, 300))).isoformat(),
                "notes": f"Expansion of {PRODUCTS[product]} seats.",
            })

    errors_by_product: dict[str, list[str]] = {}
    for code, (product, *_rest) in ERRORS.items():
        errors_by_product.setdefault(product, []).append(code)

    for cust in w.customers:
        for _ in range(rng.randint(*tickets_per_customer)):
            product = rng.choice(cust["products"].split(","))
            code = rng.choice(errors_by_product[product])
            _, symptom, _cause, _fix = ERRORS[code]
            tid = f"TCK-{1000 + len(w.tickets)}"
            created = base + timedelta(hours=rng.randint(0, 24 * 300))
            requester = f"{rng.choice(['ops', 'it', 'admin', 'jordan', 'casey', 'alex'])}@{cust['domain']}"
            w.tickets.append({
                "id": tid,
                "customer_id": cust["id"],
                "requester_email": requester,
                "product": product,
                "subject": f"{PRODUCTS[product]}: {symptom}",
                "description": (
                    f"Since this morning our {PRODUCTS[product]} {symptom}. "
                    f"Error code {code} shows in the logs. Please advise."
                ),
                "error_code": code,
                "status": rng.choice(TICKET_STATUS),
                "priority": rng.choice(PRIORITY),
                "assignee": rng.choice(ENGINEERS),
                "created_at": created.isoformat(),
                "updated_at": (created + timedelta(hours=rng.randint(1, 72))).isoformat(),
            })

    for slug, title in PRODUCTS.items():
        codes = errors_by_product[slug]
        w.pages.append({
            "id": f"product-{slug}",
            "title": f"{title} overview",
            "path": f"products/product-{slug}.md",
            "body": (
                f"# {title} overview\n\n{title} is part of the platform. "
                f"Known error codes: {', '.join(codes)}. See the runbooks for each code.\n\n"
                f"Owner team: {rng.choice(['Platform', 'Integrations', 'Data', 'Identity'])}."
            ),
        })
    for code, (product, symptom, cause, fix) in ERRORS.items():
        w.pages.append({
            "id": f"runbook-{code.lower()}",
            "title": f"Runbook {code}: {symptom}",
            "path": f"runbooks/runbook-{code.lower()}.md",
            "body": (
                f"# Runbook {code}\n\nProduct: {PRODUCTS[product]}\n\n## Symptom\n{symptom.capitalize()}.\n\n"
                f"## Root cause\nUsually {cause}.\n\n## Fix\nTo resolve, {fix}.\n"
            ),
        })

    for t in w.tickets:
        if rng.random() < 0.7:
            ts = datetime.fromisoformat(t["created_at"]) + timedelta(minutes=rng.randint(5, 600))
            lines = [
                f"anyone looking at {t['id']}? priority {t['priority']}",
                f"{t['id']} looks like the known {t['error_code']} issue, following the runbook",
                f"pushed a fix for {t['id']}, waiting on the customer to confirm",
            ]
            for j in range(rng.randint(1, 3)):
                w.messages.append({
                    "id": f"MSG-{len(w.messages) + 1:05d}",
                    "channel": rng.choice(CHANNELS),
                    "user": rng.choice(ENGINEERS),
                    "ts": (ts + timedelta(minutes=7 * j)).isoformat(),
                    "text": lines[j],
                })
    return w
