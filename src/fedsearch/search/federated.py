"""Federated query execution.

1. Plan:     NL query -> QueryPlan (intent, target systems, entities, keywords).
2. Resolve:  plan entities -> canonical keys via the registry learned at ingest ("Globx" -> customer:CUST-0001).
3. Retrieve: hybrid BM25+dense search inside each planned system, plus an entity-anchored list
             (documents tagged with the resolved entities).
4. Expand:   cross-reference. Entities found on the strongest hits (e.g. the ticket ids of Globex's SSO
             tickets) pull in linked documents from *other* systems (chat threads about those tickets).
5. Fuse:     weighted reciprocal-rank fusion -> one ranked list, regardless of which system it came from.
"""

from __future__ import annotations

from collections import Counter

from fedsearch.index.store import HybridIndex, rrf
from fedsearch.schemas import ALL_SOURCES, QueryPlan, SearchHit

# Which entity types are worth following from one system into another. Ticket ids are specific links;
# following error codes or products would fan out to unrelated customers.
XREF_TYPES = ("ticket",)


class FederatedSearcher:
    def __init__(self, index: HybridIndex, per_source_k: int = 20, xref_seed_k: int = 10):
        self.index = index
        self.per_source_k = per_source_k
        self.xref_seed_k = xref_seed_k

    def resolve(self, plan: QueryPlan) -> list[str]:
        keys = []
        for e in plan.entities:
            k = self.index.registry.resolve(e.type, e.value)
            if k and k not in keys:
                keys.append(k)
        return keys

    def search(self, plan: QueryPlan, k: int = 10) -> list[SearchHit]:
        idx = self.index
        sources = [s for s in plan.sources if s in ALL_SOURCES] or list(ALL_SOURCES)
        allowed = set(idx.candidates(sources).tolist())
        text_query = " ".join([plan.rewritten_query, *plan.keywords])
        resolved = self.resolve(plan)
        reasons: dict[int, list[str]] = {}
        lists: list[tuple[list[int], float]] = []

        # Per-system hybrid retrieval (each system contributes its own ranked list).
        for s in sources:
            hits = idx.hybrid(text_query, [s], self.per_source_k)
            lists.append(([i for i, _, _ in hits], 1.0))
            for i, _, r in hits:
                reasons.setdefault(i, []).extend(r)

        # Entity-anchored retrieval: docs tagged with resolved entities, most entities matched first.
        lex = idx.lexical_scores(text_query)
        anchored: list[int] = []
        if resolved:
            match = Counter(i for key in resolved for i in idx.by_entity.get(key, ()) if i in allowed)
            anchored = sorted(match, key=lambda i: (-match[i], -lex[i]))
            lists.append((anchored, 1.0))
            for i in anchored:
                reasons.setdefault(i, []).append("entity:" + ",".join(k for k in resolved if i in idx.by_entity[k]))
        # Docs matching *all* resolved entities are the strongest evidence.
        full = [i for i in anchored if match[i] == len(resolved)] if resolved else []
        if full:
            lists.append((full, 4.0))

        # Cross-reference expansion: follow linkable entities from the top full matches into other systems.
        full = full[: self.xref_seed_k]
        xref: list[int] = []
        for i in full:
            for ent in idx.docs[i].entities:
                if ent.split(":", 1)[0] in XREF_TYPES and ent not in resolved:
                    for j in sorted(idx.by_entity[ent], key=lambda j: -lex[j]):
                        if j in allowed and j not in full and j not in xref:
                            xref.append(j)
                            reasons.setdefault(j, []).append(f"xref:{ent}")
        if xref:
            lists.append((xref, 2.0))

        fused = rrf(lists)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [SearchHit(doc=idx.docs[i], score=round(s, 5), reasons=sorted(set(reasons.get(i, []))))
                for i, s in ranked]
