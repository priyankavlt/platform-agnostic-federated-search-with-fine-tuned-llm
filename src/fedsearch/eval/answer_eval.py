"""Answer quality: groundedness and citation checks (no LLM judge required).

* citation_validity  - cited doc ids were actually in the retrieved context (no invented sources)
* quote_support      - each citation quote appears verbatim in the cited document
* citation_precision - cited docs are gold-relevant for the query
* groundedness       - share of answer sentences whose content words are covered (>= threshold) by the
                       text of the documents they cite (or, if uncited, by any cited doc)
* fact_recall        - expected key facts (from the synthetic world) present in the answer
"""

from __future__ import annotations

import re

from fedsearch.index.store import tokenize
from fedsearch.schemas import GroundedAnswer, SearchHit
from fedsearch.search.answer import quote_in_doc

_SENT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[([^\]]+)\]")


def evaluate_answer(ans: GroundedAnswer, hits: list[SearchHit], relevant: set[str], facts: list[str],
                    coverage_threshold: float = 0.6) -> dict[str, float]:
    ctx = {h.doc.id: f"{h.doc.title}\n{h.doc.text}" for h in hits}
    cites = ans.citations
    valid = [c for c in cites if c.doc_id in ctx]
    supported = [c for c in valid if quote_in_doc(c.quote, ctx[c.doc_id])]
    cited_ids = {c.doc_id for c in valid}

    sentences = [s for s in _SENT.split(ans.answer) if tokenize(_CITE.sub("", s))]
    grounded = 0
    for s in sentences:
        ids = [i for i in _CITE.findall(s) if i in ctx] or list(cited_ids)
        support = set().union(*(set(tokenize(ctx[i])) for i in ids)) if ids else set()
        toks = set(tokenize(_CITE.sub("", s)))
        if toks and len(toks & support) / len(toks) >= coverage_threshold:
            grounded += 1

    low = ans.answer.lower()
    return {
        "abstained": float(ans.insufficient_context),
        "citation_validity": len(valid) / len(cites) if cites else 0.0,
        "quote_support": len(supported) / len(cites) if cites else 0.0,
        "citation_precision": len(cited_ids & relevant) / len(cited_ids) if cited_ids else 0.0,
        "groundedness": grounded / len(sentences) if sentences else 0.0,
        "fact_recall": sum(1 for f in facts if f.lower() in low) / len(facts) if facts else 0.0,
    }
