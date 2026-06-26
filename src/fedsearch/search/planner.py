"""Query planners: natural language -> :class:`QueryPlan`.

``LLMPlanner`` calls the fine-tuned model (served by vLLM) through PydanticAI, which validates the output
against the QueryPlan schema and re-asks the model on validation failure. ``RulePlanner`` is a
deterministic gazetteer/regex baseline used for offline runs and as the comparison point in evals.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from pydantic_ai import Agent

from fedsearch.config import Settings
from fedsearch.index.store import tokenize
from fedsearch.ingest.entities import ID_PATTERNS, EntityRegistry, norm
from fedsearch.llm import make_model, model_settings, output_spec
from fedsearch.schemas import Entity, QueryPlan
from fedsearch.search.prompts import PLANNER_SYSTEM


class Planner(ABC):
    @abstractmethod
    async def plan(self, query: str) -> QueryPlan: ...


ISSUE_WORDS = re.compile(r"\b(problems?|issues?|tickets?|complaints?|trouble|escalations?|reported|bugs?)\b", re.IGNORECASE)
FIX_WORDS = re.compile(r"\b(fix|cause|causes|workaround|runbook|resolve|root cause)\b", re.IGNORECASE)


class RulePlanner(Planner):
    def __init__(self, registry: EntityRegistry):
        self.registry = registry

    def _find(self, etype: str, query: str) -> list[str]:
        padded = f" {norm(query)} "
        hits = {c for a, c in self.registry.aliases.get(etype, {}).items() if len(a) >= 3 and f" {a} " in padded}
        return [self.registry.names.get(f"{etype}:{c}", c) for c in sorted(hits)]

    async def plan(self, query: str) -> QueryPlan:
        ents: list[Entity] = []
        for etype in ("ticket", "error_code"):
            ents += [Entity(type=etype, value=m.group(0).upper()) for m in ID_PATTERNS[etype].finditer(query)]
        customers = self._find("customer", query)
        products = self._find("product", query)
        ents += [Entity(type="customer", value=c) for c in customers]
        ents += [Entity(type="product", value=p) for p in products]
        types = {e.type for e in ents}

        if "ticket" in types:
            intent, sources = "discussion", ["chat", "support"]
        elif "error_code" in types or FIX_WORDS.search(query):
            intent, sources = "troubleshoot", ["docs", "support"]
        elif "customer" in types and ("product" in types or ISSUE_WORDS.search(query)):
            intent, sources = "customer_issues", ["support", "chat"]
        elif "customer" in types:
            intent, sources = "account_overview", ["crm"]
        elif "product" in types:
            intent, sources = "general", ["docs"]
        else:
            intent, sources = "general", []
        kw = list(dict.fromkeys(tokenize(query)))
        rewritten = " ".join([query, *(e.value for e in ents)])
        return QueryPlan(intent=intent, sources=sources, entities=ents, keywords=kw, rewritten_query=rewritten)


class LLMPlanner(Planner):
    def __init__(self, settings: Settings, model=None):
        self.agent = Agent(
            model or make_model(settings),
            output_type=output_spec(settings, QueryPlan),
            instructions=PLANNER_SYSTEM,
            model_settings=model_settings(settings, max_tokens=512),
            retries=2,
        )

    async def plan(self, query: str) -> QueryPlan:
        result = await self.agent.run(query)
        return result.output


def make_planner(settings: Settings, registry: EntityRegistry) -> Planner:
    return LLMPlanner(settings) if settings.planner == "llm" else RulePlanner(registry)
