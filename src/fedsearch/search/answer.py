"""Grounded answer generation with enforced, verifiable citations."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext

from fedsearch.config import Settings
from fedsearch.index.store import tokenize
from fedsearch.llm import make_model, model_settings, output_spec
from fedsearch.schemas import Citation, GroundedAnswer, SearchHit
from fedsearch.search.prompts import ANSWER_SYSTEM, render_context


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_in_doc(quote: str, text: str) -> bool:
    return bool(quote.strip()) and _ws(quote) in _ws(text)


class Answerer(ABC):
    @abstractmethod
    async def answer(self, query: str, hits: list[SearchHit]) -> GroundedAnswer: ...


@dataclass
class _Deps:
    docs: dict[str, str]


class LLMAnswerer(Answerer):
    def __init__(self, settings: Settings, model=None, context_k: int = 8):
        self.context_k = context_k
        self.agent: Agent[_Deps, GroundedAnswer] = Agent(
            model or make_model(settings),
            deps_type=_Deps,
            output_type=output_spec(settings, GroundedAnswer),
            instructions=ANSWER_SYSTEM,
            model_settings=model_settings(settings, max_tokens=1024),
            retries=2,
        )

        @self.agent.output_validator
        def _check_citations(ctx: RunContext[_Deps], out: GroundedAnswer) -> GroundedAnswer:
            problems = []
            for c in out.citations:
                if c.doc_id not in ctx.deps.docs:
                    problems.append(f"{c.doc_id!r} is not one of the provided document ids")
                elif not quote_in_doc(c.quote, ctx.deps.docs[c.doc_id]):
                    problems.append(f"quote {c.quote!r} does not appear verbatim in {c.doc_id}")
            if not out.insufficient_context and not out.citations:
                problems.append("an answer must cite at least one document")
            if problems:
                raise ModelRetry("Fix citations: " + "; ".join(problems))
            return out

    async def answer(self, query: str, hits: list[SearchHit]) -> GroundedAnswer:
        ctx_hits = hits[: self.context_k]
        deps = _Deps(docs={h.doc.id: f"{h.doc.title}\n{h.doc.text}" for h in ctx_hits})
        prompt = f"Documents:\n\n{render_context(ctx_hits)}\n\nQuestion: {query}"
        return (await self.agent.run(prompt, deps=deps)).output


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


class ExtractiveAnswerer(Answerer):
    """LLM-free fallback: picks the best-matching sentence from each top document and cites it."""

    def __init__(self, max_sentences: int = 4):
        self.max_sentences = max_sentences

    async def answer(self, query: str, hits: list[SearchHit]) -> GroundedAnswer:
        q = set(tokenize(query))
        picked: list[Citation] = []
        seen: set[str] = set()
        for h in hits:
            sents = [s.strip(" #") for s in _SENT.split(h.doc.text) if len(s.strip(" #")) > 12]
            sents = [s for s in sents if s.lower() not in seen]
            if not sents:
                continue
            best = max(sents, key=lambda s: len(q & set(tokenize(s))))
            seen.add(best.lower())
            picked.append(Citation(doc_id=h.doc.id, quote=best))
            if len(picked) >= self.max_sentences:
                break
        if not picked:
            return GroundedAnswer(answer="No relevant documents were found.", insufficient_context=True)
        body = " ".join(f"{c.quote.rstrip('.')} [{c.doc_id}]." for c in picked)
        return GroundedAnswer(answer=body, citations=picked)


def make_answerer(settings: Settings) -> Answerer:
    return LLMAnswerer(settings) if settings.answerer == "llm" else ExtractiveAnswerer()
