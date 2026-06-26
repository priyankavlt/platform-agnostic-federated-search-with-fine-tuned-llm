"""Shared data contracts. These are also the schemas the fine-tuned model is trained to emit."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceName = Literal["crm", "support", "docs", "chat"]
ALL_SOURCES: tuple[SourceName, ...] = ("crm", "support", "docs", "chat")

EntityType = Literal["customer", "ticket", "product", "error_code", "person"]

Intent = Literal[
    "account_overview",  # status/health of a customer account
    "customer_issues",  # problems a customer has reported
    "troubleshoot",  # how to fix an error / known issue
    "discussion",  # what people said about something
    "general",  # anything else
]


class Record(BaseModel):
    """A raw record as emitted by a connector (mirrors the Airbyte protocol RECORD message)."""

    source: str
    stream: str
    data: dict[str, Any]
    emitted_at: int


class Entity(BaseModel):
    type: EntityType
    value: str = Field(description="Surface form as written, e.g. 'Globex' or 'TCK-1042'.")


class Document(BaseModel):
    """A normalized, source-agnostic searchable unit."""

    id: str  # globally unique: "<source>:<stream>:<pk>"
    source: str
    stream: str
    title: str
    text: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Canonical entity keys, e.g. "customer:CUST-0007", "ticket:TCK-1042", "product:billing-export".
    entities: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Structured interpretation of a natural-language query. The fine-tuned model produces this."""

    intent: Intent
    sources: list[SourceName] = Field(
        description="Systems likely to hold relevant data. Empty means search everything."
    )
    entities: list[Entity] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list, description="Salient search terms, lowercase.")
    rewritten_query: str = Field(description="Self-contained query suitable for retrieval.")


class SearchHit(BaseModel):
    doc: Document
    score: float
    # Why this hit was returned: "lexical", "dense", or "xref:<entity>" for cross-referenced hits.
    reasons: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    doc_id: str
    quote: str = Field(description="Verbatim span copied from the cited document that supports the claim.")


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_context: bool = Field(
        default=False, description="True when the retrieved documents do not contain the answer."
    )


class SearchResponse(BaseModel):
    query: str
    plan: QueryPlan
    hits: list[SearchHit]
    answer: GroundedAnswer | None = None
