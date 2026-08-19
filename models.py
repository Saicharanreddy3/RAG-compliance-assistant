"""Domain objects passed between ingestion, retrieval, and generation."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with enough metadata to cite it."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    ordinal: int
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation_label(self) -> str:
        parts = [self.title]
        if self.section:
            parts.append(self.section)
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return " \u2014 ".join(parts)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": asdict(self.chunk), "score": round(self.score, 6)}


@dataclass(frozen=True)
class Citation:
    marker: int
    chunk_id: str
    doc_id: str
    label: str
    quote: str
    score: float


@dataclass(frozen=True)
class Answer:
    """Structured output returned by the API and scored by the eval suite."""

    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    retrieved: list[ScoredChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def cited_doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.citations}
