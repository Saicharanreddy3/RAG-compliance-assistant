"""Request/response schemas. These are the contract the regression tests pin."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["How long must audit logs be retained?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationModel(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    label: str
    quote: str
    score: float


class RetrievedChunkModel(BaseModel):
    chunk_id: str
    doc_id: str
    section: str | None = None
    score: float
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(
        ..., description="False when the answer is unsupported or the system abstained."
    )
    citations: list[CitationModel]
    retrieved: list[RetrievedChunkModel]
    metadata: dict[str, Any]


class IngestRequest(BaseModel):
    directory: str | None = Field(default=None, description="Defaults to DOCUMENTS_DIR.")
    reset: bool = Field(default=False, description="Clear the index before ingesting.")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    indexed: int
    total_vectors: int


class HealthResponse(BaseModel):
    status: str
    vectors: int
    embedding_provider: str
    llm_provider: str
    prompt_version: str
