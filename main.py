"""FastAPI service.

Deployed behind AWS API Gateway; the handler itself is transport-agnostic so it
runs the same under uvicorn locally, in a container on EKS, or via Mangum in
Lambda.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_assistant.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from rag_assistant.config import get_settings
from rag_assistant.generation import prompts
from rag_assistant.generation.pipeline import RAGPipeline
from rag_assistant.ingestion.chunker import chunk_documents
from rag_assistant.ingestion.loader import load_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def set_pipeline(pipeline: RAGPipeline | None) -> None:
    """Test hook for injecting a pipeline backed by fixtures."""
    global _pipeline
    _pipeline = pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting: embeddings=%s llm=%s backend=%s",
        settings.embedding.provider,
        settings.generation.provider,
        settings.retrieval.backend,
    )
    get_pipeline()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RAG-Powered Compliance Document Assistant",
    description="Grounded question answering over compliance documents, with source citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        vectors=pipeline.retriever.store.count(),
        embedding_provider=settings.embedding.provider,
        llm_provider=settings.generation.provider,
        prompt_version=prompts.PROMPT_VERSION,
    )


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(request: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    if pipeline.retriever.store.count() == 0:
        raise HTTPException(status_code=409, detail="Index is empty. POST /ingest first.")

    result = pipeline.answer(request.question, top_k=request.top_k)

    return AskResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        citations=[asdict(c) for c in result.citations],
        retrieved=[
            {
                "chunk_id": s.chunk.chunk_id,
                "doc_id": s.chunk.doc_id,
                "section": s.chunk.section,
                "score": round(s.score, 4),
                "preview": s.chunk.text[:200],
            }
            for s in result.retrieved
        ],
        metadata=result.metadata,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["admin"])
def ingest(request: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    settings = get_settings()
    directory = Path(request.directory) if request.directory else settings.documents_dir

    try:
        documents = load_directory(directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not documents:
        raise HTTPException(status_code=400, detail=f"No supported documents found in {directory}")

    if request.reset:
        pipeline.retriever.store.clear()

    chunks = chunk_documents(documents)
    indexed = pipeline.retriever.index(chunks)

    return IngestResponse(
        documents=len(documents),
        chunks=len(chunks),
        indexed=indexed,
        total_vectors=pipeline.retriever.store.count(),
    )
