"""Central configuration.

Every provider is selectable via environment variables so the same code path
runs locally (no cloud credentials) and in AWS. Defaults are the local ones,
which keeps `pytest` and the eval suite runnable on a clean checkout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    min_chunk_chars: int = field(default_factory=lambda: _env_int("MIN_CHUNK_CHARS", 80))


@dataclass(frozen=True)
class RetrievalConfig:
    # "faiss" (local, default) or "pinecone"
    backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "faiss"))
    top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_CANDIDATE_K", 20))
    min_score: float = field(default_factory=lambda: _env_float("RETRIEVAL_MIN_SCORE", 0.15))
    index_path: Path = field(
        default_factory=lambda: Path(_env("FAISS_INDEX_PATH", str(PROJECT_ROOT / "data" / "index")))
    )
    pinecone_index: str = field(default_factory=lambda: _env("PINECONE_INDEX", "compliance-docs"))
    pinecone_namespace: str = field(default_factory=lambda: _env("PINECONE_NAMESPACE", "default"))


@dataclass(frozen=True)
class EmbeddingConfig:
    # "hash" (deterministic, offline), "bedrock", or "openai"
    provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "hash"))
    model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"))
    dimension: int = field(default_factory=lambda: _env_int("EMBEDDING_DIMENSION", 1024))
    batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 16))


@dataclass(frozen=True)
class GenerationConfig:
    # "echo" (offline stub), "bedrock", or "openai"
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "echo"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))
    # Refuse to answer when the best retrieved chunk is below this score.
    grounding_threshold: float = field(
        default_factory=lambda: _env_float("GROUNDING_THRESHOLD", 0.2)
    )


@dataclass(frozen=True)
class Settings:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    documents_dir: Path = field(
        default_factory=lambda: Path(_env("DOCUMENTS_DIR", str(PROJECT_ROOT / "data" / "documents")))
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that patch environment variables."""
    get_settings.cache_clear()
