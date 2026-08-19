"""Scoring functions for the regression suite.

Deliberately non-LLM: every metric here is deterministic, so a score change
always means a system change, never judge variance. LLM-as-judge scoring can be
layered on top for fluency, but the gates in CI run on these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_assistant.models import Answer

_ABSTAIN_MARKERS = (
    "do not contain enough information",
    "does not contain enough information",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_abstention(answer: Answer) -> bool:
    normalized = normalize(answer.answer)
    return any(marker in normalized for marker in _ABSTAIN_MARKERS)


def retrieval_hit(answer: Answer, expected_doc_ids: list[str], k: int | None = None) -> bool:
    """True if any expected document appears in the top-k retrieved chunks."""
    if not expected_doc_ids:
        return True
    retrieved = answer.retrieved[:k] if k else answer.retrieved
    found = {s.chunk.doc_id for s in retrieved}
    return bool(found & set(expected_doc_ids))


def reciprocal_rank(answer: Answer, expected_doc_ids: list[str]) -> float:
    """MRR contribution for this case; 0.0 when no expected doc was retrieved."""
    if not expected_doc_ids:
        return 1.0
    for rank, scored in enumerate(answer.retrieved, start=1):
        if scored.chunk.doc_id in expected_doc_ids:
            return 1.0 / rank
    return 0.0


def keyword_recall(answer: Answer, expected_keywords: list[str]) -> float:
    """Fraction of expected keyword groups present.

    Keywords are treated as alternatives when they express the same fact
    ("18" / "eighteen"), so this returns 1.0 if any single keyword matches --
    the caller supplies one group per case.
    """
    if not expected_keywords:
        return 1.0
    normalized = normalize(answer.answer)
    return 1.0 if any(normalize(k) in normalized for k in expected_keywords) else 0.0


def citation_precision(answer: Answer) -> float:
    """Fraction of citations that resolve to a chunk that was actually retrieved."""
    if not answer.citations:
        return 0.0
    retrieved_ids = {s.chunk.chunk_id for s in answer.retrieved}
    valid = sum(1 for c in answer.citations if c.chunk_id in retrieved_ids)
    return valid / len(answer.citations)


def citation_correctness(answer: Answer, expected_doc_ids: list[str]) -> float:
    """Fraction of citations pointing at an expected source document."""
    if not expected_doc_ids:
        return 1.0
    if not answer.citations:
        return 0.0
    correct = sum(1 for c in answer.citations if c.doc_id in expected_doc_ids)
    return correct / len(answer.citations)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    question: str
    passed: bool
    failures: list[str]
    scores: dict[str, float]
    answer: str
    latency_ms: float


def score_case(case: dict, answer: Answer) -> CaseResult:
    expected_docs = case.get("expected_doc_ids", [])
    expected_keywords = case.get("expected_keywords", [])
    must_abstain = case.get("must_abstain", False)
    must_be_grounded = case.get("must_be_grounded", not must_abstain)

    abstained = is_abstention(answer)
    failures: list[str] = []

    scores = {
        "retrieval_hit": float(retrieval_hit(answer, expected_docs)),
        "reciprocal_rank": reciprocal_rank(answer, expected_docs),
        "keyword_recall": keyword_recall(answer, expected_keywords) if not must_abstain else 1.0,
        "citation_precision": citation_precision(answer) if not must_abstain else 1.0,
        "citation_correctness": citation_correctness(answer, expected_docs)
        if not must_abstain
        else 1.0,
        "abstained": float(abstained),
    }

    if must_abstain:
        if not abstained:
            failures.append("expected abstention but the system answered")
        if answer.citations:
            failures.append("abstention case produced citations")
    else:
        if abstained:
            failures.append("abstained on an answerable question")
        if scores["retrieval_hit"] < 1.0:
            failures.append(f"no expected document in retrieval (expected {expected_docs})")
        if scores["keyword_recall"] < 1.0:
            failures.append(f"answer missing expected fact (any of {expected_keywords})")
        if must_be_grounded and not answer.grounded:
            failures.append("answer was not grounded")
        if scores["citation_precision"] < 1.0:
            failures.append("answer contains citations that do not resolve to retrieved chunks")

    return CaseResult(
        case_id=case["id"],
        category=case.get("category", "uncategorized"),
        question=case["question"],
        passed=not failures,
        failures=failures,
        scores=scores,
        answer=answer.answer,
        latency_ms=float(answer.metadata.get("latency_ms", 0.0)),
    )
