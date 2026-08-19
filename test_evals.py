"""Tests for the scoring code.

The eval harness gates merges, so its metrics need tests of their own -- a
silently broken metric is worse than no metric.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.metrics import (
    citation_correctness,
    citation_precision,
    is_abstention,
    keyword_recall,
    reciprocal_rank,
    retrieval_hit,
    score_case,
)

from rag_assistant.models import Answer, Chunk, Citation, ScoredChunk

DATASET = Path(__file__).resolve().parents[1] / "evals" / "regression_set.json"


def make_answer(
    text: str = "Audit logs are retained for eighteen (18) months. [1]",
    doc_ids: tuple[str, ...] = ("retention_policy",),
    citation_doc: str = "retention_policy",
    with_citation: bool = True,
) -> Answer:
    retrieved = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"{doc_id}::000{i}",
                doc_id=doc_id,
                title=doc_id,
                text="text",
                ordinal=i,
            ),
            score=0.9 - 0.1 * i,
        )
        for i, doc_id in enumerate(doc_ids)
    ]
    citations = (
        [
            Citation(
                marker=1,
                chunk_id=f"{citation_doc}::0000",
                doc_id=citation_doc,
                label=citation_doc,
                quote="text",
                score=0.9,
            )
        ]
        if with_citation
        else []
    )
    return Answer(
        question="q",
        answer=text,
        citations=citations,
        grounded=bool(citations),
        retrieved=retrieved,
        metadata={"latency_ms": 1.0},
    )


def test_abstention_is_detected():
    assert is_abstention(
        make_answer("The provided documents do not contain enough information to answer this question.")
    )
    assert not is_abstention(make_answer())


def test_retrieval_hit_and_miss():
    answer = make_answer(doc_ids=("retention_policy", "access_standard"))
    assert retrieval_hit(answer, ["access_standard"])
    assert not retrieval_hit(answer, ["incident_plan"])


def test_retrieval_hit_is_vacuously_true_for_abstention_cases():
    assert retrieval_hit(make_answer(), [])


def test_reciprocal_rank_reflects_position():
    answer = make_answer(doc_ids=("other_doc", "retention_policy"))
    assert reciprocal_rank(answer, ["retention_policy"]) == 0.5
    assert reciprocal_rank(answer, ["other_doc"]) == 1.0
    assert reciprocal_rank(answer, ["missing_doc"]) == 0.0


def test_keyword_recall_treats_keywords_as_alternatives():
    answer = make_answer("Retained for eighteen (18) months.")
    assert keyword_recall(answer, ["18", "eighteen"]) == 1.0
    assert keyword_recall(answer, ["seven", "7"]) == 0.0
    assert keyword_recall(answer, []) == 1.0


def test_citation_precision_penalizes_unresolvable_citations():
    good = make_answer()
    assert citation_precision(good) == 1.0

    bad = make_answer(citation_doc="ghost_doc")
    assert citation_precision(bad) == 0.0

    assert citation_precision(make_answer(with_citation=False)) == 0.0


def test_citation_correctness_checks_the_source_document():
    assert citation_correctness(make_answer(), ["retention_policy"]) == 1.0
    assert citation_correctness(make_answer(), ["access_standard"]) == 0.0


def test_score_case_passes_a_correct_lookup():
    case = {
        "id": "t1",
        "question": "q",
        "expected_doc_ids": ["retention_policy"],
        "expected_keywords": ["eighteen"],
        "category": "lookup",
    }
    result = score_case(case, make_answer())
    assert result.passed, result.failures


def test_score_case_fails_when_the_fact_is_missing():
    case = {
        "id": "t2",
        "question": "q",
        "expected_doc_ids": ["retention_policy"],
        "expected_keywords": ["seven"],
        "category": "lookup",
    }
    result = score_case(case, make_answer())
    assert not result.passed
    assert any("missing expected fact" in f for f in result.failures)


def test_score_case_fails_when_abstention_was_required():
    case = {"id": "t3", "question": "q", "must_abstain": True, "category": "abstention"}
    result = score_case(case, make_answer())
    assert not result.passed
    assert any("expected abstention" in f for f in result.failures)


def test_score_case_passes_a_correct_abstention():
    case = {"id": "t4", "question": "q", "must_abstain": True, "category": "abstention"}
    answer = make_answer(
        "The provided documents do not contain enough information to answer this question.",
        with_citation=False,
    )
    assert score_case(case, answer).passed


def test_regression_set_is_well_formed():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    ids = [c["id"] for c in dataset["cases"]]

    assert dataset["dataset_version"], "dataset must be versioned"
    assert len(ids) == len(set(ids)), "case ids must be unique"

    for case in dataset["cases"]:
        assert case["question"].strip(), f"{case['id']} has an empty question"
        if case.get("must_abstain"):
            assert not case.get("expected_doc_ids"), f"{case['id']} cannot both abstain and cite"
        else:
            assert case.get("expected_doc_ids"), f"{case['id']} needs expected_doc_ids"
            assert case.get("expected_keywords"), f"{case['id']} needs expected_keywords"


def test_regression_set_covers_abstention():
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    categories = {c.get("category") for c in dataset["cases"]}
    assert "abstention" in categories, "the suite must test refusal, not just recall"
