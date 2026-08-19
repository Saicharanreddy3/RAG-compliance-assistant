from __future__ import annotations

from rag_assistant.generation import prompts
from rag_assistant.generation.pipeline import RAGPipeline


class StubLLM:
    """Returns a scripted answer so citation handling can be tested in isolation."""

    model = "stub"

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system, self.last_user = system, user
        return self.response


def test_answer_includes_resolvable_citations(pipeline):
    result = pipeline.answer("How long must audit logs be retained?")
    assert result.grounded
    assert result.citations
    retrieved_ids = {s.chunk.chunk_id for s in result.retrieved}
    assert all(c.chunk_id in retrieved_ids for c in result.citations)


def test_abstains_when_nothing_relevant_is_retrieved(pipeline):
    result = pipeline.answer("What is the company's holiday schedule in Antarctica?")
    assert not result.grounded
    assert prompts.INSUFFICIENT_CONTEXT_ANSWER.lower() in result.answer.lower()
    assert result.citations == []


def test_abstains_when_index_is_empty(empty_retriever):
    pipeline = RAGPipeline(retriever=empty_retriever, llm=StubLLM("Anything [1]"))
    result = pipeline.answer("How long must audit logs be retained?")
    assert not result.grounded
    assert result.metadata["reason"] == "below_grounding_threshold"


def test_hallucinated_citation_markers_are_stripped(retriever):
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=StubLLM("Audit logs are kept for eighteen months [1] and reviewed yearly [99]."),
    )
    result = pipeline.answer("How long must audit logs be retained?")
    assert "[99]" not in result.answer
    assert "[1]" in result.answer
    assert all(c.marker <= len(result.retrieved) for c in result.citations)


def test_uncited_answer_is_marked_ungrounded(retriever):
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=StubLLM("Audit logs are retained for eighteen months."),
    )
    result = pipeline.answer("How long must audit logs be retained?")
    assert result.citations == []
    assert result.grounded is False


def test_model_refusal_is_treated_as_abstention(retriever):
    pipeline = RAGPipeline(
        retriever=retriever, llm=StubLLM(prompts.INSUFFICIENT_CONTEXT_ANSWER)
    )
    result = pipeline.answer("How long must audit logs be retained?")
    assert not result.grounded
    assert result.metadata["reason"] == "model_refused"


def test_duplicate_markers_collapse_to_one_citation(retriever):
    pipeline = RAGPipeline(
        retriever=retriever, llm=StubLLM("First point [1]. Second point [1]. Third [1].")
    )
    result = pipeline.answer("How long must audit logs be retained?")
    assert len(result.citations) == 1
    assert result.citations[0].marker == 1


def test_citations_are_returned_in_marker_order(retriever):
    pipeline = RAGPipeline(retriever=retriever, llm=StubLLM("C [3]. A [1]. B [2]."))
    result = pipeline.answer("retention requirements")
    assert [c.marker for c in result.citations] == sorted(c.marker for c in result.citations)


def test_prompt_contains_numbered_excerpts_and_the_question(retriever):
    llm = StubLLM("Answer [1]")
    RAGPipeline(retriever=retriever, llm=llm).answer("How long are audit logs kept?")
    assert "[1] Source:" in llm.last_user
    assert "How long are audit logs kept?" in llm.last_user
    assert "only the numbered excerpts" in llm.last_system.lower()


def test_metadata_records_provenance(pipeline):
    result = pipeline.answer("How long must audit logs be retained?")
    assert result.metadata["prompt_version"] == prompts.PROMPT_VERSION
    assert result.metadata["latency_ms"] >= 0
    assert result.metadata["retrieved_count"] == len(result.retrieved)


def test_pipeline_is_deterministic(pipeline):
    question = "How long must financial records be retained?"
    first = pipeline.answer(question)
    second = pipeline.answer(question)
    assert first.answer == second.answer
    assert [c.chunk_id for c in first.citations] == [c.chunk_id for c in second.citations]
