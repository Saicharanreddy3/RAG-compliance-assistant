from __future__ import annotations

from rag_assistant.ingestion.chunker import chunk_documents
from rag_assistant.retrieval.embeddings import HashingEmbedder
from rag_assistant.retrieval.retriever import contextualize, tokenize
from rag_assistant.retrieval.vector_store import LocalVectorStore


def test_retrieves_the_correct_document(retriever):
    results = retriever.retrieve("How long are audit logs retained?")
    assert results
    assert results[0].chunk.doc_id == "retention_policy"
    assert "eighteen" in results[0].chunk.text.lower()


def test_retrieval_is_deterministic(retriever):
    query = "What is the password length requirement?"
    first = [(s.chunk.chunk_id, round(s.score, 8)) for s in retriever.retrieve(query)]
    second = [(s.chunk.chunk_id, round(s.score, 8)) for s in retriever.retrieve(query)]
    assert first == second


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("   ") == []


def test_respects_top_k(retriever):
    assert len(retriever.retrieve("retention requirements for records", top_k=2)) <= 2


def test_scores_are_sorted_descending(retriever):
    scores = [s.score for s in retriever.retrieve("audit log retention")]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_query_is_filtered_out(retriever):
    results = retriever.retrieve("zzzz quantum llama tapdancing xylophone")
    assert all(s.score >= retriever.config.min_score for s in results)


def test_tokenizer_collapses_plurals():
    assert "password" in tokenize("Passwords must be long")
    assert "record" in tokenize("Records are retained")
    # Words genuinely ending in double-s must not be truncated.
    assert "access" in tokenize("access is revoked")


def test_tokenizer_drops_stopwords():
    tokens = tokenize("what is the retention period")
    assert "the" not in tokens and "is" not in tokens
    assert "retention" in tokens


def test_contextualize_includes_title_and_section(corpus):
    chunk = chunk_documents(corpus)[0]
    text = contextualize(chunk)
    assert chunk.title in text
    assert chunk.text in text


def test_idf_downweights_common_terms(retriever):
    idf = retriever._get_idf()
    assert idf, "expected a populated IDF table"
    # "password" appears in one chunk; "retention" appears across many.
    assert idf.get("password", 0) > idf.get("retention", 0)


def test_local_store_roundtrips_across_instances(tmp_path, corpus):
    chunks = chunk_documents(corpus)
    embedder = HashingEmbedder(dimension=256)
    vectors = embedder.embed_documents([c.text for c in chunks])

    store = LocalVectorStore(tmp_path / "idx")
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)

    # A fresh instance must see the persisted vectors.
    reopened = LocalVectorStore(tmp_path / "idx")
    assert reopened.count() == len(chunks)
    assert reopened.query(vectors[0], top_k=1)[0].chunk.chunk_id == chunks[0].chunk_id


def test_upsert_is_idempotent(tmp_path, corpus):
    chunks = chunk_documents(corpus)
    vectors = HashingEmbedder(dimension=256).embed_documents([c.text for c in chunks])
    store = LocalVectorStore(tmp_path / "idx")
    store.upsert(chunks, vectors)
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)


def test_clear_empties_the_store(tmp_path, corpus):
    chunks = chunk_documents(corpus)
    vectors = HashingEmbedder(dimension=256).embed_documents([c.text for c in chunks])
    store = LocalVectorStore(tmp_path / "idx")
    store.upsert(chunks, vectors)
    store.clear()
    assert store.count() == 0


def test_embeddings_are_normalized():
    vector = HashingEmbedder(dimension=256).embed_query("retention policy")
    magnitude = sum(v * v for v in vector) ** 0.5
    assert abs(magnitude - 1.0) < 1e-6


def test_embedding_of_empty_text_does_not_crash():
    # Regression: signed-hash collisions could cancel to zero and hit log(0).
    assert HashingEmbedder(dimension=64).embed_query("") == [0.0] * 64
