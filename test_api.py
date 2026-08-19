"""API contract tests.

These pin the response shape. Downstream consumers (and API Gateway mappings)
depend on these field names, so a rename should fail loudly here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.main import app, set_pipeline


@pytest.fixture
def client(pipeline):
    set_pipeline(pipeline)
    with TestClient(app) as test_client:
        yield test_client
    set_pipeline(None)


@pytest.fixture
def empty_client(empty_retriever):
    from rag_assistant.generation.llm import EchoLLM
    from rag_assistant.generation.pipeline import RAGPipeline

    set_pipeline(RAGPipeline(retriever=empty_retriever, llm=EchoLLM()))
    with TestClient(app) as test_client:
        yield test_client
    set_pipeline(None)


def test_health_reports_index_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vectors"] > 0
    assert "prompt_version" in body


def test_ask_returns_the_expected_contract(client):
    response = client.post("/ask", json={"question": "How long must audit logs be retained?"})
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"question", "answer", "grounded", "citations", "retrieved", "metadata"}
    assert body["grounded"] is True
    assert body["citations"], "expected at least one citation"

    citation = body["citations"][0]
    assert set(citation) == {"marker", "chunk_id", "doc_id", "label", "quote", "score"}

    chunk = body["retrieved"][0]
    assert set(chunk) == {"chunk_id", "doc_id", "section", "score", "preview"}


def test_ask_marks_unanswerable_questions_ungrounded(client):
    response = client.post("/ask", json={"question": "What is the CEO's home address?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["citations"] == []


def test_ask_rejects_a_too_short_question(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_ask_rejects_a_missing_question(client):
    assert client.post("/ask", json={}).status_code == 422


def test_ask_rejects_out_of_range_top_k(client):
    response = client.post("/ask", json={"question": "audit log retention", "top_k": 999})
    assert response.status_code == 422


def test_ask_on_an_empty_index_returns_409(empty_client):
    response = empty_client.post("/ask", json={"question": "How long are audit logs kept?"})
    assert response.status_code == 409
    assert "ingest" in response.json()["detail"].lower()


def test_ingest_on_a_missing_directory_returns_404(client):
    response = client.post("/ingest", json={"directory": "/nonexistent/path/xyz"})
    assert response.status_code == 404


def test_ingest_indexes_documents(client, tmp_path):
    (tmp_path / "policy.md").write_text(
        "# Travel Policy\n\n## 2. Limits\n\n"
        "Employees may expense meals up to fifty (50) dollars per day while travelling "
        "on approved company business.",
        encoding="utf-8",
    )
    response = client.post("/ingest", json={"directory": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 1
    assert body["chunks"] >= 1
    assert body["total_vectors"] >= body["indexed"]


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/ask" in schema["paths"]
    assert "/health" in schema["paths"]
