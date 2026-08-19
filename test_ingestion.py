from __future__ import annotations

import pytest

from rag_assistant.config import ChunkingConfig
from rag_assistant.ingestion.chunker import chunk_document
from rag_assistant.ingestion.loader import load_directory, load_document
from rag_assistant.models import Document

CONFIG = ChunkingConfig(chunk_size=300, chunk_overlap=50, min_chunk_chars=20)


def make_doc(text: str) -> Document:
    return Document(doc_id="test_doc", title="Test Doc", text=text)


def test_chunker_attaches_markdown_section_headings():
    doc = make_doc(
        "# Policy\n\n## 3.1 Financial Records\n\n"
        + "Financial records must be retained for seven years. " * 3
        + "\n\n## 3.2 Audit Logs\n\n"
        + "Audit logs must be retained for eighteen months. " * 3
    )
    chunks = chunk_document(doc, CONFIG)
    sections = {c.section for c in chunks}
    assert "3.1 Financial Records" in sections
    assert "3.2 Audit Logs" in sections


def test_chunker_detects_numbered_headings_without_markdown():
    doc = make_doc(
        "4.2 Data Retention\n\n" + "Records are kept for the required period. " * 4
    )
    chunks = chunk_document(doc, CONFIG)
    assert chunks[0].section == "4.2 Data Retention"


def test_chunker_is_deterministic():
    doc = make_doc("## A\n\n" + "Some policy text here. " * 40)
    first = chunk_document(doc, CONFIG)
    second = chunk_document(doc, CONFIG)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunker_respects_max_size_with_tolerance_for_overlap():
    doc = make_doc("## A\n\n" + "word " * 500)
    chunks = chunk_document(doc, CONFIG)
    assert chunks, "expected at least one chunk"
    # Overlap is prepended, so allow the configured overlap as slack.
    assert all(len(c.text) <= CONFIG.chunk_size + CONFIG.chunk_overlap + 50 for c in chunks)


def test_chunker_drops_fragments_below_minimum_length():
    doc = make_doc("## A\n\nHi.\n\n## B\n\n" + "Real content that is long enough to keep. " * 3)
    chunks = chunk_document(doc, CONFIG)
    assert all(len(c.text) >= CONFIG.min_chunk_chars for c in chunks)


def test_chunk_ids_are_unique_and_ordered():
    doc = make_doc("## A\n\n" + "Policy text. " * 100)
    chunks = chunk_document(doc, CONFIG)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.ordinal for c in chunks] == sorted(c.ordinal for c in chunks)


def test_citation_label_includes_section():
    doc = make_doc("## 3.1 Financial Records\n\n" + "Retention is seven years. " * 4)
    chunk = chunk_document(doc, CONFIG)[0]
    assert "Test Doc" in chunk.citation_label()
    assert "3.1 Financial Records" in chunk.citation_label()


def test_loader_uses_h1_as_title(tmp_path):
    path = tmp_path / "some_file.md"
    path.write_text("# Real Title\n\nBody text that is long enough to matter.", encoding="utf-8")
    doc = load_document(path)
    assert doc.title == "Real Title"
    assert doc.doc_id == "some_file"


def test_loader_rejects_empty_documents(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(path)


def test_loader_skips_unsupported_files(tmp_path):
    (tmp_path / "good.md").write_text("# Good\n\nEnough content here to load.", encoding="utf-8")
    (tmp_path / "ignored.xyz").write_text("nope", encoding="utf-8")
    docs = load_directory(tmp_path)
    assert [d.doc_id for d in docs] == ["good"]


def test_loader_is_deterministically_ordered(tmp_path):
    for name in ["c", "a", "b"]:
        (tmp_path / f"{name}.md").write_text(f"# {name}\n\nSome body text here.", encoding="utf-8")
    assert [d.doc_id for d in load_directory(tmp_path)] == ["a", "b", "c"]
