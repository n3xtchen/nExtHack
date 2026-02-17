import os
from rag4pdf.build import load_markdown_documents, split_documents
from langchain_core.documents import Document

def test_load_markdown_documents(tmp_path):
    # Create some dummy markdown files
    d = tmp_path / "docs"
    d.mkdir()
    (d / "test1.md").write_text("Content of test 1", encoding="utf-8")
    (d / "test2.md").write_text("Content of test 2", encoding="utf-8")
    (d / "ignore.txt").write_text("Ignore me", encoding="utf-8")

    pattern = str(d / "*.md")
    docs = load_markdown_documents(pattern)

    assert len(docs) == 2
    contents = [doc.page_content for doc in docs]
    assert "Content of test 1" in contents
    assert "Content of test 2" in contents
    assert docs[0].metadata["source"].endswith(".md")

def test_split_documents():
    docs = [
        Document(page_content="This is a long document. " * 10, metadata={"source": "test.md"})
    ]

    # Each "This is a long document. " is 25 chars. 10 times is 250 chars.
    # Split with chunk_size=100
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)
    assert chunks[0].metadata["source"] == "test.md"
