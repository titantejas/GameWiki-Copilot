from pathlib import Path

from rag import Retriever, build_index, load_documents


def test_loader_finds_docs():
    docs = load_documents()
    assert len(docs) >= 5
    assert all(d["source"].endswith(".md") for d in docs)


def test_build_and_search(tmp_path: Path = None):
    info = build_index()
    assert info["docs"] >= 5
    r = Retriever()
    hits = r.search("Mage patch 1.5 buffs", top_k=2)
    assert len(hits) == 2
    assert any("1.5" in h["text"] or "Mage" in h["text"] for h in hits)


def test_generate_fallback():
    from llm import generate

    r = Retriever()
    chunks = r.search("How to upgrade weapons?", top_k=2)
    answer, provider, model = generate("How to upgrade weapons?", chunks)
    assert isinstance(answer, str) and len(answer) > 20
    assert provider in ("extractive", "ollama", "openai", "ollama-fallback", "openai-fallback")
