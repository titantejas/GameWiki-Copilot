"""Core RAG pipeline: load -> chunk (LangChain) -> embed -> FAISS search."""
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).parent
DATA_DIR = BASE.parent / "data"
INDEX_DIR = BASE / "faiss_index"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_PATH = INDEX_DIR / "index.faiss"
META_PATH = INDEX_DIR / "meta.pkl"

_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)


def load_documents(data_dir: Path = DATA_DIR):
    docs = []
    for f in sorted(data_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for i, chunk in enumerate(_splitter.split_text(text)):
            docs.append({"source": f.name, "chunk_id": i, "text": chunk})
    return docs


class TfidfEmbeddings:
    """Lightweight offline embedding backend (no torch/download).
    Production can swap in HuggingFaceEmbeddings / OpenAIEmbeddings with same interface."""

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: list[str]):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
        mat = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        # L2-normalize for cosine-via-inner-product
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms

    def encode(self, texts: list[str]) -> np.ndarray:
        assert self.vectorizer is not None, "call fit() first or load()"
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms

    def dim(self) -> int:
        assert self.vectorizer is not None
        return len(self.vectorizer.vocabulary_)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)


def try_hf_embeddings():
    """Use sentence-transformers if installed, else None (fallback to TF-IDF)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)
    except Exception:
        return None


def build_index(data_dir: Path = DATA_DIR) -> dict:
    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"No .md files in {data_dir}")
    texts = [d["text"] for d in docs]

    hf = try_hf_embeddings()
    if hf is not None:
        vecs = np.array(hf.embed_documents(texts), dtype=np.float32)
        faiss.normalize_L2(vecs)
        backend = "hf:" + os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        extra = {"hf_model": os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")}
    else:
        emb = TfidfEmbeddings()
        vecs = emb.fit(texts)
        emb.save(INDEX_DIR / "vectorizer.pkl")
        backend = "tfidf"
        extra = {}

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"docs": docs, "backend": backend, **extra}, f)
    return {"docs": len(docs), "dim": vecs.shape[1], "backend": backend}


class Retriever:
    def __init__(self):
        if not FAISS_PATH.exists() or not META_PATH.exists():
            build_index()
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.docs = meta["docs"]
        self.backend = meta.get("backend", "tfidf")
        self._emb = None
        self._hf = None
        if self.backend.startswith("hf:"):
            self._hf = try_hf_embeddings()
        if self._hf is None:
            self._emb = TfidfEmbeddings()
            self._emb.load(INDEX_DIR / "vectorizer.pkl")

    def embed_query(self, query: str) -> np.ndarray:
        if self._hf is not None:
            v = np.array([self._hf.embed_query(query)], dtype=np.float32)
            faiss.normalize_L2(v)
            return v
        return self._emb.encode([query])

    def search(self, query: str, top_k: int = 4):
        q = self.embed_query(query)
        scores, ids = self.index.search(q, min(top_k, len(self.docs)))
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            d = self.docs[int(idx)]
            out.append({**d, "score": float(score)})
        return out
