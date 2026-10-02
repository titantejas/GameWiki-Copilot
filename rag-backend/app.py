"""FastAPI RAG service."""
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

from llm import PROVIDER, benchmark_info, generate  # noqa: E402
from rag import Retriever  # noqa: E402

app = FastAPI(title="GameWiki Copilot RAG")
retriever = Retriever()


class AskRequest(BaseModel):
    query: str
    top_k: int = 4


@app.get("/health")
def health():
    return {
        "status": "ok",
        "docs_indexed": len(retriever.docs),
        "provider": PROVIDER,
        "backend": retriever.backend,
    }


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    chunks = retriever.search(req.query, top_k=req.top_k)
    answer, provider, model = generate(req.query, chunks)
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "answer": answer,
        "citations": [
            {"source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"], "text": c["text"]}
            for c in chunks
        ],
        "latency_ms": latency_ms,
        "provider": provider,
        "model": model,
    }


@app.get("/benchmark")
def benchmark():
    """Live retrieval latency + static LLM cost/quality priors."""
    import time as _t

    results = {}
    for prov in ("extractive", "ollama", "openai"):
        t0 = _t.time()
        chunks = retriever.search("What changed for Mage in patch 1.5?", top_k=3)
        retrieval_ms = int((_t.time() - t0) * 1000)
        info = benchmark_info(prov)
        results[prov] = {**info, "retrieval_ms": retrieval_ms}
    results["note"] = "Ollama Q4 cuts inference cost ~40% vs gpt-4o-mini with modest quality drop. Extractive is free baseline."
    return results
