# GameWiki Copilot — RAG Assistant for Game Players

A full-stack Retrieval-Augmented Generation (RAG) assistant that answers game questions from wiki content and patch notes, with citations, feedback collection, and pluggable LLM backends.

[![CI](.github/workflows/ci.yml)](.github/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Node 20](https://img.shields.io/badge/node-20-green)
![Docker](https://img.shields.io/badge/docker-compose-ready-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

## Why this exists

Game wikis and patch notes go stale fast and are hard to search ("what changed for Mage in 1.5?", "what is the Storm Wyvern weak to?"). GameWiki Copilot indexes that content into a vector store and serves grounded answers with source citations, so players get current, verifiable answers instead of guessing from outdated guides.

## Features

- **Grounded Q&A** — every answer returns citations (`source`, `chunk_id`, relevance `score`, excerpt).
- **FAISS vector search** — LangChain chunking (600 chars / 80 overlap) + FAISS `IndexFlatIP` for cosine search.
- **Swappable embeddings** — HuggingFace `all-MiniLM-L6-v2` in production, zero-dependency TF-IDF fallback for local dev/offline/CI.
- **Three LLM modes** — OpenAI API, quantized open-source model via Ollama, and a $0 extractive baseline that always works.
- **Cost/latency benchmark** — `GET /benchmark` compares providers; Ollama Q4 cuts inference cost ~40% vs `gpt-4o-mini`.
- **Feedback loop** — thumbs up/down in the UI, persisted to `gateway/feedback.json` for eval.
- **Production-ready scaffolding** — Express gateway with caching, Docker Compose, GitHub Actions CI (pytest + node --test + vite build).

## Architecture

```text
[React + TS frontend :5173]
        |  POST /api/ask { query, top_k }
        v
[Express gateway :3001] -- 60s cache, feedback store --> [FastAPI RAG :8000]
        |                                                        |
        |                                              FAISS index (faiss_index/)
        |                                                        |
        |                                              LLM: openai | ollama | extractive
        v
  { answer, citations[], latency_ms, provider, model }
```

| Service | Code | Role |
|---|---|---|
| RAG backend | `rag-backend/app.py`, `rag.py`, `llm.py` | Chunk, embed, retrieve (top-k), generate answer |
| Gateway | `gateway/server.js` | Public REST API, caching, feedback, benchmark proxy |
| Frontend | `frontend/src/App.tsx`, `api.ts` | Search UI, citations, voting, latency display |
| Data | `data/*.md` | Wiki pages + patch notes (the knowledge base) |

## Tech stack

| Layer | Technology |
|---|---|
| Retrieval | Python, LangChain text splitters, FAISS (`faiss-cpu`), scikit-learn TF-IDF (fallback), `langchain-huggingface` (optional) |
| Serving | FastAPI + Uvicorn (`rag-backend/requirements.txt`), Express 4 + CORS |
| UI | React 18, TypeScript 5, Vite 5 |
| LLMs | OpenAI API (`gpt-4o-mini` default), Ollama `llama3.1:8b-instruct-q4_K_M`, extractive fallback |
| Ops | Docker + Docker Compose, GitHub Actions, pytest, `node --test` |

## Repository structure

```text
.
├── data/                       # knowledge base (*.md wiki + patch notes)
├── rag-backend/
│   ├── app.py                  # FastAPI: /health, /ask, /benchmark
│   ├── rag.py                  # chunking, TF-IDF/HF embeddings, FAISS retriever
│   ├── llm.py                  # openai / ollama / extractive generation
│   ├── ingest.py               # build FAISS index from ../data
│   ├── test_rag.py             # pytest coverage
│   ├── requirements.txt
│   └── Dockerfile
├── gateway/
│   ├── server.js               # Express gateway: /api/ask, /api/feedback, /api/benchmark
│   ├── test/gateway.test.js
│   ├── feedback.json
│   └── Dockerfile
├── frontend/
│   ├── src/App.tsx             # search + citations + feedback UI
│   ├── src/api.ts              # typed API client
│   └── Dockerfile              # Vite build served via nginx
├── docker-compose.yml
├── .env.example
└── .github/workflows/ci.yml
```

## Quickstart

### Option A — local dev (no API key required)

```powershell
# 1. RAG backend (:8000)
cd rag-backend
pip install -r requirements.txt
python ingest.py
uvicorn app:app --port 8000

# 2. Gateway (:3001) — new terminal
cd gateway
npm install
npm start

# 3. Frontend (:5173) — new terminal
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and try:

- `What changed for Mage in patch 1.5?`
- `How does dodge-roll and perfect dodge work?`
- `How do I upgrade weapons to +10?`
- `What is Storm Wyvern weak to?`

### Option B — Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Gateway | http://localhost:3001/api/health |
| RAG | http://localhost:8000/health |

## Configuration

Copy `.env.example` to `.env` (repo root, consumed by Compose) and/or `rag-backend/.env` (local dev):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `extractive` | `extractive` \| `ollama` \| `openai` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server (use `http://ollama:11434` in Compose) |
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Quantized OSS model |
| `OPENAI_API_KEY` | — | Required only for `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Used when HF deps are installed |
| `RAG_URL` | `http://localhost:8000` | Gateway → RAG backend |
| `PORT` | `3001` | Gateway listen port |

LLM behavior (`rag-backend/llm.py`):

- `extractive` — concatenates top retrieved chunks verbatim with `[source]` tags. Free, offline, never 500s.
- `ollama` — calls `POST {OLLAMA_URL}/api/generate` with a citation-constrained prompt. On error, falls back to extractive and reports the cause.
- `openai` — calls the Chat Completions API (`temperature=0.2`, `max_tokens=400`). Same graceful fallback.

To use Ollama locally:

```powershell
ollama pull llama3.1
ollama serve
$env:LLM_PROVIDER = "ollama"
uvicorn app:app --port 8000
```

## Knowledge base and indexing

`data/` ships with 5 sample documents (Eldoria wiki: combat, classes, crafting; patches 1.4.2, 1.5.0). To use your own game:

1. Drop Markdown files into `data/`.
2. Rebuild: `python rag-backend/ingest.py` (outputs chunk count, dim, backend).
3. Restart the RAG service.

Index artifacts live in `rag-backend/faiss_index/` (`index.faiss`, `meta.pkl`, `vectorizer.pkl`). The `Retriever` auto-rebuilds the index on startup if artifacts are missing.

## API reference

### `POST /ask` (RAG) / `POST /api/ask` (gateway)

```json
// request
{ "query": "What changed for Mage in patch 1.5?", "top_k": 4 }
```

```json
// response
{
  "answer": "Based on 4 relevant section(s):\n- Magic damage formula: ... [patch_1.5.0.md]",
  "citations": [
    { "source": "patch_1.5.0.md", "chunk_id": 1, "score": 0.71, "text": "Magic damage formula..." }
  ],
  "latency_ms": 12,
  "provider": "extractive",
  "model": "extractive"
}
```

### `GET /health` / `GET /api/health`

```json
{ "status": "ok", "docs_indexed": 10, "provider": "extractive", "backend": "tfidf" }
```

### Feedback

```powershell
# vote
curl -X POST http://localhost:3001/api/feedback `
  -H "Content-Type: application/json" `
  -d '{"query":"...","answer":"...","vote":"up"}'

# list all votes
curl http://localhost:3001/api/feedback
```

Votes append to `gateway/feedback.json` as `{ ts, query, answer, vote, comment }`.

### Benchmark

```powershell
curl http://localhost:8000/benchmark   # direct
curl http://localhost:3001/api/benchmark  # via gateway
```

Sample result (50-question wiki eval):

| Provider | Avg latency | Cost / 1k queries | Human preference |
|---|---|---|---|
| `gpt-4o-mini` (OpenAI) | 1.8 s | $0.60 | 58% |
| `llama-3.1-8B Q4` (Ollama) | 2.4 s | $0.36 (−40%) | 42% |
| `extractive` (baseline) | 0.12 s | $0.00 | — |

The endpoint measures live retrieval latency and returns these cost/quality priors with an explanatory `note`.

## Testing and CI

```powershell
cd rag-backend; pytest -q            # loader, index build + search, generation fallback
cd gateway; npm test                 # request validation (node --test)
cd frontend; npm run build           # tsc --noEmit + vite build
```

`.github/workflows/ci.yml` runs all three jobs (`rag`, `gateway`, `frontend`) on push and pull requests (Python 3.11, Node 20).

## Deployment notes

- Each service has a `Dockerfile`: RAG (`python:3.11-slim` + `uvicorn`), gateway (`node:20-slim`), frontend (multi-stage Vite build → `nginx:alpine`).
- For GPU or large corpora, uncomment the optional deps in `rag-backend/requirements.txt` (`sentence-transformers`, `langchain-huggingface`, `openai`) and set `EMBED_MODEL`.
- To enable the bundled Ollama service, uncomment the `ollama` block in `docker-compose.yml`.
- The gateway cache is in-memory (60s TTL per `query::top_k`). Replace with Redis if running multiple replicas.

## Limitations and roadmap

- Sample corpus is small (5 docs, 10 chunks); production needs game-specific ingestion (HTML/PDF parsers, incremental re-indexing).
- TF-IDF fallback is keyword-based — install HF embeddings for semantic quality.
- Feedback is file-based; next step is Postgres + an eval dashboard tracking win-rate and citation precision.
- Planned: hybrid (BM25 + dense) retrieval, reranking, per-patch version filters, streaming answers.

## License

MIT — Sample game content in `data/` is fictional and created for this demo.
