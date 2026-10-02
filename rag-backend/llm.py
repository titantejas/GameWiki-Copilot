"""LLM layer: OpenAI API + Ollama (quantized OSS) + extractive fallback."""
import os
import time

import httpx

PROVIDER = os.getenv("LLM_PROVIDER", "extractive").lower()  # extractive | ollama | openai
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM = (
    "You are GameWiki Copilot, a helpful game assistant. "
    "Answer ONLY from the provided context. Cite sources like [patch_1.5.0.md]. "
    "If the answer is not in context, say you don't know."
)


def build_prompt(query: str, chunks: list[dict]) -> str:
    ctx = "\n\n".join(f"[{c['source']} #{c['chunk_id']}]\n{c['text']}" for c in chunks)
    return f"{SYSTEM}\n\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer with citations:"


def extractive_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't know — no relevant wiki content found."
    lines = [f"Based on {len(chunks)} relevant section(s):", ""]
    for c in chunks[:3]:
        snippet = c["text"].strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        lines.append(f"- {snippet} [{c['source']}]")
    lines.append("")
    lines.append("Tip: set LLM_PROVIDER=ollama or openai for a generated answer.")
    return "\n".join(lines)


def ollama_answer(prompt: str) -> str:
    r = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def openai_answer(prompt: str) -> str:
    from openai import OpenAI  # optional dep; only needed when provider=openai

    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def generate(query: str, chunks: list[dict]) -> tuple[str, str, str]:
    """Returns (answer, provider, model). Falls back to extractive on any LLM error."""
    provider = PROVIDER
    prompt = build_prompt(query, chunks)
    try:
        if provider == "ollama":
            return ollama_answer(prompt), "ollama", OLLAMA_MODEL
        if provider == "openai":
            return openai_answer(prompt), "openai", OPENAI_MODEL
    except Exception as e:  # fall through to extractive so MVP never 500s
        return f"{extractive_answer(query, chunks)}\n\n(LLM {provider} failed: {e})", f"{provider}-fallback", "extractive"
    return extractive_answer(query, chunks), "extractive", "extractive"


def benchmark_info(provider: str) -> dict:
    """Static cost/latency priors (measured on 50 QA eval, see README). Runtime latency measured live."""
    table = {
        "extractive": {"avg_latency_s": 0.12, "cost_per_1k": 0.0, "win_rate": None},
        "ollama": {"avg_latency_s": 2.4, "cost_per_1k": 0.36, "win_rate": 0.42},
        "openai": {"avg_latency_s": 1.8, "cost_per_1k": 0.60, "win_rate": 0.58},
    }
    return table.get(provider, table["extractive"])
