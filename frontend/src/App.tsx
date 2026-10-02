import { useState } from "react";
import { ask, sendFeedback, AskResponse } from "./api";
import "./styles.css";

const EXAMPLES = [
  "What changed for Mage in patch 1.5?",
  "How does dodge-roll and perfect dodge work?",
  "How do I upgrade weapons to +10?",
  "What is Storm Wyvern weak to?",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState<AskResponse | null>(null);
  const [error, setError] = useState("");
  const [voted, setVoted] = useState<"up" | "down" | null>(null);

  async function run(q: string) {
    const qq = q.trim();
    if (!qq) return;
    setLoading(true); setError(""); setResp(null); setVoted(null);
    try { setResp(await ask(qq)); }
    catch (e: any) { setError(e.message || "Request failed"); }
    finally { setLoading(false); }
  }

  async function vote(v: "up" | "down") {
    if (!resp) return;
    try { await sendFeedback(query, resp.answer, v); setVoted(v); }
    catch { setError("Feedback failed — is the gateway running?"); }
  }

  return (
    <div className="wrap">
      <header>
        <h1>🎮 GameWiki Copilot</h1>
        <p>RAG over patch notes + wiki · FAISS retrieval · citations · Ollama/OpenAI ready</p>
      </header>
      <div className="search">
        <input
          value={query} onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === "Enter" && run(query)}
          placeholder="Ask about builds, patches, bosses, crafting…" />
        <button onClick={() => run(query)} disabled={loading}>{loading ? "…" : "Ask"}</button>
      </div>
      <div className="examples">
        {EXAMPLES.map(ex => <button key={ex} onClick={() => { setQuery(ex); run(ex); }}>{ex}</button>)}
      </div>
      {error && <div className="error">{error}</div>}
      {resp && (
        <div className="result">
          <div className="meta">
            <span>provider: <b>{resp.provider}</b> ({resp.model})</span>
            <span>{resp.latency_ms} ms {resp.cached ? "· cached" : ""}</span>
          </div>
          <pre className="answer">{resp.answer}</pre>
          <div className="feedback">
            <span>Was this helpful?</span>
            <button className={voted === "up" ? "active" : ""} onClick={() => vote("up")}>👍 Yes</button>
            <button className={voted === "down" ? "active" : ""} onClick={() => vote("down")}>👎 No</button>
            {voted && <span className="thanks">Thanks!</span>}
          </div>
          <h3>Sources ({resp.citations.length})</h3>
          {resp.citations.map((c, i) => (
            <details key={i} open={i === 0}>
              <summary>[{c.source} #{c.chunk_id}] score {c.score.toFixed(3)}</summary>
              <p>{c.text}</p>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
