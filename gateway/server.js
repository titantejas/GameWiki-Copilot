/** Express REST gateway -> proxies to Python RAG, owns feedback + benchmark. */
const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const RAG_URL = process.env.RAG_URL || "http://localhost:8000";
const PORT = process.env.PORT || 3001;
const FEEDBACK_FILE = path.join(__dirname, "feedback.json");

const app = express();
app.use(cors());
app.use(express.json());

// Simple in-memory cache: query -> response (60s TTL)
const cache = new Map();
const TTL = 60_000;

function readFeedback() {
  try {
    return JSON.parse(fs.readFileSync(FEEDBACK_FILE, "utf8"));
  } catch {
    return [];
  }
}

app.get("/api/health", async (req, res) => {
  try {
    const r = await fetch(`${RAG_URL}/health`);
    const body = await r.json();
    res.json({ gateway: "ok", rag: body });
  } catch (e) {
    res.status(502).json({ gateway: "ok", rag: "unreachable", error: String(e) });
  }
});

app.post("/api/ask", async (req, res) => {
  const { query, top_k = 4 } = req.body || {};
  if (!query || typeof query !== "string") return res.status(400).json({ error: "query required" });
  const key = `${query}::${top_k}`;
  const hit = cache.get(key);
  if (hit && Date.now() - hit.t < TTL) return res.json({ ...hit.data, cached: true });
  try {
    const r = await fetch(`${RAG_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k }),
    });
    if (!r.ok) throw new Error(`RAG ${r.status}`);
    const data = await r.json();
    cache.set(key, { t: Date.now(), data });
    res.json(data);
  } catch (e) {
    res.status(502).json({ error: "RAG backend unreachable", detail: String(e) });
  }
});

app.post("/api/feedback", (req, res) => {
  const { query, answer, vote, comment = "" } = req.body || {};
  if (!query || !vote) return res.status(400).json({ error: "query and vote required" });
  if (!["up", "down"].includes(vote)) return res.status(400).json({ error: "vote must be up|down" });
  const all = readFeedback();
  all.push({ ts: new Date().toISOString(), query, answer: (answer || "").slice(0, 2000), vote, comment });
  fs.writeFileSync(FEEDBACK_FILE, JSON.stringify(all, null, 2));
  res.json({ ok: true, total: all.length });
});

app.get("/api/feedback", (req, res) => {
  res.json(readFeedback());
});

app.get("/api/benchmark", async (req, res) => {
  try {
    const r = await fetch(`${RAG_URL}/benchmark`);
    res.json(await r.json());
  } catch (e) {
    res.status(502).json({ error: String(e) });
  }
});

if (require.main === module) {
  app.listen(PORT, () => console.log(`Gateway :${PORT} -> RAG ${RAG_URL}`));
}
module.exports = app;
