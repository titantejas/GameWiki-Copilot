export interface Citation { source: string; chunk_id: number; score: number; text: string; }
export interface AskResponse {
  answer: string; citations: Citation[]; latency_ms: number;
  provider: string; model: string; cached?: boolean;
}

export async function ask(query: string, top_k = 4): Promise<AskResponse> {
  const r = await fetch("/api/ask", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k })
  });
  if (!r.ok) throw new Error(`Ask failed: ${r.status}`);
  return r.json();
}

export async function sendFeedback(query: string, answer: string, vote: "up" | "down") {
  const r = await fetch("/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, answer, vote })
  });
  if (!r.ok) throw new Error("Feedback failed");
  return r.json();
}
