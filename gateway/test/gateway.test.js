const test = require("node:test");
const assert = require("node:assert");
const app = require("../server.js");

test("feedback validation rejects bad vote", async () => {
  const server = app.listen(0);
  const port = server.address().port;
  try {
    const r = await fetch(`http://localhost:${port}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "hi", vote: "meh" }),
    });
    assert.equal(r.status, 400);
    const r2 = await fetch(`http://localhost:${port}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(r2.status, 400);
  } finally {
    server.close();
  }
});

test("ask validation requires query (no RAG needed)", async () => {
  const server = app.listen(0);
  const port = server.address().port;
  try {
    const r = await fetch(`http://localhost:${port}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(r.status, 400);
  } finally {
    server.close();
  }
});
