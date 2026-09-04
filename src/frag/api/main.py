from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from frag.api.schemas import AskRequest, EvalRequest, QueryRequest
from frag.eval.harness import evaluate
from frag.rag.controller import RagController
from frag.sources.loader import load_live_sources, load_sample_sources
from frag.utils.config import configure_runtime

configure_runtime()

app = FastAPI(title="Actor-Critic Financial RAG", debug=True)


@lru_cache(maxsize=1)
def get_controller() -> RagController:
    """Construct the controller lazily on first use.

    Building it at import time would open a live Qdrant/Ollama connection
    simply to import this module, which breaks test collection and any
    tooling that imports the app without a configured backend. A cached
    accessor defers that cost to the first request instead.
    """
    return RagController()


@lru_cache(maxsize=1)
def get_orchestrator():
    """Build the router+agent orchestrator over the same store; graph tool if present."""
    from frag.agent.loop import AgentLoop
    from frag.agent.orchestrator import Orchestrator
    from frag.agent.tools import (
        ToolRegistry,
        make_calc_tool,
        make_graph_lookup_tool,
        make_retrieve_tool,
    )
    from frag.rag import prompts
    from frag.rag.openrouter_client import OpenRouterClient

    controller = get_controller()
    tools = [make_retrieve_tool(controller.store), make_calc_tool()]
    graph_path = os.getenv("GRAPH_PATH")
    if graph_path and os.path.exists(graph_path):
        from frag.kg.graph import PropertyGraph
        from frag.kg.graph_rag import GraphRAGRetriever

        graph = GraphRAGRetriever(PropertyGraph.load(graph_path))
        tools.insert(1, make_graph_lookup_tool(graph))

    llm = OpenRouterClient("AGENT_MODEL")
    agent = AgentLoop(llm, ToolRegistry(tools), prompts.get("agent").body)
    return Orchestrator(controller, agent, rewrite_llm=llm, route_llm=llm, critic=controller.critic)


@app.on_event("startup")
def startup_event():
    """Startup event to optionally ingest sample documents based on environment variable."""
    if os.getenv("AUTO_INGEST_SAMPLE", "false").lower() == "true":
        docs = load_sample_sources()
        get_controller().ingest(docs)


@app.get("/v1/health")
def health():
    """Health check endpoint to verify that the API is running."""
    try:
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/", response_class=HTMLResponse)
def ui_page():
    """Serve a simple conversational web UI for end users."""
    return HTMLResponse(
        r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Financial RAG Chat</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Inter, system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100dvh; }
    .shell { max-width: 860px; margin: 0 auto; padding: 28px 20px; }
    .card  { background: #111827; border: 1px solid #1e293b; border-radius: 18px; padding: 24px; box-shadow: 0 16px 48px rgba(0,0,0,0.4); }

    /* Header */
    .hdr { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
    .hdr h1 { font-size: 1.2rem; font-weight: 700; color: #f1f5f9; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 6px #22c55e; }
    .sub { color: #64748b; font-size: 0.88rem; margin-bottom: 16px; }

    /* Message window */
    .messages { min-height: 340px; max-height: 500px; overflow-y: auto; padding: 10px 6px;
      border-radius: 12px; background: #020617; border: 1px solid #1e293b; margin-bottom: 14px;
      scroll-behavior: smooth; }
    .messages::-webkit-scrollbar { width: 5px; }
    .messages::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

    /* Bubbles */
    .msg { margin: 8px 0; padding: 11px 14px; border-radius: 12px; line-height: 1.55; font-size: 0.93rem; }
    .user      { background: #1e40af; margin-left: 48px; border-radius: 12px 12px 4px 12px; }
    .assistant { background: #1e293b; margin-right: 48px; border-radius: 12px 12px 12px 4px; }

    /* Answer block */
    .answer-label { font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .answer-text  { color: #f1f5f9; line-height: 1.65; }
    .abstained-warn { color: #fb923c; font-style: italic; font-size: 0.9rem; }

    /* Citations */
    .citations { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; align-items: center; }
    .cit-label { font-size: 0.78rem; color: #64748b; margin-right: 2px; }
    .cit-tag   { background: #0c2340; color: #7dd3fc; border: 1px solid #1d4ed8;
      border-radius: 5px; padding: 1px 7px; font-size: 0.8rem; font-family: monospace; }

    /* Quality toggle */
    .quality-toggle { display: flex; align-items: center; gap: 8px; margin-top: 10px;
      cursor: pointer; user-select: none; padding: 5px 8px; border-radius: 8px 8px 0 0;
      border: 1px solid #1e293b; background: #0f172a; width: fit-content;
      transition: background 0.15s; }
    .quality-toggle:hover { background: #1e293b; }
    .toggle-arrow { font-size: 0.7rem; color: #475569; transition: transform 0.2s; display: inline-block; }
    .toggle-arrow.open { transform: rotate(90deg); }
    .q-label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .score-pill { padding: 1px 9px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
    .score-high { background: #14532d; color: #86efac; }
    .score-med  { background: #78350f; color: #fde68a; }
    .score-low  { background: #7f1d1d; color: #fca5a5; }
    .status-pill { font-size: 0.76rem; padding: 1px 8px; border-radius: 999px; font-weight: 600; }
    .s-accepted  { background: #14532d; color: #86efac; }
    .s-abstained { background: #7c2d12; color: #fed7aa; }

    /* Drawer */
    .quality-drawer { overflow: hidden; max-height: 0; transition: max-height 0.28s ease; }
    .quality-drawer.open { max-height: 220px; }
    .drawer-inner { padding: 10px 10px 8px 10px; border: 1px solid #1e293b;
      border-top: none; border-radius: 0 8px 8px 8px; background: #0a0f1e; }

    /* Sub-score bars */
    .sub-row { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
    .sub-name { font-size: 0.78rem; color: #94a3b8; width: 100px; flex-shrink: 0; }
    .bar-track { flex: 1; height: 5px; background: #1e293b; border-radius: 3px; overflow: hidden; }
    .bar-fill  { height: 100%; border-radius: 3px; transition: width 0.45s cubic-bezier(0.16,1,0.3,1); }
    .bar-fill.high { background: #4ade80; }
    .bar-fill.med  { background: #fbbf24; }
    .bar-fill.low  { background: #f87171; }
    .sub-val { font-size: 0.78rem; color: #94a3b8; width: 34px; text-align: right; flex-shrink: 0; }
    .issues-row { font-size: 0.78rem; color: #64748b; margin-top: 6px;
      padding-top: 7px; border-top: 1px solid #1e293b; }
    .issues-row span { color: #94a3b8; }

    /* Input form */
    form { display: flex; gap: 8px; }
    input[type=text] { flex: 1; padding: 11px 14px; border-radius: 10px;
      border: 1px solid #334155; background: #0f172a; color: #f8fafc;
      font-size: 0.93rem; outline: none; transition: border-color 0.15s; }
    input[type=text]:focus { border-color: #38bdf8; }
    select#ticker { padding: 11px 10px; border-radius: 10px;
      border: 1px solid #334155; background: #0f172a; color: #f8fafc;
      font-size: 0.93rem; outline: none; }
    .send-btn { padding: 11px 20px; border-radius: 10px; border: 0;
      background: #0ea5e9; color: #082f49; font-weight: 700; font-size: 0.93rem;
      cursor: pointer; transition: background 0.15s; white-space: nowrap; }
    .send-btn:hover    { background: #38bdf8; }
    .send-btn:disabled { opacity: 0.5; cursor: wait; }
    .hint { color: #334155; font-size: 0.82rem; margin-top: 10px; }

    /* Thinking indicator */
    .thinking { display: flex; align-items: center; gap: 9px; color: #64748b; }
    .spinner { width: 14px; height: 14px; border-radius: 50%;
      border: 2px solid #334155; border-top-color: #38bdf8;
      animation: spin 0.7s linear infinite; flex-shrink: 0; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .thinking-elapsed { font-variant-numeric: tabular-nums; color: #475569; }
  </style>
</head>
<body>
<div class="shell">
  <div class="card">
    <div class="hdr"><div class="dot"></div><h1>Financial RAG Chat</h1></div>
    <p class="sub">Ask questions about indexed SEC filings and market data.</p>
    <div id="messages" class="messages"></div>
    <form id="chat-form">
      <select id="mode" title="RAG = single-shot retrieval; Agent = multi-hop tool-loop">
        <option value="rag">RAG</option>
        <option value="agent">Agent</option>
      </select>
      <select id="ticker">
        <option value="">All tickers</option>
        <option value="NVDA">NVDA</option>
        <option value="AAPL">AAPL</option>
        <option value="MSFT">MSFT</option>
      </select>
      <input id="query" type="text" placeholder="e.g. What changed in revenue and liquidity?" autocomplete="off" />
      <button class="send-btn" type="submit">Ask</button>
    </form>
    <p class="hint">Try: &ldquo;Summarize the latest filings&rdquo; &bull; &ldquo;What is the 10-year yield trend?&rdquo;</p>
  </div>
</div>
<script>
  const form     = document.getElementById('chat-form');
  const input    = document.getElementById('query');
  const ticker   = document.getElementById('ticker');
  const messages = document.getElementById('messages');
  const btn      = form.querySelector('.send-btn');

  function addMsg(cls, html) {
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    d.innerHTML = html;
    messages.appendChild(d);
    messages.scrollTop = messages.scrollHeight;
    return d;
  }

  function barCls(v)  { return v >= 0.8 ? 'high' : v >= 0.5 ? 'med' : 'low'; }
  function pillCls(v) { return v >= 0.8 ? 'score-high' : v >= 0.5 ? 'score-med' : 'score-low'; }

  function parseNotes(notes) {
    const r = { faithfulness: null, completeness: null, citations: null, issues: 'none' };
    if (!notes) return r;
    const m = notes.match(/faithfulness=([\d.]+).*?completeness=([\d.]+).*?citations=([\d.]+)/);
    if (m) { r.faithfulness = +m[1]; r.completeness = +m[2]; r.citations = +m[3]; }
    const im = notes.match(/issues:\s*(.+)$/);
    if (im) r.issues = im[1].trim();
    return r;
  }

  function subBar(label, value) {
    if (value === null) return '';
    const pct = Math.round(value * 100);
    return `<div class="sub-row">
      <span class="sub-name">${label}</span>
      <div class="bar-track"><div class="bar-fill ${barCls(value)}" style="width:${pct}%"></div></div>
      <span class="sub-val">${pct}%</span>
    </div>`;
  }

  let tid = 0;
  function renderReply(data) {
    const answer    = data.answer || 'No answer returned.';
    const citations = Array.isArray(data.citations) ? data.citations : [];
    const score     = typeof data.critic_score === 'number' ? data.critic_score : null;
    const status    = data.status || 'unknown';
    const notes     = parseNotes(data.critic_notes);
    const accepted  = status === 'accepted';
    const id        = 'qt' + (++tid);

    let html = `<div class="answer-label">Answer</div>`;
    html += accepted
      ? `<div class="answer-text">${answer}</div>`
      : `<div class="abstained-warn">&#x26A0;&#xFE0F; Answer withheld &mdash; quality below threshold (${Math.round(score*100)}%).</div>`;

    if (citations.length) {
      html += `<div class="citations"><span class="cit-label">Sources</span>${citations.map(c=>`<span class="cit-tag">${c}</span>`).join('')}</div>`;
    }

    if (score !== null) {
      const pct   = Math.round(score * 100);
      const sCls  = pillCls(score);
      const stCls = accepted ? 's-accepted' : 's-abstained';
      html += `
        <div class="quality-toggle" onclick="toggleDrawer('${id}')" role="button" tabindex="0"
             onkeydown="if(event.key==='Enter'||event.key===' ')toggleDrawer('${id}')" aria-expanded="false" id="${id}-btn">
          <span class="toggle-arrow" id="${id}-arrow">&#9658;</span>
          <span class="q-label">Quality</span>
          <span class="score-pill ${sCls}">${pct}%</span>
          <span class="status-pill ${stCls}">${status}</span>
        </div>
        <div class="quality-drawer" id="${id}">
          <div class="drawer-inner">
            ${subBar('Faithfulness', notes.faithfulness)}
            ${subBar('Completeness', notes.completeness)}
            ${subBar('Citations',    notes.citations)}
            <div class="issues-row">Issues: <span>${notes.issues}</span></div>
          </div>
        </div>`;
    }
    return html;
  }

  function renderAgentReply(data) {
    const answered = data.status === 'answered';
    const tools = (data.trace || []).filter(s => s.tool).map(s => s.tool);
    let html = `<div class="answer-label">Answer</div>`;
    html += answered
      ? `<div class="answer-text">${data.answer || ''}</div>`
      : `<div class="abstained-warn">&#x26A0;&#xFE0F; ${data.answer || 'No answer.'}</div>`;
    const cost = typeof data.total_cost === 'number' ? '$' + data.total_cost.toFixed(4) : 'n/a';
    html += `<div class="citations"><span class="cit-label">Route</span>` +
            `<span class="cit-tag">${data.route || '?'}</span>` +
            (tools.length ? `<span class="cit-label">Tools</span>` +
              tools.map(t=>`<span class="cit-tag">${t}</span>`).join('') : '') +
            `<span class="cit-label">Cost</span><span class="cit-tag">${cost}</span></div>`;
    return html;
  }

  function toggleDrawer(id) {
    const drawer = document.getElementById(id);
    const arrow  = document.getElementById(id + '-arrow');
    const button = document.getElementById(id + '-btn');
    const open   = drawer.classList.toggle('open');
    arrow.classList.toggle('open', open);
    button.setAttribute('aria-expanded', String(open));
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    addMsg('user', `<strong>You:</strong> ${q}`);
    input.value = '';
    btn.disabled = true;
    input.disabled = true;

    const thinking = addMsg('assistant',
      `<div class="thinking"><span class="spinner"></span>
         <span>Retrieving evidence, drafting, and critiquing&hellip; please be patient</span>
         <span class="thinking-elapsed" id="elapsed">0s</span>
       </div>`);
    const elapsedEl = thinking.querySelector('#elapsed');
    const startedAt = Date.now();
    const timer = setInterval(() => {
      elapsedEl.textContent = Math.round((Date.now() - startedAt) / 1000) + 's';
    }, 1000);

    try {
      const agentMode = document.getElementById('mode').value === 'agent';
      let res;
      if (agentMode) {
        res = await fetch('/v1/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
      } else {
        const filter = ticker.value ? { ticker: ticker.value } : null;
        res = await fetch('/v1/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, top_k: 8, metadata_filter: filter })
        });
      }
      const data = await res.json();
      messages.removeChild(thinking);
      addMsg('assistant', agentMode ? renderAgentReply(data) : renderReply(data));
    } catch (_) {
      messages.removeChild(thinking);
      addMsg('assistant', '<span style="color:#f87171">&#x26A0; Request failed &mdash; please try again.</span>');
    } finally {
      clearInterval(timer);
      btn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  });
</script>
</body>
</html>
        """,
        status_code=200,
    )


@app.post("/v1/ingest-live")
def ingest_live(include_fred: bool = False):
    docs = load_live_sources(
        headers={"User-Agent": os.getenv("SEC_USER_AGENT")}
        if os.getenv("SEC_USER_AGENT")
        else None,
        fred_series_ids=["DGS10"] if include_fred else [],
        fred_api_key=os.getenv("FRED_API_KEY"),
    )
    controller = get_controller()
    indexed = controller.ingest(docs)
    return {
        "status": "ok",
        "documents_loaded": len(docs),
        "documents_indexed": indexed,
        "total_indexed": controller.store.count(),
    }


@app.post("/v1/ingest-sample")
def ingest_sample():
    docs = load_sample_sources()
    controller = get_controller()
    indexed = controller.ingest(docs)
    return {"status": "ok", "documents_indexed": indexed, "total_indexed": controller.store.count()}


@app.post("/v1/query")
def query(req: QueryRequest):
    """Full actor-critic pipeline: retrieve -> actor generates -> critic scores."""
    return get_controller().answer_with_critique(
        req.query, top_k=req.top_k or 8, metadata_filter=req.metadata_filter
    )


@app.post("/v1/ask")
def ask(req: AskRequest):
    """Route the question: single-fact -> RAG; multi-hop -> the agent tool-loop."""
    return get_orchestrator().answer(req.question, history=req.history)


@app.post("/v1/eval")
def eval_run(req: EvalRequest):
    return evaluate(req.predictions, req.golden)
