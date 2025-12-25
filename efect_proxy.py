import os
import time
import uuid
import json
import hashlib
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("EFECT_DEFAULT_MODEL", "gpt-4.1-mini").strip()
TOKEN_SECRET = os.getenv("EFECT_TOKEN_SECRET", "change-me").strip()

# -----------------------------
# App
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory stores (Render free tier resets on redeploy)
# -----------------------------
SESSIONS: Dict[str, List[Dict[str, str]]] = {}          # session_id -> [{"role","content"}, ...]
SESSION_SUMMARY: Dict[str, str] = {}                    # session_id -> summary string
PROFILE: Dict[str, str] = {}                            # key -> value
KNOWLEDGE: List[Dict[str, str]] = []                    # [{"id","title","content"}]

# -----------------------------
# Models
# -----------------------------
class ChatV2Request(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    use_memory: bool = True
    use_knowledge: bool = True
    top_k: int = 3

class KnowledgeUpsertRequest(BaseModel):
    title: str
    content: str
    id: Optional[str] = None

class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ProfileSetRequest(BaseModel):
    key: str
    value: str

# -----------------------------
# Helpers
# -----------------------------
def require_token(authorization: Optional[str]):
    # Optional auth gate (keep permissive by default)
    # If you want to require bearer, set EFECT_REQUIRE_BEARER=1
    if os.getenv("EFECT_REQUIRE_BEARER", "0") != "1":
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    # simple check: token must equal TOKEN_SECRET
    if token != TOKEN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid token")

def simple_rank_knowledge(query: str, top_k: int) -> List[Dict[str, str]]:
    # lightweight ranking: token overlap + substring bonus
    q = query.lower()
    q_tokens = set([t for t in "".join([c if c.isalnum() else " " for c in q]).split() if t])

    scored = []
    for doc in KNOWLEDGE:
        text = (doc["title"] + "\n" + doc["content"]).lower()
        tokens = set([t for t in "".join([c if c.isalnum() else " " for c in text]).split() if t])
        overlap = len(q_tokens & tokens)
        bonus = 3 if q in text else 0
        scored.append((overlap + bonus, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:max(0, top_k)] if s > 0]

def openai_responses(messages: List[Dict[str, str]], model: str) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY is not set on the server. Add it in Render → Environment."

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": messages,  # Responses API accepts role/content list as input
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        return f"OpenAI error {r.status_code}: {r.text}"

    data = r.json()
    # Extract text output safely
    out_text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out_text += c.get("text", "")
    return out_text.strip() or "(No text returned)"

def should_summarize(session_id: str) -> bool:
    # summarize every 10 user turns
    turns = SESSIONS.get(session_id, [])
    user_turns = sum(1 for t in turns if t["role"] == "user")
    return user_turns > 0 and user_turns % 10 == 0

def summarize_session(session_id: str, model: str) -> str:
    turns = SESSIONS.get(session_id, [])[-30:]  # cap
    prompt = [
        {"role": "system", "content": "Summarize the conversation so far into concise bullet points with key decisions, constraints, and open questions."},
        {"role": "user", "content": json.dumps(turns, ensure_ascii=False)},
    ]
    return openai_responses(prompt, model)

def default_system_prompt() -> str:
    return (
        "You are EFECT AI, specialized in UEFN / Verse debugging and building.\n"
        "Priorities:\n"
        "1) Correctness over speed\n"
        "2) Use Verse failure contexts correctly (decides/suspends)\n"
        "3) Provide minimal, compiling code fixes and clear UEFN device wiring steps\n"
        "4) If uncertain, ask for the exact error line and the full function/file\n"
        "Constraints:\n"
        "- Do not assist with cheating, exploits, or bypassing anti-cheat.\n"
        "- Focus on legitimate UEFN development.\n"
    )

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "EFECT AI server running"}

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    # Clean UI (no proxy fields)
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>EFECT AI</title>
  <style>
    body{background:#0b0b0b;color:#e8e8e8;font-family:system-ui,Segoe UI,Arial;margin:0}
    header{padding:16px 20px;border-bottom:1px solid #1e1e1e;display:flex;align-items:center;gap:12px}
    .logo{color:#00ff66;font-weight:800;letter-spacing:1px}
    .wrap{max-width:980px;margin:0 auto;padding:18px}
    .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
    select,input,button,textarea{background:#111;color:#e8e8e8;border:1px solid #2a2a2a;border-radius:10px;padding:10px 12px}
    button{cursor:pointer}
    button.primary{background:#00ff66;color:#001b09;border:none;font-weight:700}
    .chat{margin-top:14px;border:1px solid #222;border-radius:14px;min-height:420px;padding:14px;background:#0f0f0f}
    .msg{padding:10px 12px;border-radius:12px;margin:10px 0;white-space:pre-wrap;line-height:1.35}
    .u{background:#141414;border:1px solid #262626}
    .a{background:#0f1a12;border:1px solid #1d3a2a}
    .footer{display:flex;gap:10px;margin-top:12px}
    textarea{flex:1;min-height:54px;max-height:160px;resize:vertical}
    .pill{font-size:12px;border:1px solid #2a2a2a;border-radius:999px;padding:6px 10px;color:#bdbdbd}
  </style>
</head>
<body>
<header>
  <div class="logo">EFECT</div>
  <div class="pill">UEFN / Verse Debugging Mode</div>
</header>

<div class="wrap">
  <div class="row">
    <label class="pill">Model</label>
    <select id="model">
      <option value="gpt-4.1-mini">gpt-4.1-mini</option>
      <option value="gpt-4.1">gpt-4.1</option>
    </select>
    <label class="pill"><input type="checkbox" id="mem" checked> Memory</label>
    <label class="pill"><input type="checkbox" id="kn" checked> Knowledge</label>
    <button id="new" class="primary">New Chat</button>
  </div>

  <div id="chat" class="chat"></div>

  <div class="footer">
    <textarea id="msg" placeholder="Paste Verse errors + code here..."></textarea>
    <button id="send" class="primary">Send</button>
  </div>
</div>

<script>
  const chatEl = document.getElementById('chat');
  const msgEl = document.getElementById('msg');
  const modelEl = document.getElementById('model');
  const memEl = document.getElementById('mem');
  const knEl = document.getElementById('kn');
  let sessionId = localStorage.getItem('efect_session_id') || '';

  function add(role, text){
    const d = document.createElement('div');
    d.className = 'msg ' + (role === 'user' ? 'u' : 'a');
    d.textContent = (role === 'user' ? 'You: ' : 'EFECT AI: ') + text;
    chatEl.appendChild(d);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  async function send(){
    const text = msgEl.value.trim();
    if(!text) return;
    msgEl.value = '';
    add('user', text);

    const res = await fetch('/api/chat_v2', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        message: text,
        session_id: sessionId || null,
        model: modelEl.value,
        use_memory: memEl.checked,
        use_knowledge: knEl.checked,
        top_k: 3
      })
    });
    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem('efect_session_id', sessionId);
    add('assistant', data.reply || JSON.stringify(data));
  }

  document.getElementById('send').onclick = send;
  msgEl.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
  document.getElementById('new').onclick = ()=>{
    sessionId = '';
    localStorage.removeItem('efect_session_id');
    chatEl.innerHTML = '';
    add('assistant', 'New chat started. Paste your Verse error + code.');
  };

  // boot message
  add('assistant', 'Ready. Paste your Verse compiler error and the full function/file.');
</script>
</body>
</html>
"""

@app.get("/api/sessions")
def list_sessions():
    return list(SESSIONS.keys())

@app.post("/api/profile")
def set_profile(req: ProfileSetRequest, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    PROFILE[req.key.strip()] = req.value.strip()
    return {"ok": True, "profile": PROFILE}

@app.post("/api/knowledge/upsert")
def knowledge_upsert(req: KnowledgeUpsertRequest, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    doc_id = req.id or hashlib.sha1((req.title + req.content).encode("utf-8")).hexdigest()[:12]
    # update if exists
    for d in KNOWLEDGE:
        if d["id"] == doc_id:
            d["title"] = req.title
            d["content"] = req.content
            return {"ok": True, "id": doc_id, "updated": True}
    KNOWLEDGE.append({"id": doc_id, "title": req.title, "content": req.content})
    return {"ok": True, "id": doc_id, "updated": False}

@app.post("/api/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest):
    hits = simple_rank_knowledge(req.query, req.top_k)
    return {"hits": hits}

@app.post("/api/chat_v2")
def chat_v2(req: ChatV2Request):
    sid = req.session_id or str(uuid.uuid4())
    model = (req.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    turns = SESSIONS.setdefault(sid, [])
    turns.append({"role": "user", "content": req.message})

    # Build context
    sys = default_system_prompt()
    parts = []

    if req.use_memory:
        prof = "\n".join([f"- {k}: {v}" for k, v in PROFILE.items()]) if PROFILE else ""
        summ = SESSION_SUMMARY.get(sid, "")
        if prof:
            parts.append({"role": "system", "content": "User Profile:\n" + prof})
        if summ:
            parts.append({"role": "system", "content": "Session Summary:\n" + summ})

    if req.use_knowledge:
        hits = simple_rank_knowledge(req.message, max(1, min(10, req.top_k)))
        if hits:
            kb = "\n\n".join([f"[{h['id']}] {h['title']}\n{h['content']}" for h in hits])
            parts.append({"role": "system", "content": "Relevant EFECT Knowledge:\n" + kb})

    # Recent context window
    recent = turns[-12:]

    messages = [{"role": "system", "content": sys}] + parts + recent
    reply = openai_responses(messages, model)

    turns.append({"role": "assistant", "content": reply})

    if should_summarize(sid):
        SESSION_SUMMARY[sid] = summarize_session(sid, model)

    return {"session_id": sid, "reply": reply}
