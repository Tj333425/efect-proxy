import os
import time
import uuid
import json
import hashlib
import sqlite3
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================================================
# CONFIG
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("EFECT_DEFAULT_MODEL", "gpt-4.1-mini").strip()
TOKEN_SECRET = os.getenv("EFECT_TOKEN_SECRET", "change-me").strip()

# If EFECT_REQUIRE_BEARER=1, protected endpoints require:
# Authorization: Bearer <EFECT_TOKEN_SECRET>
REQUIRE_BEARER = os.getenv("EFECT_REQUIRE_BEARER", "0").strip() == "1"

# Storage (Render free tier may be ephemeral)
DATA_DIR = os.getenv("EFECT_DATA_DIR", ".").strip()
DATA_PATH = Path(DATA_DIR)
DATA_PATH.mkdir(parents=True, exist_ok=True)

DB_PATH = os.getenv("EFECT_DB_PATH", str(DATA_PATH / "efect_ai.db")).strip()

UPLOAD_DIR = Path(os.getenv("EFECT_UPLOAD_DIR", str(DATA_PATH / "uploads"))).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("EFECT_MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Inject attached text file caps
MAX_ATTACH_TEXT_KB = int(os.getenv("EFECT_MAX_ATTACH_TEXT_KB", "200"))
MAX_ATTACH_TEXT_BYTES = MAX_ATTACH_TEXT_KB * 1024


# =========================================================
# APP
# =========================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DB HELPERS
# =========================================================
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = db()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """)

    # Migration: title column
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
    except sqlite3.OperationalError:
        pass

    cur.execute("""
    CREATE TABLE IF NOT EXISTS turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        session_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS profile (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS knowledge (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        stored_name TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        content_type TEXT NOT NULL,
        saved_at INTEGER NOT NULL
    )
    """)

    con.commit()
    con.close()


@app.on_event("startup")
def _startup():
    init_db()


# =========================================================
# AUTH
# =========================================================
def require_token(authorization: Optional[str]) -> None:
    if not REQUIRE_BEARER:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != TOKEN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid token")


# =========================================================
# OPENAI (Responses API) - NON-STREAM
# =========================================================
def openai_responses(messages: List[Dict[str, str]], model: str) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY is not set on the server. Add it in your host environment variables."

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": messages}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=90)
    except Exception as e:
        return f"OpenAI request failed: {e}"

    if r.status_code >= 400:
        return f"OpenAI error {r.status_code}: {r.text}"

    data = r.json()
    out_text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out_text += c.get("text", "")
    return out_text.strip() or "(No text returned)"


# =========================================================
# OPENAI (Responses API) - STREAM
# Notes:
# Streaming events include type="response.output_text.delta" with delta text. :contentReference[oaicite:0]{index=0}
# =========================================================
def openai_responses_stream(messages: List[Dict[str, str]], model: str):
    if not OPENAI_API_KEY:
        # Stream an error as plain text so UI still works
        yield "OPENAI_API_KEY is not set on the server.\n"
        return

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": messages, "stream": True}

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
        if r.status_code >= 400:
            yield f"OpenAI error {r.status_code}: {r.text}"
            return

        # SSE comes as lines like:
        # event: response.output_text.delta
        # data: {...json...}
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue

            line = raw.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    evt = json.loads(data_str)
                except Exception:
                    continue

                # Some servers wrap as {"type": "...", ...}
                evt_type = evt.get("type") or evt.get("event")
                if evt_type == "response.output_text.delta":
                    delta = evt.get("delta") or ""
                    if delta:
                        yield delta
                elif evt_type == "response.completed":
                    break
                elif evt_type == "response.error":
                    err = evt.get("error") or evt
                    yield f"\n[Error] {err}\n"


# =========================================================
# PROMPTING (wide intelligence + routing)
# =========================================================
def default_system_prompt() -> str:
    return (
        "You are EFECT AI: a wide-scope assistant for (1) Branding/Design, (2) Apps/Websites/EXEs, "
        "(3) Fortnite Creative/UEFN, and (5) PC Optimization/Troubleshooting, while still handling general questions.\n\n"
        "Routing:\n"
        "- First classify the user request into a mode:\n"
        "  A) Design/Brand (logos, thumbnails, boards, colors, style)\n"
        "  B) Software Dev (websites, APIs, EXE packaging, bugs)\n"
        "  C) UEFN/Verse (device wiring, Verse errors, Creative workflows)\n"
        "  D) PC Optimization (performance, latency, troubleshooting)\n"
        "  E) General\n"
        "- Then answer in that mode using correct conventions and step-by-step actions.\n\n"
        "Working style:\n"
        "- Prefer actionable steps, minimal working examples, and checklists.\n"
        "- If debugging, ask only for the minimum missing info (error lines, file, steps to reproduce).\n"
        "- If the user wants 'same style,' preserve branding and match layout/typography.\n\n"
        "Safety:\n"
        "- Do not assist with cheating, exploits, malware, or bypassing protections. Provide legitimate alternatives.\n\n"
        "Output:\n"
        "- Be direct and specific. Provide clear next steps and defaults when details are missing."
    )


def should_summarize(session_id: str) -> bool:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM turns WHERE session_id=? AND role='user'", (session_id,))
    user_turns = int(cur.fetchone()["c"])
    con.close()
    return user_turns > 0 and (user_turns % 10 == 0)


def summarize_session(session_id: str, model: str) -> str:
    con = db()
    cur = con.cursor()
    cur.execute(
        "SELECT role, content FROM turns WHERE session_id=? ORDER BY id DESC LIMIT 30",
        (session_id,),
    )
    rows = list(reversed(cur.fetchall()))
    con.close()

    turns = [{"role": r["role"], "content": r["content"]} for r in rows]
    prompt = [
        {"role": "system", "content": "Summarize the conversation so far into concise bullet points with key decisions, constraints, and open questions."},
        {"role": "user", "content": json.dumps(turns, ensure_ascii=False)},
    ]
    return openai_responses(prompt, model)


# =========================================================
# KNOWLEDGE SEARCH (lightweight)
# =========================================================
def tokenize(text: str) -> List[str]:
    t = text.lower()
    cleaned = "".join([c if c.isalnum() else " " for c in t])
    return [x for x in cleaned.split() if x]


def rank_knowledge(query: str, top_k: int) -> List[Dict[str, str]]:
    q = query.lower()
    q_tokens = set(tokenize(q))

    con = db()
    cur = con.cursor()
    cur.execute("SELECT id, title, content FROM knowledge")
    docs = cur.fetchall()
    con.close()

    scored: List[Tuple[int, Dict[str, str]]] = []
    for d in docs:
        text = (d["title"] + "\n" + d["content"]).lower()
        tokens = set(tokenize(text))
        overlap = len(q_tokens & tokens)
        bonus = 3 if q in text else 0
        score = overlap + bonus
        if score > 0:
            scored.append((score, {"id": d["id"], "title": d["title"], "content": d["content"]}))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:max(0, top_k)]]


# =========================================================
# FILE HELPERS
# =========================================================
TEXT_EXTS = {
    "txt", "log", "md", "json", "verse", "ini", "cfg", "yaml", "yml", "csv", "py", "js", "ts", "html", "css"
}

def is_text_like(filename: str, content_type: str) -> bool:
    ext = ""
    if "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
    if ext in TEXT_EXTS:
        return True
    if content_type.startswith("text/"):
        return True
    if content_type == "application/json":
        return True
    return False


def read_text_file(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode("latin-1", errors="replace")


# =========================================================
# REQUEST MODELS
# =========================================================
class ChatV2Request(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    use_memory: bool = True
    use_knowledge: bool = True
    top_k: int = 3
    file_ids: Optional[List[str]] = None


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


class SessionRenameRequest(BaseModel):
    session_id: str
    title: str


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root():
    return {"status": "EFECT AI server running"}


# ---------------------------
# ChatGPT-like UI at /chat (STREAMING ENABLED)
# ---------------------------
@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>EFECT AI</title>
  <style>
    :root{
      --bg:#0b0b0b; --panel:#0f0f0f; --panel2:#111;
      --border:#232323; --text:#e9e9e9; --muted:#a0a0a0;
      --accent:#00ff66; --accentText:#001b09;
      --bubbleU:#141414; --bubbleA:#0f1a12;
      --shadow: 0 8px 30px rgba(0,0,0,.45);
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Arial}
    .app{display:flex;height:100vh;overflow:hidden}
    .sidebar{
      width:300px;min-width:260px;max-width:360px;
      border-right:1px solid var(--border);
      background:linear-gradient(180deg,#0c0c0c,#090909);
      padding:14px;display:flex;flex-direction:column;gap:12px;
    }
    .brand{display:flex;align-items:center;justify-content:space-between}
    .logo{font-weight:900;letter-spacing:1px;color:var(--accent)}
    .btn{border:1px solid var(--border);background:var(--panel2);color:var(--text);
      padding:10px 12px;border-radius:12px;cursor:pointer}
    .btn.primary{background:var(--accent);border:none;color:var(--accentText);font-weight:800}
    .btn.small{padding:8px 10px;border-radius:10px;font-size:13px}
    .stack{display:flex;gap:10px;flex-wrap:wrap}
    .sessions{overflow:auto;flex:1;border:1px solid var(--border);border-radius:14px;background:rgba(15,15,15,.7)}
    .sess{padding:10px 12px;border-bottom:1px solid #1b1b1b;cursor:pointer}
    .sess:hover{background:#101010}
    .sess.active{background:#0f1a12;border-left:3px solid var(--accent)}
    .sessTitle{font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .sessMeta{font-size:12px;color:var(--muted);margin-top:4px}
    .main{flex:1;display:flex;flex-direction:column}
    .topbar{
      padding:14px 16px;border-bottom:1px solid var(--border);
      display:flex;align-items:center;justify-content:space-between;gap:12px
    }
    .pill{font-size:12px;color:var(--muted);border:1px solid var(--border);padding:6px 10px;border-radius:999px}
    .content{flex:1;overflow:auto;padding:18px;display:flex;justify-content:center}
    .chatWrap{width:min(980px,100%);display:flex;flex-direction:column;gap:12px}
    .msg{padding:12px 14px;border-radius:16px;line-height:1.35;white-space:pre-wrap;box-shadow:var(--shadow);border:1px solid var(--border)}
    .msg.user{background:var(--bubbleU)}
    .msg.ai{background:var(--bubbleA);border-color:#1d3a2a}
    .msgHead{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:6px}
    .name{font-weight:800;font-size:13px}
    .tools{display:flex;gap:8px}
    .composer{
      border-top:1px solid var(--border);
      padding:14px 16px;display:flex;justify-content:center;background:#0c0c0c
    }
    .composerWrap{width:min(980px,100%);display:flex;flex-direction:column;gap:10px}
    textarea{
      width:100%;min-height:56px;max-height:180px;resize:vertical;
      background:var(--panel2);color:var(--text);border:1px solid var(--border);
      border-radius:16px;padding:12px 12px;font-size:14px;outline:none
    }
    .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    select,input[type="text"]{
      background:var(--panel2);color:var(--text);border:1px solid var(--border);
      border-radius:12px;padding:10px 12px
    }
    input[type="file"]{display:none}
    .right{margin-left:auto}
    .typing{font-size:13px;color:var(--muted);padding:6px 2px}
    @media (max-width: 860px){
      .sidebar{display:none}
    }
  </style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="logo">EFECT AI</div>
      <button class="btn small" id="refreshSessions">↻</button>
    </div>

    <button class="btn primary" id="newChat">New chat</button>

    <div class="stack">
      <label class="pill"><input type="checkbox" id="mem" checked> Memory</label>
      <label class="pill"><input type="checkbox" id="kn" checked> Knowledge</label>
    </div>

    <div class="stack">
      <select id="model">
        <option value="gpt-4.1-mini">gpt-4.1-mini</option>
        <option value="gpt-4.1">gpt-4.1</option>
      </select>
    </div>

    <div class="sessions" id="sessions"></div>

    <div class="stack">
      <input type="text" id="renameInput" placeholder="Rename current chat..." />
      <button class="btn small" id="renameBtn">Rename</button>
    </div>

    <div class="stack">
      <button class="btn small" id="uploadBtn">Upload file</button>
      <input id="filePicker" type="file" />
    </div>

    <div class="pill">
      Upload any file. Text files can be streamed into context.
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div class="pill">Streaming • ChatGPT-style</div>
      <div class="stack right">
        <button class="btn small" id="regenerate">Regenerate</button>
        <button class="btn small" id="clear">Clear</button>
      </div>
    </div>

    <div class="content">
      <div class="chatWrap" id="chat"></div>
    </div>

    <div class="composer">
      <div class="composerWrap">
        <div class="row">
          <span class="pill">Attach file IDs (optional)</span>
          <input type="text" id="attachIds" placeholder="ab12cd34ef56, 1122aabbccdd" style="flex:1"/>
        </div>
        <textarea id="msg" placeholder="Message EFECT AI… (Enter to send, Shift+Enter newline)"></textarea>
        <div class="row">
          <div class="typing" id="typing" style="display:none">EFECT AI is thinking…</div>
          <button class="btn primary right" id="send">Send</button>
        </div>
      </div>
    </div>
  </main>
</div>

<script>
  const chatEl = document.getElementById('chat');
  const sessionsEl = document.getElementById('sessions');
  const msgEl = document.getElementById('msg');
  const modelEl = document.getElementById('model');
  const memEl = document.getElementById('mem');
  const knEl = document.getElementById('kn');
  const typingEl = document.getElementById('typing');
  const attachEl = document.getElementById('attachIds');
  const renameInput = document.getElementById('renameInput');
  const filePicker = document.getElementById('filePicker');

  let sessionId = localStorage.getItem('efect_session_id') || '';
  let lastUserMessage = '';

  function addMsg(role, text){
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + (role==='user' ? 'user' : 'ai');

    const head = document.createElement('div');
    head.className = 'msgHead';
    head.innerHTML = `<div class="name">${role==='user' ? 'You' : 'EFECT AI'}</div>`;

    const tools = document.createElement('div');
    tools.className = 'tools';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'btn small';
    copyBtn.textContent = 'Copy';
    copyBtn.onclick = async ()=>{ try{ await navigator.clipboard.writeText(text); }catch(e){} };
    tools.appendChild(copyBtn);
    head.appendChild(tools);

    const body = document.createElement('div');
    body.textContent = text;

    wrap.appendChild(head);
    wrap.appendChild(body);
    chatEl.appendChild(wrap);
    chatEl.scrollTop = chatEl.scrollHeight;
    return body; // return node so we can stream into it
  }

  function parseAttach(){
    const raw = attachEl.value.trim();
    if(!raw) return null;
    return raw.split(',').map(x=>x.trim()).filter(Boolean);
  }

  async function loadSessions(){
    sessionsEl.innerHTML = '';
    const r = await fetch('/api/sessions');
    const data = await r.json();

    data.forEach(s=>{
      const d = document.createElement('div');
      d.className = 'sess' + (s.session_id === sessionId ? ' active' : '');
      const title = (s.title && String(s.title).trim()) ? s.title : ('Chat ' + s.session_id.slice(0,8));
      const when = s.updated_at ? new Date(s.updated_at*1000).toLocaleString() : '';
      d.innerHTML = `<div class="sessTitle">${title}</div><div class="sessMeta">${when}</div>`;
      d.onclick = ()=>{
        sessionId = s.session_id;
        localStorage.setItem('efect_session_id', sessionId);
        loadSessions();
        addMsg('assistant', 'Switched chat. Continue here.');
      };
      sessionsEl.appendChild(d);
    });
  }

  async function sendStream(message){
    typingEl.style.display = 'block';

    const res = await fetch('/api/chat_stream', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        message,
        session_id: sessionId || null,
        model: modelEl.value,
        use_memory: memEl.checked,
        use_knowledge: knEl.checked,
        top_k: 3,
        file_ids: parseAttach()
      })
    });

    // session id comes back in header for streaming
    const newSid = res.headers.get('x-efect-session-id');
    if (newSid) {
      sessionId = newSid;
      localStorage.setItem('efect_session_id', sessionId);
    }

    const aiBodyNode = addMsg('assistant', ''); // stream into this node
    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let full = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      const chunk = decoder.decode(value, {stream:true});
      full += chunk;
      aiBodyNode.textContent = full;
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    typingEl.style.display = 'none';
    await loadSessions();
    return full;
  }

  async function send(){
    const text = msgEl.value.trim();
    if(!text) return;
    msgEl.value = '';
    lastUserMessage = text;
    addMsg('user', text);
    await sendStream(text);
  }

  async function regenerate(){
    if(!lastUserMessage) return;
    addMsg('assistant', 'Regenerating…');
    await sendStream(lastUserMessage);
  }

  async function renameChat(){
    const title = renameInput.value.trim();
    if(!title || !sessionId) return;
    await fetch('/api/sessions/rename', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ session_id: sessionId, title })
    });
    renameInput.value = '';
    await loadSessions();
  }

  async function uploadPickedFile(file){
    if(!file) return;
    const fd = new FormData();
    fd.append('file', file);

    const r = await fetch('/api/files/upload', { method:'POST', body: fd });
    const data = await r.json();
    if(data && data.file && data.file.id){
      const current = attachEl.value.trim();
      attachEl.value = current ? (current + ', ' + data.file.id) : data.file.id;
      addMsg('assistant', `Uploaded "${data.file.filename}" → file_id: ${data.file.id}. (Added to Attach IDs)`);
    } else {
      addMsg('assistant', 'Upload failed: ' + JSON.stringify(data));
    }
  }

  document.getElementById('send').onclick = send;
  document.getElementById('newChat').onclick = ()=>{
    sessionId = '';
    localStorage.removeItem('efect_session_id');
    chatEl.innerHTML = '';
    addMsg('assistant', 'New chat started.');
    loadSessions();
  };
  document.getElementById('clear').onclick = ()=>{ chatEl.innerHTML=''; };
  document.getElementById('regenerate').onclick = regenerate;
  document.getElementById('renameBtn').onclick = renameChat;
  document.getElementById('refreshSessions').onclick = loadSessions;

  msgEl.addEventListener('keydown', (e)=>{
    if(e.key==='Enter' && !e.shiftKey){
      e.preventDefault();
      send();
    }
  });

  document.getElementById('uploadBtn').onclick = ()=> filePicker.click();
  filePicker.addEventListener('change', async (e)=>{
    const f = e.target.files && e.target.files[0];
    await uploadPickedFile(f);
    filePicker.value = '';
  });

  addMsg('assistant', 'Ready. Streaming is on.');
  loadSessions();
</script>
</body>
</html>
"""


# ---------------------------
# Sessions API (ChatGPT-style list)
# ---------------------------
@app.get("/api/sessions")
def list_sessions():
    con = db()
    cur = con.cursor()
    cur.execute(
        "SELECT session_id, COALESCE(title,'') AS title, updated_at "
        "FROM sessions ORDER BY updated_at DESC LIMIT 200"
    )
    rows = cur.fetchall()
    con.close()
    return [{"session_id": r["session_id"], "title": r["title"], "updated_at": r["updated_at"]} for r in rows]


@app.post("/api/sessions/rename")
def rename_session(req: SessionRenameRequest, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    title = req.title.strip()[:60]
    con = db()
    cur = con.cursor()
    cur.execute("UPDATE sessions SET title=? WHERE session_id=?", (title, req.session_id))
    con.commit()
    con.close()
    return {"ok": True}


# ---------------------------
# Profile / Knowledge
# ---------------------------
@app.post("/api/profile")
def set_profile(req: ProfileSetRequest, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    k = req.key.strip()
    v = req.value.strip()
    con = db()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO profile(key,value,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (k, v, int(time.time()))
    )
    con.commit()
    con.close()
    return {"ok": True, "key": k, "value": v}


@app.post("/api/knowledge/upsert")
def knowledge_upsert(req: KnowledgeUpsertRequest, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    doc_id = req.id or hashlib.sha1((req.title + req.content).encode("utf-8")).hexdigest()[:12]
    con = db()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO knowledge(id,title,content,updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET title=excluded.title, content=excluded.content, updated_at=excluded.updated_at",
        (doc_id, req.title, req.content, int(time.time()))
    )
    con.commit()
    con.close()
    return {"ok": True, "id": doc_id}


@app.post("/api/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest):
    hits = rank_knowledge(req.query, max(1, min(20, req.top_k)))
    return {"hits": hits}


# ---------------------------
# Files (any type)
# ---------------------------
@app.post("/api/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    require_token(authorization)

    data = await file.read()
    size = len(data)
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {MAX_UPLOAD_MB} MB.")

    file_id = uuid.uuid4().hex[:12]
    original = (file.filename or "upload.bin").replace("\\", "_").replace("/", "_")[:240]

    guessed_type = file.content_type or mimetypes.guess_type(original)[0] or "application/octet-stream"
    stored_name = f"{file_id}__{original}"
    path = UPLOAD_DIR / stored_name
    path.write_bytes(data)

    con = db()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO files(id,filename,stored_name,bytes,content_type,saved_at) VALUES(?,?,?,?,?,?)",
        (file_id, original, stored_name, size, guessed_type, int(time.time()))
    )
    con.commit()
    con.close()

    return {"ok": True, "file": {
        "id": file_id,
        "filename": original,
        "stored_name": stored_name,
        "bytes": size,
        "content_type": guessed_type,
        "saved_at": int(time.time())
    }}


@app.get("/api/files")
def list_files(authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    con = db()
    cur = con.cursor()
    cur.execute("SELECT id, filename, bytes, content_type, saved_at FROM files ORDER BY saved_at DESC LIMIT 500")
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/api/files/{file_id}")
def download_file(file_id: str, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    con = db()
    cur = con.cursor()
    cur.execute("SELECT id, filename, stored_name, content_type FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")

    path = UPLOAD_DIR / row["stored_name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    return FileResponse(
        str(path),
        filename=row["filename"],
        media_type=row["content_type"] or "application/octet-stream",
    )


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    con = db()
    cur = con.cursor()
    cur.execute("SELECT stored_name FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise HTTPException(status_code=404, detail="File not found")

    stored_name = row["stored_name"]
    cur.execute("DELETE FROM files WHERE id=?", (file_id,))
    con.commit()
    con.close()

    path = UPLOAD_DIR / stored_name
    if path.exists():
        path.unlink()

    return {"ok": True, "deleted": file_id}


# =========================================================
# CONTEXT BUILDER (shared by non-stream + stream)
# =========================================================
def build_messages_for_request(req: ChatV2Request, sid: str, model: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Returns (messages_to_model, session_id).
    Also ensures the user's turn is stored.
    """
    now = int(time.time())
    con = db()
    cur = con.cursor()

    # Upsert session
    cur.execute(
        "INSERT INTO sessions(session_id,created_at,updated_at,title) VALUES(?,?,?,?) "
        "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at",
        (sid, now, now, None)
    )

    # Auto-title on first user message if empty
    cur.execute("SELECT title FROM sessions WHERE session_id=?", (sid,))
    trow = cur.fetchone()
    if trow and (trow["title"] is None or str(trow["title"]).strip() == ""):
        auto = req.message.strip().split("\n")[0][:40]
        cur.execute("UPDATE sessions SET title=? WHERE session_id=?", (auto, sid))

    # Save user turn
    cur.execute(
        "INSERT INTO turns(session_id,role,content,created_at) VALUES(?,?,?,?)",
        (sid, "user", req.message, now)
    )
    con.commit()

    parts: List[Dict[str, str]] = []

    if req.use_memory:
        cur.execute("SELECT key, value FROM profile ORDER BY updated_at DESC LIMIT 100")
        prof_rows = cur.fetchall()
        if prof_rows:
            prof = "\n".join([f"- {r['key']}: {r['value']}" for r in prof_rows])
            parts.append({"role": "system", "content": "User Profile:\n" + prof})

        cur.execute("SELECT summary FROM summaries WHERE session_id=?", (sid,))
        srow = cur.fetchone()
        if srow:
            parts.append({"role": "system", "content": "Session Summary:\n" + srow["summary"]})

    if req.use_knowledge:
        hits = rank_knowledge(req.message, max(1, min(10, req.top_k)))
        if hits:
            kb = "\n\n".join([f"[{h['id']}] {h['title']}\n{h['content']}" for h in hits])
            parts.append({"role": "system", "content": "Relevant EFECT Knowledge:\n" + kb})

    if req.file_ids:
        attach_blocks = []
        for fid in req.file_ids[:10]:
            cur.execute("SELECT id, filename, stored_name, bytes, content_type FROM files WHERE id=?", (fid,))
            frow = cur.fetchone()
            if not frow:
                attach_blocks.append(f"[{fid}] (missing file id)")
                continue

            filename = frow["filename"]
            ctype = frow["content_type"] or "application/octet-stream"
            path = UPLOAD_DIR / frow["stored_name"]

            if not path.exists():
                attach_blocks.append(f"[{fid}] {filename} (missing on disk)")
                continue

            if is_text_like(filename, ctype):
                text = read_text_file(path, MAX_ATTACH_TEXT_BYTES)
                attach_blocks.append(f"[{fid}] {filename}\n{text}")
            else:
                attach_blocks.append(f"[{fid}] {filename} (binary saved: {ctype}, {frow['bytes']} bytes)")

        if attach_blocks:
            parts.append({"role": "system", "content": "Attached Files:\n\n" + "\n\n---\n\n".join(attach_blocks)})

    # Recent turns
    cur.execute(
        "SELECT role, content FROM turns WHERE session_id=? ORDER BY id DESC LIMIT 12",
        (sid,)
    )
    recent_rows = list(reversed(cur.fetchall()))
    recent = [{"role": r["role"], "content": r["content"]} for r in recent_rows]

    con.close()

    messages = [{"role": "system", "content": default_system_prompt()}] + parts + recent
    return messages, sid


def save_assistant_turn_and_summary(sid: str, reply: str, model: str) -> None:
    con2 = db()
    cur2 = con2.cursor()
    cur2.execute(
        "INSERT INTO turns(session_id,role,content,created_at) VALUES(?,?,?,?)",
        (sid, "assistant", reply, int(time.time()))
    )
    cur2.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (int(time.time()), sid))
    con2.commit()
    con2.close()

    if should_summarize(sid):
        summ = summarize_session(sid, model)
        con3 = db()
        cur3 = con3.cursor()
        cur3.execute(
            "INSERT INTO summaries(session_id,summary,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
            (sid, summ, int(time.time()))
        )
        con3.commit()
        con3.close()


# ---------------------------
# Non-stream chat (kept for compatibility)
# ---------------------------
@app.post("/api/chat_v2")
def chat_v2(req: ChatV2Request):
    sid = req.session_id or str(uuid.uuid4())
    model = (req.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    messages, sid = build_messages_for_request(req, sid, model)
    reply = openai_responses(messages, model)
    save_assistant_turn_and_summary(sid, reply, model)
    return {"session_id": sid, "reply": reply}


# ---------------------------
# STREAMING chat endpoint
# Returns a streamed text/plain body.
# Session id is returned in header: x-efect-session-id
# ---------------------------
@app.post("/api/chat_stream")
def chat_stream(req: ChatV2Request):
    sid = req.session_id or str(uuid.uuid4())
    model = (req.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    messages, sid = build_messages_for_request(req, sid, model)

    def gen():
        full = ""
        try:
            for chunk in openai_responses_stream(messages, model):
                full += chunk
                yield chunk
        finally:
            # Persist the full assistant message after streaming ends
            if full.strip():
                save_assistant_turn_and_summary(sid, full, model)

    headers = {"x-efect-session-id": sid}
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8", headers=headers)
