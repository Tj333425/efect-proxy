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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================================================
# CONFIG (Render Environment Variables)
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("EFECT_DEFAULT_MODEL", "gpt-4.1-mini").strip()
TOKEN_SECRET = os.getenv("EFECT_TOKEN_SECRET", "change-me").strip()

# If EFECT_REQUIRE_BEARER=1, then every protected endpoint requires:
# Authorization: Bearer <EFECT_TOKEN_SECRET>
REQUIRE_BEARER = os.getenv("EFECT_REQUIRE_BEARER", "0").strip() == "1"

# Persistent storage path (Render Disk mount path recommended: /var/data)
DATA_DIR = os.getenv("EFECT_DATA_DIR", "/var/data").strip()
DATA_PATH = Path(DATA_DIR)
DATA_PATH.mkdir(parents=True, exist_ok=True)

DB_PATH = os.getenv("EFECT_DB_PATH", str(DATA_PATH / "efect_ai.db")).strip()

UPLOAD_DIR = Path(os.getenv("EFECT_UPLOAD_DIR", str(DATA_PATH / "uploads"))).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("EFECT_MAX_UPLOAD_MB", "100"))  # Any file type
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# For injecting attached text files into chat:
MAX_ATTACH_TEXT_KB = int(os.getenv("EFECT_MAX_ATTACH_TEXT_KB", "200"))  # per file
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
# OPENAI (Responses API)
# =========================================================
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
        "input": messages,
    }
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
# PROMPTING
# =========================================================
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


def should_summarize(session_id: str) -> bool:
    # summarize every 10 user turns
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
    # JSON often comes as application/json
    if content_type in ("application/json",):
        return True
    return False


def read_text_file(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    # try utf-8 fallback latin-1
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
    file_ids: Optional[List[str]] = None  # Attach saved files


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


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root():
    return {"status": "EFECT AI server running"}


@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    # Clean UI, supports optional file attach by file_id (manual for now)
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
    .hint{color:#9a9a9a;font-size:12px;margin-top:8px}
  </style>
</head>
<body>
<header>
  <div class="logo">EFECT</div>
  <div class="pill">UEFN / Verse Debugging</div>
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

  <div class="hint">
    File uploads are via API right now: POST /api/files/upload (form-data key "file").
    Then attach by file_id in chat using the "Attach IDs" box.
  </div>

  <div class="row" style="margin-top:10px">
    <input id="attach" style="flex:1" placeholder="Attach IDs (comma-separated), e.g. ab12cd34ef56, 1122aabbccdd">
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
  const attachEl = document.getElementById('attach');

  let sessionId = localStorage.getItem('efect_session_id') || '';

  function add(role, text){
    const d = document.createElement('div');
    d.className = 'msg ' + (role === 'user' ? 'u' : 'a');
    d.textContent = (role === 'user' ? 'You: ' : 'EFECT AI: ') + text;
    chatEl.appendChild(d);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function parseAttach(){
    const raw = attachEl.value.trim();
    if(!raw) return null;
    return raw.split(',').map(x=>x.trim()).filter(Boolean);
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
        top_k: 3,
        file_ids: parseAttach()
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
    add('assistant', 'New chat started. Paste your Verse compiler error + code.');
  };

  add('assistant', 'Ready. Paste your Verse error and the full function/file. Upload files via /api/files/upload then attach IDs.');
</script>
</body>
</html>
"""


@app.get("/api/sessions")
def list_sessions():
    con = db()
    cur = con.cursor()
    cur.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT 200")
    rows = cur.fetchall()
    con.close()
    return [r["session_id"] for r in rows]


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


@app.post("/api/chat_v2")
def chat_v2(req: ChatV2Request):
    sid = req.session_id or str(uuid.uuid4())
    model = (req.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    now = int(time.time())

    con = db()
    cur = con.cursor()

    # Upsert session
    cur.execute(
        "INSERT INTO sessions(session_id,created_at,updated_at) VALUES(?,?,?) "
        "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at",
        (sid, now, now)
    )

    # Insert user turn
    cur.execute(
        "INSERT INTO turns(session_id,role,content,created_at) VALUES(?,?,?,?)",
        (sid, "user", req.message, now)
    )
    con.commit()

    # Build memory blocks
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

    # Attach files (only inject text-like files; always store any type)
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
    reply = openai_responses(messages, model)

    # Save assistant turn and possibly summary
    con2 = db()
    cur2 = con2.cursor()
    cur2.execute(
        "INSERT INTO turns(session_id,role,content,created_at) VALUES(?,?,?,?)",
        (sid, "assistant", reply, int(time.time()))
    )
    cur2.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (int(time.time()), sid))

    con2.commit()
    con2.close()

    # Summarize every 10 user turns
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

    return {"session_id": sid, "reply": reply}
