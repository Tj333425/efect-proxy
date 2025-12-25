import os
import time
import uuid
import json
import hashlib
import hmac
import base64
import sqlite3
import mimetypes
import io
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================================================
# CONFIG
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("EFECT_DEFAULT_MODEL", "gpt-4.1-mini").strip()

TOKEN_SECRET = os.getenv("EFECT_TOKEN_SECRET", "change-me").strip()
REQUIRE_BEARER = os.getenv("EFECT_REQUIRE_BEARER", "0").strip() == "1"

DATA_DIR = os.getenv("EFECT_DATA_DIR", ".").strip()
DATA_PATH = Path(DATA_DIR)
DATA_PATH.mkdir(parents=True, exist_ok=True)

DB_PATH = os.getenv("EFECT_DB_PATH", str(DATA_PATH / "efect_ai.db")).strip()

UPLOAD_DIR = Path(os.getenv("EFECT_UPLOAD_DIR", str(DATA_PATH / "uploads"))).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("EFECT_MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

MAX_ATTACH_TEXT_KB = int(os.getenv("EFECT_MAX_ATTACH_TEXT_KB", "200"))
MAX_ATTACH_TEXT_BYTES = MAX_ATTACH_TEXT_KB * 1024

PREVIEW_TOKEN_TTL_SECONDS = int(os.getenv("EFECT_PREVIEW_TTL_SECONDS", "86400"))  # 24h default

# Workspace limits (keep stable)
WORKSPACE_MAX_FILES = int(os.getenv("EFECT_WORKSPACE_MAX_FILES", "80"))
WORKSPACE_MAX_TOTAL_KB = int(os.getenv("EFECT_WORKSPACE_MAX_TOTAL_KB", "1500"))
WORKSPACE_MAX_TOTAL_BYTES = WORKSPACE_MAX_TOTAL_KB * 1024


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
# DB
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
# SIGNED PREVIEW TOKENS
# =========================================================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def make_preview_token(file_id: str, stored_name: str, exp: int) -> str:
    msg = f"{file_id}|{stored_name}|{exp}".encode("utf-8")
    sig = hmac.new(TOKEN_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
    return _b64url(sig)


def verify_preview_token(file_id: str, stored_name: str, exp: int, token: str) -> bool:
    if exp < int(time.time()):
        return False
    expected = make_preview_token(file_id, stored_name, exp)
    return hmac.compare_digest(expected, token)


# =========================================================
# OPENAI (Responses API)
# =========================================================
def openai_responses(messages: List[Dict[str, str]], model: str) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY is not set on the server. Add it in your host environment variables."

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": messages}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
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


def openai_responses_stream(messages: List[Dict[str, str]], model: str):
    if not OPENAI_API_KEY:
        yield "OPENAI_API_KEY is not set on the server.\n"
        return

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": messages, "stream": True}

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
        if r.status_code >= 400:
            yield f"OpenAI error {r.status_code}: {r.text}"
            return

        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                evt = json.loads(data_str)
            except Exception:
                continue

            evt_type = evt.get("type") or evt.get("event")

            if evt_type == "response.output_text.delta":
                delta = evt.get("delta") or ""
                if delta:
                    yield delta
                continue

            if evt_type and "delta" in evt and isinstance(evt.get("delta"), str):
                yield evt["delta"]
                continue

            if evt_type in ("response.completed", "response.complete"):
                break

            if evt_type in ("response.error", "error"):
                err = evt.get("error") or evt
                yield f"\n[Error] {err}\n"
                break


# =========================================================
# PROMPTING
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
    cur.execute("SELECT role, content FROM turns WHERE session_id=? ORDER BY id DESC LIMIT 30", (session_id,))
    rows = list(reversed(cur.fetchall()))
    con.close()

    turns = [{"role": r["role"], "content": r["content"]} for r in rows]
    prompt = [
        {"role": "system", "content": "Summarize the conversation so far into concise bullet points with key decisions, constraints, and open questions."},
        {"role": "user", "content": json.dumps(turns, ensure_ascii=False)},
    ]
    return openai_responses(prompt, model)


# =========================================================
# KNOWLEDGE SEARCH
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


def clamp_workspace(files: Dict[str, str]) -> Dict[str, str]:
    items = list(files.items())[:WORKSPACE_MAX_FILES]
    out: Dict[str, str] = {}
    total = 0
    for name, content in items:
        safe_name = name.replace("\\", "/").lstrip("/").strip()[:180]
        if not safe_name:
            continue
        if not isinstance(content, str):
            continue
        b = content.encode("utf-8", errors="ignore")
        if total + len(b) > WORKSPACE_MAX_TOTAL_BYTES:
            break
        out[safe_name] = content
        total += len(b)
    return out


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


class SessionRenameRequest(BaseModel):
    session_id: str
    title: str


class EditWorkspaceRequest(BaseModel):
    instructions: str
    files: Dict[str, str]
    model: Optional[str] = None


class ExportWorkspaceZipRequest(BaseModel):
    files: Dict[str, str]
    zip_name: Optional[str] = None


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def root():
    return {"status": "EFECT AI server running"}


# ---------------------------
# IMAGE EDIT (ChatGPT-style)
# ---------------------------
@app.post("/api/image_edit")
async def image_edit(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None),
):
    """
    Upload an image + instruction prompt, returns edited image bytes (PNG).
    No persistent storage required.
    """
    require_token(authorization)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    img_bytes = await image.read()
    if len(img_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (50MB max)")

    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    files = {
        "image": (image.filename or "image.png", img_bytes, image.content_type or "image/png"),
    }
    data = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": "1024x1024",
    }

    try:
        r = requests.post(url, headers=headers, files=files, data=data, timeout=180)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}")

    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"OpenAI error {r.status_code}: {r.text}")

    out = r.json()
    try:
        b64 = out["data"][0]["b64_json"]
        edited_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected image response: {out}")

    return Response(content=edited_bytes, media_type="image/png")


# ---------------------------
# Chat UI at /chat
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
      width:320px;min-width:280px;max-width:400px;
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
    .pill{font-size:12px;color:var(--muted);border:1px solid var(--border);padding:6px 10px;border-radius:999px}
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
    .tabs{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .tab{border:1px solid var(--border);background:var(--panel2);color:var(--text);
      padding:8px 12px;border-radius:12px;cursor:pointer;font-size:13px}
    .tab.active{background:#0f1a12;border-color:#1d3a2a}

    .content{flex:1;overflow:auto;padding:18px;display:flex;justify-content:center}
    .chatWrap{width:min(1050px,100%);display:flex;flex-direction:column;gap:12px}

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
    .composerWrap{width:min(1050px,100%);display:flex;flex-direction:column;gap:10px}
    textarea{
      width:100%;min-height:56px;max-height:220px;resize:vertical;
      background:var(--panel2);color:var(--text);border:1px solid var(--border);
      border-radius:16px;padding:12px 12px;font-size:14px;outline:none
    }
    select,input[type="text"]{
      background:var(--panel2);color:var(--text);border:1px solid var(--border);
      border-radius:12px;padding:10px 12px
    }
    input[type="file"]{display:none}
    .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .right{margin-left:auto}
    .typing{font-size:13px;color:var(--muted);padding:6px 2px}

    /* Attachment cards (ChatGPT-like) */
    .attachTray{display:flex;flex-wrap:wrap;gap:10px}
    .attachCard{
      display:flex;gap:10px;align-items:center;
      border:1px solid var(--border);
      background:rgba(15,15,15,.85);
      border-radius:14px;
      padding:10px;
      max-width:520px;
      box-shadow:var(--shadow);
    }
    .attachThumb{
      width:76px;height:76px;border-radius:12px;object-fit:cover;
      border:1px solid var(--border);cursor:pointer;
      background:#0a0a0a;
    }
    .attachInfo{display:flex;flex-direction:column;gap:4px;min-width:200px}
    .attachName{font-size:13px;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .attachMeta{font-size:12px;color:var(--muted)}
    .attachActions{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}

    /* Modal viewer */
    .modal{
      position:fixed;inset:0;display:none;align-items:center;justify-content:center;
      background:rgba(0,0,0,.72);z-index:9999;padding:18px;
    }
    .modalInner{
      width:min(980px,100%);max-height:90vh;
      border:1px solid var(--border);border-radius:18px;
      background:#0c0c0c;box-shadow:var(--shadow);overflow:hidden;
      display:flex;flex-direction:column;
    }
    .modalTop{
      padding:10px 12px;border-bottom:1px solid #1b1b1b;
      display:flex;align-items:center;justify-content:space-between;gap:10px
    }
    .modalTitle{font-weight:900}
    .modalBody{padding:12px;display:flex;justify-content:center;align-items:center;overflow:auto}
    .modalImg{max-width:100%;max-height:75vh;border-radius:14px;border:1px solid var(--border)}

    @media (max-width: 980px){
      .sidebar{display:none}
    }
  </style>
</head>
<body>

<div class="modal" id="modal">
  <div class="modalInner">
    <div class="modalTop">
      <div class="modalTitle" id="modalTitle">Preview</div>
      <div class="stack">
        <button class="btn small" id="modalDownload">Download</button>
        <button class="btn small" id="modalClose">Close</button>
      </div>
    </div>
    <div class="modalBody">
      <img id="modalImg" class="modalImg" src="" alt="preview"/>
    </div>
  </div>
</div>

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
      <button class="btn small" id="uploadBtn">Attach file</button>
      <input id="filePicker" type="file" />
    </div>

    <div class="pill">
      ChatGPT-like: Attachments appear inside your message bubble. Images support Edit + Regenerate + Modal.
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div class="tabs">
        <button class="tab active" id="tabChat">Chat</button>
        <div class="pill">Streaming • Attachments • Image Edit</div>
      </div>
      <div class="stack right">
        <button class="btn small" id="regenerateChat">Regenerate chat</button>
        <button class="btn small" id="clear">Clear</button>
      </div>
    </div>

    <div class="content">
      <div class="chatWrap" id="chatView"></div>
    </div>

    <div class="composer" id="composerChat">
      <div class="composerWrap">

        <div class="row">
          <div class="pill">Pending attachments</div>
          <button class="btn small" id="clearPending">Clear</button>
        </div>

        <div class="attachTray" id="pendingTray"></div>

        <div class="row">
          <div class="pill">Image edit prompt</div>
          <input type="text" id="imgPrompt" placeholder='e.g. "make this pink and black"' style="flex:1"/>
          <button class="btn small" id="imgEditBtn">Edit image</button>
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
  // ===========================
  // State
  // ===========================
  const chatView = document.getElementById('chatView');
  const sessionsEl = document.getElementById('sessions');
  const msgEl = document.getElementById('msg');
  const modelEl = document.getElementById('model');
  const memEl = document.getElementById('mem');
  const knEl = document.getElementById('kn');
  const typingEl = document.getElementById('typing');
  const renameInput = document.getElementById('renameInput');
  const filePicker = document.getElementById('filePicker');
  const pendingTray = document.getElementById('pendingTray');

  const imgPromptEl = document.getElementById('imgPrompt');

  // Modal
  const modal = document.getElementById('modal');
  const modalImg = document.getElementById('modalImg');
  const modalTitle = document.getElementById('modalTitle');
  const modalClose = document.getElementById('modalClose');
  const modalDownload = document.getElementById('modalDownload');
  let modalDownloadName = 'download.png';
  let modalDownloadUrl = '';

  function openModal(url, title, dlName){
    modalImg.src = url;
    modalTitle.textContent = title || 'Preview';
    modalDownloadName = dlName || 'download.png';
    modalDownloadUrl = url;
    modal.style.display = 'flex';
  }
  function closeModal(){
    modal.style.display = 'none';
    modalImg.src = '';
    modalDownloadUrl = '';
  }
  modalClose.onclick = closeModal;
  modal.onclick = (e)=>{ if(e.target === modal) closeModal(); };
  modalDownload.onclick = ()=>{
    if(!modalDownloadUrl) return;
    const a = document.createElement('a');
    a.href = modalDownloadUrl;
    a.download = modalDownloadName;
    document.body.appendChild(a); a.click(); a.remove();
  };

  let sessionId = localStorage.getItem('efect_session_id') || '';
  let lastUserMessage = '';

  // Pending attachments (ChatGPT-like)
  // Each item: { kind:'server'|'local', type:'image'|'file', name, size, file, url, file_id, content_type, preview_url }
  let pending = [];

  // Image edit regenerate memory
  // lastImageEdit = { imageFile: File, prompt: string }
  let lastImageEdit = null;

  const TEXT_EXTS = new Set(['txt','log','md','json','verse','ini','cfg','yaml','yml','csv','py','js','ts','html','css','tsx','jsx']);
  function isTextName(name){
    const ext = (name.split('.').pop() || '').toLowerCase();
    return TEXT_EXTS.has(ext);
  }

  function fmtBytes(n){
    if(!n && n !== 0) return '';
    const kb = n/1024;
    if(kb < 1024) return kb.toFixed(1) + ' KB';
    return (kb/1024).toFixed(1) + ' MB';
  }

  function downloadBlobUrl(url, filename){
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'download';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function addMsg(role, text, attachments){
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
    copyBtn.onclick = async ()=>{ try{ await navigator.clipboard.writeText(text || ''); }catch(e){} };
    tools.appendChild(copyBtn);
    head.appendChild(tools);

    wrap.appendChild(head);

    if(attachments && attachments.length){
      const tray = document.createElement('div');
      tray.className = 'attachTray';
      attachments.forEach(att=>{
        tray.appendChild(renderAttachmentCard(att));
      });
      wrap.appendChild(tray);
    }

    if(text && text.length){
      const body = document.createElement('div');
      body.textContent = text;
      wrap.appendChild(body);
    }

    chatView.appendChild(wrap);
    chatView.scrollTop = chatView.scrollHeight;
    return wrap;
  }

  function renderAttachmentCard(att){
    const card = document.createElement('div');
    card.className = 'attachCard';

    const info = document.createElement('div');
    info.className = 'attachInfo';

    const name = document.createElement('div');
    name.className = 'attachName';
    name.textContent = att.name || 'attachment';

    const meta = document.createElement('div');
    meta.className = 'attachMeta';
    meta.textContent = `${att.type === 'image' ? 'Image' : 'File'} • ${fmtBytes(att.size)}${att.file_id ? ' • id: '+att.file_id : ''}`;

    info.appendChild(name);
    info.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'attachActions';

    if(att.type === 'image'){
      const img = document.createElement('img');
      img.className = 'attachThumb';
      img.src = att.url || att.preview_url || '';
      img.alt = att.name || 'image';
      img.onclick = ()=> openModal(img.src, att.name, (att.name || 'image') );

      card.appendChild(img);

      const viewBtn = document.createElement('button');
      viewBtn.className = 'btn small';
      viewBtn.textContent = 'View';
      viewBtn.onclick = ()=> openModal(img.src, att.name, (att.name || 'image'));

      const dlBtn = document.createElement('button');
      dlBtn.className = 'btn small';
      dlBtn.textContent = 'Download';
      dlBtn.onclick = async ()=>{
        if(att.kind === 'server' && att.file_id){
          // download through authenticated endpoint if enabled
          window.open(`/api/files/${att.file_id}`, '_blank');
        } else {
          downloadBlobUrl(img.src, att.name || 'image.png');
        }
      };

      actions.appendChild(viewBtn);
      actions.appendChild(dlBtn);
    } else {
      // non-image file card
      const icon = document.createElement('div');
      icon.className = 'attachThumb mono';
      icon.style.display='flex';
      icon.style.alignItems='center';
      icon.style.justifyContent='center';
      icon.style.fontSize='12px';
      icon.textContent = (att.name || 'file').split('.').pop().toUpperCase().slice(0,5);
      card.appendChild(icon);

      const dlBtn = document.createElement('button');
      dlBtn.className = 'btn small';
      dlBtn.textContent = 'Download';
      dlBtn.onclick = ()=>{
        if(att.kind === 'server' && att.file_id){
          window.open(`/api/files/${att.file_id}`, '_blank');
        } else if(att.url){
          downloadBlobUrl(att.url, att.name || 'file');
        }
      };
      actions.appendChild(dlBtn);
    }

    card.appendChild(info);
    card.appendChild(actions);
    return card;
  }

  function renderPendingTray(){
    pendingTray.innerHTML = '';
    if(pending.length === 0){
      const empty = document.createElement('div');
      empty.className = 'pill';
      empty.textContent = 'No pending attachments.';
      pendingTray.appendChild(empty);
      return;
    }
    pending.forEach((att, idx)=>{
      const card = renderAttachmentCard(att);
      // add remove btn
      const rm = document.createElement('button');
      rm.className = 'btn small';
      rm.textContent = 'Remove';
      rm.onclick = ()=>{
        pending.splice(idx,1);
        renderPendingTray();
      };
      card.querySelector('.attachActions').appendChild(rm);
      pendingTray.appendChild(card);
    });
  }

  function pendingServerIds(){
    return pending.filter(x=>x.kind==='server' && x.file_id).map(x=>x.file_id);
  }

  function lastPendingImage(){
    for(let i=pending.length-1;i>=0;i--){
      if(pending[i].type==='image') return pending[i];
    }
    return null;
  }

  // ===========================
  // Sessions
  // ===========================
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

  // ===========================
  // Chat streaming
  // ===========================
  async function sendStream(message, attachmentsForBubble){
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
        file_ids: pendingServerIds()
      })
    });

    const newSid = res.headers.get('x-efect-session-id');
    if (newSid) {
      sessionId = newSid;
      localStorage.setItem('efect_session_id', sessionId);
    }

    // show user bubble with inline attachments (ChatGPT feel)
    addMsg('user', message, attachmentsForBubble || []);

    const aiWrap = addMsg('assistant', '', []);
    const aiBody = document.createElement('div');
    aiWrap.appendChild(aiBody);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let full = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      const chunk = decoder.decode(value, {stream:true});
      full += chunk;
      aiBody.textContent = full;
      chatView.scrollTop = chatView.scrollHeight;
    }

    typingEl.style.display = 'none';
    await loadSessions();
    return full;
  }

  async function send(){
    const text = msgEl.value.trim();
    if(!text) return;

    // take snapshot of pending attachments into this message bubble
    const snapshot = pending.slice();
    // clear pending (like ChatGPT)
    pending = [];
    renderPendingTray();

    msgEl.value = '';
    lastUserMessage = text;
    await sendStream(text, snapshot);
  }

  async function regenerateChat(){
    if(!lastUserMessage) return;
    await sendStream(lastUserMessage, []);
  }

  // ===========================
  // Server upload (adds to pending)
  // ===========================
  async function uploadToServer(file){
    const fd = new FormData();
    fd.append('file', file);

    const r = await fetch('/api/files/upload', { method:'POST', body: fd });
    const data = await r.json();

    if(data && data.file && data.file.id){
      const f = data.file;
      const isImg = (f.content_type || '').startsWith('image/');
      const att = {
        kind: 'server',
        type: isImg ? 'image' : 'file',
        name: f.filename,
        size: f.bytes,
        file_id: f.id,
        content_type: f.content_type,
        preview_url: f.preview_url,
        url: isImg ? f.preview_url : ''
      };
      pending.push(att);
      renderPendingTray();
      addMsg('assistant', `Attached "${f.filename}" to your pending attachments. Send a message to include it in chat.`, []);
    } else {
      addMsg('assistant', 'Upload failed: ' + JSON.stringify(data), []);
    }
  }

  // ===========================
  // Image Edit + Regenerate
  // ===========================
  async function runImageEdit(imageFile, prompt, originalName){
    typingEl.style.display = 'block';

    const fd = new FormData();
    fd.append('prompt', prompt);
    fd.append('image', imageFile);

    const r = await fetch('/api/image_edit', { method:'POST', body: fd });
    typingEl.style.display = 'none';

    if(!r.ok){
      const t = await r.text();
      addMsg('assistant', 'Image edit failed: ' + t, []);
      return;
    }

    const blob = await r.blob();
    const url = URL.createObjectURL(blob);

    // Remember for regenerate
    lastImageEdit = { imageFile, prompt, originalName: originalName || imageFile.name || 'image.png' };

    // Show as AI message with attachment card
    const editedAtt = {
      kind: 'local',
      type: 'image',
      name: `Edited_${(originalName || imageFile.name || 'image').replace(/\s+/g,'_')}.png`,
      size: blob.size,
      url: url
    };

    const aiMsg = addMsg('assistant', `Image edit: ${prompt}`, [editedAtt]);

    // Add regenerate button under the message (ChatGPT feel)
    const controls = document.createElement('div');
    controls.className = 'row';
    controls.style.marginTop = '10px';

    const regen = document.createElement('button');
    regen.className = 'btn small';
    regen.textContent = 'Regenerate';
    regen.onclick = async ()=>{
      if(!lastImageEdit) return;
      await runImageEdit(lastImageEdit.imageFile, lastImageEdit.prompt, lastImageEdit.originalName);
    };

    const view = document.createElement('button');
    view.className = 'btn small';
    view.textContent = 'View';
    view.onclick = ()=> openModal(url, editedAtt.name, editedAtt.name);

    const dl = document.createElement('button');
    dl.className = 'btn small';
    dl.textContent = 'Download';
    dl.onclick = ()=> downloadBlobUrl(url, editedAtt.name);

    controls.appendChild(regen);
    controls.appendChild(view);
    controls.appendChild(dl);
    aiMsg.appendChild(controls);

    chatView.scrollTop = chatView.scrollHeight;
  }

  document.getElementById('imgEditBtn').onclick = async ()=>{
    const prompt = (imgPromptEl.value || '').trim();
    if(!prompt){
      addMsg('assistant','Type an image edit instruction first (example: "make this pink and black").', []);
      return;
    }

    const imgAtt = lastPendingImage();
    if(!imgAtt){
      addMsg('assistant','Attach an image first using “Attach file”. It will appear in Pending attachments.', []);
      return;
    }

    // Prefer local file if we have it; but pending uses server upload.
    // For ChatGPT-like behavior without paid storage, we edit from the local file BEFORE uploading.
    // Here we only have server item, so we prompt user to re-attach as local OR we fetch and blob it.
    // We will fetch preview_url and convert to File.
    let fileToEdit = null;

    try{
      const resp = await fetch(imgAtt.preview_url || imgAtt.url);
      const blob = await resp.blob();
      const name = imgAtt.name || 'image.png';
      fileToEdit = new File([blob], name, { type: blob.type || 'image/png' });
    }catch(e){
      addMsg('assistant', 'Could not load the attached image for editing. Re-attach the image and try again.', []);
      return;
    }

    // Show user “tool action” bubble with the image attachment + prompt
    addMsg('user', `[Image Edit] ${prompt}`, [imgAtt]);

    await runImageEdit(fileToEdit, prompt, imgAtt.name || fileToEdit.name);
  };

  document.getElementById('regenerateChat').onclick = regenerateChat;

  // ===========================
  // Wire UI
  // ===========================
  document.getElementById('send').onclick = send;
  document.getElementById('newChat').onclick = ()=>{
    sessionId = '';
    localStorage.removeItem('efect_session_id');
    chatView.innerHTML = '';
    pending = [];
    renderPendingTray();
    addMsg('assistant', 'New chat started.', []);
    loadSessions();
  };
  document.getElementById('clear').onclick = ()=>{ chatView.innerHTML = ''; };
  document.getElementById('renameBtn').onclick = renameChat;
  document.getElementById('refreshSessions').onclick = loadSessions;

  document.getElementById('uploadBtn').onclick = ()=> filePicker.click();
  filePicker.addEventListener('change', async (e)=>{
    const f = e.target.files && e.target.files[0];
    if(!f) return;

    // Upload to server so it can be used as chat attachment IDs.
    // (Also enables image edit by fetching preview_url)
    await uploadToServer(f);
    filePicker.value = '';
  });

  document.getElementById('clearPending').onclick = ()=>{
    pending = [];
    renderPendingTray();
  };

  msgEl.addEventListener('keydown', (e)=>{
    if(e.key==='Enter' && !e.shiftKey){
      e.preventDefault();
      send();
    }
  });

  addMsg('assistant', 'Ready. Attach files → they appear in your message bubble when you Send. Use Image Edit for ChatGPT-style edits (Regenerate + Modal included).', []);
  renderPendingTray();
  loadSessions();
</script>
</body>
</html>
"""


@app.head("/chat")
def chat_head():
    return Response(status_code=200)


# ---------------------------
# Sessions API
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
# Files (server uploads)
# ---------------------------
@app.post("/api/files/upload")
async def upload_file(
    request: Request,
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

    if REQUIRE_BEARER:
        exp = int(time.time()) + PREVIEW_TOKEN_TTL_SECONDS
        tok = make_preview_token(file_id, stored_name, exp)
        preview_url = f"/api/files/{file_id}/preview?exp={exp}&token={tok}"
    else:
        preview_url = f"/api/files/{file_id}"

    return {"ok": True, "file": {
        "id": file_id,
        "filename": original,
        "stored_name": stored_name,
        "bytes": size,
        "content_type": guessed_type,
        "saved_at": int(time.time()),
        "preview_url": preview_url
    }}


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


@app.get("/api/files/{file_id}/preview")
def preview_file(file_id: str, exp: int, token: str):
    con = db()
    cur = con.cursor()
    cur.execute("SELECT id, filename, stored_name, content_type FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")

    stored_name = row["stored_name"]
    if REQUIRE_BEARER:
        if not verify_preview_token(file_id, stored_name, exp, token):
            raise HTTPException(status_code=403, detail="Invalid or expired preview token")

    path = UPLOAD_DIR / stored_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    headers = {"Cache-Control": "private, max-age=60"}
    return FileResponse(
        str(path),
        filename=row["filename"],
        media_type=row["content_type"] or "application/octet-stream",
        headers=headers
    )


# ---------------------------
# Chat context builder + saving
# ---------------------------
def build_messages_for_request(req: ChatV2Request, sid: str, model: str) -> Tuple[List[Dict[str, str]], str]:
    now = int(time.time())
    con = db()
    cur = con.cursor()

    cur.execute(
        "INSERT INTO sessions(session_id,created_at,updated_at,title) VALUES(?,?,?,?) "
        "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at",
        (sid, now, now, None)
    )

    cur.execute("SELECT title FROM sessions WHERE session_id=?", (sid,))
    trow = cur.fetchone()
    if trow and (trow["title"] is None or str(trow["title"]).strip() == ""):
        auto = req.message.strip().split("\n")[0][:40]
        cur.execute("UPDATE sessions SET title=? WHERE session_id=?", (auto, sid))

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

    cur.execute("SELECT role, content FROM turns WHERE session_id=? ORDER BY id DESC LIMIT 12", (sid,))
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
# Streaming chat
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
            if full.strip():
                save_assistant_turn_and_summary(sid, full, model)

    headers = {
        "x-efect-session-id": sid,
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8", headers=headers)


# =========================================================
# Optional: workspace ZIP export endpoints (kept for later)
# =========================================================
@app.post("/api/workspace/export_zip")
def export_workspace_zip(req: ExportWorkspaceZipRequest):
    files = clamp_workspace(req.files or {})
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    zip_name = (req.zip_name or "EFECT_workspace.zip").strip()
    if not zip_name.lower().endswith(".zip"):
        zip_name += ".zip"

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, content in files.items():
            z.writestr(path, content)

    mem.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{zip_name}"'}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)
