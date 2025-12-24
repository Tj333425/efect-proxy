# efect_proxy.py
# EFECT Proxy + ChatGPT-like Web UI (single-file)
#
# Routes:
#   GET  /            -> Chat UI homepage
#   GET  /token       -> Token generator page (invite-code)
#   POST /token/new   -> Create a token (invite-code required)
#   GET  /download    -> Download page for EXE (uses EXE_URL env)
#   GET  /health      -> {"ok": true}
#   GET  /stats       -> basic usage stats
#   POST /responses   -> Desktop app endpoint (non-stream)
#   POST /stream      -> Browser endpoint (streaming text)
#   POST /upload      -> Upload file/image (optional)
#
# Env vars (Render):
#   OPENAI_API_KEY       = your real OpenAI key (server-side only)
#   EFECT_PROXY_TOKENS   = comma-separated tokens allowed (optional if using token store)
#   TOKEN_INVITE_CODE    = invite code to generate tokens (e.g. "EFECT2025")
#   EXE_URL              = direct URL to your EXE asset (GitHub Release asset URL recommended)
#   SITE_TITLE           = optional override title
#
# Start command (Render):
#   uvicorn efect_proxy:app --host 0.0.0.0 --port $PORT

import os
import re
import json
import time
import base64
import secrets
from typing import Optional, Dict, Any, Generator

import requests
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"

SITE_TITLE = os.getenv("SITE_TITLE", "EFECT AI")
EXE_URL = os.getenv("EXE_URL", "").strip()  # set to a downloadable link (GitHub release asset)

# Optional static allowlist tokens (you can still generate + store tokens using /token)
TOKENS_ENV = os.getenv("EFECT_PROXY_TOKENS", "").strip()
TOKENS_ALLOWLIST = {t.strip() for t in TOKENS_ENV.split(",") if t.strip()}

# Invite code for token generator
TOKEN_INVITE_CODE = os.getenv("TOKEN_INVITE_CODE", "").strip()

# Simple JSON storage (works fine for small scale)
DATA_DIR = os.getenv("DATA_DIR", "/tmp")
TOKENS_PATH = os.path.join(DATA_DIR, "efect_tokens.json")
USAGE_PATH = os.path.join(DATA_DIR, "efect_usage.json")

app = FastAPI(title="EFECT Proxy")


def _load_json(path: str, default: Any) -> Any:
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _token_store() -> Dict[str, Any]:
    return _load_json(TOKENS_PATH, {"tokens": {}})


def _usage_store() -> Dict[str, Any]:
    return _load_json(USAGE_PATH, {"by_token": {}, "total_requests": 0, "total_stream_requests": 0})


def _inc_usage(token: str, kind: str) -> None:
    u = _usage_store()
    u["total_requests"] = int(u.get("total_requests", 0)) + (1 if kind == "responses" else 0)
    u["total_stream_requests"] = int(u.get("total_stream_requests", 0)) + (1 if kind == "stream" else 0)
    bt = u.setdefault("by_token", {})
    row = bt.setdefault(token, {"requests": 0, "stream_requests": 0, "last_used": None})
    if kind == "responses":
        row["requests"] += 1
    else:
        row["stream_requests"] += 1
    row["last_used"] = int(time.time())
    _save_json(USAGE_PATH, u)


def _is_valid_token(token: str) -> bool:
    if not token:
        return False
    if token in TOKENS_ALLOWLIST:
        return True
    store = _token_store()
    return token in store.get("tokens", {})


def _require_token(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing proxy token (Authorization: Bearer <token>)")
    token = auth.split(" ", 1)[1].strip()
    if not _is_valid_token(token):
        raise HTTPException(status_code=403, detail="Invalid proxy token")
    return token


def _mask(s: str) -> str:
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return s[:3] + "*" * (len(s) - 6) + s[-3:]


def _html_home() -> str:
    # Single-file Chat UI (no build, no frameworks)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{SITE_TITLE}</title>
  <style>
    :root {{
      --bg:#0b0b0b;
      --panel:#0f1010;
      --panel2:#0d0f0f;
      --border:#222;
      --text:#e8e8e8;
      --muted:#a0a0a0;
      --green:#63ff73;
      --green2:#1bff4a;
      --shadow: 0 0 32px rgba(99,255,115,.12);
      --radius: 18px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
    }}
    *{{box-sizing:border-box}}
    body {{
      margin:0; background:radial-gradient(1200px 600px at 50% 20%, rgba(99,255,115,.10), transparent 60%), var(--bg);
      color:var(--text); font-family:var(--sans); height:100vh; overflow:hidden;
    }}
    .app {{
      display:flex; height:100vh;
    }}
    .sidebar {{
      width:320px; background:linear-gradient(180deg, rgba(15,16,16,.95), rgba(10,10,10,.95));
      border-right:1px solid var(--border); padding:16px; display:flex; flex-direction:column; gap:12px;
    }}
    .brand {{
      padding:14px 14px; border:1px solid #1d1d1d; border-radius:16px;
      background:rgba(0,0,0,.35); box-shadow:var(--shadow);
    }}
    .brand h1 {{
      margin:0; font-size:22px; letter-spacing:.08em; color:var(--green);
      text-shadow:0 0 18px rgba(99,255,115,.22);
    }}
    .brand .sub {{ margin-top:6px; color:var(--muted); font-size:13px; line-height:1.35; }}
    .btn {{
      display:inline-flex; align-items:center; justify-content:center;
      border-radius:12px; border:1px solid #2a2a2a; padding:10px 12px;
      background:rgba(255,255,255,.02); color:var(--text); cursor:pointer; gap:10px;
    }}
    .btn:hover {{ border-color:#3a3a3a; }}
    .btn.primary {{ border-color: rgba(99,255,115,.45); }}
    .row {{ display:flex; gap:10px; }}
    .row .btn {{ flex:1; }}
    .input {{
      width:100%; background:rgba(0,0,0,.35); border:1px solid #2a2a2a;
      color:var(--text); padding:10px 12px; border-radius:12px; outline:none;
    }}
    .input:focus {{ border-color: rgba(99,255,115,.55); box-shadow:0 0 0 3px rgba(99,255,115,.08); }}
    .tiny {{ font-size:12px; color:var(--muted); line-height:1.4; }}
    .main {{
      flex:1; display:flex; flex-direction:column;
    }}
    .topbar {{
      height:60px; border-bottom:1px solid var(--border); display:flex; align-items:center;
      padding:0 18px; gap:12px; background:rgba(0,0,0,.25);
    }}
    .dot {{
      width:10px; height:10px; border-radius:999px; background:var(--green);
      box-shadow:0 0 14px rgba(99,255,115,.65);
    }}
    .topbar .title {{ font-weight:650; letter-spacing:.04em; }}
    .topbar .right {{ margin-left:auto; display:flex; gap:10px; align-items:center; }}
    .link {{
      color:var(--green); text-decoration:none; font-size:13px;
    }}
    .link:hover {{ text-decoration:underline; }}

    .chat {{
      flex:1; overflow:auto; padding:18px; display:flex; flex-direction:column; gap:12px;
    }}
    .msg {{
      max-width:980px; width:100%;
      border:1px solid #202020; background:rgba(0,0,0,.25);
      border-radius:16px; padding:12px 14px; box-shadow: 0 0 18px rgba(0,0,0,.15);
    }}
    .msg.user {{ border-color: #242424; align-self:flex-end; background:rgba(255,255,255,.03); }}
    .msg.ai {{ border-color: rgba(99,255,115,.20); align-self:flex-start; }}
    .meta {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; color:var(--muted); font-size:12px; }}
    .pill {{
      font-size:11px; padding:3px 8px; border:1px solid #2a2a2a; border-radius:999px;
      background:rgba(255,255,255,.02);
    }}
    pre {{
      background:rgba(0,0,0,.45); border:1px solid #2a2a2a; padding:10px; border-radius:12px; overflow:auto;
      font-family:var(--mono); font-size:12.5px;
    }}
    code {{ font-family:var(--mono); }}
    .composer {{
      border-top:1px solid var(--border); padding:14px; background:rgba(0,0,0,.25);
    }}
    .composer .wrap {{
      max-width:1100px; margin:0 auto; display:flex; gap:10px; align-items:flex-end;
    }}
    textarea {{
      flex:1; min-height:48px; max-height:180px; resize:none;
      background:rgba(0,0,0,.35); border:1px solid #2a2a2a; color:var(--text);
      padding:12px 12px; border-radius:14px; outline:none; font-family:var(--sans);
    }}
    textarea:focus {{ border-color: rgba(99,255,115,.55); box-shadow:0 0 0 3px rgba(99,255,115,.08); }}
    .send {{
      width:120px; height:48px; border-radius:14px; border:1px solid rgba(99,255,115,.45);
      background:rgba(99,255,115,.08); color:var(--text); cursor:pointer; font-weight:650;
    }}
    .send:disabled {{ opacity:.55; cursor:not-allowed; }}
    .notice {{
      max-width:1100px; margin:10px auto 0; color:var(--muted); font-size:12px; display:flex; gap:10px; flex-wrap:wrap;
    attaching {{
      color:var(--muted);
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="sidebar">
      <div class="brand">
        <h1>EFECT AI</h1>
        <div class="sub">Browser-based chat UI (streaming).<br/>Token-gated proxy + uploads.</div>
      </div>

      <div class="row">
        <button class="btn primary" id="newChatBtn">New chat</button>
        <button class="btn" id="clearBtn">Clear</button>
      </div>

      <div>
        <div class="tiny" style="margin-bottom:6px;">Proxy token (required)</div>
        <input class="input" id="token" placeholder="efect-xxxxx" />
        <div class="tiny" style="margin-top:8px;">
          Need a token? Go to <a class="link" href="/token" target="_blank">/token</a>
        </div>
      </div>

      <div>
        <div class="tiny" style="margin-bottom:6px;">Model (optional)</div>
        <input class="input" id="model" placeholder="gpt-4.1-mini (or any you allow)" value="gpt-4.1-mini"/>
      </div>

      <div class="tiny" style="margin-top:auto;">
        Downloads: <a class="link" href="/download" target="_blank">/download</a><br/>
        Status: <a class="link" href="/health" target="_blank">/health</a> • <a class="link" href="/stats" target="_blank">/stats</a>
      </div>
    </div>

    <div class="main">
      <div class="topbar">
        <div class="dot"></div>
        <div class="title">EFECT AI — Live Chat</div>
        <div class="right">
          <a class="link" href="/download" target="_blank">Download Desktop App</a>
        </div>
      </div>

      <div class="chat" id="chat"></div>

      <div class="composer">
        <div class="wrap">
          <textarea id="msg" placeholder="Message EFECT AI... (Shift+Enter for new line)"></textarea>
          <button class="send" id="sendBtn">Send</button>
        </div>
        <div class="notice">
          <span>Tip: If you get 401/403, your token is missing or invalid.</span>
          <span>Streaming is enabled by default (closest to ChatGPT feel).</span>
        </div>
      </div>
    </div>
  </div>

<script>
const chat = document.getElementById("chat");
const msg = document.getElementById("msg");
const sendBtn = document.getElementById("sendBtn");
const tokenEl = document.getElementById("token");
const modelEl = document.getElementById("model");
const clearBtn = document.getElementById("clearBtn");
const newChatBtn = document.getElementById("newChatBtn");

let history = []; // [{role, content}]
let busy = false;

function scrollDown() {{
  chat.scrollTop = chat.scrollHeight;
}}

function escapeHtml(s) {{
  return s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}}

function renderMarkdownish(text) {{
  // minimal: code fences + inline code
  // ```lang ... ```
  const esc = escapeHtml(text);
  // code blocks
  let out = esc.replace(/```([\\s\\S]*?)```/g, (m, p1) => {{
    return `<pre><code>${p1.trim()}</code></pre>`;
  }});
  // inline code
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  // line breaks
  out = out.replace(/\\n/g, "<br/>");
  return out;
}}

function addMsg(role, content) {{
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "ai");
  const who = role === "user" ? "You" : "EFECT AI";
  div.innerHTML = `
    <div class="meta">
      <span class="pill">${who}</span>
      <span>${new Date().toLocaleTimeString()}</span>
    </div>
    <div class="body">${renderMarkdownish(content)}</div>
  `;
  chat.appendChild(div);
  scrollDown();
  return div;
}}

function setBusy(v) {{
  busy = v;
  sendBtn.disabled = v;
}}

function newChat() {{
  history = [];
  chat.innerHTML = "";
  addMsg("ai", "EFECT AI online. Drop your question.");
}}

clearBtn.addEventListener("click", () => {{
  chat.innerHTML = "";
  history = [];
}});

newChatBtn.addEventListener("click", () => newChat());

msg.addEventListener("keydown", (e) => {{
  if (e.key === "Enter" && !e.shiftKey) {{
    e.preventDefault();
    sendBtn.click();
  }}
}});

async function streamChat(userText) {{
  const token = tokenEl.value.trim();
  if (!token) {{
    addMsg("ai", "Missing token. Put your token in the left sidebar.");
    return;
  }}

  const model = (modelEl.value || "gpt-4.1-mini").trim();

  // Build OpenAI Responses-style input
  const payload = {{
    model,
    input: [
      ...history.map(m => ({{ role: m.role, content: [{{type:"input_text", text: m.content}}] }})),
      {{ role: "user", content: [{{type:"input_text", text: userText}}] }}
    ],
  }};

  setBusy(true);

  addMsg("user", userText);
  history.push({{role:"user", content:userText}});

  const aiDiv = addMsg("assistant", "");
  const bodyEl = aiDiv.querySelector(".body");

  try {{
    const res = await fetch("/stream", {{
      method: "POST",
      headers: {{
        "Content-Type": "application/json",
        "Authorization": "Bearer " + token
      }},
      body: JSON.stringify(payload)
    }});

    if (!res.ok) {{
      const t = await res.text();
      bodyEl.innerHTML = renderMarkdownish("Error: " + res.status + "\\n" + t);
      setBusy(false);
      return;
    }}

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = "";

    while (true) {{
      const {{value, done}} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, {{stream:true}});
      full += chunk;
      bodyEl.innerHTML = renderMarkdownish(full);
      scrollDown();
    }}

    history.push({{role:"assistant", content: full}});
  }} catch (err) {{
    bodyEl.innerHTML = renderMarkdownish("Network error: " + String(err));
  }} finally {{
    setBusy(false);
  }}
}}

sendBtn.addEventListener("click", async () => {{
  if (busy) return;
  const text = msg.value.trim();
  if (!text) return;
  msg.value = "";
  await streamChat(text);
}});

newChat();
</script>
</body>
</html>
"""


def _html_token_page() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{SITE_TITLE} — Token Generator</title>
  <style>
    body {{
      margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background:#0b0b0b; color:#e8e8e8; height:100vh; display:flex; align-items:center; justify-content:center;
    }}
    .box {{
      width:min(720px, 92vw); border:1px solid #222; border-radius:18px; padding:22px;
      background:rgba(255,255,255,.02);
      box-shadow:0 0 32px rgba(99,255,115,.10);
    }}
    h1 {{ margin:0; color:#63ff73; letter-spacing:.08em; }}
    p {{ color:#a0a0a0; line-height:1.45; }}
    .row {{ display:flex; gap:10px; margin-top:14px; }}
    input {{
      flex:1; padding:12px 12px; border-radius:12px; border:1px solid #2a2a2a;
      background:rgba(0,0,0,.35); color:#e8e8e8; outline:none;
    }}
    input:focus {{ border-color: rgba(99,255,115,.55); box-shadow:0 0 0 3px rgba(99,255,115,.08); }}
    button {{
      padding:12px 14px; border-radius:12px; border:1px solid rgba(99,255,115,.45);
      background:rgba(99,255,115,.08); color:#e8e8e8; cursor:pointer; font-weight:650;
      width:160px;
    }}
    pre {{
      margin-top:14px; padding:12px; border-radius:12px;
      background:rgba(0,0,0,.45); border:1px solid #2a2a2a; overflow:auto;
    }}
    a {{ color:#63ff73; }}
  </style>
</head>
<body>
  <div class="box">
    <h1>EFECT TOKEN</h1>
    <p>
      Generate an access token for the EFECT AI proxy.<br/>
      This is protected by an invite code so random people can’t mint unlimited tokens.
    </p>

    <div class="row">
      <input id="code" placeholder="Invite code"/>
      <button id="gen">Generate</button>
    </div>

    <pre id="out">Token will appear here…</pre>

    <p>
      Use the token on the homepage: <a href="/" target="_blank">/</a><br/>
      Download desktop app: <a href="/download" target="_blank">/download</a>
    </p>
  </div>

<script>
const out = document.getElementById("out");
document.getElementById("gen").addEventListener("click", async () => {{
  const invite = document.getElementById("code").value.trim();
  out.textContent = "Generating…";
  const r = await fetch("/token/new", {{
    method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{invite}})
  }});
  const t = await r.text();
  try {{
    const j = JSON.parse(t);
    if (!r.ok) {{
      out.textContent = "Error: " + (j.detail || t);
      return;
    }}
    out.textContent = j.token;
  }} catch {{
    out.textContent = t;
  }}
}});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _html_home()


@app.get("/token", response_class=HTMLResponse)
def token_page() -> str:
    return _html_token_page()


@app.post("/token/new")
async def token_new(req: Request) -> JSONResponse:
    body = await req.json()
    invite = str(body.get("invite", "")).strip()

    if not TOKEN_INVITE_CODE:
        raise HTTPException(status_code=500, detail="Server missing TOKEN_INVITE_CODE env var")
    if invite != TOKEN_INVITE_CODE:
        raise HTTPException(status_code=403, detail="Invalid invite code")

    store = _token_store()
    tokens = store.setdefault("tokens", {})

    # Create a short, shareable token
    tok = "efect-" + secrets.token_urlsafe(12).replace("_", "").replace("-", "")[:16].lower()
    tokens[tok] = {"created_at": int(time.time()), "revoked": False}
    _save_json(TOKENS_PATH, store)

    return JSONResponse({"token": tok})


@app.get("/download", response_class=HTMLResponse)
def download_page() -> str:
    # If EXE_URL is set, show a download button. Otherwise instructions.
    btn = ""
    if EXE_URL:
        btn = f'<p><a href="{EXE_URL}" style="color:#0b0b0b;background:#63ff73;padding:10px 14px;border-radius:12px;display:inline-block;text-decoration:none;font-weight:700;">Download EFECT AI Desktop App</a></p>'
    else:
        btn = "<p><b>EXE_URL is not set</b>. Add EXE_URL in Render env vars to a GitHub Release asset direct URL.</p>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{SITE_TITLE} — Download</title>
  <style>
    body{{margin:0;background:#0b0b0b;color:#e8e8e8;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;display:flex;align-items:center;justify-content:center;height:100vh}}
    .box{{width:min(760px,92vw);border:1px solid #222;border-radius:18px;padding:22px;background:rgba(255,255,255,.02);box-shadow:0 0 32px rgba(99,255,115,.10)}}
    h1{{margin:0;color:#63ff73;letter-spacing:.08em}}
    p{{color:#a0a0a0;line-height:1.45}}
    a{{color:#63ff73}}
  </style>
</head>
<body>
  <div class="box">
    <h1>EFECT AI DESKTOP</h1>
    <p>Download the EFECT AI desktop app.</p>
    {btn}
    <p>Back to chat: <a href="/">/</a></p>
  </div>
</body>
</html>
"""


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    u = _usage_store()
    # Return masked token keys for safety
    masked = { _mask(k): v for k, v in u.get("by_token", {}).items() }
    return {
        "ok": True,
        "total_requests": u.get("total_requests", 0),
        "total_stream_requests": u.get("total_stream_requests", 0),
        "by_token": masked,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_exe_url": bool(EXE_URL),
    }


def _openai_headers() -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


@app.post("/responses")
async def responses(req: Request) -> JSONResponse:
    token = _require_token(req)
    _inc_usage(token, "responses")

    body = await req.json()

    # Basic safety: ensure model exists
    if "model" not in body:
        body["model"] = "gpt-4.1-mini"

    try:
        r = requests.post(OPENAI_RESPONSES_URL, headers=_openai_headers(), json=body, timeout=300)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    if r.status_code >= 400:
        return JSONResponse(status_code=r.status_code, content=r.json() if _is_json(r.text) else {"detail": r.text})

    return JSONResponse(r.json())


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


@app.post("/stream")
async def stream(req: Request) -> StreamingResponse:
    """
    Streams plain text back to the browser (closest ChatGPT feel).
    Frontend reads this as a text stream and appends as it arrives.
    """
    token = _require_token(req)
    _inc_usage(token, "stream")

    body = await req.json()
    body["stream"] = True
    if "model" not in body:
        body["model"] = "gpt-4.1-mini"

    def gen() -> Generator[bytes, None, None]:
        try:
            with requests.post(
                OPENAI_RESPONSES_URL,
                headers=_openai_headers(),
                json=body,
                stream=True,
                timeout=300,
            ) as r:
                if r.status_code >= 400:
                    # Emit error as text so UI shows it
                    try:
                        err = r.json()
                    except Exception:
                        err = {"detail": r.text}
                    yield (json.dumps(err, indent=2)).encode("utf-8")
                    return

                # OpenAI streams SSE-like lines. We extract text deltas if present,
                # otherwise pass through any readable text fragments.
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    # Many lines start with: "data: {...}"
                    if raw.startswith("data:"):
                        data = raw[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            evt = json.loads(data)
                        except Exception:
                            continue

                        # Try to pull text from common fields
                        delta = (
                            evt.get("delta")
                            or evt.get("text")
                            or ""
                        )

                        # Responses API commonly nests output text in:
                        # evt["response"]["output"] ... but varies; handle generically:
                        if not delta:
                            delta = _extract_any_text(evt)

                        if delta:
                            yield delta.encode("utf-8")
                    else:
                        # Fallback: pass through raw lines that might contain content
                        if "{" not in raw and raw.strip():
                            yield raw.encode("utf-8")
        except requests.RequestException as e:
            yield f"\n[Upstream error] {e}\n".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


def _extract_any_text(obj: Any) -> str:
    """
    Heuristic: walk dict/list and collect small text fragments that look like deltas.
    Keeps this lightweight and safe.
    """
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("text", "delta") and isinstance(v, str) and v:
                    out.append(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return "".join(out)


@app.post("/upload")
async def upload(req: Request, file: UploadFile = File(...)) -> JSONResponse:
    """
    Optional helper for web/desktop:
    - Images -> return data_url (base64) for input_image
    - PDFs/others -> upload to OpenAI Files, return file_id
    """
    _require_token(req)

    filename = file.filename or "upload"
    content = await file.read()
    ext = os.path.splitext(filename)[1].lower()

    # Image -> data URL
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")
        b64 = base64.b64encode(content).decode("utf-8")
        return JSONResponse({"kind": "image", "name": filename, "data_url": f"data:{mime};base64,{b64}"})

    # Otherwise upload to OpenAI Files
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    files = {"file": (filename, content, "application/octet-stream")}
    data = {"purpose": "assistants"}  # widely accepted purpose
    try:
        r = requests.post(
            OPENAI_FILES_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files,
            data=data,
            timeout=300,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    if r.status_code >= 400:
        return JSONResponse(status_code=r.status_code, content=r.json() if _is_json(r.text) else {"detail": r.text})

    j = r.json()
    return JSONResponse({"kind": "file", "name": filename, "file_id": j.get("id"), "raw": j})
