import os
import json
import time
import secrets
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

# -------------------------
# ENV (Render)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

TOKENS_ENV = os.getenv("EFECT_PROXY_TOKENS", "").strip()
TOKENS = set(t.strip() for t in TOKENS_ENV.split(",") if t.strip())

ADMIN_SECRET = os.getenv("EFECT_ADMIN_SECRET", "").strip()
DOWNLOAD_URL = os.getenv("EFECT_DOWNLOAD_URL", "").strip()
FORCE_MODEL = os.getenv("EFECT_FORCE_MODEL", "").strip()

# -------------------------
# Simple in-memory usage stats
# -------------------------
START_TS = time.time()
REQ_COUNT_TOTAL = 0
REQ_COUNT_BY_TOKEN: Dict[str, int] = {}
LAST_REQ_TS_BY_TOKEN: Dict[str, float] = {}

# -------------------------
# EFECT Prompt Packs (edit freely)
# -------------------------
QUESTIONS: List[Dict[str, str]] = [
    {
        "title": "Creative ideas",
        "prompt": "Give me 10 creative content ideas for EFECT. Make them unique, high-energy, and easy to execute."
    },
    {
        "title": "Rewrite like a pro",
        "prompt": "Rewrite this text to be more professional and persuasive while keeping it short:\n\nPASTE TEXT HERE"
    },
    {
        "title": "Brand voice",
        "prompt": "Define EFECT's brand voice in 8 bullet points (tone, vocabulary, do/don't)."
    },
    {
        "title": "YouTube title pack",
        "prompt": "Generate 15 YouTube titles for EFECT that are aggressive, clickable, and not cringe. Include 5 that use numbers."
    },
    {
        "title": "ChatGPT-like open-ended",
        "prompt": "Be my general-purpose chatbot. Ask 1 clarifying question only if absolutely needed; otherwise answer directly and creatively."
    },
]

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="EFECT Proxy")


# -------------------------
# Helpers
# -------------------------
def require_token(req: Request) -> str:
    """
    Token is required via:
      Authorization: Bearer <token>
    If EFECT_PROXY_TOKENS is empty, treat as open (not recommended).
    """
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing proxy token")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing proxy token")
    if TOKENS and token not in TOKENS:
        raise HTTPException(status_code=403, detail="Invalid proxy token")
    return token


def bump_usage(token: str):
    global REQ_COUNT_TOTAL
    REQ_COUNT_TOTAL += 1
    REQ_COUNT_BY_TOKEN[token] = REQ_COUNT_BY_TOKEN.get(token, 0) + 1
    LAST_REQ_TS_BY_TOKEN[token] = time.time()


def extract_output_text(openai_json: Dict[str, Any]) -> str:
    out = []
    for item in openai_json.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for c in (item.get("content") or []):
            if c.get("type") == "output_text" and "text" in c:
                out.append(c["text"])
    return "".join(out).strip()


def themed_page(title: str, body_html: str) -> str:
    # Single-file EFECT theme
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0b0b0b;
      --panel: #101010;
      --panel2: #0f0f0f;
      --text: #e8e8e8;
      --muted: #9aa0aa;
      --green: #1bff4a;
      --border: #2a2a2a;
      --shadow: rgba(0,0,0,.45);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(1100px 650px at 50% 35%, rgba(27,255,74,.08), transparent 60%),
        var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      min-height: 100vh;
    }}
    .top {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(11,11,11,.88);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
    }}
    .nav {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }}
    .brand {{
      display:flex;
      align-items: baseline;
      gap: 10px;
      font-weight: 900;
      letter-spacing: .5px;
    }}
    .brand .efect {{
      color: var(--green);
      font-size: 20px;
      text-shadow: 0 0 18px rgba(27,255,74,.30);
    }}
    .brand .sub {{
      color: var(--text);
      font-size: 16px;
      opacity: .9;
    }}
    .links {{
      display:flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    a.btn {{
      display:inline-block;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(16,16,16,.7);
      color: var(--text);
      text-decoration: none;
      font-size: 14px;
    }}
    a.btn:hover {{
      border-color: rgba(27,255,74,.65);
      box-shadow: 0 0 18px rgba(27,255,74,.12);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 18px;
    }}
    .card {{
      background: rgba(16,16,16,.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 0 60px var(--shadow);
      overflow: hidden;
    }}
    .muted {{ color: var(--muted); }}
    code {{
      background:#0c0c0c;
      border: 1px solid #222;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    /* Inputs */
    input, textarea, select {{
      width: 100%;
      background: var(--panel2);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 12px;
      padding: 10px 12px;
      outline: none;
    }}
    textarea {{ resize: vertical; }}
    .btnSolid {{
      border: 1px solid rgba(27,255,74,.55);
      background: rgba(27,255,74,.12);
      color: var(--text);
      cursor: pointer;
      padding: 10px 12px;
      border-radius: 12px;
      font-weight: 700;
    }}
    .btnSolid:hover {{
      box-shadow: 0 0 18px rgba(27,255,74,.14);
    }}
  </style>
</head>
<body>
  <div class="top">
    <div class="nav">
      <div class="brand">
        <div class="efect">EFECT</div>
        <div class="sub">AI</div>
      </div>
      <div class="links">
        <a class="btn" href="/">Home</a>
        <a class="btn" href="/chat">Chat</a>
        <a class="btn" href="/questions">Questions</a>
        <a class="btn" href="/stats">Status</a>
        <a class="btn" href="/download">Download</a>
      </div>
    </div>
  </div>

  <div class="wrap">
    {body_html}
  </div>
</body>
</html>"""


def home_html() -> str:
    body = """
    <div class="card" style="padding:42px;">
      <div style="text-align:center;">
        <div style="font-size:56px;font-weight:900;color:var(--green);text-shadow:0 0 18px rgba(27,255,74,.35);letter-spacing:1px;">
          EFECT AI
        </div>
        <div class="muted" style="margin-top:10px;font-size:18px;line-height:1.5;">
          Secure Online AI Proxy<br/>
          Browser Chat • Desktop App • File Uploads (optional)
        </div>
        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:26px;">
          <a class="btn" href="/chat">Open Chat UI</a>
          <a class="btn" href="/questions">Question Packs</a>
          <a class="btn" href="/health">Health</a>
          <a class="btn" href="/stats">Status & Usage</a>
        </div>
        <div style="margin-top:18px;font-weight:800;color:var(--green);">Status: ONLINE</div>
        <div class="muted" style="margin-top:18px;font-size:14px;">
          API route: <code>POST /responses</code> • Test: <code>/health</code>
        </div>
      </div>
    </div>
    """
    return themed_page("EFECT AI", body)


def questions_html() -> str:
    # Render questions as clickable "Use in Chat" buttons
    items = []
    for i, q in enumerate(QUESTIONS):
        title = (q.get("title") or f"Question {i+1}").replace("<", "&lt;").replace(">", "&gt;")
        prompt = (q.get("prompt") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        items.append(f"""
          <div style="border:1px solid var(--border);border-radius:16px;padding:16px;background:rgba(15,15,15,.7);">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
              <div style="font-weight:900;color:var(--green);font-size:16px;">{title}</div>
              <button class="btnSolid" onclick="usePrompt({json.dumps(q.get('prompt',''))})">Use in Chat</button>
            </div>
            <div class="muted" style="margin-top:10px;white-space:pre-wrap;line-height:1.45;">{prompt}</div>
          </div>
        """)

    body = f"""
    <div class="card" style="padding:22px;">
      <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center;">
        <div>
          <div style="font-size:22px;font-weight:900;">EFECT Questions</div>
          <div class="muted" style="margin-top:6px;">
            Click <b>Use in Chat</b> to load a prompt into the chat box.
          </div>
        </div>
        <a class="btn" href="/chat">Go to Chat</a>
      </div>

      <div style="display:grid;gap:14px;margin-top:18px;">
        {''.join(items)}
      </div>
    </div>

    <script>
      function usePrompt(text) {{
        // Store to localStorage so /chat can auto-fill
        localStorage.setItem("efect_prefill", text || "");
        window.location.href = "/chat";
      }}
    </script>
    """
    return themed_page("EFECT AI • Questions", body)


def chat_html() -> str:
    # Single-file Chat UI that calls POST /responses with Bearer token
    body = """
    <div class="card">
      <div style="padding:18px 18px 0 18px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:flex-end;">
        <div>
          <div style="font-size:22px;font-weight:900;">Browser Chat</div>
          <div class="muted" style="margin-top:6px;">Closest “ChatGPT-like” feel, running through your EFECT proxy.</div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <a class="btn" href="/questions">Questions</a>
          <a class="btn" href="/stats">Status</a>
        </div>
      </div>

      <div style="padding:18px;display:grid;gap:12px;border-top:1px solid var(--border);margin-top:14px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
          <div>
            <div class="muted" style="margin:0 0 6px 2px;font-size:13px;">Proxy Base URL</div>
            <input id="proxy" placeholder="https://efect-proxy.onrender.com" />
          </div>
          <div>
            <div class="muted" style="margin:0 0 6px 2px;font-size:13px;">EFECT Token (Bearer)</div>
            <input id="token" placeholder="efect-xxxx" />
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 220px;gap:12px;">
          <div>
            <div class="muted" style="margin:0 0 6px 2px;font-size:13px;">System Prompt (optional)</div>
            <textarea id="system" rows="3" placeholder="You are EFECT AI..."></textarea>
          </div>
          <div>
            <div class="muted" style="margin:0 0 6px 2px;font-size:13px;">Model</div>
            <input id="model" placeholder="gpt-4.1-mini" />
            <div style="margin-top:10px;display:flex;gap:10px;">
              <button class="btnSolid" style="width:100%;" onclick="clearChat()">Clear</button>
            </div>
          </div>
        </div>

        <div id="chat" style="height:460px;overflow:auto;padding:14px;border:1px solid var(--border);border-radius:16px;background:rgba(15,15,15,.7);">
          <!-- messages -->
        </div>

        <div style="display:grid;grid-template-columns:1fr 140px;gap:12px;align-items:end;">
          <div>
            <div class="muted" style="margin:0 0 6px 2px;font-size:13px;">Message</div>
            <textarea id="msg" rows="3" placeholder="Type here... (Shift+Enter new line)"></textarea>
            <div class="muted" style="margin-top:8px;font-size:12px;">
              Tip: If you get 401/403, your token is missing/invalid.
            </div>
          </div>
          <button class="btnSolid" style="height:56px;" onclick="send()">Send</button>
        </div>

        <div id="err" class="muted" style="font-size:13px;white-space:pre-wrap;"></div>
      </div>
    </div>

    <script>
      const els = {
        proxy: document.getElementById("proxy"),
        token: document.getElementById("token"),
        system: document.getElementById("system"),
        model: document.getElementById("model"),
        chat: document.getElementById("chat"),
        msg: document.getElementById("msg"),
        err: document.getElementById("err"),
      };

      // Load saved settings
      function loadSettings() {
        els.proxy.value = localStorage.getItem("efect_proxy") || window.location.origin;
        els.token.value = localStorage.getItem("efect_token") || "";
        els.system.value = localStorage.getItem("efect_system") || "";
        els.model.value = localStorage.getItem("efect_model") || "gpt-4.1-mini";

        const prefill = localStorage.getItem("efect_prefill");
        if (prefill) {
          els.msg.value = prefill;
          localStorage.removeItem("efect_prefill");
        }
      }

      function saveSettings() {
        localStorage.setItem("efect_proxy", els.proxy.value.trim());
        localStorage.setItem("efect_token", els.token.value.trim());
        localStorage.setItem("efect_system", els.system.value);
        localStorage.setItem("efect_model", els.model.value.trim());
      }

      function scrollBottom() {
        els.chat.scrollTop = els.chat.scrollHeight;
      }

      function escapeHtml(s) {
        return (s||"")
          .replaceAll("&","&amp;")
          .replaceAll("<","&lt;")
          .replaceAll(">","&gt;");
      }

      // History used to build /responses payload with correct types
      let history = []; // {role:'user'|'assistant', text:'...'}

      function renderMessage(role, text) {
        const wrap = document.createElement("div");
        wrap.style.margin = "0 0 12px 0";
        wrap.style.display = "flex";
        wrap.style.justifyContent = (role === "user" ? "flex-end" : "flex-start");

        const bubble = document.createElement("div");
        bubble.style.maxWidth = "82%";
        bubble.style.padding = "12px 12px";
        bubble.style.borderRadius = "14px";
        bubble.style.border = "1px solid var(--border)";
        bubble.style.background = (role === "user" ? "rgba(27,255,74,.10)" : "rgba(16,16,16,.85)");
        bubble.style.whiteSpace = "pre-wrap";
        bubble.style.lineHeight = "1.45";

        const label = document.createElement("div");
        label.style.fontSize = "12px";
        label.style.opacity = "0.75";
        label.style.marginBottom = "6px";
        label.innerText = (role === "user" ? "You" : "EFECT AI");

        const content = document.createElement("div");
        content.innerHTML = escapeHtml(text);

        bubble.appendChild(label);
        bubble.appendChild(content);
        wrap.appendChild(bubble);
        els.chat.appendChild(wrap);
        scrollBottom();
      }

      function clearChat() {
        history = [];
        els.chat.innerHTML = "";
        els.err.innerText = "";
      }

      function buildPayload(userText) {
        const system = (els.system.value || "").trim();
        const model = (els.model.value || "gpt-4.1-mini").trim();

        const input = [];
        if (system) {
          input.push({ role: "system", content: [{ type: "input_text", text: system }] });
        }

        // Critical fix: assistant history uses output_text
        for (const m of history) {
          if (!m.text) continue;
          if (m.role === "user") {
            input.push({ role: "user", content: [{ type: "input_text", text: m.text }] });
          } else if (m.role === "assistant") {
            input.push({ role: "assistant", content: [{ type: "output_text", text: m.text }] });
          }
        }

        input.push({ role: "user", content: [{ type: "input_text", text: userText }] });

        return { model, input, stream: false };
      }

      async function send() {
        els.err.innerText = "";
        const proxy = (els.proxy.value || "").trim().replace(/\\/+$/,"");
        const token = (els.token.value || "").trim();
        const text = (els.msg.value || "").trim();

        if (!proxy.startsWith("http")) {
          els.err.innerText = "Set Proxy Base URL to https://...";
          return;
        }
        if (!token) {
          els.err.innerText = "Enter your EFECT token.";
          return;
        }
        if (!text) return;

        saveSettings();

        renderMessage("user", text);
        history.push({ role: "user", text });
        els.msg.value = "";

        const payload = buildPayload(text);

        try {
          const res = await fetch(proxy + "/responses", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": "Bearer " + token
            },
            body: JSON.stringify(payload)
          });

          const data = await res.json();

          if (!res.ok) {
            els.err.innerText = JSON.stringify(data, null, 2);
            renderMessage("assistant", "Error: " + (data?.error?.message || data?.detail || "Request failed"));
            history.push({ role: "assistant", text: "Error: request failed." });
            return;
          }

          const out = (data.output_text || "").trim() || "(No output_text returned.)";
          renderMessage("assistant", out);
          history.push({ role: "assistant", text: out });

        } catch (e) {
          els.err.innerText = String(e);
          renderMessage("assistant", "Error: " + String(e));
          history.push({ role: "assistant", text: "Error: network failure." });
        }
      }

      // Enter to send, Shift+Enter for newline
      els.msg.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          send();
        }
      });

      loadSettings();
    </script>
    """
    return themed_page("EFECT AI • Chat", body)


# -------------------------
# Core routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return home_html()


@app.get("/chat", response_class=HTMLResponse)
def chat():
    return chat_html()


@app.get("/questions", response_class=HTMLResponse)
def questions():
    return questions_html()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/download")
def download():
    if not DOWNLOAD_URL:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "EFECT_DOWNLOAD_URL not set. Set it on Render to your GitHub Release asset URL.",
                "example": "https://github.com/<you>/<repo>/releases/download/<tag>/EFECT_AI.exe",
            },
        )
    return RedirectResponse(DOWNLOAD_URL, status_code=302)


@app.get("/stats")
def stats():
    uptime = int(time.time() - START_TS)
    return {
        "ok": True,
        "uptime_seconds": uptime,
        "total_requests": REQ_COUNT_TOTAL,
        "tokens_configured": len(TOKENS),
        "requests_by_token": REQ_COUNT_BY_TOKEN,
        "last_request_ts_by_token": LAST_REQ_TS_BY_TOKEN,
    }


@app.post("/token")
async def token_generator(req: Request):
    """
    Admin-only token generator.
    Header: X-Admin-Secret: <EFECT_ADMIN_SECRET>
    Note: This adds the token in-memory; for permanent use add it to EFECT_PROXY_TOKENS in Render.
    """
    if not ADMIN_SECRET:
        raise HTTPException(status_code=400, detail="EFECT_ADMIN_SECRET not set on server")
    admin = (req.headers.get("x-admin-secret") or "").strip()
    if admin != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    new_token = "efect-" + secrets.token_urlsafe(18)
    TOKENS.add(new_token)
    return {
        "token": new_token,
        "note": "Added in-memory. For permanent access add it to EFECT_PROXY_TOKENS in Render env.",
    }


# -------------------------
# Proxy routes
# -------------------------
@app.post("/responses")
async def responses(req: Request):
    """
    Proxy forward to OpenAI Responses API.
    Enforces stream=False to keep UI stable and avoid duplicates.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    token = require_token(req)
    bump_usage(token)

    body = await req.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    # Keep stable (prevents duplicate text in naive clients)
    body["stream"] = False

    # Optional force model
    if FORCE_MODEL:
        body["model"] = FORCE_MODEL

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=body, timeout=300)

    if r.status_code >= 400:
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            return JSONResponse(status_code=r.status_code, content={"error": {"message": r.text}})

    data = r.json()
    out_text = extract_output_text(data)

    return {
        "output_text": out_text,
        "raw": data,  # remove later if you want
    }


@app.post("/upload")
async def upload(req: Request, file: UploadFile = File(...)):
    """
    Optional placeholder. Not required for /chat.
    """
    token = require_token(req)
    bump_usage(token)
    contents = await file.read()
    return {"ok": True, "filename": file.filename, "bytes": len(contents)}
