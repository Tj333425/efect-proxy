import os
import json
import time
import secrets
from typing import Optional, Dict, Any, Tuple

import requests
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

# Server-only secrets (set in Render)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Gate access with shared tokens
# Example: EFECT_PROXY_TOKENS="token1,token2,token3"
TOKENS_ENV = os.getenv("EFECT_PROXY_TOKENS", "").strip()
TOKENS = set(t.strip() for t in TOKENS_ENV.split(",") if t.strip())

# Optional: allow creating tokens via /token page
ADMIN_SECRET = os.getenv("EFECT_ADMIN_SECRET", "").strip()

# Optional: redirect /download to a GitHub Release asset link (set this)
# Example:
# EFECT_DOWNLOAD_URL="https://github.com/TJ333425/efect-proxy/releases/download/v1.0.0/EFECT_AI.exe"
DOWNLOAD_URL = os.getenv("EFECT_DOWNLOAD_URL", "").strip()

# Optional: force a model regardless of what clients send
# Example: EFECT_FORCE_MODEL="gpt-4.1-mini"
FORCE_MODEL = os.getenv("EFECT_FORCE_MODEL", "").strip()

# Simple in-memory usage stats
START_TS = time.time()
REQ_COUNT_TOTAL = 0
REQ_COUNT_BY_TOKEN: Dict[str, int] = {}
LAST_REQ_TS_BY_TOKEN: Dict[str, float] = {}

app = FastAPI(title="EFECT Proxy")


# -------------------------
# Helpers
# -------------------------
def require_token(req: Request) -> str:
    """
    Token is required in:
      Authorization: Bearer <token>
    Must be present in EFECT_PROXY_TOKENS env (unless no tokens are set, then open).
    """
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing proxy token")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing proxy token")

    # If TOKENS is empty, treat as open (not recommended). Otherwise enforce.
    if TOKENS and token not in TOKENS:
        raise HTTPException(status_code=403, detail="Invalid proxy token")
    return token


def bump_usage(token: str):
    global REQ_COUNT_TOTAL
    REQ_COUNT_TOTAL += 1
    REQ_COUNT_BY_TOKEN[token] = REQ_COUNT_BY_TOKEN.get(token, 0) + 1
    LAST_REQ_TS_BY_TOKEN[token] = time.time()


def extract_output_text(openai_json: Dict[str, Any]) -> str:
    """
    Responses API returns output as a list; we aggregate output_text blocks.
    """
    out = []
    for item in openai_json.get("output", []) or []:
        # item: {"type":"message","role":"assistant","content":[...]}
        if item.get("type") != "message":
            continue
        content = item.get("content", []) or []
        for c in content:
            if c.get("type") == "output_text" and "text" in c:
                out.append(c["text"])
    return "".join(out).strip()


def safe_json(req: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


def themed_home_html(status_text: str) -> str:
    # Pure Python string (NO JS template literals) to avoid SyntaxError
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>EFECT AI</title>
  <style>
    :root {{
      --bg: #0b0b0b;
      --panel: #101010;
      --text: #e8e8e8;
      --muted: #aaaaaa;
      --green: #1bff4a;
      --border: #2a2a2a;
    }}
    body {{
      margin: 0;
      background: radial-gradient(1200px 700px at 50% 40%, rgba(27,255,74,.08), transparent 60%),
                  var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }}
    .box {{
      width: min(720px, 92vw);
      background: rgba(16,16,16,.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 42px 42px;
      box-shadow: 0 0 60px rgba(0,0,0,.45);
      text-align: center;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 56px;
      letter-spacing: 1px;
      color: var(--green);
      text-shadow: 0 0 18px rgba(27,255,74,.35);
    }}
    .sub {{
      color: var(--muted);
      font-size: 18px;
      margin-top: 10px;
      line-height: 1.5;
    }}
    .row {{
      margin-top: 26px;
      display: flex;
      gap: 12px;
      justify-content: center;
      flex-wrap: wrap;
    }}
    a.btn {{
      display: inline-block;
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
      color: var(--text);
      text-decoration: none;
      background: #0f0f0f;
    }}
    a.btn:hover {{
      border-color: rgba(27,255,74,.7);
      box-shadow: 0 0 18px rgba(27,255,74,.15);
    }}
    .status {{
      margin-top: 18px;
      font-weight: 800;
      color: var(--green);
    }}
    .hint {{
      margin-top: 18px;
      color: #bfbfbf;
      font-size: 14px;
      opacity: .9;
    }}
    code {{
      background: #0c0c0c;
      border: 1px solid #222;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="box">
    <h1>EFECT AI</h1>
    <div class="sub">
      Secure Online AI Proxy<br/>
      Desktop App • File Uploads • Screenshots
    </div>
    <div class="row">
      <a class="btn" href="/stats">Status & Usage</a>
      <a class="btn" href="/download">Download Desktop App</a>
      <a class="btn" href="/health">Health</a>
    </div>
    <div class="status">Status: {status_text}</div>
    <div class="hint">
      API route: <code>POST /responses</code> • Test: <code>/health</code>
    </div>
  </div>
</body>
</html>"""


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return themed_home_html("ONLINE")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/download")
def download():
    """
    Redirect to your GitHub Release EXE asset link.
    Set EFECT_DOWNLOAD_URL in Render env to enable.
    """
    if not DOWNLOAD_URL:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Download URL not set. Set EFECT_DOWNLOAD_URL on Render.",
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
    Send header: X-Admin-Secret: <EFECT_ADMIN_SECRET>
    It returns a new token. You must manually add it to EFECT_PROXY_TOKENS on Render to persist.
    """
    if not ADMIN_SECRET:
        raise HTTPException(status_code=400, detail="EFECT_ADMIN_SECRET not set on server")
    admin = req.headers.get("x-admin-secret", "").strip()
    if admin != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    new_token = "efect-" + secrets.token_urlsafe(18)
    # Add in-memory so it works immediately until restart
    TOKENS.add(new_token)
    return {
        "token": new_token,
        "note": "Token added in-memory. For permanent access, add it to EFECT_PROXY_TOKENS in Render env.",
    }


@app.post("/responses")
async def responses(req: Request):
    """
    Desktop/web calls this. We forward to OpenAI /v1/responses.
    IMPORTANT: client must send correct message formatting:
      role user/system => content type input_text
      role assistant => content type output_text
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    token = require_token(req)
    bump_usage(token)

    body = await req.json()
    body = safe_json(req, body)

    # Enforce non-stream to keep clients simple and prevent duplicate UI bugs
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
        # Pass through OpenAI error cleanly
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            return JSONResponse(status_code=r.status_code, content={"error": {"message": r.text}})

    data = r.json()
    out_text = extract_output_text(data)
    return {
        "output_text": out_text,
        "raw": data,  # keep for debugging; you can remove later
    }


@app.post("/upload")
async def upload(req: Request, file: UploadFile = File(...)):
    """
    Optional: if you later want file uploads.
    Right now it just accepts the file and returns basic metadata.
    You can expand this to OpenAI Files API later.
    """
    token = require_token(req)
    bump_usage(token)

    contents = await file.read()
    size = len(contents)
    return {"ok": True, "filename": file.filename, "bytes": size}
