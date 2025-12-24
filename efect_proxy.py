# efect_proxy.py
# EFECT Proxy (FastAPI) + EFECT landing page + download route + uploads + responses
# Deploy command (Render): uvicorn efect_proxy:app --host 0.0.0.0 --port $PORT

import os
import json
import time
from typing import Optional, Set, Dict, Any

import requests
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse, JSONResponse

OPENAI_RESPONSES_URL = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
OPENAI_FILES_URL = os.getenv("OPENAI_FILES_URL", "https://api.openai.com/v1/files")

# Server-only secret (set this in Render env vars)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Token gate (set this in Render env vars)
# Example: EFECT_PROXY_TOKENS=efect-12345,efect-99999
RAW_TOKENS = os.getenv("EFECT_PROXY_TOKENS", "").strip()
TOKENS: Set[str] = set(t.strip() for t in RAW_TOKENS.split(",") if t.strip())

# Optional: where /download redirects (recommended: GitHub Releases asset)
# Example: https://github.com/<you>/<repo>/releases/latest/download/EFECT_AI.exe
EXE_DOWNLOAD_URL = os.getenv(
    "EFECT_EXE_URL",
    "https://github.com/Tj333425/efect-proxy/releases/latest/download/EFECT_AI.exe"
)

# Very small usage stats (resets on deploy/restart)
STATS = {
    "started_at": int(time.time()),
    "total_requests": 0,
    "total_uploads": 0,
    "last_request_at": None,
    "last_upload_at": None,
    "last_error": None,
}

app = FastAPI(title="EFECT Proxy")


def _require_token(req: Request) -> Optional[str]:
    """
    If TOKENS is set (non-empty), require Authorization: Bearer <token>.
    If TOKENS is empty, allow all (dev mode) but still parse token if provided.
    """
    auth = req.headers.get("authorization", "")
    token = None
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()

    if TOKENS:
        if not token:
            raise HTTPException(status_code=401, detail="Missing proxy token")
        if token not in TOKENS:
            raise HTTPException(status_code=403, detail="Invalid proxy token")

    return token


def _openai_headers() -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


@app.get("/", response_class=HTMLResponse)
def home():
    # Simple EFECT landing page (searchable link + button)
    return """
<!DOCTYPE html>
<html>
<head>
  <title>EFECT AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {
      margin: 0;
      background: #0b0b0b;
      color: #b6ffb3;
      font-family: Segoe UI, Arial, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
    }
    .box {
      background: #101010;
      border: 1px solid #2a2a2a;
      border-radius: 18px;
      padding: 44px;
      width: min(560px, calc(100vw - 40px));
      text-align: center;
      box-shadow: 0 0 40px rgba(0,0,0,0.45);
    }
    h1 {
      font-size: 52px;
      margin: 0 0 10px 0;
      color: #7CFF7A;
      letter-spacing: 1px;
    }
    p {
      color: #d0d0d0;
      font-size: 18px;
      margin: 8px 0;
    }
    .status {
      margin-top: 14px;
      color: #7CFF7A;
      font-weight: 800;
      font-size: 16px;
    }
    .btnRow {
      margin-top: 26px;
      display: flex;
      gap: 12px;
      justify-content: center;
      flex-wrap: wrap;
    }
    .btn {
      display: inline-block;
      padding: 14px 22px;
      border-radius: 12px;
      background: #7CFF7A;
      color: #000;
      text-decoration: none;
      font-weight: 800;
      font-size: 16px;
      border: 1px solid rgba(0,0,0,0.2);
      min-width: 180px;
    }
    .btn.secondary {
      background: transparent;
      color: #b6ffb3;
      border: 1px solid #2a2a2a;
    }
    .btn:hover { opacity: 0.92; }
    .small {
      margin-top: 18px;
      color: #9a9a9a;
      font-size: 13px;
      line-height: 1.4;
    }
    code {
      color: #7CFF7A;
      background: #0d0d0d;
      padding: 2px 6px;
      border-radius: 8px;
      border: 1px solid #222;
    }
  </style>
</head>
<body>
  <div class="box">
    <h1>EFECT AI</h1>
    <p>Secure Online AI Proxy</p>
    <p>Desktop App • File Uploads • Screenshots</p>
    <div class="status">Status: ONLINE</div>

    <div class="btnRow">
      <a class="btn" href="/download">Download EFECT AI</a>
      <a class="btn secondary" href="/stats">Status / Usage</a>
    </div>

    <div class="small">
      API endpoints: <code>/responses</code> and <code>/upload</code><br/>
      Health check: <code>/health</code>
    </div>
  </div>
</body>
</html>
"""


@app.get("/download")
def download():
    # Keep /download on your domain, but serve the exe from GitHub Releases (recommended)
    return RedirectResponse(EXE_DOWNLOAD_URL)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stats")
def stats():
    # simple stats page (JSON)
    return JSONResponse(
        {
            "ok": True,
            "started_at": STATS["started_at"],
            "uptime_seconds": int(time.time()) - int(STATS["started_at"]),
            "total_requests": STATS["total_requests"],
            "total_uploads": STATS["total_uploads"],
            "last_request_at": STATS["last_request_at"],
            "last_upload_at": STATS["last_upload_at"],
            "token_gate_enabled": bool(TOKENS),
            "last_error": STATS["last_error"],
        }
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...), req: Request = None):
    """
    Upload a file to OpenAI Files API and return {"id": "<file_id>"}.
    Client can then send that id as input_file in /responses.
    """
    _require_token(req)
    STATS["total_uploads"] += 1
    STATS["last_upload_at"] = int(time.time())

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # NOTE: purpose may vary by OpenAI API behavior. "assistants" is widely used.
        files = {
            "file": (file.filename or "upload.bin", content, file.content_type or "application/octet-stream")
        }
        data = {"purpose": "assistants"}

        r = requests.post(
            OPENAI_FILES_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files,
            data=data,
            timeout=300,
        )

        if r.status_code >= 400:
            STATS["last_error"] = f"Upload failed {r.status_code}: {r.text[:500]}"
            raise HTTPException(status_code=r.status_code, detail=r.text)

        j = r.json()
        file_id = j.get("id")
        if not file_id:
            raise HTTPException(status_code=500, detail="Upload succeeded but no file id returned")

        return {"id": file_id, "filename": file.filename}

    except HTTPException:
        raise
    except Exception as e:
        STATS["last_error"] = f"Upload exception: {str(e)[:500]}"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/responses")
async def responses(req: Request):
    """
    Forward a Responses API request to OpenAI.
    Supports both normal JSON and streaming.
    Client must send Authorization: Bearer <EFECT_TOKEN> (if token gate enabled).
    """
    _require_token(req)
    STATS["total_requests"] += 1
    STATS["last_request_at"] = int(time.time())

    body: Dict[str, Any] = await req.json()
    stream = bool(body.get("stream", False))

    # Forward request unchanged (you can enforce model here if you want)
    # body["model"] = body.get("model", "gpt-4.1-mini")

    try:
        if not stream:
            r = requests.post(
                OPENAI_RESPONSES_URL,
                headers=_openai_headers(),
                data=json.dumps(body),
                timeout=300,
            )
            if r.status_code >= 400:
                STATS["last_error"] = f"Responses error {r.status_code}: {r.text[:500]}"
                return JSONResponse(status_code=r.status_code, content=r.json() if r.headers.get("content-type","").startswith("application/json") else {"error": r.text})
            return JSONResponse(content=r.json())

        # Streaming
        upstream = requests.post(
            OPENAI_RESPONSES_URL,
            headers=_openai_headers(),
            data=json.dumps(body),
            stream=True,
            timeout=300,
        )
        if upstream.status_code >= 400:
            STATS["last_error"] = f"Stream error {upstream.status_code}: {upstream.text[:500]}"
            return JSONResponse(status_code=upstream.status_code, content={"error": upstream.text})

        def gen():
            try:
                for line in upstream.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    yield (line + "\n").encode("utf-8")
            finally:
                upstream.close()

        return StreamingResponse(gen(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        STATS["last_error"] = f"Responses exception: {str(e)[:500]}"
        raise HTTPException(status_code=500, detail=str(e))
