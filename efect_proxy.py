import os
import json
import time
from typing import Optional, Any, Dict

import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

OPENAI_URL = "https://api.openai.com/v1/responses"

# Set on Render (never in your EXE)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Shared token gate for your users
# Example: EFECT_PROXY_TOKENS="token1,token2,token3"
TOKENS = {t.strip() for t in os.getenv("EFECT_PROXY_TOKENS", "").split(",") if t.strip()}

app = FastAPI(title="EFECT Proxy")


# ----------------------------
# Helpers
# ----------------------------
def require_token(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing proxy token")
    token = auth.split(" ", 1)[1].strip()
    if TOKENS and token not in TOKENS:
        raise HTTPException(status_code=403, detail="Invalid proxy token")
    return token


def normalize_input_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix common client mistakes:
    - Convert content type 'input_text' -> 'text' (Responses API expects 'text')
    - If user sends 'messages', convert to 'input'
    - Ensure there's at least one user message
    """
    p = dict(payload)

    # Allow either {input: [...]} or {messages: [...]} from clients
    if "input" not in p and "messages" in p:
        p["input"] = p.pop("messages")

    if "input" not in p or not isinstance(p["input"], list):
        # Fallback: accept a plain string
        text = p.get("text") or p.get("prompt") or ""
        p["input"] = [{"role": "user", "content": [{"type": "text", "text": str(text)}]}]

    # Normalize each message content
    for msg in p["input"]:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            for c in content:
                # Fix the exact error you hit:
                if c.get("type") == "input_text":
                    c["type"] = "text"
                # Some people send {"text": "..."} without type
                if "type" not in c and "text" in c:
                    c["type"] = "text"

    return p


def extract_output_text(resp_json: Dict[str, Any]) -> str:
    """
    Prevent repetition: use only the final merged output text.
    """
    if isinstance(resp_json, dict) and "output_text" in resp_json:
        return resp_json.get("output_text") or ""

    # Fallback: look for message content output_text
    parts = []
    for item in resp_json.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "".join(parts)


def openai_headers() -> Dict[str, str]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    # Simple EFECT branded homepage (you can expand later)
    return """
<!DOCTYPE html>
<html>
<head>
  <title>EFECT AI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { margin:0; background:#0b0b0b; color:#b6ff3b; font-family:Segoe UI, Arial, sans-serif;
           display:flex; align-items:center; justify-content:center; height:100vh; }
    .box { background:#101010; border:1px solid #2a2a2a; border-radius:18px; padding:42px 48px;
           width:min(560px, 92vw); text-align:center; box-shadow:0 0 45px rgba(182,255,59,0.08); }
    h1 { margin:0 0 10px; font-size:52px; letter-spacing:1px; text-shadow:0 0 18px rgba(182,255,59,0.25); }
    p { margin:10px 0; color:#cfcfcf; font-size:18px; }
    .status { margin-top:18px; font-weight:800; color:#b6ff3b; }
    a { color:#b6ff3b; text-decoration:none; }
  </style>
</head>
<body>
  <div class="box">
    <h1>EFECT AI</h1>
    <p>Secure Online AI Proxy</p>
    <p>Desktop App • File Uploads • Screenshots</p>
    <div class="status">Status: ONLINE</div>
  </div>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


@app.post("/responses")
async def responses(req: Request):
    """
    Main endpoint your desktop app should call.
    - Fixes 'input_text' -> 'text'
    - Prevents repeated output
    - Returns the full OpenAI JSON AND includes clean 'output_text'
    """
    require_token(req)

    body = await req.json()
    body = normalize_input_payload(body)

    # Optional: force a model if you want
    body.setdefault("model", "gpt-4.1-mini")

    r = requests.post(OPENAI_URL, headers=openai_headers(), json=body, timeout=300)

    if r.status_code >= 400:
        # Pass through OpenAI error
        try:
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception:
            raise HTTPException(status_code=r.status_code, detail=r.text)

    resp_json = r.json()

    # Ensure a single clean output_text (prevents repeats in clients)
    resp_json["output_text"] = extract_output_text(resp_json)

    return JSONResponse(content=resp_json)


@app.post("/responses/stream")
async def responses_stream(req: Request):
    """
    Streaming version (optional).
    IMPORTANT: streaming can cause repeats if your client appends each chunk incorrectly.
    This stream sends OpenAI's stream directly. Your client must render properly.
    """
    require_token(req)

    body = await req.json()
    body = normalize_input_payload(body)
    body.setdefault("model", "gpt-4.1-mini")
    body["stream"] = True

    def gen():
        with requests.post(OPENAI_URL, headers=openai_headers(), json=body, stream=True, timeout=300) as r:
            if r.status_code >= 400:
                # stream an error as plain text
                yield (r.text or "OpenAI error").encode("utf-8")
                return
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")
