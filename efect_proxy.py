import os
import json
import requests
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="EFECT Proxy")

OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
TOKENS = set(
    t.strip() for t in (os.getenv("EFECT_PROXY_TOKENS") or "").split(",") if t.strip()
)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

def require_token(req: Request):
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing EFECT token")
    token = auth.split(" ", 1)[1]
    if TOKENS and token not in TOKENS:
        raise HTTPException(status_code=403, detail="Invalid EFECT token")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload")
async def upload(req: Request, file: UploadFile = File(...)):
    require_token(req)

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": (file.filename, await file.read(), file.content_type)}
    data = {"purpose": "user_data"}

    r = requests.post(
        OPENAI_FILES_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=300
    )

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()

@app.post("/responses")
async def responses(req: Request):
    require_token(req)
    body = await req.json()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if body.get("stream") else "application/json",
    }

    upstream = requests.post(
        OPENAI_URL,
        headers=headers,
        data=json.dumps(body),
        stream=bool(body.get("stream")),
        timeout=300,
    )

    if upstream.status_code >= 400:
        raise HTTPException(status_code=upstream.status_code, detail=upstream.text)

    if body.get("stream"):
        def gen():
            for chunk in upstream.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    return upstream.json()
