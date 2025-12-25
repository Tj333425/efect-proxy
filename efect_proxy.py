from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (proof-of-upgrade)
SESSIONS = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.get("/")
def root():
    return {"status": "EFECT AI server running"}

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    return """
    <html>
    <body style="background:black;color:#00ff66;font-family:sans-serif">
      <h1>EFECT AI</h1>
      <p>Upgraded server active.</p>
    </body>
    </html>
    """

# This is the route you tested (it must exist)
@app.get("/api/sessions")
def list_sessions():
    return list(SESSIONS.keys())

@app.post("/api/chat_v2")
def chat_v2(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    history = SESSIONS.setdefault(sid, [])
    history.append({"role": "user", "content": req.message})

    reply = f"[EFECT AI] Session {sid[:8]} received: {req.message}"

    history.append({"role": "assistant", "content": reply})
    return {"session_id": sid, "reply": reply}
