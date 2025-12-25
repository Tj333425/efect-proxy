# efect_proxy.py
# EFECT Proxy + Website (Chat UI, Token Generator, Download, Status/Stats)
#
# ENV VARS (Render -> Environment):
#   OPENAI_API_KEY            = your real OpenAI key (server only)
#   EFECT_TOKEN_SECRET        = long random string (used to sign tokens)
#   EFECT_INVITE_CODES        = comma-separated invite codes users can use to mint tokens (ex: "EFECT-ALPHA,EFECT-BETA")
#   EFECT_DOWNLOAD_URL        = (optional) URL to your EXE download (GitHub Release asset recommended)
#   EFECT_ALLOWED_ORIGINS     = (optional) CORS origins, default "*"
#   EFECT_DEFAULT_MODEL       = (optional) default model, default "gpt-4.1-mini"

import os
import json
import time
import hmac
import hashlib
import base64
import secrets
import sqlite3
import uuid
from typing import Optional, Dict, Any, Tuple, List

import requests
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TOKEN_SECRET = os.getenv("EFECT_TOKEN_SECRET", "").strip()
INVITE_CODES = {c.strip() for c in os.getenv("EFECT_INVITE_CODES", "").split(",") if c.strip()}
ALLOWED_ORIGINS = os.getenv("EFECT_ALLOWED_ORIGINS", "*").strip() or "*"
DOWNLOAD_URL = os.getenv("EFECT_DOWNLOAD_URL", "").strip()
DEFAULT_MODEL = os.getenv("EFECT_DEFAULT_MODEL", "gpt-4.1-mini").strip()

# Memory/RAG
DB_PATH = os.getenv("EFECT_DB_PATH", "efect_ai.db").strip() or "efect_ai.db"
EMBEDDING_MODEL = os.getenv("EFECT_EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"
SUMMARY_MODEL = os.getenv("EFECT_SUMMARY_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

# Legacy shared tokens (optional). You can keep using this too.
LEGACY_TOKENS = {t.strip() for t in os.getenv("EFECT_PROXY_TOKENS", "").split(",") if t.strip()}

START_TS = time.time()

# Simple in-memory usage stats (resets on redeploy)
STATS = {
    "requests_total": 0,
    "chat_total": 0,
    "responses_total": 0,
    "upload_total": 0,
    "by_token": {},     # token_fingerprint -> count
    "recent": [],       # list of (ts, path, status)
}

# ---------- Persistent Memory + Knowledge (SQLite) ----------

def _db() -> sqlite3.Connection:
    # check_same_thread=False because FastAPI can use threads
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                owner_fp   TEXT NOT NULL,
                created_ts INTEGER NOT NULL,
                updated_ts INTEGER NOT NULL,
                title      TEXT,
                summary    TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                ts         INTEGER NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                owner_fp TEXT NOT NULL,
                k        TEXT NOT NULL,
                v        TEXT NOT NULL,
                updated_ts INTEGER NOT NULL,
                PRIMARY KEY(owner_fp, k)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id         TEXT PRIMARY KEY,
                owner_fp   TEXT NOT NULL,
                title      TEXT,
                content    TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_ts INTEGER NOT NULL,
                updated_ts INTEGER NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _get_or_create_session(owner_fp: str, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
    sid = session_id or str(uuid.uuid4())
    now = int(time.time())
    conn = _db()
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT session_id FROM sessions WHERE session_id=? AND owner_fp=?", (sid, owner_fp)).fetchone()
        if row:
            cur.execute("UPDATE sessions SET updated_ts=? WHERE session_id=?", (now, sid))
            conn.commit()
            return sid
        cur.execute(
            "INSERT INTO sessions(session_id, owner_fp, created_ts, updated_ts, title, summary) VALUES(?,?,?,?,?,?)",
            (sid, owner_fp, now, now, (title or None), None),
        )
        conn.commit()
        return sid
    finally:
        conn.close()


def _add_message(owner_fp: str, session_id: str, role: str, content: str) -> None:
    now = int(time.time())
    conn = _db()
    try:
        cur = conn.cursor()
        # Ensure session belongs to owner
        row = cur.execute("SELECT session_id FROM sessions WHERE session_id=? AND owner_fp=?", (session_id, owner_fp)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown session")
        cur.execute("INSERT INTO messages(session_id, role, content, ts) VALUES(?,?,?,?)", (session_id, role, content, now))
        cur.execute("UPDATE sessions SET updated_ts=? WHERE session_id=?", (now, session_id))
        conn.commit()
    finally:
        conn.close()


def _get_recent_messages(owner_fp: str, session_id: str, limit: int = 18) -> List[Dict[str, str]]:
    conn = _db()
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT session_id FROM sessions WHERE session_id=? AND owner_fp=?", (session_id, owner_fp)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown session")
        rows = cur.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, max(1, min(100, int(limit)))),
        ).fetchall()
        items = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        return items
    finally:
        conn.close()


def _get_session_summary(owner_fp: str, session_id: str) -> str:
    conn = _db()
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT summary FROM sessions WHERE session_id=? AND owner_fp=?", (session_id, owner_fp)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown session")
        return (row["summary"] or "").strip()
    finally:
        conn.close()


def _set_session_summary(owner_fp: str, session_id: str, summary: str) -> None:
    now = int(time.time())
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE sessions SET summary=?, updated_ts=? WHERE session_id=? AND owner_fp=?",
            (summary.strip(), now, session_id, owner_fp),
        )
        conn.commit()
    finally:
        conn.close()


def _profile_get(owner_fp: str) -> Dict[str, str]:
    conn = _db()
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT k, v FROM user_profile WHERE owner_fp=?", (owner_fp,)).fetchall()
        return {r["k"]: r["v"] for r in rows}
    finally:
        conn.close()


def _profile_set(owner_fp: str, k: str, v: str) -> None:
    now = int(time.time())
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_profile(owner_fp,k,v,updated_ts) VALUES(?,?,?,?) "
            "ON CONFLICT(owner_fp,k) DO UPDATE SET v=excluded.v, updated_ts=excluded.updated_ts",
            (owner_fp, k[:64], v[:2048], now),
        )
        conn.commit()
    finally:
        conn.close()


def _cosine(a: List[float], b: List[float]) -> float:
    # small, dependency-free cosine similarity
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _embed_text(text: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": text[:8000]}
    r = requests.post(OPENAI_EMBEDDINGS_URL, headers=headers, json=payload, timeout=120)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    j = r.json()
    data = j.get("data") or []
    if not data:
        return []
    emb = data[0].get("embedding")
    return emb if isinstance(emb, list) else []


def _knowledge_upsert(owner_fp: str, title: str, content: str, kid: Optional[str] = None) -> str:
    now = int(time.time())
    doc_id = kid or str(uuid.uuid4())
    emb = _embed_text(f"{title}\n\n{content}")
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO knowledge(id, owner_fp, title, content, embedding_json, created_ts, updated_ts) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, content=excluded.content, embedding_json=excluded.embedding_json, updated_ts=excluded.updated_ts",
            (doc_id, owner_fp, (title or None), content, json.dumps(emb), now, now),
        )
        conn.commit()
        return doc_id
    finally:
        conn.close()


def _knowledge_search(owner_fp: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    qemb = _embed_text(query)
    conn = _db()
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT id, title, content, embedding_json FROM knowledge WHERE owner_fp=?", (owner_fp,)).fetchall()
        scored = []
        for r in rows:
            try:
                emb = json.loads(r["embedding_json"])
            except Exception:
                emb = []
            score = _cosine(qemb, emb)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, r in scored[: max(0, min(12, int(top_k)))] :
            out.append({
                "id": r["id"],
                "title": r["title"] or "Untitled",
                "score": float(score),
                # keep snippets short
                "content": (r["content"] or "")[:2000],
            })
        return out
    finally:
        conn.close()


# ---------- Local DB (Memory + Knowledge) ----------

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
              session_id TEXT PRIMARY KEY,
              owner_fp TEXT NOT NULL,
              title TEXT,
              summary TEXT,
              created_ts INTEGER NOT NULL,
              updated_ts INTEGER NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_ts INTEGER NOT NULL,
              FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
              owner_fp TEXT NOT NULL,
              k TEXT NOT NULL,
              v TEXT NOT NULL,
              updated_ts INTEGER NOT NULL,
              PRIMARY KEY(owner_fp, k)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
              id TEXT PRIMARY KEY,
              owner_fp TEXT NOT NULL,
              title TEXT,
              content TEXT NOT NULL,
              embedding_json TEXT NOT NULL,
              created_ts INTEGER NOT NULL,
              updated_ts INTEGER NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_owner ON knowledge(owner_fp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)")
        con.commit()
    finally:
        con.close()


def get_or_create_session(owner_fp: str, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
    sid = (session_id or "").strip() or uuid.uuid4().hex
    now = int(time.time())
    con = _db()
    try:
        cur = con.cursor()
        row = cur.execute("SELECT session_id FROM chat_sessions WHERE session_id=? AND owner_fp=?", (sid, owner_fp)).fetchone()
        if row:
            cur.execute("UPDATE chat_sessions SET updated_ts=? WHERE session_id=?", (now, sid))
            con.commit()
            return sid
        cur.execute(
            "INSERT INTO chat_sessions(session_id, owner_fp, title, summary, created_ts, updated_ts) VALUES (?,?,?,?,?,?)",
            (sid, owner_fp, (title or "")[:80], "", now, now),
        )
        con.commit()
        return sid
    finally:
        con.close()


def add_message(owner_fp: str, session_id: str, role: str, content: str) -> None:
    now = int(time.time())
    con = _db()
    try:
        cur = con.cursor()
        # verify ownership
        row = cur.execute("SELECT session_id FROM chat_sessions WHERE session_id=? AND owner_fp=?", (session_id, owner_fp)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown session")
        cur.execute(
            "INSERT INTO chat_messages(session_id, role, content, created_ts) VALUES (?,?,?,?)",
            (session_id, role, content, now),
        )
        cur.execute("UPDATE chat_sessions SET updated_ts=? WHERE session_id=?", (now, session_id))
        con.commit()
    finally:
        con.close()


def get_recent_messages(owner_fp: str, session_id: str, limit: int = 14) -> List[Dict[str, str]]:
    con = _db()
    try:
        cur = con.cursor()
        row = cur.execute("SELECT session_id FROM chat_sessions WHERE session_id=? AND owner_fp=?", (session_id, owner_fp)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown session")
        rows = cur.execute(
            "SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, max(1, min(50, int(limit)))),
        ).fetchall()
        msgs = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        return msgs
    finally:
        con.close()


def get_session_summary(owner_fp: str, session_id: str) -> str:
    con = _db()
    try:
        cur = con.cursor()
        row = cur.execute(
            "SELECT summary FROM chat_sessions WHERE session_id=? AND owner_fp=?",
            (session_id, owner_fp),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown session")
        return (row["summary"] or "").strip()
    finally:
        con.close()


def set_session_summary(owner_fp: str, session_id: str, summary: str) -> None:
    con = _db()
    try:
        cur = con.cursor()
        row = cur.execute("SELECT session_id FROM chat_sessions WHERE session_id=? AND owner_fp=?", (session_id, owner_fp)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Unknown session")
        cur.execute("UPDATE chat_sessions SET summary=?, updated_ts=? WHERE session_id=?", (summary[:5000], int(time.time()), session_id))
        con.commit()
    finally:
        con.close()


def get_user_profile_kv(owner_fp: str) -> Dict[str, str]:
    con = _db()
    try:
        cur = con.cursor()
        rows = cur.execute("SELECT k, v FROM user_profile WHERE owner_fp=?", (owner_fp,)).fetchall()
        return {r["k"]: r["v"] for r in rows}
    finally:
        con.close()


def upsert_user_profile(owner_fp: str, k: str, v: str) -> None:
    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO user_profile(owner_fp, k, v, updated_ts) VALUES (?,?,?,?) "
            "ON CONFLICT(owner_fp, k) DO UPDATE SET v=excluded.v, updated_ts=excluded.updated_ts",
            (owner_fp, k[:64], v[:500], int(time.time())),
        )
        con.commit()
    finally:
        con.close()


def _cosine(a: List[float], b: List[float]) -> float:
    # no numpy dependency; small vectors are fine.
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def embed_text(text: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": text[:8000]}
    r = requests.post(OPENAI_EMBEDDINGS_URL, headers=headers, json=payload, timeout=120)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    j = r.json()
    # OpenAI embeddings response: data[0].embedding
    emb = j.get("data", [{}])[0].get("embedding")
    if not isinstance(emb, list):
        raise HTTPException(status_code=500, detail="Embedding response missing embedding")
    return [float(x) for x in emb]


def knowledge_upsert(owner_fp: str, title: str, content: str, doc_id: Optional[str] = None) -> str:
    kid = (doc_id or "").strip() or uuid.uuid4().hex
    now = int(time.time())
    emb = embed_text(f"{title}\n\n{content}")
    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO knowledge(id, owner_fp, title, content, embedding_json, created_ts, updated_ts) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, content=excluded.content, embedding_json=excluded.embedding_json, updated_ts=excluded.updated_ts",
            (kid, owner_fp, (title or "")[:120], content[:200000], json.dumps(emb), now, now),
        )
        con.commit()
        return kid
    finally:
        con.close()


def knowledge_search(owner_fp: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
    qemb = embed_text(query)
    con = _db()
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT id, title, content, embedding_json FROM knowledge WHERE owner_fp=?",
            (owner_fp,),
        ).fetchall()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in rows:
            try:
                emb = json.loads(r["embedding_json"]) if r["embedding_json"] else []
                score = _cosine(qemb, emb)
            except Exception:
                score = -1.0
            scored.append((score, {"id": r["id"], "title": r["title"], "content": r["content"], "score": score}))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [s[1] for s in scored[: max(1, min(10, int(k)))] if s[0] > 0.15]
        return out
    finally:
        con.close()

def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + pad)

def _hmac_sha256(secret: str, msg: str) -> str:
    return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()

def _token_fingerprint(tok: str) -> str:
    # don’t store raw tokens in stats
    return hashlib.sha256(tok.encode("utf-8")).hexdigest()[:12]

def mint_token(display_name: str, days: int = 30) -> str:
    if not TOKEN_SECRET:
        raise HTTPException(status_code=500, detail="Server missing EFECT_TOKEN_SECRET")
    now = int(time.time())
    payload = {
        "v": 1,
        "sub": (display_name or "user")[:64],
        "iat": now,
        "exp": now + days * 86400,
        "nonce": secrets.token_urlsafe(10),
    }
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)
    sig = _hmac_sha256(TOKEN_SECRET, payload_b64)
    # Token format: efect.<payload_b64>.<sig>
    return f"efect.{payload_b64}.{sig}"

def verify_token(tok: str) -> bool:
    if not tok:
        return False

    # Accept legacy tokens if you still use them
    if tok in LEGACY_TOKENS:
        return True

    # Verify signed tokens
    if not tok.startswith("efect."):
        return False

    parts = tok.split(".")
    if len(parts) != 3:
        return False

    _, payload_b64, sig = parts
    if not TOKEN_SECRET:
        return False

    expected = _hmac_sha256(TOKEN_SECRET, payload_b64)
    if not hmac.compare_digest(expected, sig):
        return False

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return False

    exp = int(payload.get("exp", 0))
    return int(time.time()) < exp

def get_bearer_token(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return ""

def require_token(req: Request) -> str:
    tok = get_bearer_token(req)
    if not tok:
        raise HTTPException(status_code=401, detail="Missing proxy token (Authorization: Bearer ...)")
    if not verify_token(tok):
        raise HTTPException(status_code=403, detail="Invalid proxy token")
    return tok

def extract_output_text(resp_json: Dict[str, Any]) -> str:
    """
    Responses API shape varies; try best-effort to extract output_text.
    """
    # Common: resp_json["output"][...]["content"][...{"type":"output_text","text":...}]
    out = resp_json.get("output", [])
    if isinstance(out, list):
        for item in out:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                        t = c.get("text")
                        if isinstance(t, str) and t.strip():
                            return t
    # Fallbacks
    txt = resp_json.get("text")
    if isinstance(txt, str):
        return txt
    return json.dumps(resp_json, ensure_ascii=False)


def build_core_instructions() -> str:
    # Centralized system prompt for consistent quality
    return (
        "You are EFECT AI, a practical technical assistant for the EFECT brand and its community.\n"
        "Operating principles:\n"
        "- Be accurate and explicit; do not guess.\n"
        "- If the user is missing details, ask targeted questions, but still provide the best safe default path.\n"
        "- Provide step-by-step instructions and copy/paste-ready code when applicable.\n"
        "- When debugging, identify the most likely root cause, then propose a minimal fix and a robust fix.\n"
        "Safety constraints:\n"
        "- No cheats, exploits, account theft, bypassing payments, malware, or deceptive instructions.\n"
        "- If the user asks for disallowed content, refuse and offer a safe alternative.\n"
    )


def _format_profile(profile: Dict[str, str]) -> str:
    if not profile:
        return ""
    # Keep it small and structured
    lines = []
    for k in sorted(profile.keys()):
        lines.append(f"- {k}: {profile[k]}")
    return "User profile (persisted):\n" + "\n".join(lines)


def _format_knowledge_hits(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    blocks = []
    for h in hits:
        title = h.get("title") or "Untitled"
        cid = h.get("id")
        snippet = (h.get("content") or "").strip()
        blocks.append(f"[Doc {cid}] {title}\n{snippet}")
    return "Reference docs (use when relevant; cite Doc IDs when you rely on them):\n\n" + "\n\n".join(blocks)


def _maybe_update_summary(owner_fp: str, session_id: str) -> None:
    """Summarize occasionally to keep context compact."""
    # Summarize every 10 user messages (approx) to avoid latency.
    conn = _db()
    try:
        cur = conn.cursor()
        n = cur.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE session_id=? AND role='user'",
            (session_id,),
        ).fetchone()["c"]
    finally:
        conn.close()
    if not n or (n % 10) != 0:
        return

    recent = _get_recent_messages(owner_fp, session_id, limit=30)
    prior = _get_session_summary(owner_fp, session_id)

    prompt = (
        "You maintain a running memory summary of a chat session.\n"
        "Update the existing summary using the new recent messages.\n"
        "Rules: be concise; keep stable facts, preferences, ongoing tasks; omit small talk.\n"
        "Return ONLY the updated summary text.\n"
    )

    payload = {
        "model": SUMMARY_MODEL,
        "instructions": prompt,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": f"Existing summary:\n{prior}\n\nRecent messages:\n{json.dumps(recent, ensure_ascii=False)}"}]}
        ],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=120)
    if r.status_code >= 400:
        return
    summary = extract_output_text(r.json()).strip()
    if summary:
        _set_session_summary(owner_fp, session_id, summary)

app = FastAPI(title="EFECT Proxy")

# Ensure DB is ready (memory + knowledge)
try:
    _init_db()
except Exception:
    # Do not crash the server if storage cannot initialize; endpoints will raise useful errors.
    pass

# CORS (helps if desktop app or other domains call this)
origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def stats_mw(request: Request, call_next):
    t0 = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        STATS["requests_total"] += 1
        ts = int(time.time())
        path = request.url.path
        STATS["recent"].append((ts, path, status))
        if len(STATS["recent"]) > 80:
            STATS["recent"] = STATS["recent"][-80:]
        _ = time.time() - t0

# ---------- Website Pages ----------

def page_shell(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <meta name="description" content="EFECT AI — Secure Online AI Proxy with chat, uploads, and desktop app." />
  <style>
    :root {{
      --bg: #0b0b0b;
      --panel: #101010;
      --border: #2a2a2a;
      --text: #eaeaea;
      --muted: #a0a0a0;
      --green: #1bff4a;
      --green2: #6bff6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: radial-gradient(1200px 700px at 60% 35%, rgba(27,255,74,0.08), transparent 60%),
                  radial-gradient(900px 500px at 30% 70%, rgba(27,255,74,0.05), transparent 60%),
                  var(--bg);
      color: var(--text);
    }}
    .nav {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 22px;
      border-bottom: 1px solid var(--border);
      background: rgba(8,8,8,0.65);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    .brand {{
      display:flex; align-items:center; gap:12px;
    }}
    .logo {{
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 18px rgba(27,255,74,0.65);
    }}
    .title {{
      font-weight: 800; letter-spacing: 1px;
      color: var(--green);
      text-shadow: 0 0 12px rgba(27,255,74,0.25);
    }}
    .links a {{
      color: var(--muted);
      text-decoration: none;
      margin-left: 14px;
      font-size: 14px;
    }}
    .links a:hover {{ color: var(--green2); }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 22px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: rgba(16,16,16,0.9);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 0 0 1px rgba(27,255,74,0.05), 0 14px 36px rgba(0,0,0,0.45);
    }}
    .btn {{
      display:inline-flex; align-items:center; justify-content:center;
      border: 1px solid rgba(27,255,74,0.45);
      background: rgba(27,255,74,0.12);
      color: var(--green);
      padding: 10px 12px;
      border-radius: 10px;
      font-weight: 700;
      cursor: pointer;
      text-decoration:none;
      gap: 8px;
    }}
    .btn:hover {{ background: rgba(27,255,74,0.18); }}
    .btn2 {{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--text);
    }}
    .btn2:hover {{ background: rgba(255,255,255,0.06); }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display:inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid rgba(27,255,74,0.35);
      color: var(--green2);
      background: rgba(27,255,74,0.06);
      margin-left: 10px;
    }}
    .chatlog {{
      height: 420px;
      overflow: auto;
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 14px;
      background: rgba(0,0,0,0.25);
    }}
    .msg {{
      margin: 0 0 12px 0;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.07);
      background: rgba(255,255,255,0.03);
    }}
    .msg.you {{ border-color: rgba(27,255,74,0.20); }}
    .msg .who {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .row {{
      display:flex; gap:10px; margin-top:12px;
    }}
    input[type="text"], textarea {{
      width: 100%;
      background: rgba(0,0,0,0.25);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      outline: none;
    }}
    textarea {{ min-height: 92px; resize: vertical; }}
    .small {{ font-size: 13px; }}
    .kpi {{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 10px;
    }}
    .kpi .box {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: rgba(0,0,0,0.18);
    }}
    .kpi .val {{ font-size: 20px; font-weight: 800; color: var(--green2); }}
    .kpi .lab {{ font-size: 12px; color: var(--muted); }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 12px;
    }}
    .note {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="nav">
    <div class="brand">
      <div class="logo"></div>
      <div class="title">EFECT AI</div>
      <div class="pill">ONLINE</div>
    </div>
    <div class="links">
      <a href="/">Home</a>
      <a href="/chat">Chat</a>
      <a href="/token">Token</a>
      <a href="/status">Status</a>
      <a href="/download">Download</a>
    </div>
  </div>
  <div class="wrap">
    {body_html}
    <div class="footer">
      EFECT AI Proxy • Secure routing • No cheats/spoofers/bypasses.
    </div>
  </div>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    body = f"""
    <div class="grid">
      <div class="card">
        <h2 style="margin:0 0 10px 0; color: var(--green); letter-spacing:1px;">EFECT AI</h2>
        <div class="muted small">Secure Online AI Proxy • Desktop App • Uploads • Screenshots</div>
        <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
          <a class="btn" href="/chat">Open Web Chat</a>
          <a class="btn btn2" href="/token">Generate Token</a>
          <a class="btn btn2" href="/status">View Status</a>
          <a class="btn btn2" href="/download">Download Desktop App</a>
        </div>
        <div style="margin-top:14px;" class="note">
          Tip: Use <b>Generate Token</b> to get an access token, then paste it into the desktop app or the web chat.
        </div>
      </div>
      <div class="card">
        <h3 style="margin:0 0 10px 0;">Quick Status</h3>
        <div class="kpi">
          <div class="box">
            <div class="val">{STATS["requests_total"]}</div>
            <div class="lab">Requests (since deploy)</div>
          </div>
          <div class="box">
            <div class="val">{int(time.time()-START_TS)}s</div>
            <div class="lab">Uptime</div>
          </div>
        </div>
        <div style="margin-top:12px;" class="note">
          If the site feels “slow” on free tier, the instance may have slept. First request can take ~30–60s to wake.
        </div>
      </div>
    </div>
    """
    return page_shell("EFECT AI", body)

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    body = """
    <div class="grid">
      <div class="card">
        <h3 style="margin:0 0 10px 0;">Web Chat</h3>
        <div class="muted small">Paste your token once. It saves in your browser.</div>
        <div style="margin-top:12px;">
          <div class="row">
            <input id="tok" type="text" placeholder="EFECT Token (Bearer)" />
            <button class="btn btn2" onclick="saveToken()">Save</button>
          </div>
          <div class="row">
            <input id="model" type="text" placeholder="Model (default: gpt-4.1-mini)" />
            <button class="btn btn2" onclick="saveModel()">Save</button>
          </div>
        </div>

        <div style="margin-top:12px;" class="chatlog" id="chatlog"></div>

        <div style="margin-top:12px;">
          <textarea id="msg" placeholder="Type your message..."></textarea>
          <div class="row" style="align-items:center;">
            <input id="file" type="file" />
            <button class="btn" onclick="send()">Send</button>
            <button class="btn btn2" onclick="clearChat()">Clear</button>
            <button class="btn btn2" onclick="newChat()">New Chat</button>
          </div>
          <div class="note" style="margin-top:10px;">
            Images are sent inline. PDFs/other files upload first, then attach via file_id.
          </div>
        </div>
      </div>

      <div class="card">
        <h3 style="margin:0 0 10px 0;">Links</h3>
        <div style="display:flex; gap:10px; flex-wrap:wrap;">
          <a class="btn btn2" href="/token">Generate Token</a>
          <a class="btn btn2" href="/status">Status</a>
          <a class="btn btn2" href="/download">Download App</a>
        </div>
        <div style="margin-top:12px;" class="note">
          This web chat talks to <code>/api/chat_v2</code> on this server (same domain) with memory + knowledge.
        </div>
      </div>
    </div>

    <script>
      const log = document.getElementById("chatlog");
      const tokEl = document.getElementById("tok");
      const modelEl = document.getElementById("model");
      const msgEl = document.getElementById("msg");
      const fileEl = document.getElementById("file");

      function append(who, text, you=false) {
        const div = document.createElement("div");
        div.className = "msg" + (you ? " you" : "");
        div.innerHTML = `<div class="who">${who}</div><div class="txt"></div>`;
        div.querySelector(".txt").innerText = text;
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
      }

      function saveToken() {
        localStorage.setItem("efect_token", tokEl.value.trim());
        append("SYSTEM", "Token saved in browser.", false);
      }
      function saveModel() {
        localStorage.setItem("efect_model", modelEl.value.trim());
        append("SYSTEM", "Model saved in browser.", false);
      }
      function loadSaved() {
        tokEl.value = localStorage.getItem("efect_token") || "";
        modelEl.value = localStorage.getItem("efect_model") || "";
      }

      function getSessionId() {
        let sid = localStorage.getItem("efect_session_id") || "";
        if (!sid) {
          // simple uuid v4 generator
          sid = ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
          );
          localStorage.setItem("efect_session_id", sid);
        }
        return sid;
      }

      function newChat() {
        localStorage.removeItem("efect_session_id");
        clearChat();
        append("SYSTEM", "New chat session started.", false);
      }
      function clearChat() {
        log.innerHTML = "";
      }

      async function uploadIfNeeded(file, token) {
        if (!file) return { kind: "none" };

        const name = file.name.toLowerCase();
        if (name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".webp") || name.endsWith(".bmp")) {
          // inline image
          const dataUrl = await new Promise((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => resolve(r.result);
            r.onerror = reject;
            r.readAsDataURL(file);
          });
          return { kind: "image", data_url: dataUrl, filename: file.name };
        }

        // upload file to server -> OpenAI Files -> returns file_id
        const fd = new FormData();
        fd.append("file", file);

        const resp = await fetch("/upload", {
          method: "POST",
          headers: { "Authorization": "Bearer " + token },
          body: fd
        });

        if (!resp.ok) {
          const t = await resp.text();
          throw new Error("Upload failed: " + t);
        }
        const j = await resp.json();
        return { kind: "file", file_id: j.id, filename: file.name };
      }

      async function send() {
        const token = (tokEl.value.trim() || localStorage.getItem("efect_token") || "").trim();
        if (!token) {
          append("SYSTEM", "Missing token. Go to /token to generate one.", false);
          return;
        }
        const model = (modelEl.value.trim() || localStorage.getItem("efect_model") || "").trim();

        const text = msgEl.value.trim();
        if (!text && !fileEl.files[0]) return;

        append("YOU", text || "[attachment]", true);
        msgEl.value = "";

        let attachment = { kind: "none" };
        try {
          attachment = await uploadIfNeeded(fileEl.files[0], token);
        } catch (e) {
          append("SYSTEM", String(e), false);
          return;
        } finally {
          fileEl.value = "";
        }

        const payload = {
          model: model || null,
          message: text || "",
          attachment: attachment,
          session_id: getSessionId(),
          use_memory: true,
          use_knowledge: true,
          top_k: 4
        };

        const resp = await fetch("/api/chat_v2", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token
          },
          body: JSON.stringify(payload)
        });

        if (!resp.ok) {
          const t = await resp.text();
          append("SYSTEM", "Error: " + t, false);
          return;
        }
        const j = await resp.json();
        if (j.session_id) {
          localStorage.setItem("efect_session_id", String(j.session_id));
        }
        append("EFECT AI", j.text || "(no text)", false);
      }

      loadSaved();
      append("SYSTEM", "Ready. Paste token and start chatting. Click Clear to clear UI; use New Chat to reset memory.", false);
    </script>
    """
    return page_shell("EFECT AI • Chat", body)

@app.get("/token", response_class=HTMLResponse)
def token_page():
    body = """
    <div class="card">
      <h3 style="margin:0 0 10px 0;">Token Generator</h3>
      <div class="muted small">Enter an invite code to mint your EFECT token.</div>

      <div style="margin-top:12px;" class="row">
        <input id="code" type="text" placeholder="Invite code (ex: EFECT-ALPHA)" />
        <input id="name" type="text" placeholder="Display name (optional)" />
      </div>

      <div style="margin-top:12px;" class="row">
        <button class="btn" onclick="gen()">Generate Token</button>
        <button class="btn btn2" onclick="copyTok()">Copy</button>
        <button class="btn btn2" onclick="saveTok()">Save to Browser</button>
      </div>

      <div style="margin-top:12px;">
        <textarea id="out" placeholder="Your token will appear here..." readonly></textarea>
      </div>

      <div class="note" style="margin-top:12px;">
        If you want users to generate tokens, set <code>EFECT_INVITE_CODES</code> on the server.
      </div>
    </div>

    <script>
      const out = document.getElementById("out");
      async function gen() {
        const code = document.getElementById("code").value.trim();
        const name = document.getElementById("name").value.trim();

        const resp = await fetch("/api/token", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ invite_code: code, name: name })
        });

        const text = await resp.text();
        if (!resp.ok) {
          out.value = "ERROR: " + text;
          return;
        }
        const j = JSON.parse(text);
        out.value = j.token;
      }
      function copyTok() {
        navigator.clipboard.writeText(out.value || "");
      }
      function saveTok() {
        localStorage.setItem("efect_token", out.value.trim());
        alert("Saved token to browser.");
      }
    </script>
    """
    return page_shell("EFECT AI • Token", body)

@app.get("/status", response_class=HTMLResponse)
def status_page():
    uptime = int(time.time() - START_TS)
    recent = "\n".join([f"{ts}  {path}  {status}" for (ts, path, status) in STATS["recent"][-25:]])
    body = f"""
    <div class="grid">
      <div class="card">
        <h3 style="margin:0 0 10px 0;">Service Status</h3>
        <div class="kpi">
          <div class="box"><div class="val">ONLINE</div><div class="lab">State</div></div>
          <div class="box"><div class="val">{uptime}s</div><div class="lab">Uptime</div></div>
          <div class="box"><div class="val">{STATS["requests_total"]}</div><div class="lab">Requests</div></div>
          <div class="box"><div class="val">{STATS["chat_total"]}</div><div class="lab">Web Chat Calls</div></div>
        </div>
        <div class="note" style="margin-top:12px;">
          This page is public. The API routes that cost money are protected by EFECT tokens.
        </div>
      </div>

      <div class="card">
        <h3 style="margin:0 0 10px 0;">Recent Requests</h3>
        <pre style="white-space:pre-wrap; margin:0; color: var(--muted); border:1px solid var(--border); border-radius:12px; padding:12px; background: rgba(0,0,0,0.2);">{recent}</pre>
      </div>
    </div>
    """
    return page_shell("EFECT AI • Status", body)

@app.get("/metrics")
def metrics():
    # Safe, public summary
    uptime = int(time.time() - START_TS)
    return {
        "ok": True,
        "uptime_s": uptime,
        "requests_total": STATS["requests_total"],
        "chat_total": STATS["chat_total"],
        "responses_total": STATS["responses_total"],
        "upload_total": STATS["upload_total"],
    }

@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nAllow: /\n"

@app.get("/sitemap.xml", response_class=PlainTextResponse)
def sitemap():
    # simple sitemap to help “searchable link” indexing
    base = ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{base}/</loc></url>
  <url><loc>{base}/chat</loc></url>
  <url><loc>{base}/token</loc></url>
  <url><loc>{base}/status</loc></url>
</urlset>
"""

@app.get("/download")
def download():
    # Best practice: set EFECT_DOWNLOAD_URL to a GitHub Release asset.
    if DOWNLOAD_URL:
        return RedirectResponse(DOWNLOAD_URL, status_code=302)

    # Optional: if you commit an EXE under ./static/EFECT_AI.exe, serve it
    static_path = os.path.join(os.path.dirname(__file__), "static", "EFECT_AI.exe")
    if os.path.exists(static_path):
        return FileResponse(static_path, filename="EFECT_AI.exe")

    # If neither exists:
    return JSONResponse(
        status_code=404,
        content={"detail": "No download configured. Set EFECT_DOWNLOAD_URL or add static/EFECT_AI.exe"},
    )

# ---------- API Routes ----------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/token")
async def api_token(payload: Dict[str, Any]):
    invite = (payload.get("invite_code") or "").strip()
    name = (payload.get("name") or "").strip()

    if not INVITE_CODES:
        raise HTTPException(status_code=500, detail="Server missing EFECT_INVITE_CODES")
    if invite not in INVITE_CODES:
        raise HTTPException(status_code=403, detail="Invalid invite code")

    tok = mint_token(name or "user", days=30)
    return {"token": tok, "expires_days": 30}

@app.post("/upload")
async def upload(req: Request, file: UploadFile = File(...)):
    tok = require_token(req)
    STATS["upload_total"] += 1
    STATS["by_token"][_token_fingerprint(tok)] = STATS["by_token"].get(_token_fingerprint(tok), 0) + 1

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    # Upload to OpenAI Files API
    # Note: purpose sometimes varies; "assistants" is commonly accepted.
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    files = {
        "file": (file.filename, await file.read(), file.content_type or "application/octet-stream")
    }
    data = {"purpose": "assistants"}

    r = requests.post(OPENAI_FILES_URL, headers=headers, files=files, data=data, timeout=300)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    return r.json()

@app.post("/responses")
async def responses(req: Request):
    tok = require_token(req)
    STATS["responses_total"] += 1
    STATS["by_token"][_token_fingerprint(tok)] = STATS["by_token"].get(_token_fingerprint(tok), 0) + 1

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    body = await req.json()
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    stream = bool(body.get("stream", False))
    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=body, stream=stream, timeout=300)

    if not stream:
        if r.status_code >= 400:
            return JSONResponse(status_code=r.status_code, content={"detail": r.text})
        return JSONResponse(status_code=200, content=r.json())

    def gen():
        try:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            try:
                r.close()
            except Exception:
                pass

    # pass through content-type if present
    ctype = r.headers.get("content-type", "application/octet-stream")
    return StreamingResponse(gen(), status_code=r.status_code, media_type=ctype)


# ---------- EFECT AI v2 (Memory + Knowledge) ----------

@app.get("/api/sessions")
async def list_sessions(req: Request):
    tok = require_token(req)
    owner_fp = _token_fingerprint(tok)
    conn = _db()
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT session_id, created_ts, updated_ts, title FROM sessions WHERE owner_fp=? ORDER BY updated_ts DESC LIMIT 50",
            (owner_fp,),
        ).fetchall()
        return {
            "sessions": [
                {
                    "session_id": r["session_id"],
                    "created_ts": r["created_ts"],
                    "updated_ts": r["updated_ts"],
                    "title": r["title"] or "Chat",
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


@app.post("/api/profile")
async def update_profile(req: Request):
    tok = require_token(req)
    owner_fp = _token_fingerprint(tok)
    body = await req.json()
    k = (body.get("key") or "").strip()
    v = (body.get("value") or "").strip()
    if not k or not v:
        raise HTTPException(status_code=400, detail="key and value are required")
    _profile_set(owner_fp, k, v)
    return {"ok": True}


@app.post("/api/knowledge/upsert")
async def knowledge_upsert(req: Request):
    tok = require_token(req)
    owner_fp = _token_fingerprint(tok)
    body = await req.json()
    title = (body.get("title") or "").strip()
    content = (body.get("content") or "").strip()
    kid = (body.get("id") or "").strip() or None
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    doc_id = _knowledge_upsert(owner_fp, title, content, kid=kid)
    return {"id": doc_id}


@app.post("/api/knowledge/search")
async def knowledge_search(req: Request):
    tok = require_token(req)
    owner_fp = _token_fingerprint(tok)
    body = await req.json()
    q = (body.get("query") or "").strip()
    top_k = int(body.get("top_k") or 4)
    if not q:
        raise HTTPException(status_code=400, detail="query is required")
    return {"hits": _knowledge_search(owner_fp, q, top_k=top_k)}


@app.post("/api/chat_v2")
async def api_chat_v2(req: Request):
    tok = require_token(req)
    STATS["chat_total"] += 1
    STATS["by_token"][_token_fingerprint(tok)] = STATS["by_token"].get(_token_fingerprint(tok), 0) + 1

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    owner_fp = _token_fingerprint(tok)
    body = await req.json()
    message = (body.get("message") or "").strip()
    attachment = body.get("attachment") or {"kind": "none"}
    model = (body.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    session_id = (body.get("session_id") or "").strip() or None
    title = (body.get("title") or "").strip() or None

    use_memory = bool(body.get("use_memory", True))
    use_knowledge = bool(body.get("use_knowledge", True))
    top_k = int(body.get("top_k") or 4)

    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    sid = _get_or_create_session(owner_fp, session_id=session_id, title=title)

    # Record a readable user message for persistence
    stored_user_text = message
    if isinstance(attachment, dict) and attachment.get("kind") in ("image", "file"):
        fn = attachment.get("filename") or attachment.get("kind")
        stored_user_text = (message + "\n\n" if message else "") + f"[attachment: {fn}]"
    _add_message(owner_fp, sid, "user", stored_user_text)

    # Compact memory signals
    profile = _profile_get(owner_fp) if use_memory else {}
    summary = _get_session_summary(owner_fp, sid) if use_memory else ""

    hits = _knowledge_search(owner_fp, message, top_k=top_k) if use_knowledge else []

    instructions = build_core_instructions()
    memory_block = ""
    if profile or summary:
        memory_block = (
            "\n\n---\n"
            + (f"Session summary (persisted):\n{summary}\n\n" if summary else "")
            + (_format_profile(profile) + "\n" if profile else "")
        )
    knowledge_block = "\n\n---\n" + _format_knowledge_hits(hits) if hits else ""

    full_instructions = instructions + memory_block + knowledge_block

    # Provide last turns as conversational grounding (kept small)
    # We rebuild the latest user message to include attachments (if provided).
    recent = _get_recent_messages(owner_fp, sid, limit=14)
    convo: List[Dict[str, Any]] = []
    if recent:
        recent = recent[:-1]  # drop the stored user message we just added
    for m in recent:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            convo.append({"role": role, "content": [{"type": "input_text", "text": content}]})

    # Current user turn (with optional attachment)
    parts: List[Dict[str, Any]] = []
    if message:
        parts.append({"type": "input_text", "text": message})
    if isinstance(attachment, dict):
        if attachment.get("kind") == "image" and attachment.get("data_url"):
            parts.append({"type": "input_image", "image_url": attachment["data_url"]})
        elif attachment.get("kind") == "file" and attachment.get("file_id"):
            parts.append({"type": "input_file", "file_id": attachment["file_id"]})

    convo.append({"role": "user", "content": parts if parts else [{"type": "input_text", "text": ""}]})

    payload = {
        "model": model,
        "instructions": full_instructions,
        "input": convo,
        "stream": False,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=300)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    j = r.json()
    text = extract_output_text(j).strip()
    if text:
        _add_message(owner_fp, sid, "assistant", text)
    # Opportunistic summary update
    if use_memory:
        try:
            _maybe_update_summary(owner_fp, sid)
        except Exception:
            pass

    return {
        "session_id": sid,
        "text": text,
        "knowledge_hits": [{"id": h["id"], "title": h["title"], "score": h["score"]} for h in hits],
    }

@app.post("/api/chat")
async def api_chat(req: Request):
    tok = require_token(req)
    STATS["chat_total"] += 1
    STATS["by_token"][_token_fingerprint(tok)] = STATS["by_token"].get(_token_fingerprint(tok), 0) + 1

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    body = await req.json()
    message = (body.get("message") or "").strip()
    model = (body.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    attachment = body.get("attachment") or {"kind": "none"}
    parts: List[Dict[str, Any]] = []
    if message:
        parts.append({"type": "input_text", "text": message})

    # Handle attachments from web chat
    if isinstance(attachment, dict):
        if attachment.get("kind") == "image" and attachment.get("data_url"):
            parts.append({"type": "input_image", "image_url": attachment["data_url"]})
        elif attachment.get("kind") == "file" and attachment.get("file_id"):
            parts.append({"type": "input_file", "file_id": attachment["file_id"]})

    # Build a safe, EFECT-branded instruction set
    instructions = (
        "You are EFECT AI, an assistant for the EFECT brand.\n"
        "Be practical and step-by-step.\n"
        "No cheats/spoofers/bypasses.\n"
        "If user attaches screenshots/files, analyze them.\n"
    )

    payload = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": parts}] if parts else (message or ""),
        "stream": False
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=300)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    j = r.json()
    text = extract_output_text(j)
    return {"text": text, "raw": j}

