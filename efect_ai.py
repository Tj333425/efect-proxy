# efect_ai.py
# EFECT AI client with:
# - Online proxy support (HTTPS)
# - Attach button + Screenshot button
# - Drag & drop files into the chat
# - Images: sent as input_image (base64 data URL)
# - PDFs: uploaded to proxy -> file_id -> sent as input_file (recommended for PDFs) :contentReference[oaicite:5]{index=5}
# - Text/log/md: inlined as input_text
#
# Install deps:
#   py -m pip install requests pillow
# Drag-drop support (recommended):
#   py -m pip install tkinterdnd2
#
# Build EXE:
#   py -m pip install pyinstaller
#   py -m PyInstaller --onefile --windowed --name "EFECT AI" --icon efect.ico efect_ai.py

import os
import json
import time
import base64
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from io import BytesIO

import requests
from PIL import ImageGrab

# Drag & drop (optional)
DND_OK = False
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_OK = True
except Exception:
    TkinterDnD = None
    DND_FILES = None

APP_NAME = "EFECT_AI"
DEFAULT_MODEL = "gpt-4.1-mini"

DEFAULT_SETTINGS = {
    "proxy_base_url": "https://YOUR-PROXY-DOMAIN",  # <-- change after deploy
    "model": DEFAULT_MODEL,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are EFECT AI, an assistant for the EFECT brand.\n"
    "Be practical and step-by-step. No cheats/spoofers/bypasses.\n"
    "If user attaches screenshots/files, analyze them.\n"
)

def appdata_dir() -> str:
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    p = os.path.join(base, APP_NAME)
    os.makedirs(p, exist_ok=True)
    return p

def p_settings() -> str:
    return os.path.join(appdata_dir(), "settings.json")

def safe_load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_save_json(path: str, obj) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def parse_sse_stream(iter_lines, stop_flag):
    """
    Streams Response API output text deltas.
    The Responses streaming docs describe SSE event streams. :contentReference[oaicite:6]{index=6}
    """
    event_type = None
    for raw_line in iter_lines:
        if stop_flag.is_set():
            break
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            event_type = None
            continue
        if line.startswith("event:"):
            event_type = line.split("event:", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_str = line.split("data:", 1)[1].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except Exception:
                continue
            typ = obj.get("type") or event_type
            # Common delta event for text output
            if typ == "response.output_text.delta":
                delta = obj.get("delta", "")
                if delta:
                    yield delta

def is_image_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def is_text_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".txt", ".log", ".md"]

def is_pdf_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".pdf"

def file_to_data_url(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    elif ext == ".bmp":
        mime = "image/bmp"
    else:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return mime, f"data:{mime};base64,{b64}"

class EfectAIApp:
    def __init__(self, root: tk.Tk):
        self.root = root

        self.settings = safe_load_json(p_settings(), DEFAULT_SETTINGS)
        if "proxy_base_url" not in self.settings:
            self.settings["proxy_base_url"] = DEFAULT_SETTINGS["proxy_base_url"]
        if "model" not in self.settings:
            self.settings["model"] = DEFAULT_SETTINGS["model"]

        self.stop_flag = threading.Event()
        self.attachments = []  # list of dicts: {kind, name, ...}

        # Theme
        self.bg = "#0B0B0F"
        self.panel = "#0F1016"
        self.fg = "#E8E8F0"
        self.muted = "#9AA0AA"
        self.green = "#1BFF4A"
        self.border = "#2A2D3A"

        self.root.title("EFECT AI")
        self.root.geometry("1100x740")
        self.root.minsize(900, 620)
        self.root.configure(bg=self.bg)

        self.token_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value=self.settings.get("model", DEFAULT_MODEL))
        self.proxy_var = tk.StringVar(value=self.settings.get("proxy_base_url", DEFAULT_SETTINGS["proxy_base_url"]))
        self.input_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.panel, highlightbackground=self.green, highlightthickness=1)
        header.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(16, 8))

        left = tk.Frame(header, bg=self.panel)
        left.pack(side=tk.LEFT, padx=14, pady=10)
        tk.Label(left, text="EFECT", fg=self.green, bg=self.panel, font=("Segoe UI", 20, "bold")).pack(side=tk.LEFT)
        tk.Label(left, text=" AI", fg=self.fg, bg=self.panel, font=("Segoe UI", 20, "bold")).pack(side=tk.LEFT)

        right = tk.Frame(header, bg=self.panel)
        right.pack(side=tk.RIGHT, padx=14, pady=10)

        tk.Label(right, text="Proxy:", fg=self.muted, bg=self.panel, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Entry(
            right, textvariable=self.proxy_var, width=34, bg=self.bg, fg=self.fg,
            insertbackground=self.green, highlightbackground=self.border, highlightthickness=1, bd=0
        ).pack(side=tk.LEFT, padx=(0, 8), ipady=6)

        tk.Label(right, text="Model:", fg=self.muted, bg=self.panel, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Entry(
            right, textvariable=self.model_var, width=14, bg=self.bg, fg=self.fg,
            insertbackground=self.green, highlightbackground=self.border, highlightthickness=1, bd=0
        ).pack(side=tk.LEFT, padx=(0, 8), ipady=6)

        tk.Label(right, text="Token:", fg=self.muted, bg=self.panel, font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Entry(
            right, textvariable=self.token_var, width=22, bg=self.bg, fg=self.fg,
            insertbackground=self.green, highlightbackground=self.border, highlightthickness=1, bd=0, show="•"
        ).pack(side=tk.LEFT, ipady=6)

        tk.Button(
            right, text="Save", command=self.save_settings,
            bg=self.panel, fg=self.green, activebackground=self.panel, activeforeground=self.green,
            bd=1, highlightthickness=1, highlightbackground=self.border,
            font=("Segoe UI", 10, "bold"), padx=10
        ).pack(side=tk.LEFT, padx=(8, 0), ipady=3)

        chat_frame = tk.Frame(self.root, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=8)

        self.chat_text = tk.Text(
            chat_frame, bg=self.panel, fg=self.fg, insertbackground=self.green,
            wrap="word", font=("Segoe UI", 12), bd=0, padx=14, pady=12
        )
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = tk.Scrollbar(chat_frame, orient="vertical", command=self.chat_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.configure(yscrollcommand=scroll.set, state="disabled")

        # Drag & drop hook (optional)
        if DND_OK:
            try:
                # register drop on the chat area
                self.chat_text.drop_target_register(DND_FILES)
                self.chat_text.dnd_bind("<<Drop>>", self._on_drop_files)
                self._append_line("[EFECT] Drag & drop enabled. Drop files into the chat to attach.\n")
            except Exception:
                self._append_line("[EFECT] Drag & drop init failed. Attach button still works.\n")
        else:
            self._append_line("[EFECT] Drag & drop not installed. Run: py -m pip install tkinterdnd2\n")

        # Attachments bar
        attach_bar = tk.Frame(self.root, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        attach_bar.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(0, 8))

        tk.Button(
            attach_bar, text="Attach File", command=self.attach_file_dialog,
            bg=self.panel, fg=self.green, bd=1, highlightthickness=1, highlightbackground=self.border,
            font=("Segoe UI", 10, "bold"), padx=12, pady=6
        ).pack(side=tk.LEFT, padx=12, pady=10)

        tk.Button(
            attach_bar, text="Screenshot", command=self.attach_screenshot,
            bg=self.panel, fg=self.green, bd=1, highlightthickness=1, highlightbackground=self.border,
            font=("Segoe UI", 10, "bold"), padx=12, pady=6
        ).pack(side=tk.LEFT, padx=(0, 12), pady=10)

        tk.Button(
            attach_bar, text="Clear", command=self.clear_attachments,
            bg=self.panel, fg=self.muted, bd=1, highlightthickness=1, highlightbackground=self.border,
            font=("Segoe UI", 10, "bold"), padx=12, pady=6
        ).pack(side=tk.LEFT, padx=(0, 12), pady=10)

        self.attach_label = tk.Label(
            attach_bar, text="Attachments: none",
            bg=self.panel, fg=self.muted, font=("Segoe UI", 10)
        )
        self.attach_label.pack(side=tk.LEFT, padx=(0, 12), pady=10)

        bottom = tk.Frame(self.root, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(8, 16))

        self.input_entry = tk.Entry(
            bottom, textvariable=self.input_var, bg=self.bg, fg=self.fg,
            insertbackground=self.green, font=("Segoe UI", 12), bd=0
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 8), pady=12, ipady=8)
        self.input_entry.bind("<Return>", lambda e: self.send())

        tk.Button(
            bottom, text="Stop", command=self.stop_stream,
            bg=self.panel, fg=self.green, bd=1, highlightthickness=1, highlightbackground=self.border,
            font=("Segoe UI", 12, "bold"), padx=18
        ).pack(side=tk.RIGHT, padx=(0, 12), pady=12, ipady=6)

        tk.Button(
            bottom, text="Send", command=self.send,
            bg=self.green, fg=self.bg, bd=0, font=("Segoe UI", 12, "bold"), padx=18
        ).pack(side=tk.RIGHT, padx=(8, 8), pady=12, ipady=6)

    def _append(self, text: str):
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", text)
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _append_line(self, text: str):
        self._append(text + "\n")

    def save_settings(self):
        self.settings["proxy_base_url"] = self.proxy_var.get().strip()
        self.settings["model"] = self.model_var.get().strip() or DEFAULT_MODEL
        safe_save_json(p_settings(), self.settings)
        self._append_line("[EFECT] Settings saved.\n")

    def stop_stream(self):
        self.stop_flag.set()

    # ---------------- Attachments ----------------

    def _refresh_attachment_label(self):
        if not self.attachments:
            self.attach_label.config(text="Attachments: none")
            return
        names = [a.get("name", "file") for a in self.attachments][-3:]
        more = "" if len(self.attachments) <= 3 else f" (+{len(self.attachments)-3})"
        self.attach_label.config(text="Attachments: " + ", ".join(names) + more)

    def clear_attachments(self):
        self.attachments.clear()
        self._refresh_attachment_label()
        self._append_line("[EFECT] Cleared attachments.\n")

    def attach_file_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Attach files",
            filetypes=[
                ("Supported", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.txt;*.log;*.md;*.pdf"),
                ("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"),
                ("Text", "*.txt;*.log;*.md"),
                ("PDF", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self._attach_paths(list(paths))

    def _on_drop_files(self, event):
        # tkinterdnd2 provides a string that may include braces for paths with spaces
        data = event.data
        paths = self._split_dnd_files(data)
        if paths:
            self._attach_paths(paths)

    def _split_dnd_files(self, data: str):
        # Handles:
        #   {C:\Path With Spaces\a.pdf} C:\Other\b.png
        out = []
        cur = ""
        in_brace = False
        for ch in data:
            if ch == "{":
                in_brace = True
                cur = ""
            elif ch == "}":
                in_brace = False
                if cur.strip():
                    out.append(cur.strip())
                cur = ""
            elif ch == " " and not in_brace:
                if cur.strip():
                    out.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            out.append(cur.strip())
        # filter real files
        out2 = []
        for p in out:
            p2 = p.strip().strip('"')
            if os.path.isfile(p2):
                out2.append(p2)
        return out2

    def _attach_paths(self, paths):
        for path in paths:
            try:
                self._attach_single_path(path)
            except Exception as e:
                self._append_line(f"[EFECT] Attach failed: {os.path.basename(path)} ({e})\n")
        self._refresh_attachment_label()

    def _attach_single_path(self, path: str):
        name = os.path.basename(path)

        if is_image_path(path):
            mime, data_url = file_to_data_url(path)
            self.attachments.append({"kind": "image_data_url", "name": name, "data_url": data_url})
            self._append_line(f"[EFECT] Attached image: {name}\n")
            return

        if is_text_path(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if len(text) > 120_000:
                text = text[:120_000] + "\n…(truncated)…"
            self.attachments.append({"kind": "inline_text", "name": name, "text": text})
            self._append_line(f"[EFECT] Attached text: {name}\n")
            return

        if is_pdf_path(path):
            # Upload on send (so you can attach multiple PDFs without delay)
            self.attachments.append({"kind": "pdf_path", "name": name, "path": path})
            self._append_line(f"[EFECT] Attached PDF: {name} (will upload on send)\n")
            return

        self._append_line(f"[EFECT] Unsupported file type: {name}\n")

    def attach_screenshot(self):
        img = ImageGrab.grab(all_screens=True)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        self.attachments.append({"kind": "image_data_url", "name": "screenshot.png", "data_url": data_url})
        self._refresh_attachment_label()
        self._append_line("[EFECT] Attached screenshot.\n")

    # ---------------- Proxy helpers ----------------

    def _proxy_base(self) -> str:
        base = (self.proxy_var.get().strip() or "").rstrip("/")
        return base

    def _token(self) -> str:
        return (self.token_var.get().strip() or "")

    def upload_pdf_to_proxy(self, path: str) -> str | None:
        """
        Calls POST /upload on your proxy; proxy uploads to OpenAI Files API and returns file_id. :contentReference[oaicite:7]{index=7}
        """
        base = self._proxy_base()
        if not base.startswith("http"):
            self._append_line("[EFECT] Set Proxy URL first (https://...).\n")
            return None
        token = self._token()
        if not token:
            self._append_line("[EFECT] Missing EFECT token.\n")
            return None

        url = base + "/upload"
        headers = {"Authorization": f"Bearer {token}"}

        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "application/pdf")}
            r = requests.post(url, headers=headers, files=files, timeout=300)

        if r.status_code >= 400:
            self._append_line(f"[EFECT] Upload error: {r.status_code} {r.text}\n")
            return None

        j = r.json()
        return j.get("id")

    # ---------------- Send / streaming ----------------

    def send(self):
        msg = self.input_var.get().strip()
        if not msg and not self.attachments:
            return

        token = self._token()
        if not token:
            messagebox.showerror("EFECT AI", "Missing EFECT token.")
            return

        base = self._proxy_base()
        if not base.startswith("http"):
            messagebox.showerror("EFECT AI", "Set Proxy URL (https://YOUR-PROXY...) and click Save.")
            return

        model = self.model_var.get().strip() or DEFAULT_MODEL
        self.settings["model"] = model
        self.settings["proxy_base_url"] = base
        safe_save_json(p_settings(), self.settings)

        self.input_var.set("")
        if msg:
            self._append_line(f"You: {msg}\n")
        else:
            self._append_line("You: (sent with attachments)\n")

        self.stop_flag.clear()
        threading.Thread(target=self._worker_stream, args=(base, token, model, msg), daemon=True).start()

    def _worker_stream(self, base: str, token: str, model: str, msg: str):
        def ui_append(s: str):
            self.root.after(0, lambda: self._append(s))

        def ui_line(s: str):
            self.root.after(0, lambda: self._append_line(s))

        # Build structured content parts for Responses API. :contentReference[oaicite:8]{index=8}
        content_parts = []
        content_parts.append({"type": "input_text", "text": f"{DEFAULT_SYSTEM_PROMPT}\n\nUSER: {msg or '(see attachments)'}"})

        # Upload PDFs now to get file_ids
        pdf_items = [a for a in self.attachments if a["kind"] == "pdf_path"]
        for a in pdf_items:
            ui_line(f"[EFECT] Uploading PDF: {a['name']}…\n")
            file_id = self.upload_pdf_to_proxy(a["path"])
            if not file_id:
                ui_line(f"[EFECT] PDF upload failed: {a['name']}\n")
                continue
            content_parts.append({"type": "input_file", "file_id": file_id})

        # Images and inline text
        for a in self.attachments:
            if a["kind"] == "image_data_url":
                content_parts.append({"type": "input_image", "image_url": a["data_url"]})
            elif a["kind"] == "inline_text":
                content_parts.append({"type": "input_text", "text": f"FILE: {a['name']}\n\n{a['text']}"})

        payload = {
            "model": model,
            "input": [{
                "role": "user",
                "content": content_parts
            }],
            "stream": True
        }

        url = base.rstrip("/") + "/responses"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        ui_append("EFECT AI: ")

        assistant_chunks = []
        try:
            with requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=300) as r:
                if r.status_code >= 400:
                    raise RuntimeError(f"{r.status_code}: {r.text}")

                for delta in parse_sse_stream(r.iter_lines(decode_unicode=True), self.stop_flag):
                    assistant_chunks.append(delta)
                    ui_append(delta)

            ui_line("\n")

        except Exception as ex:
            ui_line(f"\n[EFECT] Error: {ex}\n")

        # Clear attachments after send
        self.attachments.clear()
        self.root.after(0, self._refresh_attachment_label)

def main():
    if DND_OK:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = EfectAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
