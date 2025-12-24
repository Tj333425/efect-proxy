# efect_ai.py
# EFECT AI Desktop Client (Tkinter)
# - Online proxy support (HTTPS)
# - Attach button + Screenshot button
# - Drag & drop files into the chat (optional via tkinterdnd2)
# - Images: sent as input_image (base64 data URL)
# - PDFs/other files: uploaded to proxy -> returns file_id -> sent as input_file
#
# Install deps:
#   py -m pip install requests pillow
# Optional drag-drop:
#   py -m pip install tkinterdnd2
#
# Build EXE:
#   py -m pip install pyinstaller
#   py -m PyInstaller --onefile --windowed --name "EFECT AI" efect_ai.py

import os
import json
import time
import base64
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox
from io import BytesIO
from typing import Optional, Tuple

import requests
from PIL import ImageGrab

# Drag & drop (optional)
DND_OK = False
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # type: ignore
    DND_OK = True
except Exception:
    TkinterDnD = None
    DND_FILES = None

APP_NAME = "EFECT_AI"
DEFAULT_MODEL = "gpt-4.1-mini"

DEFAULT_SETTINGS = {
    "proxy_base_url": "https://efect-proxy.onrender.com",
    "model": DEFAULT_MODEL,
    "token": "efect-12345",
    "auto_open_website": False,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are EFECT AI, an assistant for the EFECT brand.\n"
    "Be practical and step-by-step.\n"
    "No cheats/spoofers/bypasses.\n"
    "If user attaches screenshots/files, analyze them.\n"
)

def appdata_dir() -> str:
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    p = os.path.join(base, APP_NAME)
    os.makedirs(p, exist_ok=True)
    return p

def settings_path() -> str:
    return os.path.join(appdata_dir(), "settings.json")

def safe_load_json(path: str, default: dict) -> dict:
    try:
        if not os.path.exists(path):
            return dict(default)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(default)
        merged = dict(default)
        merged.update(data)
        return merged
    except Exception:
        return dict(default)

def safe_save_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

def file_to_data_url(path: str) -> Tuple[str, str]:
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
        self.settings = safe_load_json(settings_path(), DEFAULT_SETTINGS)
        self.stop_flag = threading.Event()
        self.attachments = []  # list of dicts: {kind, name, path, data_url, file_id}

        # Theme
        self.bg = "#0B0B0B"
        self.panel = "#0F1016"
        self.fg = "#E8E8F0"
        self.muted = "#9AA0AA"
        self.green = "#7CFF7A"
        self.border = "#2A2D3A"

        self.root.title("EFECT AI")
        self.root.geometry("1100x740")
        self.root.minsize(900, 620)
        self.root.configure(bg=self.bg)

        self.token_var = tk.StringVar(value=self.settings.get("token", ""))
        self.model_var = tk.StringVar(value=self.settings.get("model", DEFAULT_MODEL))
        self.proxy_var = tk.StringVar(value=self.settings.get("proxy_base_url", "https://efect-proxy.onrender.com"))
        self.auto_open_var = tk.BooleanVar(value=bool(self.settings.get("auto_open_website", False)))

        self._build_ui()

        # Optional: auto-open website
        if self.auto_open_var.get():
            try:
                webbrowser.open(self.proxy_var.get().rstrip("/"))
            except Exception:
                pass

    def _save_settings(self):
        self.settings["token"] = self.token_var.get().strip()
        self.settings["model"] = self.model_var.get().strip() or DEFAULT_MODEL
        self.settings["proxy_base_url"] = self.proxy_var.get().strip()
        self.settings["auto_open_website"] = bool(self.auto_open_var.get())
        safe_save_json(settings_path(), self.settings)

    def _build_ui(self):
        header = tk.Frame(self.root, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        header.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(16, 10))

        left = tk.Frame(header, bg=self.panel)
        left.pack(side=tk.LEFT, padx=14, pady=10)
        tk.Label(left, text="EFECT", fg=self.green, bg=self.panel, font=("Segoe UI", 22, "bold")).pack(side=tk.LEFT)
        tk.Label(left, text=" AI", fg=self.fg, bg=self.panel, font=("Segoe UI", 22, "bold")).pack(side=tk.LEFT)
        tk.Label(left, text="  •  Online", fg=self.muted, bg=self.panel, font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=(12, 0))

        right = tk.Frame(header, bg=self.panel)
        right.pack(side=tk.RIGHT, padx=14, pady=10)

        tk.Label(right, text="Model:", fg=self.muted, bg=self.panel, font=("Segoe UI", 11)).grid(row=0, column=0, sticky="e", padx=(0, 8))
        tk.Entry(right, textvariable=self.model_var, width=18, bg=self.bg, fg=self.fg, insertbackground=self.green,
                 highlightbackground=self.border, highlightthickness=1).grid(row=0, column=1, padx=(0, 14))

        tk.Label(right, text="Token:", fg=self.muted, bg=self.panel, font=("Segoe UI", 11)).grid(row=0, column=2, sticky="e", padx=(0, 8))
        tk.Entry(right, textvariable=self.token_var, width=22, bg=self.bg, fg=self.fg, insertbackground=self.green,
                 highlightbackground=self.border, highlightthickness=1, show="•").grid(row=0, column=3, padx=(0, 14))

        tk.Button(right, text="Open Website", command=self.open_website,
                  bg=self.bg, fg=self.fg, activebackground=self.panel, activeforeground=self.fg,
                  highlightbackground=self.border, highlightthickness=1, padx=14, pady=6).grid(row=0, column=4, padx=(0, 10))

        tk.Checkbutton(
            right, text="Auto-open", variable=self.auto_open_var,
            bg=self.panel, fg=self.muted, activebackground=self.panel, activeforeground=self.muted,
            selectcolor=self.bg, command=self._save_settings
        ).grid(row=0, column=5, padx=(0, 0))

        # Proxy URL row
        proxy_row = tk.Frame(self.root, bg=self.bg)
        proxy_row.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(0, 10))

        tk.Label(proxy_row, text="Proxy URL:", fg=self.muted, bg=self.bg, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=(0, 8))
        tk.Entry(proxy_row, textvariable=self.proxy_var, bg=self.panel, fg=self.fg, insertbackground=self.green,
                 highlightbackground=self.border, highlightthickness=1).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        tk.Button(proxy_row, text="Save", command=self._save_settings,
                  bg=self.green, fg="#000", activebackground=self.green, activeforeground="#000",
                  padx=16, pady=6).pack(side=tk.LEFT)

        # Chat area
        chat_frame = tk.Frame(self.root, bg=self.panel, highlightbackground=self.border, highlightthickness=1)
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))

        self.chat = tk.Text(chat_frame, bg=self.panel, fg=self.fg, insertbackground=self.green, wrap=tk.WORD,
                            font=("Segoe UI", 11), bd=0)
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=12)

        scrollbar = tk.Scrollbar(chat_frame, command=self.chat.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.config(yscrollcommand=scrollbar.set)

        self._append_line("EFECT AI: Hello! Ask me anything. Use Attach or Screenshot for files/images.\n")

        # Bottom input
        bottom = tk.Frame(self.root, bg=self.bg)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 16))

        self.input = tk.Text(bottom, height=3, bg=self.panel, fg=self.fg, insertbackground=self.green,
                             highlightbackground=self.border, highlightthickness=1, font=("Segoe UI", 11))
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        controls = tk.Frame(bottom, bg=self.bg)
        controls.pack(side=tk.RIGHT)

        tk.Button(controls, text="Attach", command=self.attach_file,
                  bg=self.bg, fg=self.fg, highlightbackground=self.border, highlightthickness=1,
                  padx=14, pady=8).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(controls, text="Screenshot", command=self.screenshot,
                  bg=self.bg, fg=self.fg, highlightbackground=self.border, highlightthickness=1,
                  padx=14, pady=8).pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(controls, text="Send", command=self.send,
                  bg=self.green, fg="#000", padx=18, pady=10).pack(side=tk.LEFT)

        # Optional drag & drop
        if DND_OK and isinstance(self.root, TkinterDnD.Tk):
            try:
                self.chat.drop_target_register(DND_FILES)
                self.chat.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

    def _append_line(self, text: str):
        self.chat.insert(tk.END, text)
        self.chat.see(tk.END)

    def open_website(self):
        url = self.proxy_var.get().strip().rstrip("/")
        if not url:
            messagebox.showerror("EFECT AI", "Proxy URL is empty.")
            return
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("EFECT AI", str(e))

    def _proxy_base(self) -> str:
        return self.proxy_var.get().strip().rstrip("/")

    def _token(self) -> str:
        return self.token_var.get().strip()

    def attach_file(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self._add_attachment(path)

    def _add_attachment(self, path: str):
        name = os.path.basename(path)
        if is_image(path):
            mime, data_url = file_to_data_url(path)
            self.attachments.append({"kind": "image", "name": name, "path": path, "mime": mime, "data_url": data_url})
            self._append_line(f"[EFECT] Attached image: {name}\n")
        else:
            self.attachments.append({"kind": "file", "name": name, "path": path})
            self._append_line(f"[EFECT] Attached file: {name}\n")

    def _on_drop(self, event):
        # event.data can contain one or multiple paths
        data = event.data
        # Clean braces on windows paths
        paths = []
        cur = ""
        in_brace = False
        for ch in data:
            if ch == "{":
                in_brace = True
                cur = ""
            elif ch == "}":
                in_brace = False
                if cur.strip():
                    paths.append(cur.strip())
                cur = ""
            elif ch == " " and not in_brace:
                if cur.strip():
                    paths.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            paths.append(cur.strip())

        for p in paths:
            p = p.strip()
            if os.path.exists(p):
                self._add_attachment(p)

    def screenshot(self):
        try:
            img = ImageGrab.grab()
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"
            name = f"screenshot_{int(time.time())}.png"
            self.attachments.append({"kind": "image", "name": name, "mime": "image/png", "data_url": data_url})
            self._append_line(f"[EFECT] Captured screenshot: {name}\n")
        except Exception as e:
            messagebox.showerror("EFECT AI", f"Screenshot failed: {e}")

    def _upload_file_to_proxy(self, path: str) -> Optional[str]:
        base = self._proxy_base()
        if not base.startswith("http"):
            self._append_line("[EFECT] Set Proxy URL first (https://...)\n")
            return None

        token = self._token()
        if not token:
            self._append_line("[EFECT] Missing EFECT token.\n")
            return None

        url = base + "/upload"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f, "application/octet-stream")}
                r = requests.post(url, headers=headers, files=files, timeout=300)
            if r.status_code >= 400:
                self._append_line(f"[EFECT] Upload error: {r.status_code} {r.text}\n")
                return None
            j = r.json()
            return j.get("id")
        except Exception as e:
            self._append_line(f"[EFECT] Upload exception: {e}\n")
            return None

    def send(self):
        msg = self.input.get("1.0", tk.END).strip()
        if not msg and not self.attachments:
            return

        self._save_settings()

        # display user message
        if msg:
            self._append_line(f"You: {msg}\n")

        # disable send while working
        self.input.delete("1.0", tk.END)

        t = threading.Thread(target=self._send_worker, args=(msg, list(self.attachments)), daemon=True)
        self.attachments.clear()
        t.start()

    def _send_worker(self, msg: str, attachments: list):
        try:
            base = self._proxy_base()
            token = self._token()
            model = self.model_var.get().strip() or DEFAULT_MODEL

            if not base.startswith("http"):
                self._append_line("[EFECT] Error: Proxy URL must start with http(s)\n")
                return
            if not token:
                self._append_line("[EFECT] Error: Missing EFECT token.\n")
                return

            # Build Responses input (text + images + files)
            content = []
            if msg:
                content.append({"type": "input_text", "text": msg})

            # Images: inline data URLs
            for a in attachments:
                if a.get("kind") == "image":
                    content.append({"type": "input_image", "image_url": a["data_url"]})

            # Files: upload to proxy -> use file_id
            for a in attachments:
                if a.get("kind") == "file":
                    file_id = self._upload_file_to_proxy(a["path"])
                    if file_id:
                        content.append({"type": "input_file", "file_id": file_id})
                    else:
                        self._append_line(f"[EFECT] Skipped file (upload failed): {a.get('name')}\n")

            payload = {
                "model": model,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": DEFAULT_SYSTEM_PROMPT}]},
                    {"role": "user", "content": content if content else [{"type": "input_text", "text": msg or ""}]},
                ],
                "stream": False,
            }

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            r = requests.post(base + "/responses", headers=headers, json=payload, timeout=300)
            if r.status_code >= 400:
                self._append_line(f"EFECT AI:\n[EFECT] Error: {r.text}\n")
                return

            data = r.json()

            # Extract assistant text from Responses API output
            out_text = None
            try:
                # common shape: data["output"][0]["content"][0]["text"]
                output = data.get("output", [])
                for item in output:
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") in ("output_text", "text"):
                                out_text = c.get("text")
                                break
                    if out_text:
                        break
            except Exception:
                out_text = None

            if not out_text:
                out_text = json.dumps(data, indent=2)[:4000]

            self._append_line(f"EFECT AI: {out_text}\n\n")

        except Exception as e:
            self._append_line(f"EFECT AI:\n[EFECT] Error: {e}\n")


def main():
    if DND_OK:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = EfectAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
