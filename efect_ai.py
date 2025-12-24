import os
import json
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import List, Dict, Any

import requests

APP_NAME = "EFECT_AI"

DEFAULT_MODEL = "gpt-4.1-mini"

DEFAULT_SYSTEM_PROMPT = (
    "You are EFECT AI, a helpful assistant for the EFECT brand.\n"
    "Be practical, step-by-step, and creative for open-ended requests.\n"
    "If the user asks for something unsafe or disallowed, refuse clearly.\n"
)

DEFAULT_SETTINGS = {
    "proxy_base_url": "https://efect-proxy.onrender.com",
    "model": DEFAULT_MODEL,
    "token": "",
}

THEME = {
    "bg": "#0B0B0B",
    "panel": "#101010",
    "fg": "#E8E8E8",
    "muted": "#9AA0AA",
    "green": "#1BFF4A",
    "border": "#2A2A2A",
    "danger": "#FF4A4A",
}


def appdata_dir() -> str:
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    p = os.path.join(base, APP_NAME)
    os.makedirs(p, exist_ok=True)
    return p


def settings_path() -> str:
    return os.path.join(appdata_dir(), "settings.json")


def safe_load_settings() -> Dict[str, Any]:
    p = settings_path()
    if not os.path.exists(p):
        return dict(DEFAULT_SETTINGS)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = dict(DEFAULT_SETTINGS)
        out.update({k: v for k, v in data.items() if k in out})
        return out
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(data: Dict[str, Any]) -> None:
    try:
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def build_responses_payload(system_prompt: str, history: List[Dict[str, str]], user_text: str, model: str) -> Dict[str, Any]:
    """
    CRITICAL FIX:
      - user/system => input_text
      - assistant => output_text
    """
    inp = []

    if system_prompt.strip():
        inp.append({
            "role": "system",
            "content": [{"type": "input_text", "text": system_prompt}]
        })

    for m in history:
        role = (m.get("role") or "").strip().lower()
        text = (m.get("text") or "").strip()
        if not text:
            continue

        if role == "user":
            inp.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
        elif role == "assistant":
            inp.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})

    if user_text.strip():
        inp.append({"role": "user", "content": [{"type": "input_text", "text": user_text}]})

    return {
        "model": model,
        "input": inp,
        "stream": False,  # prevents duplicate UI issues
    }


class EfectAIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EFECT AI")
        self.root.configure(bg=THEME["bg"])
        self.root.geometry("1100x740")
        self.root.minsize(900, 620)

        self.settings = safe_load_settings()

        self.proxy_var = tk.StringVar(value=self.settings["proxy_base_url"])
        self.model_var = tk.StringVar(value=self.settings["model"])
        self.token_var = tk.StringVar(value=self.settings["token"])

        # Chat history used for context
        self.chat_history: List[Dict[str, str]] = []

        self._build_ui()

    def _build_ui(self):
        header = tk.Frame(self.root, bg=THEME["panel"], highlightbackground=THEME["border"], highlightthickness=1)
        header.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(16, 10))

        left = tk.Frame(header, bg=THEME["panel"])
        left.pack(side=tk.LEFT, padx=14, pady=10)
        tk.Label(left, text="EFECT", fg=THEME["green"], bg=THEME["panel"], font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(left, text="AI", fg=THEME["fg"], bg=THEME["panel"], font=("Segoe UI", 18, "bold")).pack(anchor="w")

        right = tk.Frame(header, bg=THEME["panel"])
        right.pack(side=tk.RIGHT, padx=14, pady=10)

        tk.Label(right, text="Proxy:", fg=THEME["muted"], bg=THEME["panel"], font=("Segoe UI", 10)).grid(row=0, column=0, sticky="e", padx=(0, 6))
        tk.Entry(right, textvariable=self.proxy_var, width=40, bg=THEME["bg"], fg=THEME["fg"],
                 insertbackground=THEME["green"], highlightbackground=THEME["border"], highlightthickness=1).grid(row=0, column=1, padx=(0, 10))

        tk.Label(right, text="Model:", fg=THEME["muted"], bg=THEME["panel"], font=("Segoe UI", 10)).grid(row=0, column=2, sticky="e", padx=(0, 6))
        tk.Entry(right, textvariable=self.model_var, width=18, bg=THEME["bg"], fg=THEME["fg"],
                 insertbackground=THEME["green"], highlightbackground=THEME["border"], highlightthickness=1).grid(row=0, column=3, padx=(0, 10))

        tk.Label(right, text="Token:", fg=THEME["muted"], bg=THEME["panel"], font=("Segoe UI", 10)).grid(row=1, column=0, sticky="e", padx=(0, 6), pady=(8, 0))
        tk.Entry(right, textvariable=self.token_var, width=40, show="•", bg=THEME["bg"], fg=THEME["fg"],
                 insertbackground=THEME["green"], highlightbackground=THEME["border"], highlightthickness=1).grid(row=1, column=1, padx=(0, 10), pady=(8, 0))

        tk.Button(right, text="Save", command=self._save_settings, bg=THEME["panel"], fg=THEME["fg"],
                  activebackground=THEME["panel"], activeforeground=THEME["fg"], highlightbackground=THEME["border"]).grid(row=1, column=2, columnspan=2, sticky="we", pady=(8, 0))

        # Main layout
        main = tk.Frame(self.root, bg=THEME["bg"])
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        self.chat = tk.Text(main, bg=THEME["bg"], fg=THEME["fg"], insertbackground=THEME["green"],
                            wrap="word", bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        self.chat.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.chat.config(state="disabled")

        footer = tk.Frame(main, bg=THEME["bg"])
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        self.input = tk.Text(footer, height=3, bg=THEME["panel"], fg=THEME["fg"], insertbackground=THEME["green"],
                             wrap="word", bd=0, highlightthickness=1, highlightbackground=THEME["border"])
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True)

        send_btn = tk.Button(footer, text="Send", command=self.send, bg=THEME["green"], fg="#000000",
                             activebackground=THEME["green"], activeforeground="#000000", bd=0, padx=18, pady=10)
        send_btn.pack(side=tk.RIGHT, padx=(10, 0))

        self._append_line("EFECT AI ready. Type a message and press Send.\n", tag="muted")

    def _save_settings(self):
        self.settings["proxy_base_url"] = self.proxy_var.get().strip()
        self.settings["model"] = self.model_var.get().strip() or DEFAULT_MODEL
        self.settings["token"] = self.token_var.get().strip()
        save_settings(self.settings)
        messagebox.showinfo("Saved", "Settings saved.")

    def _append_line(self, text: str, tag: str = "normal"):
        self.chat.config(state="normal")
        if tag == "muted":
            self.chat.insert("end", text)
        else:
            self.chat.insert("end", text)
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _append_msg(self, who: str, text: str):
        self.chat.config(state="normal")
        self.chat.insert("end", f"{who}\n", ())
        self.chat.insert("end", f"{text}\n\n", ())
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _get_input_text(self) -> str:
        return self.input.get("1.0", "end").strip()

    def _clear_input(self):
        self.input.delete("1.0", "end")

    def send(self):
        msg = self._get_input_text()
        if not msg:
            return

        proxy = self.proxy_var.get().strip().rstrip("/")
        token = self.token_var.get().strip()
        model = self.model_var.get().strip() or DEFAULT_MODEL

        if not proxy.startswith("http"):
            messagebox.showerror("Proxy URL", "Set Proxy to a valid URL (https://...)")
            return
        if not token:
            messagebox.showerror("Token", "Enter your EFECT proxy token.")
            return

        self._append_msg("You", msg)
        self.chat_history.append({"role": "user", "text": msg})
        self._clear_input()

        def worker():
            try:
                payload = build_responses_payload(
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    history=self.chat_history[-20:],  # keep recent context
                    user_text=msg,
                    model=model,
                )
                r = requests.post(
                    proxy + "/responses",
                    headers={"Authorization": f"Bearer {token}"},
                    json=payload,
                    timeout=300
                )
                data = r.json()

                if r.status_code >= 400:
                    # Show error
                    err_txt = json.dumps(data, indent=2)
                    self.root.after(0, lambda: self._append_msg("EFECT AI", err_txt))
                    return

                out = (data.get("output_text") or "").strip()
                if not out:
                    out = "(No output_text returned. Check proxy response.)"

                # Save assistant message into history (as plain text)
                self.chat_history.append({"role": "assistant", "text": out})
                self.root.after(0, lambda: self._append_msg("EFECT AI", out))

            except Exception as e:
                self.root.after(0, lambda: self._append_msg("EFECT AI", f"Error: {e}"))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = EfectAIApp(root)
    root.mainloop()
