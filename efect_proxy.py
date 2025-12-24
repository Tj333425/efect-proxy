from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="EFECT Proxy")

@app.get("/", response_class=HTMLResponse)
@app.head("/")
def home():
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
                border-radius: 16px;
                padding: 40px;
                width: 520px;
                text-align: center;
            }
            h1 { font-size: 48px; margin: 0 0 10px; }
            p { color: #cccccc; font-size: 18px; margin: 8px 0; }
            .status { margin-top: 18px; color: #6bff6b; font-weight: 700; }
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
