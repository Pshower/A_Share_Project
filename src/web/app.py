"""Local research HTTP API. Start with python -m src.web.serve.

This workbench used to serve the PPO training line -- datasets, experiments, models, online
downloads, training jobs.  That line is closed and its code is gone, so what remains is the one
thing still read every day: the execution books and the shadow recorder.  The API is read-only by
construction; nothing here can place an order.
"""

import secrets
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.data.preprocess import ROOT


def create_app(root=ROOT):
    root = Path(root).resolve()
    token = secrets.token_urlsafe(32)

    app = FastAPI(title="A-share Execution Workbench", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        if request.url.hostname not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"error": "Untrusted Host"}, status_code=403)
        origin = request.headers.get("origin")
        expected = str(request.base_url).rstrip("/")
        if origin and origin != expected:
            return JSONResponse({"error": "Cross-origin requests are disabled"}, status_code=403)
        if request.method not in {"GET", "HEAD"} and not secrets.compare_digest(
                request.headers.get("x-csrf-token", ""), token):
            return JSONResponse({"error": "Invalid local session token"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"error": str(error)}, status_code=409)

    @app.get("/api/session")
    def session():
        return dict(token=token)

    @app.get("/api/system")
    def system():
        return dict(interpreter=sys.executable, environment=Path(sys.prefix).name,
                    workspace=str(root), price_basis="hfq_research")

    @app.get("/api/execution/books")
    def execution_books():
        from . import execution
        return execution.books(root)

    @app.get("/api/execution/books/{name}")
    def execution_book(name: str):
        from . import execution
        return execution.book(root, name)

    @app.get("/api/execution/shadow")
    def execution_shadow():
        from . import execution
        return execution.shadow(root)

    dist = root / "web/dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
