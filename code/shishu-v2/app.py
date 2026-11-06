"""SHISHU v2 HTTP and WebSocket entrypoint. Run with PYTHONPATH=code/shishu-v2."""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

import db
import gemini_live
import openai_live
import seed

COOKIE_NAME = "shishu_auth"
_STATIC = Path(__file__).resolve().parent / "static"
_PUBLIC_PATHS = {"/health", "/login", "/api/login", "/api/logout", "/static/login.html"}


def _passcode() -> str:
    return os.environ.get("APP_PASSCODE") or "shishu2026"


def _authorized(value: str | None) -> bool:
    if not value:
        return False
    expected = _passcode()
    try:
        return secrets.compare_digest(value, expected)
    except Exception:
        return False


def _audio_dir() -> Path:
    return (Path(db.DATA_DIR) / "audio").resolve()


def _resolve_audio(audio_path: str) -> Path | None:
    raw = (audio_path or "").strip()
    if not raw:
        return None
    root = _audio_dir()
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        resolved = (root / candidate.name).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None
    if not resolved.is_file():
        return None
    return resolved


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    seed.ensure_seeded()
    yield


app = FastAPI(lifespan=_lifespan)


@app.middleware("http")
async def _require_login(request: Request, call_next):
    path = request.url.path
    if path in _PUBLIC_PATHS or _authorized(request.cookies.get(COOKIE_NAME)):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"ok": False}, status_code=401)
    if path in {"/", "/admin"}:
        nxt = quote(path, safe="/")
        return RedirectResponse(url=f"/login?next={nxt}", status_code=302)
    return await call_next(request)


def _page(name: str) -> FileResponse:
    return FileResponse(_STATIC / name, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/login")
def login_page() -> FileResponse:
    return _page("login.html")


@app.get("/static/login.html")
def static_login_page() -> FileResponse:
    return _page("login.html")


@app.get("/")
def call_page() -> FileResponse:
    return _page("call.html")


@app.get("/admin")
def admin_page() -> FileResponse:
    return _page("admin.html")


@app.post("/api/login")
async def login(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    supplied = ""
    if isinstance(body, dict) and isinstance(body.get("passcode"), str):
        supplied = body["passcode"]
    expected = _passcode()
    try:
        ok = secrets.compare_digest(supplied, expected)
    except Exception:
        ok = False
    if not ok:
        return JSONResponse({"ok": False}, status_code=401)
    response = JSONResponse({"ok": True})
    response.set_cookie(
        COOKIE_NAME,
        expected,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 24 * 14,
    )
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@app.get("/api/config")
def get_config() -> dict:
    return {"active_engine": db.get_setting("active_engine", "gemini")}


@app.post("/api/config")
async def set_config(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    engine = body.get("active_engine") if isinstance(body, dict) else None
    if engine not in {"gemini", "openai"}:
        raise HTTPException(status_code=400, detail="active_engine must be gemini or openai")
    db.set_setting("active_engine", engine)
    return {"ok": True, "active_engine": engine}


@app.get("/api/calls")
def list_calls():
    return db.list_calls()


@app.get("/api/calls/{call_id}")
def get_call(call_id: int):
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@app.get("/api/audio/{call_id}")
def get_audio(call_id: int):
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    path = _resolve_audio(str(call.get("audio_path") or ""))
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


async def _reject_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws.close(code=1008)


@app.websocket("/ws/gemini")
async def ws_gemini(ws: WebSocket) -> None:
    if not _authorized(ws.cookies.get(COOKIE_NAME)):
        await _reject_ws(ws)
        return
    # handle_gemini_ws accepts while the socket is still CONNECTING.
    await gemini_live.handle_gemini_ws(ws)


@app.websocket("/ws/openai")
async def ws_openai(ws: WebSocket) -> None:
    if not _authorized(ws.cookies.get(COOKIE_NAME)):
        await _reject_ws(ws)
        return
    # handle_openai_ws always accepts. Do not accept here.
    await openai_live.handle_openai_ws(ws)
