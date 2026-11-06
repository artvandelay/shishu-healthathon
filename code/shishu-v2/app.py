"""SHISHU v2 HTTP and WebSocket entrypoint. Run with PYTHONPATH=code/shishu-v2."""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, unquote

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
ROLE_COOKIE = "shishu_role"
USER_COOKIE = "shishu_user"
_STATIC = Path(__file__).resolve().parent / "static"
_PUBLIC_PATHS = {"/health", "/login", "/api/login", "/api/logout", "/static/login.html"}
_COOKIE_MAX_AGE = 60 * 60 * 24 * 14


def _passcode() -> str:
    return os.environ.get("APP_PASSCODE") or "shishu2026"


def _admin_passcode() -> str:
    return os.environ.get("ADMIN_PASSCODE") or "shishu2026admin"


def _match(supplied: str, expected: str) -> bool:
    if not supplied or not expected:
        return False
    try:
        return secrets.compare_digest(supplied, expected)
    except Exception:
        return False


def _role_from_auth_cookie(value: str | None) -> str:
    if value in {"user", "admin"}:
        return value
    if _match(value or "", _admin_passcode()):
        return "admin"
    if _match(value or "", _passcode()):
        return "user"
    return ""


def _authorized(value: str | None) -> bool:
    return bool(_role_from_auth_cookie(value))


def _cookie_user(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        return unquote(raw)
    except Exception:
        return raw


def _identity_from_cookies(cookies) -> tuple[str, str]:
    role = cookies.get(ROLE_COOKIE) or _role_from_auth_cookie(cookies.get(COOKIE_NAME))
    if role not in {"user", "admin"}:
        role = "user" if _authorized(cookies.get(COOKIE_NAME)) else ""
    username = _cookie_user(cookies.get(USER_COOKIE)) or "Guest"
    return role, username


def _require_admin(request: Request) -> None:
    role, _username = _identity_from_cookies(request.cookies)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin login required")


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
def admin_page(request: Request):
    role, _username = _identity_from_cookies(request.cookies)
    if role != "admin":
        return RedirectResponse(url="/login?next=/admin", status_code=302)
    return _page("admin.html")


def _set_session(response: JSONResponse, role: str, username: str) -> None:
    common = {
        "httponly": True,
        "samesite": "lax",
        "path": "/",
        "max_age": _COOKIE_MAX_AGE,
    }
    response.set_cookie(COOKIE_NAME, role, **common)
    response.set_cookie(ROLE_COOKIE, role, **common)
    response.set_cookie(USER_COOKIE, quote(username, safe=""), **common)


@app.post("/api/login")
async def login(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    username = "Guest"
    supplied = ""
    if isinstance(body, dict):
        raw_name = body.get("username")
        if isinstance(raw_name, str) and raw_name.strip():
            username = raw_name.strip()[:80]
        if isinstance(body.get("passcode"), str):
            supplied = body["passcode"]
    if _match(supplied, _admin_passcode()):
        role = "admin"
    elif _match(supplied, _passcode()):
        role = "user"
    else:
        return JSONResponse({"ok": False}, status_code=401)
    response = JSONResponse({"ok": True, "role": role, "username": username})
    _set_session(response, role, username)
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    response.delete_cookie(ROLE_COOKIE, path="/")
    response.delete_cookie(USER_COOKIE, path="/")
    return response


@app.get("/api/config")
def get_config(request: Request) -> dict:
    role, username = _identity_from_cookies(request.cookies)
    return {
        "active_engine": db.get_setting("active_engine", "gemini"),
        "role": role or "user",
        "username": username,
        "settings": db.get_all_settings(),
    }


@app.get("/api/settings")
def get_settings(request: Request) -> dict:
    _require_admin(request)
    return db.get_all_settings()


@app.post("/api/settings")
async def post_settings(request: Request):
    _require_admin(request)
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a settings object")
    settings = db.update_settings(body)
    return {"ok": True, "settings": settings}


@app.post("/api/config")
async def set_config(request: Request):
    _require_admin(request)
    try:
        body = await request.json()
    except Exception:
        body = None
    engine = body.get("active_engine") if isinstance(body, dict) else None
    if engine not in {"gemini", "openai"}:
        raise HTTPException(status_code=400, detail="active_engine must be gemini or openai")
    db.set_setting("active_engine", engine)
    return {"ok": True, "active_engine": engine}


@app.get("/api/bugs")
def get_bugs(request: Request):
    _require_admin(request)
    return db.list_bug_reports()


@app.post("/api/bugs")
async def post_bug(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a bug report object")
    role, username = _identity_from_cookies(request.cookies)
    try:
        call_id = int(body.get("call_id") or 0)
    except (TypeError, ValueError):
        call_id = 0
    try:
        elapsed = int(body.get("call_elapsed_sec") or 0)
    except (TypeError, ValueError):
        elapsed = 0
    extra = body.get("client_state") if isinstance(body.get("client_state"), dict) else {}
    bug = db.create_bug_report(
        call_id,
        username or role or "user",
        str(body.get("category") or ""),
        str(body.get("note") or ""),
        elapsed,
        extra,
    )
    return {"ok": True, "bug": bug}


@app.get("/api/calls")
def list_calls(request: Request):
    _require_admin(request)
    return db.list_calls()


@app.get("/api/calls/{call_id}")
def get_call(call_id: int, request: Request):
    _require_admin(request)
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@app.get("/api/audio/{call_id}")
def get_audio(call_id: int, request: Request):
    _require_admin(request)
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


def _ws_identity(ws: WebSocket) -> tuple[str, str]:
    role, username = _identity_from_cookies(ws.cookies)
    return role or "user", username


@app.websocket("/ws/gemini")
async def ws_gemini(ws: WebSocket) -> None:
    if not _authorized(ws.cookies.get(COOKIE_NAME)):
        await _reject_ws(ws)
        return
    role, username = _ws_identity(ws)
    await gemini_live.handle_gemini_ws(ws, caller_name=username, user_role=role)


@app.websocket("/ws/openai")
async def ws_openai(ws: WebSocket) -> None:
    if not _authorized(ws.cookies.get(COOKIE_NAME)):
        await _reject_ws(ws)
        return
    role, username = _ws_identity(ws)
    await openai_live.handle_openai_ws(ws, caller_name=username, user_role=role)
