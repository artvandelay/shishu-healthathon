"""SHISHU v2 HTTP and WebSocket entrypoint. Run with PYTHONPATH=code/shishu-v2."""

import base64
import hashlib
import hmac
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

import answer_extractor
import db
import gemini_live
import mother_report
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
    return os.environ.get("ADMIN_PASSCODE") or ""


def _match(supplied: str, expected: str) -> bool:
    if not supplied or not expected:
        return False
    try:
        return secrets.compare_digest(supplied, expected)
    except Exception:
        return False


def _cookie_secure(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "")
    scheme = forwarded.split(",")[0].strip().lower() if forwarded else request.url.scheme
    return scheme == "https"


def _session_token(role: str, username: str) -> str:
    exp = int(time.time()) + _COOKIE_MAX_AGE
    user_b64 = base64.urlsafe_b64encode(username.encode()).decode().rstrip("=")
    body = f"v1.{role}.{exp}.{user_b64}"
    sig = hmac.new(
        db.get_session_secret().encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{body}.{sig}"


def _read_session(value: str | None) -> tuple[str, str]:
    if not value or value in {"user", "admin"}:
        return "", ""
    parts = value.split(".")
    if len(parts) != 5 or parts[0] != "v1":
        return "", ""
    body = ".".join(parts[:4])
    expected = hmac.new(
        db.get_session_secret().encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()
    try:
        if not hmac.compare_digest(parts[4], expected):
            return "", ""
    except Exception:
        return "", ""
    role = parts[1]
    if role not in {"user", "admin"}:
        return "", ""
    try:
        exp = int(parts[2])
    except ValueError:
        return "", ""
    if exp < int(time.time()):
        return "", ""
    pad = "=" * (-len(parts[3]) % 4)
    try:
        username = base64.urlsafe_b64decode(parts[3] + pad).decode()
    except Exception:
        return "", ""
    return role, username or "Guest"


def _authorized(value: str | None) -> bool:
    role, _username = _read_session(value)
    return role in {"user", "admin"}


def _identity_from_cookies(cookies) -> tuple[str, str]:
    role, username = _read_session(cookies.get(COOKIE_NAME))
    if role not in {"user", "admin"}:
        return "", ""
    return role, username or "Guest"


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


@app.get("/static/shishu-logo.jpg")
def shishu_logo() -> FileResponse:
    return FileResponse(_STATIC / "shishu-logo.jpg", media_type="image/jpeg")


@app.get("/static/shishu-logo.png")
def shishu_logo_png() -> FileResponse:
    return FileResponse(_STATIC / "shishu-logo.png", media_type="image/png")


@app.get("/")
def call_page() -> FileResponse:
    return _page("call.html")


@app.get("/admin")
def admin_page(request: Request):
    role, _username = _identity_from_cookies(request.cookies)
    if role != "admin":
        return RedirectResponse(url="/login?next=/admin", status_code=302)
    return _page("admin.html")


def _set_session(response: JSONResponse, request: Request, role: str, username: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        _session_token(role, username),
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        path="/",
        max_age=_COOKIE_MAX_AGE,
    )


def _clear_named_cookie(response: JSONResponse, name: str, secure: bool) -> None:
    response.delete_cookie(name, path="/", secure=secure, samesite="lax")


def _clear_session(response: JSONResponse, request: Request) -> None:
    secure = _cookie_secure(request)
    for name in (COOKIE_NAME, ROLE_COOKIE, USER_COOKIE):
        _clear_named_cookie(response, name, False)
        if secure:
            _clear_named_cookie(response, name, True)


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
    _set_session(response, request, role, username)
    return response


@app.post("/api/logout")
def logout(request: Request):
    response = JSONResponse({"ok": True})
    _clear_session(response, request)
    return response


@app.get("/api/config")
def get_config(request: Request) -> dict:
    role, username = _identity_from_cookies(request.cookies)
    settings = db.get_all_settings()
    settings["active_engine"] = "gemini"
    return {
        "active_engine": "gemini",
        "role": role or "user",
        "username": username,
        "settings": settings,
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
    body["active_engine"] = "gemini"
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
    db.set_setting("active_engine", "gemini")
    return {"ok": True, "active_engine": "gemini"}


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


@app.get("/api/calls/{call_id}/mother.pdf")
async def mother_pdf(call_id: int, request: Request):
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    if str(call.get("status") or "") == "active":
        raise HTTPException(status_code=409, detail="Call is still running")
    role, username = _identity_from_cookies(request.cookies)
    owner = role == "user" and (call.get("caller_name") or "") == username
    if role != "admin" and not owner:
        raise HTTPException(status_code=403, detail="Not allowed")
    try:
        answers = await answer_extractor.cached_answers(call)
        pdf = mother_report.render_pdf(mother_report.build(call, answers))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not build the report") from exc
    name = mother_report.filename(call)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


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
