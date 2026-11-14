"""SHISHU v2 HTTP and WebSocket entrypoint. Run with PYTHONPATH=code/shishu-v2."""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
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

import db
import gemini_live
import report_service
import seed

logger = logging.getLogger(__name__)

COOKIE_NAME = "shishu_auth"
ROLE_COOKIE = "shishu_role"
USER_COOKIE = "shishu_user"
_STATIC = Path(__file__).resolve().parent / "static"
_PUBLIC_PATHS = {"/health", "/login", "/api/login", "/api/logout", "/static/login.html"}
_COOKIE_MAX_AGE = 60 * 60 * 24 * 14
_CLINIC_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_STATIC_TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".js": "text/javascript",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ttf": "font/ttf",
    ".woff2": "font/woff2",
}
_SHARE_EVENTS = frozenset({"share_clicked"})
_BUG_BYTE_CAP = 512 * 1024
_BUG_NOTE_CAP = 2000
_LOGIN_FAILS = 15
_LOGIN_WINDOW_SEC = 60.0
_PDF_REBUILDS = 5
_PDF_WINDOW_SEC = 60.0
_rate_lock = threading.Lock()
_login_failures: dict[str, list[float]] = {}
_pdf_rebuilds: dict[str, list[float]] = {}


def _passcode() -> str:
    configured = (os.environ.get("APP_PASSCODE") or "").strip()
    if configured:
        return configured
    if not (os.environ.get("RAILWAY_ENVIRONMENT") or "").strip():
        return "shishu2026"
    return ""


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


def _client_ip(request: Request) -> str:
    client = request.client
    if client is None:
        return ""
    return client.host or ""


def _prune_hits(stamps: list[float], now: float, window: float) -> list[float]:
    return [stamp for stamp in stamps if now - stamp < window]


def _rate_limited(bucket: dict[str, list[float]], key: str, limit: int, window: float) -> bool:
    """True when this key already has `limit` hits inside the window."""
    now = time.monotonic()
    with _rate_lock:
        stamps = _prune_hits(bucket.get(key, []), now, window)
        bucket[key] = stamps
        return len(stamps) >= limit


def _rate_hit(bucket: dict[str, list[float]], key: str, window: float) -> None:
    now = time.monotonic()
    with _rate_lock:
        stamps = _prune_hits(bucket.get(key, []), now, window)
        stamps.append(now)
        bucket[key] = stamps


def _reset_rate_limits() -> None:
    with _rate_lock:
        _login_failures.clear()
        _pdf_rebuilds.clear()


def _owns_call(role: str, call_id: int, share_token: str = "") -> bool:
    """Admin owns any saved call. A mother owns a call only through its share token."""
    if call_id <= 0:
        return False
    if role == "admin":
        return True
    token = (share_token or "").strip()
    if not token:
        return False
    call = db.get_call_by_token(token)
    if call is None:
        return False
    return int(call.get("id") or 0) == int(call_id)


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
    if not (os.environ.get("APP_PASSCODE") or "").strip():
        logger.warning("APP_PASSCODE is unset")
    seed.ensure_seeded()
    yield


app = FastAPI(lifespan=_lifespan)


@app.middleware("http")
async def _require_login(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/calls/by-token/"):
        return await call_next(request)
    if path in _PUBLIC_PATHS or _authorized(request.cookies.get(COOKIE_NAME)):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"ok": False}, status_code=401)
    if path in {"/", "/admin"}:
        target = path
        if request.url.query:
            target = f"{path}?{request.url.query}"
        nxt = quote(target, safe="/")
        return RedirectResponse(url=f"/login?next={nxt}", status_code=302)
    return await call_next(request)


def _page(name: str) -> FileResponse:
    return FileResponse(_STATIC / name, media_type="text/html")


def _clinic_slug(value: str) -> str:
    slug = (value or "").strip().lower()
    return slug if _CLINIC_RE.fullmatch(slug) else ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/login")
def login_page() -> FileResponse:
    return _page("login.html")


@app.get("/static/{asset}")
def static_asset(asset: str) -> FileResponse:
    name = Path(asset).name
    if name != asset or not name or name.startswith("."):
        raise HTTPException(status_code=404, detail="Not found")
    path = (_STATIC / name).resolve()
    try:
        path.relative_to(_STATIC.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    media = _STATIC_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)


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
    ip = _client_ip(request)
    if _rate_limited(_login_failures, ip, _LOGIN_FAILS, _LOGIN_WINDOW_SEC):
        return JSONResponse({"ok": False}, status_code=429)
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
        _rate_hit(_login_failures, ip, _LOGIN_WINDOW_SEC)
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
    if isinstance(engine, str):
        engine = engine.strip().lower()
    # A stored or posted "openai" value is the removed live-voice engine.
    if engine == "openai":
        engine = "gemini"
    if engine != "gemini":
        raise HTTPException(status_code=400, detail="active_engine must be gemini")
    db.set_setting("active_engine", "gemini")
    return {"ok": True, "active_engine": "gemini"}


@app.get("/api/bugs")
def get_bugs(request: Request):
    _require_admin(request)
    return db.list_bug_reports()


@app.post("/api/bugs")
async def post_bug(request: Request):
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > _BUG_BYTE_CAP:
                raise HTTPException(status_code=413, detail="Bug report is too large")
        except ValueError:
            pass
    raw = await request.body()
    if len(raw) > _BUG_BYTE_CAP:
        raise HTTPException(status_code=413, detail="Bug report is too large")
    try:
        body = json.loads(raw) if raw else None
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
    share_token = str(body.get("share_token") or "").strip()
    note = str(body.get("note") or "")[:_BUG_NOTE_CAP]
    bug = db.create_bug_report(
        call_id,
        username or role or "user",
        str(body.get("category") or ""),
        note,
        elapsed,
        extra,
        _owns_call(role, call_id, share_token),
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


def _pdf_response(pdf: bytes, name: str) -> Response:
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def _authorize_mother_pdf(call_id: int, request: Request) -> None:
    """Integer call ids are admin-only. Mothers use the unguessable share token."""
    _require_admin(request)
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    if str(call.get("status") or "") == "active":
        raise HTTPException(status_code=409, detail="Call is still running")


@app.get("/api/calls/{call_id}/mother.pdf")
async def mother_pdf(call_id: int, request: Request):
    _authorize_mother_pdf(call_id, request)
    ready = report_service.ready_pdf(call_id)
    if ready is None:
        raise HTTPException(status_code=409, detail="Report is not ready")
    pdf, name = ready
    return _pdf_response(pdf, name)


def _call_for_token(share_token: str) -> dict:
    call = db.get_call_by_token(share_token)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    if str(call.get("status") or "") == "active":
        raise HTTPException(status_code=409, detail="Call is still running")
    return call


@app.get("/api/calls/by-token/{share_token}/mother.pdf")
async def mother_pdf_by_token(share_token: str):
    call = _call_for_token(share_token)
    ready = report_service.ready_pdf(int(call["id"]))
    if ready is None:
        raise HTTPException(status_code=409, detail="Report is not ready")
    pdf, name = ready
    return _pdf_response(pdf, name)


@app.post("/api/calls/by-token/{share_token}/events")
async def call_event_by_token(share_token: str, request: Request):
    call = db.get_call_by_token(share_token)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    event = str(body.get("event") or "").strip()
    if event not in _SHARE_EVENTS:
        raise HTTPException(status_code=400, detail="Unknown event")
    db.append_call_event(int(call["id"]), event)
    return {"ok": True}


@app.post("/api/calls/{call_id}/review")
async def post_clinician_review(call_id: int, request: Request):
    _require_admin(request)
    if db.get_call(call_id) is None:
        raise HTTPException(status_code=404, detail="Call not found")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    rating = str(body.get("rating") or "").strip()
    note = str(body.get("note") or "")
    try:
        saved = db.save_clinician_review(call_id, rating, note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unknown rating") from exc
    return {"ok": True, "clinician_review": saved}


@app.post("/api/calls/by-token/{share_token}/mother.pdf")
async def rebuild_mother_pdf_by_token(share_token: str):
    call = _call_for_token(share_token)
    if _rate_limited(_pdf_rebuilds, share_token, _PDF_REBUILDS, _PDF_WINDOW_SEC):
        raise HTTPException(status_code=429, detail="Too many report rebuilds")
    _rate_hit(_pdf_rebuilds, share_token, _PDF_WINDOW_SEC)
    try:
        pdf, name = await report_service.mother_pdf(int(call["id"]))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Call not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not build the report") from exc
    return _pdf_response(pdf, name)


@app.post("/api/calls/{call_id}/mother.pdf")
async def rebuild_mother_pdf(call_id: int, request: Request):
    _authorize_mother_pdf(call_id, request)
    try:
        pdf, name = await report_service.mother_pdf(call_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Call not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not build the report") from exc
    return _pdf_response(pdf, name)


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


async def _begin_report(call_id: int, snapshot: dict) -> None:
    """App-owned hook. The live session does not import the report modules."""
    await report_service.begin_answers(call_id, snapshot)


async def _prepare_mother_pdf(call_id: int) -> bool:
    """Build the one PDF for this call. Share stays hidden until this is true."""
    try:
        await report_service.mother_pdf(call_id)
    except Exception:
        return False
    return report_service.ready_pdf(call_id) is not None


def _ws_identity(ws: WebSocket) -> tuple[str, str]:
    role, username = _identity_from_cookies(ws.cookies)
    return role or "user", username


@app.websocket("/ws/gemini")
async def ws_gemini(ws: WebSocket) -> None:
    if not _authorized(ws.cookies.get(COOKIE_NAME)):
        await _reject_ws(ws)
        return
    role, username = _ws_identity(ws)
    clinic = _clinic_slug(ws.query_params.get("clinic") or "")
    await gemini_live.handle_gemini_ws(
        ws,
        caller_name=username,
        user_role=role,
        on_call_completed=_prepare_mother_pdf,
        on_end_accepted=_begin_report,
        clinic_code=clinic,
    )
