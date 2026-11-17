"""Moldable V3 HTTP and WebSocket entrypoint (`intake/app.py`)."""

from __future__ import annotations

import base64, hashlib, hmac, json, logging, os, re, secrets, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from intake import db, reports  # noqa: E402
from intake.pack import list_pack_ids, load_pack  # noqa: E402

log = logging.getLogger("intake.app")
COOKIE, STATIC = "shishu_auth", Path(__file__).resolve().parent / "static"
PAGES = {"/": "call.html", "/login": "login.html", "/admin": "admin.html"}
PUBLIC = {"/health", "/login", "/api/login", "/api/logout", "/api/pack"}
OPEN, CLINIC = ("/static/", "/api/pack-asset/", "/api/calls/by-token/"), ("/api/calls", "/api/audio", "/admin")
TYPES, ASSETS = {".css": "text/css", ".js": "text/javascript"}, {"logo": "image/png", "consent_pdf": "application/pdf"}
LIMITS = {"login": (15, 60.0), "pdf": (5, 60.0), "share": (20, 60.0)}
MAX_AGE, BUG_CAP, NOTE_CAP = 60 * 60 * 24 * 14, 512 * 1024, 2000
HITS: dict[str, dict[str, list[float]]] = {name: {} for name in LIMITS}
LOCK = threading.Lock()

def _sign(body: str) -> str: return hmac.new(db.get_session_secret().encode(), body.encode(), hashlib.sha256).hexdigest()

def _session(role: str, username: str) -> str:
    body = f"v1.{role}.{int(time.time()) + MAX_AGE}.{base64.urlsafe_b64encode(username.encode()).decode().rstrip('=')}"
    return f"{body}.{_sign(body)}"
def _auth(conn: Request | WebSocket) -> tuple[str, str]:
    parts = (conn.cookies.get(COOKIE) or "").split(".")
    if len(parts) != 5 or parts[0] != "v1" or parts[1] not in {"user", "admin"}: return "", ""
    try: name, fresh = base64.urlsafe_b64decode(parts[3] + "=" * (-len(parts[3]) % 4)).decode(), float(parts[2]) > time.time()
    except Exception: return "", ""  # noqa: BLE001
    return (parts[1], name or "Guest") if fresh and hmac.compare_digest(parts[4], _sign(".".join(parts[:4]))) else ("", "")
def _over(name: str, key: str, *, count: bool = True) -> bool:
    limit, window = LIMITS[name]
    now, bucket = time.monotonic(), HITS[name]
    with LOCK:
        recent = [at for at in bucket.get(key, []) if now - at < window]
        over = len(recent) >= limit
        bucket[key] = [*recent, now] if (count and not over) else recent
    return over
def _pack(pack_id: str | None = None):
    try: return load_pack(pack_id)
    except LookupError as exc: raise HTTPException(404, "Unknown pack") from exc
def _call(call_id: int | None = None, token: str = "", *, live: bool = False) -> dict:
    call = db.get_call_by_token(token) if token else db.get_call(call_id or 0)
    if call is None: raise HTTPException(404, "Call not found")
    if not live and call.get("status") == "active": raise HTTPException(409, "Call is still running")
    return call
def _clinic_code(raw: str | None) -> str:
    """A clinic code only as a plain slug, else the configured default — never a path or free text."""
    for value in ((raw or "").strip().lower(), (os.environ.get("DEFAULT_CLINIC_CODE") or "").strip().lower()):
        if re.fullmatch(r"[a-z0-9_-]{1,40}", value): return value
    return ""
def _inside(root: Path, *parts: str) -> Path:
    base, path = root.resolve(), root.joinpath(*parts).resolve()
    if base not in path.parents: raise HTTPException(404, "Not found")
    return path
async def _prepare_pdf(call_id: int) -> bool:
    try: return await reports.ensure_patient_pdf(call_id) is not None and reports.ready_pdf(call_id) is not None
    except Exception:  # noqa: BLE001
        log.exception("Failed to prepare PDF for call %s", call_id)
        return False
@asynccontextmanager
async def _lifespan(_app: FastAPI):
    reports.ensure_seeded()
    yield
app = FastAPI(lifespan=_lifespan)

@app.middleware("http")
async def _gate(request: Request, call_next):
    path, (role, username) = request.url.path, _auth(request)
    request.state.role, request.state.username = role, username
    admin_only = request.method == "GET" if path == "/api/bugs" else path.startswith(CLINIC)
    if path in PUBLIC or path.startswith(OPEN) or (role and (role == "admin" or not admin_only)): return await call_next(request)
    if path.startswith("/api/"): return JSONResponse({"ok": False}, status_code=403 if role else 401)
    return RedirectResponse(f"/login?next={quote(f'{path}?{request.url.query}' if request.url.query else path, safe='/')}", status_code=302)
app.get("/health")(lambda: {"status": "ok", "build": reports.build_info()["tag"]})
for _route, _page in PAGES.items(): app.get(_route)(lambda html=_page: FileResponse(STATIC / html, media_type="text/html"))

@app.get("/static/{asset}")
def static_asset(asset: str) -> FileResponse:
    path = _inside(STATIC, asset)
    if path.suffix.lower() not in TYPES or not path.is_file(): raise HTTPException(404, "Not found")
    return FileResponse(path, media_type=TYPES[path.suffix.lower()])
@app.post("/api/login")
@app.post("/api/logout")
async def auth(request: Request):
    if request.url.path == "/api/logout":
        res = JSONResponse({"ok": True})
        res.delete_cookie(COOKIE, path="/")
        return res
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or getattr(request.client, "host", "")
    if _over("login", ip, count=False): return JSONResponse({"ok": False}, status_code=429)
    try: said = await request.json()
    except ValueError: said = {}
    said = said if isinstance(said, dict) else {}
    pw, user = str(said.get("passcode") or "").strip(), str(said.get("username") or "").strip()[:80] or "Guest"
    role = next((n for n, k in (("admin", "ADMIN_PASSCODE"), ("user", "APP_PASSCODE")) if (exp := (os.environ.get(k) or "").strip()) and pw and secrets.compare_digest(pw, exp)), "")
    if not role:
        _over("login", ip)
        return JSONResponse({"ok": False}, status_code=401)
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip() or request.url.scheme
    res = JSONResponse({"ok": True, "role": role, "username": user})
    res.set_cookie(COOKIE, _session(role, user), max_age=MAX_AGE, path="/", httponly=True, samesite="lax", secure=proto == "https")
    return res
@app.get("/api/pack")
def get_pack(request: Request, pack: str | None = None):
    cfg, (role, username) = _pack(pack), _auth(request)
    return {"id": cfg.id, "title": cfg.meta.get("title", cfg.id), "subtitle_native": cfg.meta.get("subtitle_native", ""), "ui": cfg.meta.get("ui") or {},
            "assets_url": {k: f"/api/pack-asset/{k}?pack={cfg.id}" for k in ASSETS}, "role": role, "username": username,
            "available_packs": [{"id": p, "title": load_pack(p).meta.get("title", p)} for p in list_pack_ids()]}
@app.get("/api/pack-asset/{asset_key}")
def pack_asset(asset_key: str, pack: str | None = None) -> FileResponse:
    cfg = _pack(pack)
    name = str((cfg.meta.get("assets") or {}).get(asset_key) or "").strip()
    if not name or asset_key not in ASSETS or not (path := _inside(cfg.root, "assets", name)).is_file(): raise HTTPException(404, "Unknown asset")
    return FileResponse(path, media_type=ASSETS[asset_key])
app.get("/api/calls")(lambda pack=None: db.list_calls(pack or ""))
app.get("/api/calls/{call_id}")(lambda call_id: _call(call_id, live=True))
app.get("/api/bugs")(lambda: db.list_bug_reports())

@app.api_route("/api/calls/{call_id}/patient.pdf", methods=["GET", "POST"])
@app.api_route("/api/calls/by-token/{share_token}/patient.pdf", methods=["GET", "POST"])
async def patient_pdf(request: Request, call_id: int | None = None, share_token: str | None = None):
    call = _call(call_id) if share_token is None else _call(token=share_token)
    if request.method == "POST" and share_token and _over("pdf", str(call["share_token"])): raise HTTPException(429, "Too many report rebuilds")
    try: built = await reports.ensure_patient_pdf(call["id"]) if request.method == "POST" else reports.ready_pdf(call["id"])
    except Exception as exc: raise HTTPException(500, "Could not build the report") from exc
    if built is None: raise HTTPException(409, "Report is not ready")
    return Response(built[0], media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{built[1]}"'})
@app.post("/api/calls/by-token/{share_token}/events")
async def call_event(share_token: str, request: Request):
    call = _call(token=share_token)
    if _over("share", str(call["share_token"])): raise HTTPException(429, "Too many events")
    try: body = await request.json()
    except ValueError: body = {}
    if not isinstance(body, dict) or str(body.get("event") or "") != "share_clicked": raise HTTPException(400, "Unknown event")
    db.append_call_event(int(call["id"]), "share_clicked")
    return {"ok": True}
@app.get("/api/audio/{call_id}")
def get_audio(call_id: int):
    raw = str(_call(call_id, live=True).get("audio_path") or "").strip()
    if not raw or not (path := _inside(Path(db.DATA_DIR) / "audio", Path(raw).name)).is_file(): raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
@app.post("/api/bugs")
async def post_bug(request: Request):
    if len(raw := await request.body()) > BUG_CAP: raise HTTPException(413, "Bug report is too large")
    try: rep = json.loads(raw)
    except ValueError: rep = None
    if not isinstance(rep, dict): raise HTTPException(400, "Expected a bug report object")
    try: cid, elapsed = int(rep.get("call_id") or 0), int(rep.get("call_elapsed_sec") or 0)
    except (TypeError, ValueError): cid = elapsed = 0
    tok = str(rep.get("share_token") or "")
    owned = request.state.role == "admin" or (cid > 0 and bool(tok) and (c := db.get_call_by_token(tok)) is not None and int(c["id"]) == cid)
    client = rep.get("client_state") if isinstance(rep.get("client_state"), dict) else {}
    return {"ok": True, "bug": db.create_bug_report(cid, request.state.username or "user", str(rep.get("category") or ""), str(rep.get("note") or "")[:NOTE_CAP], elapsed, client, owned)}
@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    role, username = _auth(ws)
    try: pack = load_pack(ws.query_params.get("pack"))
    except LookupError: pack = None
    if not role or pack is None:
        await ws.accept()
        await ws.close(code=1008)
        return
    await ws.accept()
    from intake import live
    await live.handle_live_ws(ws, pack=pack, caller_name=username, user_role=role, clinic_code=_clinic_code(ws.query_params.get("clinic")),
                              resume_token=str(ws.query_params.get("resume") or "").strip(), on_call_completed=_prepare_pdf)
