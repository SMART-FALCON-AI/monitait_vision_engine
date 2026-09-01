"""Support router for MVE — operator-facing support chat backed by Matrix.

4.0.381. The operator asks a question from the Support tab and a human answers
from a normal Matrix client (Element on chat.virasad.ir). MVE renders its OWN
chat UI and talks to the homeserver over the client-server API; the operator
NEVER logs in to Matrix. Each site authenticates as one bot account and all of
its conversation lives in one room, so support staff get real history,
notifications and multi-agent handover for free.

Wired in vision_engine/main.py before commands_router (the catch-all):

    from routers.support import router as support_router
    app.include_router(support_router)

Configuration (all env, nothing hard-coded per site):

    MVE_SUPPORT_HOMESERVER   https://chat.virasad.ir      (default)
    MVE_SUPPORT_USER         @mve-hashtgerd:virasad.ir    (or a bare localpart)
    MVE_SUPPORT_PASSWORD     ...                          (password login)
    MVE_SUPPORT_TOKEN        ...                          (static access token; wins over password)
    MVE_SUPPORT_ROOM_ID      !abc:virasad.ir              (explicit room; skips resolve/create)
    MVE_SUPPORT_ROOM_ALIAS   #mve-hashtgerd:virasad.ir    (resolved, else CREATED with this alias)
    MVE_SUPPORT_INVITE       @ali:virasad.ir,@sara:...    (invited when the room is created)
    MVE_SUPPORT_SITE         Hashtgerd                    (room name / label; defaults to hostname)
    MVE_SUPPORT_TIMEOUT      15                           (seconds, per HTTP call)

Design contract (do not weaken) — mirrors routers/knowledge.py:
  - Every call is timeout-bounded and returns a structured error instead of
    raising. The homeserver being down must DEGRADE the Support tab, never
    break the dashboard or the capture loop.
  - Blocking `requests` calls run off the event loop (asyncio.to_thread) so a
    slow homeserver can't freeze every MVE endpoint.
  - Nothing here is required for MVE to boot: with no env set,
    /api/support/status reports configured=false and the tab renders a
    "not configured" panel instead of erroring.
  - Secrets (password, access token) are NEVER returned by any endpoint and are
    scrubbed out of every error string that can reach a client or the log.
"""
import asyncio
import logging
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_HOMESERVER = "https://chat.virasad.ir"

# cv2.CAP_PROP_GAIN — hard-coded so diagnostics never import cv2 just to read a key.
_CAP_PROP_GAIN = 14


def _env(name: str, default: str = "") -> str:
    try:
        return (os.environ.get(name) or default).strip()
    except Exception:
        return default


def _timeout() -> float:
    """Per-call HTTP timeout. A bad env value must never crash the import."""
    try:
        v = float(_env("MVE_SUPPORT_TIMEOUT", "15") or 15)
        return v if v > 0 else 15.0
    except Exception:
        return 15.0


def _homeserver() -> str:
    return (_env("MVE_SUPPORT_HOMESERVER", DEFAULT_HOMESERVER) or DEFAULT_HOMESERVER).rstrip("/")


def _server_name() -> str:
    """Host part of the homeserver URL, used to complete bare user/room ids."""
    return _homeserver().split("://", 1)[-1].split("/", 1)[0]


def _site_name() -> str:
    name = _env("MVE_SUPPORT_SITE")
    if name:
        return name
    try:
        return socket.gethostname() or "MVE"
    except Exception:
        return "MVE"


# --- process-wide session state --------------------------------------------
# token:   access token (from env, or minted by password login)
# room_id: resolved once, then reused for the process lifetime
_STATE: Dict[str, Optional[str]] = {"token": None, "user_id": None, "room_id": None}
_STATE_LOCK = threading.Lock()
_SESSION = requests.Session()


def _scrub(text: Any) -> str:
    """Strip anything secret out of a string before it reaches a client or log."""
    out = str(text)
    for var in ("MVE_SUPPORT_PASSWORD", "MVE_SUPPORT_TOKEN"):
        secret = _env(var)
        if secret and len(secret) >= 4:
            out = out.replace(secret, "***")
    with _STATE_LOCK:
        tok = _STATE.get("token")
    if tok and len(tok) >= 8:
        out = out.replace(tok, "***")
    return out


def is_configured() -> bool:
    """True when there is enough env to even attempt a connection."""
    return bool(
        _env("MVE_SUPPORT_TOKEN")
        or (_env("MVE_SUPPORT_USER") and _env("MVE_SUPPORT_PASSWORD"))
    )


class SupportError(Exception):
    """Carries a client-safe message; never contains a secret."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = _scrub(message)
        self.status = status


def _quote(v: str) -> str:
    return requests.utils.quote(v, safe="")


def _full_user_id(user: str) -> str:
    """Accept `bot`, `@bot`, or a full `@bot:server` and return a full MXID."""
    u = (user or "").strip()
    if not u:
        return u
    if not u.startswith("@"):
        u = "@" + u
    if ":" not in u[1:]:
        u = u + ":" + _server_name()
    return u


def _err_of(resp) -> str:
    try:
        j = resp.json()
        return str(j.get("error") or j.get("errcode") or resp.text[:200])
    except Exception:
        return (resp.text or "")[:200]


def _login() -> str:
    """Mint an access token via m.login.password. Returns the token."""
    user = _env("MVE_SUPPORT_USER")
    password = _env("MVE_SUPPORT_PASSWORD")
    if not user or not password:
        raise SupportError("support chat is not configured (no token, and no user/password)", 503)
    body = {
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": _full_user_id(user)},
        "password": password,
        "initial_device_display_name": "MVE " + _site_name(),
    }
    try:
        r = _SESSION.post(_homeserver() + "/_matrix/client/v3/login", json=body, timeout=_timeout())
    except requests.RequestException as e:
        raise SupportError("cannot reach the support server: " + str(e), 502)
    if r.status_code != 200:
        raise SupportError(
            "support login rejected (HTTP {}): {}".format(r.status_code, _err_of(r)), 502)
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise SupportError("support login returned no access token", 502)
    with _STATE_LOCK:
        _STATE["token"] = token
        _STATE["user_id"] = data.get("user_id")
    logger.info("[support] logged in to %s as %s", _homeserver(), data.get("user_id"))
    return token


def _token(force_new: bool = False) -> str:
    """Current access token — an env token wins; else the cached login, else login."""
    env_token = _env("MVE_SUPPORT_TOKEN")
    if env_token:
        return env_token
    if not force_new:
        with _STATE_LOCK:
            tok = _STATE.get("token")
        if tok:
            return tok
    return _login()


def _call(method: str, path: str, *, params=None, json_body=None, retry_auth: bool = True):
    """One authenticated client-server API call. Re-logins once on a bad token."""
    headers = {"Authorization": "Bearer " + _token()}
    try:
        r = _SESSION.request(method, _homeserver() + path, params=params, json=json_body,
                             headers=headers, timeout=_timeout())
    except requests.RequestException as e:
        raise SupportError("cannot reach the support server: " + str(e), 502)
    # A token minted by a previous process (or expired) → mint a fresh one, once.
    if r.status_code in (401, 403) and retry_auth and not _env("MVE_SUPPORT_TOKEN"):
        try:
            errcode = r.json().get("errcode", "")
        except Exception:
            errcode = ""
        if r.status_code == 401 or errcode in ("M_UNKNOWN_TOKEN", "M_MISSING_TOKEN"):
            logger.info("[support] access token rejected, re-logging in")
            _token(force_new=True)
            return _call(method, path, params=params, json_body=json_body, retry_auth=False)
    if r.status_code >= 400:
        raise SupportError(
            "support server error (HTTP {}): {}".format(r.status_code, _err_of(r)), 502)
    try:
        return r.json()
    except Exception:
        return {}


def _resolve_room() -> str:
    """Room for this site: explicit id → resolve alias → create with that alias.

    Cached for the process lifetime; a wrong/stale cache self-heals on restart.
    """
    with _STATE_LOCK:
        rid = _STATE.get("room_id")
    if rid:
        return rid

    explicit = _env("MVE_SUPPORT_ROOM_ID")
    if explicit:
        with _STATE_LOCK:
            _STATE["room_id"] = explicit
        return explicit

    alias = _env("MVE_SUPPORT_ROOM_ALIAS")
    if not alias:
        raise SupportError(
            "support chat has no room configured "
            "(set MVE_SUPPORT_ROOM_ID or MVE_SUPPORT_ROOM_ALIAS)", 503)
    if not alias.startswith("#"):
        alias = "#" + alias
    if ":" not in alias[1:]:
        alias = alias + ":" + _server_name()

    # 1) the room already exists → use it, and make sure we are joined
    try:
        found = _call("GET", "/_matrix/client/v3/directory/room/" + _quote(alias))
        rid = found.get("room_id")
        if rid:
            try:
                _call("POST", "/_matrix/client/v3/join/" + _quote(rid), json_body={})
            except SupportError:
                pass  # already a member — join is idempotent-ish, failure here is fine
            with _STATE_LOCK:
                _STATE["room_id"] = rid
            logger.info("[support] using existing room %s (%s)", rid, alias)
            return rid
    except SupportError as e:
        logger.info("[support] alias %s not resolvable (%s) — creating the room", alias, e.message)

    # 2) create it, inviting whoever env names as support staff
    invites = [u.strip() for u in _env("MVE_SUPPORT_INVITE").split(",") if u.strip()]
    body = {
        "visibility": "private",
        "room_alias_name": alias[1:].split(":", 1)[0],
        "name": "MVE Support — " + _site_name(),
        "topic": ("Support channel for MVE site " + _site_name()
                  + ". Messages here reach the operator's Support tab."),
        "preset": "private_chat",
        "invite": [_full_user_id(u) for u in invites],
    }
    created = _call("POST", "/_matrix/client/v3/createRoom", json_body=body)
    rid = created.get("room_id")
    if not rid:
        raise SupportError("could not create the support room", 502)
    with _STATE_LOCK:
        _STATE["room_id"] = rid
    logger.info("[support] created room %s (%s), invited %s", rid, alias, invites)
    return rid


# --- diagnostics ------------------------------------------------------------

def collect_diagnostics(app_state) -> Dict[str, Any]:
    """A snapshot support staff would otherwise have to ask for one field at a
    time. Every lookup is individually guarded — a diagnostics failure must
    never stop the operator's message from being sent."""
    d: Dict[str, Any] = {"site": _site_name()}
    try:
        from routers.health import get_current_version
        d["version"] = get_current_version("")
    except Exception:
        pass
    try:
        d["hostname"] = socket.gethostname()
    except Exception:
        pass
    try:
        watcher = getattr(app_state, "watcher_instance", None)
        cams = getattr(watcher, "cameras", {}) or {}
        rows = []
        for cid in sorted(cams, key=lambda k: int(k) if str(k).isdigit() else 0):
            cam = cams[cid]
            rows.append({
                "id": cid,
                "source": getattr(cam, "source", None),
                "ok": bool(getattr(cam, "success", False)),
                "gain": (getattr(cam, "_saved_props", {}) or {}).get(_CAP_PROP_GAIN),
            })
        d["cameras"] = rows
        d["camera_count"] = len(rows)
    except Exception:
        pass
    try:
        from config import load_service_config
        svc = load_service_config() or {}
        d["current_state"] = svc.get("current_state_name")
        shipment = svc.get("current_shipment")
        d["shipment"] = shipment.get("id") if isinstance(shipment, dict) else shipment
        pc = svc.get("pipeline_config") or {}
        d["pipeline"] = pc.get("current_pipeline_name") or pc.get("current")
    except Exception:
        pass
    try:
        import psutil
        d["cpu_percent"] = psutil.cpu_percent(interval=None)
        d["mem_percent"] = psutil.virtual_memory().percent
        d["disk_percent"] = psutil.disk_usage("/").percent
    except Exception:
        pass
    return d


def _format_diagnostics(d: Dict[str, Any]) -> str:
    lines = ["— MVE diagnostics (" + str(d.get("site", "?")) + ") —"]
    for key in ("version", "hostname", "camera_count", "current_state", "shipment",
                "pipeline", "cpu_percent", "mem_percent", "disk_percent"):
        if d.get(key) is not None:
            lines.append("{}: {}".format(key, d[key]))
    for cam in (d.get("cameras") or []):
        lines.append("  cam {}: ok={} gain={} {}".format(
            cam.get("id"), cam.get("ok"), cam.get("gain"), cam.get("source")))
    return "\n".join(lines)


# --- blocking workers (always run via asyncio.to_thread) --------------------

def _send_sync(text: str, diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    room = _resolve_room()
    body = text if not diagnostics else text + "\n\n" + _format_diagnostics(diagnostics)
    txn = "mve{}".format(int(time.time() * 1000))
    res = _call("PUT",
                "/_matrix/client/v3/rooms/{}/send/m.room.message/{}".format(_quote(room), txn),
                json_body={"msgtype": "m.text", "body": body})
    return {"ok": True, "event_id": res.get("event_id"), "room_id": room}


def _whoami_cached() -> Optional[str]:
    with _STATE_LOCK:
        me = _STATE.get("user_id")
    if me:
        return me
    try:
        me = _call("GET", "/_matrix/client/v3/account/whoami").get("user_id")
    except SupportError:
        return None
    with _STATE_LOCK:
        _STATE["user_id"] = me
    return me


def _messages_sync(limit: int) -> Dict[str, Any]:
    room = _resolve_room()
    me = _whoami_cached()
    res = _call("GET", "/_matrix/client/v3/rooms/{}/messages".format(_quote(room)),
                params={"dir": "b", "limit": max(1, min(int(limit), 200))})
    out: List[Dict[str, Any]] = []
    for ev in (res.get("chunk") or []):
        if ev.get("type") != "m.room.message":
            continue
        body = (ev.get("content") or {}).get("body")
        if not body:
            continue  # redacted, or a non-text message type
        sender = ev.get("sender")
        out.append({
            "event_id": ev.get("event_id"),
            "sender": sender,
            "sender_name": (sender or "").lstrip("@").split(":", 1)[0],
            "mine": bool(me and sender == me),
            "body": body,
            "ts": ev.get("origin_server_ts"),
        })
    out.reverse()  # dir=b returns newest-first; the UI renders oldest-first
    return {"ok": True, "room_id": room, "messages": out}


def _status_sync() -> Dict[str, Any]:
    """Never raises — this is what the tab polls to decide what to render."""
    info: Dict[str, Any] = {
        "configured": is_configured(),
        "homeserver": _homeserver(),
        "site": _site_name(),
    }
    if not info["configured"]:
        info["connected"] = False
        info["error"] = "not configured"
        return info
    try:
        who = _call("GET", "/_matrix/client/v3/account/whoami")
        info["user_id"] = who.get("user_id")
        with _STATE_LOCK:
            _STATE["user_id"] = who.get("user_id")
        info["room_id"] = _resolve_room()
        info["connected"] = True
    except SupportError as e:
        info["connected"] = False
        info["error"] = e.message
    except Exception as e:
        info["connected"] = False
        info["error"] = _scrub(e)
    return info


# --- routes -----------------------------------------------------------------

@router.get("/api/support/status")
async def support_status():
    """Is support chat usable, and as whom/where. Safe to poll; never 5xx."""
    try:
        return JSONResponse(content=await asyncio.to_thread(_status_sync))
    except Exception as e:
        logger.warning("[support] status failed: %s", _scrub(e))
        return JSONResponse(content={"configured": is_configured(), "connected": False,
                                     "error": _scrub(e)})


@router.get("/api/support/messages")
async def support_messages(limit: int = 50):
    try:
        return JSONResponse(content=await asyncio.to_thread(_messages_sync, limit))
    except SupportError as e:
        return JSONResponse(content={"ok": False, "error": e.message}, status_code=e.status)
    except Exception as e:
        logger.warning("[support] messages failed: %s", _scrub(e))
        return JSONResponse(content={"ok": False, "error": _scrub(e)}, status_code=502)


@router.post("/api/support/send")
async def support_send(request: Request):
    """Send the operator's message.

    Body: {text, include_diagnostics?, context?}
      include_diagnostics — appends the snapshot from /api/support/diagnostics
      context             — prepended verbatim; used by the AI tab's
                            "Ask a human" escalation to carry the transcript
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse(content={"ok": False, "error": "empty message"}, status_code=400)
    context = (body.get("context") or "").strip()
    if context:
        text = context[:4000] + "\n\n" + text
    if len(text) > 8000:
        text = text[:8000] + " …"
    diags = None
    if body.get("include_diagnostics"):
        try:
            diags = collect_diagnostics(request.app.state)
        except Exception as e:
            logger.warning("[support] diagnostics collection failed: %s", _scrub(e))
    try:
        return JSONResponse(content=await asyncio.to_thread(_send_sync, text, diags))
    except SupportError as e:
        return JSONResponse(content={"ok": False, "error": e.message}, status_code=e.status)
    except Exception as e:
        logger.warning("[support] send failed: %s", _scrub(e))
        return JSONResponse(content={"ok": False, "error": _scrub(e)}, status_code=502)


@router.get("/api/support/diagnostics")
async def support_diagnostics(request: Request):
    """Preview exactly what `include_diagnostics` would attach."""
    try:
        d = collect_diagnostics(request.app.state)
        return JSONResponse(content={"ok": True, "diagnostics": d, "text": _format_diagnostics(d)})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": _scrub(e)}, status_code=500)
