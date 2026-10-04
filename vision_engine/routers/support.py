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
    MVE_SUPPORT_SITE         Hashtgerd                    (human room NAME; defaults to hostname)
    MVE_SUPPORT_FINGERPRINT  a3f9c21b4e77                 (clone guard; see below)
    MVE_SUPPORT_ADMINS       @support:chat.virasad.ir     (may run !commands; default = _INVITE)
    MVE_SUPPORT_COMMANDS     1                            (0 disables the listener entirely)
    MVE_SUPPORT_POLL_SEC     15                           (command listener poll interval)
    MVE_SUPPORT_TIMEOUT      15                           (seconds, per HTTP call)

Support-side commands (4.0.384). Rather than making the OPERATOR decide what to
attach — they don't know what's useful, and attaching everything to every message
is noise — support PULLS what it needs by typing a command in the room:

    !help  !status  !diag  !cameras  !logs [n]  !version

A background listener runs from MVE boot, independent of whether anyone has the
Support tab open, so support can pull diagnostics from a site at 3am. It is
strictly READ-ONLY and strictly ALLOW-LISTED: there is no shell, no arbitrary
verb, and nothing that mutates state. Only senders in MVE_SUPPORT_ADMINS are
obeyed, only in this site's own room, and each event is executed at most once.

Identity is the machine's HARDWARE FINGERPRINT (services/machine_id.py), not a
hand-typed name. With no MVE_SUPPORT_ROOM_ALIAS set, the room alias defaults to
`#mve-<fingerprint>`, so a new site needs NO per-site configuration at all — it
resolves or creates its own room on first use. MVE_SUPPORT_SITE stays a purely
cosmetic display name for the room, safe to change at any time.

Clone guard: set MVE_SUPPORT_FINGERPRINT to the fingerprint the credential was
issued for. If the machine reports a different one — i.e. this disk image was
cloned onto other hardware — support REFUSES to connect rather than authenticate
as, and post into the room of, the machine it was cloned from. DMI/SMBIOS is read
from the motherboard, so a disk clone always reports a different value.

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
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, Response

from services.machine_id import machine_fingerprint, fingerprint_sources

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


def _fingerprint() -> str:
    try:
        return machine_fingerprint()
    except Exception as e:
        logger.warning("[support] fingerprint unavailable: %s", e)
        return ""


def clone_mismatch() -> Optional[str]:
    """Non-None when this credential was issued for DIFFERENT hardware.

    Returns a human-readable reason. Guards the golden-image clone case: an image
    copied to a new machine must not keep authenticating as the machine it came
    from and posting into that site's room.
    """
    expected = _env("MVE_SUPPORT_FINGERPRINT")
    if not expected:
        return None                      # guard not armed for this site
    actual = _fingerprint()
    if not actual:
        return None                      # cannot read hardware id — do not lock the site out
    if actual != expected:
        return ("support credential was issued for machine {} but this machine is {} "
                "- refusing to use another site's identity (re-enroll this machine)"
                .format(expected, actual))
    return None


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
        # 4.0.382 - no per-site configuration needed: derive the alias from the
        # hardware fingerprint. Stable for the life of the motherboard, unique
        # across the fleet, and requires nothing typed by an operator.
        fp = _fingerprint()
        if not fp:
            raise SupportError(
                "support chat has no room configured and this machine has no readable "
                "hardware id (set MVE_SUPPORT_ROOM_ID or MVE_SUPPORT_ROOM_ALIAS)", 503)
        alias = "mve-" + fp
        logger.info("[support] no room alias configured; using fingerprint alias #%s", alias)
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
        d["machine"] = _fingerprint()
        d["machine_sources"] = fingerprint_sources()
    except Exception:
        pass
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
    for key in ("version", "machine", "hostname", "camera_count", "current_state", "shipment",
                "pipeline", "cpu_percent", "mem_percent", "disk_percent"):
        if d.get(key) is not None:
            lines.append("{}: {}".format(key, d[key]))
    for cam in (d.get("cameras") or []):
        lines.append("  cam {}: ok={} gain={} {}".format(
            cam.get("id"), cam.get("ok"), cam.get("gain"), cam.get("source")))
    return "\n".join(lines)


# --- blocking workers (always run via asyncio.to_thread) --------------------

def _guard_or_raise():
    mismatch = clone_mismatch()
    if mismatch:
        raise SupportError(mismatch, 409)


def _send_sync(text: str, diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    _guard_or_raise()
    room = _resolve_room()
    body = text if not diagnostics else text + "\n\n" + _format_diagnostics(diagnostics)
    # 4.0.384 - always identify the box in one short line. The operator used to
    # have to tick "attach diagnostics", which asked them to judge something they
    # cannot: they do not know what support needs, and attaching everything to
    # every message buries the actual question. One header line, plus support
    # pulling !diag on demand, is strictly better for both sides.
    try:
        _hdr = "[{} - v{}]".format(_site_name(), _short_version())
        if _hdr not in body:
            body = _hdr + "\n" + body
    except Exception:
        pass
    txn = "mve{}".format(int(time.time() * 1000))
    res = _call("PUT",
                "/_matrix/client/v3/rooms/{}/send/m.room.message/{}".format(_quote(room), txn),
                json_body={"msgtype": "m.text", "body": body})
    return {"ok": True, "event_id": res.get("event_id"), "room_id": room}


def _short_version() -> str:
    try:
        from routers.health import get_current_version
        return get_current_version("") or "?"
    except Exception:
        return "?"


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
    _guard_or_raise()
    room = _resolve_room()
    me = _whoami_cached()
    res = _call("GET", "/_matrix/client/v3/rooms/{}/messages".format(_quote(room)),
                params={"dir": "b", "limit": max(1, min(int(limit), 200))})
    out: List[Dict[str, Any]] = []
    for ev in (res.get("chunk") or []):
        if ev.get("type") != "m.room.message":
            continue
        content = ev.get("content") or {}
        body = content.get("body")
        if not body:
            continue  # redacted event
        sender = ev.get("sender")
        msgtype = content.get("msgtype") or "m.text"
        item = {
            "event_id": ev.get("event_id"),
            "sender": sender,
            "sender_name": (sender or "").lstrip("@").split(":", 1)[0],
            "mine": bool(me and sender == me),
            "body": body,
            "msgtype": msgtype,
            "ts": ev.get("origin_server_ts"),
        }
        # 4.0.383 - attachments. For m.image/m.file/m.video/m.audio the `body` is only
        # the FILENAME; the bytes live at an mxc:// URI that needs the bot's token to
        # fetch. Hand the UI a same-origin proxy URL instead of the mxc, so the browser
        # never sees a Matrix token and authenticated-media servers still work.
        mxc = content.get("url")
        if isinstance(mxc, str) and mxc.startswith("mxc://"):
            item["mxc"] = mxc
            item["media_url"] = "/api/support/media?mxc=" + requests.utils.quote(mxc, safe="")
            info = content.get("info") or {}
            item["mimetype"] = info.get("mimetype")
            item["width"] = info.get("w")
            item["height"] = info.get("h")
            item["size"] = info.get("size")
        out.append(item)
    out.reverse()  # dir=b returns newest-first; the UI renders oldest-first
    return {"ok": True, "room_id": room, "messages": out}


def _media_sync(mxc: str):
    """Fetch an mxc:// attachment as (bytes, content_type).

    Synapse 1.100+ serves media through an AUTHENTICATED endpoint; the legacy
    unauthenticated one is being retired. Try the authenticated route first and
    fall back, so this keeps working on both old and new homeservers.
    """
    _guard_or_raise()
    if not isinstance(mxc, str) or not mxc.startswith("mxc://"):
        raise SupportError("not a media URI", 400)
    rest = mxc[len("mxc://"):]
    if "/" not in rest:
        raise SupportError("malformed media URI", 400)
    server, media_id = rest.split("/", 1)
    if not server or not media_id or "/" in media_id or ".." in media_id:
        raise SupportError("malformed media URI", 400)

    paths = (
        "/_matrix/client/v1/media/download/{}/{}".format(_quote(server), _quote(media_id)),
        "/_matrix/media/v3/download/{}/{}".format(_quote(server), _quote(media_id)),
    )
    headers = {"Authorization": "Bearer " + _token()}
    last = None
    for path in paths:
        try:
            r = _SESSION.get(_homeserver() + path, headers=headers,
                             timeout=_timeout(), stream=False)
        except requests.RequestException as e:
            last = "cannot reach the support server: " + str(e)
            continue
        if r.status_code == 200:
            ctype = r.headers.get("Content-Type") or "application/octet-stream"
            return r.content, ctype
        last = "HTTP {}: {}".format(r.status_code, _err_of(r))
    raise SupportError("could not fetch attachment ({})".format(last or "unknown"), 502)


# --- outbound attachments (operator -> support) -----------------------------
# 4.0.403 — the operator can now send a screenshot / log / exported config INTO
# the support room, mirroring the inbound media path (_media_sync). Upload the
# bytes to the homeserver media repo, then post an m.image/m.file message whose
# `url` is the returned mxc:// — exactly the shape _messages_sync already renders.

def _msgtype_for(ctype: str) -> str:
    c = (ctype or "").lower()
    if c.startswith("image/"):
        return "m.image"
    if c.startswith("video/"):
        return "m.video"
    if c.startswith("audio/"):
        return "m.audio"
    return "m.file"


def _max_upload_bytes() -> int:
    try:
        mb = int(_env("MVE_SUPPORT_MAX_UPLOAD_MB", "25") or 25)
        return max(1, mb) * 1024 * 1024
    except Exception:
        return 25 * 1024 * 1024


def _upload_sync(file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """Upload a file to the homeserver and post it into the room as m.image/m.file."""
    _guard_or_raise()
    if not file_bytes:
        raise SupportError("empty file", 400)
    cap = _max_upload_bytes()
    if len(file_bytes) > cap:
        raise SupportError("file too large (max {} MB)".format(cap // (1024 * 1024)), 413)
    room = _resolve_room()
    fname = (filename or "upload").replace("\n", " ").replace("\r", " ").strip() or "upload"
    ctype = (content_type or "application/octet-stream").split(";", 1)[0].strip() or "application/octet-stream"

    # 1) upload the raw bytes (NOT json) to the media repo; re-login once on a bad token
    def _do_upload(tok: str):
        return _SESSION.post(
            _homeserver() + "/_matrix/media/v3/upload",
            params={"filename": fname},
            data=file_bytes,
            headers={"Authorization": "Bearer " + tok, "Content-Type": ctype},
            timeout=max(_timeout(), 60.0),
        )
    try:
        r = _do_upload(_token())
        if r.status_code in (401, 403) and not _env("MVE_SUPPORT_TOKEN"):
            r = _do_upload(_token(force_new=True))
    except requests.RequestException as e:
        raise SupportError("cannot reach the support server: " + str(e), 502)
    if r.status_code >= 400:
        raise SupportError("media upload failed (HTTP {}): {}".format(r.status_code, _err_of(r)), 502)
    try:
        mxc = (r.json() or {}).get("content_uri")
    except Exception:
        mxc = None
    if not isinstance(mxc, str) or not mxc.startswith("mxc://"):
        raise SupportError("media upload returned no content uri", 502)

    # 2) post the attachment message into the room (same shape the UI renders inbound)
    content = {
        "msgtype": _msgtype_for(ctype),
        "body": fname,
        "url": mxc,
        "info": {"mimetype": ctype, "size": len(file_bytes)},
    }
    txn = "mveup{}".format(int(time.time() * 1000))
    res = _call("PUT",
                "/_matrix/client/v3/rooms/{}/send/m.room.message/{}".format(_quote(room), txn),
                json_body=content)
    return {"ok": True, "event_id": res.get("event_id"), "room_id": room,
            "mxc": mxc, "msgtype": content["msgtype"], "filename": fname}


def _status_sync() -> Dict[str, Any]:
    """Never raises — this is what the tab polls to decide what to render."""
    info: Dict[str, Any] = {
        "configured": is_configured(),
        "homeserver": _homeserver(),
        "site": _site_name(),
    }
    info["fingerprint"] = _fingerprint()
    if not info["configured"]:
        info["connected"] = False
        info["error"] = "not configured"
        return info
    mismatch = clone_mismatch()
    if mismatch:
        info["connected"] = False
        info["cloned"] = True
        info["error"] = mismatch
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


# --- in-memory log ring (feeds !logs) ---------------------------------------
# A logging handler is cheaper and far more portable than shelling out to
# `docker logs`: it needs no docker socket, works identically on every site, and
# cannot be used to read anything but MVE's own log records.

_LOG_RING: List[str] = []
_LOG_RING_MAX = 400
_LOG_RING_LOCK = threading.Lock()


class _RingHandler(logging.Handler):
    def emit(self, record):
        try:
            line = self.format(record)
        except Exception:
            return
        with _LOG_RING_LOCK:
            _LOG_RING.append(line)
            if len(_LOG_RING) > _LOG_RING_MAX:
                del _LOG_RING[:len(_LOG_RING) - _LOG_RING_MAX]


def install_log_ring():
    """Attach the ring handler to the root logger. Idempotent."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, _RingHandler):
            return
    h = _RingHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    h.setLevel(logging.INFO)
    root.addHandler(h)
    logger.info("[support] log ring installed (%d lines)", _LOG_RING_MAX)


def _recent_logs(n: int) -> str:
    with _LOG_RING_LOCK:
        lines = _LOG_RING[-max(1, min(n, 200)):]
    return "\n".join(lines) if lines else "(no log lines buffered yet)"


# --- commands ---------------------------------------------------------------

def _admins() -> List[str]:
    """Who may run commands. Defaults to whoever was invited to the room."""
    raw = _env("MVE_SUPPORT_ADMINS") or _env("MVE_SUPPORT_INVITE")
    return [_full_user_id(u.strip()) for u in raw.split(",") if u.strip()]


def _cmd_help(app_state) -> str:
    return ("MVE support commands (read-only):\n"
            "  !status        health summary\n"
            "  !diag          full diagnostics snapshot\n"
            "  !cameras       per-camera source / ok / gain\n"
            "  !logs [n]      last n log lines (default 50, max 200)\n"
            "  !version       MVE version + machine fingerprint\n"
            "  !help          this list")


def _cmd_status(app_state) -> str:
    d = collect_diagnostics(app_state)
    return ("status: site={} version={} machine={}\n"
            "cameras={} state={} shipment={}\n"
            "cpu={}% mem={}% disk={}%").format(
        d.get("site"), d.get("version"), d.get("machine"),
        d.get("camera_count"), d.get("current_state"), d.get("shipment"),
        d.get("cpu_percent"), d.get("mem_percent"), d.get("disk_percent"))


def _cmd_diag(app_state) -> str:
    return _format_diagnostics(collect_diagnostics(app_state))


def _cmd_cameras(app_state) -> str:
    d = collect_diagnostics(app_state)
    cams = d.get("cameras") or []
    if not cams:
        return "no cameras configured"
    return "\n".join("cam {}: ok={} gain={} {}".format(
        c.get("id"), c.get("ok"), c.get("gain"), c.get("source")) for c in cams)


def _cmd_version(app_state) -> str:
    d = collect_diagnostics(app_state)
    return "version={} machine={} site={}\nsources={}".format(
        d.get("version"), d.get("machine"), d.get("site"), d.get("machine_sources"))


_COMMANDS = {
    "help": _cmd_help,
    "status": _cmd_status,
    "diag": _cmd_diag,
    "diagnostics": _cmd_diag,
    "cameras": _cmd_cameras,
    "version": _cmd_version,
}


def handle_command(body: str, app_state) -> Optional[str]:
    """Map one '!verb args' line to its reply, or None if it is not a command."""
    text = (body or "").strip()
    if not text.startswith("!"):
        return None
    parts = text[1:].split()
    if not parts:
        return None
    verb = parts[0].lower()
    if verb in ("logs", "log"):
        n = 50
        if len(parts) > 1:
            try:
                n = int(parts[1])
            except ValueError:
                pass
        return "last {} log lines:\n{}".format(min(max(n, 1), 200), _recent_logs(n))
    fn = _COMMANDS.get(verb)
    if not fn:
        return "unknown command '!{}'. Try !help".format(verb)
    try:
        return fn(app_state)
    except Exception as e:
        return "command failed: {}".format(_scrub(e))


# --- background command listener --------------------------------------------
# Runs from MVE boot regardless of whether the Support tab is open, so support can
# pull diagnostics from an unattended site. Independent of the UI poller.

_CMD_STATE: Dict[str, Any] = {"since_ts": 0, "done": set(), "task": None}


def _poll_sec() -> float:
    try:
        v = float(_env("MVE_SUPPORT_POLL_SEC", "15") or 15)
        return v if v >= 5 else 15.0
    except Exception:
        return 15.0


def _commands_enabled() -> bool:
    return _env("MVE_SUPPORT_COMMANDS", "1") not in ("0", "false", "no", "off")


def _scan_commands_sync(app_state) -> int:
    """One pass: find unhandled commands from an allowed sender and answer them."""
    if not is_configured() or clone_mismatch():
        return 0
    admins = set(_admins())
    if not admins:
        return 0
    me = _whoami_cached()
    room = _resolve_room()
    res = _call("GET", "/_matrix/client/v3/rooms/{}/messages".format(_quote(room)),
                params={"dir": "b", "limit": 20})
    events = list(res.get("chunk") or [])
    events.reverse()   # oldest-first so replies come out in order

    # First pass ever: don't answer backlog, just mark the high-water mark.
    if not _CMD_STATE["since_ts"]:
        for ev in events:
            _CMD_STATE["since_ts"] = max(_CMD_STATE["since_ts"], ev.get("origin_server_ts") or 0)
        return 0

    handled = 0
    for ev in events:
        ts = ev.get("origin_server_ts") or 0
        eid = ev.get("event_id")
        if ts <= _CMD_STATE["since_ts"] or eid in _CMD_STATE["done"]:
            continue
        _CMD_STATE["since_ts"] = max(_CMD_STATE["since_ts"], ts)
        if ev.get("type") != "m.room.message":
            continue
        sender = ev.get("sender")
        if sender == me or sender not in admins:
            continue   # never obey ourselves; never obey a non-admin
        body = (ev.get("content") or {}).get("body") or ""
        reply = handle_command(body, app_state)
        if reply is None:
            continue
        _CMD_STATE["done"].add(eid)
        if len(_CMD_STATE["done"]) > 500:
            _CMD_STATE["done"] = set(list(_CMD_STATE["done"])[-250:])
        logger.info("[support] command from %s: %s", sender, body.strip()[:60])
        try:
            _send_sync(reply[:8000], None)
            handled += 1
        except SupportError as e:
            logger.warning("[support] could not answer command: %s", e.message)
    return handled


async def _command_loop(app):
    logger.info("[support] command listener started (every %.0fs)", _poll_sec())
    while True:
        try:
            await asyncio.sleep(_poll_sec())
            if not _commands_enabled():
                continue
            await asyncio.to_thread(_scan_commands_sync, app.state)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A listener failure must never take the app down or stop the loop.
            logger.warning("[support] command listener pass failed: %s", _scrub(e))


def start_command_listener(app):
    """Called from main.py's startup event. No-op when support isn't configured."""
    if not _commands_enabled():
        logger.info("[support] command listener disabled by env")
        return
    if not is_configured():
        logger.info("[support] command listener not started (support not configured)")
        return
    if _CMD_STATE.get("task"):
        return
    install_log_ring()
    _CMD_STATE["task"] = asyncio.create_task(_command_loop(app))


def stop_command_listener():
    task = _CMD_STATE.get("task")
    if task:
        task.cancel()
        _CMD_STATE["task"] = None


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


@router.get("/api/support/media")
async def support_media(mxc: str = ""):
    """Stream one attachment from the room, authenticated server-side.

    The browser gets a same-origin URL and never handles a Matrix access token.
    """
    try:
        data, ctype = await asyncio.to_thread(_media_sync, mxc)
        return Response(content=data, media_type=ctype,
                        headers={"Cache-Control": "private, max-age=3600"})
    except SupportError as e:
        return JSONResponse(content={"ok": False, "error": e.message}, status_code=e.status)
    except Exception as e:
        logger.warning("[support] media fetch failed: %s", _scrub(e))
        return JSONResponse(content={"ok": False, "error": _scrub(e)}, status_code=502)


@router.post("/api/support/upload")
async def support_upload(file: UploadFile = File(...)):
    """Operator sends a screenshot / log / exported config into the support room.

    multipart/form-data with field `file`. Uploaded to the homeserver media repo,
    then posted as m.image (images) or m.file (everything else).
    """
    try:
        data = await file.read()
    except Exception as e:
        logger.warning("[support] upload read failed: %s", _scrub(e))
        return JSONResponse(content={"ok": False, "error": "could not read the file"}, status_code=400)
    try:
        res = await asyncio.to_thread(
            _upload_sync, data, file.filename or "upload",
            file.content_type or "application/octet-stream")
        return JSONResponse(content=res)
    except SupportError as e:
        return JSONResponse(content={"ok": False, "error": e.message}, status_code=e.status)
    except Exception as e:
        logger.warning("[support] upload failed: %s", _scrub(e))
        return JSONResponse(content={"ok": False, "error": _scrub(e)}, status_code=502)


@router.get("/api/support/diagnostics")
async def support_diagnostics(request: Request):
    """Preview exactly what `include_diagnostics` would attach."""
    try:
        d = collect_diagnostics(request.app.state)
        return JSONResponse(content={"ok": True, "diagnostics": d, "text": _format_diagnostics(d)})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": _scrub(e)}, status_code=500)
