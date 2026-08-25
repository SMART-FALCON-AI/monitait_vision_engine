"""Audit log + role-based access control (RBAC) for MVE — 4.0.357.

Two capabilities, both SAFE to ship on a running line:

  * AUDIT LOG — always on. Every mutating request to a protected path
    (procedures, pipelines, camera config, timeline, ejector, config) is
    recorded with actor/role/method/path/time. Recording is best-effort and
    wrapped so it can NEVER break a request.

  * RBAC — OPT-IN, default OFF. When `rbac_enabled` is false (the default and the
    behaviour of every existing install) nothing is enforced — the line runs
    exactly as before. When an admin turns it on, mutating requests to protected
    paths require a role, resolved from a bearer token / cookie. Enforcement
    fail-OPEN on any internal error, so a bug here can never lock the plant out.

Storage is the same TimescaleDB the rest of MVE uses (mve_users / mve_auth_tokens
/ mve_audit_log), self-bootstrapping so no migration step is needed. A default
`admin`/`admin` user is seeded so enabling RBAC never locks everyone out; change
it immediately.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from services.db import get_db_connection, release_db_connection

logger = logging.getLogger(__name__)

ROLE_RANK = {"viewer": 0, "operator": 1, "engineer": 2, "admin": 3}
DEFAULT_ROLE = "viewer"
TOKEN_TTL_SEC = 12 * 3600

# path-prefix -> minimum role to MUTATE it (POST/PUT/PATCH/DELETE). Anything not
# listed but under /api needs at least 'operator'. Read (GET) is never gated.
_PROTECTED: List[Tuple[str, str]] = [
    ("/api/procedures", "engineer"),
    ("/api/timeline_config", "engineer"),
    ("/api/pipelines/activate", "engineer"),
    ("/api/models/activate-weights", "engineer"),
    ("/api/cameras/config", "engineer"),
    ("/api/camera/", "engineer"),          # /api/camera/{id}/config|restart
    ("/api/config", "engineer"),
    ("/api/color_config", "engineer"),
    ("/api/audio_settings", "operator"),
    ("/api/auth/users", "admin"),
    ("/api/auth/config", "admin"),
    ("/api/audit", "engineer"),            # viewing the log (GET) — see required_role_for
]

_schema_ready = False


# --------------------------------------------------------------------------- #
# tiny DB helpers (tuple cursors — MVE's pool hands back raw connections)
# --------------------------------------------------------------------------- #
def _exec(sql: str, params: tuple = (), fetch: Optional[str] = None):
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("db unavailable")
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        out = cur.fetchone() if fetch == "one" else cur.fetchall() if fetch == "all" else None
        conn.commit()
        cur.close()
        return out
    finally:
        release_db_connection(conn)


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_users (
            username    TEXT PRIMARY KEY,
            role        TEXT NOT NULL DEFAULT 'operator',
            salt        TEXT NOT NULL,
            pw_hash     TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );""")
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_auth_tokens (
            token       TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            expires_at  DOUBLE PRECISION NOT NULL
        );""")
    _exec("""
        CREATE TABLE IF NOT EXISTS mve_audit_log (
            id          BIGSERIAL PRIMARY KEY,
            ts          TIMESTAMPTZ DEFAULT NOW(),
            actor       TEXT,
            role        TEXT,
            method      TEXT,
            path        TEXT,
            status      INTEGER,
            detail      TEXT
        );""")
    # seed a default admin so enabling RBAC can never lock everyone out
    row = _exec("SELECT COUNT(*) FROM mve_users WHERE role='admin'", fetch="one")
    if not row or int(row[0] or 0) == 0:
        _create_user("admin", "admin", "admin")
        logger.warning("security: seeded default admin/admin — CHANGE THIS PASSWORD")
    _schema_ready = True


# --------------------------------------------------------------------------- #
# passwords / users
# --------------------------------------------------------------------------- #
def _hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex()


def _create_user(username: str, password: str, role: str = "operator") -> None:
    salt = secrets.token_hex(16)
    _exec("INSERT INTO mve_users (username, role, salt, pw_hash) VALUES (%s,%s,%s,%s) "
          "ON CONFLICT (username) DO UPDATE SET role=EXCLUDED.role, salt=EXCLUDED.salt, "
          "pw_hash=EXCLUDED.pw_hash",
          (username, role if role in ROLE_RANK else "operator", salt, _hash(password, salt)))


def create_user(username: str, password: str, role: str = "operator") -> Dict[str, Any]:
    _ensure_schema()
    if not username or not password:
        raise ValueError("username and password required")
    _create_user(username.strip(), password, role)
    return {"username": username.strip(), "role": role}


def list_users() -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("SELECT username, role, created_at FROM mve_users ORDER BY username", fetch="all") or []
    return [{"username": r[0], "role": r[1], "created_at": str(r[2])} for r in rows]


def delete_user(username: str) -> None:
    _ensure_schema()
    _exec("DELETE FROM mve_users WHERE username=%s", (username,))
    _exec("DELETE FROM mve_auth_tokens WHERE username=%s", (username,))


# --------------------------------------------------------------------------- #
# login / tokens
# --------------------------------------------------------------------------- #
def login(username: str, password: str) -> Dict[str, Any]:
    _ensure_schema()
    row = _exec("SELECT role, salt, pw_hash FROM mve_users WHERE username=%s", (username,), fetch="one")
    if not row or _hash(password, row[1]) != row[2]:
        raise ValueError("invalid credentials")
    token = secrets.token_urlsafe(32)
    _exec("INSERT INTO mve_auth_tokens (token, username, expires_at) VALUES (%s,%s,%s)",
          (token, username, time.time() + TOKEN_TTL_SEC))
    return {"token": token, "username": username, "role": row[0]}


def logout(token: str) -> None:
    try:
        _exec("DELETE FROM mve_auth_tokens WHERE token=%s", (token,))
    except Exception:
        pass


def resolve(token: Optional[str]) -> Tuple[Optional[str], str]:
    """(username, role) for a token, or (None, DEFAULT_ROLE) if absent/expired."""
    if not token:
        return None, DEFAULT_ROLE
    try:
        row = _exec("SELECT u.username, u.role, t.expires_at FROM mve_auth_tokens t "
                    "JOIN mve_users u ON u.username=t.username WHERE t.token=%s", (token,), fetch="one")
        if not row or float(row[2]) < time.time():
            return None, DEFAULT_ROLE
        return row[0], row[1]
    except Exception:
        return None, DEFAULT_ROLE


# --------------------------------------------------------------------------- #
# RBAC toggle (opt-in) + path policy
# --------------------------------------------------------------------------- #
_rbac_cache = {"on": None, "ts": 0.0}


def rbac_enabled() -> bool:
    """True only if an admin has explicitly turned RBAC on. Default False → the
    line behaves exactly as before. Cached 5s. Env override MVE_RBAC_ENABLED=1."""
    if os.environ.get("MVE_RBAC_ENABLED", "").lower() in ("1", "true", "yes"):
        return True
    now = time.time()
    if _rbac_cache["on"] is not None and now - _rbac_cache["ts"] < 5:
        return _rbac_cache["on"]
    on = False
    try:
        from config import load_data_file
        on = bool((load_data_file().get("service_config", {}) or {}).get("rbac_enabled", False))
    except Exception:
        on = False
    _rbac_cache.update(on=on, ts=now)
    return on


def set_rbac_enabled(on: bool) -> None:
    try:
        from config import load_data_file, save_data_file
        data = load_data_file()
        data.setdefault("service_config", {})["rbac_enabled"] = bool(on)
        save_data_file(data)
        _rbac_cache.update(on=bool(on), ts=time.time())
    except Exception as e:
        raise RuntimeError(f"could not persist rbac_enabled: {e}")


def required_role_for(method: str, path: str) -> Optional[str]:
    """Minimum role to perform `method path`, or None if not gated.
    Reads (GET/HEAD/OPTIONS) are never gated, except the audit log view."""
    if path.startswith("/api/audit"):
        return "engineer"                       # even GET of the audit log is gated
    if method.upper() in ("GET", "HEAD", "OPTIONS"):
        return None
    for prefix, role in _PROTECTED:
        if path.startswith(prefix):
            return role
    if path.startswith("/api/"):
        return "operator"                       # any other mutating /api call
    return None


def is_mutating_protected(method: str, path: str) -> bool:
    return required_role_for(method, path) is not None and (
        method.upper() not in ("GET", "HEAD", "OPTIONS") or path.startswith("/api/audit"))


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def audit_record(actor: Optional[str], role: str, method: str, path: str,
                 status: Optional[int] = None, detail: str = "") -> None:
    """Best-effort — must never raise into the request path."""
    try:
        _ensure_schema()
        _exec("INSERT INTO mve_audit_log (actor, role, method, path, status, detail) "
              "VALUES (%s,%s,%s,%s,%s,%s)",
              (actor or "-", role, method, path[:400], status, (detail or "")[:2000]))
    except Exception as e:
        logger.debug("audit_record skipped: %s", e)


def get_audit(limit: int = 200) -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("SELECT ts, actor, role, method, path, status FROM mve_audit_log "
                 "ORDER BY id DESC LIMIT %s", (int(limit),), fetch="all") or []
    return [{"ts": str(r[0]), "actor": r[1], "role": r[2], "method": r[3],
             "path": r[4], "status": r[5]} for r in rows]


def token_from_request(request) -> Optional[str]:
    """Bearer header or `mve_token` cookie."""
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get("mve_token")
