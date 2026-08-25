"""Auth + audit endpoints (4.0.357). Pairs with services/security.py.

Login issues a bearer token (also set as an httponly cookie). Everything here is
inert for existing installs until an admin enables RBAC — the endpoints simply
manage users, the audit log, and the RBAC toggle.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse

from services import security as sec

logger = logging.getLogger(__name__)
router = APIRouter()


def _actor(request: Request):
    return sec.resolve(sec.token_from_request(request))


def _require(request: Request, min_role: str):
    """Returns (username, role) or raises a 403 JSONResponse-worthy error.
    Enforced only when RBAC is enabled; otherwise everyone is treated as admin."""
    user, role = _actor(request)
    if not sec.rbac_enabled():
        return user, "admin"
    if sec.ROLE_RANK.get(role, 0) < sec.ROLE_RANK.get(min_role, 0):
        raise PermissionError(min_role)
    return user, role


@router.post("/api/auth/login")
async def login(request: Request):
    body = await request.json()
    try:
        res = sec.login(str(body.get("username", "")), str(body.get("password", "")))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=401)
    resp = JSONResponse({"username": res["username"], "role": res["role"], "token": res["token"]})
    resp.set_cookie("mve_token", res["token"], max_age=sec.TOKEN_TTL_SEC,
                    httponly=True, samesite="lax")
    return resp


@router.post("/api/auth/logout")
async def logout(request: Request):
    sec.logout(sec.token_from_request(request) or "")
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("mve_token")
    return resp


@router.get("/api/auth/me")
def me(request: Request) -> Dict[str, Any]:
    user, role = _actor(request)
    return {"username": user, "role": role, "rbac_enabled": sec.rbac_enabled(),
            "authenticated": user is not None}


@router.get("/api/auth/config")
def auth_config(request: Request) -> Dict[str, Any]:
    return {"rbac_enabled": sec.rbac_enabled()}


@router.post("/api/auth/config")
async def set_auth_config(request: Request):
    try:
        _require(request, "admin")
    except PermissionError:
        return JSONResponse({"error": "admin role required"}, status_code=403)
    body = await request.json()
    try:
        sec.set_rbac_enabled(bool(body.get("rbac_enabled")))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"rbac_enabled": sec.rbac_enabled()}


@router.get("/api/auth/users")
def users(request: Request):
    try:
        _require(request, "admin")
    except PermissionError:
        return JSONResponse({"error": "admin role required"}, status_code=403)
    return {"users": sec.list_users(), "roles": list(sec.ROLE_RANK.keys())}


@router.post("/api/auth/users")
async def add_user(request: Request):
    try:
        _require(request, "admin")
    except PermissionError:
        return JSONResponse({"error": "admin role required"}, status_code=403)
    body = await request.json()
    try:
        return sec.create_user(str(body.get("username", "")), str(body.get("password", "")),
                               str(body.get("role", "operator")))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.delete("/api/auth/users/{username}")
def del_user(request: Request, username: str):
    try:
        _require(request, "admin")
    except PermissionError:
        return JSONResponse({"error": "admin role required"}, status_code=403)
    sec.delete_user(username)
    return {"ok": True}


@router.get("/api/audit")
def audit(request: Request, limit: int = 200):
    try:
        _require(request, "engineer")
    except PermissionError:
        return JSONResponse({"error": "engineer role required"}, status_code=403)
    return {"entries": sec.get_audit(limit)}
