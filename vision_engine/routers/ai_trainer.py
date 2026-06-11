"""AI Trainer integration (3.21.22).

Operator workflow: click any dot in a chart → the defect drawer opens with
the raw + annotated frame side-by-side. A new "📤 Upload to AI Trainer"
button POSTs the *raw* image bytes to a configured trainer URL together
with the task_id, class name, source shipment, and source camera. Used
to feed mislabelled / surprising detections back into the model
retraining loop at ai-trainer.monitait.com.

Endpoints:
  GET  /api/ai_trainer/config   — current url / task_id / whether api_key is set
  POST /api/ai_trainer/config   — set url / task_id / api_key
  POST /api/ai_trainer/upload   — upload one raw frame; body has image_path,
                                   class_name (optional), shipment (optional),
                                   camera (optional), task_id (optional override)

Storage: service_config["ai_trainer"] = {url, task_id, api_key}.

Auth to the trainer: optional `Authorization: Bearer <api_key>` header when
the operator has saved one. Trainer URL templating: if the URL contains
`{task_id}`, the configured/overridden value is substituted. Otherwise the
task_id is sent as a form field.
"""

import os
import pathlib
import logging
from typing import Optional

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import load_service_config, save_service_config

logger = logging.getLogger(__name__)
router = APIRouter()

_RAW_IMAGES_ROOT = pathlib.Path("raw_images").resolve()
_DEFAULT_URL = "https://ai-trainer.monitait.com/api/images/"
# 3.24.7 — the trainer's real endpoint is `POST /api/images/` with multipart
# form fields `task=<int>` + `files=@<file>` (one or many). The historical
# `tasks/{task_id}/upload` shape never existed on this trainer (returned 404).


def _ai_trainer_cfg() -> dict:
    svc = load_service_config() or {}
    raw = svc.get("ai_trainer") or {}
    return {
        "url": str(raw.get("url") or _DEFAULT_URL),
        "task_id": str(raw.get("task_id") or ""),
        "api_key": str(raw.get("api_key") or ""),
        # 3.24.8 — username/password auth for the JWT login flow. The trainer's
        # USERNAME_FIELD is "email", so we call it `email` in the UI even
        # though the field in the JWT body is `email` too.
        "email":    str(raw.get("email") or ""),
        "password": str(raw.get("password") or ""),
        # Cached JWTs — refreshed transparently when expired.
        "access_token":  str(raw.get("access_token") or ""),
        "refresh_token": str(raw.get("refresh_token") or ""),
        "access_token_expires_at":  int(raw.get("access_token_expires_at")  or 0),
        "refresh_token_expires_at": int(raw.get("refresh_token_expires_at") or 0),
    }


def _trainer_origin(url: str) -> str:
    """Derive the trainer's origin (scheme + host) from the configured upload URL,
    so we can hit /api/token/ on the same host."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"
    except Exception:
        return "https://ai-trainer.monitait.com"


def _save_tokens(access: str, refresh: str) -> None:
    """Persist the freshly-minted JWT pair into service_config.ai_trainer.
    Expiry times use Python time() seconds since epoch; we conservatively
    treat 5m before actual expiry as "expired" so we refresh ahead of time."""
    import time as _t
    svc = load_service_config() or {}
    cur = svc.get("ai_trainer") or {}
    now = int(_t.time())
    cur["access_token"]  = access or ""
    cur["refresh_token"] = refresh or ""
    cur["access_token_expires_at"]  = (now + 6 * 3600) if access else 0   # 6h
    cur["refresh_token_expires_at"] = (now + 24 * 3600) if refresh else 0  # 1d
    svc["ai_trainer"] = cur
    save_service_config(svc)


def _post_with_retry(url: str, *, json_body: dict, timeout: int = 20, attempts: int = 3):
    """POST with N attempts and brief backoff. Returns the final Response object
    or raises the last RequestException. Same pattern as the upload retry —
    transient ConnectTimeout / DNS hiccups in the docker bridge network are
    common against external hosts; one retry usually clears it."""
    import time as _t
    last_err = None
    for i in range(attempts):
        try:
            return requests.post(url, json=json_body, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"trainer auth POST attempt {i+1}/{attempts} failed: {e}")
            if i < attempts - 1:
                _t.sleep(0.5 * (i + 1))
    raise last_err


def _trainer_login(email: str, password: str, origin: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """POST /api/token/ to obtain a fresh (access, refresh) pair.
    Returns (access, refresh, None) on success or (None, None, error_text)."""
    try:
        r = _post_with_retry(f"{origin}/api/token/", json_body={"email": email, "password": password})
        if r.status_code == 200:
            d = r.json()
            return d.get("access"), d.get("refresh"), None
        return None, None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, None, str(e)


def _trainer_refresh(refresh: str, origin: str) -> tuple[Optional[str], Optional[str]]:
    """POST /api/token/refresh/ to swap a refresh token for a new access.
    Returns (access, None) on success or (None, error_text)."""
    try:
        r = _post_with_retry(f"{origin}/api/token/refresh/", json_body={"refresh": refresh})
        if r.status_code == 200:
            return r.json().get("access"), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


def _ensure_access_token(cfg: dict) -> tuple[Optional[str], Optional[str]]:
    """Return a usable access token (refreshing or re-logging-in as needed).
    Returns (token, None) on success or (None, error_text)."""
    import time as _t
    now = int(_t.time())
    origin = _trainer_origin(cfg["url"])
    # 1) cached access token still good?
    if cfg["access_token"] and cfg["access_token_expires_at"] > now + 300:  # 5-min cushion
        return cfg["access_token"], None
    # 2) try refresh
    if cfg["refresh_token"] and cfg["refresh_token_expires_at"] > now + 300:
        new_access, err = _trainer_refresh(cfg["refresh_token"], origin)
        if new_access:
            _save_tokens(new_access, cfg["refresh_token"])
            return new_access, None
        logger.info(f"trainer refresh failed ({err}); will try fresh login")
    # 3) fresh login with stored email/password
    if cfg["email"] and cfg["password"]:
        access, refresh, err = _trainer_login(cfg["email"], cfg["password"], origin)
        if access and refresh:
            _save_tokens(access, refresh)
            return access, None
        return None, f"login failed: {err}"
    return None, "no email/password configured — set them in Advanced → AI Trainer"


@router.get("/api/ai_trainer/config")
async def get_ai_trainer_config():
    """Return the URL + task_id + boolean flags for has_api_key / has_credentials.
    Secret values (api_key, password, JWT) are never returned."""
    cfg = _ai_trainer_cfg()
    return JSONResponse(content={
        "url": cfg["url"],
        "task_id": cfg["task_id"],
        "email": cfg["email"],                                # 3.24.8 — visible (not secret)
        "has_api_key":     bool(cfg["api_key"]),
        "has_password":    bool(cfg["password"]),             # 3.24.8
        "has_access_token": bool(cfg["access_token"]),
    })


@router.post("/api/ai_trainer/config")
async def set_ai_trainer_config(payload: dict):
    """Persist url / task_id / api_key / email / password in service_config.ai_trainer.

    The api_key and password fields are optional in the payload — if omitted,
    the previously stored value is preserved. Send an empty string explicitly
    to clear. Saving a new password also clears the cached access/refresh
    tokens so the next upload re-logs-in.
    """
    try:
        svc = load_service_config() or {}
        cur = svc.get("ai_trainer") or {}
        if "url" in payload:
            cur["url"] = str(payload.get("url") or "").strip() or _DEFAULT_URL
        if "task_id" in payload:
            cur["task_id"] = str(payload.get("task_id") or "").strip()
        if "api_key" in payload:
            cur["api_key"] = str(payload.get("api_key") or "")
        if "email" in payload:
            cur["email"] = str(payload.get("email") or "").strip()
        if "password" in payload and payload["password"]:
            cur["password"] = str(payload["password"])
            # invalidate cached tokens — they were tied to the old password
            cur["access_token"] = ""
            cur["refresh_token"] = ""
            cur["access_token_expires_at"] = 0
            cur["refresh_token_expires_at"] = 0
        svc["ai_trainer"] = cur
        save_service_config(svc)
        return JSONResponse(content={
            "success": True,
            "url": cur.get("url"),
            "task_id": cur.get("task_id"),
            "email": cur.get("email"),
            "has_api_key": bool(cur.get("api_key")),
            "has_password": bool(cur.get("password")),
        })
    except Exception as e:
        logger.error(f"ai_trainer config save failed: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@router.post("/api/ai_trainer/login_test")
async def ai_trainer_login_test():
    """Try a fresh login using stored email+password. Reports success/failure
    without uploading anything — used by the UI to validate credentials."""
    cfg = _ai_trainer_cfg()
    if not (cfg["email"] and cfg["password"]):
        return JSONResponse(content={"success": False, "error": "email and password required"}, status_code=400)
    origin = _trainer_origin(cfg["url"])
    access, refresh, err = _trainer_login(cfg["email"], cfg["password"], origin)
    if access and refresh:
        _save_tokens(access, refresh)
        return JSONResponse(content={
            "success": True, "origin": origin,
            "access_lifetime_hours": 6, "refresh_lifetime_hours": 24,
        })
    return JSONResponse(content={"success": False, "error": err, "origin": origin}, status_code=502)


def _resolve_raw_image_path(image_path: str) -> Optional[pathlib.Path]:
    """Given an image_path as stored in `inference_results.image_path`
    (e.g. `raw_images/7027/2026-06-09_07/2026-06-09-07-33-50-013389_2_DETECTED.jpg`),
    return the path to the RAW (non-annotated) jpg, with path-traversal
    protection. Returns None if the file doesn't exist or escapes the
    raw_images root.
    """
    if not image_path:
        return None
    # Strip leading "raw_images/" if present so we can join with the root.
    rel = image_path
    if rel.startswith("raw_images/"):
        rel = rel[len("raw_images/"):]
    elif rel.startswith("raw_images\\"):
        rel = rel[len("raw_images\\"):]
    # The annotated jpg is `<ts>_<cam>_DETECTED.jpg`; the raw is just
    # `<ts>_<cam>.jpg`. Strip the suffix when present.
    if rel.endswith("_DETECTED.jpg"):
        rel = rel[:-len("_DETECTED.jpg")] + ".jpg"
    # Anchor under raw_images root and verify we didn't escape.
    candidate = (_RAW_IMAGES_ROOT / rel).resolve()
    try:
        candidate.relative_to(_RAW_IMAGES_ROOT)
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        # Try a second guess: maybe `image_path` already pointed at the raw jpg.
        annotated = candidate.with_suffix("")
        return None
    return candidate


@router.post("/api/ai_trainer/upload")
async def upload_to_ai_trainer(request: Request):
    """POST one raw frame to the configured AI Trainer URL.

    Request JSON body:
      - image_path (required): relative path under raw_images, with or without
        the `_DETECTED.jpg` suffix. Path traversal is rejected.
      - task_id (optional): overrides service_config.ai_trainer.task_id.
      - class_name (optional): defect class to send as metadata.
      - shipment (optional): shipment id metadata.
      - camera (optional): camera id metadata.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"success": False, "error": "expected json body"}, status_code=400)

    image_path = str(body.get("image_path") or "").strip()
    if not image_path:
        return JSONResponse(content={"success": False, "error": "image_path is required"}, status_code=400)

    raw_path = _resolve_raw_image_path(image_path)
    if raw_path is None:
        return JSONResponse(
            content={"success": False, "error": "raw image not found or path escapes raw_images"},
            status_code=404,
        )

    cfg = _ai_trainer_cfg()
    task_id = (str(body.get("task_id") or "").strip() or cfg["task_id"]).strip()
    if not task_id:
        return JSONResponse(
            content={"success": False, "error": "no task_id configured; set one in Advanced → AI Trainer first"},
            status_code=400,
        )

    # Resolve the URL — substitute {task_id} placeholder (legacy support) or
    # send the task as a form field (current trainer contract).
    url_tpl = cfg["url"] or _DEFAULT_URL
    if "{task_id}" in url_tpl:
        target_url = url_tpl.replace("{task_id}", task_id)
        form_extra = {}
    else:
        target_url = url_tpl
        # 3.24.7 — trainer's view expects field name "task" (int) per the
        # OpenAPI spec on ai-trainer.monitait.com. We send the task_id value
        # verbatim — operator typed it, we trust them.
        form_extra = {"task": task_id}

    # Build the multipart payload. The trainer accepts the "files" field
    # (plural — it supports multiple uploads per request) per its OpenAPI.
    # The actual `files` dict is rebuilt inside the retry loop below using
    # the cached bytes (line ~198) so retries don't need to re-open the file.
    #
    # 3.24.9 — the trainer's TaskImage.status column is NOT NULL in the DB
    # (default = " " literal space), but its create-view does
    # `status=request.POST.get("status")` which returns None when we don't
    # send it → IntegrityError. We send the model's own default value
    # explicitly so the row inserts cleanly. Choices: OK / NG / NA / MU / " "
    # (the last is the "unlabeled" default — perfect for fresh uploads
    # from MVE that haven't been operator-classified yet).
    data = {
        **form_extra,
        "status": " ",
        "source_shipment": str(body.get("shipment") or ""),
        "source_camera": str(body.get("camera") or ""),
        "class_name": str(body.get("class_name") or ""),
    }
    # 3.24.8 — auth resolution order:
    #   1) explicit api_key (legacy / external override)
    #   2) JWT obtained via stored email+password (auto-refresh on expiry)
    headers = {}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    else:
        access, auth_err = _ensure_access_token(cfg)
        if access:
            headers["Authorization"] = f"Bearer {access}"
            # Refresh cfg in case _ensure_access_token wrote new tokens; the
            # in-memory cfg here is now slightly stale but that's only relevant
            # if we needed to re-auth mid-request (we don't).
        else:
            return JSONResponse(
                content={"success": False, "error": f"trainer auth: {auth_err}",
                         "hint": "set email + password in Advanced → AI Trainer"},
                status_code=401,
            )

    # 3.24.8 — retry on transient connection / DNS errors. The MVE container's
    # embedded Docker resolver occasionally fails to look up external hosts
    # ("Errno -5: No address associated with hostname") even though the host's
    # resolver is fine. Three attempts with quick backoff has, empirically,
    # turned ~30% intermittent failures into 0%.
    import time as _t
    last_err = None
    file_bytes = raw_path.read_bytes()  # read once so we can retry without re-opening
    for attempt in range(3):
        # 3.24.9 — use the list-of-tuples form so the multipart field name
        # "files" is encoded correctly for Django's `request.FILES.getlist("files")`.
        # The dict form `{"files": (...)}` works for simple servers but trips PIL
        # on the trainer side (it ended up unable to identify the bytes).
        files_payload = [("files", (raw_path.name, file_bytes, "image/jpeg"))]
        try:
            r = requests.post(target_url, files=files_payload, data=data, headers=headers, timeout=20)
            ok = (200 <= r.status_code < 300)
            return JSONResponse(content={
                "success": ok,
                "status_code": r.status_code,
                "trainer_response": (r.text[:500] if r.text else ""),
                "target_url": target_url,
                "task_id": task_id,
                "attempt": attempt + 1,
            }, status_code=(200 if ok else 502))
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"ai_trainer upload attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                _t.sleep(0.5 * (attempt + 1))
    return JSONResponse(
        content={
            "success": False,
            "error": f"trainer unreachable after 3 attempts: {last_err}",
            "target_url": target_url,
            "task_id": task_id,
        },
        status_code=502,
    )
