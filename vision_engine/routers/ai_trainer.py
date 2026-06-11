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
    }


@router.get("/api/ai_trainer/config")
async def get_ai_trainer_config():
    """Return the URL + task_id, plus a boolean so the UI knows whether a
    key was previously saved (we never return the key itself)."""
    cfg = _ai_trainer_cfg()
    return JSONResponse(content={
        "url": cfg["url"],
        "task_id": cfg["task_id"],
        "has_api_key": bool(cfg["api_key"]),
    })


@router.post("/api/ai_trainer/config")
async def set_ai_trainer_config(payload: dict):
    """Persist url / task_id / api_key in service_config.ai_trainer.

    The api_key field is optional in the payload — if omitted, the previously
    stored key is preserved. Send an empty string explicitly to clear it.
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
        svc["ai_trainer"] = cur
        save_service_config(svc)
        return JSONResponse(content={
            "success": True,
            "url": cur.get("url"),
            "task_id": cur.get("task_id"),
            "has_api_key": bool(cur.get("api_key")),
        })
    except Exception as e:
        logger.error(f"ai_trainer config save failed: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


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
    files = {"files": (raw_path.name, open(str(raw_path), "rb"), "image/jpeg")}
    data = {
        **form_extra,
        "source_shipment": str(body.get("shipment") or ""),
        "source_camera": str(body.get("camera") or ""),
        "class_name": str(body.get("class_name") or ""),
    }
    headers = {}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    try:
        r = requests.post(target_url, files=files, data=data, headers=headers, timeout=15)
        ok = (200 <= r.status_code < 300)
        return JSONResponse(content={
            "success": ok,
            "status_code": r.status_code,
            "trainer_response": (r.text[:500] if r.text else ""),
            "target_url": target_url,
            "task_id": task_id,
        }, status_code=(200 if ok else 502))
    except requests.RequestException as e:
        logger.warning(f"ai_trainer upload failed: {e}")
        return JSONResponse(
            content={"success": False, "error": f"trainer unreachable: {e}", "target_url": target_url},
            status_code=502,
        )
    finally:
        try:
            files["file"][1].close()
        except Exception:
            pass
