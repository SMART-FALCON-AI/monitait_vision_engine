"""MVE-side plumbing for the anomaly-inference worker.

Direction of flow: **MVE → worker** (always outbound; matches how YOLO + math
already work). This module owns:

  1. POST /api/anomaly/build-baseline
     Body: {mode, phase, value?, window_pct?, max_frames?}
     - Resolves the raw JPG file set from raw_images/<shipment>/<hour>/
       based on the operator's baseline mode + phase (same picker the
       colour heatmap uses on the Charts tab).
     - Multipart-POSTs those JPGs to the anomaly worker's /set-baseline.
     - Persists {mode, phase, value, n_frames, built_at} in
       service_config.anomaly_baseline so the next MVE boot can replay it
       against the worker (self-healing if the worker's RAM is lost).

  2. GET /api/anomaly/baseline
     Returns the persisted anomaly-baseline metadata + a live health
     probe of the worker.

The unified-reference-frames-pool design (see MEMORY): each of the four
colour-baseline modes maps onto anomaly like this —

    camera         → last N frames (across the visible window) per phase
    shipment_start → first ~60 seconds of the active shipment
    reference_frame→ frames within ± window_pct% of the picked encoder/time
    target         → COLOUR-ONLY (target is a synthetic L*a*b* triplet,
                     no raw frames exist to feed the anomaly worker with)
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import load_service_config, save_service_config

logger = logging.getLogger(__name__)
router = APIRouter()

# Anomaly worker default URL. Overridable via env if the compose service
# name ever changes. The `/set-baseline` and `/detect` endpoints share the
# same base so we derive both from one env var.
ANOMALY_WORKER_BASE = os.environ.get(
    "ANOMALY_WORKER_BASE",
    "http://anomaly_inference:4444/v1/anomaly-detection/anomaly_v1",
)

RAW_IMAGES_ROOT = Path("raw_images")

# Filename pattern the watcher writes:
#   raw_images/<shipment>/YYYY-MM-DD_HH/YYYY-MM-DD-HH-MM-SS-nnnnnn_p<phase>_<cam>.jpg
_FN_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d+)_p(\d+)_(\d+)\.jpg$"
)

# Cap on frames POSTed to the worker per baseline build. The worker itself
# truncates at 500; we pick a smaller number here to keep multipart uploads
# under a couple of hundred MB and the memory bank aligned with typical
# industrial baselines (25-60 frames is plenty for pixel-stat).
DEFAULT_MAX_FRAMES = 50


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def _parse_filename(name: str) -> Optional[Dict[str, Any]]:
    """Extract timestamp / phase / camera from a raw-image filename.

    Returns None if the name doesn't match the watcher's convention (older
    frames from before the phase-suffix change, or one-off files someone
    dropped in). Skipped rather than crashed on.
    """
    m = _FN_RE.match(name)
    if not m:
        return None
    y, mo, d, h, mi, s, us, ph, cam = m.groups()
    try:
        ts = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s),
                      int(us[:6]) if len(us) >= 6 else int(us))
    except ValueError:
        return None
    return {"ts": ts, "phase": ph, "cam": int(cam)}


def _list_frames(shipment: str, hours_back: int = 6) -> List[Tuple[Path, Dict[str, Any]]]:
    """List (path, meta) for JPGs under raw_images/<shipment>/ in the last
    N hours, in reverse-chronological order.

    We scan hour-chunk directories rather than the whole shipment tree to
    keep this fast even for a multi-day shipment (raw_images can hold
    ~10k JPGs per hour). Six hours back is enough for camera / reference
    modes; shipment_start mode explicitly overrides by walking from
    shipment-directory mtime.
    """
    root = RAW_IMAGES_ROOT / shipment
    if not root.is_dir():
        return []
    now = datetime.now()
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for h in range(hours_back + 1):
        hour_dir = root / (now - timedelta(hours=h)).strftime("%Y-%m-%d_%H")
        if not hour_dir.is_dir():
            continue
        for entry in os.scandir(hour_dir):
            if not entry.is_file() or not entry.name.endswith(".jpg"):
                continue
            meta = _parse_filename(entry.name)
            if meta is None:
                continue
            out.append((Path(entry.path), meta))
    out.sort(key=lambda p: p[1]["ts"], reverse=True)
    return out


def _select_frames_camera_mode(
    shipment: str, phase: Optional[str], max_frames: int
) -> List[Path]:
    """`camera` mode: newest frames within the last 6 hours.

    We take a mix across whichever cameras are present rather than
    forcing per-cam quotas — the anomaly worker builds ONE global
    baseline right now, so a mix is what we want.
    """
    frames = _list_frames(shipment, hours_back=6)
    if phase not in (None, "", "all"):
        frames = [(p, m) for p, m in frames if m["phase"] == str(phase)]
    return [p for p, _ in frames[:max_frames]]


def _select_frames_shipment_start(
    shipment: str, phase: Optional[str], max_frames: int
) -> List[Path]:
    """`shipment_start` mode: the first ~60s of frames after the shipment
    directory was created. Uses filename timestamps rather than filesystem
    mtime so out-of-order writes (rare, but possible on a busy disk) don't
    mis-classify frames as "start"."""
    root = RAW_IMAGES_ROOT / shipment
    if not root.is_dir():
        return []
    # Walk all hour dirs and take the earliest frames overall.
    everything: List[Tuple[Path, Dict[str, Any]]] = []
    try:
        for hour_dir in sorted(root.iterdir()):
            if not hour_dir.is_dir():
                continue
            for entry in os.scandir(hour_dir):
                if not entry.is_file() or not entry.name.endswith(".jpg"):
                    continue
                meta = _parse_filename(entry.name)
                if meta is None:
                    continue
                everything.append((Path(entry.path), meta))
    except OSError as e:
        logger.warning(f"shipment_start scan failed: {e}")
        return []
    if not everything:
        return []
    if phase not in (None, "", "all"):
        everything = [(p, m) for p, m in everything if m["phase"] == str(phase)]
    everything.sort(key=lambda pm: pm[1]["ts"])
    if not everything:
        return []
    t0 = everything[0][1]["ts"]
    window_end = t0 + timedelta(seconds=60)
    within = [p for p, m in everything if m["ts"] <= window_end]
    return within[:max_frames]


def _select_frames_reference(
    shipment: str,
    phase: Optional[str],
    value: float,
    axis: str,
    window_pct: float,
    max_frames: int,
) -> List[Path]:
    """`reference_frame` mode: frames within ± window_pct% of the picked
    value on the chart. `axis` is 'time' or 'encoder'.

    We only have timestamps in filenames — no encoder value. For axis='time'
    we filter directly. For axis='encoder' we DON'T have per-file encoder
    info, so we fall back to filtering by the wall-clock range of frames
    whose encoder value in the DB matches ± window. That's a DB read
    though, so keep V1 simple: for encoder axis, we approximate by taking
    the newest N frames as if the user picked "recent" (mirrors the
    behaviour of camera mode) and log a warning. Encoder-precise
    selection is a follow-up once we plumb encoder into the raw filename.
    """
    if axis == "time":
        # `value` is a ms-epoch (matches the chart's x-axis unit for time).
        pivot = datetime.fromtimestamp(float(value) / 1000.0)
        # Get everything from a reasonably wide window then narrow.
        frames = _list_frames(shipment, hours_back=24)
        if not frames:
            return []
        if phase not in (None, "", "all"):
            frames = [(p, m) for p, m in frames if m["phase"] == str(phase)]
        # Compute half-window in seconds. We take window_pct% of the
        # span of frames we found.
        if not frames:
            return []
        earliest = min(m["ts"] for _, m in frames)
        latest = max(m["ts"] for _, m in frames)
        span_s = max(1.0, (latest - earliest).total_seconds())
        half = timedelta(seconds=span_s * float(window_pct) / 2.0)
        keep = [p for p, m in frames if abs((m["ts"] - pivot).total_seconds())
                <= half.total_seconds()]
        return keep[:max_frames]

    # axis == "encoder" (or anything else): degrade to camera mode + warn.
    logger.warning(
        "reference_frame mode with axis=%s not fully supported yet — "
        "falling back to newest %d frames.",
        axis, max_frames,
    )
    return _select_frames_camera_mode(shipment, phase, max_frames)


def _select_frames_for_mode(
    mode: str,
    shipment: str,
    phase: Optional[str],
    value: Optional[float],
    axis: str,
    window_pct: float,
    max_frames: int,
) -> List[Path]:
    """Dispatch on baseline mode. Returns the list of JPG paths."""
    if mode == "camera":
        return _select_frames_camera_mode(shipment, phase, max_frames)
    if mode == "shipment_start":
        return _select_frames_shipment_start(shipment, phase, max_frames)
    if mode == "reference_frame":
        if value is None:
            return []
        return _select_frames_reference(
            shipment, phase, float(value), axis, window_pct, max_frames
        )
    # "target" and any unknown mode: no raw-frame source → empty.
    return []


# ---------------------------------------------------------------------------
# Worker interaction
# ---------------------------------------------------------------------------

def _post_frames_to_worker(
    frames: List[Path],
    camera_id: str,
    phase: str,
    mode: str,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """Multipart-POST the frames to the anomaly worker's /set-baseline.

    Returns whatever the worker responds with, plus a `success` flag and
    an `error` string on failure. Never raises so the calling route can
    surface partial info to the operator.
    """
    files_payload: List[Tuple[str, Tuple[str, bytes, str]]] = []
    read_failures = 0
    for p in frames:
        try:
            with open(p, "rb") as fh:
                data = fh.read()
            files_payload.append(("files", (p.name, data, "image/jpeg")))
        except OSError as e:
            read_failures += 1
            logger.debug(f"skipping unreadable baseline frame {p}: {e}")

    if not files_payload:
        return {"success": False, "error": "no readable baseline frames"}

    form_data = {
        "camera_id": str(camera_id),
        "phase":     str(phase),
        "mode":      str(mode),
    }
    url = f"{ANOMALY_WORKER_BASE}/set-baseline"
    try:
        r = requests.post(url, files=files_payload, data=form_data, timeout=timeout_s)
    except requests.RequestException as e:
        return {"success": False, "error": f"POST to worker failed: {e}"}
    if r.status_code >= 400:
        return {
            "success": False,
            "error":   f"worker returned {r.status_code}: {r.text[:400]}",
        }
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:400]}
    body["success"] = True
    body["read_failures"] = read_failures
    return body


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

class BuildBaselineRequest(BaseModel):
    mode: str = Field(..., description="camera | shipment_start | reference_frame | target")
    phase: Optional[str] = Field("0", description="phase suffix filter or 'all'")
    camera_id: Optional[str] = Field(
        "_global",
        description="baseline key on the worker; V1 uses a single global bank",
    )
    value: Optional[float] = Field(
        None,
        description="required when mode='reference_frame'. ms-epoch (axis=time) or encoder count",
    )
    axis: Optional[str] = Field(
        "time",
        description="'time' or 'encoder'; determines what `value` means",
    )
    window_pct: Optional[float] = Field(
        0.05,
        description="reference-frame window as fraction of visible span (default 5%)",
    )
    max_frames: Optional[int] = Field(
        DEFAULT_MAX_FRAMES,
        description=f"upper bound on frames sent to worker (default {DEFAULT_MAX_FRAMES})",
    )


@router.post("/api/anomaly/build-baseline")
async def build_baseline(request: Request, body: BuildBaselineRequest):
    """Build the anomaly worker's baseline from raw frames on disk.

    Frame source is picked by `mode` (same picker the colour heatmap uses).
    `target` mode is deliberately unsupported — target is a synthetic
    L*a*b* triplet with no raw-frame counterpart.
    """
    svc = load_service_config() or {}
    shipment = str(svc.get("current_shipment") or "").strip()
    if not shipment or shipment == "no_shipment":
        return JSONResponse(status_code=400, content={
            "success": False,
            "error":   "no active shipment — cannot select baseline frames",
        })

    if body.mode == "target":
        # Explicit: colour target has no raw-frame source; skip anomaly.
        return JSONResponse(content={
            "success":     False,
            "skipped":     "target mode is colour-only",
            "explanation": "🎯 Target is a synthetic L*a*b* triplet (no raw frames).",
        })

    frames = _select_frames_for_mode(
        mode=body.mode,
        shipment=shipment,
        phase=body.phase,
        value=body.value,
        axis=body.axis or "time",
        window_pct=body.window_pct or 0.05,
        max_frames=int(body.max_frames or DEFAULT_MAX_FRAMES),
    )
    if not frames:
        return JSONResponse(content={
            "success": False,
            "error":   f"no frames found for mode={body.mode} phase={body.phase}",
        })

    result = _post_frames_to_worker(
        frames=frames,
        camera_id=body.camera_id or "_global",
        phase=body.phase or "0",
        mode=body.mode,
    )
    if not result.get("success"):
        return JSONResponse(status_code=502, content={
            "success": False,
            "frames_selected": len(frames),
            "worker_error":    result.get("error"),
        })

    # Persist so the operator sees a summary + the next MVE boot can replay.
    svc["anomaly_baseline"] = {
        "mode":       body.mode,
        "phase":      body.phase,
        "value":      body.value,
        "axis":       body.axis or "time",
        "window_pct": body.window_pct or 0.05,
        "camera_id":  body.camera_id or "_global",
        "n_frames":   int(result.get("n_frames", len(frames))),
        "built_at":   int(time.time()),
    }
    save_service_config(svc)
    return JSONResponse(content={
        "success":    True,
        "mode":       body.mode,
        "n_frames":   int(result.get("n_frames", len(frames))),
        "skipped":    int(result.get("skipped", 0)),
        "read_failures": int(result.get("read_failures", 0)),
        "worker":     result,
    })


@router.get("/api/anomaly/baseline")
async def get_baseline(request: Request):
    """Return the persisted anomaly-baseline metadata + a live worker health
    probe. The frontend uses this to render "Anomaly baseline: reference_frame
    · 48 frames · 3m ago" next to the Charts-tab base-mode buttons."""
    svc = load_service_config() or {}
    persisted = svc.get("anomaly_baseline") or {}
    worker_status = _probe_worker_health()
    return JSONResponse(content={
        "persisted": persisted,
        "worker":    worker_status,
    })


def _probe_worker_health(timeout_s: float = 3.0) -> Dict[str, Any]:
    """Hit the worker's /health (or /detect if health isn't wired) with a
    short timeout so this endpoint stays snappy even when the worker is
    down. Never raises."""
    for path in ("/health",):
        try:
            r = requests.get(f"{ANOMALY_WORKER_BASE}{path}", timeout=timeout_s)
            if r.ok:
                try:
                    return {"reachable": True, **r.json()}
                except Exception:
                    return {"reachable": True, "raw": r.text[:200]}
        except requests.RequestException:
            continue
    return {"reachable": False}


# ---------------------------------------------------------------------------
# Boot-time self-heal — call from main.py apply_saved_config_at_startup
# ---------------------------------------------------------------------------

def replay_baseline_on_boot() -> Dict[str, Any]:
    """Idempotent: if service_config.anomaly_baseline exists, re-run the
    baseline build against currently-available frames so the worker's RAM
    is warm even after a fresh container start.

    Safe to call even when the worker is down — logs a warning and returns.
    """
    svc = load_service_config() or {}
    saved = svc.get("anomaly_baseline") or {}
    if not saved:
        return {"replayed": False, "reason": "no persisted baseline"}
    shipment = str(svc.get("current_shipment") or "").strip()
    if not shipment or shipment == "no_shipment":
        return {"replayed": False, "reason": "no active shipment"}

    mode = str(saved.get("mode") or "camera")
    if mode == "target":
        return {"replayed": False, "reason": "target mode is colour-only"}
    frames = _select_frames_for_mode(
        mode=mode,
        shipment=shipment,
        phase=saved.get("phase"),
        value=saved.get("value"),
        axis=saved.get("axis") or "time",
        window_pct=saved.get("window_pct") or 0.05,
        max_frames=int(saved.get("n_frames") or DEFAULT_MAX_FRAMES),
    )
    if not frames:
        return {"replayed": False, "reason": "no matching frames on disk"}
    result = _post_frames_to_worker(
        frames=frames,
        camera_id=saved.get("camera_id") or "_global",
        phase=saved.get("phase") or "0",
        mode=mode,
    )
    if not result.get("success"):
        logger.warning(f"anomaly baseline replay failed: {result.get('error')}")
        return {"replayed": False, "reason": result.get("error")}
    logger.info(
        f"anomaly baseline replayed on boot: mode={mode} "
        f"n_frames={result.get('n_frames')}"
    )
    return {"replayed": True, "n_frames": result.get("n_frames")}
