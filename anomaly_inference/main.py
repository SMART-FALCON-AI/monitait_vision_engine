"""Anomaly Inference Worker (v0.1).

A baseline-anomaly detector that consumes the SAME baseline picker the
colour heatmap uses on the MVE side (camera median / shipment_start /
target / reference_frame). The MVE side picks the frame set per
camera × phase, posts them here as a baseline, and on each subsequent
/detect call we compare the input against the matching baseline and
return bounding boxes for any region whose pixel statistics deviate
significantly from baseline mean/σ.

V1 algorithm — per-pixel mean + σ on LAB (CIELAB) channels:

1. Baseline build: collect N frames per (camera, phase, mode). Resize
   to a fixed 320×180 work resolution (≈4× faster than full 1280×720
   with no loss of regional accuracy). Convert each to LAB.  Stack
   into (N, H, W, 3) → mean and std per pixel per channel → store as
   {.npz} with arrays `mean`, `std`, `n`, `meta`.

2. Detect: resize the incoming frame to 320×180, convert to LAB.
   Compute per-pixel per-channel z-score: |x - μ| / max(σ, σ_floor).
   Take the max over channels → single anomaly score map.

3. Threshold the score map at z >= z_threshold (default 3.0).
   Morphological open+close to suppress salt-and-pepper noise.
   cv2.connectedComponentsWithStats → bbox per blob.

4. For each blob: confidence = clamp(max_z_in_region / 6.0, 0, 1).
   Filter out blobs whose area < min_area_px or confidence <
   min_confidence — but per the MVE design (anomaly is a YOLO-like
   class), we DON'T enforce min_confidence here. We return everything
   above the z threshold; MVE's standard per-class min_confidence
   filter (`audio_settings[anomaly].min_confidence`) takes care of
   that downstream — same as YOLO.

5. Rescale bbox coords back to the full input resolution before
   returning.

Response shape mirrors yolo_inference's `{"output": [...]}` so the
MVE detection pipeline can merge anomaly bboxes into the same
`detections` jsonb without any special case.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=os.environ.get("ANOMALY_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] anomaly_inference: %(message)s",
)
logger = logging.getLogger("anomaly_inference")

# ---------------------------------------------------------------------------
# Tunables (env-overridable)

BASELINES_DIR = Path(os.environ.get("ANOMALY_BASELINES_DIR", "/baselines"))
BASELINES_DIR.mkdir(parents=True, exist_ok=True)

# Internal work resolution. 320×180 keeps regional detail for a 1280×720
# input (4× downscale) and brings per-frame inference under ~10ms on CPU.
WORK_W = int(os.environ.get("ANOMALY_WORK_W", 320))
WORK_H = int(os.environ.get("ANOMALY_WORK_H", 180))

# σ floor — avoid divide-by-near-zero on flat patches (e.g. uniform
# background) where any tiny deviation would look like infinite z-score.
SIGMA_FLOOR = float(os.environ.get("ANOMALY_SIGMA_FLOOR", 4.0))

# z-score threshold above which a pixel is considered anomalous. Higher =
# fewer / stronger anomalies. 3.0 ≈ outside the 99.7% baseline distribution
# (rough — actual baseline isn't Gaussian).
Z_THRESHOLD = float(os.environ.get("ANOMALY_Z_THRESHOLD", 3.0))

# Minimum blob area at work resolution. 60 px² at 320×180 ≈ 960 px² at
# 1280×720 (after rescale) — drops fly-on-fabric noise but keeps real
# defects. Scaled up so the operator-facing area is intuitive.
MIN_AREA_PX_WORK = int(os.environ.get("ANOMALY_MIN_AREA_WORK", 60))

# Morphology kernel size. 3×3 open removes single-pixel noise; 5×5 close
# joins close blobs (a smudge spread over two adjacent pixels).
MORPH_OPEN_K = int(os.environ.get("ANOMALY_MORPH_OPEN_K", 3))
MORPH_CLOSE_K = int(os.environ.get("ANOMALY_MORPH_CLOSE_K", 5))

# Hard cap on bboxes returned per frame. Without this a wildly off-baseline
# frame (e.g. operator hand in shot) could fill the response with hundreds
# of tiny anomalies and downstream DB writes balloon.
MAX_BBOXES_PER_FRAME = int(os.environ.get("ANOMALY_MAX_BBOXES", 32))

# Class name returned for every anomaly bbox. Plural-instance, single
# class — matches YOLO semantics; the operator filters via the standard
# per-class min_confidence on the MVE side.
ANOMALY_CLASS_NAME = os.environ.get("ANOMALY_CLASS_NAME", "anomaly")

# ---------------------------------------------------------------------------
# Baseline cache — in-memory map of (camera_id, phase, mode) → (mean, std, n).
# Populated on /set-baseline and from disk on startup. Keeping the loaded
# arrays in RAM avoids re-reading from disk on every /detect call.

_baselines: dict[tuple[str, str, str], dict] = {}


def _baseline_key(camera_id: str, phase: str, mode: str) -> tuple[str, str, str]:
    return (str(camera_id), str(phase), str(mode))


def _baseline_path(camera_id: str, phase: str, mode: str) -> Path:
    # Safe filename — only alnum + underscore.
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in str(s))
    return BASELINES_DIR / f"{_safe(camera_id)}_p{_safe(phase)}_{_safe(mode)}.npz"


def _load_one_from_disk(camera_id: str, phase: str, mode: str) -> bool:
    """Load a single baseline from disk into the in-memory cache.

    Returns True on success. Used by /detect to recover when another
    uvicorn worker built the baseline (set-baseline lands on worker A,
    detect on worker B — each has its own _baselines dict). Reading the
    .npz on miss makes the workers eventually-consistent.
    """
    path = _baseline_path(camera_id, phase, mode)
    if not path.exists():
        return False
    try:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        _baselines[_baseline_key(camera_id, phase, mode)] = {
            "mean": data["mean"],
            "std": data["std"],
            "n": int(data["n"]),
            "meta": meta,
            "path": str(path),
        }
        return True
    except Exception as e:
        logger.warning(f"baseline load failed for {path.name}: {e}")
        return False


def _load_baselines_from_disk():
    """Re-populate the in-memory cache from /baselines on startup."""
    loaded = 0
    for p in BASELINES_DIR.glob("*.npz"):
        try:
            data = np.load(p, allow_pickle=True)
            meta = json.loads(str(data["meta"]))
            cam, ph, mode = meta["camera_id"], meta["phase"], meta["mode"]
            _baselines[_baseline_key(cam, ph, mode)] = {
                "mean": data["mean"],
                "std": data["std"],
                "n": int(data["n"]),
                "meta": meta,
                "path": str(p),
            }
            loaded += 1
        except Exception as e:
            logger.warning(f"baseline load failed for {p.name}: {e}")
    logger.info(f"loaded {loaded} baseline(s) from {BASELINES_DIR}")


# ---------------------------------------------------------------------------
# FastAPI app

app = FastAPI(title="anomaly_inference", version="0.1.0")


@app.on_event("startup")
def _on_startup():
    _load_baselines_from_disk()


# ---------------------------------------------------------------------------
# Helpers

def _decode_image(raw: bytes) -> Optional[np.ndarray]:
    """Decode an image upload to a BGR ndarray. Returns None on failure."""
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _to_work(img_bgr: np.ndarray) -> np.ndarray:
    """Resize BGR → fixed work resolution, then convert to LAB."""
    if img_bgr.shape[1] != WORK_W or img_bgr.shape[0] != WORK_H:
        img_bgr = cv2.resize(img_bgr, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)


def _bboxes_from_score_map(
    score_map: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> List[dict]:
    """Threshold + morphology + connected components → bbox list at orig res."""
    mask = (score_map >= Z_THRESHOLD).astype(np.uint8)
    if not mask.any():
        return []

    if MORPH_OPEN_K > 1:
        k = np.ones((MORPH_OPEN_K, MORPH_OPEN_K), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if MORPH_CLOSE_K > 1:
        k = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    sx = orig_w / WORK_W
    sy = orig_h / WORK_H

    out: list[dict] = []
    # label 0 is background — skip
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA_PX_WORK:
            continue
        region_score = float(score_map[y:y+h, x:x+w].max())
        # Confidence = z / 6 clamped to 0..1. z=3 → 0.50, z=6 → 1.00, z=9 → 1.00.
        # Operator's per-class min_confidence on MVE side does the real
        # filtering; we just need a stable monotonic confidence.
        conf = max(0.0, min(1.0, region_score / 6.0))
        out.append({
            "name": ANOMALY_CLASS_NAME,
            "confidence": round(conf, 4),
            "xmin": int(round(x * sx)),
            "ymin": int(round(y * sy)),
            "xmax": int(round((x + w) * sx)),
            "ymax": int(round((y + h) * sy)),
            "score_z": round(region_score, 3),
            "area_work_px": int(area),
        })

    # Strongest first; cap.
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out[:MAX_BBOXES_PER_FRAME]


# ---------------------------------------------------------------------------
# Endpoints (URL shape mirrors yolo_inference)

@app.get("/v1/anomaly-detection/anomaly_v1/health")
def health():
    return {
        "ok": True,
        "baselines_loaded": len(_baselines),
        "work_resolution": [WORK_W, WORK_H],
        "z_threshold": Z_THRESHOLD,
        "class_name": ANOMALY_CLASS_NAME,
    }


@app.get("/v1/anomaly-detection/anomaly_v1/classes")
def classes():
    # Only ever one class — operator treats anomaly the same as a YOLO
    # class in the Process tab (Show / min_conf / Severity).
    return {"classes": [ANOMALY_CLASS_NAME]}


@app.get("/v1/anomaly-detection/anomaly_v1/baselines")
def list_baselines():
    # Always reflect what's on disk (truth) — another worker may have just
    # written a baseline this worker hasn't loaded yet.
    _load_baselines_from_disk()
    out = []
    for (cam, phase, mode), b in _baselines.items():
        out.append({
            "camera_id": cam,
            "phase": phase,
            "mode": mode,
            "n_frames": int(b["n"]),
            "meta": b.get("meta", {}),
        })
    return {"baselines": out}


@app.post("/v1/anomaly-detection/anomaly_v1/set-baseline")
async def set_baseline(
    camera_id: str = Form(...),
    phase: str = Form("0"),
    mode: str = Form("camera"),
    files: List[UploadFile] = File(...),
):
    """Build a baseline from N reference frames.

    Expects multipart upload: `camera_id`, `phase`, `mode`, plus one or more
    image files in `files`. Computes per-pixel mean + std on the LAB
    channels at the worker's work resolution, writes a .npz to the
    baselines volume, and populates the in-memory cache.
    """
    if not files or len(files) < 3:
        raise HTTPException(status_code=400, detail="need at least 3 baseline frames")

    if len(files) > 500:
        # If someone POSTs 5k frames we'd OOM on the stack; truncate.
        files = files[:500]

    stack = []
    skipped = 0
    for f in files:
        raw = await f.read()
        img = _decode_image(raw)
        if img is None:
            skipped += 1
            continue
        try:
            lab = _to_work(img)
            stack.append(lab.astype(np.float32))
        except Exception as e:
            logger.warning(f"baseline frame decode/convert failed ({f.filename}): {e}")
            skipped += 1

    if len(stack) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"only {len(stack)} usable frames after decode ({skipped} skipped)",
        )

    arr = np.stack(stack, axis=0)             # (N, H, W, 3)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)

    meta = {
        "camera_id": str(camera_id),
        "phase": str(phase),
        "mode": str(mode),
        "n_frames": int(len(stack)),
        "skipped": int(skipped),
        "built_at": int(time.time()),
        "work_w": WORK_W,
        "work_h": WORK_H,
    }
    path = _baseline_path(camera_id, phase, mode)
    np.savez_compressed(path, mean=mean, std=std, n=int(len(stack)), meta=json.dumps(meta))

    _baselines[_baseline_key(camera_id, phase, mode)] = {
        "mean": mean,
        "std": std,
        "n": int(len(stack)),
        "meta": meta,
        "path": str(path),
    }
    logger.info(
        f"baseline built: cam={camera_id} phase={phase} mode={mode} "
        f"n={len(stack)} skipped={skipped} → {path.name}"
    )
    return {"status": "ok", **meta, "path": str(path)}


@app.post("/v1/anomaly-detection/anomaly_v1/detect")
async def detect(
    image: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    camera_id: str = Form("_global"),
    phase: str = Form("0"),
    mode: str = Form("global"),
):
    """Detect anomalies against the matching baseline.

    Pipeline-compatible: accepts the YOLO-style multipart field name `image`
    (what MVE's services.pipeline._run_yolo_inference posts) AND the legacy
    `image_file` field. camera_id/phase/mode are optional — when missing
    we fall back to a single `_global` baseline shared across all
    cameras, so the worker can be wired into MVE's existing pipeline
    mechanism with no code change on the MVE side.

    Returns a FLAT LIST of detection dicts (not `{"output": [...]}`)
    so MVE's pipeline manager can extend its detections list directly.
    Each entry mirrors a YOLO detection: name + confidence + xmin/ymin/
    xmax/ymax. Two extra diagnostic fields are added: `score_z` (raw max
    z-score in the region) and `area_work_px` (region area at work
    resolution). These survive into the detections jsonb so they're
    inspectable from Charts / DB without changing any schema.

    V1 limit (deliberate): one shared baseline across cameras. Per-camera
    per-phase per-mode baselines are computed and stored, but at inference
    time we only consult the `_global` key because the pipeline doesn't
    pass camera metadata. V2 (with proper MVE wiring) will route to the
    matching per-camera baseline.
    """
    uf = image or image_file
    if uf is None:
        raise HTTPException(status_code=400, detail="multipart 'image' (or 'image_file') required")

    key = _baseline_key(camera_id, phase, mode)
    # V1 fallback ladder:
    #   1. exact (camera_id, phase, mode) in this worker's in-memory cache
    #   2. exact key on disk (another uvicorn worker built it) — lazy-load
    #   3. _global baseline in memory
    #   4. _global baseline on disk — lazy-load
    #   5. nothing → return empty list (pipeline merges zero anomalies)
    if key not in _baselines:
        if not _load_one_from_disk(camera_id, phase, mode):
            global_key = _baseline_key("_global", "0", "global")
            if global_key not in _baselines and not _load_one_from_disk("_global", "0", "global"):
                return []
            key = global_key
        # else: lazy-load populated _baselines for the original key — use it

    raw = await uf.read()
    img = _decode_image(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="failed to decode image")

    orig_h, orig_w = img.shape[:2]
    lab = _to_work(img).astype(np.float32)

    b = _baselines[key]
    mean = b["mean"]
    std = b["std"]

    diff = np.abs(lab - mean)
    sigma = np.maximum(std, SIGMA_FLOOR)
    z = diff / sigma                # per-channel z-score
    z_max = z.max(axis=2)           # collapse over LAB channels

    bboxes = _bboxes_from_score_map(z_max, orig_w, orig_h)
    return bboxes
