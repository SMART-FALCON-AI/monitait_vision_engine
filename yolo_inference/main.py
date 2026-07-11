"""
Run a rest API exposing the yolov5s object detection model.

MVE-batch v1 — adds GET /capabilities + POST /v1/.../batch-detect/ per
docs/YOLO_BATCH_ENDPOINT_SPEC.md in the MVE repo. Single-image /detect endpoint
is unchanged so existing clients keep working.
"""
import io
import json
import os
from typing import List

from fastapi import FastAPI, File, UploadFile, Request, Form
from PIL import Image

from detect import Detector

app = FastAPI(title="YOLOv5 Service")

DETECTION_URL = "/v1/object-detection/yolov5s"
# Advertise this to MVE via /capabilities. Larger batches vectorize better on the
# GPU but at some point you run out of VRAM. 16 is a safe default for a single-GPU
# node; override via env if you have more memory (or less).
YOLO_MAX_BATCH = int(os.environ.get("YOLO_MAX_BATCH", "16"))

detector = Detector()


@app.get("/capabilities")
async def capabilities():
    """MVE probes this to auto-detect batch support. Returned as-is by
    PipelineManager.probe_batch_capability().
    """
    return {"batch": True, "max_batch": YOLO_MAX_BATCH}


@app.post(os.path.join(DETECTION_URL, 'detect') + '/')
@app.post(os.path.join(DETECTION_URL, 'detect'))
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    results = detector.detect(img)
    # Parse JSON string to prevent double-encoding by FastAPI
    return json.loads(results.to_json(orient="records"))


@app.post(os.path.join(DETECTION_URL, 'batch-detect') + '/')
@app.post(os.path.join(DETECTION_URL, 'batch-detect'))
async def batch_predict(image: List[UploadFile] = File(...)):
    """MVE-batch v1 — accepts N JPEG images as repeated `image` multipart parts,
    returns { "results": [ [det, det, ...], [det, ...], ... ] } in input order.

    Length of `results` MUST equal N; MVE explicitly falls back to per-image
    calls on any length mismatch. Per-image detection shape identical to /detect
    (yolov5 pandas.xyxy row → JSON dict) so downstream code is agnostic.
    """
    imgs = []
    for part in image:
        contents = await part.read()
        imgs.append(Image.open(io.BytesIO(contents)))

    per_image_dfs = detector.detect_batch(imgs)  # list of pandas DataFrames

    per_image_dets = [
        json.loads(df.to_json(orient="records")) for df in per_image_dfs
    ]
    return {"results": per_image_dets}


@app.post(os.path.join(DETECTION_URL, 'set-model'))
async def set_model(request: Request, model_path: str = Form(...)):
    """v4.0.98 — Persistent set-model with 1-click revert.

    Detector's `__init__` hardcodes `path="best.pt"`, so any weight loaded via
    the old set-model was lost on container restart. New behavior:

      1. Move current `best.pt` -> `old_best.pt` (backup, overwrites any prior).
      2. Copy `<model_path>` -> `best.pt` (new active weight; source file
         preserved so operator can re-select it later).
      3. Load the new model into memory.

    On the next container restart, `detect.py` still loads `best.pt` — but
    `best.pt` IS NOW the chosen weight, so persistence is automatic.

    One-click revert: call POST .../revert-model. It swaps `best.pt` <->
    `old_best.pt` and reloads. Second revert swaps back.
    """
    import shutil
    form_data = await request.form()
    model_path = form_data.get('model_path') or model_path
    if not model_path:
        return {'error': 'model_path required'}

    # Resolve relative names (e.g. "zarrin_best.pt") against the /weights or
    # /code bind-mount. Absolute paths pass through.
    candidates = [model_path]
    if not os.path.isabs(model_path):
        candidates += [
            os.path.join('/weights', model_path),
            os.path.join('/code', model_path),
        ]
    src = next((p for p in candidates if os.path.exists(p)), None)
    if src is None:
        return {'error': f'weight file not found: {model_path}',
                'searched': candidates}

    try:
        # Idempotent: if src IS best.pt, still succeed but skip the swap.
        best = '/code/best.pt'
        if os.path.abspath(src) != os.path.abspath(best):
            if os.path.exists(best):
                # Move current best.pt -> old_best.pt so operator can revert.
                # Overwrites any prior old_best.pt (single-slot undo).
                shutil.move(best, '/code/old_best.pt')
            shutil.copy2(src, best)
        # Load into memory — subsequent /detect calls use the new weights.
        detector.set_model(best)
        return {'message': 'ok',
                'active_weight': best,
                'source_used': src,
                'previous_saved_as': '/code/old_best.pt' if os.path.exists('/code/old_best.pt') else None}
    except Exception as e:
        return {'error': f'set-model failed: {e}'}


@app.post(os.path.join(DETECTION_URL, 'revert-model'))
async def revert_model():
    """v4.0.98 — One-click revert to the previous best.pt.

    Swaps `best.pt` and `old_best.pt`, then reloads. Second revert swaps back
    (so operator can flip between two weights freely without needing the
    original source file). Returns 400 if no `old_best.pt` exists.
    """
    import shutil
    best = '/code/best.pt'
    old  = '/code/old_best.pt'
    if not os.path.exists(old):
        return {'error': 'no previous weight to revert to (old_best.pt not found)'}
    try:
        tmp = '/code/best.pt.swap'
        # Atomic-ish 3-step swap
        shutil.move(best, tmp)
        shutil.move(old, best)
        shutil.move(tmp, old)
        detector.set_model(best)
        return {'message': 'ok', 'active_weight': best,
                'note': 'best.pt <-> old_best.pt swapped; revert again to swap back'}
    except Exception as e:
        return {'error': f'revert-model failed: {e}'}


@app.get(os.path.join(DETECTION_URL, 'health'))
async def health_check():
    """Health check endpoint for monitoring."""
    return {'status': 'healthy', 'service': 'YOLO Inference'}


@app.get(os.path.join(DETECTION_URL, 'classes'))
async def get_classes():
    """Return the list of class names from the loaded model."""
    names = getattr(detector.model, 'names', {})
    if isinstance(names, dict):
        class_list = list(names.values())
    elif isinstance(names, list):
        class_list = names
    else:
        class_list = []
    return {'classes': class_list}
