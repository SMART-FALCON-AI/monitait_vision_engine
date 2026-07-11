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
    form_data = await request.form()
    model_path = form_data.get('model_path')
    detector.set_model(model_path)
    return {'message': 'ok'}


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
