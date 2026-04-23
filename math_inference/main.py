"""Math inference service — pure-math measurement channels.

HTTP contract mirrors yolo_inference:
    POST /v1/math/analyze  with files={"image": <bytes>}
Returns a JSON list of detection dicts. Each detection has a flat `name`
(rule-able), a `confidence` 0..1 from a fixed absolute math map, and a real
bounding box in frame coordinates.

MVE attaches encoder, cam_id, shipment, capture_t on its side; the worker is
stateless and context-free.
"""
import os
import logging

from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

from math_worker import MathWorker, USING_GPU

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("math_inference")

app = FastAPI(title="Math Inference Service")

# URL shape mirrors yolo_inference:
#   yolo → /v1/object-detection/yolov5s/{detect,health,classes,set-model}
#   math → /v1/math-analysis/math_v1/{detect,health,channels,set-config}
DETECTION_URL = "/v1/math-analysis/math_v1"

worker = MathWorker(
    tiles_x     = int(os.environ.get("MATH_TILES_X", "1")),
    tiles_y     = int(os.environ.get("MATH_TILES_Y", "1")),
    bands       = int(os.environ.get("MATH_BANDS", "8")),
    fft_top_k   = int(os.environ.get("MATH_FFT_TOP_K", "3")),
    flat_field  = os.environ.get("MATH_FLAT_FIELD_ENABLE", "false").lower() == "true",
)
logger.info(
    f"math worker ready — device={'GPU' if USING_GPU else 'CPU'}, "
    f"tiles={worker.tiles_x}x{worker.tiles_y}, bands={worker.bands}, "
    f"fft_top_k={worker.fft_top_k}, flat_field={worker.flat_field}"
)


@app.post(DETECTION_URL + "/detect/")
@app.post(DETECTION_URL + "/detect")
async def detect(image: UploadFile = File(...)):
    """Emit a flat list of math-channel detections for one frame.

    Same HTTP contract as /v1/object-detection/yolov5s/detect — MVE's
    `_run_yolo_inference` treats both identically.
    """
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        logger.warning("imdecode failed — returning empty detections")
        return []
    try:
        return worker.analyze(bgr)
    except Exception as e:
        logger.exception(f"analyze failed: {e}")
        return []


@app.get(DETECTION_URL + "/health")
async def health():
    return {
        "status":  "healthy",
        "service": "Math Inference",
        "device":  "GPU" if USING_GPU else "CPU",
    }


@app.get(DETECTION_URL + "/channels")
async def channels():
    """Advertise the channel names this worker emits (parallels YOLO /classes)."""
    return {"channels": worker.channel_names()}


@app.post(DETECTION_URL + "/set-config")
async def set_config(
    tiles_x:    int  = None,
    tiles_y:    int  = None,
    bands:      int  = None,
    fft_top_k:  int  = None,
    flat_field: bool = None,
):
    """Runtime config tweak without restart — parallels YOLO /set-model."""
    if tiles_x    is not None: worker.tiles_x    = max(1, int(tiles_x))
    if tiles_y    is not None: worker.tiles_y    = max(1, int(tiles_y))
    if bands      is not None: worker.bands      = max(1, int(bands))
    if fft_top_k  is not None: worker.fft_top_k  = max(1, int(fft_top_k))
    if flat_field is not None: worker.flat_field = bool(flat_field)
    return {"message": "ok",
            "config": {"tiles_x": worker.tiles_x, "tiles_y": worker.tiles_y,
                       "bands": worker.bands, "fft_top_k": worker.fft_top_k,
                       "flat_field": worker.flat_field}}
