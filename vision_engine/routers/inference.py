"""Inference, pipeline, and model management routes."""

from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import pathlib
from config import (
    YOLO_INFERENCE_URL, GRADIO_MODEL, GRADIO_CONFIDENCE_THRESHOLD,
    inference_times, frame_intervals, last_inference_timestamp,
    max_inference_samples, capture_timestamps, max_capture_samples,
    load_service_config, save_service_config, PARENT_OBJECT_LIST,
)
from services.pipeline import InferenceModel, PipelinePhase, Pipeline, PipelineManager
from redis import Redis
from datetime import datetime
import config
import time
import json
import logging
import requests

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/inference/stats")
async def get_inference_stats(request: Request):
    """Get inference performance statistics."""
    watcher = request.app.state.watcher_instance
    pm = request.app.state.pipeline_manager
    sm = request.app.state.state_manager

    # Get pipeline name from pipeline manager (or fallback to URL-based detection)
    if pm and pm.current_pipeline:
        pipeline_name = pm.current_pipeline.name if hasattr(pm.current_pipeline, 'name') else str(pm.current_pipeline)
        service_type = pipeline_name
        service_url = YOLO_INFERENCE_URL
    elif "hf.space" in YOLO_INFERENCE_URL or "huggingface" in YOLO_INFERENCE_URL:
        service_type = "Gradio (Cloud)"
        service_url = YOLO_INFERENCE_URL
    else:
        service_type = "YOLO (Local)"
        service_url = YOLO_INFERENCE_URL

    # Get current state name and shipment
    state_name = sm.current_state.name if sm and sm.current_state else "unknown"
    shipment_id = watcher.shipment if watcher else "unknown"

    # Try to get timing data from Redis (cross-process)
    redis_inference_times = []
    inf_frame_timestamps = []
    cap_frame_timestamps = []

    try:
        redis_conn = Redis("redis", 6379, db=0)

        # Per-request inference latency (last 10)
        times_raw = redis_conn.lrange("inference_times", 0, -1)
        redis_inference_times = [float(t.decode('utf-8')) for t in times_raw if t]

        # Total inference throughput timestamps (all workers, last 200)
        inf_raw = redis_conn.lrange("inf_frame_timestamps", 0, -1)
        inf_frame_timestamps = [float(t.decode('utf-8')) for t in inf_raw if t]

        # Total capture throughput timestamps (all cameras, last 200)
        cap_raw = redis_conn.lrange("cap_frame_timestamps", 0, -1)
        cap_frame_timestamps = [float(t.decode('utf-8')) for t in cap_raw if t]
    except Exception as e:
        logger.warning(f"Failed to read timing from Redis: {e}")

    use_inference_times = redis_inference_times if redis_inference_times else config.inference_times

    # Average inference latency per request
    if use_inference_times:
        avg_inference = sum(use_inference_times) / len(use_inference_times)
        min_inference = min(use_inference_times)
        max_inference = max(use_inference_times)
    else:
        avg_inference = 0
        min_inference = 0
        max_inference = 0

    # FPS = frames counted over last 10 seconds, divided by 10
    now = time.time()
    inference_fps = len([t for t in inf_frame_timestamps if now - t <= 10.0]) / 10.0
    capture_fps = len([t for t in cap_frame_timestamps if now - t <= 10.0]) / 10.0

    # Autoscaler status and system capacity from app.state
    autoscaler = getattr(request.app.state, 'autoscaler', {})
    system_capacity = getattr(request.app.state, 'system_capacity', {})

    return JSONResponse(content={
        "service_type": service_type,
        "service_url": service_url,
        "state_name": state_name,
        "shipment_id": shipment_id,
        # Per-request inference latency
        "avg_inference_time_ms": round(avg_inference, 1),
        "min_inference_time_ms": round(min_inference, 1),
        "max_inference_time_ms": round(max_inference, 1),
        # True throughput FPS (all workers/cameras combined)
        "inference_fps": round(inference_fps, 2),
        "capture_fps": round(capture_fps, 2),
        # Autoscaler status + system capacity
        "autoscaler": autoscaler,
        "system_capacity": system_capacity,
        # Sample counts
        "inference_sample_count": len(use_inference_times),
        "inf_throughput_samples": len(inf_frame_timestamps),
        "cap_throughput_samples": len(cap_frame_timestamps),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@router.get("/api/model_classes")
async def get_model_classes(request: Request):
    """Get the list of object class names from the active inference model."""
    pm = request.app.state.pipeline_manager
    classes = []
    try:
        # Try YOLO service classes endpoint
        active_url = YOLO_INFERENCE_URL
        if pm and pm.get_current_model():
            active_url = pm.get_current_model().inference_url
        # Build classes URL from the inference URL
        classes_url = active_url.rstrip('/').rsplit('/detect', 1)[0] + '/classes'
        resp = requests.get(classes_url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            classes = data.get('classes', [])
    except Exception as e:
        logger.debug(f"Could not fetch model classes: {e}")
    # Also include PARENT_OBJECT_LIST as known objects
    for obj in PARENT_OBJECT_LIST:
        if obj not in classes and obj != '_root':
            classes.append(obj)
    return {"classes": classes}


@router.get("/api/inference")
async def get_inference_config():
    """Get inference module configuration (fast endpoint for UI)."""
    try:
        svc_config = load_service_config()
        if not svc_config or "inference" not in svc_config:
            return JSONResponse(content={
                "current_module": "gradio_hf",
                "modules": {}
            })
        return JSONResponse(content=svc_config["inference"])
    except Exception as e:
        logger.error(f"Error reading inference config: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ===== PIPELINE MANAGEMENT ENDPOINTS =====

@router.get("/api/pipelines")
async def get_pipelines(request: Request):
    """Get all pipelines and models configuration."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    return JSONResponse(content=pm.get_status())


@router.get("/api/pipelines/current")
async def get_current_pipeline(request: Request):
    """Get the currently active pipeline."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pm.current_pipeline:
        return JSONResponse(content={
            "pipeline": pm.current_pipeline.to_dict(),
            "current_model": pm.get_current_model().to_dict() if pm.get_current_model() else None
        })
    return JSONResponse(content={"pipeline": None, "current_model": None})


@router.post("/api/pipelines/activate/{pipeline_name}")
async def activate_pipeline(pipeline_name: str, request: Request):
    """Set the active pipeline by name."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pm.set_current_pipeline(pipeline_name):
        # Auto-save pipeline configuration to DATA_FILE
        try:
            svc_config = load_service_config() or {}
            svc_config["pipeline_config"] = pm.to_config()
            save_service_config(svc_config)
            logger.info(f"Pipeline '{pipeline_name}' activated and saved to DATA_FILE")
        except Exception as save_err:
            logger.error(f"Error saving after pipeline activation to DATA_FILE: {save_err}")

        return JSONResponse(content={
            "success": True,
            "message": f"Activated pipeline: {pipeline_name} (saved)",
            "current_model": pm.get_current_model().to_dict() if pm.get_current_model() else None
        })
    return JSONResponse(content={"error": f"Pipeline not found: {pipeline_name}"}, status_code=404)


@router.post("/api/pipelines")
async def create_or_update_pipeline(request: Request):
    """Create or update a pipeline."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    try:
        body = await request.json()
        pipeline = Pipeline.from_dict(body)

        if pm.add_pipeline(pipeline):
            # Auto-save pipeline configuration to DATA_FILE
            try:
                svc_config = load_service_config() or {}
                svc_config["pipeline_config"] = pm.to_config()
                save_service_config(svc_config)
                logger.info(f"Pipeline '{pipeline.name}' added and saved to DATA_FILE")
            except Exception as save_err:
                logger.error(f"Error saving pipeline to DATA_FILE: {save_err}")

            return JSONResponse(content={
                "success": True,
                "message": f"Pipeline '{pipeline.name}' created/updated and saved",
                "pipeline": pipeline.to_dict()
            })
        return JSONResponse(content={"error": "Failed to add pipeline"}, status_code=500)
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@router.delete("/api/pipelines/{pipeline_name}")
async def delete_pipeline(pipeline_name: str, request: Request):
    """Delete a pipeline."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pm.remove_pipeline(pipeline_name):
        # Auto-save pipeline configuration to DATA_FILE
        try:
            svc_config = load_service_config() or {}
            svc_config["pipeline_config"] = pm.to_config()
            save_service_config(svc_config)
            logger.info(f"Pipeline '{pipeline_name}' deleted and saved to DATA_FILE")
        except Exception as save_err:
            logger.error(f"Error saving after pipeline deletion to DATA_FILE: {save_err}")

        return JSONResponse(content={"success": True, "message": f"Pipeline '{pipeline_name}' deleted and saved"})
    return JSONResponse(content={"error": f"Cannot delete pipeline: {pipeline_name}"}, status_code=400)


@router.get("/api/models")
async def get_models(request: Request):
    """Get all available inference models."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    return JSONResponse(content={
        "models": {mid: m.to_dict() for mid, m in pm.models.items()}
    })


@router.post("/api/models")
async def create_or_update_model(request: Request):
    """Create or update an inference model."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    try:
        body = await request.json()
        model_id = body.get("model_id")
        if not model_id:
            return JSONResponse(content={"error": "model_id is required"}, status_code=400)

        model = InferenceModel.from_dict(body)

        if pm.add_model(model_id, model):
            # Auto-save pipeline configuration to DATA_FILE
            try:
                svc_config = load_service_config() or {}
                svc_config["pipeline_config"] = pm.to_config()
                save_service_config(svc_config)
                logger.info(f"Model '{model_id}' added and saved to DATA_FILE")
            except Exception as save_err:
                logger.error(f"Error saving model to DATA_FILE: {save_err}")
                # Don't fail the request if save fails, just log it

            return JSONResponse(content={
                "success": True,
                "message": f"Model '{model_id}' created/updated and saved",
                "model": model.to_dict()
            })
        return JSONResponse(content={"error": "Failed to add model"}, status_code=500)
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@router.delete("/api/models/{model_id}")
async def delete_model(model_id: str, request: Request):
    """Delete an inference model."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    if pm.remove_model(model_id):
        # Auto-save pipeline configuration to DATA_FILE
        try:
            svc_config = load_service_config() or {}
            svc_config["pipeline_config"] = pm.to_config()
            save_service_config(svc_config)
            logger.info(f"Model '{model_id}' deleted and saved to DATA_FILE")
        except Exception as save_err:
            logger.error(f"Error saving after model deletion to DATA_FILE: {save_err}")

        return JSONResponse(content={"success": True, "message": f"Model '{model_id}' deleted and saved"})
    return JSONResponse(content={"error": f"Cannot delete model: {model_id}"}, status_code=400)


@router.get("/api/gradio/models")
async def get_gradio_models(url: str):
    """Fetch available models from a Gradio/FastAPI endpoint (HuggingFace Spaces compatible)."""
    try:
        base_url = url.rstrip('/')
        models = []

        # Try multiple endpoints in order of preference
        endpoints_to_try = [
            (f"{base_url}/models", "fastapi_models"),  # Our unified API format (hg-codes api.py)
            (f"{base_url}/", "root"),  # Root endpoint with models list
            (f"{base_url}/api/", "gradio_api"),  # Gradio 4.x API info
            (f"{base_url}/info", "gradio_info"),  # Legacy Gradio
            (f"{base_url}/config", "gradio_config"),  # Gradio config
        ]

        for endpoint, source in endpoints_to_try:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except:
                        continue  # Not JSON, skip

                    # Our unified FastAPI format: {"models": {"key": "Display Name"}}
                    if "models" in data and isinstance(data["models"], dict):
                        # Return display names (values)
                        models = list(data["models"].values())
                        if models:
                            return JSONResponse(content={"models": models, "source": source})

                    # Our unified FastAPI format: {"models": ["name1", "name2"]} or root with models list
                    if "models" in data and isinstance(data["models"], list):
                        return JSONResponse(content={"models": data["models"], "source": source})

                    # Gradio 4.x API format - look for named_endpoints
                    if "named_endpoints" in data:
                        for ep_name in data["named_endpoints"]:
                            models.append(ep_name.strip('/'))
                        if models:
                            return JSONResponse(content={"models": models, "source": source})

                    # Check for components with choices (dropdowns)
                    if "components" in data:
                        for comp in data["components"]:
                            if comp.get("type") == "dropdown" and "choices" in comp.get("props", {}):
                                choices = comp["props"]["choices"]
                                if isinstance(choices, list):
                                    models.extend([c if isinstance(c, str) else c[0] for c in choices])
                    if models:
                        return JSONResponse(content={"models": models, "source": source})

            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
                continue

        # Try to infer from common patterns
        if "datamatrix" in base_url.lower() or "data-matrix" in base_url.lower():
            models = ["Data Matrix", "predict"]
        else:
            models = ["predict", "N/A"]

        return JSONResponse(content={
            "models": models,
            "note": "Could not auto-detect models, showing common defaults"
        })
    except Exception as e:
        logger.error(f"Error fetching Gradio models from {url}: {e}")
        return JSONResponse(content={
            "models": ["Data Matrix", "predict", "N/A"],
            "error": str(e)
        }, status_code=200)  # Return 200 with defaults instead of 500


WEIGHTS_DIR = pathlib.Path("/weights")
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB


def _yolo_base_url():
    """Derive YOLO service base URL (without /detect/) from the configured detection URL."""
    url = config.YOLO_INFERENCE_URL.rstrip("/")
    if url.endswith("/detect"):
        url = url[:-len("/detect")]
    return url


def _call_set_model_all_replicas(model_path: str):
    """Call set-model on yolo_inference, repeating to cover all replicas via DNS round-robin."""
    base = _yolo_base_url()
    replicas = int(os.environ.get("YOLO_REPLICAS", 2))
    attempts = max(replicas * 3, 6)
    successes = 0
    last_error = None
    for _ in range(attempts):
        try:
            resp = requests.post(f"{base}/set-model", data={"model_path": model_path}, timeout=30)
            if resp.status_code == 200:
                successes += 1
        except Exception as e:
            last_error = e
    return successes, last_error


def _fetch_classes():
    """Fetch class names from the currently loaded YOLO model."""
    base = _yolo_base_url()
    try:
        resp = requests.get(f"{base}/classes", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("classes", [])
        else:
            logger.warning(f"YOLO /classes returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to fetch YOLO classes: {e}")
    return []


@router.get("/api/models/weights")
def list_weights():
    """List available .pt weight files in /weights/ directory, plus current active classes."""
    weights = []
    if WEIGHTS_DIR.exists():
        for f in sorted(WEIGHTS_DIR.glob("*.pt")):
            stat = f.stat()
            weights.append({
                "name": f.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "modified": stat.st_mtime,
            })

    current_classes = _fetch_classes()
    return JSONResponse(content={"weights": weights, "current_classes": current_classes})


@router.post("/api/models/upload-weights")
async def upload_weights(file: UploadFile = File(...)):
    """Upload a .pt weight file and activate it on all YOLO replicas."""
    if not file.filename or not file.filename.endswith(".pt"):
        return JSONResponse(content={"error": "Only .pt files are allowed"}, status_code=400)

    # Sanitize filename
    safe_name = pathlib.Path(file.filename).name
    if not safe_name or safe_name.startswith("."):
        return JSONResponse(content={"error": "Invalid filename"}, status_code=400)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / safe_name

    # Stream to disk with size check
    total = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                f.close()
                dest.unlink(missing_ok=True)
                return JSONResponse(content={"error": "File exceeds 500MB limit"}, status_code=413)
            f.write(chunk)

    size_mb = round(total / (1024 * 1024), 1)
    logger.info(f"Uploaded weight file: {safe_name} ({size_mb}MB)")

    # Activate on all YOLO replicas
    model_path = f"/weights/{safe_name}"
    successes, last_error = _call_set_model_all_replicas(model_path)
    if successes == 0:
        err_msg = str(last_error) if last_error else "unknown error"
        return JSONResponse(content={
            "error": f"File uploaded but failed to activate model: {err_msg}",
            "filename": safe_name,
            "size_mb": size_mb,
        }, status_code=502)

    classes = _fetch_classes()
    logger.info(f"Activated {safe_name} on {successes} replica(s), classes: {classes}")

    return JSONResponse(content={
        "filename": safe_name,
        "path": model_path,
        "size_mb": size_mb,
        "classes": classes,
        "replicas_updated": successes,
    })


@router.post("/api/models/activate-weights")
async def activate_weights(request: Request):
    """Activate an existing .pt weight file on all YOLO replicas."""
    body = await request.json()
    filename = body.get("filename", "").strip()
    if not filename or not filename.endswith(".pt"):
        return JSONResponse(content={"error": "Invalid filename"}, status_code=400)

    safe_name = pathlib.Path(filename).name
    dest = WEIGHTS_DIR / safe_name
    if not dest.exists():
        return JSONResponse(content={"error": f"Weight file not found: {safe_name}"}, status_code=404)

    model_path = f"/weights/{safe_name}"
    successes, last_error = _call_set_model_all_replicas(model_path)
    if successes == 0:
        err_msg = str(last_error) if last_error else "unknown error"
        return JSONResponse(content={"error": f"Failed to activate model: {err_msg}"}, status_code=502)

    classes = _fetch_classes()
    logger.info(f"Activated {safe_name} on {successes} replica(s), classes: {classes}")

    return JSONResponse(content={
        "filename": safe_name,
        "path": model_path,
        "classes": classes,
        "replicas_updated": successes,
    })
