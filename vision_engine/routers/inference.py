"""Inference, pipeline, and model management routes."""

from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from fastapi import Depends as Depends
from routers._body import json_body as _json_body_dep, body as _req_body  # 4.0.425
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
def get_inference_stats(request: Request):
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
        redis_conn = Redis("redis", 6379, db=config.REDIS_DB)

        # Per-request inference latency (last 10)
        times_raw = redis_conn.lrange("inference_times", 0, -1)
        redis_inference_times = [float(t.decode('utf-8')) for t in times_raw if t]

        # Total inference throughput timestamps (all workers, last 2000)
        inf_raw = redis_conn.lrange("inf_frame_timestamps", 0, -1)
        inf_frame_timestamps = [float(t.decode('utf-8')) for t in inf_raw if t]

        # Total capture throughput timestamps (all cameras, last 2000)
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
def get_model_classes(request: Request):
    """Get the list of object class names from the active inference model."""
    pm = request.app.state.pipeline_manager
    classes = []
    try:
        # Try YOLO service classes endpoint
        # 4.0.438 — the weights model (object-detection phase), not just the first phase;
        # live config value, not the import-time copy
        active_url = config.YOLO_INFERENCE_URL
        _wm = _weights_model(pm) if pm else None
        if _wm is not None:
            active_url = _wm.inference_url
        # Build classes URL from the inference URL
        classes_url = active_url.rstrip('/').rsplit('/detect', 1)[0] + '/classes'
        resp = requests.get(classes_url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            classes = data.get('classes', [])
    except Exception as e:
        logger.debug(f"Could not fetch model classes: {e}")
    # Also include PARENT_OBJECT_LIST as known objects
    # 4.0.438 — the LIVE list (config.*): the import-time copy was the env default
    # (box, pack, DP-105, jean, knit), not the operator's configured parents.
    for obj in (getattr(config, "PARENT_OBJECT_LIST", None) or []):
        if obj not in classes and obj != '_root':
            classes.append(obj)
    return {"classes": classes}


@router.get("/api/inference")
def get_inference_config():
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
def get_pipelines(request: Request):
    """Get all pipelines and models configuration."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    return JSONResponse(content=pm.get_status())


@router.get("/api/pipelines/current")
def get_current_pipeline(request: Request):
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


def _push_pipeline_weight(pm, pipeline_name, request):
    """Push the pipeline's weight_file to YOLO (all replicas) and, once accepted, mirror it to
    best.pt. Returns the weight status dict, or None when the pipeline names no weight."""
    pobj = pm.pipelines.get(pipeline_name) if hasattr(pm, "pipelines") else None
    wf = (getattr(pobj, "weight_file", "") or "").strip() if pobj else ""
    if not wf:
        return None
    safe_name = pathlib.Path(wf).name
    dest = WEIGHTS_DIR / safe_name
    if not dest.exists():
        logger.warning(f"Pipeline '{pipeline_name}' weight_file '{safe_name}' not found in {WEIGHTS_DIR}")
        return {"weight_file": safe_name, "error": "weight file not found in /weights"}
    successes, last_error = _call_set_model_all_replicas(f"/weights/{safe_name}", request=request)
    # 4.0.438 — mirror to best.pt (the weight yolo loads on restart) only AFTER yolo
    # accepted it. It was copied first, so a weight yolo can't load became the
    # restart weight even though activation failed.
    if successes > 0:
        _mirror_best(dest, safe_name)
    logger.info(f"Pipeline '{pipeline_name}' weight '{safe_name}' pushed to {successes} replica(s)")
    # 4.0.456 — accepted by some calls but REFUSED by others (a worker out of memory, a load
    # error): yolo workers may now run DIFFERENT weights. It was reported as plain success.
    partial = bool(successes and last_error)
    return {"weight_file": safe_name, "replicas_updated": successes,
            "error": (str(last_error) if last_error else None), "partial": partial}


def _validate_pipeline_body(pm, body):
    """4.0.456 — a pipeline that cannot run inference was accepted: no name (it silently
    overwrote "default"), no phases, or a phase whose model doesn't exist — every frame of
    such a pipeline was stored as a CLEAN product. Returns an error text or None."""
    if not isinstance(body, dict):
        return "expected a JSON object"
    if not str(body.get("name") or "").strip():
        return "a pipeline needs a name"
    phases = body.get("phases")
    if not isinstance(phases, list) or not phases:
        return "a pipeline needs at least one phase"
    models = getattr(pm, "models", {}) or {}
    for i, ph in enumerate(phases):
        mid = (ph or {}).get("model_id") if isinstance(ph, dict) else None
        if not mid or mid not in models:
            return f"phase {i + 1}: unknown model {mid!r}"
    return None


@router.post("/api/pipelines/activate/{pipeline_name}")
def activate_pipeline(pipeline_name: str, request: Request):
    """Set the active pipeline by name."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    # 4.0.456 — a pipeline whose models were deleted since (delete_model only guards the
    # ACTIVE pipeline) ran nothing and stored every frame as clean
    _pobj = pm.pipelines.get(pipeline_name) if hasattr(pm, "pipelines") else None
    if _pobj is not None:
        _missing = [ph.model_id for ph in (getattr(_pobj, "phases", None) or [])
                    if getattr(ph, "enabled", True) and ph.model_id not in (getattr(pm, "models", {}) or {})]
        if _missing or not [ph for ph in (getattr(_pobj, "phases", None) or []) if getattr(ph, "enabled", True)]:
            return JSONResponse(content={
                "success": False,
                "error": (f"Pipeline '{pipeline_name}' cannot run: "
                          + (f"its model(s) {_missing} no longer exist" if _missing else "it has no enabled phase")
                          + " — fix the pipeline first.")}, status_code=400)

    if pm.set_current_pipeline(pipeline_name):
        # Auto-save pipeline configuration to DATA_FILE
        try:
            svc_config = load_service_config() or {}
            svc_config["pipeline_config"] = pm.to_config()
            save_service_config(svc_config)
            logger.info(f"Pipeline '{pipeline_name}' activated and saved to DATA_FILE")
        except Exception as save_err:
            logger.error(f"Error saving after pipeline activation to DATA_FILE: {save_err}")

        # 4.0.286 — per-pipeline weight. If the activated pipeline names a
        # weight_file, push it to YOLO via the EXISTING /set-model API (NO change
        # to the YOLO container) — identical mechanism to /api/models/activate-weights.
        # Mirror to /weights/best.pt so the choice survives a restart. Non-fatal:
        # the pipeline stays active even if the weight push fails.
        weight_status = None
        try:
            weight_status = _push_pipeline_weight(pm, pipeline_name, request)
        except Exception as _we:
            logger.error(f"Error pushing pipeline weight on activate: {_we}")
            weight_status = {"error": str(_we)}

        # 4.0.438 — a pipeline whose weight did NOT reach yolo is not "activated" for the
        # operator: yolo keeps running the previous weight, so this pipeline's rules would
        # run against the wrong classes. It answered success:true with the error buried in
        # weight.error, which neither UI caller reads (both showed green "activated").
        if weight_status and weight_status.get("error") and not weight_status.get("replicas_updated"):
            return JSONResponse(content={
                "success": False,
                "error": (f"Pipeline '{pipeline_name}' is selected in MVE, but YOLO could NOT load its weight "
                          f"'{weight_status.get('weight_file', '?')}': {weight_status['error']}. "
                          f"YOLO is still running the previous weight."),
                "current_model": pm.get_current_model().to_dict() if pm.get_current_model() else None,
                "weight": weight_status,
            }, status_code=502)

        _warn = None
        if weight_status and weight_status.get("partial"):
            _warn = (f"some YOLO workers refused the weight ({weight_status.get('error')}) — they may still run "
                     f"the previous one; activate again")
        return JSONResponse(content={
            "success": True,
            "message": f"Activated pipeline: {pipeline_name} (saved)",
            "current_model": pm.get_current_model().to_dict() if pm.get_current_model() else None,
            "weight": weight_status,
            "warning": _warn,
        })
    return JSONResponse(content={"error": f"Pipeline not found: {pipeline_name}"}, status_code=404)


@router.post("/api/pipelines")
def create_or_update_pipeline(request: Request, _mve_body=Depends(_json_body_dep)):
    """Create or update a pipeline."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    try:
        body = _req_body(_mve_body)
        _err = _validate_pipeline_body(pm, body)
        if _err:
            return JSONResponse(content={"error": _err}, status_code=400)
        pipeline = Pipeline.from_dict(body)
        _prev = pm.pipelines.get(pipeline.name) if hasattr(pm, "pipelines") else None
        _prev_wf = (getattr(_prev, "weight_file", "") or "").strip() if _prev else ""
        _is_active = getattr(getattr(pm, "current_pipeline", None), "name", None) == pipeline.name

        if pm.add_pipeline(pipeline):
            # Auto-save pipeline configuration to DATA_FILE
            try:
                svc_config = load_service_config() or {}
                svc_config["pipeline_config"] = pm.to_config()
                save_service_config(svc_config)
                logger.info(f"Pipeline '{pipeline.name}' added and saved to DATA_FILE")
            except Exception as save_err:
                logger.error(f"Error saving pipeline to DATA_FILE: {save_err}")

            # 4.0.456 — editing the ACTIVE pipeline's weight only changed MVE's record: the
            # weight is pushed on ACTIVATE only, so YOLO kept the old one (also after a
            # restart — best.pt was never mirrored) while the UI showed the new one.
            weight_status = None
            _new_wf = (getattr(pipeline, "weight_file", "") or "").strip()
            if _is_active and _new_wf and _new_wf != _prev_wf:
                try:
                    weight_status = _push_pipeline_weight(pm, pipeline.name, request)
                except Exception as _we:
                    weight_status = {"weight_file": _new_wf, "error": str(_we)}
                if weight_status and weight_status.get("error") and not weight_status.get("replicas_updated"):
                    return JSONResponse(content={
                        "success": False,
                        "error": (f"Pipeline '{pipeline.name}' saved, but YOLO could NOT load its new weight "
                                  f"'{weight_status.get('weight_file', '?')}': {weight_status['error']}. "
                                  f"YOLO is still running the previous weight."),
                        "pipeline": pipeline.to_dict(), "weight": weight_status,
                    }, status_code=502)

            return JSONResponse(content={
                "success": True,
                "message": f"Pipeline '{pipeline.name}' created/updated and saved"
                           + (" — its new weight is loaded on YOLO" if weight_status else ""),
                "pipeline": pipeline.to_dict(),
                "weight": weight_status,
            })
        return JSONResponse(content={"error": "Failed to add pipeline"}, status_code=500)
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)


@router.delete("/api/pipelines/{pipeline_name}")
def delete_pipeline(pipeline_name: str, request: Request):
    """Delete a pipeline."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)
    # 4.0.438 — not the ACTIVE pipeline: inference kept running the deleted object, and on
    # restart MVE fell back to "default" (the Gradio HF pipeline → no local YOLO at all).
    if getattr(getattr(pm, "current_pipeline", None), "name", None) == pipeline_name:
        return JSONResponse(content={"error": f"Pipeline '{pipeline_name}' is active — activate another pipeline first"}, status_code=409)

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
def get_models(request: Request):
    """Get all available inference models."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    return JSONResponse(content={
        "models": {mid: m.to_dict() for mid, m in pm.models.items()}
    })


@router.post("/api/models")
def create_or_update_model(request: Request, _mve_body=Depends(_json_body_dep)):
    """Create or update an inference model."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)

    try:
        body = _req_body(_mve_body)
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
def delete_model(model_id: str, request: Request):
    """Delete an inference model."""
    pm = request.app.state.pipeline_manager
    if pm is None:
        return JSONResponse(content={"error": "Pipeline manager not initialized"}, status_code=503)
    # 4.0.438 — not a model the ACTIVE pipeline uses: that phase silently skipped every frame.
    try:
        _cp = getattr(pm, "current_pipeline", None)
        if _cp and any(getattr(ph, "model_id", None) == model_id for ph in (_cp.phases or [])):
            return JSONResponse(content={"error": f"Model '{model_id}' is used by the active pipeline '{_cp.name}' — remove it from that pipeline or activate another first"}, status_code=409)
    except Exception:
        pass

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
def get_gradio_models(url: str):
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


def _weights_model(pm):
    """The model of the active pipeline that takes WEIGHTS (/set-model, /classes).

    4.0.438 — this was get_current_model(), the FIRST enabled phase of any type. The math
    and anomaly services are registered as model_type "yolo" too (YOLO-compatible API), so
    a pipeline [math, yolo] or [gradio, yolo] sent set-model to the math service / HF
    Space (404). Prefer an enabled phase on an object-detection service, then any enabled
    yolo-type phase, then the old behaviour."""
    if pm is None:
        return None
    try:
        cp = getattr(pm, "current_pipeline", None)
        phases = sorted(cp.phases, key=lambda p: p.order) if cp and cp.phases else []
        cands = [pm.models[p.model_id] for p in phases if p.enabled and p.model_id in pm.models]
        for m in cands:
            if m.model_type == "yolo" and "/object-detection/" in (m.inference_url or ""):
                return m
        for m in cands:
            if m.model_type == "yolo":
                return m
    except Exception:
        pass
    return pm.get_current_model()


def _yolo_base_url(request: "Request | None" = None):
    """Derive YOLO service base URL (without /detect/) from the ACTIVE pipeline.

    4.0.47 — previously this read `config.YOLO_INFERENCE_URL` (a static env-var
    snapshot at process start). When the operator switched to "Local YOLO" via
    the Inference tab, the PipelineManager.current_pipeline got updated but
    this helper kept pointing at the legacy HuggingFace Space URL. Result:
    /set-model called the HF Space → 404 Not Found → "Failed to activate
    model: unknown error" in the UI. Verified on vteam12 2026-06-22: logs
    showed `base='https://smartfalcon-ai-industrial-defect-detection.hf.space'`.

    Resolution order:
      1. pipeline_manager.get_current_model().inference_url  (operator's choice)
      2. config.YOLO_INFERENCE_URL                            (legacy fallback)
    Then strip a trailing /detect/ to get the base.
    """
    url = None
    try:
        if request is not None:
            pm = getattr(request.app.state, "pipeline_manager", None)
            m = _weights_model(pm)
            if m is not None:
                url = m.inference_url
    except Exception:
        url = None
    if not url:
        url = config.YOLO_INFERENCE_URL
    url = (url or "").rstrip("/")
    if url.endswith("/detect"):
        url = url[:-len("/detect")]
    return url


def _mirror_best(dest, safe_name):
    """3.21.12 — copy the ACTIVATED weight to /weights/best.pt, the single-file bind-mount
    yolo loads at restart, so the choice survives a container recreate.
    4.0.438 — called only AFTER yolo accepted the weight: it used to run before set-model,
    so a weight yolo cannot load became the restart weight (yolo then failed to start)."""
    try:
        import shutil as _shutil
        best_dest = WEIGHTS_DIR / "best.pt"
        if dest.resolve() != best_dest.resolve():
            tmp = WEIGHTS_DIR / "best.pt.part"
            _shutil.copyfile(dest, tmp)
            os.replace(tmp, best_dest)
            logger.info(f"Mirrored {safe_name} -> /weights/best.pt for restart persistence")
    except Exception as _e:
        logger.warning(f"Failed to mirror {safe_name} -> /weights/best.pt: {_e}")


def _call_set_model_all_replicas(model_path: str, request=None, base=None):
    """Call set-model on yolo_inference, repeating to cover all replicas via DNS round-robin.
    Pass an explicit `base` when calling from a BACKGROUND thread — the request object is
    only valid during the request lifecycle."""
    if base is None:
        base = _yolo_base_url(request)
    replicas = int(os.environ.get("YOLO_REPLICAS", 2))
    # 4.0.293 — uvicorn spreads set-model over N workers by OS connection routing, so we
    # must send MANY more calls than workers to reliably hit every one (coupon-collector).
    # Under-shooting left some workers on the OLD weight after a switch. Cover up to
    # YOLO_WORKERS×4 (env-tunable), min 30. GPU headroom (right-sized workers) means each
    # in-place swap is cheap, so the extra calls are safe.
    _workers = int(os.environ.get("YOLO_WORKERS", 4) or 4)
    attempts = max(replicas * 3, _workers * 4, int(os.environ.get("SET_MODEL_ATTEMPTS", "30")))
    # 4.0.295 — WALL-CLOCK deadline + shorter per-attempt timeout so a slow/hung yolo
    # can't make the ACTIVATE request outlast the frontend/nginx timeout, which showed as
    # "Activate pipeline failed: Failed to fetch". Bounds the whole call to ~deadline secs.
    import time as _t
    _deadline = _t.monotonic() + float(os.environ.get("SET_MODEL_DEADLINE", "25"))
    _per_timeout = int(os.environ.get("SET_MODEL_TIMEOUT", "8"))
    successes = 0
    last_error = None
    # 4.0.47 — log every attempt so we can see WHY successes stays 0 even when
    # a direct curl to the same URL returns 200. Earlier "unknown error"
    # response on the UI gave us nothing to debug; this surfaces it.
    logger.info(f"set-model: base={base!r} model_path={model_path!r} attempts={attempts} deadline={os.environ.get('SET_MODEL_DEADLINE','25')}s")
    for i in range(attempts):
        if _t.monotonic() > _deadline:
            logger.info(f"set-model: hit deadline after {i} attempts ({successes} ok)")
            break
        try:
            resp = requests.post(f"{base}/set-model", data={"model_path": model_path}, timeout=_per_timeout)
            logger.info(f"set-model attempt {i+1}/{attempts}: status={resp.status_code} body={resp.text[:200]!r}")
            # 4.0.428 — yolo_inference answers a FAILED load with HTTP 200 + {"error": ...};
            # that was counted as a success, so Activate reported OK while yolo kept the
            # old weight. A 200 whose JSON body is an error is a failure.
            _err_body = None
            if resp.status_code == 200:
                try:
                    _j = resp.json()
                    if isinstance(_j, dict) and _j.get("error") and not _j.get("message"):
                        _err_body = str(_j.get("error"))[:200]
                except Exception:
                    pass  # non-JSON 200 (other inference services) = success
            if resp.status_code == 200 and _err_body is None:
                successes += 1
            elif _err_body is not None:
                last_error = _err_body
            else:
                # Capture the non-200 as last_error too so the caller's error
                # message is informative (was always "unknown error" because
                # last_error stayed None when status was non-200).
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error = e
            logger.info(f"set-model attempt {i+1}/{attempts}: exception={e!r}")
    logger.info(f"set-model done: successes={successes}/{attempts} last_error={last_error!r}")
    return successes, last_error


def _fetch_classes(request=None):
    """Fetch class names from the currently loaded YOLO model."""
    base = _yolo_base_url(request)
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
def list_weights(request: Request):
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

    current_classes = _fetch_classes(request)
    return JSONResponse(content={"weights": weights, "current_classes": current_classes})


@router.post("/api/models/upload-weights")
def upload_weights(request: Request, file: UploadFile = File(...), activate: int = 1):
    """Upload a .pt weight file. activate=1 (default) also pushes it to all YOLO
    replicas + mirrors best.pt. activate=0 (pipeline card) ONLY saves + names it, so
    uploading a weight for a NON-active pipeline never disturbs the live YOLO model —
    the pipeline's own activation pushes it later."""
    if not file.filename or not file.filename.endswith(".pt"):
        return JSONResponse(content={"error": "Only .pt files are allowed"}, status_code=400)

    # Sanitize filename
    safe_name = pathlib.Path(file.filename).name
    if not safe_name or safe_name.startswith("."):
        return JSONResponse(content={"error": "Invalid filename"}, status_code=400)

    # 4.0.438 — "best.pt" IS the weight yolo loads on restart: an upload under that name
    # replaced it before yolo had even tried to load it. Keep it under its own name (the UI
    # uses the returned filename); activation mirrors it to best.pt once yolo accepted it.
    if safe_name.lower() == "best.pt":
        import time as _tu
        safe_name = "uploaded_" + _tu.strftime("%Y%m%d-%H%M%S") + "_best.pt"

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / safe_name
    part = WEIGHTS_DIR / (safe_name + ".part")

    # Stream to disk with size check.
    # 4.0.438 — into a .part file, renamed over the real name only when complete: writing
    # the target directly truncated an existing weight of the same name at once (an
    # over-limit or failed re-upload — ENOSPC, dropped connection — destroyed the original).
    total = 0
    try:
        with open(part, "wb") as f:
            while chunk := file.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_SIZE:
                    f.close()
                    part.unlink(missing_ok=True)
                    return JSONResponse(content={"error": "File exceeds 500MB limit"}, status_code=413)
                f.write(chunk)
        os.replace(part, dest)
    except Exception as _ue:
        try:
            part.unlink(missing_ok=True)
        except Exception:
            pass
        logger.error(f"Weight upload failed for {safe_name}: {_ue}")
        return JSONResponse(content={"error": f"Upload failed: {_ue}"}, status_code=500)

    size_mb = round(total / (1024 * 1024), 1)
    logger.info(f"Uploaded weight file: {safe_name} ({size_mb}MB)")

    # 4.0.289 — activate=0: SAVE + NAME only (pipeline card). Don't mirror best.pt and
    # don't push to YOLO, so a weight uploaded for a non-active pipeline leaves the live
    # model untouched. The pipeline's activation (activate_pipeline) does the push.
    if not activate:
        return JSONResponse(content={
            "filename": safe_name,
            "path": f"/weights/{safe_name}",
            "size_mb": size_mb,
            "activated": False,
        })

    # Activate on all YOLO replicas (use the just-uploaded path)
    model_path = f"/weights/{safe_name}"
    successes, last_error = _call_set_model_all_replicas(model_path, request=request)
    if successes == 0:
        err_msg = str(last_error) if last_error else "unknown error"
        return JSONResponse(content={
            "error": f"File uploaded but failed to activate model: {err_msg}",
            "filename": safe_name,
            "size_mb": size_mb,
        }, status_code=502)
    _mirror_best(dest, safe_name)   # 4.0.438 — only once yolo accepted it

    classes = _fetch_classes(request)
    logger.info(f"Activated {safe_name} on {successes} replica(s), classes: {classes}")

    return JSONResponse(content={
        "filename": safe_name,
        "path": model_path,
        "size_mb": size_mb,
        "classes": classes,
        "replicas_updated": successes,
    })


@router.post("/api/models/activate-weights")
def activate_weights(request: Request, _mve_body=Depends(_json_body_dep)):
    """Activate an existing .pt weight file on all YOLO replicas."""
    body = _req_body(_mve_body)
    filename = body.get("filename", "").strip()
    if not filename or not filename.endswith(".pt"):
        return JSONResponse(content={"error": "Invalid filename"}, status_code=400)

    safe_name = pathlib.Path(filename).name
    dest = WEIGHTS_DIR / safe_name
    if not dest.exists():
        return JSONResponse(content={"error": f"Weight file not found: {safe_name}"}, status_code=404)

    model_path = f"/weights/{safe_name}"
    successes, last_error = _call_set_model_all_replicas(model_path, request=request)
    if successes == 0:
        err_msg = str(last_error) if last_error else "unknown error"
        return JSONResponse(content={"error": f"Failed to activate model: {err_msg}"}, status_code=502)
    _mirror_best(dest, safe_name)   # 4.0.438 — only once yolo accepted it

    classes = _fetch_classes(request)
    logger.info(f"Activated {safe_name} on {successes} replica(s), classes: {classes}")

    return JSONResponse(content={
        "filename": safe_name,
        "path": model_path,
        "classes": classes,
        "replicas_updated": successes,
    })
