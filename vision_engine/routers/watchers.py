"""External-machine (Watcher Jet) metrics — ingestion + read API (4.0.359).

Point a Watcher Jet (or any compatible device) at this MVE instead of the Monitait
cloud by changing its URL to `http://<mve-host>/api/factory/update-watcher/`. Its
production counts + analog telemetry then land in TimescaleDB (see
services/watcher_metrics.py) and are charted in the Charts tab.

The three /api/factory/* endpoints mirror the cloud's shapes so no firmware change is
needed beyond the URL. This is a METRICS-only build: the image-upload leg is accepted
and acked (so image-mode devices don't retry-storm) but the image is discarded.

Registered in main.py BEFORE commands_router (the catch-all).
"""
import logging

from fastapi import APIRouter, Request
from fastapi import Depends as Depends
from routers._body import json_body as _json_body_dep, body as _req_body  # 4.0.425
from fastapi.responses import JSONResponse

from services import watcher_metrics as wm

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
# ingestion — Watcher-Jet compatible
# --------------------------------------------------------------------------- #
@router.post("/api/factory/update-watcher/")
def update_watcher(request: Request, _mve_body=Depends(_json_body_dep)):
    """No-image path: the device POSTs its JSON report and expects HTTP 200."""
    try:
        payload = _req_body(_mve_body)
        r = wm.ingest(payload)
        return {"status": "ok", **r}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.warning("update-watcher ingest failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/factory/image-update-watcher-data/")
def image_update_watcher_data(request: Request, _mve_body=Depends(_json_body_dep)):
    """Image path, step 1: store the metrics and hand back an id the device then
    references when it uploads the image."""
    try:
        payload = _req_body(_mve_body)
        r = wm.ingest(payload)
        return {"status": "ok", "_id": r.get("id"), "register_id": r.get("register_id")}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.warning("image-update-watcher-data ingest failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/factory/image-update-watcher/")
def image_update_watcher(request: Request):
    """Image path, step 2: metrics-only build — accept and ack, discard the image."""
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# read — consumed by the Charts tab
# --------------------------------------------------------------------------- #
@router.get("/api/watchers")
def watchers():
    """List machines that have reported, with last-seen + latest counts."""
    try:
        return {"watchers": wm.list_watchers()}
    except Exception as e:
        logger.warning("list watchers failed: %s", e)
        return {"watchers": [], "error": str(e)}


@router.get("/api/watchers/metrics")
def watcher_metrics_ep(register_id: str, since_ms: int = 0, until_ms: int = 0,
                       limit: int = 5000):
    """Time-series for one machine (OK/NG + the analog keys in extra_info)."""
    try:
        return wm.metrics(register_id, since_ms or None, until_ms or None, limit)
    except Exception as e:
        logger.warning("watcher metrics failed: %s", e)
        return JSONResponse({"error": str(e), "points": [], "metric_keys": []},
                            status_code=500)



@router.post("/api/watchers/register")
def watcher_register(request: Request, _mve_body=Depends(_json_body_dep)):
    """Register / update a machine: name, line (group), sensor, type, ideal_rate
    (units/hr, used for the OEE Performance factor); 4.0.417 Hardware-tab station
    setup: next_watcher (the station this one feeds), ok_factor / ng_factor (units
    per counted OK / NG pulse), assigned. Only provided fields change."""
    try:
        b = _req_body(_mve_body)
        ir = b.get("ideal_rate")
        if ir not in (None, ""):
            ir = float(ir)
            # 4.0.459 — 0 / negative / NaN made OEE 0 % (or NaN) with no warning
            if not (ir == ir and ir not in (float("inf"), float("-inf")) and ir > 0):
                raise ValueError("ideal_rate must be a positive number")
        _as = b.get("assigned")
        if isinstance(_as, str):
            _as = _as.strip().lower() in ("true", "1", "yes", "on")   # bool("false") was True
        return wm.register_watcher(
            str(b.get("register_id", "")),
            name=b.get("name"), line=b.get("line"),
            sensor=b.get("sensor"), wtype=b.get("type") or b.get("wtype"),
            ideal_rate=(ir if ir not in (None, "") else None),
            next_watcher=b.get("next_watcher"),
            ok_factor=b.get("ok_factor"), ng_factor=b.get("ng_factor"),
            assigned=_as,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/watchers/delete")
def watcher_delete(request: Request, _mve_body=Depends(_json_body_dep)):
    """4.0.417 — remove a watcher from the station list (history kept)."""
    try:
        b = _req_body(_mve_body)
        return wm.delete_watcher(str(b.get("register_id", "")))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/watchers/oee")
def watcher_oee(register_id: str, since_ms: int = 0, until_ms: int = 0):
    """OEE (Availability × Performance × Quality) for one machine over a window."""
    try:
        return wm.compute_oee(register_id, since_ms or None, until_ms or None)
    except Exception as e:
        logger.warning("oee failed: %s", e)
        return JSONResponse({"error": str(e), "oee": None}, status_code=500)


@router.post("/api/watchers/line_name")
def watcher_line_name(request: Request, _mve_body=Depends(_json_body_dep)):
    """4.0.420 — name a line (free text, any language; "" = show the station chain).
    `station` = any station of that line."""
    try:
        b = _req_body(_mve_body)
        return wm.set_line_name(b.get("station"), b.get("name"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/watchers/camera_options")
def watcher_camera_options():
    """4.0.419 — cameras + object classes (with their Store flag) for a camera station."""
    try:
        return wm.camera_options()
    except Exception as e:
        logger.warning("camera options failed: %s", e)
        return JSONResponse({"error": str(e), "cameras": [], "classes": []}, status_code=500)


@router.post("/api/watchers/camera_station")
def watcher_camera_station(request: Request, _mve_body=Depends(_json_body_dep)):
    """4.0.419 — create (no register_id) or update a camera station: name, cameras
    (ids; empty = all), ok_classes, ng_classes."""
    try:
        b = _req_body(_mve_body)
        return wm.save_camera_station(b.get("register_id"), b.get("name"), b.get("cameras"),
                                      b.get("ok_classes"), b.get("ng_classes"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/watchers/camera_count_now")
def watcher_camera_count_now():
    """4.0.419 — run one camera-station counting pass now (it also runs every minute)."""
    try:
        return {"counted": wm.count_camera_stations()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/watchers/lines")
def watcher_lines(since_ms: int = 0, until_ms: int = 0):
    """4.0.418 — OEE per line (stations chained by next station) with each station's
    input / OK / NG and the gap to the station before it."""
    try:
        return wm.compute_lines(since_ms or None, until_ms or None)
    except Exception as e:
        logger.warning("line oee failed: %s", e)
        return JSONResponse({"error": str(e), "lines": []}, status_code=500)
