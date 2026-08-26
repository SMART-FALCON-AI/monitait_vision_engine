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
from fastapi.responses import JSONResponse

from services import watcher_metrics as wm

logger = logging.getLogger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------- #
# ingestion — Watcher-Jet compatible
# --------------------------------------------------------------------------- #
@router.post("/api/factory/update-watcher/")
async def update_watcher(request: Request):
    """No-image path: the device POSTs its JSON report and expects HTTP 200."""
    try:
        payload = await request.json()
        r = wm.ingest(payload)
        return {"status": "ok", **r}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.warning("update-watcher ingest failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/factory/image-update-watcher-data/")
async def image_update_watcher_data(request: Request):
    """Image path, step 1: store the metrics and hand back an id the device then
    references when it uploads the image."""
    try:
        payload = await request.json()
        r = wm.ingest(payload)
        return {"status": "ok", "_id": r.get("id"), "register_id": r.get("register_id")}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.warning("image-update-watcher-data ingest failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/factory/image-update-watcher/")
async def image_update_watcher(request: Request):
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


@router.post("/api/watchers/name")
async def watcher_name(request: Request):
    """Give a machine a friendly display name."""
    try:
        b = await request.json()
        wm.set_name(str(b.get("register_id", "")), str(b.get("name", "")))
        return {"ok": True}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
