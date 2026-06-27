"""Health, status, and system metrics routes."""

import os
import time
import json
import logging
import requests
import psutil
import psycopg2
from datetime import datetime
from redis import Redis

from fastapi import APIRouter, Request
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    FileResponse,
    HTMLResponse,
    RedirectResponse,
)

import config as cfg_module  # Direct module ref for runtime-updated values (SSE)

from config import (
    YOLO_INFERENCE_URL,
    EJECTOR_OFFSET,
    EJECTOR_ENABLED,
    HISTOGRAM_ENABLED,
    WATCHER_USB,
    SERIAL_BAUDRATE,
    SERIAL_MODE,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    DETECTION_EVENTS_REDIS_KEY,
    DETECTION_EVENTS_MAX_SIZE,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared requests session for health checks
_session = requests.Session()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get("/api/config/db_status")
async def config_db_status():
    """3.22.0 — show which storage backend is currently authoritative.

    Returns:
      backend: "db" if mve_config_kv has rows, "file" otherwise
      db_reachable: True if we can talk to Postgres at all
      key_count: number of top-level keys in DB (None if unreachable)
      file_exists: True if the legacy .env.prepared_query_data file is on disk
    """
    out = {"backend": "file", "db_reachable": False, "key_count": None, "file_exists": False}
    try:
        from services.config_db import history_count
        cnt = history_count()
        if cnt is not None:
            out["db_reachable"] = True
            out["key_count"] = cnt
            if cnt > 0:
                out["backend"] = "db"
    except Exception:
        pass
    try:
        from config import DATA_FILE
        out["file_exists"] = os.path.exists(DATA_FILE)
    except Exception:
        pass
    return JSONResponse(content=out)


@router.get("/api/shipments/next_code")
async def next_shipment_code(request: Request, local_date: str = ""):
    """4.0.9 — shortened format `yymmddXXYYZZZ` (13 chars, was 15):

      * yymmdd    — date (2-digit year)
      * XX        — 2-digit ID of the currently-active capture state
                    (alphabetical index across all defined states, padded)
      * YY        — 2-digit ID of the currently-active inference pipeline
      * ZZZ       — chronological per-day counter

    State/pipeline IDs are derived from their alphabetical position in the
    saved config so the mapping is deterministic across restarts and across
    sites (provided the same names exist). When the operator adds a new
    state, the IDs of existing states are stable as long as the new name
    sorts after them.

    Hybrid counter for ZZZ (unchanged from 3.25.0):
      1) Redis-backed per-day counter `shipment_seq:<yyyymmdd>` — back-to-back
         🎲 clicks return 001, 002, 003 even before the operator saves.
      2) DB scan over `inference_results` for any shipment matching today's
         date — if a higher ZZZ is already in the DB (cross-operator,
         restored backup, or pre-3.25.12 short codes), use that.
      3) next ZZZ = max(redis, db) + 1; persist to redis with EOD expiry.
    """
    from datetime import datetime as _dt
    # 4.0.9 — date prefix shortened to 2-digit year (yymmdd) per operator request.
    # Redis key keeps the full year so per-day counters never collide across
    # century boundaries even if the same yymmdd recurs in 2126.
    # 4.0.38 — accept `local_date` (yymmdd) from the caller. MVE containers
    # run in UTC; in Iran (+0330) the container thinks it's still yesterday
    # for the first ~3.5h of the operator's day. When the JS sends its own
    # local yymmdd we honour that, with strict regex validation so a bogus
    # input can't poison Redis keys or the response prefix. Falls back to
    # container clock for legacy callers (no param).
    import re as _re_ld
    _ld = str(local_date or "").strip()
    if _ld and _re_ld.fullmatch(r"\d{6}", _ld):
        today = _ld
        # Reconstruct yyyymmdd from yymmdd by assuming 21xx for now (the
        # legacy code also assumed current century). This is only used as a
        # Redis key so a wrong century guess just means a fresh per-day
        # counter — no data loss.
        century = _dt.now().strftime("%Y")[:2]
        today_full = century + _ld
    else:
        today = _dt.now().strftime("%y%m%d")              # 6 chars for the code
        today_full = _dt.now().strftime("%Y%m%d")         # 8 chars for the Redis key
    redis_key = f"shipment_seq:{today_full}"

    # --- A) Derive 2-digit IDs for the currently-active state + pipeline.
    state_id = 0
    pipeline_id = 0
    try:
        from config import load_service_config as _lsc
        svc = _lsc() or {}
        states = svc.get("states") or {}
        active_state = svc.get("current_state_name") or "default"
        if isinstance(states, dict) and states:
            ordered = sorted(states.keys())
            try:
                state_id = ordered.index(active_state)
            except ValueError:
                state_id = 0
        # Pipeline: prefer the live PipelineManager (covers in-memory swap
        # before service_config sync), fall back to config.
        try:
            pm = request.app.state.pipeline_manager
            pipelines = getattr(pm, "pipelines", None) or {}
            cur_pipe = getattr(pm, "current_pipeline", None)
            active_pipeline = (cur_pipe.name if cur_pipe and hasattr(cur_pipe, "name") else None) or ""
            if isinstance(pipelines, dict) and pipelines:
                ordered_p = sorted(pipelines.keys())
                if active_pipeline in ordered_p:
                    pipeline_id = ordered_p.index(active_pipeline)
        except Exception:
            # Fall back to config-only lookup
            try:
                pipelines = svc.get("pipelines") or {}
                active_pipeline = svc.get("current_pipeline_name") or ""
                if isinstance(pipelines, dict) and pipelines:
                    ordered_p = sorted(pipelines.keys())
                    if active_pipeline in ordered_p:
                        pipeline_id = ordered_p.index(active_pipeline)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"shipment_code: state/pipeline lookup failed: {e}")

    state_id = max(0, min(99, int(state_id)))
    pipeline_id = max(0, min(99, int(pipeline_id)))
    prefix = f"{today}{state_id:02d}{pipeline_id:02d}"  # 10-char prefix (yymmdd + XX + YY)

    # --- 1) Redis counter (or 0 if unreachable / unset)
    redis_max = 0
    try:
        from redis import Redis
        from config import REDIS_DB
        r = Redis("redis", 6379, db=REDIS_DB)
        v = r.get(redis_key)
        if v:
            redis_max = int(v.decode() if isinstance(v, bytes) else v)
    except Exception as e:
        logger.debug(f"shipment_seq redis read failed: {e}")

    # --- 2) DB scan — find max ZZZ over BOTH new (15-char) and legacy (11-char)
    # codes for today, so the counter never collides with a pre-3.25.12 code.
    db_max = 0
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                # 4.0.9 new format: yymmddXXYYZZZ (13 chars).
                cur.execute(
                    """
                    SELECT MAX(CAST(RIGHT(image_path, 3) AS INTEGER))
                    FROM (
                      SELECT DISTINCT SPLIT_PART(image_path, '/', 1) AS image_path
                      FROM inference_results
                      WHERE time::date = CURRENT_DATE
                    ) ships
                    WHERE image_path ~ %s
                    """,
                    (rf'^{today}\d{{7}}$',),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    db_max = max(db_max, int(row[0]))
                # 3.25.12 format: yyyymmddXXYYZZZ (15 chars).
                cur.execute(
                    """
                    SELECT MAX(CAST(RIGHT(image_path, 3) AS INTEGER))
                    FROM (
                      SELECT DISTINCT SPLIT_PART(image_path, '/', 1) AS image_path
                      FROM inference_results
                      WHERE time::date = CURRENT_DATE
                    ) ships
                    WHERE image_path ~ %s
                    """,
                    (rf'^{today_full}\d{{7}}$',),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    db_max = max(db_max, int(row[0]))
                # 3.25.0 legacy: yyyymmddZZZ (11 chars).
                cur.execute(
                    """
                    SELECT MAX(CAST(SUBSTRING(image_path FROM %s) AS INTEGER))
                    FROM (
                      SELECT DISTINCT SPLIT_PART(image_path, '/', 1) AS image_path
                      FROM inference_results
                      WHERE time::date = CURRENT_DATE
                    ) ships
                    WHERE image_path ~ %s
                    """,
                    (rf'^{today_full}(\d{{3}})$', rf'^{today_full}\d{{3}}$'),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    db_max = max(db_max, int(row[0]))
                cur.close()
            finally:
                try:
                    release_db_connection(conn)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"next_shipment_code db lookup failed: {e}")

    # --- 3) next = max(redis, db) + 1, persist in redis with EOD expiry
    next_num = max(redis_max, db_max) + 1
    try:
        from redis import Redis
        from config import REDIS_DB
        r = Redis("redis", 6379, db=REDIS_DB)
        now = _dt.now()
        end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0)
        ttl = max(60, int((end_of_day - now).total_seconds()))
        r.setex(redis_key, ttl, str(next_num))
    except Exception as e:
        logger.debug(f"shipment_seq redis write failed: {e}")

    code = f"{prefix}{next_num:03d}"  # yymmddXXYYZZZ (13 chars)
    return JSONResponse(content={
        "code": code, "today": today,
        "state_id": state_id, "pipeline_id": pipeline_id,
        "seq": next_num, "redis_seq": redis_max, "db_seq": db_max,
    })


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    try:
        watcher = request.app.state.watcher_instance
        pipeline_manager = request.app.state.pipeline_manager

        # Check if watcher is available
        device_ok = watcher is not None and watcher.health_check

        # Check Redis connection
        redis_ok = False
        if watcher and watcher.redis_connection:
            try:
                watcher.redis_connection.redis_connection.ping()
                redis_ok = True
            except:
                pass

        # Check cameras dynamically (verify device node exists for USB)
        cameras_status = {}
        if watcher and hasattr(watcher, 'cameras'):
            for cam_id, cam in watcher.cameras.items():
                cam_path = watcher.camera_paths[cam_id - 1] if cam_id <= len(watcher.camera_paths) else None
                is_usb = True
                if hasattr(watcher, 'camera_metadata') and cam_id in watcher.camera_metadata:
                    is_usb = watcher.camera_metadata[cam_id].get("type", "usb") == "usb"
                device_exists = os.path.exists(cam_path) if is_usb and cam_path else True
                cameras_status[f"cam_{cam_id}"] = (cam is not None and cam.success and device_exists) if cam else False

        # Check serial availability
        serial_available = watcher.serial_available if watcher else False

        # Check YOLO/Gradio inference availability
        # Use pipeline manager's current model URL if available (more accurate than static global)
        yolo_ok = False
        active_inference_url = YOLO_INFERENCE_URL
        if pipeline_manager and pipeline_manager.get_current_model():
            active_inference_url = pipeline_manager.get_current_model().inference_url
        try:
            # Check if Gradio client is initialized (for HuggingFace URLs)
            if "hf.space" in active_inference_url or "huggingface" in active_inference_url:
                gradio_health_status = "unknown"
                gradio_needs_restart = False

                # Try comprehensive health check
                try:
                    # 1. Check if health endpoint responds
                    base_url = active_inference_url.split('/api/')[0] if '/api/' in active_inference_url else active_inference_url.rsplit('/', 1)[0]
                    health_url = f"{base_url}/api/health"

                    health_response = _session.get(health_url, timeout=5)

                    if health_response.status_code == 200:
                        # 2. Check recent successful inference from Redis
                        last_detection_ts = 0.0
                        if watcher and watcher.redis_connection:
                            try:
                                cached_ts = watcher.redis_connection.redis_connection.get("gradio_last_detection_timestamp")
                                if cached_ts:
                                    last_detection_ts = float(cached_ts.decode('utf-8'))
                            except Exception as e:
                                logger.warning(f"Failed to read detection timestamp from Redis: {e}")

                        # 3. Determine health based on recent activity
                        if last_detection_ts > 0:
                            age = time.time() - last_detection_ts
                            if age < 60.0:  # Less than 1 minute - healthy
                                yolo_ok = True
                                gradio_health_status = "healthy"
                            elif age < 300.0:  # Less than 5 minutes - warning
                                yolo_ok = True
                                gradio_health_status = "warning"
                            else:  # More than 5 minutes - may need restart
                                yolo_ok = False
                                gradio_health_status = "stale"
                                gradio_needs_restart = True
                            logger.info(f"Gradio health: {gradio_health_status}, last inference {age:.1f}s ago")
                        else:
                            # Health endpoint OK but no recent inferences
                            yolo_ok = True
                            gradio_health_status = "idle"
                            logger.info(f"Gradio health: {gradio_health_status} (no recent inferences)")
                    else:
                        raise Exception(f"Health endpoint returned {health_response.status_code}")

                except Exception as e:
                    # Health endpoint failed - space may be sleeping or needs restart
                    yolo_ok = False
                    gradio_health_status = "offline"
                    gradio_needs_restart = True
                    logger.error(f"Gradio health check failed: {e}")

                # Store status in Redis for monitoring
                try:
                    if watcher and watcher.redis_connection:
                        watcher.redis_connection.redis_connection.set("gradio_health_status", gradio_health_status)
                        watcher.redis_connection.redis_connection.set("gradio_needs_restart", str(gradio_needs_restart))
                except:
                    pass
            else:
                # For traditional YOLO endpoint, try a quick ping
                # Try /health first, then fall back to just checking if server responds (even 404 means it's alive)
                health_url = active_inference_url.replace('/detect/', '/').replace('/detect', '/')
                response = requests.get(health_url, timeout=2)
                # Any response means server is running (200, 404, etc.)
                yolo_ok = response.status_code in [200, 404, 405]
        except Exception as e:
            logger.error(f"[HEALTH] Error checking YOLO status: {e}")
            yolo_ok = False

        # Consider healthy if redis works and either serial is connected or we're in camera-only mode
        status_code = 200 if redis_ok else 503

        # Get Gradio-specific health info from Redis
        gradio_health_info = {}
        if "hf.space" in active_inference_url or "huggingface" in active_inference_url:
            try:
                if watcher and watcher.redis_connection:
                    health_status = watcher.redis_connection.redis_connection.get("gradio_health_status")
                    needs_restart = watcher.redis_connection.redis_connection.get("gradio_needs_restart")
                    gradio_health_info = {
                        "gradio_status": health_status.decode('utf-8') if health_status else "unknown",
                        "gradio_needs_restart": needs_restart.decode('utf-8') == "True" if needs_restart else False
                    }
            except:
                pass

        # Check TimescaleDB/PostgreSQL connection
        db_ok = False
        try:
            conn = psycopg2.connect(host=POSTGRES_HOST, port=POSTGRES_PORT, database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=2)
            conn.close()
            db_ok = True
        except Exception:
            db_ok = False

        response_content = {
            "status": "healthy" if status_code == 200 else "unhealthy",
            "version": request.app.version,
            "device": "connected" if device_ok else ("camera-only" if not serial_available else "disconnected"),
            "serial": "connected" if serial_available else "not available",
            "redis": "connected" if redis_ok else "disconnected",
            "db": "connected" if db_ok else "disconnected",
            "yolo": "connected" if yolo_ok else "not available",
            "cameras": cameras_status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add Gradio-specific info if available
        if gradio_health_info:
            response_content.update(gradio_health_info)

        return JSONResponse(
            status_code=status_code,
            content=response_content
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )


# ---------------------------------------------------------------------------
# GET /api/health/watchdog  (4.0.47)
# Frame-recency liveness check for the docker-compose healthcheck.
#
# /health (above) checks dependency CONNECTIVITY (redis/db/yolo all up). That
# returns 200 even when MVE has stopped processing frames — the 2026-06-20
# incident on khoy is the canonical case: redis/db/yolo healthy, watcher
# alive, but the encoder USB drop killed the inference pipeline. The dashboard
# stayed at "Last 24 hours / No data" while /health said "healthy".
#
# This endpoint exposes the actual frame-throughput as seen by Redis lists
# `inf_frame_timestamps` and `cap_frame_timestamps` (populated by
# services/detection.py:~1166 and services/watcher.py:~1310 respectively).
# It does NOT 503 by default — measurement only, so external callers can
# decide their own thresholds. The docker-compose healthcheck wraps this in
# a shell that 503s when both counters are stale beyond N seconds AND the
# operator has the line marked "running" (i.e. shipment != no_shipment).
# Marked-stopped shipments are fine to have zero frames.
# ---------------------------------------------------------------------------
@router.get("/api/health/watchdog")
async def health_watchdog(request: Request, stale_seconds: int = 120):
    """Frame-recency liveness for docker healthcheck + external monitoring.

    Returns:
      now_ts: int — current epoch seconds (server clock)
      last_inference_ts / last_capture_ts: float|null — newest ts in Redis
      seconds_since_inference / seconds_since_capture: int|null — now_ts - last
      inference_in_last_n / capture_in_last_n: int — count in last `stale_seconds`
      shipment: str — current active shipment (so the healthcheck can ignore
                no_shipment cases where empty is expected)
      stalled: bool — True if shipment != no_shipment AND both counters are 0
                in last `stale_seconds`. The compose-level healthcheck uses
                this as its 503 gate.
    """
    import time as _t
    now_ts = int(_t.time())
    cutoff = now_ts - max(5, int(stale_seconds))

    inf_last = None
    cap_last = None
    inf_count = 0
    cap_count = 0
    shipment = "no_shipment"
    try:
        watcher = request.app.state.watcher_instance
        if watcher and watcher.redis_connection:
            r = watcher.redis_connection.redis_connection
            # Newest timestamp is at index 0 (most-recent-first LPUSH pattern
            # used in detection.py / watcher.py — see existing usage in the
            # SSE stream around line 638).
            try:
                _i0 = r.lindex("inf_frame_timestamps", 0)
                if _i0 is not None:
                    inf_last = float(_i0.decode() if isinstance(_i0, bytes) else _i0)
            except Exception:
                pass
            try:
                _c0 = r.lindex("cap_frame_timestamps", 0)
                if _c0 is not None:
                    cap_last = float(_c0.decode() if isinstance(_c0, bytes) else _c0)
            except Exception:
                pass
            try:
                ship = r.get("shipment")
                if ship:
                    shipment = ship.decode() if isinstance(ship, bytes) else str(ship)
            except Exception:
                pass
            # Count recent frames by walking the head of the list (cheap — Redis
            # LRANGE 0..2000 returns at worst a few thousand small strings).
            for key, sink in (("inf_frame_timestamps", "inf"), ("cap_frame_timestamps", "cap")):
                try:
                    raw = r.lrange(key, 0, 2000) or []
                    n = 0
                    for v in raw:
                        try:
                            ts = float(v.decode() if isinstance(v, bytes) else v)
                        except Exception:
                            continue
                        if ts >= cutoff:
                            n += 1
                        else:
                            # List is newest-first so once we hit a stale ts
                            # everything after is older too — early-out.
                            break
                    if sink == "inf":
                        inf_count = n
                    else:
                        cap_count = n
                except Exception:
                    pass
    except Exception as _e:
        logger.debug(f"watchdog: redis read failed: {_e}")

    sec_since_inf = (now_ts - int(inf_last)) if inf_last else None
    sec_since_cap = (now_ts - int(cap_last)) if cap_last else None

    # 4.0.48 — robust "stuck" definition (avoids false-positive restarts on
    # idle lines):
    #
    #   stuck = captures > 0 AND inferences == 0
    #
    # That is the ONLY shape that means a real downstream failure (frames
    # are being grabbed but the inference pipeline is dropping/ignoring
    # them). Every other combination is either fine or genuinely-idle:
    #
    #   captures=0 inferences=0 → idle (line stopped, no parts) → NOT stalled
    #   captures>0 inferences>0 → healthy production
    #   captures=0 inferences>0 → impossible by construction; treat healthy
    #
    # Earlier rule (`inf==0 AND cap==0` triggers restart) was too eager —
    # it would restart MVE every time the operator paused for a shift
    # change. Restart loops on a healthy-but-idle machine waste startup
    # time + reset shipment / in-memory state.
    #
    # Also still gated on shipment != "no_shipment" so a parked line stays
    # peaceful regardless of capture/inference counts.
    stuck_inference_broken = (cap_count > 0 and inf_count == 0)
    stalled = bool(
        shipment and shipment != "no_shipment"
        and stuck_inference_broken
    )

    # Diagnostic reason string so the docker healthcheck log lines and
    # /api/health/watchdog consumers know why (or why not).
    if shipment in (None, "", "no_shipment"):
        reason = "shipment-parked"
    elif cap_count == 0 and inf_count == 0:
        reason = "idle"           # NOT stuck — line just isn't capturing
    elif stuck_inference_broken:
        reason = "inference-stuck"  # frames in, no frames out → restart
    else:
        reason = "healthy"

    return JSONResponse(content={
        "now_ts": now_ts,
        "stale_seconds": stale_seconds,
        "shipment": shipment,
        "last_inference_ts": inf_last,
        "last_capture_ts": cap_last,
        "seconds_since_inference": sec_since_inf,
        "seconds_since_capture": sec_since_cap,
        "inference_in_last_n": inf_count,
        "capture_in_last_n": cap_count,
        "stalled": stalled,
        "reason": reason,
    })


# ---------------------------------------------------------------------------
# GET /api/system/metrics
# ---------------------------------------------------------------------------
@router.get("/api/system/metrics")
async def get_system_metrics(request: Request):
    """Get system resource usage (CPU, RAM, Disk)."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)

        # Memory (RAM) usage
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_used_gb = mem.used / (1024**3)
        mem_available_gb = mem.available / (1024**3)
        mem_percent = mem.percent

        # Disk usage — use the data volume mount (reflects real host disk)
        import os
        _disk_path = '/code/raw_images' if os.path.exists('/code/raw_images') else '/'
        disk = psutil.disk_usage(_disk_path)
        disk_total_gb = disk.total / (1024**3)
        disk_used_gb = disk.used / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        disk_percent = disk.percent

        return JSONResponse(content={
            "cpu": {
                "percent": round(cpu_percent, 1),
                "cores_logical": cpu_count_logical,
                "cores_physical": cpu_count_physical
            },
            "memory": {
                "total_gb": round(mem_total_gb, 2),
                "used_gb": round(mem_used_gb, 2),
                "available_gb": round(mem_available_gb, 2),
                "percent": round(mem_percent, 1)
            },
            "disk": {
                "total_gb": round(disk_total_gb, 2),
                "used_gb": round(disk_used_gb, 2),
                "free_gb": round(disk_free_gb, 2),
                "percent": round(disk_percent, 1)
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# GET / - Redirect to /status
# ---------------------------------------------------------------------------
@router.get("/")
async def root_redirect(request: Request):
    """Redirect root to status page."""
    return RedirectResponse(url="/status")


# ---------------------------------------------------------------------------
# GET /status - Serve status HTML page
# ---------------------------------------------------------------------------
@router.get("/status")
async def status_page(request: Request):
    """Serve the status page with control panel.

    The HTML itself is served no-cache (always fresh), but the referenced
    `/static/js/*.js` files are served by the StaticFiles mount which the
    browser caches aggressively. After an MVE upgrade the fresh HTML would
    keep loading a STALE cached JS (e.g. an audio.js without the Store
    checkbox), which looked like "the feature didn't deploy". To fix it
    permanently we rewrite each local static JS/CSS include to carry a
    `?v=<app-version>` query string — the version changes every release, so
    the browser is forced to refetch the JS whenever MVE is upgraded.
    """
    import re
    try:
        with open("static/status.html", "r", encoding="utf-8") as f:
            html = f.read()
        ver = getattr(request.app, "version", None) or "dev"
        # Append ?v=<ver> to local /static/js/*.js and /static/css/*.css includes.
        html = re.sub(r'(/static/(?:js|css)/[\w\-./]+\.(?:js|css))"', rf'\1?v={ver}"', html)
        return HTMLResponse(content=html, headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        })
    except Exception as e:
        logger.error(f"Status page error: {e}")
        return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)


# ---------------------------------------------------------------------------
# GET /api/status - API status data
# ---------------------------------------------------------------------------
@router.get("/api/status")
async def api_status(request: Request):
    """API endpoint for real-time status data."""
    try:
        watcher = request.app.state.watcher_instance

        return JSONResponse(content={
            "encoder_value": watcher.encoder_value if watcher else 0,
            "pps": getattr(watcher, "pulses_per_second", 0) if watcher else 0,
            "ppm": getattr(watcher, "pulses_per_minute", 0) if watcher else 0,
            "downtime_seconds": getattr(watcher, "downtime_seconds", 0) if watcher else 0,
            "ok_counter": getattr(watcher, "ok_counter", 0) if watcher else 0,
            "ng_counter": getattr(watcher, "ng_counter", 0) if watcher else 0,
            "eject_ok_counter": getattr(watcher, "eject_ok_counter", 0) if watcher else 0,
            "eject_ng_counter": getattr(watcher, "eject_ng_counter", 0) if watcher else 0,
            "analog_value": getattr(watcher, "analog_value", 0) if watcher else 0,
            "power_value": getattr(watcher, "power_value", 0) if watcher else 0,
            "is_moving": watcher.is_moving if watcher else False,
            "shipment": watcher.shipment if watcher else "no_shipment",
            "ejector_queue_length": len(watcher.ejection_queue) if watcher else 0,
            "ejector_running": watcher.ejector_running if watcher else False,
            "ejector_offset": cfg_module.EJECTOR_OFFSET,
            "ejector_delay": cfg_module.EJECTOR_DELAY,
            "ejector_enabled": cfg_module.EJECTOR_ENABLED,
            "histogram_enabled": HISTOGRAM_ENABLED,
            "status": {
                "U": getattr(watcher, "u_status", False) if watcher else False,
                "B": getattr(watcher, "b_status", False) if watcher else False,
                "warning": getattr(watcher, "warning_status", False) if watcher else False,
                "raw": getattr(watcher, "status_value", 0) if watcher else 0,
            } if watcher else {"U": False, "B": False, "warning": False, "raw": 0},
            "verbose_data": {
                "OOD": getattr(watcher, "ok_offset_delay", 0) if watcher else 0,
                "ODP": getattr(watcher, "ok_duration_pulses", 0) if watcher else 0,
                "ODL": getattr(watcher, "ok_duration_percent", 0) if watcher else 0,
                "OEF": getattr(watcher, "ok_encoder_factor", 0) if watcher else 0,
                "NOD": getattr(watcher, "ng_offset_delay", 0) if watcher else 0,
                "NDP": getattr(watcher, "ng_duration_pulses", 0) if watcher else 0,
                "NDL": getattr(watcher, "ng_duration_percent", 0) if watcher else 0,
                "NEF": getattr(watcher, "ng_encoder_factor", 0) if watcher else 0,
                "EXT": getattr(watcher, "external_reset", 0) if watcher else 0,
                "BUD": getattr(watcher, "baud_rate", SERIAL_BAUDRATE) if watcher else SERIAL_BAUDRATE,
                "DWT": getattr(watcher, "downtime_threshold", 0) if watcher else 0,
            },
            "data": watcher.data if watcher else {},
            "serial_device": {
                "connected": getattr(watcher, "serial_available", False) if watcher else False,
                "port": getattr(watcher, "serial_port", WATCHER_USB) if watcher else WATCHER_USB,
                "baudrate": getattr(watcher, "serial_baudrate", SERIAL_BAUDRATE) if watcher else SERIAL_BAUDRATE,
                "mode": getattr(watcher, "serial_mode", SERIAL_MODE) if watcher else SERIAL_MODE,
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"API status error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# GET /api/status/stream - SSE stream
# ---------------------------------------------------------------------------
@router.get("/api/status/stream")
async def status_stream(request: Request):
    """Server-Sent Events stream for real-time status updates."""
    watcher = request.app.state.watcher_instance

    def generate():
        last_data = None
        _db_ok = False
        _db_check_time = 0

        # Re-read inference buffers from Redis each iteration (cross-process safe)
        while True:
            try:
                # Periodic DB health check (every 30s)
                if time.time() - _db_check_time > 30:
                    try:
                        _c = psycopg2.connect(host=POSTGRES_HOST, port=POSTGRES_PORT, database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD, connect_timeout=2)
                        _c.close()
                        _db_ok = True
                    except Exception:
                        _db_ok = False
                    _db_check_time = time.time()

                # Fetch inference timing data from Redis
                inference_times = []
                inf_frame_timestamps = []
                cap_frame_timestamps = []
                try:
                    redis_conn = Redis("redis", 6379, db=cfg_module.REDIS_DB)
                    times_raw = redis_conn.lrange("inference_times", 0, -1)
                    inference_times = [float(t.decode('utf-8')) for t in times_raw if t]
                    inf_raw = redis_conn.lrange("inf_frame_timestamps", 0, -1)
                    inf_frame_timestamps = [float(t.decode('utf-8')) for t in inf_raw if t]
                    cap_raw = redis_conn.lrange("cap_frame_timestamps", 0, -1)
                    cap_frame_timestamps = [float(t.decode('utf-8')) for t in cap_raw if t]
                except Exception:
                    pass

                # True FPS: count frames in last 5 seconds
                _now = time.time()
                _inf_recent = [t for t in inf_frame_timestamps if _now - t <= 5.0]
                _inf_fps = (len(_inf_recent) - 1) / (max(_inf_recent) - min(_inf_recent)) if len(_inf_recent) >= 2 and max(_inf_recent) > min(_inf_recent) else 0
                _cap_recent = [t for t in cap_frame_timestamps if _now - t <= 5.0]
                _cap_fps = (len(_cap_recent) - 1) / (max(_cap_recent) - min(_cap_recent)) if len(_cap_recent) >= 2 and max(_cap_recent) > min(_cap_recent) else 0

                # Build current status data
                current_data = {
                    "encoder_value": watcher.encoder_value if watcher else 0,
                    "pps": getattr(watcher, "pulses_per_second", 0) if watcher else 0,
                    "ppm": getattr(watcher, "pulses_per_minute", 0) if watcher else 0,
                    "downtime_seconds": getattr(watcher, "downtime_seconds", 0) if watcher else 0,
                    "ok_counter": getattr(watcher, "ok_counter", 0) if watcher else 0,
                    "ng_counter": getattr(watcher, "ng_counter", 0) if watcher else 0,
                    "eject_ok_counter": getattr(watcher, "eject_ok_counter", 0) if watcher else 0,
                    "eject_ng_counter": getattr(watcher, "eject_ng_counter", 0) if watcher else 0,
                    "analog_value": getattr(watcher, "analog_value", 0) if watcher else 0,
                    "power_value": getattr(watcher, "power_value", 0) if watcher else 0,
                    "is_moving": watcher.is_moving if watcher else False,
                    "shipment": watcher.shipment if watcher else "no_shipment",
                    "ejector_queue_length": len(watcher.ejection_queue) if watcher else 0,
                    "ejector_running": watcher.ejector_running if watcher else False,
                    "ejector_enabled": cfg_module.EJECTOR_ENABLED,
                    "ejector_offset": cfg_module.EJECTOR_OFFSET,
                    "ejector_delay": cfg_module.EJECTOR_DELAY,
                    "status": {
                        "U": getattr(watcher, "u_status", False) if watcher else False,
                        "B": getattr(watcher, "b_status", False) if watcher else False,
                        "warning": getattr(watcher, "warning_status", False) if watcher else False,
                        "raw": getattr(watcher, "status_value", 0) if watcher else 0,
                    },
                    "verbose_data": {
                        "OOD": getattr(watcher, "ok_offset_delay", 0) if watcher else 0,
                        "ODP": getattr(watcher, "ok_duration_pulses", 0) if watcher else 0,
                        "ODL": getattr(watcher, "ok_duration_percent", 0) if watcher else 0,
                        "OEF": getattr(watcher, "ok_encoder_factor", 0) if watcher else 0,
                        "NOD": getattr(watcher, "ng_offset_delay", 0) if watcher else 0,
                        "NDP": getattr(watcher, "ng_duration_pulses", 0) if watcher else 0,
                        "NDL": getattr(watcher, "ng_duration_percent", 0) if watcher else 0,
                        "NEF": getattr(watcher, "ng_encoder_factor", 0) if watcher else 0,
                        "EXT": getattr(watcher, "external_reset", 0) if watcher else 0,
                        "BUD": getattr(watcher, "baud_rate", SERIAL_BAUDRATE) if watcher else SERIAL_BAUDRATE,
                        "DWT": getattr(watcher, "downtime_threshold", 0) if watcher else 0,
                    },
                    "serial_device": {
                        "connected": getattr(watcher, "serial_available", False) if watcher else False,
                        "port": getattr(watcher, "serial_port", WATCHER_USB) if watcher else WATCHER_USB,
                        "baudrate": getattr(watcher, "serial_baudrate", SERIAL_BAUDRATE) if watcher else SERIAL_BAUDRATE,
                        "mode": getattr(watcher, "serial_mode", SERIAL_MODE) if watcher else SERIAL_MODE,
                    },
                    # Camera status
                    "cameras": {
                        name: {
                            "active": hasattr(cam, 'frame') and cam.frame is not None,
                            "name": getattr(cam, 'name', name),
                            "type": getattr(cam, 'camera_type', 'unknown'),
                        }
                        for name, cam in (watcher.cameras.items() if watcher and watcher.cameras else {})
                    },
                    # Inference stats
                    "inference": {
                        "avg_time_ms": sum(inference_times) / len(inference_times) if inference_times else 0,
                        "inference_fps": round(_inf_fps, 2),
                        "capture_fps": round(_cap_fps, 2),
                        "samples": len(inference_times),
                    },
                    # Health status
                    "health": {
                        "redis": "connected" if (watcher and watcher.redis_connection and watcher.redis_connection.redis_connection) else "disconnected",
                        "db": "connected" if _db_ok else "disconnected",
                        "gradio_needs_restart": False,  # Will be updated if needed
                    },
                    # Latest detection event for audio notification
                    "detection_event": None,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Check for new detection events from Redis
                try:
                    if watcher and watcher.redis_connection and watcher.redis_connection.redis_connection:
                        # Get recent events from Redis (LRANGE gets from front of list)
                        raw_events = watcher.redis_connection.redis_connection.lrange(DETECTION_EVENTS_REDIS_KEY, 0, 10)
                        if raw_events:
                            current_time = time.time()
                            # Parse events and find most recent one less than 5 seconds old
                            for raw_event in raw_events:
                                try:
                                    event = json.loads(raw_event)
                                    age = current_time - event.get("timestamp", 0)
                                    if age < 5.0:
                                        current_data["detection_event"] = event
                                        break
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.error(f"[SSE] Error reading detection events from Redis: {e}")

                # Send data if changed or if there's a detection event
                data_json = json.dumps(current_data)
                if data_json != last_data or current_data["detection_event"]:
                    yield f"data: {data_json}\n\n"
                    last_data = data_json

                time.sleep(0.1)  # Check for changes every 100ms
            except Exception as e:
                logger.error(f"SSE status stream error: {e}")
                time.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
