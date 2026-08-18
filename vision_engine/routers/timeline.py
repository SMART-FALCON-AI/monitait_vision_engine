import pathlib
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response, FileResponse
from config import (load_data_file, save_data_file, TIMELINE_REDIS_PREFIX,
                    TIMELINE_THUMBNAIL_WIDTH, TIMELINE_THUMBNAIL_HEIGHT,
                    latest_detections, latest_detections_timestamp,
                    DETECTION_EVENTS_REDIS_KEY)
import config as cfg_module
import cv2, numpy as np, time, json, pickle, logging, os
from redis import Redis
from services.detection import evaluate_eject_from_detections
from services.render import draw_detection_on

logger = logging.getLogger(__name__)
router = APIRouter()

# v4.0.75 — in-memory TTL cache for the expensive chart endpoints.
# /api/detection_charts, /api/quality/heatmap, /api/area_stats,
# /api/detection_stats each scan inference_results via
# jsonb_array_elements(detections) which is single-digit-seconds-per-call
# on a busy hypertable. The dashboard polls each of them every 5–10 s from
# multiple tabs — under load, six concurrent 15-s SQL scans saturate
# Chrome's per-origin socket limit and every OTHER endpoint (including
# POST /api/config for shipment save) queues behind them.
#
# A 20-second in-memory TTL cache turns a 15-s scan into a sub-millisecond
# lookup for every subsequent poll within that window — dashboards refresh
# instantly, the DB only sees one query per 20 s per parameter permutation.
# `_endpoint_cache` maps a hash-of-params → (timestamp, payload_dict).
# LRU-capped at 256 entries so a distinct query permutation storm can't
# balloon the process footprint.
import hashlib as _hashlib
_ENDPOINT_CACHE_TTL_SEC = 20.0
_ENDPOINT_CACHE_MAX = 256
_endpoint_cache: dict = {}

def _endpoint_cache_key(prefix: str, *args) -> str:
    """Deterministic cache key from the endpoint name + all query params."""
    raw = prefix + "|" + "|".join(str(a) for a in args)
    return _hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()

def _endpoint_cache_get(key: str):
    v = _endpoint_cache.get(key)
    if v is None:
        return None
    ts, payload = v
    if time.time() - ts < _ENDPOINT_CACHE_TTL_SEC:
        return payload
    return None

def _endpoint_cache_put(key: str, payload) -> None:
    global _endpoint_cache
    if len(_endpoint_cache) >= _ENDPOINT_CACHE_MAX:
        # Drop oldest half — cheaper than a strict LRU and avoids unbounded growth.
        items = sorted(_endpoint_cache.items(), key=lambda kv: kv[1][0])
        _endpoint_cache = dict(items[len(items) // 2:])
    _endpoint_cache[key] = (time.time(), payload)

# Absolute path to raw_images for path traversal protection
_RAW_IMAGES_ROOT = pathlib.Path("raw_images").resolve()

# Header strip height in pixels
_HEADER_HEIGHT = 28


def _list_shipment_dirs():
    """Return shipment-id directory names under raw_images/. Used to surface
    freshly-created shipments in the dropdown even before any detection has
    been recorded against them (otherwise the DB DISTINCT query returns
    only shipments that already have rows in the time window).
    """
    try:
        if not _RAW_IMAGES_ROOT.exists():
            return []
        return sorted([p.name for p in _RAW_IMAGES_ROOT.iterdir() if p.is_dir() and not p.name.startswith('.')])
    except Exception:
        return []


def _unpack_timeline_entry(frame_data):
    """Unpack a timeline Redis entry, handling 2/3/4-tuple formats.
    Returns (ts, jpeg_bytes, detections, meta) where meta is a dict with d_path, encoder.
    """
    unpacked = pickle.loads(frame_data)
    n = len(unpacked)
    if n >= 4:
        ts, jpeg_bytes, detections, meta = unpacked[0], unpacked[1], unpacked[2], unpacked[3]
    elif n == 3:
        ts, jpeg_bytes, detections = unpacked
        meta = {}
    else:
        ts, jpeg_bytes = unpacked[0], unpacked[1]
        detections = None
        meta = {}
    return ts, jpeg_bytes, detections, meta if isinstance(meta, dict) else {}


def _build_header_strip(columns_meta, thumb_width, num_columns, procedures, current_encoder, ejector_offset):
    """Render a metadata header strip above the timeline composite.

    Args:
        columns_meta: list of dicts per column with 'detections', 'encoder', 'should_eject'
        thumb_width: pixel width of each column
        num_columns: number of columns
        procedures: eject procedure rules
        current_encoder: live encoder value from watcher
        ejector_offset: EJECTOR_OFFSET config value

    Returns:
        numpy array of shape (_HEADER_HEIGHT, total_width, 3)
    """
    total_width = thumb_width * num_columns
    header = np.zeros((_HEADER_HEIGHT, total_width, 3), dtype=np.uint8)
    header[:] = (50, 50, 50)  # dark gray default

    ejector_target = (current_encoder - ejector_offset) if current_encoder is not None else None

    for i, col in enumerate(columns_meta):
        x_start = i * thumb_width
        x_end = x_start + thumb_width

        # Evaluate eject from detections
        all_dets = col.get('all_detections', [])
        if procedures and all_dets:
            should_eject, reasons = evaluate_eject_from_detections(all_dets, procedures)
        else:
            should_eject = None  # unknown
            reasons = []

        # Store reasons back in col for metadata
        col['should_eject'] = should_eject
        col['eject_reasons'] = reasons

        # Background color
        if should_eject is True:
            color = (0, 0, 180)  # red BGR
        elif should_eject is False:
            color = (0, 140, 0)  # green BGR
        else:
            color = (60, 60, 60)  # gray

        cv2.rectangle(header, (x_start, 0), (x_end - 1, _HEADER_HEIGHT - 1), color, -1)

        # Thin separator line between columns
        cv2.line(header, (x_end - 1, 0), (x_end - 1, _HEADER_HEIGHT - 1), (30, 30, 30), 1)

        # Encoder + timestamp on same line (top-left, small)
        enc_val = col.get('encoder')
        col_ts = col.get('ts')
        label_parts = []
        if enc_val is not None:
            label_parts.append(str(int(enc_val)))
        if col_ts is not None:
            label_parts.append(time.strftime("%H:%M:%S", time.localtime(col_ts)))
        if label_parts:
            label = " | ".join(label_parts)
            cv2.putText(header, label, (x_start + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (220, 220, 220), 1)

        # Eject reason text (bottom-left, small, only for red columns)
        if should_eject is True and reasons:
            reason_text = "; ".join(reasons)
            # Truncate to fit column width (~6px per char at scale 0.25)
            max_chars = max(5, (thumb_width - 6) // 5)
            if len(reason_text) > max_chars:
                reason_text = reason_text[:max_chars - 2] + ".."
            cv2.putText(header, reason_text, (x_start + 2, _HEADER_HEIGHT - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        # Ejector position marker (downward triangle)
        if ejector_target is not None and enc_val is not None:
            if abs(enc_val - ejector_target) <= max(1, abs(ejector_offset) * 0.1):
                cx = x_start + thumb_width // 2
                pts = np.array([[cx - 5, _HEADER_HEIGHT - 8], [cx + 5, _HEADER_HEIGHT - 8], [cx, _HEADER_HEIGHT - 1]], np.int32)
                cv2.fillPoly(header, [pts], (0, 100, 255))  # orange triangle

        # Capture marker (rightmost column on page 0 — caller marks it)
        if col.get('is_capture'):
            cx = x_start + thumb_width // 2
            cv2.circle(header, (cx, _HEADER_HEIGHT - 5), 3, (255, 200, 0), -1)  # cyan dot

    return header


def generate_detection_stream():
    """Generator function that yields MJPEG frames from latest detections."""
    import glob
    last_file = None
    last_mtime = 0

    while True:
        try:
            # Find all _DETECTED.jpg files in raw_images directory
            pattern = os.path.join("raw_images", "**", "*_DETECTED.jpg")
            detected_files = glob.glob(pattern, recursive=True)

            if detected_files:
                # Get the most recent file by modification time
                latest_file = max(detected_files, key=os.path.getmtime)
                current_mtime = os.path.getmtime(latest_file)

                # Only send if file has changed or it's a new file
                if latest_file != last_file or current_mtime != last_mtime:
                    # Read the image
                    with open(latest_file, 'rb') as f:
                        image_data = f.read()

                    # Yield the frame in MJPEG format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')

                    last_file = latest_file
                    last_mtime = current_mtime

            # Wait a bit before checking for new frames (2 times per second)
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error in detection stream: {e}")
            time.sleep(1)


@router.get("/api/latest_detections")
def get_latest_detections(request: Request):
    """Get the most recent detection events for audio notification (polling fallback)."""
    try:
        watcher = request.app.state.watcher_instance
        if watcher and watcher.redis_connection and watcher.redis_connection.redis_connection:
            # Get most recent events from Redis
            raw_events = watcher.redis_connection.redis_connection.lrange(DETECTION_EVENTS_REDIS_KEY, 0, 10)
            if raw_events:
                current_time = time.time()
                for raw_event in raw_events:
                    try:
                        event = json.loads(raw_event)
                        age = current_time - event.get("timestamp", 0)
                        if age < 5.0:
                            return {
                                "has_detection": True,
                                "event": event
                            }
                    except json.JSONDecodeError:
                        continue

        return {"has_detection": False}
    except Exception as e:
        logger.error(f"Error getting latest detections: {e}")
        return {"has_detection": False, "error": str(e)}


@router.get("/timeline_feed")
def timeline_feed(request: Request):
    """MJPEG stream showing camera timeline (history view).

    Shows a grid: columns = cameras, rows = time (newest at bottom).
    Lightweight alternative to full stitching service.
    """
    def generate():
        while True:
            try:
                composite = get_timeline_composite(request)
                if composite:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + composite + b'\r\n')
                time.sleep(0.5)  # Update 2 FPS for timeline
            except Exception as e:
                logger.error(f"Timeline feed error: {e}")
                time.sleep(1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


def get_timeline_composite(request: Request):
    """Concatenate all camera frames: horizontal axis = time, vertical axis = cameras."""
    try:
        # Get configuration (with defaults)
        try:
            tl_config = request.app.state.timeline_config
            quality = tl_config.get('image_quality', 85)
            image_rotation = tl_config.get('image_rotation', 0)
        except Exception as e:
            logger.debug(f"Could not load timeline config for stitching, using defaults: {e}")
            tl_config = None
            quality = 85
            image_rotation = 0

        # Create Redis connection
        redis_client = Redis("redis", 6379, db=cfg_module.REDIS_DB)

        # Find all camera timeline keys (timeline:1, timeline:2, ...)
        all_keys = redis_client.keys(f"{TIMELINE_REDIS_PREFIX}*")
        if not all_keys:
            logger.debug("Timeline: No frames in Redis")
            return None

        # Sort keys numerically by camera ID
        def extract_cam_id(key):
            k = key.decode() if isinstance(key, bytes) else key
            try:
                return int(k.split(":")[-1])
            except ValueError:
                return 0
        camera_order = tl_config.get('camera_order', 'normal') if tl_config else 'normal'
        if camera_order == 'custom':
            custom_order_str = tl_config.get('custom_camera_order', '') if tl_config else ''
            order_list = [int(x.strip()) for x in custom_order_str.split(',') if x.strip().isdigit()]
            if order_list:
                def _custom_key(key):
                    cid = extract_cam_id(key)
                    try:
                        return order_list.index(cid)
                    except ValueError:
                        return len(order_list) + cid
                all_keys.sort(key=_custom_key)
            else:
                all_keys.sort(key=extract_cam_id)
        else:
            all_keys.sort(key=extract_cam_id, reverse=(camera_order == 'reverse'))

        # Build one horizontal row per camera (left=oldest, right=newest)
        camera_rows = []
        for key in all_keys:
            frames_raw = redis_client.lrange(key, 0, -1)
            if not frames_raw:
                continue
            # Unpack all frames (handle both 2-tuple and 3-tuple formats)
            all_frames = []
            for frame_data in frames_raw:
                unpacked = pickle.loads(frame_data)
                ts = unpacked[0]
                jpeg_bytes = unpacked[1]
                all_frames.append((ts, jpeg_bytes))
            # Sort chronologically
            all_frames.sort(key=lambda x: x[0])
            row_frames = []
            for ts, jpeg_bytes in all_frames:
                thumb = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if thumb is not None:
                    if image_rotation == 90:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                    elif image_rotation == 180:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_180)
                    elif image_rotation == 270:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    row_frames.append(thumb)
            if row_frames:
                camera_rows.append(np.hstack(row_frames))

        if not camera_rows:
            logger.debug("Timeline: No valid frames")
            return None

        # Pad all rows to the same width before vertical stacking
        max_width = max(row.shape[1] for row in camera_rows)
        padded_rows = []
        for row in camera_rows:
            if row.shape[1] < max_width:
                pad = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                row = np.hstack([row, pad])
            padded_rows.append(row)

        # Stack vertically: camera 1 on top, camera 2 below, etc.
        composite = np.vstack(padded_rows)

        logger.info(f"Timeline: Stitched {len(camera_rows)} camera row(s) into composite")

        # Encode final composite with configured quality
        _, jpeg = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

    except Exception as e:
        logger.error(f"Error generating timeline composite: {e}", exc_info=True)
        return None


@router.get("/timeline_image")
def timeline_image(request: Request, page: int = 0):
    """Get timeline frames for a specific page. Horizontal = time, vertical = cameras.

    Args:
        page: Page number (0=latest frames, 1=previous page, etc.)
    """
    try:
        # Get configuration (with defaults)
        try:
            tl_config = request.app.state.timeline_config
            quality = tl_config.get('image_quality', 85)
            frames_per_page = tl_config.get('num_rows', 10)
            image_rotation = tl_config.get('image_rotation', 0)
        except:
            quality = 85
            frames_per_page = 10
            image_rotation = 0

        # Create Redis connection
        redis_client = Redis("redis", 6379, db=cfg_module.REDIS_DB)

        # Find all camera timeline keys
        all_keys = redis_client.keys(f"{TIMELINE_REDIS_PREFIX}*")
        if not all_keys:
            placeholder = np.zeros((180, 240, 3), dtype=np.uint8)
            placeholder[:] = (60, 60, 60)
            cv2.putText(placeholder, "No frames yet", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return Response(content=jpeg.tobytes(), media_type="image/jpeg")

        # Sort keys numerically by camera ID
        def extract_cam_id(key):
            k = key.decode() if isinstance(key, bytes) else key
            try:
                return int(k.split(":")[-1])
            except ValueError:
                return 0
        all_keys.sort(key=extract_cam_id)

        # Determine pagination from the camera with the most frames
        max_total_frames = 0
        camera_frames_raw = {}
        for key in all_keys:
            frames_raw = redis_client.lrange(key, 0, -1)
            cam_id = extract_cam_id(key)
            camera_frames_raw[cam_id] = frames_raw if frames_raw else []
            max_total_frames = max(max_total_frames, len(camera_frames_raw[cam_id]))

        total_pages = max(1, (max_total_frames + frames_per_page - 1) // frames_per_page)
        if page < 0 or page >= total_pages:
            page = 0

        # Get bbox rendering config
        try:
            show_bbox = tl_config.get('show_bounding_boxes', True)
            obj_filters = tl_config.get('object_filters', {})
        except:
            show_bbox = True
            obj_filters = {}

        # Get procedures config for eject evaluation
        try:
            procedures = tl_config.get('procedures', [])
        except Exception:
            procedures = []

        # Get current encoder from watcher for ejector marker
        try:
            watcher_inst = request.app.state.watcher_instance
            current_encoder = getattr(watcher_inst, 'encoder_value', None)
        except Exception:
            current_encoder = None
        ejector_offset = int(os.environ.get("EJECTOR_OFFSET", 0))

        # Build one horizontal row per camera for this page
        # Also collect per-column metadata across all cameras
        camera_order = tl_config.get('camera_order', 'normal')
        if camera_order == 'custom':
            custom_order_str = tl_config.get('custom_camera_order', '')
            order_list = [int(x.strip()) for x in custom_order_str.split(',') if x.strip().isdigit()]
            if order_list:
                def _custom_cam_key(cid):
                    try:
                        return order_list.index(cid)
                    except ValueError:
                        return len(order_list) + cid
                sorted_cam_ids = sorted(camera_frames_raw.keys(), key=_custom_cam_key)
            else:
                sorted_cam_ids = sorted(camera_frames_raw.keys())
        else:
            sorted_cam_ids = sorted(camera_frames_raw.keys(), reverse=(camera_order == 'reverse'))
        cam_page_slices = {}  # cam_id -> list of (ts, jpeg_bytes, detections, meta)
        num_columns = 0

        for cam_id in sorted_cam_ids:
            frames_raw = camera_frames_raw[cam_id]

            # Unpack all frames and sort by capture timestamp (chronological order)
            all_frames = []
            for frame_data in frames_raw:
                ts, jpeg_bytes, detections, meta = _unpack_timeline_entry(frame_data)
                all_frames.append((ts, jpeg_bytes, detections, meta))
            all_frames.sort(key=lambda x: x[0])

            # Paginate after sorting
            total = len(all_frames)
            end_index = total - (page * frames_per_page)
            start_index = max(0, end_index - frames_per_page)
            page_slice = all_frames[start_index:end_index] if end_index > 0 else []

            cam_page_slices[cam_id] = page_slice
            num_columns = max(num_columns, len(page_slice))

        # Build per-column metadata (across all cameras)
        columns_meta = []
        for col_idx in range(num_columns):
            col_all_dets = []
            col_d_paths = {}
            col_encoder = None
            col_ts = None
            for cam_id in sorted_cam_ids:
                ps = cam_page_slices.get(cam_id, [])
                if col_idx < len(ps):
                    ts, jpeg_bytes, detections, meta = ps[col_idx]
                    if detections:
                        col_all_dets.extend(detections)
                    d_path = meta.get('d_path') if meta else None
                    col_d_paths[str(cam_id)] = d_path
                    if meta and meta.get('encoder') is not None:
                        col_encoder = meta['encoder']
                    if col_ts is None:
                        col_ts = ts
            columns_meta.append({
                'all_detections': col_all_dets,
                'd_paths': col_d_paths,
                'encoder': col_encoder,
                'ts': col_ts,
                'is_capture': (page == 0 and col_idx == num_columns - 1),
            })

        # Render camera rows with bboxes and rotation
        camera_rows = []
        thumb_width = 0
        thumb_height = 240

        for cam_id in sorted_cam_ids:
            page_slice = cam_page_slices.get(cam_id, [])
            row_frames = []
            for ts, jpeg_bytes, detections, meta in page_slice:
                thumb = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if thumb is not None:
                    # Draw bounding boxes if enabled and detections exist
                    if show_bbox and detections:
                        # Scale bbox coords from original resolution to thumbnail
                        th, tw = thumb.shape[:2]
                        orig_h = meta.get('orig_h', th) if meta else th
                        orig_w = meta.get('orig_w', tw) if meta else tw
                        sx = tw / orig_w if orig_w else 1
                        sy = th / orig_h if orig_h else 1
                        kv_y = 4
                        for det in detections:
                            kv_y = draw_detection_on(
                                thumb, det, sx=sx, sy=sy, kv_y=kv_y,
                                bbox_thickness=2, obj_filters=obj_filters,
                            )

                    # Apply configurable rotation (0, 90, 180, 270)
                    if image_rotation == 90:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                    elif image_rotation == 180:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_180)
                    elif image_rotation == 270:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    row_frames.append(thumb)
                    # Record thumb dimensions (after rotation)
                    if thumb_width == 0:
                        thumb_height, thumb_width = thumb.shape[:2]

            if row_frames:
                camera_rows.append(np.hstack(row_frames))

        if not camera_rows:
            placeholder = np.zeros((180, 240, 3), dtype=np.uint8)
            placeholder[:] = (60, 60, 60)
            cv2.putText(placeholder, "No frames", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return Response(content=jpeg.tobytes(), media_type="image/jpeg")

        # Build metadata header strip
        header = _build_header_strip(columns_meta, thumb_width, num_columns, procedures, current_encoder, ejector_offset)

        # Pad all rows to same width and vstack (cameras stacked vertically)
        max_width = max(row.shape[1] for row in camera_rows)
        # Ensure header matches width
        if header.shape[1] < max_width:
            pad = np.zeros((_HEADER_HEIGHT, max_width - header.shape[1], 3), dtype=np.uint8)
            header = np.hstack([header, pad])
        elif header.shape[1] > max_width:
            header = header[:, :max_width]

        padded_rows = [header]
        for row in camera_rows:
            if row.shape[1] < max_width:
                pad = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                row = np.hstack([row, pad])
            padded_rows.append(row)

        composite = np.vstack(padded_rows)

        logger.info(f"Timeline page {page}: {len(camera_rows)} camera(s), {max_total_frames} max frames")

        _, jpeg = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error getting timeline image: {e}")
        placeholder = np.zeros((180, 240, 3), dtype=np.uint8)
        placeholder[:] = (60, 60, 60)
        _, jpeg = cv2.imencode('.jpg', placeholder)
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/api/timeline_count")
def timeline_count(request: Request):
    """Get the total number of frames and pages in the timeline (across all cameras)."""
    try:
        # Get configuration
        try:
            tl_config = request.app.state.timeline_config
            rows_per_page = tl_config.get('num_rows', 10)
        except:
            rows_per_page = 10

        redis_client = Redis("redis", 6379, db=cfg_module.REDIS_DB)
        all_keys = redis_client.keys(f"{TIMELINE_REDIS_PREFIX}*")
        # Use the camera with most frames to determine page count
        max_frames = 0
        total_all = 0
        num_cameras = 0
        for key in all_keys:
            n = redis_client.llen(key)
            if n > 0:
                num_cameras += 1
                total_all += n
                if n > max_frames:
                    max_frames = n
        total_pages = (max_frames + rows_per_page - 1) // rows_per_page if max_frames > 0 else 0

        return {
            "total_frames": total_all,
            "total_pages": total_pages,
            "rows_per_page": rows_per_page,
            "num_cameras": num_cameras,
            "max_frames_per_camera": max_frames
        }
    except Exception as e:
        logger.error(f"Error getting timeline count: {e}")
        return {"total_frames": 0, "total_pages": 0, "rows_per_page": 10}


# 3.21.12 — module-level cache for per-class confidence baselines (p50, p95).
# Recomputed on demand if older than _BASELINE_TTL_SEC.
_baseline_cache = {"computed_at": 0.0, "baselines": {}}
_BASELINE_TTL_SEC = 3600  # 1 hour


# 3.21.13 — verdict thresholds (configurable later via UI; in-code defaults for now).
# score >= RELEASE_SCORE: green light, ship.
# RELEASE_SCORE > score >= REINSPECT_SCORE: yellow, re-inspect before ship.
# score < REINSPECT_SCORE: red, hold; do not ship without QA sign-off.
#
# 4.0.103 — these are now the FALLBACKS only. Real values are read from
# `service_config` in `.env.prepared_query_data` per request (see helpers
# below) so the operator can tune them via the Process tab without a code
# push. `_quality_thresholds()` returns the live values and is what every
# scoring code path should use — NEVER read the module-level constants
# directly except as a default.
QUALITY_RELEASE_SCORE = 85
QUALITY_REINSPECT_SCORE = 60
QUALITY_MIN_ROWS = 100         # < this many detections in the window → verdict = PENDING
QUALITY_MIN_ENCODER_SPAN = 100  # encoder span < this → verdict = PENDING (unit-less; site-tuned)


def _quality_thresholds():
    """Return {release, reinspect, min_rows, min_span} — read live from
    service_config so operator changes take effect immediately with no
    restart. On any read failure we fall back to the module-level
    constants above so scoring never crashes."""
    try:
        from config import load_service_config as _lsc
        svc = _lsc() or {}
        return {
            "release":   float(svc.get("quality_release_score",   QUALITY_RELEASE_SCORE)),
            "reinspect": float(svc.get("quality_reinspect_score", QUALITY_REINSPECT_SCORE)),
            "min_rows":  int(svc.get("quality_min_rows",           QUALITY_MIN_ROWS)),
            "min_span":  float(svc.get("quality_min_encoder_span", QUALITY_MIN_ENCODER_SPAN)),
        }
    except Exception:
        return {
            "release":   float(QUALITY_RELEASE_SCORE),
            "reinspect": float(QUALITY_REINSPECT_SCORE),
            "min_rows":  int(QUALITY_MIN_ROWS),
            "min_span":  float(QUALITY_MIN_ENCODER_SPAN),
        }


def _compute_conf_baselines():
    """Per-class confidence baselines over last 7 days of stored detections.

    3.21.24 — now returns per-class AND per-camera percentiles plus a p5 so
    operators can drive per-camera min_confidence sliders off real-world data.

    Shape:
      {
        "TB": {
          "p5": 0.08, "p50": 0.12, "p95": 0.14, "n": 14524,           # overall
          "by_camera": {
            "1": {"p5": 0.14, "p50": 0.22, "p95": 0.31, "n": 6203},
            "2": {"p5": 0.06, "p50": 0.09, "p95": 0.12, "n": 8321}
          }
        },
        ...
      }
    """
    import time as _t
    if _t.time() - _baseline_cache["computed_at"] < _BASELINE_TTL_SEC and _baseline_cache["baselines"]:
        return _baseline_cache["baselines"]

    from services.db import get_db_connection, release_db_connection
    out = {}
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return _baseline_cache.get("baselines", {})
        cur = conn.cursor()
        # Overall (no camera grouping)
        cur.execute(
            """
            SELECT (det->>'name') AS cls,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->>'confidence')::float) AS p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->>'confidence')::float) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->>'confidence')::float) AS p95,
                   COUNT(*) AS n
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL '7 days'
              AND (det->>'confidence') IS NOT NULL
            GROUP BY (det->>'name')
            HAVING COUNT(*) >= 5
            """
        )
        for cls, p5, p50, p95, n in cur.fetchall():
            if cls:
                out[str(cls)] = {
                    "p5":  round(float(p5  or 0.0), 4),
                    "p50": round(float(p50 or 0.0), 4),
                    "p95": round(float(p95 or 0.0), 4),
                    "n":   int(n or 0),
                    "by_camera": {},
                }
        # Per-camera breakdown
        cur.execute(
            """
            SELECT (det->>'name') AS cls,
                   (det->>'_cam') AS cam,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->>'confidence')::float) AS p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->>'confidence')::float) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->>'confidence')::float) AS p95,
                   COUNT(*) AS n
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL '7 days'
              AND (det->>'confidence') IS NOT NULL
              AND (det->>'_cam') IS NOT NULL
            GROUP BY (det->>'name'), (det->>'_cam')
            HAVING COUNT(*) >= 5
            """
        )
        for cls, cam, p5, p50, p95, n in cur.fetchall():
            if not cls or cam is None:
                continue
            cls_s = str(cls)
            # If this class wasn't in the overall pass (rare — e.g. only one cam
            # so the per-class HAVING failed), seed it.
            if cls_s not in out:
                out[cls_s] = {"p5": 0.0, "p50": 0.0, "p95": 0.0, "n": 0, "by_camera": {}}
            out[cls_s]["by_camera"][str(cam)] = {
                "p5":  round(float(p5  or 0.0), 4),
                "p50": round(float(p50 or 0.0), 4),
                "p95": round(float(p95 or 0.0), 4),
                "n":   int(n or 0),
            }
        cur.close()
    except Exception as e:
        logger.warning(f"conf_baselines compute failed: {e}")
    finally:
        if conn is not None:
            try:
                release_db_connection(conn)
            except Exception:
                pass

    _baseline_cache["baselines"] = out
    _baseline_cache["computed_at"] = _t.time()
    return out


@router.get("/api/conf_baselines")
def get_conf_baselines():
    """Per-class confidence baselines (p50, p95, n) from last 7 days of stored
    detections. Cached for 1 hour. Used by the Process tab to display a
    read-only badge under each class card showing 'normal range'."""
    return JSONResponse(content={"baselines": _compute_conf_baselines()})


@router.post("/api/conf_baselines/recompute")
def recompute_conf_baselines():
    """Force-invalidate the baseline cache (used by admin/dev tools)."""
    _baseline_cache["computed_at"] = 0.0
    return JSONResponse(content={"baselines": _compute_conf_baselines()})


@router.get("/api/color_drift")
def get_color_drift(window: str = "7d"):
    """3.22.4 — Per-class CIELAB color stats (absolute L*, a*, b* percentiles).

    Replaces the ΔE drift readout that 3.22.2 originally shipped. Absolute
    LAB values are more diagnostic: an operator looking at `L: 50 · 54 · 58`
    can tell at a glance both what the typical color IS and how wide its
    spread is — a single ΔE number couldn't do that. And when something
    drifts, the operator can see WHICH channel moved (L = exposure/dye fade,
    a = green↔red, b = blue↔yellow).

    For each class with stored detections carrying `lab_color`, returns:
      - L / a / b: {p5, p50, p95} of each CIELAB component over the window
      - n: count of detections with a stored lab_color
      - by_camera: same metrics broken out per camera

    The operator opts a class in via the ColorE checkbox on the Process
    tab. The annotator then extracts `lab_color` for that class only,
    so this endpoint only sees samples for classes color_e was on for.
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days", "30d": "30 days"}
    interval = _windows.get(window, "7 days")
    out: dict = {}
    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content={"window": window, "classes": {}})
        cur = conn.cursor()
        # v4.0.102 — same 3s-timeout rescue as area_stats. This endpoint's
        # per-class percentile_cont pass over ~900k jsonb_array_elements + the
        # SECOND per-camera GROUP BY (same rows re-scanned per class) was the
        # heaviest analytical query on khoy; audit clocked it at 18 s to
        # timeout+fail. 15 s covers the common case; for very large windows
        # (30d) the operator will still see empty payload if we exceed 15 s,
        # which is the correct failure mode (materialised view is the real fix).
        cur.execute("SET LOCAL statement_timeout = 15000")
        # Overall (no camera grouping): one row per class, with p5/p50/p95 for L, a, b
        cur.execute(
            """
            SELECT (det->>'name') AS cls,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->'lab_color'->>0)::float) AS L_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->'lab_color'->>0)::float) AS L_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->'lab_color'->>0)::float) AS L_p95,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->'lab_color'->>1)::float) AS a_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->'lab_color'->>1)::float) AS a_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->'lab_color'->>1)::float) AS a_p95,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->'lab_color'->>2)::float) AS b_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->'lab_color'->>2)::float) AS b_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->'lab_color'->>2)::float) AS b_p95,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY COALESCE((det->>'E')::float, |/(
                       POWER((det->'lab_color'->>0)::float, 2) +
                       POWER((det->'lab_color'->>1)::float, 2) +
                       POWER((det->'lab_color'->>2)::float, 2)
                   ))) AS E_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY COALESCE((det->>'E')::float, |/(
                       POWER((det->'lab_color'->>0)::float, 2) +
                       POWER((det->'lab_color'->>1)::float, 2) +
                       POWER((det->'lab_color'->>2)::float, 2)
                   ))) AS E_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY COALESCE((det->>'E')::float, |/(
                       POWER((det->'lab_color'->>0)::float, 2) +
                       POWER((det->'lab_color'->>1)::float, 2) +
                       POWER((det->'lab_color'->>2)::float, 2)
                   ))) AS E_p95,
                   COUNT(*) AS n
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s
              AND (det ? 'lab_color')
              AND (det->>'name') IS NOT NULL
            GROUP BY (det->>'name')
            HAVING COUNT(*) >= 5
            """,
            (interval,),
        )
        for row in cur.fetchall():
            cls, Lp5, Lp50, Lp95, ap5, ap50, ap95, bp5, bp50, bp95, Ep5, Ep50, Ep95, n = row
            if not cls:
                continue
            out[str(cls)] = {
                "n": int(n or 0),
                "L": {"p5": round(float(Lp5  or 0), 2),
                      "p50": round(float(Lp50 or 0), 2),
                      "p95": round(float(Lp95 or 0), 2)},
                "a": {"p5": round(float(ap5  or 0), 2),
                      "p50": round(float(ap50 or 0), 2),
                      "p95": round(float(ap95 or 0), 2)},
                "b": {"p5": round(float(bp5  or 0), 2),
                      "p50": round(float(bp50 or 0), 2),
                      "p95": round(float(bp95 or 0), 2)},
                "E": {"p5": round(float(Ep5  or 0), 2),
                      "p50": round(float(Ep50 or 0), 2),
                      "p95": round(float(Ep95 or 0), 2)},
                "by_camera": {},
            }
        # Per-camera
        cur.execute(
            """
            SELECT (det->>'name') AS cls,
                   (det->>'_cam') AS cam,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->'lab_color'->>0)::float) AS L_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->'lab_color'->>0)::float) AS L_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->'lab_color'->>0)::float) AS L_p95,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->'lab_color'->>1)::float) AS a_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->'lab_color'->>1)::float) AS a_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->'lab_color'->>1)::float) AS a_p95,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY (det->'lab_color'->>2)::float) AS b_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY (det->'lab_color'->>2)::float) AS b_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY (det->'lab_color'->>2)::float) AS b_p95,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY COALESCE((det->>'E')::float, |/(
                       POWER((det->'lab_color'->>0)::float, 2) +
                       POWER((det->'lab_color'->>1)::float, 2) +
                       POWER((det->'lab_color'->>2)::float, 2)
                   ))) AS E_p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY COALESCE((det->>'E')::float, |/(
                       POWER((det->'lab_color'->>0)::float, 2) +
                       POWER((det->'lab_color'->>1)::float, 2) +
                       POWER((det->'lab_color'->>2)::float, 2)
                   ))) AS E_p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY COALESCE((det->>'E')::float, |/(
                       POWER((det->'lab_color'->>0)::float, 2) +
                       POWER((det->'lab_color'->>1)::float, 2) +
                       POWER((det->'lab_color'->>2)::float, 2)
                   ))) AS E_p95,
                   COUNT(*) AS n
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s
              AND (det ? 'lab_color')
              AND (det->>'name') IS NOT NULL
              AND (det->>'_cam') IS NOT NULL
            GROUP BY (det->>'name'), (det->>'_cam')
            HAVING COUNT(*) >= 5
            """,
            (interval,),
        )
        for row in cur.fetchall():
            cls, cam, Lp5, Lp50, Lp95, ap5, ap50, ap95, bp5, bp50, bp95, Ep5, Ep50, Ep95, n = row
            cls_s = str(cls)
            if cls_s not in out:
                continue
            out[cls_s]["by_camera"][str(cam)] = {
                "n": int(n or 0),
                "L": {"p5": round(float(Lp5  or 0), 2),
                      "p50": round(float(Lp50 or 0), 2),
                      "p95": round(float(Lp95 or 0), 2)},
                "a": {"p5": round(float(ap5  or 0), 2),
                      "p50": round(float(ap50 or 0), 2),
                      "p95": round(float(ap95 or 0), 2)},
                "b": {"p5": round(float(bp5  or 0), 2),
                      "p50": round(float(bp50 or 0), 2),
                      "p95": round(float(bp95 or 0), 2)},
                "E": {"p5": round(float(Ep5  or 0), 2),
                      "p50": round(float(Ep50 or 0), 2),
                      "p95": round(float(Ep95 or 0), 2)},
            }
        cur.close()
    except Exception as e:
        logger.warning(f"color_drift compute failed: {e}")
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass
    return JSONResponse(content={"window": window, "classes": out})


@router.get("/api/area_stats")
def get_area_stats(window: str = "7d"):
    """3.22.3 — Per-class bbox-area percentiles over the requested window.

    Computed straight from the stored xmin/xmax/ymin/ymax — no per-class
    extraction toggle is needed because the data is already there. The
    Area checkbox on each card is a pure display preference.

    Returns p5 / p50 / p95 of bbox area (pixel²) for each class with
    enough detections in the window, with a per-camera breakdown.
    Operators use it to tune: "TB normally covers 12k–35k px²; the ones
    coming in at 2k px² are probably false positives, raise min_conf".
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days", "30d": "30 days"}
    interval = _windows.get(window, "7 days")
    out: dict = {}
    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content={"window": window, "classes": {}})
        cur = conn.cursor()
        # v4.0.102 — same 3s-timeout rescue as detection_stats/detection_charts.
        # jsonb_array_elements over 900k+ rows with two percentile_cont
        # aggregations was aborting at 3s and returning `classes={}` silently.
        cur.execute("SET LOCAL statement_timeout = 15000")
        # Overall percentiles
        cur.execute(
            """
            SELECT (det->>'name') AS cls,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY COALESCE(
                       (det->>'area')::float,
                       ((det->>'xmax')::float - (det->>'xmin')::float) *
                       ((det->>'ymax')::float - (det->>'ymin')::float))) AS p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY COALESCE(
                       (det->>'area')::float,
                       ((det->>'xmax')::float - (det->>'xmin')::float) *
                       ((det->>'ymax')::float - (det->>'ymin')::float))) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY COALESCE(
                       (det->>'area')::float,
                       ((det->>'xmax')::float - (det->>'xmin')::float) *
                       ((det->>'ymax')::float - (det->>'ymin')::float))) AS p95,
                   COUNT(*) AS n
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s
              AND (det ? 'xmin') AND (det ? 'xmax')
              AND (det ? 'ymin') AND (det ? 'ymax')
              AND (det->>'name') IS NOT NULL
            GROUP BY (det->>'name')
            HAVING COUNT(*) >= 5
            """,
            (interval,),
        )
        for cls, p5, p50, p95, n in cur.fetchall():
            if not cls:
                continue
            out[str(cls)] = {
                "p5":  int(round(float(p5  or 0))),
                "p50": int(round(float(p50 or 0))),
                "p95": int(round(float(p95 or 0))),
                "n":   int(n or 0),
                "by_camera": {},
            }
        # Per-camera breakdown
        cur.execute(
            """
            SELECT (det->>'name') AS cls,
                   (det->>'_cam') AS cam,
                   percentile_cont(0.05) WITHIN GROUP (ORDER BY COALESCE(
                       (det->>'area')::float,
                       ((det->>'xmax')::float - (det->>'xmin')::float) *
                       ((det->>'ymax')::float - (det->>'ymin')::float))) AS p5,
                   percentile_cont(0.50) WITHIN GROUP (ORDER BY COALESCE(
                       (det->>'area')::float,
                       ((det->>'xmax')::float - (det->>'xmin')::float) *
                       ((det->>'ymax')::float - (det->>'ymin')::float))) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY COALESCE(
                       (det->>'area')::float,
                       ((det->>'xmax')::float - (det->>'xmin')::float) *
                       ((det->>'ymax')::float - (det->>'ymin')::float))) AS p95,
                   COUNT(*) AS n
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s
              AND (det ? 'xmin') AND (det ? 'xmax')
              AND (det ? 'ymin') AND (det ? 'ymax')
              AND (det->>'name') IS NOT NULL
              AND (det->>'_cam') IS NOT NULL
            GROUP BY (det->>'name'), (det->>'_cam')
            HAVING COUNT(*) >= 5
            """,
            (interval,),
        )
        for cls, cam, p5, p50, p95, n in cur.fetchall():
            cls_s = str(cls)
            if cls_s not in out:
                continue
            out[cls_s]["by_camera"][str(cam)] = {
                "p5":  int(round(float(p5  or 0))),
                "p50": int(round(float(p50 or 0))),
                "p95": int(round(float(p95 or 0))),
                "n":   int(n or 0),
            }
        cur.close()
    except Exception as e:
        logger.warning(f"area_stats compute failed: {e}")
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass
    return JSONResponse(content={"window": window, "classes": out})


@router.get("/api/active_classes")
def get_active_classes(window: str = "1h"):
    """Classes with at least one detection in the recent window — used by the
    Process tab "Show only active" filter (3.21.24).

    Returns:
      {
        "window": "1h",
        "names": ["TB", "mean_L", ...],      # de-duplicated, sorted by total count desc
        "by_camera": {"1": ["TB"], "2": ["TB", "mean_L"], ...},
        "counts":   {"TB": 14524, "mean_L": 312, ...},
      }
    """
    _windows = {"15m": "15 minutes", "1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
    interval = _windows.get(window, "1 hour")
    out = {"window": window, "names": [], "by_camera": {}, "counts": {}}
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=out)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT (det->>'name') AS cls, (det->>'_cam') AS cam, COUNT(*) AS n
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s
              AND (det->>'name') IS NOT NULL
            GROUP BY (det->>'name'), (det->>'_cam')
            """,
            (interval,),
        )
        for cls, cam, n in cur.fetchall():
            cls_s = str(cls)
            cnt = int(n or 0)
            out["counts"][cls_s] = out["counts"].get(cls_s, 0) + cnt
            if cam is not None:
                out["by_camera"].setdefault(str(cam), []).append(cls_s)
        cur.close()
        # Sort de-duplicated names by total count desc (busiest first)
        out["names"] = sorted(out["counts"].keys(), key=lambda k: -out["counts"][k])
        # De-dup the by_camera lists, preserve order
        for k, names in out["by_camera"].items():
            seen, uniq = set(), []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    uniq.append(n)
            out["by_camera"][k] = uniq
    except Exception as e:
        logger.warning(f"active_classes query failed: {e}")
    finally:
        try:
            release_db_connection(conn)
        except Exception:
            pass
    return JSONResponse(content=out)


def _compute_quality_payload(shipment: str = "", window: str = "24h") -> dict:
    """Shared computation for the score endpoint and PDF report.

    Returns the full payload dict (same shape the /api/shipment_quality_score
    endpoint serializes). Returns the `empty` payload if the DB is unreachable
    or has no rows for the requested window/shipment.
    """
    # 4.0.29d — added 30d / 90d so the "Score per shipment" chart can score
    # shipments older than 24h. Before, /api/quality/shipments listed 30 days
    # of shipments but their score was computed with a hard-coded 24h window,
    # so anything inactive for >24h came back score=None and rendered as a
    # zero-height bar.
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours",
                "7d": "7 days", "30d": "30 days", "90d": "90 days"}
    interval = _windows.get(window, "24 hours")
    # 4.0.196 — same override as /api/detection_charts + /api/shipment_spans.
    # A specific shipment being picked means the operator wants THAT shipment's
    # data regardless of the "Last N hours" preset. Widen to 90 days so an
    # older shipment (customer picked one from days back) still returns rows
    # instead of an empty Quality Score panel + "no data in this window yet."
    if shipment:
        interval = "90 days"
    ship_clause = "AND shipment = %s" if shipment else ""
    base_params = [interval] + ([shipment] if shipment else [])

    empty = {
        "shipment": shipment, "window": window,
        "score": None, "verdict": "NO_DATA",
        "impact_total": 0.0, "impact_by_class": {},
        "total_detections": 0, "top_defects": [],
        "encoder_min": None, "encoder_max": None, "encoder_span": 0,
        "encoder_unit": "encoder_unit",
        # v4.0.140 — both new and legacy fields present on the empty payload
        # so a caller iterating .get(...) uniformly against either sees None
        # instead of KeyError.
        "encoder_units_per_unit":  None,
        "encoder_units_per_meter": None,
        "impact_per_unit": 0.0, "impact_per_unit_label": "/unit",
        "first_ts": None, "last_ts": None, "duration_sec": 0,
        "throughput": 0.0, "throughput_label": "units/sec",
        "normalized_by": "none",
        "persisted": False,
        "ejection_impact": 0.0, "ejection_counts": {},
    }

    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        from config import load_service_config as _lsc
        conn = get_db_connection()
        if conn is None:
            return empty

        _svc = _lsc() or {}
        _audio = _svc.get("audio_settings", {}) or {}
        severities = {k: int(v.get("severity", 0) or 0) for k, v in _audio.items() if isinstance(v, dict)}
        # 3.25.8 — ejection procedures contribute severity too. Map procedure_name → severity (0–100).
        # Only procedures with Store=ON are written to ejection_events so other procedures have no rows
        # to score against; keying by name is safe (procedure rename = retroactive re-attribution, which
        # is the expected behavior for an analytics view).
        _procs = _svc.get("procedures", []) or []
        proc_severities = {
            str(p.get("name") or ""): int(p.get("severity") or 0)
            for p in _procs if isinstance(p, dict) and p.get("name")
        }
        encoder_unit = str(_svc.get("encoder_unit") or "encoder_unit")
        # v4.0.140 — read new `encoder_units_per_unit`; fall back to legacy
        # `encoder_units_per_meter` so calibration set before v4.0.140 still
        # drives the shipment-quality normalisation without operator action.
        _upu_raw = _svc.get("encoder_units_per_unit")
        if _upu_raw is None:
            _upu_raw = _svc.get("encoder_units_per_meter")
        try:
            encoder_units_per_meter = float(_upu_raw or 0)
        except (TypeError, ValueError):
            encoder_units_per_meter = 0.0
        encoder_units_per_unit = encoder_units_per_meter   # v4.0.140 alias for readability

        # v4.0.111 — also exclude classes the operator hid in the Process tab.
        # Same source (audio_settings[cls].show === False) that
        # detection_charts / quality/heatmap already use. Without this, an
        # operator-hidden class (e.g. `jean` — dominant frame-label not a
        # defect) still counted toward total_detections, top_defects, and
        # every downstream aggregate. `_compute_quality_payload` is the
        # heart of the score system so this change ripples cleanly.
        _hidden_by_process = sorted([
            str(cls) for cls, cfg in _audio.items()
            if isinstance(cfg, dict) and cfg.get("show") is False
        ])
        _hp_sql = ""
        _hp_args = []
        if _hidden_by_process:
            _hp_sql = " AND (det->>'name') <> ALL(%s)"
            _hp_args = [_hidden_by_process]

        cur = conn.cursor()
        # v4.0.107 — filter out synthetic pipeline entries whose class name
        # starts with an underscore. `_color` (LAB colour analysis output),
        # `_lab`, and future `_*` markers are added by post-processing;
        # they are NOT real defect detections. Prior to this filter they
        # counted as real detections in `by_class`, showed up in the
        # operator's top-defects list, and — with confidence 1.0 — could
        # dominate the confidence-sum-based impact calculation for any
        # class whose configured severity was zero. Same pattern the
        # detection_charts and detection_stats endpoints use (see line
        # ~2704 comment "4.0.29: skip synthetic").
        cur.execute(
            f"""
            SELECT (det->>'name') AS cls,
                   COUNT(*)       AS n,
                   SUM((det->>'confidence')::float) AS conf_sum
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s {ship_clause}
              AND (det->>'confidence') IS NOT NULL
              AND COALESCE(det->>'name', '') !~ '^_'
              {_hp_sql}
            GROUP BY (det->>'name')
            """,
            list(base_params) + _hp_args,
        )
        cls_rows = cur.fetchall()

        cur.execute(
            f"""
            SELECT MIN(encoder_value) AS enc_min,
                   MAX(encoder_value) AS enc_max,
                   COUNT(*) AS frames,
                   MIN(time)::timestamptz AS first_ts,
                   MAX(time)::timestamptz AS last_ts
            FROM inference_results
            WHERE time > NOW() - INTERVAL %s {ship_clause}
            """,
            base_params,
        )
        enc_min, enc_max, frame_count, first_ts, last_ts = cur.fetchone()
        cur.close()

        if not cls_rows and not frame_count:
            return empty

        impact_by_class = {}
        count_by_class = {}
        total_detections = 0
        for cls, n, conf_sum in cls_rows:
            cls_s = str(cls) if cls is not None else ""
            n_i = int(n or 0)
            conf_sum_f = float(conf_sum or 0.0)
            total_detections += n_i
            count_by_class[cls_s] = n_i
            sev = severities.get(cls_s, 0)
            if sev > 0:
                impact_by_class[cls_s] = round((sev / 100.0) * conf_sum_f, 3)

        # 3.25.8 — fold per-procedure ejection severity into the same impact ledger.
        # ejection_events has one row per fired (Store=ON) procedure; we count rows
        # per procedure_name within the window and add (severity/100 × count) to a
        # virtual "ejection:<name>" key so the existing impact_total / top_defects /
        # PDF report machinery picks them up alongside detection classes.
        ejection_impact = 0.0
        ejection_counts: dict = {}
        try:
            cur2 = conn.cursor()
            ej_ship_clause = "AND shipment = %s" if shipment else ""
            ej_params = [interval] + ([shipment] if shipment else [])
            cur2.execute(
                f"""
                SELECT procedure_name, COUNT(*) AS n
                FROM ejection_events
                WHERE time > NOW() - INTERVAL %s {ej_ship_clause}
                GROUP BY procedure_name
                """,
                ej_params,
            )
            for proc_name, n in cur2.fetchall():
                pname = str(proc_name) if proc_name is not None else ""
                n_i = int(n or 0)
                if not pname or n_i <= 0:
                    continue
                ejection_counts[pname] = n_i
                psev = proc_severities.get(pname, 0)
                if psev > 0:
                    imp = round((psev / 100.0) * n_i, 3)
                    ejection_impact += imp
                    # Surface in impact_by_class under a namespaced key so the existing
                    # top_defects renderer + PDF table treat it as just another row.
                    impact_by_class[f"⏏ {pname}"] = imp
                    count_by_class[f"⏏ {pname}"] = n_i
            cur2.close()
        except Exception as _ej_e:
            logger.warning(f"ejection-impact merge failed: {_ej_e}")

        impact_total = round(sum(impact_by_class.values()), 3)

        encoder_span = 0
        if enc_min is not None and enc_max is not None:
            encoder_span = int(max(0, enc_max - enc_min))

        if encoder_span > 0:
            denom = float(encoder_span)
            normalized_by = "encoder"
            impact_per_unit_label = f"/{encoder_unit}" if encoder_unit != "encoder_unit" else "/encoder_unit"
        elif frame_count and frame_count > 0:
            denom = float(frame_count)
            normalized_by = "frame"
            impact_per_unit_label = "/frame"
        else:
            denom = 1.0
            normalized_by = "none"
            impact_per_unit_label = "/(none)"

        impact_per_unit = impact_total / denom if denom > 0 else 0.0

        # 3.25.12 — score_scale_factor is the global "loudness knob" — multiplies
        # impact_per_unit before the 100×(1−x) formula. Default 1.0 makes a long
        # shipment with a handful of defects score ~99.99 (impact_per_unit on
        # encoder-normalised lines is typically ~1e-4). The operator calibrates
        # this once via POST /api/score/calibrate so the p50 of recent shipment
        # scores lands at a target (default 85). Without it, severity changes
        # can't move the score and the quality strip stays uniformly green.
        try:
            score_scale_factor = float(_svc.get("score_scale_factor") or 1.0)
        except (TypeError, ValueError):
            score_scale_factor = 1.0
        # 4.0.61 — dual scoring. The RELATIVE score uses the operator-calibrated
        # score_scale_factor and is what "matches your fleet's own baseline";
        # the ABSOLUTE score uses a fixed unity coefficient and shows the raw
        # severity×confidence impact without any calibration knob, so a site
        # that hasn't been calibrated (or slid out of calibration) still has
        # a defensible number to look at. UI renders BOTH side-by-side so the
        # operator never has to click "Calibrate" just to get a number.
        impact_per_unit_scaled = impact_per_unit * score_scale_factor
        score_relative = max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, impact_per_unit_scaled))))
        score_absolute = max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, impact_per_unit))))
        # Historical field `score` = the relative one, to preserve every
        # existing consumer (PDF report, quality strip, top-N leaderboard,
        # thresholds against release/re-inspect gates).
        score = score_relative

        # 4.0.103 — thresholds are LIVE from service_config so operator
        # changes take effect immediately.
        _q = _quality_thresholds()
        # 4.0.103 — PENDING verdict. If the sample is too thin to say
        # anything meaningful, refuse to slap a RELEASE/RE-INSPECT/HOLD
        # sticker on it. Reasons the operator asked for this: on khoy,
        # brand-new shipments were showing RE-INSPECT at score=60.3 with
        # `encoder_span=0` — undefined-behaviour scoring wearing a lab
        # coat. Now those come back as PENDING with a machine-readable
        # `pending_reason`, and the UI can show "gathering data…" instead.
        # We count detections (`total_detections`) rather than raw frames
        # so a shipment with 500 empty frames doesn't score PENDING away
        # from a real 100-defect verdict.
        _n_rows = int(total_detections or 0)
        _span = float(encoder_span or 0)
        if _n_rows < _q["min_rows"] or _span < _q["min_span"]:
            verdict = "PENDING"
            pending_reason = (
                f"insufficient sample: rows={_n_rows}<{_q['min_rows']}"
                if _n_rows < _q["min_rows"] else
                f"insufficient span: {_span:.0f}<{_q['min_span']:.0f}"
            )
        else:
            pending_reason = None
            if score >= _q["release"]:
                verdict = "RELEASE"
            elif score >= _q["reinspect"]:
                verdict = "RE-INSPECT"
            else:
                verdict = "HOLD"

        duration_sec = 0
        if first_ts and last_ts:
            try:
                duration_sec = max(0, int((last_ts - first_ts).total_seconds()))
            except Exception:
                duration_sec = 0
        throughput = (encoder_span / duration_sec) if (encoder_span and duration_sec) else 0.0

        top_defects = []
        for cls, imp in sorted(impact_by_class.items(), key=lambda kv: -kv[1])[:5]:
            n = count_by_class.get(cls, 0)
            # 3.25.8 — ejection rows are namespaced "⏏ <proc_name>"; look up severity
            # against the procedure map so the top-defect row shows the right value.
            if cls.startswith("⏏ "):
                sev = proc_severities.get(cls[2:], 0)
                kind = "ejection"
            else:
                sev = severities.get(cls, 0)
                kind = "detection"
            top_defects.append({
                "class":         cls,
                "impact":        round(imp, 2),
                "impact_per_unit": round(imp / denom, 4) if denom > 0 else 0.0,
                "count":         n,
                "count_per_unit": round(n / denom, 4) if denom > 0 else 0.0,
                "severity":      sev,
                "kind":          kind,
            })

        return {
            "shipment": shipment, "window": window,
            "score": round(score, 1),
            # 4.0.61 — dual-mode scoring exposed on the payload so the UI can
            # render both without a second endpoint. See the score calc above
            # for the definition of each; `score` stays as the relative one
            # for backwards compat with the PDF / strip / leaderboard readers.
            "score_absolute": round(score_absolute, 1),
            "score_relative": round(score_relative, 1),
            "score_scale_factor": round(score_scale_factor, 4),
            "verdict": verdict,
            # 4.0.103 — new field. Non-null ONLY when verdict=="PENDING". Gives
            # the UI a human-readable reason to show ("gathering data" etc.).
            "pending_reason": pending_reason,
            # 4.0.103 — thresholds now include min_rows + min_span so the
            # frontend can render "PENDING (78/100 rows)" style progress hints
            # instead of leaving the operator guessing what "PENDING" means.
            "thresholds": {
                "release":   _q["release"],
                "reinspect": _q["reinspect"],
                "min_rows":  _q["min_rows"],
                "min_span":  _q["min_span"],
            },
            "impact_total": impact_total,
            "impact_by_class": impact_by_class,
            "impact_per_unit": round(impact_per_unit, 4),
            "impact_per_unit_label": impact_per_unit_label,
            "total_detections": total_detections,
            "top_defects": top_defects,
            "encoder_min": int(enc_min) if enc_min is not None else None,
            "encoder_max": int(enc_max) if enc_max is not None else None,
            "encoder_span": encoder_span,
            "encoder_unit": encoder_unit,
            # v4.0.140 — new field is primary; legacy alias kept for one release
            # so any consumer (report generator, dashboards) that still reads
            # the old name doesn't regress.
            "encoder_units_per_unit":  encoder_units_per_unit or None,
            "encoder_units_per_meter": encoder_units_per_meter or None,
            "frame_count": int(frame_count or 0),
            "first_ts": first_ts.isoformat() if first_ts else None,
            "last_ts":  last_ts.isoformat()  if last_ts  else None,
            "duration_sec": duration_sec,
            "throughput": round(throughput, 3),
            "throughput_label": (f"{encoder_unit}/sec" if encoder_unit != "encoder_unit" else "encoder_units/sec"),
            "normalized_by": normalized_by,
            "persisted": True,
            # 3.25.8 — surface the ejection contribution separately for transparency.
            "ejection_impact": round(ejection_impact, 3),
            "ejection_counts": ejection_counts,
            # 3.25.12 — surface the calibration knob so the UI + /api/score/calibrate
            # can read/show the currently-applied loudness.
            "score_scale_factor": round(score_scale_factor, 4),
        }
    except Exception as e:
        logger.warning(f"_compute_quality_payload failed: {e}")
        return empty
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


# 4.0.103 — operator-facing threshold config. GET returns live values so
# the Process tab UI can populate its inputs; POST validates + persists to
# `service_config` in the data file. Reads are lock-free (dict copy) and
# writes go through save_service_config which is atomic.
@router.get("/api/quality/thresholds")
def get_quality_thresholds():
    """Return the live release/reinspect/min_rows/min_span thresholds.

    UI shows these on the Process tab so QC can tune "what counts as
    RELEASE" without editing YAML or restarting MVE. Values change take
    effect on the NEXT score computation (no restart).
    """
    return JSONResponse(content=_quality_thresholds())


@router.post("/api/quality/thresholds")
async def set_quality_thresholds(request: Request):
    """Persist new release/reinspect/min_rows/min_span values.

    Body: JSON with any subset of {release, reinspect, min_rows, min_span}.
    Missing keys keep their current value. Invalid values are rejected.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "invalid JSON body"}, status_code=400)

    try:
        from config import load_service_config as _lsc, save_service_config as _svc_save
    except Exception as e:
        return JSONResponse(content={"error": f"config module unavailable: {e}"}, status_code=500)

    cur = _lsc() or {}
    errors = []
    # Each knob has its own type + range. release MUST be > reinspect so
    # the verdict ladder still makes sense.
    def _mk_num(k, lo, hi, typ):
        if k not in body: return None
        try: v = typ(body[k])
        except (TypeError, ValueError):
            errors.append(f"{k}: expected number"); return None
        if v < lo or v > hi:
            errors.append(f"{k}: must be in [{lo}, {hi}]"); return None
        return v

    new_release   = _mk_num("release",   0, 100, float)
    new_reinspect = _mk_num("reinspect", 0, 100, float)
    new_min_rows  = _mk_num("min_rows",  0, 10_000_000, int)
    new_min_span  = _mk_num("min_span",  0, 1e12, float)

    # Ordering check: new (or existing) release must be > reinspect. This
    # matters even when only one knob was supplied — we compare against the
    # current value for the other one.
    _q = _quality_thresholds()
    eff_release   = new_release   if new_release   is not None else _q["release"]
    eff_reinspect = new_reinspect if new_reinspect is not None else _q["reinspect"]
    if eff_release <= eff_reinspect:
        errors.append(f"release ({eff_release}) must be greater than reinspect ({eff_reinspect})")

    if errors:
        return JSONResponse(content={"error": "validation failed", "details": errors}, status_code=400)

    if new_release   is not None: cur["quality_release_score"]     = new_release
    if new_reinspect is not None: cur["quality_reinspect_score"]   = new_reinspect
    if new_min_rows  is not None: cur["quality_min_rows"]           = new_min_rows
    if new_min_span  is not None: cur["quality_min_encoder_span"]   = new_min_span

    ok = _svc_save(cur)
    if not ok:
        return JSONResponse(content={"error": "failed to persist config"}, status_code=500)
    return JSONResponse(content={"ok": True, "thresholds": _quality_thresholds()})



# 3.25.12 — auto-calibrate the score_scale_factor so the p50 of recent shipment
# scores lands on a target (default 85). Without this calibration, scores are
# typically stuck near 100 because impact_per_unit on encoder-normalised lines
# is ~1e-4 — the formula 100*(1-impact_per_unit) bottoms out at 99.99. The
# operator runs this once per site; the resulting scale_factor stays in config.
@router.post("/api/score/calibrate")
def calibrate_score_scale(payload: dict = None):
    """Auto-tune `score_scale_factor` against recent shipment data.

    Body (all optional):
      {
        "target_p50": 85,          // desired median shipment score (0..100)
        "target_p5":  60,          // desired floor (worst 5% lands at or below)
        "window":     "7d",        // history window to sample
        "min_shipments": 5,        // bail if fewer than this many distinct shipments
        "apply":      false,       // if true, write to service_config and persist
      }

    Returns the recommended scale_factor + the projected p5 / p50 / p95
    of the score distribution after applying it. No DB / config mutation
    unless `apply=true`.
    """
    payload = payload or {}
    target_p50 = float(payload.get("target_p50") or 85)
    target_p5  = float(payload.get("target_p5")  or 60)
    window     = str(payload.get("window") or "7d")
    min_n      = int(payload.get("min_shipments") or 5)
    apply      = bool(payload.get("apply") or False)

    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days", "30d": "30 days"}
    interval = _windows.get(window, "7 days")

    try:
        from services.db import get_db_connection, release_db_connection
        from config import load_service_config as _lsc, save_service_config as _ssc
    except Exception as e:
        return JSONResponse(content={"error": f"db/config import failed: {e}"}, status_code=500)

    conn = get_db_connection()
    if conn is None:
        return JSONResponse(content={"error": "DB unreachable"}, status_code=503)

    try:
        # Distinct recent shipments (segment 2 of image_path, per 3.25.4 fix).
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT SPLIT_PART(image_path, '/', 2) AS ship
            FROM inference_results
            WHERE time > NOW() - INTERVAL %s
              AND image_path IS NOT NULL
              AND SPLIT_PART(image_path, '/', 2) NOT IN ('', 'no_shipment')
            """,
            [interval],
        )
        ships = [r[0] for r in cur.fetchall() if r[0]]
        cur.close()
    finally:
        try:
            release_db_connection(conn)
        except Exception:
            pass

    if len(ships) < min_n:
        return JSONResponse(content={
            "error": f"Need at least {min_n} shipments in {window}, found {len(ships)}.",
            "shipments_found": len(ships),
        }, status_code=400)

    # Compute UNSCALED impact_per_unit for each shipment.
    # 4.0.29 — FIX runaway-calibration bug: `_compute_quality_payload`
    # returns the RAW `impact_per_unit` at line 1213/1272 (`impact_total /
    # denom`, no scale applied). The previous logic here divided by
    # `current_scale_factor` thinking the payload was already scaled — that
    # comment was wrong, the payload was always raw. With a 16M scale stored
    # in config, every calibration click computed `ipu_raw = ipu / 16M` →
    # vanishingly small → recommended new scale near the 1e9 cap → next click
    # divided by the new larger scale → bigger still → pinned at the cap.
    # No division needed; the payload IS the raw value.
    ipu_list = []
    for s in ships:
        try:
            pl = _compute_quality_payload(shipment=s, window=window)
            ipu_raw = float(pl.get("impact_per_unit") or 0.0)
            if ipu_raw > 0:
                ipu_list.append(ipu_raw)
        except Exception:
            continue

    if len(ipu_list) < min_n:
        # 4.0.61 — actionable calibration diagnostics. The old error was
        # "Recent shipments have zero impact — nothing to calibrate against"
        # which is symptom, not cause. Impact is
        # `sum((severity/100) × confidence)` over detections in classes with
        # severity>0. It goes to zero when EITHER (a) no class has severity
        # configured, OR (b) the detected classes DO have severities but those
        # specific defects didn't appear in the recent window. Tell the
        # operator which one they're in and what to click.
        try:
            svc_diag = _lsc() or {}
            audio_diag = svc_diag.get("audio_settings", {}) or {}
            classes_with_sev = sorted(
                cls for cls, meta in audio_diag.items()
                if isinstance(meta, dict) and int(meta.get("severity", 0) or 0) > 0
            )
            proc_diag = svc_diag.get("ejection_procedures", []) or []
            procs_with_sev = sorted(
                p.get("name", "") for p in proc_diag
                if isinstance(p, dict) and int(p.get("severity", 0) or 0) > 0 and p.get("name")
            )
        except Exception:
            classes_with_sev, procs_with_sev = [], []
        try:
            conn2 = get_db_connection()
            recent_classes = []
            if conn2 is not None:
                cur3 = conn2.cursor()
                cur3.execute(
                    """
                    SELECT DISTINCT elem->>'name' AS cls, COUNT(*) AS n
                    FROM inference_results, jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s
                      AND COALESCE(elem->>'name','') !~ '^_'
                    GROUP BY cls
                    ORDER BY n DESC
                    LIMIT 15
                    """,
                    [interval],
                )
                recent_classes = [{"class": r[0], "count": int(r[1] or 0)} for r in cur3.fetchall() if r[0]]
                cur3.close()
                release_db_connection(conn2)
        except Exception:
            recent_classes = []
        if not classes_with_sev and not procs_with_sev:
            headline = ("No class or ejection procedure has a severity > 0 configured — the "
                        "quality score has nothing to weight. Set severities in Process tab "
                        "→ per-class row → Severity, or in Ejection Procedures → Severity.")
        else:
            in_data = {rc["class"] for rc in recent_classes}
            covered = [c for c in classes_with_sev if c in in_data]
            missing = [c for c in classes_with_sev if c not in in_data]
            if covered:
                headline = (f"Severities are set on {len(classes_with_sev)} class(es) but only "
                            f"{len(covered)} appeared in {window}, and their impact totals "
                            f"were too small to calibrate against. Try a longer window "
                            f"(e.g. 30d), or raise severities on the classes actually being "
                            f"detected below.")
            else:
                headline = (f"Severities are set on {len(classes_with_sev)} class(es) "
                            f"({', '.join(classes_with_sev[:6])}{'…' if len(classes_with_sev) > 6 else ''}) "
                            f"but NONE of them were detected in {window}. Either extend the "
                            f"window, or set severity on the classes currently being detected "
                            f"(see 'top detected classes' below).")
        return JSONResponse(content={
            "error": headline,
            "diagnostic": {
                "window": window,
                "shipments_found": len(ships),
                "shipments_with_impact": len(ipu_list),
                "classes_with_severity": classes_with_sev,
                "procedures_with_severity": procs_with_sev,
                "top_detected_classes_in_window": recent_classes,
            },
        }, status_code=400)

    ipu_list.sort()
    def _pct(arr, p):
        if not arr: return 0.0
        k = max(0, min(len(arr)-1, int(round((len(arr)-1) * p))))
        return arr[k]
    p5_raw  = _pct(ipu_list, 0.05)
    p50_raw = _pct(ipu_list, 0.50)
    p95_raw = _pct(ipu_list, 0.95)

    # We want:
    #   target_p50 = score of the MEDIAN shipment  => scale uses p50_raw
    #   target_p5  = score of the WORST shipment   => scale uses p95_raw
    #     (5th-percentile-of-scores corresponds to 95th-percentile-of-impact
    #      because score is monotonically decreasing in impact)
    # Pick the LARGER of the two so we MEET both targets (the projection
    # function later uses _score(p95_raw) for projected_p5 — bug fixed
    # here so that mapping is consistent on both sides).
    # 4.0.29 — Was previously using p5_raw (lowest impact) on the second
    # candidate, which "constrained the BEST shipment to score 60" and
    # ALWAYS dominated max() because (0.4 / very-small) is huge. That made
    # every recommendation push median + best down toward 60 simultaneously
    # while saturating the worst at 0. Using p95_raw (highest impact) makes
    # the candidate represent its true semantic — "the worst-impact shipment
    # must score ≥ 60" — which is normally a SMALLER scale than the p50
    # candidate, so max() correctly hits target_p50=85.
    candidates = []
    if p50_raw > 0:
        candidates.append(max(0.0, (1.0 - target_p50 / 100.0) / p50_raw))
    if p95_raw > 0:
        candidates.append(max(0.0, (1.0 - target_p5  / 100.0) / p95_raw))
    if not candidates:
        return JSONResponse(content={
            "error": "All ipu values are 0 — can't solve for scale.",
        }, status_code=400)
    new_scale = max(candidates)
    # Sanity bounds: don't allow runaway factors or zero.
    new_scale = max(0.001, min(1e9, new_scale))

    # Project p5 / p50 / p95 of scores AFTER applying new_scale.
    def _score(ipu):
        return max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, ipu * new_scale))))
    projected_p5  = _score(p95_raw)   # worst-impact -> lowest score
    projected_p50 = _score(p50_raw)
    projected_p95 = _score(p5_raw)    # least-impact -> highest score

    result = {
        "shipments_sampled": len(ipu_list),
        "ipu_p5_raw":  round(p5_raw, 6),
        "ipu_p50_raw": round(p50_raw, 6),
        "ipu_p95_raw": round(p95_raw, 6),
        "current_scale_factor": None,
        "recommended_scale_factor": round(new_scale, 4),
        "projected_score_p5":  round(projected_p5, 1),
        "projected_score_p50": round(projected_p50, 1),
        "projected_score_p95": round(projected_p95, 1),
        "target_p50": target_p50,
        "target_p5":  target_p5,
        "applied": False,
    }

    try:
        svc = _lsc() or {}
        result["current_scale_factor"] = float(svc.get("score_scale_factor") or 1.0)
        if apply:
            svc["score_scale_factor"] = round(new_scale, 4)
            _ssc(svc)
            result["applied"] = True
    except Exception as e:
        logger.warning(f"calibrate_score_scale config write failed: {e}")
        result["apply_error"] = str(e)

    return JSONResponse(content=result)


@router.get("/api/shipment_quality_score")
def shipment_quality_score(request: Request, shipment: str = "", window: str = "24h"):
    """3.21.14 — Shipment-level Quality Score with encoder-span normalization.

    The score is now length-aware: `impact_per_unit = impact_total / encoder_span`,
    so a long shipment is not penalized vs. a short one (same defect quality →
    same score, regardless of duration / throughput).

    Falls back to `impact_total / frame_count` when encoder data is missing
    (handy for cameras-only deployments without a roll encoder).

    Encoder rollover / reset is handled by computing `MAX - MIN` which is
    monotonic-encoder-correct; non-monotonic encoders may need
    `sum(positive deltas)` later — flagged as a known limitation.

    Returns:
      - score (0–100; higher = better)
      - verdict ('RELEASE' / 'RE-INSPECT' / 'HOLD')
      - impact_total, impact_per_unit, impact_per_unit_label
      - encoder_min, encoder_max, encoder_span, encoder_unit (user-configured label)
      - first_ts, last_ts, duration_sec, throughput (units/sec)
      - total_detections, top_defects (each with impact-per-unit and count-per-unit)
    """
    return JSONResponse(content=_compute_quality_payload(shipment, window))


def _fmt_duration(sec: int) -> str:
    sec = int(sec or 0)
    if sec <= 0:
        return "0s"
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _verdict_color(verdict: str):
    """RGB tuple in 0..1 for the verdict pill background."""
    if verdict == "RELEASE":
        return (0.20, 0.65, 0.32)  # green
    if verdict == "RE-INSPECT":
        return (0.95, 0.62, 0.07)  # amber
    if verdict == "HOLD":
        return (0.86, 0.21, 0.27)  # red
    return (0.55, 0.55, 0.55)      # grey for NO_DATA


@router.get("/api/shipment_quality_score/report.pdf")
def shipment_quality_score_report(request: Request, shipment: str = "", window: str = "24h"):
    """Render the same quality-score payload as a printable PDF.

    Streams `application/pdf` straight from ReportLab — no temp files.
    The endpoint imports reportlab lazily so the rest of the app keeps
    working even if the dependency hasn't been installed yet (operators
    get a 503 with a clear install hint instead of a server crash).
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
        )
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"error": "reportlab not installed",
                     "hint": "docker exec monitait_vision_engine pip install reportlab"},
        )

    import io, datetime as _dt
    payload = _compute_quality_payload(shipment, window)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Shipment Quality Report",
    )
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=22, spaceAfter=4)
    H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
    Small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, textColor=colors.grey)
    Body = styles["BodyText"]
    # Score paragraph: leading == fontSize so the bbox is tight to the glyphs.
    # With leading > fontSize the VALIGN=MIDDLE in the score-row table centers
    # the bbox, not the visible glyphs, so the number sat ~8pt below the pill's
    # center. leading=42 with a 42pt font puts the text baseline at the bbox
    # midline so it lines up exactly with the verdict pill across from it.
    # alignment=1 (center) horizontally centers the score in its column.
    ScoreXL = ParagraphStyle("score_xl", parent=Body, fontSize=42, leading=42, alignment=1)
    VerdictPill = ParagraphStyle("verdict_pill", parent=Body, fontSize=14, leading=16, alignment=1)
    HeaderCell = ParagraphStyle(
        "header_cell", parent=Body, fontSize=9, leading=11, alignment=1,
        textColor=colors.white, fontName="Helvetica-Bold",
    )
    # Show the configured unit when set, otherwise just "unit" (was the literal
    # placeholder "encoder_unit" which read awkwardly in the report).
    raw_unit = payload.get("encoder_unit") or "encoder_unit"
    is_default_unit = (raw_unit == "encoder_unit")
    unit = "unit" if is_default_unit else raw_unit
    unit_suffix = f"/{unit}"
    story = []

    # --- Header ---
    story.append(Paragraph("Shipment Quality Report", H1))
    subtitle = []
    subtitle.append(f"Shipment: <b>{payload.get('shipment') or 'ALL SHIPMENTS'}</b>")
    subtitle.append(f"Window: last {payload.get('window', '24h')}")
    subtitle.append(f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    story.append(Paragraph(" &nbsp; · &nbsp; ".join(subtitle), Small))
    story.append(Spacer(1, 10))

    # --- Score block: big score + tight verdict pill on same row ---
    score = payload.get("score")
    verdict = payload.get("verdict", "NO_DATA")
    score_txt = f"{score:.1f}/100" if isinstance(score, (int, float)) else "—"
    vr, vg, vb = _verdict_color(verdict)
    # Two-column table: left = big score paragraph with leading=50 (no overflow);
    # right = a small nested table that holds the colored verdict pill so the
    # pill is auto-sized to its content instead of stretching to fill 80mm.
    # Pill is sized to its content (40mm wide), 7pt vertical padding above/below.
    # Total pill height ≈ 16 (leading) + 14 (padding) = 30pt.
    pill_inner = Table([[Paragraph(f'<font color="white"><b>{verdict}</b></font>', VerdictPill)]],
                       colWidths=[40 * mm])
    pill_inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(vr, vg, vb)),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
        ("LEFTPADDING",(0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("BOX",        (0, 0), (-1, -1), 0.4, colors.Color(vr * 0.7, vg * 0.7, vb * 0.7)),
    ]))
    # Score row: VALIGN=MIDDLE on the cell + leading=fontSize on the Paragraph
    # means the score baseline lands at the same vertical center as the pill.
    # Equal top/bottom padding (12pt) ensures the row height is symmetric.
    score_tbl = Table(
        [[Paragraph(f"<b>{score_txt}</b>", ScoreXL), pill_inner]],
        colWidths=[90 * mm, 50 * mm],
    )
    score_tbl.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (0, 0), (0, 0),   "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("TOPPADDING",  (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 12),
    ]))
    story.append(score_tbl)
    thr = payload.get("thresholds") or {}
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Thresholds: RELEASE ≥ {thr.get('release','?')} &nbsp;·&nbsp; "
        f"RE-INSPECT ≥ {thr.get('reinspect','?')} &nbsp;·&nbsp; HOLD &lt; {thr.get('reinspect','?')}", Small))
    story.append(Spacer(1, 10))

    # --- KPI grid ---
    story.append(Paragraph("Production KPIs", H2))
    span = payload.get("encoder_span") or 0
    # v4.0.140 — read new field first, legacy fallback so PDF report on
    # older sites still shows the physical-length hint.
    upu = payload.get("encoder_units_per_unit") or payload.get("encoder_units_per_meter")
    length_txt = f"{span:,} {unit}"
    if upu and span:
        length_txt += f"  (≈ {span / upu:,.1f} {unit})"
    throughput_label = payload.get("throughput_label", "units/sec")
    if is_default_unit:
        throughput_label = "units/sec"
    impact_per_unit_label = unit_suffix
    kpi_rows = [
        ["Production length", length_txt],
        ["Duration",          _fmt_duration(payload.get("duration_sec", 0))],
        ["Throughput",        f"{payload.get('throughput', 0):,.2f} {throughput_label}"],
        ["Total detections",  f"{payload.get('total_detections', 0):,}"],
        ["Impact (total)",    f"{payload.get('impact_total', 0):,.2f}"],
        ["Impact per unit",   f"{payload.get('impact_per_unit', 0):.4f} {impact_per_unit_label}"],
        ["Normalized by",     payload.get("normalized_by", "—")],
        ["Frame count",       f"{payload.get('frame_count', 0):,}"],
    ]
    kpi_tbl = Table(kpi_rows, colWidths=[55 * mm, 110 * mm])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_tbl)

    # --- Top defects ---
    story.append(Paragraph("Top Defects", H2))
    tops = payload.get("top_defects") or []
    if not tops:
        story.append(Paragraph("<i>No defects with severity &gt; 0 in this window.</i>", Body))
    else:
        # Headers wrapped in Paragraph so long labels ("Count /encoder_unit")
        # break onto two lines instead of overflowing into the next column.
        header = [
            Paragraph("Class", HeaderCell),
            Paragraph("Severity", HeaderCell),
            Paragraph("Count", HeaderCell),
            Paragraph(f"Count<br/>{unit_suffix}", HeaderCell),
            Paragraph("Impact", HeaderCell),
            Paragraph(f"Impact<br/>{unit_suffix}", HeaderCell),
        ]
        body = [header]
        for d in tops:
            body.append([
                str(d.get("class", "")),
                str(d.get("severity", 0)),
                f"{d.get('count', 0):,}",
                f"{d.get('count_per_unit', 0):.4f}",
                f"{d.get('impact', 0):,.2f}",
                f"{d.get('impact_per_unit', 0):.4f}",
            ])
        # Wider per-unit columns (28mm) so the two-line header has horizontal room
        def_tbl = Table(body, colWidths=[44 * mm, 20 * mm, 22 * mm, 28 * mm, 26 * mm, 28 * mm])
        def_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.13, 0.18, 0.27)),
            ("VALIGN",     (0, 0), (-1, 0), "MIDDLE"),
            ("VALIGN",     (0, 1), (-1, -1), "MIDDLE"),
            ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
            ("ALIGN",      (0, 0), (0, -1),  "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.96, 0.96, 0.98)]),
            ("BOX",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("INNERGRID",  (0, 0), (-1, -1), 0.2, colors.lightgrey),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",(0, 0), (-1, -1), 5),
            ("RIGHTPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(def_tbl)

    # --- Footer ---
    story.append(Spacer(1, 18))
    try:
        ver = (pathlib.Path("/code/VERSION").read_text().strip()
               if pathlib.Path("/code/VERSION").exists()
               else pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("VERSION").read_text().strip())
    except Exception:
        ver = "?"
    first = payload.get("first_ts") or "—"
    last  = payload.get("last_ts")  or "—"
    story.append(Paragraph(
        f"Window: <b>{first}</b> &nbsp;→&nbsp; <b>{last}</b><br/>"
        f"Generated by Monitait Vision Engine v{ver}", Small))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()

    safe_ship = (payload.get("shipment") or "all").replace("/", "_").replace(" ", "_")[:48]
    fname = f"quality_{safe_ship}_{payload.get('window','24h')}_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/api/shipment_quality_score/trend")
def shipment_quality_score_trend(
    request: Request, shipment: str = "", window: str = "24h", buckets: int = 12,
):
    """3.21.16 — Time-bucketed impact-per-unit timeline + drift indicator.

    Splits the window into ~`buckets` equal time-buckets and computes
    `impact_per_unit` for each bucket (using the same severity map + encoder
    normalization as `/api/shipment_quality_score`). Adds a simple linear-fit
    slope so the operator sees *whether quality is drifting* even when the
    overall score still passes.

    Returns: { buckets:[{t0,t1,score,impact,impact_per_unit,encoder_span,frames,detections}],
               slope, slope_pct, slope_label, normalized_by, encoder_unit, window_first_ts, window_last_ts }

    Slope label: "improving" / "stable" / "degrading" based on
    `slope_pct = (last_third_avg - first_third_avg) / max(first_third_avg, eps)`.
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
    interval = _windows.get(window, "24 hours")
    # Bucket width: aim for `buckets` slots across the window.
    _bucket_for = {
        "1h":  "5 minutes",
        "6h":  "30 minutes",
        "24h": "2 hours",
        "7d":  "14 hours",
    }
    bucket = _bucket_for.get(window, "2 hours")
    # 4.0.196 — same override as the other shipment-aware endpoints. A
    # picked shipment means "show this shipment's trend regardless of the
    # window preset." Widen to 90 days and let the buckets span the
    # shipment's actual duration (bucket width also widens to keep the
    # bucket count in the same ballpark for an older long shipment).
    if shipment:
        interval = "90 days"
        # A 90-day trend split into 12 buckets = 7.5 days/bucket — too
        # coarse. Recompute bucket size dynamically based on the picked
        # shipment's actual first_ts → last_ts span at query time. Below
        # the CTE runs, we let Postgres decide via time_bucket(). For now
        # a reasonable default that scales with buckets param:
        bucket = "1 day"

    ship_clause = "AND shipment = %s" if shipment else ""
    base_params = [interval] + ([shipment] if shipment else [])

    empty = {
        "shipment": shipment, "window": window, "buckets": [],
        "slope": 0.0, "slope_pct": 0.0, "slope_label": "no data",
        "normalized_by": "none", "encoder_unit": "encoder_unit",
        "window_first_ts": None, "window_last_ts": None,
        "persisted": False,
    }

    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        from config import load_service_config as _lsc
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)

        _svc = _lsc() or {}
        _audio = _svc.get("audio_settings", {}) or {}
        severities = {k: (int(v.get("severity", 0) or 0)) / 100.0
                      for k, v in _audio.items() if isinstance(v, dict)}
        encoder_unit = str(_svc.get("encoder_unit") or "encoder_unit")

        cur = conn.cursor()

        # Per-bucket per-class detection sums (impact = severity * sum(conf))
        cur.execute(
            f"""
            SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt,
                   (det->>'name') AS cls,
                   COUNT(*) AS n,
                   SUM((det->>'confidence')::float) AS conf_sum
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s {ship_clause}
              AND (det->>'confidence') IS NOT NULL
            GROUP BY bkt, (det->>'name')
            ORDER BY bkt
            """,
            base_params,
        )
        cls_rows = cur.fetchall()

        # Per-bucket frame count + encoder span
        cur.execute(
            f"""
            SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt,
                   COUNT(*) AS frames,
                   MIN(encoder_value) AS enc_min,
                   MAX(encoder_value) AS enc_max,
                   MIN(time) AS t0,
                   MAX(time) AS t1
            FROM inference_results
            WHERE time > NOW() - INTERVAL %s {ship_clause}
            GROUP BY bkt
            ORDER BY bkt
            """,
            base_params,
        )
        frame_rows = cur.fetchall()
        cur.close()

        if not frame_rows:
            return JSONResponse(content=empty)

        # Index per-bucket totals (impact + count) from the class rows
        per_bucket_impact = {}
        per_bucket_count = {}
        for bkt, cls, n, conf_sum in cls_rows:
            cls_s = str(cls) if cls is not None else ""
            sev = severities.get(cls_s, 0.0)
            n_i = int(n or 0)
            cs = float(conf_sum or 0.0)
            per_bucket_impact[bkt] = per_bucket_impact.get(bkt, 0.0) + sev * cs
            per_bucket_count[bkt] = per_bucket_count.get(bkt, 0) + n_i

        # 3.25.12 — same calibration knob as the main shipment score (see
        # _compute_quality_payload). Without this the trend buckets are also
        # near-100 for any real factory throughput.
        try:
            SCALE = float(_svc.get("score_scale_factor") or 1.0)
        except (TypeError, ValueError):
            SCALE = 1.0
        result = []
        for bkt, frames, enc_min, enc_max, t0, t1 in frame_rows:
            impact = per_bucket_impact.get(bkt, 0.0)
            count = per_bucket_count.get(bkt, 0)
            enc_span = 0
            if enc_min is not None and enc_max is not None:
                enc_span = int(max(0, enc_max - enc_min))
            if enc_span > 0:
                denom = float(enc_span)
                normalized_by = "encoder"
            elif frames and frames > 0:
                denom = float(frames)
                normalized_by = "frame"
            else:
                denom = 1.0
                normalized_by = "none"
            ipu = impact / denom if denom > 0 else 0.0
            score = max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, ipu * SCALE))))
            result.append({
                "t0": t0.isoformat() if t0 else None,
                "t1": t1.isoformat() if t1 else None,
                "bucket": bkt.isoformat() if bkt else None,
                "frames": int(frames or 0),
                "detections": count,
                "encoder_span": enc_span,
                "impact": round(impact, 3),
                "impact_per_unit": round(ipu, 5),
                "score": round(score, 1),
                "normalized_by": normalized_by,
            })

        # Slope: compare last-third vs first-third of ipu values.
        # More robust than raw last-vs-first when noise is high.
        ipus = [r["impact_per_unit"] for r in result]
        slope_pct = 0.0
        slope_label = "stable"
        if len(ipus) >= 3:
            n = len(ipus)
            third = max(1, n // 3)
            first_avg = sum(ipus[:third]) / third
            last_avg  = sum(ipus[-third:]) / third
            base = max(first_avg, 1e-9)
            slope_pct = round(100.0 * (last_avg - first_avg) / base, 1)
            if first_avg < 1e-6 and last_avg < 1e-6:
                slope_label = "stable"  # both flat at ~0
            elif slope_pct > 15:
                slope_label = "degrading"
            elif slope_pct < -15:
                slope_label = "improving"
            else:
                slope_label = "stable"

        # Pick a representative normalized_by (whichever applied to most buckets)
        normalizers = [r["normalized_by"] for r in result]
        normalized_by = max(set(normalizers), key=normalizers.count) if normalizers else "none"

        return JSONResponse(content={
            "shipment": shipment, "window": window,
            "buckets": result,
            "bucket_size": bucket,
            "slope": round(slope_pct / 100.0, 4),
            "slope_pct": slope_pct,
            "slope_label": slope_label,
            "normalized_by": normalized_by,
            "encoder_unit": encoder_unit,
            "window_first_ts": result[0]["t0"] if result else None,
            "window_last_ts":  result[-1]["t1"] if result else None,
            "thresholds": {"release": QUALITY_RELEASE_SCORE, "reinspect": QUALITY_REINSPECT_SCORE},
            "persisted": True,
        })
    except Exception as e:
        logger.warning(f"shipment_quality_score_trend failed: {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


from datetime import timedelta as _td   # 3.25.4 — used by /api/quality/heatmap

@router.get("/api/quality/shipments")
def quality_shipments(request: Request, n: int = 30, window: str = "24h"):
    """3.25.4 — recent shipments + their quality scores for a per-shipment bar chart.

    Returns up to `n` shipments started within `window`, each with its
    quality score + verdict + detection count. Order is most-recent-first.
    Chart shows one bar per shipment: bar height = score, color = verdict.

    4.0.103 — `_windows` now includes 1h and 6h (previously they silently
    fell through to the 30d default because the map only had 24h/7d/30d/90d,
    which meant "operator clicked 1h, saw 30 days of data, thought it was
    1 h").
    4.0.103 — `n<=0` now means "no cap" (customer request: 7d/30d views must
    show EVERY shipment in the window, not just the first 30). We still cap
    the effective LIMIT at 5000 as a safety valve — anyone asking for more
    than 5000 shipments is either testing or writing an integration and
    should page.
    """
    _windows = {
        "1h":   "1 hour",
        "6h":   "6 hours",
        "24h":  "24 hours",
        "7d":   "7 days",
        "30d":  "30 days",
        "90d":  "90 days",
        # v4.0.131 — long-window support so operators can look at quarterly
        # / yearly score trends. Only /api/quality/shipments is bounded
        # enough to serve these (30 rows max via `n` cap). The scatter +
        # detection_stats + detection_charts endpoints would time out; the
        # Charts tab UI enters "long window" mode on the frontend and only
        # renders Score-per-shipment for these values.
        "120d": "120 days",
        "1y":   "365 days",
        # 4.0.202 — "all": the Score-per-shipment card is window-INDEPENDENT
        # (operator: "show the last N shipments regardless of the window").
        # ORDER BY ... DESC LIMIT n already returns the most recent N, so a
        # generous 90-day lookback covers the last N for any active line while
        # keeping the GROUP BY bounded (statement_timeout guards the rest).
        "all":  "90 days",
    }
    # v4.0.103 — clamp n: <=0 → uncapped (5000 max), else clamp to [1, 5000]
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 30
    effective_n = 5000 if n <= 0 else max(1, min(n, 5000))
    # v4.0.238 — the last N shipments, TIME-WINDOW-FREE (operator: "just a SQL query,
    # DESC LIMIT N, avoid anything time-related"). An unbounded GROUP BY over the whole
    # table would itself be slow, so bound the scan by ROWS not time: pull the most
    # recent ~2M rows via the time index (fast — the planner uses the time index for
    # `ORDER BY time DESC LIMIT`), then GROUP those into shipments. 2M rows always holds
    # far more than N shipments, so this returns the true last N however far back they
    # are, and never times out. `window` is ignored for the list.
    out: list = []
    rows: list = []
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content={"shipments": []})
        try:
            cur = conn.cursor()
            try:
                # 4.0.320 SCORE-LANE-ENUM — read the last N from the tiny shipment_summary cache
                # (a finished shipment was scored ONCE at finalize) instead of scanning ~2M
                # inference_results rows. Their score/verdict/length come straight from the cache
                # in the loop below (no per-shipment recompute). Falls back to the row-scan ONLY
                # when the cache is empty (a fresh deploy before the backfill has run).
                cur.execute("SET LOCAL statement_timeout = 8000")
                cur.execute(
                    """
                    SELECT shipment AS ship, t_start AS first_t, t_stop AS last_t, n_total AS n_rows
                    FROM shipment_summary
                    ORDER BY t_stop DESC NULLS LAST
                    LIMIT %s
                    """,
                    (effective_n,),
                )
                rows = cur.fetchall()
                if not rows:
                    cur.execute("SET LOCAL statement_timeout = 15000")
                    cur.execute(
                        """
                        SELECT ship, MIN(time) AS first_t, MAX(time) AS last_t, COUNT(*) AS n_rows
                        FROM (
                            SELECT SPLIT_PART(image_path, '/', 2) AS ship, time
                            FROM inference_results
                            WHERE image_path IS NOT NULL
                            ORDER BY time DESC
                            LIMIT 2000000
                        ) sub
                        WHERE ship NOT IN ('', 'no_shipment')
                        GROUP BY ship
                        ORDER BY last_t DESC
                        LIMIT %s
                        """,
                        (effective_n,),
                    )
                    rows = cur.fetchall()
            finally:
                try: cur.close()
                except Exception: pass
        finally:
            try: release_db_connection(conn)
            except Exception: pass

        # 4.0.320 — the in-progress shipment isn't finalized, so it's NOT in shipment_summary.
        # Prepend it (from Redis) so the lane still shows the active roll; the loop below live-
        # computes just that one shipment (everything else is a cache read).
        try:
            import os as _os_sl
            import redis as _rd_sl
            _cur_ship_sl = (_rd_sl.Redis(host=_os_sl.environ.get("REDIS_HOST", "redis"),
                                         port=6379, decode_responses=True).get("shipment") or "")
            if (_cur_ship_sl and _cur_ship_sl not in ("", "no_shipment")
                    and _cur_ship_sl not in [r[0] for r in rows]):
                rows = [(_cur_ship_sl, None, None, 0)] + list(rows)
        except Exception:
            pass

        # 4.0.29d — pass through the chart's window (default 30d) so the score
        # is computed against the SAME range that picked the shipment list.
        # Previously this was hard-coded to "24h", which silently zeroed scores
        # for any shipment whose last activity was older than 24h, leaving the
        # bar chart with invisible bars and X-axis labels for shipments that
        # never seemed to "load".
        # 4.0.63 — expose Absolute + Relative + raw impact_per_unit per
        # shipment so the frontend can render TWO bars side-by-side:
        # (1) absolute (context-free) and (2) window-relative (rescaled so
        # the currently visible shipments show as much visual difference as
        # possible). Absolute is stable, Relative highlights small deltas
        # between shipments in the selected window without needing the
        # operator to hit "Calibrate".
        # 4.0.319 — prefer the MATERIALIZED score/verdict/length from shipment_summary. A finished
        # shipment was scored ONCE at finalize, so reading it here skips the per-shipment
        # _compute_quality_payload recompute that made this lane slow to open. Only shipments NOT
        # in the cache (the in-progress one, or not-yet-backfilled) fall through to a live compute,
        # so the lane self-heals exactly as before while getting fast for everything already cached.
        _cached_scores = {}
        try:
            from services.shipment_stats import get_latest_shipment_scores
            for _cs in get_latest_shipment_scores(5000):
                if _cs.get("shipment"):
                    _cached_scores[_cs["shipment"]] = _cs
        except Exception:
            _cached_scores = {}
        for ship, first_t, last_t, n_rows in rows:
            _cs = _cached_scores.get(ship)
            if _cs is not None and _cs.get("score") is not None:
                out.append({
                    "shipment": ship,
                    "score":            _cs.get("score"),
                    "score_absolute":   _cs.get("score_absolute"),
                    "score_relative":   _cs.get("score"),
                    "impact_per_unit":  None,
                    "impact_total":     None,
                    "verdict":          _cs.get("verdict"),
                    "rows":             int(n_rows or 0),
                    "first_t": first_t.isoformat() if first_t else None,
                    "last_t":  last_t.isoformat()  if last_t  else None,
                    "top_defects": [],
                    "encoder_span":            _cs.get("enc_span"),
                    "encoder_min":             _cs.get("enc_min"),
                    "encoder_max":             _cs.get("enc_max"),
                    "encoder_unit":            _cs.get("length_unit") or "unit",
                    "encoder_units_per_unit":  None,
                    "encoder_units_per_meter": None,
                    "cached": True,
                })
                continue
            try:
                qp = _compute_quality_payload(shipment=ship, window="90d")
                out.append({
                    "shipment": ship,
                    "score":            qp.get("score") if qp else None,
                    "score_absolute":   qp.get("score_absolute") if qp else None,
                    "score_relative":   qp.get("score_relative") if qp else None,
                    "impact_per_unit":  qp.get("impact_per_unit") if qp else None,
                    "impact_total":     qp.get("impact_total") if qp else None,
                    "verdict":          qp.get("verdict") if qp else None,
                    "rows":             int(n_rows or 0),
                    "first_t": first_t.isoformat() if first_t else None,
                    "last_t":  last_t.isoformat()  if last_t  else None,
                    "top_defects": [d.get("class") for d in (qp.get("top_defects") or [])][:3] if qp else [],
                    # 4.0.98 — pass length + unit context through so the chart can
                    # show length labels above bars + compute normalized "impact
                    # per 100 units" (works for any product: fabric metres, glass
                    # sheets, wire kilometres, whatever your encoder measures).
                    "encoder_span":            qp.get("encoder_span") if qp else 0,
                    # 4.0.263 — real encoder start/end so the shipment tooltip can show the
                    # encoder range like it shows the time range (operator request).
                    "encoder_min":             qp.get("encoder_min") if qp else None,
                    "encoder_max":             qp.get("encoder_max") if qp else None,
                    "encoder_unit":            qp.get("encoder_unit", "unit") if qp else "unit",
                    # v4.0.140 — new field primary; legacy alias kept one release.
                    "encoder_units_per_unit":  (qp.get("encoder_units_per_unit") if qp else None),
                    "encoder_units_per_meter": (qp.get("encoder_units_per_meter") if qp else None),
                })
            except Exception:
                out.append({
                    "shipment": ship, "score": None, "score_absolute": None,
                    "score_relative": None, "impact_per_unit": None,
                    "verdict": None, "rows": int(n_rows or 0), "top_defects": [],
                })
    except Exception as e:
        logger.warning(f"quality/shipments failed: {e}")
    # v4.0.103 — return metadata so the frontend can differentiate
    # "list is capped" from "here's everything in the window".
    return JSONResponse(content={
        "shipments": out,
        "window": "recent",
        "requested_n": n,
        "cap_applied": (len(out) == effective_n and n != 0),
        "count": len(out),
    })


# 3.25.12 — ejection axis: where ejections occurred, colored by procedure name.
# Renders as a thin strip below the quality-by-time / quality-by-encoder strips.
# The Charts tab calls this with axis=time or axis=encoder. Each bucket gets:
#   - the dominant procedure name (most-fired in that bucket) → strip cell color
#   - a tooltip with the per-procedure breakdown
# 3.26.0 — single-frame detection lookup, used by the LSF annotation modal so it
# can pre-fill EVERY box on that frame (not just the dot the operator clicked).
@router.get("/api/frame_detections")
def frame_detections(image_path: str = "", request: Request = None):
    """Return all stored detections for one frame.

    Output:
      { image_path, image_url, px_per_mm, detections: [
          {name, confidence, xmin, ymin, xmax, ymax, ...}
      ] }
    Image dimensions are deliberately omitted — the browser already loads the
    image into an <img> for LSF, so it uses naturalWidth / naturalHeight for
    pixel→percent conversion. Saves one disk read per modal-open.

    4.0.15 — `px_per_mm` reflects the per-camera calibration on the camera
    whose stem matches `..._<cam_id>.jpg`. None if not calibrated; consumers
    (LSF modal title, defect drawer) only show mm when present.
    """
    if not image_path or "/" not in image_path:
        return JSONResponse(content={"error": "image_path required"}, status_code=400)
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content={"error": "db unreachable"}, status_code=503)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT detections FROM inference_results WHERE image_path = %s ORDER BY time DESC LIMIT 1",
                (image_path,),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            try: release_db_connection(conn)
            except Exception: pass
        dets = row[0] if row and row[0] else []
        # `detections` is a jsonb array — psycopg2 returns it as a Python list of dicts.
        if not isinstance(dets, list):
            dets = []
        # 3.26.2 — build the URL through the existing /api/raw_image/<path> serve route
        # (NOT a non-existent /raw_images/ path). Also strip `_DETECTED.jpg` so the
        # editor opens the RAW frame — annotating over YOLO's drawn boxes is wrong.
        rel = image_path.split("raw_images/")[-1].lstrip("/") if "raw_images/" in image_path else image_path.lstrip("/")
        # _DETECTED.jpg → .jpg ; the raw file lives at the same path without the suffix.
        rel_raw = rel.replace("_DETECTED.jpg", ".jpg")

        # 4.0.15 — px/mm calibration lookup. Filename ends with `_<cam_id>.jpg`
        # (e.g. `2026-06-15-14-44-10-579709_2.jpg` → cam 2). When the cam has a
        # `px_per_mm` attribute set on the live watcher, include it so the LSF
        # modal can show bbox dimensions in mm. Silently None on any failure.
        px_per_mm = None
        try:
            import re as _re
            m = _re.search(r"_(\d+)(?:_DETECTED)?\.jpg$", image_path)
            if m and request is not None:
                cam_id = int(m.group(1))
                watcher_inst = getattr(request.app.state, "watcher_instance", None)
                if watcher_inst is not None:
                    cam = watcher_inst.cameras.get(cam_id)
                    if cam is not None:
                        v = getattr(cam, "px_per_mm", None)
                        if v not in (None, "") and float(v) > 0:
                            px_per_mm = float(v)
        except Exception:
            px_per_mm = None

        return JSONResponse(content={
            "image_path": image_path,
            "image_url": "/api/raw_image/" + rel_raw,
            "px_per_mm": px_per_mm,
            "detections": dets,
        })
    except Exception as e:
        logger.warning(f"frame_detections failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/api/quality/ejection_axis")
def quality_ejection_axis(
    request: Request,
    axis: str = "encoder",   # 4.0.280 — encoder is the product default axis (front + back)
    window: str = "24h",
    buckets: int = 48,
    shipment: str = "",
    lo: float = 0.0,
    hi: float = 0.0,
):
    """Return per-bucket ejection event counts grouped by procedure_name.

    Output: {
        "axis": "time"|"encoder",
        "buckets": [{
            "bucket": <int>,
            "label":  <str>,                # bucket axis label
            "n":      <int>,                # total events in bucket
            "top_procedure": <str|None>,    # most-fired proc name (drives color)
            "by_procedure": {<name>: <int>, ...}
        }, ...],
        # for time:
        "t_min": iso, "t_max": iso, "secs_per_bucket": int,
        # for encoder:
        "encoder_min": int, "encoder_max": int, "width_per_bucket": int,
    }
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
    interval = _windows.get(window, "24 hours")
    # 4.0.196 — specific shipment overrides window (see /api/detection_charts)
    if shipment:
        interval = "90 days"
    # 4.0.39 — raise cap from 120 to 192 so the strip can match the colour
    # heatmap's max (N_BINS=192 in detection_charts). At 120 the heatmap had
    # more cells than the quality / ejection strip below it and the columns
    # didn't line up at high bucket counts.
    buckets = max(8, min(192, int(buckets)))
    axis_l = (axis or "time").lower()
    # v4.0.209 — SINGLE SOURCE OF X-DOMAIN. When the caller passes an explicit
    # [lo, hi] range (the scatter's exact window._mveUnifiedX domain), bucket
    # over THAT range instead of computing our own MIN/MAX. Guarantees a
    # vertical guideline at position X hits the same bucket in the scatter,
    # colour heatmap, quality strip AND this ejection strip. For encoder, lo/hi
    # are encoder units; for time, lo/hi are epoch-milliseconds.
    _explicit = (hi > lo)

    cells = []
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content={"axis": axis_l, "buckets": [], "shipment": shipment})
        try:
            cur = conn.cursor()
            ship_clause = ""
            params: list = []
            if shipment:
                ship_clause = "AND shipment = %s "
                params.append(shipment)

            if axis_l == "encoder":
                # v4.0.109 — encoder range now comes from inference_results
                # (the SAME source that quality/heatmap and detection_charts
                # use), not from ejection_events. Previously ejection_events
                # returned a NARROWER encoder range (subset — only rows that
                # ever fired an ejector), so on the Charts page the ejection
                # strip cells were narrower and misaligned with the quality
                # strip and the color heatmap that sit next to it. With the
                # shared range, empty ejection buckets simply have no
                # ejection rows in them (renders as a transparent cell),
                # keeping column-wise alignment across all three widgets.
                # Fallback: if inference_results has no encoder rows in the
                # window, fall through to the ejection-events range so the
                # strip still shows SOMETHING rather than empty.
                if _explicit:
                    # Bucket over the scatter's exact encoder domain.
                    enc_min = int(lo); enc_max = int(hi)
                else:
                    cur.execute(
                        f"""
                        SELECT MIN(encoder_value), MAX(encoder_value)
                        FROM inference_results
                        WHERE time > NOW() - INTERVAL %s
                          AND encoder_value IS NOT NULL
                          {ship_clause}
                        """,
                        [interval, *params],
                    )
                    enc_min, enc_max = cur.fetchone() or (None, None)
                    if enc_min is None or enc_max is None or enc_max - enc_min <= 0:
                        # Fall back to ejection_events range (prior behaviour).
                        cur.execute(
                            f"""
                            SELECT MIN(encoder_value), MAX(encoder_value)
                            FROM ejection_events
                            WHERE time > NOW() - INTERVAL %s
                              AND encoder_value IS NOT NULL
                              {ship_clause}
                            """,
                            [interval, *params],
                        )
                        enc_min, enc_max = cur.fetchone() or (None, None)
                    if enc_min is None or enc_max is None or enc_max - enc_min <= 0:
                        cur.close()
                        note = "no ejection events in window" if enc_min is None else \
                               "encoder reports no pulses on ejection events"
                        return JSONResponse(content={
                            "axis": "encoder", "buckets": [],
                            "shipment": shipment, "note": note,
                        })
                    enc_min = int(enc_min); enc_max = int(enc_max)
                span = enc_max - enc_min
                width = max(1, span // buckets)
                # v4.0.209 — when an explicit domain is set, EXCLUDE rows outside
                # [enc_min, enc_max] so edge buckets aren't inflated by clamping.
                _erng_sql = "AND encoder_value BETWEEN %s AND %s " if _explicit else ""
                _erng_args = [enc_min, enc_max] if _explicit else []
                cur.execute(
                    f"""
                    SELECT bucket, procedure_name, COUNT(*) AS n
                    FROM (
                      SELECT
                        LEAST(%s - 1, GREATEST(0,
                            ((encoder_value - %s)::bigint / NULLIF(%s,0)::bigint))::int
                        ) AS bucket,
                        procedure_name
                      FROM ejection_events
                      WHERE time > NOW() - INTERVAL %s
                        AND encoder_value IS NOT NULL
                        {_erng_sql}
                        {ship_clause}
                    ) src
                    GROUP BY bucket, procedure_name
                    """,
                    [buckets, enc_min, width, interval, *_erng_args, *params],
                )
                bd_data: dict = {}
                for b, name, n in cur.fetchall():
                    try: b = int(b or 0)
                    except (TypeError, ValueError): continue
                    n_i = int(n or 0)
                    pname = str(name) if name is not None else "(unknown)"
                    bd = bd_data.setdefault(b, {"n": 0, "by_procedure": {}})
                    bd["n"] += n_i
                    bd["by_procedure"][pname] = bd["by_procedure"].get(pname, 0) + n_i
                cur.close()
                for b in range(buckets):
                    bd = bd_data.get(b, {"n": 0, "by_procedure": {}})
                    top = max(bd["by_procedure"].items(), key=lambda kv: kv[1])[0] if bd["by_procedure"] else None
                    cells.append({
                        "bucket": b,
                        "label": f"{enc_min + b * width:,} – {enc_min + (b+1) * width - 1:,}",
                        "n": bd["n"],
                        "top_procedure": top,
                        "by_procedure": bd["by_procedure"],
                    })
                return JSONResponse(content={
                    "axis": "encoder", "buckets": cells,
                    "encoder_min": enc_min, "encoder_max": enc_max,
                    "width_per_bucket": width, "shipment": shipment,
                })

            # ----- TIME axis -----
            # 3.25.13 — bucket the FULL window (NOW() - interval → NOW()) instead
            # of MIN/MAX of ejection_events, so the strip's X-axis aligns with the
            # scatter (which spans the full window). Empty buckets at the edges
            # just render transparent.
            if _explicit:
                # lo/hi are epoch-milliseconds — bucket over the scatter's exact
                # time domain so the ejection strip lines up 1:1 with the scatter.
                cur.execute("SELECT to_timestamp(%s), to_timestamp(%s)", [lo / 1000.0, hi / 1000.0])
            else:
                cur.execute("SELECT NOW() - INTERVAL %s, NOW()", [interval])
            t_min, t_max = cur.fetchone() or (None, None)
            if not t_min or not t_max:
                cur.close()
                return JSONResponse(content={
                    "axis": "time", "buckets": [],
                    "shipment": shipment, "note": "no time window",
                })

            # v4.0.209 — with an explicit domain, exclude rows outside [t_min,t_max]
            # so edge buckets aren't inflated by clamping.
            _trng_sql = "AND time BETWEEN %s AND %s " if _explicit else ""
            _trng_args = [t_min, t_max] if _explicit else []
            cur.execute(
                f"""
                SELECT bucket, procedure_name, COUNT(*) AS n
                FROM (
                  SELECT
                    LEAST(%s - 1, GREATEST(0,
                        EXTRACT(EPOCH FROM (time - %s))::bigint
                          / NULLIF(EXTRACT(EPOCH FROM (%s::timestamptz - %s))::bigint / %s, 0)
                    ))::int AS bucket,
                    procedure_name
                  FROM ejection_events
                  WHERE time > NOW() - INTERVAL %s
                    {_trng_sql}
                    {ship_clause}
                ) src
                GROUP BY bucket, procedure_name
                """,
                [buckets, t_min, t_max, t_min, buckets, interval, *_trng_args, *params],
            )
            bd_time: dict = {}
            for b, name, n in cur.fetchall():
                try: b = int(b or 0)
                except (TypeError, ValueError): continue
                n_i = int(n or 0)
                pname = str(name) if name is not None else "(unknown)"
                bd = bd_time.setdefault(b, {"n": 0, "by_procedure": {}})
                bd["n"] += n_i
                bd["by_procedure"][pname] = bd["by_procedure"].get(pname, 0) + n_i
            cur.close()

            total_secs = (t_max - t_min).total_seconds()
            secs_per_bucket = max(1.0, total_secs / buckets)
            for b in range(buckets):
                bd = bd_time.get(b, {"n": 0, "by_procedure": {}})
                ts_start = t_min + _td(seconds=int(b * secs_per_bucket))
                top = max(bd["by_procedure"].items(), key=lambda kv: kv[1])[0] if bd["by_procedure"] else None
                cells.append({
                    "bucket": b,
                    "label": ts_start.strftime("%H:%M") if total_secs <= 86400 else ts_start.strftime("%m-%d %H:%M"),
                    "ts":    ts_start.isoformat(),
                    "n":     bd["n"],
                    "top_procedure": top,
                    "by_procedure": bd["by_procedure"],
                })
            return JSONResponse(content={
                "axis": "time", "buckets": cells,
                "t_min": t_min.isoformat(), "t_max": t_max.isoformat(),
                "secs_per_bucket": int(secs_per_bucket), "shipment": shipment,
            })
        finally:
            try:
                release_db_connection(conn)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"quality_ejection_axis failed: {e}")
        return JSONResponse(content={"axis": axis_l, "buckets": [], "error": str(e)})


@router.get("/api/quality/heatmap")
def quality_heatmap(
    request: Request,
    axis: str = "encoder",   # 4.0.280 — encoder is the product default axis (front + back)
    window: str = "24h",
    buckets: int = 48,
    shipment: str = "",
    lo: float = 0.0,
    hi: float = 0.0,
):
    """3.25.4 — 1D quality heatmap. `axis`: "time" or "encoder".

    Divides the window into N buckets and computes a quality score per
    bucket. Operator sees a strip like:
      00:00 ─🟢🟢🟢🟢🟡🔴🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢─ 24:00
    Each cell carries: {bucket, label, n_detections, score, top_class}.

    Score within a bucket = clip(100 - 100 × normalized_impact, 0, 100)
    where impact = Σ (severity × confidence) and the normalization
    target is "1 detection of severity 50 at confidence 0.5 per bucket
    counts as ~10 points off". Tunable later.
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
    interval = _windows.get(window, "24 hours")
    # 4.0.196 — specific shipment overrides window (see /api/detection_charts)
    if shipment:
        interval = "90 days"
    # 4.0.39 — raise cap from 120 to 192 so the strip can match the colour
    # heatmap's max (N_BINS=192 in detection_charts). At 120 the heatmap had
    # more cells than the quality / ejection strip below it and the columns
    # didn't line up at high bucket counts.
    buckets = max(8, min(192, int(buckets)))
    axis_l = (axis or "time").lower()
    # v4.0.209 — SINGLE SOURCE OF X-DOMAIN (same as quality_ejection_axis). An
    # explicit [lo, hi] range (the scatter's window._mveUnifiedX domain) makes
    # this strip bucket over the SAME range as the scatter/colour heatmap, so a
    # vertical guideline hits the same data in every layer. encoder → encoder
    # units; time → epoch-milliseconds.
    _explicit = (hi > lo)

    # pull per-class severity from audio_settings
    try:
        from config import load_service_config as _load_svc
        svc = _load_svc() or {}
        sev_map = {
            k: int(v.get("severity") or 0)
            for k, v in (svc.get("audio_settings") or {}).items()
            if isinstance(v, dict)
        }
        # 3.25.12 — same global loudness knob as the shipment quality score so
        # the heatmap strips reflect the same calibration. The legacy 0.0001
        # constant stays as the base scale; score_scale_factor multiplies it.
        try:
            _heatmap_scale = float(svc.get("score_scale_factor") or 1.0)
        except (TypeError, ValueError):
            _heatmap_scale = 1.0
        # v4.0.111 — classes where the operator un-checked "Show" in the
        # Process tab must NOT contribute to the quality strip either.
        # Otherwise a hidden class (e.g. `jean` — dominant frame-label, not
        # a defect) shows up as top_class in every bucket AND drives its
        # score. Same source (audio_settings[cls].show === False) that
        # detection_charts uses for `_hidden_by_process`.
        _hidden_by_process = sorted([
            str(cls) for cls, cfg in (svc.get("audio_settings") or {}).items()
            if isinstance(cfg, dict) and cfg.get("show") is False
        ])
    except Exception:
        sev_map = {}
        _heatmap_scale = 1.0
        _hidden_by_process = []

    cells = []
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content={"axis": axis_l, "buckets": [], "shipment": shipment})
        try:
            cur = conn.cursor()
            ship_clause = ""
            params: list = []
            if shipment:
                ship_clause = "AND image_path LIKE %s "
                params.append(f"{shipment}/%")

            if axis_l == "encoder":
                if _explicit:
                    enc_min = int(lo); enc_max = int(hi)
                else:
                    cur.execute(
                        f"""
                        SELECT MIN(encoder_value), MAX(encoder_value)
                        FROM inference_results
                        WHERE time > NOW() - INTERVAL %s
                          AND encoder_value IS NOT NULL
                          {ship_clause}
                        """,
                        [interval, *params],
                    )
                    enc_min, enc_max = cur.fetchone() or (None, None)
                    if enc_min is None or enc_max is None or enc_max - enc_min <= 0:
                        cur.close()
                        # Differentiate "no encoder at all" from "encoder reports all zeros"
                        # — the second one means the encoder is wired but not pulsing
                        # (line stopped, hardware unplugged, calibration missing).
                        note = "encoder reports no pulses (line not moving or encoder unwired)" \
                               if (enc_min == 0 and enc_max == 0) else "no encoder data in window"
                        return JSONResponse(content={
                            "axis": "encoder", "buckets": [],
                            "shipment": shipment, "note": note,
                        })
                    enc_min = int(enc_min); enc_max = int(enc_max)
                span = enc_max - enc_min
                width = max(1, span // buckets)
                # v4.0.209 — exclude rows outside the explicit domain so edge
                # buckets aren't inflated by LEAST/GREATEST clamping.
                _erng_sql = "AND encoder_value BETWEEN %s AND %s " if _explicit else ""
                _erng_args = [enc_min, enc_max] if _explicit else []
                # v4.0.111 — extend WHERE with the "Show" filter so classes the
                # operator hid in Process tab don't reach top_class or score.
                _hp_sql = ""
                _hp_args = []
                if _hidden_by_process:
                    _hp_sql = " AND (det->>'name') <> ALL(%s)"
                    _hp_args = [_hidden_by_process]
                cur.execute(
                    f"""
                    SELECT bucket, name, COUNT(*) AS n,
                           AVG((det->>'confidence')::float) AS avg_conf
                    FROM (
                      SELECT
                        LEAST(%s - 1, GREATEST(0,
                            ((encoder_value - %s)::bigint / NULLIF(%s,0)::bigint))::int
                        ) AS bucket,
                        (det->>'name') AS name,
                        det
                      FROM inference_results, LATERAL jsonb_array_elements(detections) det
                      WHERE time > NOW() - INTERVAL %s
                        AND encoder_value IS NOT NULL
                        {_erng_sql}
                        AND (det->>'name') IS NOT NULL
                        -- v4.0.110 — skip synthetic entries (`_color`, `_lab`, etc.).
                        -- Without this the quality-strip's top_class showed
                        -- `_color` as the "top defect" in every bucket (its
                        -- confidence is always 1.0 so it dominated the count
                        -- comparison at line 2785). Same filter that
                        -- detection_charts / detection_stats / _compute_quality_payload
                        -- already apply. Also removes _color from the score
                        -- calculation so buckets don't misleadingly turn red
                        -- when the actual defect density is fine.
                        AND COALESCE(det->>'name', '') !~ '^_'
                        {_hp_sql}
                        {ship_clause}
                    ) src
                    GROUP BY bucket, name
                    """,
                    [buckets, enc_min, width, interval, *_erng_args, *_hp_args, *params],
                )
                bucket_data: dict = {}
                for b, name, n, ac in cur.fetchall():
                    b = int(b or 0)
                    n = int(n or 0)
                    ac = float(ac or 0)
                    sev = sev_map.get(name, 0)
                    impact = sev * ac * n
                    bd = bucket_data.setdefault(b, {"n": 0, "impact": 0.0, "top_class": None, "top_count": 0})
                    bd["n"] += n
                    bd["impact"] += impact
                    if n > bd["top_count"]:
                        bd["top_count"] = n
                        bd["top_class"] = name
                cur.close()

                for b in range(buckets):
                    bd = bucket_data.get(b, {"n": 0, "impact": 0.0, "top_class": None})
                    score = max(0.0, 100.0 - bd["impact"] * 0.0001 * _heatmap_scale)
                    cells.append({
                        "bucket": b,
                        "label": f"{enc_min + b * width:,} – {enc_min + (b+1) * width - 1:,}",
                        "n": bd["n"],
                        "score": round(score, 1),
                        "top_class": bd["top_class"],
                    })
                return JSONResponse(content={
                    "axis": "encoder", "buckets": cells,
                    "encoder_min": enc_min, "encoder_max": enc_max,
                    "width_per_bucket": width, "shipment": shipment,
                })

            # ----- TIME axis -----
            # 3.25.13 — same alignment fix as quality_ejection_axis: bucket the
            # full window so the strip matches the scatter X-axis 1:1.
            if _explicit:
                # lo/hi are epoch-milliseconds — bucket over the scatter's exact
                # time domain so the quality strip lines up 1:1 with the scatter.
                cur.execute("SELECT to_timestamp(%s), to_timestamp(%s)", [lo / 1000.0, hi / 1000.0])
            else:
                cur.execute("SELECT NOW() - INTERVAL %s, NOW()", [interval])
            t_min, t_max = cur.fetchone() or (None, None)
            if not t_min or not t_max:
                cur.close()
                return JSONResponse(content={"axis": "time", "buckets": [], "shipment": shipment})

            # v4.0.209 — exclude rows outside the explicit domain (no edge clamp).
            _trng_sql = "AND time BETWEEN %s AND %s " if _explicit else ""
            _trng_args = [t_min, t_max] if _explicit else []
            # v4.0.111 — same Show-filter wiring as the encoder path.
            _hp_sql_t = ""
            _hp_args_t = []
            if _hidden_by_process:
                _hp_sql_t = " AND (det->>'name') <> ALL(%s)"
                _hp_args_t = [_hidden_by_process]
            cur.execute(
                f"""
                SELECT bucket, name, COUNT(*) AS n,
                       AVG((det->>'confidence')::float) AS avg_conf
                FROM (
                  SELECT
                    LEAST(%s - 1, GREATEST(0,
                        EXTRACT(EPOCH FROM (time - %s))::bigint
                          / NULLIF(EXTRACT(EPOCH FROM (%s::timestamptz - %s))::bigint / %s, 0)
                    ))::int AS bucket,
                    (det->>'name') AS name,
                    det
                  FROM inference_results, LATERAL jsonb_array_elements(detections) det
                  WHERE time > NOW() - INTERVAL %s
                    {_trng_sql}
                    AND (det->>'name') IS NOT NULL
                    -- v4.0.110 — same synthetic filter as the encoder path.
                    AND COALESCE(det->>'name', '') !~ '^_'
                    {_hp_sql_t}
                    {ship_clause}
                ) src
                GROUP BY bucket, name
                """,
                [buckets, t_min, t_max, t_min, buckets, interval, *_trng_args, *_hp_args_t, *params],
            )
            bucket_data2: dict = {}
            for b, name, n, ac in cur.fetchall():
                try:
                    b = int(b or 0)
                except (TypeError, ValueError):
                    continue
                n = int(n or 0)
                ac = float(ac or 0)
                sev = sev_map.get(name, 0)
                impact = sev * ac * n
                bd = bucket_data2.setdefault(b, {"n": 0, "impact": 0.0, "top_class": None, "top_count": 0})
                bd["n"] += n
                bd["impact"] += impact
                if n > bd["top_count"]:
                    bd["top_count"] = n
                    bd["top_class"] = name
            cur.close()

            total_secs = (t_max - t_min).total_seconds()
            secs_per_bucket = max(1.0, total_secs / buckets)
            for b in range(buckets):
                bd = bucket_data2.get(b, {"n": 0, "impact": 0.0, "top_class": None})
                ts_start = t_min + _td(seconds=int(b * secs_per_bucket))
                score = max(0.0, 100.0 - bd["impact"] * 0.0001 * _heatmap_scale)
                cells.append({
                    "bucket": b,
                    "label": ts_start.strftime("%H:%M") if total_secs <= 86400 else ts_start.strftime("%m-%d %H:%M"),
                    "ts":    ts_start.isoformat(),
                    "n":     bd["n"],
                    "score": round(score, 1),
                    "top_class": bd["top_class"],
                })
            return JSONResponse(content={
                "axis": "time", "buckets": cells,
                "t_min": t_min.isoformat(), "t_max": t_max.isoformat(),
                "secs_per_bucket": int(secs_per_bucket), "shipment": shipment,
            })
        finally:
            try: release_db_connection(conn)
            except Exception: pass
    except Exception as e:
        logger.warning(f"quality/heatmap failed: {e}")
    return JSONResponse(content={"axis": axis_l, "buckets": cells})


@router.get("/api/detection_stats")
def detection_stats(request: Request, window: str = "1h", min_conf: float = 0.0, shipment: str = ""):
    """Detection-quality insight for the Charts tab embedded panel.

    Aggregates the `inference_results` hypertable into:
      - by_class: { class_name: count }  over the window (most-frequent defects)
      - timeline: [ {t: "HH:MM", count: N}, ... ]  detections per time-bucket
      - total, window, persisted (whether any rows exist)

    Reads from TimescaleDB, so it only reflects classes with Store=ON (3.14.0
    per-class DB opt-in). If the DB is unreachable or empty, returns a well-formed
    empty payload so the frontend can render an informative "no stored detections
    yet" state instead of erroring.
    """
    # window → (interval SQL, bucket SQL, label fmt)
    # v4.0.126 — added 30d + 90d so operators can inspect long-window
    # detection distributions from the Charts tab without hitting the
    # "no stored detections" empty state on any window > 7d.
    _windows = {
        "1h":  ("1 hour",   "1 minute"),
        "6h":  ("6 hours",  "10 minutes"),
        "24h": ("24 hours", "1 hour"),
        "7d":  ("7 days",   "6 hours"),
        "30d": ("30 days",  "1 day"),
        "90d": ("90 days",  "1 day"),
    }
    interval, bucket = _windows.get(window, _windows["1h"])
    # 4.0.308 — a picked shipment is ENTIRELY independent of the timeframe (operator: "just WHERE
    # shipment = id, no time window"). So for a picked shipment the row filter is the shipment_id
    # ALONE — no time predicate at all — and only the no-shipment view uses the "last N" interval.
    # This fixes an old shipment coming back empty (and hiding the whole Detection Insights panel)
    # just because the previously-selected window had nothing in it.
    _ship = (shipment or "").strip()
    if _ship:
        _time_where = "shipment = %s"          # the shipment IS the filter — no time bound
        _base_args = [_ship]
    else:
        _time_where = "time > NOW() - INTERVAL %s"
        _base_args = [interval]
    empty = {"by_class": {}, "timeline": [], "total": 0, "window": window, "persisted": False}

    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        # v4.0.128 — autocommit mode so each query is its own tiny
        # transaction. When a query hits statement_timeout Postgres cancels
        # ONLY that query — the next cursor.execute() starts clean.
        # Savepoints on a shared cursor were fragile (a cancelled statement
        # left the cursor in a state where even ROLLBACK TO SAVEPOINT
        # failed) so the outer try/except kept swallowing the exception
        # AND every subsequent per-query catch. Autocommit isolates them.
        # Pool default statement_timeout (9 s) governs each; that's enough
        # for the sub-second by_class + impact queries and forces the slow
        # timeline query to fail fast without eating the outer budget.
        prev_autocommit = conn.autocommit
        conn.autocommit = True

        by_class = {}
        timeline = []
        try:
            cur = conn.cursor()
            # v4.0.129 — explicit per-statement timeout: 6 s for by_class
            # (has to be enough for LIMIT 100k + LATERAL unnest on khoy's
            # 8 M-row hypertable, empirically ~3-4 s at peak; anything
            # over 6 s means the DB is thrashing and we should just fail
            # fast so the panel populates from the other two queries).
            cur.execute("SET statement_timeout = 6000")
            cur.execute(
                f"""
                SELECT COALESCE(elem->>'name', elem->>'class') AS cls, COUNT(*) AS n
                FROM (
                    SELECT detections FROM inference_results
                    WHERE {_time_where}
                    ORDER BY time DESC
                    LIMIT 100000
                ) recent,
                LATERAL jsonb_array_elements(recent.detections) AS elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
                  -- 4.0.38: skip synthetic entries (e.g. `_color`) so they don't
                  -- show up as a real class in the Detections-by-class bar / pie.
                  AND COALESCE(elem->>'name', '') !~ '^_'
                GROUP BY cls
                ORDER BY n DESC
                LIMIT 30
                """,
                tuple(_base_args + [float(min_conf or 0.0)]),
            )
            for cls, n in cur.fetchall():
                if cls is not None:
                    by_class[str(cls)] = int(n)
            cur.close()
        except Exception as _e1:
            logger.warning(f"detection_stats by_class failed: {type(_e1).__name__}: {_e1}")

        # Detections-per-bucket timeline. Give it 4 s — a SUM aggregate
        # over the whole window either fits in that or misses the panel
        # (by_class is what really matters here).
        try:
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 4000")
            cur.execute(
                f"""
                SELECT time_bucket(INTERVAL %s, time) AS bkt,
                       COALESCE(SUM(detection_count), 0) AS n
                FROM inference_results
                WHERE {_time_where}
                GROUP BY bkt
                ORDER BY bkt
                """,
                tuple([bucket] + _base_args),
            )
            timeline = [{"t": b.strftime("%m-%d %H:%M"), "count": int(n)} for b, n in cur.fetchall()]
            cur.close()
        except Exception as _e2:
            logger.warning(f"detection_stats timeline failed: {type(_e2).__name__}: {_e2}")

        # 3.21.12 — impact score per class (severity × confidence, summed per class).
        # area_factor is currently 1 (no normalisation yet); will be added when we
        # have a reliable per-class typical-area baseline.
        from config import load_service_config as _lsc
        _svc = _lsc() or {}
        _audio = _svc.get("audio_settings", {}) or {}
        _severities = {k: (v.get("severity", 0) or 0) / 100.0 for k, v in _audio.items() if isinstance(v, dict)}
        impact_by_class = {}
        if any(s > 0 for s in _severities.values()):
            try:
                cur2 = conn.cursor()
                cur2.execute("SET statement_timeout = 6000")
                cur2.execute(
                    f"""
                    SELECT (det->>'name') AS cls,
                           SUM((det->>'confidence')::float) AS conf_sum
                    FROM (
                        SELECT detections FROM inference_results
                        WHERE {_time_where}
                        ORDER BY time DESC
                        LIMIT 100000
                    ) recent,
                    LATERAL jsonb_array_elements(recent.detections) det
                    WHERE COALESCE((det->>'confidence')::float, 0) >= %s
                      -- 4.0.38: skip synthetic (`_color` etc.) from impact-by-class too.
                      AND COALESCE(det->>'name', '') !~ '^_'
                    GROUP BY (det->>'name')
                    """,
                    tuple(_base_args + [float(min_conf or 0.0)]),
                )
                for cls, conf_sum in cur2.fetchall():
                    if cls and _severities.get(cls, 0) > 0:
                        impact_by_class[str(cls)] = round(_severities[cls] * float(conf_sum or 0.0), 2)
                cur2.close()
            except Exception as _e3:
                logger.warning(f"detection_stats impact_by_class failed: {type(_e3).__name__}: {_e3}")

        # Restore autocommit to the pool default so the connection is
        # clean when handed back.
        try: conn.autocommit = prev_autocommit
        except Exception: pass
        impact_total = round(sum(impact_by_class.values()), 2)

        total = sum(by_class.values())
        return JSONResponse(content={
            "by_class": by_class,
            "timeline": timeline,
            "total": total,
            "impact_by_class": impact_by_class,  # 3.21.12 — defect impact score per class
            "impact_total": impact_total,        # 3.21.12 — total impact across window
            "window": window,
            "persisted": total > 0 or len(timeline) > 0,
        })
    except Exception as e:
        logger.warning(f"detection_stats query failed (returning empty): {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            # 4.0.321 TIMEOUT-LEAK — this endpoint runs non-LOCAL `SET statement_timeout=4000/6000`
            # in autocommit mode, so the shortened cap PERSISTED on the connection after putconn and
            # the NEXT borrower inherited it (→ far more likely to silently return an empty panel at
            # 4-6s). Restore the pool default (and pool-default autocommit=False) before handing the
            # connection back, so the leak can't cross requests. Runs on success AND on exception.
            try:
                conn.autocommit = True  # so the reset SET commits immediately
                from services.db import POSTGRES_STATEMENT_TIMEOUT_MS as _pool_to
                _rc = conn.cursor(); _rc.execute("SET statement_timeout = %s" % int(_pool_to)); _rc.close()
            except Exception:
                pass
            try:
                conn.autocommit = False  # restore the pool-default transaction mode
            except Exception:
                pass
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# v4.0.210 — SINGLE SOURCE OF TRUTH for the Charts-tab strips. The quality and
# ejection strips used to come from their OWN endpoints (/api/quality/heatmap,
# /api/quality/ejection_axis) which each computed their OWN MIN/MAX range → they
# could never be *guaranteed* aligned with the scatter. Operator (emphatically):
# "get the quality/color/ejection data from the EXACT SAME api that you get the
# scatter data, so you don't misalign." These two helpers compute the quality
# score and ejection counts per bin using the IDENTICAL bounds + bin formula the
# colour heatmap uses inside /api/detection_charts (FLOOR((val-lo)*N/(hi-lo))),
# so quality cell K, ejection cell K, colour cell K and scatter bin K ALL cover
# the exact same encoder/time range. One query, one binning, zero drift.
def _dc_quality_cells(cur, axis, lo, hi, n_bins, time_sql, time_args,
                      ship_sql, ship_args, sev_map, hidden, scale, frame_cap_sql=""):
    """Severity-weighted quality score per bin over [lo, hi] with n_bins.

    Returns a DENSE list [{bucket, score, top_class, n}] for bins 0..n_bins-1
    (bin 0 = oldest/leftmost). Mirrors /api/quality/heatmap's math but bins over
    the caller's exact [lo, hi] (the colour heatmap's payload bounds). axis is
    'encoder' (lo/hi in encoder units) or 'time' (lo/hi in epoch-milliseconds).
    """
    cells = []
    try:
        lo = float(lo); hi = float(hi)
    except (TypeError, ValueError):
        return cells
    if not (hi > lo):
        return cells
    val = "encoder_value" if axis == "encoder" else "(EXTRACT(EPOCH FROM time) * 1000.0)"
    notnull = "AND encoder_value IS NOT NULL " if axis == "encoder" else ""
    hp_sql = " AND (det->>'name') <> ALL(%s)" if hidden else ""
    hp_args = [hidden] if hidden else []
    # 4.0.317 — cap SOURCE frames (capped CTE + LIMIT) BEFORE the jsonb expand so a per-bucket
    # quality query stays sub-second. Bin scores are unchanged by the cap (the impact/score is a
    # weighted average). No-op when frame_cap_sql is "" (full-window fetch).
    cur.execute(
        f"""
        WITH capped AS (
          SELECT encoder_value, time, detections
          FROM inference_results
          WHERE {time_sql}
            {notnull}
            {ship_sql}
          {frame_cap_sql}
        )
        SELECT bucket, name, COUNT(*) AS n, AVG((det->>'confidence')::float) AS avg_conf
        FROM (
          SELECT LEAST({n_bins} - 1, GREATEST(0,
                 FLOOR((({val} - %s)::numeric * {n_bins}) / (%s - %s))))::int AS bucket,
                 (det->>'name') AS name, det
          FROM capped, LATERAL jsonb_array_elements(detections) det
          WHERE (det->>'name') IS NOT NULL
            AND COALESCE(det->>'name', '') !~ '^_'
            {hp_sql}
        ) src
        GROUP BY bucket, name
        """,
        list(time_args) + list(ship_args) + [lo, hi, lo] + hp_args,
    )
    bd = {}
    for b, name, n, ac in cur.fetchall():
        try:
            b = int(b if b is not None else 0)
        except (TypeError, ValueError):
            continue
        n = int(n or 0); ac = float(ac or 0)
        sev = sev_map.get(name, 0)
        impact = sev * ac * n
        e = bd.setdefault(b, {"n": 0, "impact": 0.0, "top_class": None, "top_count": 0})
        e["n"] += n
        e["impact"] += impact
        if n > e["top_count"]:
            e["top_count"] = n
            e["top_class"] = name
    for b in range(n_bins):
        e = bd.get(b, {"n": 0, "impact": 0.0, "top_class": None})
        score = max(0.0, 100.0 - e["impact"] * 0.0001 * scale)
        cells.append({"bucket": b, "n": e["n"],
                      "score": round(score, 1), "top_class": e["top_class"]})
    return cells


def _dc_ejection_cells(cur, axis, lo, hi, n_bins, time_sql, time_args, ship_sql, ship_args):
    """Ejection-event counts per bin over [lo, hi] with n_bins. DENSE list
    [{bucket, n, top_procedure, by_procedure}] for bins 0..n_bins-1. Same bounds
    + bin formula as _dc_quality_cells and the colour heatmap → guaranteed
    aligned. Sources ejection_events (shipment column, encoder_value / time)."""
    cells = []
    try:
        lo = float(lo); hi = float(hi)
    except (TypeError, ValueError):
        return cells
    if not (hi > lo):
        return cells
    val = "encoder_value" if axis == "encoder" else "(EXTRACT(EPOCH FROM time) * 1000.0)"
    notnull = "AND encoder_value IS NOT NULL " if axis == "encoder" else ""
    cur.execute(
        f"""
        SELECT bucket, procedure_name, COUNT(*) AS n
        FROM (
          SELECT LEAST({n_bins} - 1, GREATEST(0,
                 FLOOR((({val} - %s)::numeric * {n_bins}) / (%s - %s))))::int AS bucket,
                 procedure_name
          FROM ejection_events
          WHERE {time_sql}
            {notnull}
            {ship_sql}
        ) src
        GROUP BY bucket, procedure_name
        """,
        [lo, hi, lo] + list(time_args) + list(ship_args),
    )
    bd = {}
    for b, name, n in cur.fetchall():
        try:
            b = int(b if b is not None else 0)
        except (TypeError, ValueError):
            continue
        n = int(n or 0)
        pname = str(name) if name is not None else "(unknown)"
        e = bd.setdefault(b, {"n": 0, "by_procedure": {}})
        e["n"] += n
        e["by_procedure"][pname] = e["by_procedure"].get(pname, 0) + n
    for b in range(n_bins):
        e = bd.get(b, {"n": 0, "by_procedure": {}})
        top = max(e["by_procedure"].items(), key=lambda kv: kv[1])[0] if e["by_procedure"] else None
        cells.append({"bucket": b, "n": e["n"],
                      "top_procedure": top, "by_procedure": e["by_procedure"]})
    return cells


def _dc_shipment_cells(cur, axis, lo, hi, n_bins, time_sql, time_args, ship_sql, ship_args, frame_cap_sql=""):
    """DOMINANT shipment per bin over [lo, hi] — same bounds + bin formula as the
    quality / ejection / colour cells, so the shipment lane bins line up 1:1 with
    the scatter + the other strips and stream the same way. Returns a DENSE list
    [{bucket, shipment, n}] (shipment=None, n=0 for empty bins). Operator: "do the
    shipment lane like the other strips with bins — each bin gets the biggest
    shipment in it; every bin of one shipment must share the same colour" (the
    frontend hues by shipment id, so a shipment that spans bins K..K+3 paints one
    solid bar)."""
    cells = []
    try:
        lo = float(lo); hi = float(hi)
    except (TypeError, ValueError):
        return cells
    if not (hi > lo):
        return cells
    val = "encoder_value" if axis == "encoder" else "(EXTRACT(EPOCH FROM time) * 1000.0)"
    notnull = "AND encoder_value IS NOT NULL " if axis == "encoder" else ""
    # v4.0.218 — DROP rows outside [lo,hi] instead of CLAMPING them into the edge
    # bins. The time WHERE-window (e.g. now-24h) is usually WIDER than the
    # scatter's dot-domain [lo,hi]; with a LEAST/GREATEST clamp every row older
    # than lo collapsed into bin 0 and every row newer than hi into bin N-1, so a
    # shipment painted a phantom block STUCK at the left edge (and another at the
    # right) even though no dot for it sits there. Operator: "I fucking hate your
    # idea of sticking every shipment to the left end — why does it have two, at
    # the beginning and the end?" Raw FLOOR + a bucket-range filter keeps only the
    # bins that actually fall on the scatter, so the lane fills right→left in
    # lock-step with the dots and never smears to an edge.
    # 4.0.317 — cap SOURCE rows per bucket (capped CTE + LIMIT). The dominant shipment in a bin
    # is unchanged by sampling — the biggest shipment stays biggest — so a dense bucket (khoy's
    # enc[0,3.7M] = ~865k rows → 1.5s) drops to sub-second. No-op when frame_cap_sql is "".
    cur.execute(
        f"""
        WITH capped AS (
          SELECT encoder_value, time, shipment
          FROM inference_results
          WHERE {time_sql}
            {notnull}
            AND shipment IS NOT NULL AND shipment <> ''
            {ship_sql}
          {frame_cap_sql}
        )
        SELECT bucket, shipment, COUNT(*) AS n
        FROM (
          SELECT FLOOR((({val} - %s)::numeric * {n_bins}) / (%s - %s))::int AS bucket,
                 shipment
          FROM capped
        ) src
        WHERE bucket >= 0 AND bucket < {n_bins}
        GROUP BY bucket, shipment
        """,
        list(time_args) + list(ship_args) + [lo, hi, lo],
    )
    bd = {}
    for b, sh, n in cur.fetchall():
        try:
            b = int(b if b is not None else 0)
        except (TypeError, ValueError):
            continue
        n = int(n or 0)
        e = bd.setdefault(b, {})
        e[sh] = e.get(sh, 0) + n
    for b in range(n_bins):
        counts = bd.get(b)
        if counts:
            top_sh, top_n = max(counts.items(), key=lambda kv: kv[1])
            cells.append({"bucket": b, "shipment": top_sh, "n": int(top_n)})
        else:
            cells.append({"bucket": b, "shipment": None, "n": 0})
    return cells


def _dc_load_severity():
    """Load {class: severity}, hidden-class list, and score scale once for the
    strip helpers. Mirrors /api/quality/heatmap's audio_settings read."""
    try:
        from config import load_service_config as _load_svc
        svc = _load_svc() or {}
        sev_map = {
            k: int(v.get("severity") or 0)
            for k, v in (svc.get("audio_settings") or {}).items()
            if isinstance(v, dict)
        }
        try:
            scale = float(svc.get("score_scale_factor") or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        hidden = sorted([
            str(cls) for cls, cfg in (svc.get("audio_settings") or {}).items()
            if isinstance(cfg, dict) and cfg.get("show") is False
        ])
        return sev_map, hidden, scale
    except Exception:
        return {}, [], 1.0


@router.get("/api/detection_charts")
def detection_charts(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0, baseline: str = "camera", phase: str = "", bins: int = 32, since_ms: int = 0, until_ms: int = 0, scatter_only: int = 0, dot_cap: int = 750, parent_class: str = "", include_color_slice: int = 0, color_axis: str = "encoder", bin_ref_lo: float = 0, bin_ref_hi: float = 0, include_strips: int = 0, range_only: int = 0, enc_lo: float = 0, enc_hi: float = 0, frame_cap: int = 0):
    """Rich detection analytics for the Charts tab (3.16.0).

    Returns, scoped to an optional shipment_id and a time window:
      - shipments:   distinct shipment ids in the window (for the filter dropdown)
      - size_over_time:   per time-bucket width/height percentiles (p10/p50/p90 +
                          min/max) → rendered as floating-range "candles"
      - confidence_over_time: per time-bucket confidence min/avg/max
      - confidence_by_class:  per time-bucket avg confidence split per class
                          ({buckets:[...], series:{cls:[...]}}) → one line per class
      - camera_scatter:   up to N points {x: epoch_ms, y: cam, r: conf, cls: name}
                          for the camera×time bubble chart (dot size = confidence)

    v4.0.74 — accepts optional `since_ms` + `until_ms` (millisecond unix
    timestamps) as an alternative to the `window` string. When both are
    provided the time filter becomes `time >= TO_TIMESTAMP(since_ms/1000)
    AND time < TO_TIMESTAMP(until_ms/1000)` — exact epoch-anchored slice.
    Used by the Charts tab's progressive bucket-by-bucket loader so a 24h
    view can render as 48 × 30-minute-bucket requests instead of one big
    query, letting the operator see dots/colours appear continuously from
    newest bucket to oldest instead of staring at "Loading charts…" for
    the full 4-8 s scan. Bucket resolution for the returned time_bucket()
    aggregation defaults to 1 minute in bucket-slice mode (the whole slice
    is ≤ 1 h anyway — no need for coarser buckets inside it).

    All aggregation runs over a capped recent slice (LIMIT 20000 rows) so a huge
    hypertable can't make this slow. Reads from inference_results, so it reflects
    only Store=ON classes. Returns a well-formed empty payload on any error.
    """
    # v4.0.80 — clamp dot_cap to [100, 5000]. UI options top out at 3000 but
    # the safe max is 5000; anything higher lets a huge scatter noticeably
    # slow Chart.js pan/zoom.
    dot_cap = max(10, min(5000, int(dot_cap or 750)))   # 4.0.203 — floor 100→10 (per-bucket)

    # v4.0.75 — TTL cache lookup. All params contribute to the key so
    # different dashboards with different filters get different cached
    # results but the same dashboard polling on a 5-s interval only hits
    # the DB once per 20-s window instead of every poll.
    # 4.0.317 — the key MUST include every param that changes the response. It was missing
    # enc_lo/enc_hi (added in 4.0.316) + parent_class + the colour/strip binning params, so all
    # of an encoder ladder's per-bucket calls (same window, DIFFERENT enc range) collided on one
    # key: the first bucket computed and the other 24 got its cached payload — every column
    # painted the first bucket's dots (they all clustered at one encoder position). This was the
    # real "only the newest bucket loads" bug, on top of the per-bucket slowness.
    _cache_key = _endpoint_cache_key(
        "dc", window, shipment, min_conf, baseline, phase, bins,
        since_ms, until_ms, scatter_only, dot_cap,
        enc_lo, enc_hi, parent_class, color_axis, bin_ref_lo, bin_ref_hi,
        include_strips, include_color_slice, range_only, frame_cap,
    )
    _cached = _endpoint_cache_get(_cache_key)
    if _cached is not None:
        return JSONResponse(content=_cached)

    _windows = {
        "1h":  ("1 hour",   "1 minute"),
        "6h":  ("6 hours",  "5 minutes"),
        "24h": ("24 hours", "30 minutes"),
        "7d":  ("7 days",   "6 hours"),
    }
    interval, bucket = _windows.get(window, _windows["24h"])

    # 4.0.200 — when a specific shipment is picked, drive the ENTIRE endpoint
    # through slice-mode over the shipment's ACTUAL [MIN(time), MAX(time)]
    # instead of a time window.
    #
    # Background: v4.0.196 overrode `interval = "90 days"` when a shipment was
    # picked, so the shipment's data would surface regardless of the operator's
    # window. That was wrong on two counts the operator caught:
    #   (a) it CUT shipments older than 90 days (they returned nothing), and
    #   (b) every time-bucketed panel (colour heatmap, quality/ejection strips)
    #       binned over the 90-day window, so a 30-minute shipment collapsed
    #       into ONE bin at the right edge ("only two bins have colour").
    # A specific shipment is already fully bounded by the (shipment, time)
    # index — there is nothing to "guard" with a time window; `shipment = X`
    # alone is a safe, bounded index lookup for a shipment of any age. So:
    # resolve the shipment's real span here and set since_ms/until_ms to it,
    # which flips `_use_slice` on below → every query (scatter, colour heatmap,
    # bins) runs over exactly the shipment's duration. Only do this when the
    # caller didn't already pass an explicit slice (progressive bucket loader).
    if shipment and not (bool(since_ms) and bool(until_ms)):
        try:
            from services.db import get_db_connection as _gdc_s, release_db_connection as _rdc_s
            _sconn = _gdc_s()
            if _sconn is not None:
                try:
                    _scur = _sconn.cursor()
                    _scur.execute(
                        "SELECT EXTRACT(EPOCH FROM MIN(time)) * 1000.0, "
                        "       EXTRACT(EPOCH FROM MAX(time)) * 1000.0 "
                        "FROM inference_results WHERE shipment = %s",
                        [shipment],
                    )
                    _sr = _scur.fetchone()
                    _scur.close()
                    if _sr and _sr[0] is not None and _sr[1] is not None and float(_sr[1]) > float(_sr[0]):
                        since_ms = int(float(_sr[0]))
                        until_ms = int(float(_sr[1])) + 1000   # +1s so the last row is included
                finally:
                    _rdc_s(_sconn)
        except Exception as _spe:
            logger.debug(f"shipment-span resolve failed: {_spe}")
        # Slice-aware queries (scatter, colour heatmap) now bin over the exact
        # span via since_ms/until_ms. Any query in this endpoint that is NOT
        # slice-aware still uses `interval`; widen it so it includes a shipment
        # of ANY age (the shipment filter + (shipment,time) index keep it a
        # bounded index scan — the wide interval only removes the age cut, it
        # doesn't cause a full-table scan).
        interval = "3650 days"

    # v4.0.74 — bucket-slice mode. When since_ms + until_ms are both provided
    # the request describes a specific epoch-anchored slice. Swap the time
    # predicate + parameter shape so the SAME downstream aggregation code
    # runs against `time >= to_timestamp(%s) AND time < to_timestamp(%s)`
    # instead of `time > NOW() - INTERVAL %s`. Keeps every field the client
    # already reads (camera_scatter, camera_scatter_encoder, camera_y_order,
    # color_baseline, phases_available, …) in place — only the WHERE clause
    # differs. `bucket` shrinks to 1 minute so the tiny slice still splits
    # into readable time_bucket rows if the client happens to use them.
    _use_slice = bool(since_ms) and bool(until_ms) and int(until_ms) > int(since_ms)
    # 4.0.317 — per-bucket FRAME CAP. A single bucket's colour/quality query jsonb-expands
    # every frame the bucket touches; on khoy's 9.5M-row 7d window that is ~2-7s per bucket
    # (× 25 buckets = minutes). But a bin's mean colour / quality score is identical whether
    # you expand 4k frames or 400k, so cap the SOURCE frames per bucket → each bucket drops to
    # ~0.1-0.8s and 25 buckets stream in ~30s. Cap only per-bucket (slice) queries; a full-
    # window fetch is never capped. Frontend may override; default kicks in on any slice.
    try:
        _frame_cap = int(frame_cap or 0)
    except Exception:
        _frame_cap = 0
    if _frame_cap <= 0 and _use_slice:
        _frame_cap = 8000
    _frame_cap_sql = (" LIMIT %d" % int(_frame_cap)) if _frame_cap > 0 else ""
    if _use_slice:
        bucket = "1 minute"
    empty = {"shipments": [], "size_over_time": [], "confidence_over_time": [],
             "confidence_by_class": {"buckets": [], "series": {}},
             "camera_scatter": [], "camera_scatter_encoder": [],
             "window": window, "shipment": shipment}

    # Optional shipment filter clause (parameterised)
    ship_clause = "AND shipment = %s" if shipment else ""

    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()

        # v4.0.101 — raise statement_timeout from the pool default (3 s, set in
        # services/db.py via `options='-c statement_timeout=3000'`) up to 15 s
        # for the duration of THIS transaction only. Reason: on high-volume
        # sites (khoy 2026-07-12: 1.5 M inference rows in 24 h), the CTE that
        # expands `jsonb_array_elements(detections)` and then ROW_NUMBER()s
        # over the exploded set legitimately takes 4–12 s, and the 3 s pool
        # ceiling was aborting the query before it returned — the browser
        # then rendered a blank scatter ("no dots"). 15 s is high enough to
        # accommodate the current row volume; if it ever creeps up we should
        # switch to a materialised view rather than raise this again.
        # `SET LOCAL` scopes the change to this transaction, so the connection
        # goes back to 3 s when we release it — write paths and cheap reads
        # keep their fail-fast behaviour.
        cur.execute("SET LOCAL statement_timeout = 15000")

        # 4.0.256 — cheap MIN/MAX range probe for the Charts loader. The frontend asks
        # for the ACTUAL data extent (time + encoder) in the window so it can bucket the
        # [min,max] of the SELECTED axis (operator: "check the min/max of time or encoder
        # based on the X-axis and bucket THAT"), instead of [now-window, now] which is
        # mostly empty when the line ran only recently. Pure indexed aggregate — no jsonb
        # expansion — so it returns in ms even on a huge hypertable.
        if int(range_only or 0):
            # 4.0.316 — a PICKED shipment's extent is TIME-INDEPENDENT (operator: "the shipment
            # selection should be independent entirely from the window timeframe"). Scope the
            # probe to shipment ONLY so its full encoder [min,max] is returned even when the
            # shipment predates the selected window. All-shipments keeps the window predicate.
            if shipment:
                _ro_where, _ro_args = "shipment = %s", [shipment]
            else:
                _ro_where, _ro_args = "time > NOW() - INTERVAL %s", [interval]
            cur.execute(
                f"SELECT EXTRACT(EPOCH FROM MIN(time)) * 1000.0, "
                f"       EXTRACT(EPOCH FROM MAX(time)) * 1000.0, "
                f"       MIN(encoder_value), MAX(encoder_value) "
                f"FROM inference_results "
                f"WHERE {_ro_where}",
                tuple(_ro_args),
            )
            _rr = cur.fetchone() or (None, None, None, None)
            # 4.0.313 — ROBUST encoder max. The raw MAX is dominated by rare runaway rolls (one
            # roll's encoder hit ~93M while typical rolls are ~4M), which pinned the axis to the
            # outlier and squished ALL real data into the left ~4% + starved the encoder buckets
            # (operator: "what's wrong when I select 7d?"). Encoder RESETS per roll, so the meaningful
            # extent is the TYPICAL roll length: MAX encoder PER shipment, then the 90th percentile
            # ACROSS shipments — one runaway roll can't set the axis. A single shipment (picked, or a
            # one-roll window) yields its own max, so nothing is clipped there.
            # 4.0.316 — use the TRUE encoder MAX. 4.0.314/315 "clipped" it thinking a 93M reading was
            # a runaway-roll outlier — but the data genuinely spans there: on khoy 50% of rows sit
            # above 25M encoder, so clipping dropped half the data. The encoder-range bucket queries
            # are ~1s each, so bucketing the full [min,max] is fine.
            _enc_max_robust = _rr[3]
            # 4.0.307 — general chart metadata over the FULL window, folded into this start-of-
            # load probe so the toolbar (parent-class dropdown, colour-check gate, unit label) no
            # longer depends on the 1h hydrate — which misses parents whose _color rows are older
            # than the last hour (operator: "TB shows late / the parent dropdown never appears").
            # This is the "separate general-data call" the operator asked for.
            # 4.0.311 — parent classes are a CONFIG property (audio_settings[cls].role == 'parent'),
            # NOT a last-N-hours thing. Discovering them from the window dropped parents whose _color
            # is older than the selected window — the operator saw the dropdown appear for a PICKED
            # shipment (spans all its data) but NOT in the general all-shipments view (TB's colour was
            # >24h old). Read parent-role classes straight from config so they ALWAYS show.
            _range_parents = []
            try:
                from config import load_service_config as _lsc_pr
                _as_pr = (_lsc_pr() or {}).get("audio_settings") or {}
                _range_parents = sorted([
                    str(_cls) for _cls, _cfg in _as_pr.items()
                    if isinstance(_cfg, dict) and str(_cfg.get("role") or "").strip().lower() == "parent"
                ])
            except Exception as _rpe:
                logger.debug(f"range_only parent discovery failed: {_rpe}")
            try:
                from config import load_service_config as _lsc_ro
                _ro_cfg = _lsc_ro() or {}
                _ro_color = bool(_ro_cfg.get("parent_color_check_enabled", False))
                _ro_unit = _ro_cfg.get("encoder_unit") or None
                _ro_upu = _ro_cfg.get("encoder_units_per_unit",
                                      _ro_cfg.get("encoder_units_per_meter"))
            except Exception:
                _ro_color, _ro_unit, _ro_upu = False, None, None
            try:
                release_db_connection(conn)
            except Exception:
                pass
            conn = None
            return JSONResponse(content={
                "data_t_min":   float(_rr[0]) if _rr[0] is not None else None,
                "data_t_max":   float(_rr[1]) if _rr[1] is not None else None,
                "data_enc_min": int(_rr[2]) if _rr[2] is not None else None,
                "data_enc_max": int(_enc_max_robust) if _enc_max_robust is not None else None,  # 4.0.313 — robust (per-shipment p90)
                "parent_classes_available": _range_parents,
                "parent_color_check_enabled": _ro_color,
                "encoder_unit": _ro_unit,
                "encoder_units_per_unit": _ro_upu,
                "window": window, "shipment": shipment,
            })

        # --- distinct shipments in the window (for the dropdown) ---
        cur.execute(
            f"SELECT DISTINCT shipment FROM inference_results "
            f"WHERE time > NOW() - INTERVAL %s AND shipment IS NOT NULL "
            f"ORDER BY shipment LIMIT 100",
            (interval,),
        )
        shipments = sorted(set([r[0] for r in cur.fetchall() if r[0]]) | set(_list_shipment_dirs()))

        # --- recent slice (capped) expanded to per-detection rows ---
        # Pull bucket, bbox dims, confidence, cam per detection.
        # min_conf filters at the JSONB-expand step so all downstream stats
        # (size percentiles, conf percentiles) respect the slider.
        base_params = [interval] + ([shipment] if shipment else []) + [float(min_conf or 0.0)]
        cur.execute(
            f"""
            WITH recent AS (
                SELECT time, shipment, detections
                FROM inference_results
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                ORDER BY time DESC
                LIMIT 50000  -- v4.0.78: bound the CTE so full-time-range scan can't blow the query
            ),
            expanded AS (
                SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt,
                       time,
                       (elem->>'_cam')        AS cam,
                       (elem->>'name')        AS cls,
                       (elem->>'confidence')::float AS conf,
                       ((elem->>'xmax')::float - (elem->>'xmin')::float) AS w,
                       ((elem->>'ymax')::float - (elem->>'ymin')::float) AS h
                FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
                  AND COALESCE(elem->>'name', '') !~ '^_'  -- 4.0.29: skip synthetic (_color full-frame bbox skews size percentiles)
            )
            SELECT
                bkt,
                percentile_cont(0.1) WITHIN GROUP (ORDER BY w) AS w_p10,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY w) AS w_p50,
                percentile_cont(0.9) WITHIN GROUP (ORDER BY w) AS w_p90,
                percentile_cont(0.1) WITHIN GROUP (ORDER BY h) AS h_p10,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY h) AS h_p50,
                percentile_cont(0.9) WITHIN GROUP (ORDER BY h) AS h_p90,
                MIN(conf) AS c_min, AVG(conf) AS c_avg, MAX(conf) AS c_max
            FROM expanded
            GROUP BY bkt ORDER BY bkt
            """,
            base_params,
        )
        size_over_time, confidence_over_time = [], []
        for row in cur.fetchall():
            bkt, wp10, wp50, wp90, hp10, hp50, hp90, cmin, cavg, cmax = row
            t = bkt.strftime("%m-%d %H:%M")
            size_over_time.append({
                "t": t,
                "w_lo": round(wp10 or 0, 1), "w_mid": round(wp50 or 0, 1), "w_hi": round(wp90 or 0, 1),
                "h_lo": round(hp10 or 0, 1), "h_mid": round(hp50 or 0, 1), "h_hi": round(hp90 or 0, 1),
            })
            confidence_over_time.append({
                "t": t,
                "c_min": round((cmin or 0) * 100, 1),
                "c_avg": round((cavg or 0) * 100, 1),
                "c_max": round((cmax or 0) * 100, 1),
            })

        # 4.0.37 — link Charts to Process tab "Show" toggle. Classes where
        # service_config.audio_settings[cls].show == False are excluded from
        # the scatter (they're already excluded from the detection annotator
        # via services/draw_filters.py, so this just brings the chart in line
        # with the operator's "this class shouldn't be visible anywhere"
        # decision). The per-chart legend click (sticky in localStorage) still
        # works on top of this as a transient override.
        _hidden_by_process = []
        try:
            from config import load_service_config as _lsc_ps
            _svc_ps = _lsc_ps() or {}
            _as = _svc_ps.get("audio_settings") or {}
            _hidden_by_process = sorted([
                str(cls) for cls, cfg in _as.items()
                if isinstance(cfg, dict) and cfg.get("show") is False
            ])
        except Exception as _pse:
            logger.debug(f"audio_settings show-flag lookup failed: {_pse}")
        _parent_filter_sql = ""
        _parent_filter_args = []
        if _hidden_by_process:
            _parent_filter_sql = " AND COALESCE(elem->>'name','') <> ALL(%s)"
            _parent_filter_args = [_hidden_by_process]
        # 4.0.305 — encoder-range filter for encoder-AXIS bucketing. The client divides the
        # window's [enc_min, enc_max] into N buckets and fetches each by its encoder start/stop
        # (not by time), so every bucket fills exactly ONE column of the encoder axis and the
        # span never re-derives from what a bucket happened to contain. Applied to the scatter
        # CTEs only (strips/colour are fetched once over the full span, binned by encoder).
        _enc_filter_sql = ""
        _enc_filter_args = []
        if float(enc_hi or 0) > float(enc_lo or 0):
            _enc_filter_sql = " AND encoder_value >= %s AND encoder_value < %s"
            _enc_filter_args = [float(enc_lo), float(enc_hi)]

        # --- camera scatter: stratified per-class so rare classes (spot/warp/stitch)
        # don't get swamped by dominant ones (weft_up). Up to 750 newest dots PER
        # CLASS, capped at 6000 total across classes.
        cur.execute(
            # v4.0.74 — slice-mode uses since_ms/until_ms for a narrow
            # epoch-anchored time predicate; the classic mode keeps the
            # `time > NOW() - INTERVAL %s` shape. Both branches feed the
            # SAME downstream aggregation so shape of returned rows is
            # identical. Base params re-ordered per branch to match.
            f"""
            WITH recent AS (
                SELECT time, shipment, detections, image_path, pipeline_id FROM inference_results
                WHERE { 'time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)' if _use_slice else 'time > NOW() - INTERVAL %s' } {ship_clause}{_enc_filter_sql}
                  AND %s   -- 4.0.317: FALSE for an ENCODER bucket (it reads camera_scatter_encoder, not
                           -- this time-x scatter) → CTE is empty and the whole query is trivial, so a
                           -- per-bucket encoder request no longer pays for BOTH scatter explodes.
                { '' if _use_slice else 'ORDER BY time DESC' }
                LIMIT { 20000 if _use_slice else 50000 }  -- v4.0.78: bound the CTE. 4.0.317: in slice
                             -- (per-bucket) mode DROP the ORDER BY time DESC — the (encoder_value,time)
                             -- index seeks the bucket's rows, but sorting ALL of a dense column by time
                             -- to pick the newest 50k was ~10s. Also a tighter 20k slice cap: a dense
                             -- column's cold random reads dominate, and 20k frames still yield plenty
                             -- of defects for the per-class dot_cap. A per-bucket column doesn't need "newest".
            ),
            exploded AS (
                SELECT EXTRACT(EPOCH FROM time) * 1000 AS x_ms,
                       time AS t,
                       COALESCE((elem->>'_cam')::int, 0) AS cam,
                       (elem->>'name') AS cls,
                       (elem->>'confidence')::float AS conf,
                       image_path AS img,
                       shipment AS ship,
                       pipeline_id AS pid
                FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
                  AND COALESCE(elem->>'name', '') !~ '^_'  -- 4.0.29: skip synthetic (_color etc.)
                  {_parent_filter_sql}
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY cls ORDER BY t DESC) AS rn
                FROM exploded
            )
            SELECT x_ms, cam, cls, conf, img, ship, pid FROM ranked
            WHERE rn <= %s
            ORDER BY t DESC
            LIMIT 6000
            """,
            (([int(since_ms), int(until_ms)] if _use_slice else [interval]) + ([shipment] if shipment else []) + _enc_filter_args + [not bool(_enc_filter_args)] + [float(min_conf or 0.0)] + _parent_filter_args + [int(dot_cap)]),
        )
        camera_scatter = [
            {"x": int(x), "y": cam, "cls": cls, "r": round((conf or 0), 3),
             "img": img, "ship": ship, "pid": pid}
            for x, cam, cls, conf, img, ship, pid in cur.fetchall()
        ]

        # --- camera × ENCODER scatter (3.21.0): defect map by roll position ---
        # Stratified per-class (same as camera-time scatter above) so rare
        # classes survive next to weft_up's flood.
        cur.execute(
            f"""
            WITH recent AS (
                SELECT time, shipment, detections, image_path, encoder_value, pipeline_id FROM inference_results
                WHERE { 'time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)' if _use_slice else 'time > NOW() - INTERVAL %s' } {ship_clause}
                  AND encoder_value IS NOT NULL{_enc_filter_sql}
                { '' if _use_slice else 'ORDER BY time DESC' }
                LIMIT { 20000 if _use_slice else 50000 }  -- v4.0.78: bound the CTE. 4.0.317: drop ORDER BY
                             -- in slice mode + tighter 20k slice cap so a dense per-bucket column
                             -- seeks+stops instead of sorting/reading all rows (~10s cold).
            ),
            exploded AS (
                SELECT encoder_value AS enc,
                       time AS t,
                       COALESCE((elem->>'_cam')::int, 0) AS cam,
                       (elem->>'name') AS cls,
                       (elem->>'confidence')::float AS conf,
                       image_path AS img,
                       shipment AS ship,
                       pipeline_id AS pid
                FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
                  AND COALESCE(elem->>'name', '') !~ '^_'  -- 4.0.29: skip synthetic (_color etc.)
                  {_parent_filter_sql}
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY cls ORDER BY t DESC) AS rn
                FROM exploded
            )
            SELECT enc, cam, cls, conf, img, ship, pid FROM ranked
            WHERE rn <= %s
            LIMIT 6000
            """,
            (([int(since_ms), int(until_ms)] if _use_slice else [interval]) + ([shipment] if shipment else []) + _enc_filter_args + [float(min_conf or 0.0)] + _parent_filter_args + [int(dot_cap)]),
        )
        camera_scatter_encoder = [
            {"x": int(enc or 0), "y": cam, "cls": cls,
             "r": round((conf or 0), 3), "img": img, "ship": ship, "pid": pid}
            for enc, cam, cls, conf, img, ship, pid in cur.fetchall()
        ]

        # v4.0.74 — scatter_only short-circuit. When the client is polling
        # bucket-by-bucket for progressive scatter fill (see charts.js
        # `_progressiveBucketLoad`), the OTHER heavy queries below (size
        # percentiles, confidence bands, color baseline, ejection strips,
        # phase enumeration) already ran on the initial '1h' full render
        # and their DOM stays as-is — refetching them on every bucket would
        # burn 3-5x the DB time for no visible change. Return just the
        # scatter arrays here.
        if int(scatter_only or 0):
            cur.close()
            _payload = {
                "camera_scatter": camera_scatter,
                "camera_scatter_encoder": camera_scatter_encoder,
                "camera_y_order": [],  # order was set on the initial render; keep as-is
                "window": window,
                "since_ms": int(since_ms) if _use_slice else 0,
                "until_ms": int(until_ms) if _use_slice else 0,
                "shipment": shipment,
            }
            # 4.0.163 — bucketed newest-first color loader. When the
            # progressive scatter loader passes `include_color_slice=1`, this
            # slice ALSO returns colour cells for the same time window, binned
            # against the FULL window's range (`bin_ref_lo` / `bin_ref_hi` sent
            # by the client — from the initial /api/detection_charts response's
            # color_heatmap.enc_min/enc_max or color_heatmap_time.t_min/t_max).
            # Client merges these cells into its progressive cache and
            # triggers a chart repaint per bucket, so operators see colour
            # cells fill in incrementally alongside scatter dots (newest →
            # oldest) instead of waiting for one big blob.
            if int(include_color_slice or 0) and _use_slice:
                try:
                    _N_BINS_SLICE = max(4, min(192, int(bins) if bins else 48))
                    _axis = str(color_axis or "encoder").strip().lower()
                    _lo = float(bin_ref_lo or 0.0)
                    _hi = float(bin_ref_hi or 0.0)
                    _color_slice_cells = []
                    if _hi > _lo:
                        _phase_local = (phase or "").strip()
                        _phase_cl_local = " AND elem->>'phase' = %s" if _phase_local else ""
                        _phase_ar_local = [_phase_local] if _phase_local else []
                        _parent_local = (parent_class or "").strip()
                        _parent_cl_local = " AND elem->>'parent_class' = %s" if _parent_local else ""
                        _parent_ar_local = [_parent_local] if _parent_local else []
                        # 4.0.309 — compare LIKE-FOR-LIKE. shipment_median → median vs median;
                        # everything else (shipment_avg default, camera, reference) → mean vs mean.
                        # Fixes "all cells brown": a bin-MEAN vs a cam-MEDIAN on a right-skewed E
                        # distribution made every cell read positive. delta_pct expresses the
                        # deviation as a PERCENT of the baseline (operator: "show 12% up from base").
                        _bl = str(baseline or "").strip().lower()
                        if _bl == "shipment_median":
                            _base_stat = "percentile_cont(0.5) WITHIN GROUP (ORDER BY E)"
                            _cell_stat = "percentile_cont(0.5) WITHIN GROUP (ORDER BY b.E)"
                        else:
                            _base_stat = "AVG(E)"
                            _cell_stat = "AVG(b.E)"
                        _slice_cur = conn.cursor()
                        if _axis == "time":
                            _slice_cur.execute(
                                f"""
                                WITH src AS (
                                    SELECT time, detections, image_path
                                    FROM inference_results
                                    WHERE time >= to_timestamp(%s / 1000.0)
                                      AND time <  to_timestamp(%s / 1000.0)
                                      {ship_clause}
                                      AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                                    {_frame_cap_sql}
                                ),
                                color_rows AS (
                                    SELECT EXTRACT(EPOCH FROM time) * 1000.0 AS t_ms,
                                        (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                                        (elem->>'L')::float AS L, (elem->>'E')::float AS E,
                                        (elem->>'phase') AS phase
                                    FROM src, LATERAL jsonb_array_elements(detections) elem
                                    WHERE elem->>'name' = '_color'
                                      {_phase_cl_local}{_parent_cl_local}
                                ),
                                binned AS (
                                    SELECT cr.cam, cr.phase,
                                           LEAST({_N_BINS_SLICE} - 1, GREATEST(0,
                                                FLOOR(((cr.t_ms - %s) * {_N_BINS_SLICE}) / (%s - %s))))::int AS bin,
                                           cr.L, cr.E
                                    FROM color_rows cr
                                ),
                                cam_phase_baselines AS (
                                    SELECT cam, phase, {_base_stat} AS base_E
                                    FROM binned GROUP BY cam, phase
                                )
                                SELECT b.cam, b.bin, AVG(b.L)::float, AVG(b.E)::float, COUNT(*),
                                       ({_cell_stat} - AVG(cpb.base_E))::float AS delta_e,
                                       CASE WHEN AVG(cpb.base_E) <> 0
                                            THEN (100.0 * ({_cell_stat} - AVG(cpb.base_E)) / AVG(cpb.base_E))::float
                                            ELSE 0.0 END AS delta_pct
                                FROM binned b JOIN cam_phase_baselines cpb
                                     ON cpb.cam = b.cam AND cpb.phase IS NOT DISTINCT FROM b.phase
                                GROUP BY b.cam, b.bin ORDER BY b.cam, b.bin
                                """,
                                [int(since_ms), int(until_ms)] + ([shipment] if shipment else []) + _phase_ar_local + _parent_ar_local + [_lo, _hi, _lo],
                            )
                        else:
                            # 4.0.317 — cap SOURCE frames (src CTE + LIMIT) BEFORE the jsonb
                            # expand so a per-bucket colour query stays ~0.1-0.8s. The bin means
                            # are unchanged by the cap (spatial sample within the column via the
                            # (encoder_value,time) index). Client recomputes delta vs the pooled
                            # baseline, so the per-column base_E here is just a placeholder.
                            _slice_cur.execute(
                                f"""
                                WITH src AS (
                                    SELECT encoder_value, detections, image_path
                                    FROM inference_results
                                    WHERE time >= to_timestamp(%s / 1000.0)
                                      AND time <  to_timestamp(%s / 1000.0)
                                      {ship_clause}{_enc_filter_sql}
                                      AND encoder_value IS NOT NULL
                                      AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                                    {_frame_cap_sql}
                                ),
                                color_rows AS (
                                    SELECT encoder_value AS enc,
                                        (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                                        (elem->>'L')::float AS L, (elem->>'E')::float AS E,
                                        (elem->>'phase') AS phase
                                    FROM src, LATERAL jsonb_array_elements(detections) elem
                                    WHERE elem->>'name' = '_color'
                                      {_phase_cl_local}{_parent_cl_local}
                                ),
                                binned AS (
                                    SELECT cr.cam, cr.phase,
                                           LEAST({_N_BINS_SLICE} - 1, GREATEST(0,
                                                FLOOR(((cr.enc - %s)::numeric * {_N_BINS_SLICE}) / (%s - %s))))::int AS bin,
                                           cr.L, cr.E
                                    FROM color_rows cr
                                ),
                                cam_phase_baselines AS (
                                    SELECT cam, phase, {_base_stat} AS base_E
                                    FROM binned GROUP BY cam, phase
                                )
                                SELECT b.cam, b.bin, AVG(b.L)::float, AVG(b.E)::float, COUNT(*),
                                       ({_cell_stat} - AVG(cpb.base_E))::float AS delta_e,
                                       CASE WHEN AVG(cpb.base_E) <> 0
                                            THEN (100.0 * ({_cell_stat} - AVG(cpb.base_E)) / AVG(cpb.base_E))::float
                                            ELSE 0.0 END AS delta_pct
                                FROM binned b JOIN cam_phase_baselines cpb
                                     ON cpb.cam = b.cam AND cpb.phase IS NOT DISTINCT FROM b.phase
                                GROUP BY b.cam, b.bin ORDER BY b.cam, b.bin
                                """,
                                [int(since_ms), int(until_ms)] + ([shipment] if shipment else []) + _enc_filter_args + _phase_ar_local + _parent_ar_local + [_lo, _hi, _lo],
                            )
                        for cam, bin_idx, mL, mE, n, de, dpct in _slice_cur.fetchall():
                            _color_slice_cells.append({
                                "cam": int(cam or 0), "bin": int(bin_idx or 0),
                                "L": round(mL or 0, 2), "E": round(mE or 0, 2),
                                "delta_e": round(de or 0, 2),
                                "delta_pct": round(dpct or 0, 1), "n": int(n or 0),
                            })
                        _slice_cur.close()
                    _payload["color_slice"] = {
                        "axis": _axis,
                        "since_ms": int(since_ms),
                        "until_ms": int(until_ms),
                        "n_bins": _N_BINS_SLICE,
                        "bin_ref_lo": _lo,
                        "bin_ref_hi": _hi,
                        "cells": _color_slice_cells,
                    }
                except Exception as _cse:
                    logger.debug(f"color_slice compute failed: {_cse}")
                    _payload["color_slice"] = {"cells": []}
            # v4.0.210 — quality + ejection SLICE. Streamed per bucket by the
            # progressive ladder EXACTLY like color_slice, so the quality and
            # ejection strips fill newest→oldest in lock-step with the dots AND
            # come from the SAME endpoint/binning as the scatter. Gated on
            # include_strips so unrelated scatter_only fetches skip the cost.
            if int(include_strips or 0) and _use_slice:
                try:
                    _s_axis = str(color_axis or "encoder").strip().lower()
                    _s_lo = float(bin_ref_lo or 0.0)
                    _s_hi = float(bin_ref_hi or 0.0)
                    _s_nbins = max(4, min(192, int(bins) if bins else 48))
                    # 4.0.316 — append the encoder-range filter so each ENCODER bucket's quality /
                    # ejection / shipment strip is bounded to that bucket (fast, ~1s) — same as the
                    # dots + colour. Without it, the encoder ladder had to aggregate the whole window
                    # per strip (heavy → empty lane/strips). No-op when enc_lo/enc_hi absent.
                    _s_time_sql = ("time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)"
                                   + _enc_filter_sql)
                    _s_time_args = [int(since_ms), int(until_ms)] + _enc_filter_args
                    _s_ship_sql = ship_clause  # "AND shipment = %s" or ""
                    _s_ship_args = [shipment] if shipment else []
                    _s_cur = conn.cursor()
                    try:
                        _sev_map, _hidden, _scale = _dc_load_severity()
                        _q_cells = _dc_quality_cells(
                            _s_cur, _s_axis, _s_lo, _s_hi, _s_nbins,
                            _s_time_sql, _s_time_args, _s_ship_sql, _s_ship_args,
                            _sev_map, _hidden, _scale, _frame_cap_sql)
                        _e_cells = _dc_ejection_cells(
                            _s_cur, _s_axis, _s_lo, _s_hi, _s_nbins,
                            _s_time_sql, _s_time_args, _s_ship_sql, _s_ship_args)
                        _sh_cells = _dc_shipment_cells(
                            _s_cur, _s_axis, _s_lo, _s_hi, _s_nbins,
                            _s_time_sql, _s_time_args, _s_ship_sql, _s_ship_args, _frame_cap_sql)
                    finally:
                        _s_cur.close()
                    _payload["quality_slice"] = {
                        "axis": _s_axis, "since_ms": int(since_ms), "until_ms": int(until_ms),
                        "n_bins": _s_nbins, "bin_ref_lo": _s_lo, "bin_ref_hi": _s_hi,
                        "cells": _q_cells,
                    }
                    _payload["ejection_slice"] = {
                        "axis": _s_axis, "since_ms": int(since_ms), "until_ms": int(until_ms),
                        "n_bins": _s_nbins, "bin_ref_lo": _s_lo, "bin_ref_hi": _s_hi,
                        "cells": _e_cells,
                    }
                    _payload["shipment_slice"] = {
                        "axis": _s_axis, "since_ms": int(since_ms), "until_ms": int(until_ms),
                        "n_bins": _s_nbins, "bin_ref_lo": _s_lo, "bin_ref_hi": _s_hi,
                        "cells": _sh_cells,
                    }
                except Exception as _qse:
                    logger.debug(f"quality/ejection/shipment slice compute failed: {_qse}")
                    _payload["quality_slice"] = {"cells": []}
                    _payload["ejection_slice"] = {"cells": []}
                    _payload["shipment_slice"] = {"cells": []}
            _endpoint_cache_put(_cache_key, _payload)
            return JSONResponse(content=_payload)

        # --- confidence by class over time (one line per class) ---
        cur.execute(
            f"""
            WITH recent AS (
                SELECT time, detections FROM inference_results
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                ORDER BY time DESC
                LIMIT 50000  -- v4.0.78: bound the CTE so full-time-range scan can't blow the query
            ),
            expanded AS (
                SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt,
                       (elem->>'name') AS cls,
                       (elem->>'confidence')::float AS conf
                FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
                  AND COALESCE(elem->>'name', '') !~ '^_'  -- 4.0.29: skip synthetic
            )
            SELECT bkt, cls, AVG(conf) AS c_avg
            FROM expanded
            WHERE cls IS NOT NULL
            GROUP BY bkt, cls ORDER BY bkt
            """,
            base_params,
        )
        cbc_buckets, _seen_b = [], set()
        cbc_vals = {}  # cls -> {label: avg_pct}
        for bkt, cls, cavg in cur.fetchall():
            label = bkt.strftime("%m-%d %H:%M")
            if label not in _seen_b:
                _seen_b.add(label); cbc_buckets.append(label)
            cbc_vals.setdefault(cls, {})[label] = round((cavg or 0) * 100, 1)
        confidence_by_class = {
            "buckets": cbc_buckets,
            "series": {cls: [vals.get(b) for b in cbc_buckets] for cls, vals in cbc_vals.items()},
        }
        cur.close()

        # 4.0.31 — initialise these here (default empty) so the camera_y_order
        # block below can union the cameras across scatter + heatmap without a
        # forward-reference error. The actual aggregation happens further down.
        color_heatmap = {"enc_min": None, "enc_max": None, "n_bins": 0, "cells": []}
        color_heatmap_time = {"t_min": None, "t_max": None, "n_bins": 0, "cells": []}
        # We need the heatmap cells available BEFORE building camera_y_order
        # (so the Y axis includes cameras that have ONLY _color samples and no
        # defect dots — the case when the line is idle but colour data is
        # flowing). Run the encoder + time heatmap aggregations now; the
        # downstream baselines block reuses their results.
        # 4.0.34 — `phase` (optional) lets the operator filter the colour heatmap
        # to a single capture phase (e.g. shipment / ejection). Empty string =
        # all phases collapsed (back-compat). Phase filter is heatmap-only —
        # scatter / quality / ejection charts are NOT phase-filtered.
        _phase = (phase or "").strip()
        _phase_clause = " AND elem->>'phase' = %s" if _phase else ""
        _phase_args = [_phase] if _phase else []
        # 4.0.161 — parent_class filter (heatmap-only). Multiple classes can
        # carry role="parent" (`box` AND `PVB_FILM` etc.); each frame emits
        # ONE `_color` per parent detected. The operator picks which parent
        # to view via the toolbar's "Parent class:" pills. Empty = show
        # every parent collapsed (back-compat when only one parent exists
        # OR when a site is using the whole-frame `_root` fallback).
        _parent = (parent_class or "").strip()
        _parent_clause = " AND elem->>'parent_class' = %s" if _parent else ""
        _parent_args = [_parent] if _parent else []
        # 4.0.161 — snapshot the feature flag once at request time so the
        # response payload can carry it back to the frontend without a
        # second /api/color_config round-trip.
        try:
            from config import load_service_config as _lsc_color
            _color_feature_enabled_snapshot = bool(
                (_lsc_color() or {}).get("parent_color_check_enabled", False)
            )
        except Exception:
            _color_feature_enabled_snapshot = False
        try:
            # 4.0.35 — bucket count is now user-configurable from the chart
            # toolbar. Same value powers the colour heatmap and is echoed to
            # the quality / ejection strips (the JS calls those endpoints
            # with the same `buckets` value). Clamp to a sane range.
            try:
                N_BINS = int(bins)
            except Exception:
                N_BINS = 32
            N_BINS = max(4, min(192, N_BINS))
            _hm_cur = conn.cursor()
            # v4.0.149 — kept v4.0.148 stretch: bins 0..N-1 computed against
            # the _color subset's MIN/MAX so slots stay populated, payload
            # enc_min/enc_max come from the full inference range so the
            # frontend spreads the cells across the chart's full width. The
            # v4.0.148 data was correct; the issue was visual — cells with
            # low ΔE painted at alpha=0.35 (green over dark background) were
            # barely visible, making it look like only the recent high-ΔE
            # cells rendered. Alpha bumped in the plugin (see charts.js:2216).
            _hm_cur.execute(
                f"""SELECT MIN(encoder_value), MAX(encoder_value)
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND elem->>'name' = '_color'
                      AND encoder_value IS NOT NULL
                      {_phase_clause}{_parent_clause}""",
                [interval] + ([shipment] if shipment else []) + _phase_args + _parent_args,
            )
            row = _hm_cur.fetchone()
            _color_enc_lo = int(row[0]) if row and row[0] is not None else None
            _color_enc_hi = int(row[1]) if row and row[1] is not None else None
            cells = []
            if _color_enc_lo is not None and _color_enc_hi is not None and _color_enc_hi > _color_enc_lo:
                _hm_cur.execute(
                    f"""
                    WITH color_rows AS (
                        SELECT
                            shipment,
                            encoder_value AS enc,
                            time,
                            (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                            (elem->>'L')::float AS L,
                            (elem->>'E')::float AS E,
                            (elem->>'phase') AS phase
                        FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                        WHERE time > NOW() - INTERVAL %s {ship_clause}
                          AND elem->>'name' = '_color'
                          AND encoder_value IS NOT NULL
                          AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                          {_phase_clause}{_parent_clause}
                    ),
                    -- 4.0.163 — per-(shipment, cam, phase) START baseline.
                    -- Each row scored against ITS OWN shipment's first 30 colour
                    -- samples (not the window-wide median). Prior code used the
                    -- window-wide median which mixed multiple physical products
                    -- across 24h (30+ shipments on razin) → every bin landed far
                    -- from that bogus mixed median → chart all-red regardless of
                    -- actual drift. Now each cell answers the real question
                    -- ("has THIS shipment's colour drifted from its own start?").
                    ranked AS (
                        SELECT shipment, cam, phase, time, E,
                               ROW_NUMBER() OVER (PARTITION BY shipment, cam, phase ORDER BY time) AS rn
                        FROM color_rows
                    ),
                    shipment_start_baselines AS (
                        SELECT shipment, cam, phase,
                               percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                        FROM ranked WHERE rn <= 30
                        GROUP BY shipment, cam, phase
                    ),
                    -- 4.0.173 — per-(shipment, cam, phase) WHOLE-shipment median.
                    -- Operator ask: "shipment start may be defective; give me
                    -- shipment average as an alternative baseline". Same shape
                    -- as shipment_start_baselines but no rn cap.
                    shipment_avg_baselines AS (
                        SELECT shipment, cam, phase,
                               percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                        FROM color_rows
                        GROUP BY shipment, cam, phase
                    ),
                    -- Fallback baseline for shipments too new to have 30 rows yet
                    -- (rare: shipments <5s old). Per (cam, phase) window median.
                    cam_phase_fallback AS (
                        SELECT cam, phase, percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                        FROM color_rows GROUP BY cam, phase
                    ),
                    binned AS (
                        SELECT cr.shipment, cr.cam, cr.phase,
                               LEAST({N_BINS} - 1, GREATEST(0,
                                    FLOOR(((cr.enc - %s)::numeric * {N_BINS}) / (%s - %s))))::int AS bin,
                               cr.L, cr.E
                        FROM color_rows cr
                    )
                    SELECT b.cam, b.bin, AVG(b.L)::float, AVG(b.E)::float, COUNT(*),
                           AVG(b.E - COALESCE(ssb.base_E, cpf.base_E))::float AS delta_from_start,
                           AVG(b.E - COALESCE(sab.base_E, cpf.base_E))::float AS delta_from_avg
                    FROM binned b
                    LEFT JOIN shipment_start_baselines ssb
                         ON ssb.shipment = b.shipment
                        AND ssb.cam = b.cam
                        AND ssb.phase IS NOT DISTINCT FROM b.phase
                    LEFT JOIN shipment_avg_baselines sab
                         ON sab.shipment = b.shipment
                        AND sab.cam = b.cam
                        AND sab.phase IS NOT DISTINCT FROM b.phase
                    LEFT JOIN cam_phase_fallback cpf
                         ON cpf.cam = b.cam
                        AND cpf.phase IS NOT DISTINCT FROM b.phase
                    GROUP BY b.cam, b.bin ORDER BY b.cam, b.bin
                    """,
                    [interval] + ([shipment] if shipment else []) + _phase_args + _parent_args +
                    [_color_enc_lo, _color_enc_hi, _color_enc_lo],
                )
                for row in _hm_cur.fetchall():
                    # 4.0.173 — SQL now returns (cam, bin, L, E, n, delta_from_start, delta_from_avg).
                    # Back-compat unpack: also accept the legacy 6-column shape.
                    if len(row) >= 7:
                        cam, bin_idx, mL, mE, n, de, davg = row[:7]
                    else:
                        cam, bin_idx, mL, mE, n, de = row[:6]; davg = de
                    cells.append({"cam": int(cam or 0), "bin": int(bin_idx or 0),
                                  "L": round(mL or 0, 2), "E": round(mE or 0, 2),
                                  "delta_e": round(de or 0, 2),
                                  "delta_from_avg": round(davg or 0, 2),
                                  "n": int(n or 0)})
            # Payload enc_min/enc_max from FULL inference range → frontend
            # spreads the N cells across the chart's full width.
            _hm_cur.execute(
                f"""SELECT MIN(encoder_value), MAX(encoder_value)
                    FROM inference_results
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND encoder_value IS NOT NULL""",
                [interval] + ([shipment] if shipment else []),
            )
            row = _hm_cur.fetchone()
            color_heatmap = {
                "enc_min": int(row[0]) if row and row[0] is not None else None,
                "enc_max": int(row[1]) if row and row[1] is not None else None,
                "n_bins": N_BINS, "cells": cells,
            }
            # 4.0.162 — color-change strip. Aggregate the per-(cam,bin) cells
            # into per-bin mean E across cameras, then compute |mean_E − mean_E_prev|
            # for each adjacent pair. Frontend paints one cell per bucket
            # between the quality strip and the ejection strip. Detects sudden
            # colour drift between adjacent buckets regardless of absolute
            # baseline (a solid-red heatmap can still show smooth transitions
            # here if the drift is gradual; a lime-green heatmap can show a
            # sharp step here if two consecutive buckets differ a lot).
            color_change_strip_enc = {"n_bins": N_BINS, "cells": []}
            try:
                by_bin_enc = {}
                for c in cells:
                    b = int(c.get("bin") or 0)
                    e = float(c.get("E") or 0)
                    n = int(c.get("n") or 0)
                    if b not in by_bin_enc:
                        by_bin_enc[b] = {"sum_E_weighted": 0.0, "n": 0}
                    by_bin_enc[b]["sum_E_weighted"] += e * n
                    by_bin_enc[b]["n"] += n
                prev_E = None
                ccs_cells = []
                for b in range(N_BINS):
                    agg = by_bin_enc.get(b)
                    if not agg or agg["n"] <= 0:
                        prev_E = None  # gap — reset chain so next non-empty bin has no delta
                        continue
                    mE = agg["sum_E_weighted"] / agg["n"]
                    d = 0.0 if prev_E is None else (mE - prev_E)
                    ccs_cells.append({"bin": b, "E": round(mE, 2),
                                      "delta_prev": round(d, 2), "n": agg["n"]})
                    prev_E = mE
                color_change_strip_enc["cells"] = ccs_cells
            except Exception as _cce:
                logger.debug(f"color_change_strip (encoder) compute failed: {_cce}")
            # Time-binned twin — v4.0.149 honest positioning: bins 0..N-1
            # computed against the FIXED chart window `[now - interval, now]`.
            # Bin K represents the actual time slice under strip cell K.
            # Empty slices produce no cell — honest gap, matches the chart
            # X-axis (`_scatterXMin`/`_scatterXMax`) and the strip below.
            # 4.0.200 — in slice mode (specific shipment OR progressive bucket)
            # bin over the EXACT slice [since_ms, until_ms] so the colour cells
            # span the shipment's real duration instead of the whole window.
            # `_ct_clause` / `_ct_params` are the matching row-filter predicate
            # used by the colour CTE below so its rows come from the same range.
            if _use_slice:
                _t_min = float(since_ms)
                _t_max = float(until_ms)
                _ct_clause = "time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)"
                _ct_params = [since_ms, until_ms]
            else:
                _hm_cur.execute(
                    f"""SELECT EXTRACT(EPOCH FROM (NOW() - INTERVAL %s)) * 1000.0,
                               EXTRACT(EPOCH FROM NOW()) * 1000.0""",
                    [interval],
                )
                row = _hm_cur.fetchone()
                _t_min = float(row[0]) if row and row[0] is not None else None
                _t_max = float(row[1]) if row and row[1] is not None else None
                _ct_clause = "time > NOW() - INTERVAL %s"
                _ct_params = [interval]
            tcells = []
            if _t_min is not None and _t_max is not None and _t_max > _t_min:
                _hm_cur.execute(
                    f"""
                    WITH color_rows AS (
                        SELECT
                            shipment,
                            time,
                            EXTRACT(EPOCH FROM time) * 1000.0 AS t_ms,
                            (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                            (elem->>'L')::float AS L, (elem->>'E')::float AS E,
                            (elem->>'phase') AS phase
                        FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                        WHERE {_ct_clause} {ship_clause}
                          AND elem->>'name' = '_color'
                          AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                          {_phase_clause}{_parent_clause}
                    ),
                    -- 4.0.163 — per-(shipment, cam, phase) START baseline.
                    -- See encoder-heatmap CTE above for rationale.
                    ranked AS (
                        SELECT shipment, cam, phase, time, E,
                               ROW_NUMBER() OVER (PARTITION BY shipment, cam, phase ORDER BY time) AS rn
                        FROM color_rows
                    ),
                    shipment_start_baselines AS (
                        SELECT shipment, cam, phase,
                               percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                        FROM ranked WHERE rn <= 30
                        GROUP BY shipment, cam, phase
                    ),
                    -- 4.0.173 — per-(shipment, cam, phase) whole-shipment median.
                    shipment_avg_baselines AS (
                        SELECT shipment, cam, phase,
                               percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                        FROM color_rows GROUP BY shipment, cam, phase
                    ),
                    cam_phase_fallback AS (
                        SELECT cam, phase, percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                        FROM color_rows GROUP BY cam, phase
                    ),
                    binned AS (
                        SELECT cr.shipment, cr.cam, cr.phase,
                               LEAST({N_BINS} - 1, GREATEST(0,
                                    FLOOR(((cr.t_ms - %s) * {N_BINS}) / (%s - %s))))::int AS bin,
                               cr.L, cr.E
                        FROM color_rows cr
                    )
                    SELECT b.cam, b.bin, AVG(b.L)::float, AVG(b.E)::float, COUNT(*),
                           AVG(b.E - COALESCE(ssb.base_E, cpf.base_E))::float AS delta_from_start,
                           AVG(b.E - COALESCE(sab.base_E, cpf.base_E))::float AS delta_from_avg
                    FROM binned b
                    LEFT JOIN shipment_start_baselines ssb
                         ON ssb.shipment = b.shipment
                        AND ssb.cam = b.cam
                        AND ssb.phase IS NOT DISTINCT FROM b.phase
                    LEFT JOIN shipment_avg_baselines sab
                         ON sab.shipment = b.shipment
                        AND sab.cam = b.cam
                        AND sab.phase IS NOT DISTINCT FROM b.phase
                    LEFT JOIN cam_phase_fallback cpf
                         ON cpf.cam = b.cam
                        AND cpf.phase IS NOT DISTINCT FROM b.phase
                    GROUP BY b.cam, b.bin ORDER BY b.cam, b.bin
                    """,
                    _ct_params + ([shipment] if shipment else []) + _phase_args + _parent_args +
                    [_t_min, _t_max, _t_min],
                )
                # 4.0.161 — was `_color_t_lo/_color_t_hi` (never defined
                # anywhere in this module). The whole try: block silently
                # failed, so `tcells` stayed [] and Time-axis heatmap
                # showed no cells. The actual computed variables at 3585-
                # 3586 are `_t_min` and `_t_max`; use those.
                for row in _hm_cur.fetchall():
                    if len(row) >= 7:
                        cam, bin_idx, mL, mE, n, de, davg = row[:7]
                    else:
                        cam, bin_idx, mL, mE, n, de = row[:6]; davg = de
                    tcells.append({"cam": int(cam or 0), "bin": int(bin_idx or 0),
                                   "L": round(mL or 0, 2), "E": round(mE or 0, 2),
                                   "delta_e": round(de or 0, 2),
                                   "delta_from_avg": round(davg or 0, 2),
                                   "n": int(n or 0)})
            # Payload t_min/t_max = the SAME range the cells were binned over
            # (matches the strip range and Chart.js x-scale). 4.0.200 — in
            # slice mode that's the shipment's real [since_ms, until_ms], so
            # the cells span the shipment's duration instead of collapsing
            # into the last bin of a 90-day window.
            if _use_slice:
                color_heatmap_time = {
                    "t_min": float(since_ms), "t_max": float(until_ms),
                    "n_bins": N_BINS, "cells": tcells,
                }
            else:
                _hm_cur.execute(
                    f"""SELECT EXTRACT(EPOCH FROM (NOW() - INTERVAL %s)) * 1000.0,
                               EXTRACT(EPOCH FROM NOW()) * 1000.0""",
                    [interval],
                )
                row = _hm_cur.fetchone()
                color_heatmap_time = {
                    "t_min": float(row[0]) if row and row[0] is not None else None,
                    "t_max": float(row[1]) if row and row[1] is not None else None,
                    "n_bins": N_BINS, "cells": tcells,
                }
            # 4.0.162 — time-axis color-change strip (mirrors encoder version).
            color_change_strip_time = {"n_bins": N_BINS, "cells": []}
            try:
                by_bin_t = {}
                for c in tcells:
                    b = int(c.get("bin") or 0)
                    e = float(c.get("E") or 0)
                    n = int(c.get("n") or 0)
                    if b not in by_bin_t:
                        by_bin_t[b] = {"sum_E_weighted": 0.0, "n": 0}
                    by_bin_t[b]["sum_E_weighted"] += e * n
                    by_bin_t[b]["n"] += n
                prev_E = None
                ccs_cells_t = []
                for b in range(N_BINS):
                    agg = by_bin_t.get(b)
                    if not agg or agg["n"] <= 0:
                        prev_E = None
                        continue
                    mE = agg["sum_E_weighted"] / agg["n"]
                    d = 0.0 if prev_E is None else (mE - prev_E)
                    ccs_cells_t.append({"bin": b, "E": round(mE, 2),
                                        "delta_prev": round(d, 2), "n": agg["n"]})
                    prev_E = mE
                color_change_strip_time["cells"] = ccs_cells_t
            except Exception as _ccte:
                logger.debug(f"color_change_strip (time) compute failed: {_ccte}")
            _hm_cur.close()
        except Exception as _he:
            logger.debug(f"color heatmap pre-aggregation failed: {_he}")

        # v4.0.210 — quality + ejection heatmaps in the MAIN payload, binned
        # over the SAME bounds + N_BINS the colour heatmap exposes (enc_min/
        # enc_max, t_min/t_max) using the SAME FLOOR bin formula. This is the
        # single-fetch path (picked shipment Core fetch, 1h all-shipments single
        # fetch) — the strips render straight from data.quality_heatmap /
        # data.ejection_heatmap with a GUARANTEED-identical x-domain to the
        # scatter and colour heatmap. The all-shipments ladder instead streams
        # the *_slice variants (above). Gated on include_strips.
        quality_heatmap = {"enc_min": None, "enc_max": None, "n_bins": N_BINS, "cells": []}
        quality_heatmap_time = {"t_min": None, "t_max": None, "n_bins": N_BINS, "cells": []}
        ejection_heatmap = {"enc_min": None, "enc_max": None, "n_bins": N_BINS, "cells": []}
        ejection_heatmap_time = {"t_min": None, "t_max": None, "n_bins": N_BINS, "cells": []}
        shipment_heatmap = {"enc_min": None, "enc_max": None, "n_bins": N_BINS, "cells": []}
        shipment_heatmap_time = {"t_min": None, "t_max": None, "n_bins": N_BINS, "cells": []}
        if int(include_strips or 0):
            try:
                if _use_slice:
                    _mn_time_sql = "time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)"
                    _mn_time_args = [int(since_ms), int(until_ms)]
                else:
                    _mn_time_sql = "time > NOW() - INTERVAL %s"
                    _mn_time_args = [interval]
                _mn_ship_sql = ship_clause
                _mn_ship_args = [shipment] if shipment else []
                _sev_map_m, _hidden_m, _scale_m = _dc_load_severity()
                _enc_lo = color_heatmap.get("enc_min")
                _enc_hi = color_heatmap.get("enc_max")
                _tm_lo = color_heatmap_time.get("t_min")
                _tm_hi = color_heatmap_time.get("t_max")
                _mn_cur = conn.cursor()
                try:
                    if _enc_lo is not None and _enc_hi is not None:
                        quality_heatmap = {
                            "enc_min": _enc_lo, "enc_max": _enc_hi, "n_bins": N_BINS,
                            "cells": _dc_quality_cells(_mn_cur, "encoder", _enc_lo, _enc_hi, N_BINS,
                                                       _mn_time_sql, _mn_time_args, _mn_ship_sql, _mn_ship_args,
                                                       _sev_map_m, _hidden_m, _scale_m),
                        }
                        ejection_heatmap = {
                            "enc_min": _enc_lo, "enc_max": _enc_hi, "n_bins": N_BINS,
                            "cells": _dc_ejection_cells(_mn_cur, "encoder", _enc_lo, _enc_hi, N_BINS,
                                                        _mn_time_sql, _mn_time_args, _mn_ship_sql, _mn_ship_args),
                        }
                        shipment_heatmap = {
                            "enc_min": _enc_lo, "enc_max": _enc_hi, "n_bins": N_BINS,
                            "cells": _dc_shipment_cells(_mn_cur, "encoder", _enc_lo, _enc_hi, N_BINS,
                                                        _mn_time_sql, _mn_time_args, _mn_ship_sql, _mn_ship_args),
                        }
                    if _tm_lo is not None and _tm_hi is not None:
                        quality_heatmap_time = {
                            "t_min": _tm_lo, "t_max": _tm_hi, "n_bins": N_BINS,
                            "cells": _dc_quality_cells(_mn_cur, "time", _tm_lo, _tm_hi, N_BINS,
                                                       _mn_time_sql, _mn_time_args, _mn_ship_sql, _mn_ship_args,
                                                       _sev_map_m, _hidden_m, _scale_m),
                        }
                        ejection_heatmap_time = {
                            "t_min": _tm_lo, "t_max": _tm_hi, "n_bins": N_BINS,
                            "cells": _dc_ejection_cells(_mn_cur, "time", _tm_lo, _tm_hi, N_BINS,
                                                        _mn_time_sql, _mn_time_args, _mn_ship_sql, _mn_ship_args),
                        }
                        shipment_heatmap_time = {
                            "t_min": _tm_lo, "t_max": _tm_hi, "n_bins": N_BINS,
                            "cells": _dc_shipment_cells(_mn_cur, "time", _tm_lo, _tm_hi, N_BINS,
                                                        _mn_time_sql, _mn_time_args, _mn_ship_sql, _mn_ship_args),
                        }
                finally:
                    _mn_cur.close()
            except Exception as _mse:
                logger.debug(f"main quality/ejection/shipment heatmap compute failed: {_mse}")

        # 4.0.34 — discover the phases that actually have _color data in this
        # window so the UI can render exactly the right buttons. NOT filtered
        # by the selected phase (we want the full menu either way).
        phases_available = []
        try:
            _ph_cur = conn.cursor()
            _ph_cur.execute(
                f"""SELECT DISTINCT elem->>'phase' AS phase
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND elem->>'name' = '_color'
                      AND elem->>'phase' IS NOT NULL
                    ORDER BY 1""",
                [interval] + ([shipment] if shipment else []),
            )
            phases_available = [r[0] for r in _ph_cur.fetchall() if r and r[0]]
            _ph_cur.close()
        except Exception as _pe:
            logger.debug(f"phases_available discovery failed: {_pe}")

        # 4.0.161 — discover the parent classes with _color data in this
        # window. Same idea as phases_available: NOT filtered by the current
        # parent_class selection so the operator can always switch.
        # Empty string parent_class means source="_root" (whole-frame
        # fallback); we surface that as "_root" in the list so the picker
        # can show it as a distinct choice.
        parent_classes_available = []
        try:
            _pc_cur = conn.cursor()
            _pc_cur.execute(
                f"""SELECT DISTINCT COALESCE(NULLIF(elem->>'parent_class', ''), '_root') AS pc
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND elem->>'name' = '_color'
                    ORDER BY 1""",
                [interval] + ([shipment] if shipment else []),
            )
            parent_classes_available = [r[0] for r in _pc_cur.fetchall() if r and r[0]]
            _pc_cur.close()
        except Exception as _pce:
            logger.debug(f"parent_classes_available discovery failed: {_pce}")

        # 4.0.24 — propagate the dashboard's camera column ordering into the
        # chart so the scatter Y-axis matches the timeline grid the operator
        # is already used to. Single source of truth: timeline_config.camera_order
        # + timeline_config.custom_camera_order (same fields websocket.py reads
        # to lay out the timeline columns at routers/websocket.py:150-164).
        # Any camera that appears in the data but NOT in the custom list gets
        # appended at the end in numeric order, so nothing disappears.
        camera_y_order = []
        try:
            tl_config = getattr(request.app.state, "timeline_config", {}) or {}
            mode = str(tl_config.get("camera_order") or "normal").lower()
            # Discover every camera present in the scatter payloads so the
            # axis covers EVERYTHING shown, even cams not enumerated in the
            # operator's custom list.
            # 4.0.31 — also include cameras that appear ONLY in the color
            # heatmap (i.e., no defect dots but we have _color samples). Without
            # this, when the line is stopped or the YOLO is quiet, the Y-axis
            # collapses to empty and the heatmap plugin can't map cell.cam to
            # a Y index — so the background never paints even though cells
            # exist.
            seen = set()
            for pt in camera_scatter_encoder:
                seen.add(int(pt.get("y") or 0))
            for pt in camera_scatter:
                seen.add(int(pt.get("y") or 0))
            for cell in (color_heatmap.get("cells") or []):
                seen.add(int(cell.get("cam") or 0))
            for cell in (color_heatmap_time.get("cells") or []):
                seen.add(int(cell.get("cam") or 0))
            seen.discard(0)
            if mode == "custom":
                raw = str(tl_config.get("custom_camera_order") or "").strip()
                explicit = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
                # Order = explicit list first (preserving order, dropping dups),
                # then any cameras present in data that weren't enumerated.
                seen_in_order = set()
                for cid in explicit:
                    if cid not in seen_in_order:
                        camera_y_order.append(cid)
                        seen_in_order.add(cid)
                for cid in sorted(seen):
                    if cid not in seen_in_order:
                        camera_y_order.append(cid)
            elif mode == "reverse":
                camera_y_order = sorted(seen, reverse=True)
            else:  # "normal" or anything unknown
                camera_y_order = sorted(seen)
        except Exception as _e:
            logger.debug(f"camera_y_order compute failed (falling back to numeric): {_e}")
            camera_y_order = []

        # 4.0.31 — color_heatmap + color_heatmap_time were aggregated
        # earlier (right after confidence_by_class) so camera_y_order could
        # union the heatmap-only cameras into the Y-axis order. The old
        # duplicate blocks that lived here have been removed.
        color_baseline = {}

        # 4.0.30 — baseline for the heatmap's ΔE computation. The frontend
        # picks ONE mode at a time via the `baseline` query param; we only
        # compute that one to avoid running SQL the operator didn't ask for.
        # Modes:
        #   camera          — median E across the visible window per camera
        #                     (default; auto, no operator action needed)
        #   shipment_start  — median E of the FIRST 60 s of the active shipment
        #                     per camera ("colour at shipment start = correct")
        #   target          — operator-set L*a*b* per camera in
        #                     service_config.color_target (manual gold standard)
        #   reference_frame — L*a*b* of a single operator-picked frame stored
        #                     as service_config.color_reference_frame_id
        # `color_baseline_modes` reports which modes have data available right
        # now so the frontend can enable/disable toggle buttons accordingly.
        try:
            # ---- availability survey (cheap) ----
            # We always tell the frontend which modes COULD work right now
            # so it can grey-out / enable the toggle buttons accordingly,
            # without computing values for unused modes.
            # 4.0.173 — "shipment_avg" mode always available; frontend reads
            # per-cell `delta_from_avg` from color_heatmap cells.
            color_baseline_modes = ["camera", "shipment_avg", "shipment_median"]
            try:
                from config import load_service_config as _lsc
                _svc = _lsc() or {}
            except Exception:
                _svc = {}
            if shipment:
                color_baseline_modes.append("shipment_start")
            if isinstance(_svc.get("color_target"), dict) and _svc.get("color_target"):
                color_baseline_modes.append("target")
            # 4.0.31 — reference_frame is available when EITHER an explicit
            # frame_id is configured OR the operator has clicked a position
            # on the chart (color_reference_position).
            if (_svc.get("color_reference_frame_id") or "").strip() \
               or isinstance(_svc.get("color_reference_position"), dict):
                color_baseline_modes.append("reference_frame")

            # ---- selected mode (clamp to available) ----
            mode = baseline if baseline in color_baseline_modes else "camera"

            # ---- compute only the picked baseline ----
            cur = conn.cursor()
            if mode == "camera":
                cur.execute(
                    f"""
                    WITH color_rows AS (
                        SELECT
                            (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                            (elem->>'E')::float AS E,
                            (elem->>'L')::float AS L,
                            (elem->>'a')::float AS a_ch,
                            (elem->>'b')::float AS b_ch
                        FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                        WHERE time > NOW() - INTERVAL %s {ship_clause}
                          AND elem->>'name' = '_color'
                          AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                    )
                    SELECT cam,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY E)    AS E,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY L)    AS L,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY a_ch) AS a_ch,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY b_ch) AS b_ch
                    FROM color_rows GROUP BY cam
                    """,
                    [interval] + ([shipment] if shipment else []),
                )
                for cam, E_, L_, a_, b_ in cur.fetchall():
                    if cam is None: continue
                    color_baseline[str(int(cam))] = {
                        "E": round(E_ or 0, 2), "L": round(L_ or 0, 2),
                        "a": round(a_ or 0, 2), "b": round(b_ or 0, 2),
                    }
            elif mode == "shipment_start":
                cur.execute(
                    """
                    WITH color_rows AS (
                        SELECT time,
                            (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                            (elem->>'E')::float AS E,
                            (elem->>'L')::float AS L,
                            (elem->>'a')::float AS a_ch,
                            (elem->>'b')::float AS b_ch
                        FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                        WHERE shipment = %s
                          AND elem->>'name' = '_color'
                          AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                    ),
                    start_window AS (SELECT MIN(time) AS t0 FROM color_rows)
                    SELECT cam,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY E)    AS E,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY L)    AS L,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY a_ch) AS a_ch,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY b_ch) AS b_ch
                    FROM color_rows, start_window
                    WHERE time <= t0 + INTERVAL '60 seconds'
                    GROUP BY cam
                    """,
                    [shipment],
                )
                for cam, E_, L_, a_, b_ in cur.fetchall():
                    if cam is None: continue
                    color_baseline[str(int(cam))] = {
                        "E": round(E_ or 0, 2), "L": round(L_ or 0, 2),
                        "a": round(a_ or 0, 2), "b": round(b_ or 0, 2),
                    }
            elif mode == "target":
                tgt = _svc.get("color_target") or {}
                for k, v in tgt.items():
                    try:
                        cid = str(int(k))
                        if not isinstance(v, dict): continue
                        L_ = float(v.get("L") or 0)
                        a_ = float(v.get("a") or 0)
                        b_ = float(v.get("b") or 0)
                        E_ = float(v.get("E") or (L_*L_ + a_*a_ + b_*b_) ** 0.5)
                        color_baseline[cid] = {"E": round(E_,2), "L": round(L_,2),
                                               "a": round(a_,2), "b": round(b_,2)}
                    except (TypeError, ValueError):
                        continue
            elif mode == "reference_frame":
                ref_id = (_svc.get("color_reference_frame_id") or "").strip()
                cur.execute(
                    """
                    SELECT (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                           (elem->>'E')::float AS E,
                           (elem->>'L')::float AS L,
                           (elem->>'a')::float AS a_ch,
                           (elem->>'b')::float AS b_ch
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE image_path LIKE %s
                      AND elem->>'name' = '_color'
                    ORDER BY time DESC LIMIT 50
                    """,
                    ('%' + ref_id + '%',),
                )
                for cam, E_, L_, a_, b_ in cur.fetchall():
                    if cam is None: continue
                    color_baseline[str(int(cam))] = {
                        "E": round(E_ or 0, 2), "L": round(L_ or 0, 2),
                        "a": round(a_ or 0, 2), "b": round(b_ or 0, 2),
                    }
            cur.close()
        except Exception as _be:
            logger.debug(f"color baseline compute failed: {_be}")

        # v4.0.111 — per-shipment encoder + time spans, for the new
        # "shipment lanes" strip the operator asked for. Rendered between
        # the scatter and the quality strip on the Charts tab so it's
        # instantly obvious which encoder / time range corresponds to
        # which shipment id. Same window scoping and shipment filter as
        # the rest of the endpoint. Ordered oldest-first so left → right
        # on the strip reads oldest → newest.
        shipment_spans = []
        # v4.0.132 — rollback any aborted transaction state left by
        # earlier heavy queries in this endpoint (color_heatmap, etc.).
        # Without this, on khoy the earlier queries can hit
        # statement_timeout, leave the transaction aborted, and this
        # shipment_spans query then fails with "current transaction is
        # aborted" — silently swallowed by logger.debug so the payload
        # went out with shipment_spans=[] and shipment_length_offsets={}
        # even though the /api/shipment_spans standalone endpoint (which
        # uses a fresh connection) returned the same data fine.
        try: conn.rollback()
        except Exception: pass
        try:
            _sp_cur = conn.cursor()
            # v4.0.114 — bounded-CTE variant. The v4.0.111 shape did a raw
            # `SELECT ... FROM inference_results WHERE time > NOW() - INTERVAL
            # %s GROUP BY shipment` which on khoy (1.5 M rows/hour) walked
            # every row in the window and blew past the 15 s statement_timeout
            # for anything past ~1 h. The CTE caps the base scan at 200 k
            # rows (200 k / 1.5 M ≈ 8 min of production data — plenty to
            # cover every active shipment; a shipment that hasn't produced a
            # row in the last 200 k inserts is genuinely idle and its span
            # from earlier in the window isn't interesting for the strip).
            # ORDER BY time DESC + LIMIT lets Postgres walk the time index
            # from newest → oldest, stopping when the cap is hit.
            _sp_cur.execute("SET LOCAL statement_timeout = 10000")
            _sp_cur.execute(
                f"""
                WITH recent AS (
                    SELECT shipment, encoder_value, time
                    FROM inference_results
                    WHERE { 'time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)' if _use_slice else 'time > NOW() - INTERVAL %s' } {ship_clause}
                      AND shipment IS NOT NULL
                      AND shipment <> ''
                    ORDER BY time DESC
                    LIMIT 200000
                )
                SELECT shipment,
                       MIN(encoder_value)                        AS enc_min,
                       MAX(encoder_value)                        AS enc_max,
                       EXTRACT(EPOCH FROM MIN(time)) * 1000.0    AS t_min,
                       EXTRACT(EPOCH FROM MAX(time)) * 1000.0    AS t_max,
                       COUNT(*)                                  AS n_rows
                FROM recent
                GROUP BY shipment
                ORDER BY MIN(time) ASC
                LIMIT 200
                """,
                (([int(since_ms), int(until_ms)] if _use_slice else [interval]) + ([shipment] if shipment else [])),
            )
            for _s, _e_min, _e_max, _t_min, _t_max, _n in _sp_cur.fetchall():
                if _s is None: continue
                shipment_spans.append({
                    "shipment": str(_s),
                    "enc_min": int(_e_min) if _e_min is not None else None,
                    "enc_max": int(_e_max) if _e_max is not None else None,
                    "t_min":   float(_t_min) if _t_min is not None else None,
                    "t_max":   float(_t_max) if _t_max is not None else None,
                    "n_rows":  int(_n or 0),
                })
            _sp_cur.close()
        except Exception as _spe:
            # v4.0.132 — raise from debug → warning so the operator can
            # see when this compute fails. It's the source of the length-
            # axis offsets AND the shipment strip; a silent failure here
            # is why the length toggle looked broken on khoy.
            logger.warning(f"shipment_spans compute failed: {type(_spe).__name__}: {_spe}")
            try: conn.rollback()
            except Exception: pass

        # v4.0.132 — cumulative length offsets so the frontend can render a
        # single continuous "length axis" across all shipments in the window,
        # immune to PLC-encoder resets. For each shipment (ordered by first-
        # timestamp), the offset is the sum of prior shipments' encoder spans.
        # Frontend then computes each dot's global-length x as:
        #    offsets[dot.ship].offset + max(0, dot.encoder - offsets[dot.ship].enc_min)
        # `enc_min` is included so the frontend doesn't have to look it up
        # separately. If a shipment straddles a PLC reset within its span,
        # dots after the reset will overlap dots from before the reset on
        # the length axis — that's a limitation of not storing per-row
        # length. Most shipments don't reset.
        shipment_length_offsets = {}
        try:
            # 4.0.326 — SEQUENTIAL length axis, reset-safe. Each shipment's segment width is
            # its RESET-SAFE length_raw (Σ forward encoder deltas) from shipment_summary, NOT
            # raw enc_max-enc_min. A shipment whose PLC encoder didn't cleanly reset (enc_max in
            # the billions) used to make _cum explode to tens of billions of metres and left the
            # axis mostly empty. length_raw is bounded to the true fabric length, so shipments now
            # lay out back-to-back at their real lengths in time order (operator's ribbon layout).
            _len_raw_by_ship = {}
            try:
                _sids = [str(_sp.get("shipment")) for _sp in shipment_spans if _sp.get("shipment")]
                if _sids:
                    from services.shipment_stats import get_length_raw_map
                    _len_raw_by_ship = get_length_raw_map(_sids) or {}
            except Exception as _lrm:
                logger.debug(f"length_raw map failed: {_lrm}")
            _cum = 0
            for _sp in shipment_spans:
                _sid = _sp.get("shipment")
                if not _sid:
                    continue
                _em = int(_sp.get("enc_min") or 0)
                _ex = int(_sp.get("enc_max") or 0)
                _raw_span = max(0, _ex - _em)
                # Prefer the cached reset-safe length; fall back to the raw span only for the
                # in-progress shipment (not finalized into shipment_summary yet). The current
                # shipment's encoder resets cleanly at its own start, so its raw span is bounded.
                _lr = _len_raw_by_ship.get(str(_sid))
                _span = int(_lr) if (_lr is not None and _lr > 0) else _raw_span
                shipment_length_offsets[_sid] = {
                    "offset": int(_cum),
                    "enc_min": int(_em),
                    "enc_max": int(_ex),             # 4.0.328 — raw range, for normalising dots into the segment
                    "length_span": int(_span),
                    "t_min": _sp.get("t_min"),
                    "t_max": _sp.get("t_max"),
                }
                _cum += _span
        except Exception as _sle:
            logger.debug(f"shipment_length_offsets compute failed: {_sle}")

        # v4.0.134 — class-group map for shape-based scatter markers.
        # 4 groups drive point-shape on the Camera×Encoder scatter:
        #   defect  = severity > 0 in audio_settings (real issue to flag)     → circle
        #   parent  = class in parent_object_list (base material/object)      → diamond (rectRot)
        #   marker  = class in marker_object_list (e.g. stitch, seam)         → triangle
        #   context = fallback (unclassified / low-signal)                    → cross
        # Frontend reads the map to set per-point pointStyle. Color still
        # comes from the per-class palette so shape adds a SECOND channel
        # for "type of thing" without competing with color.
        class_groups = {}
        try:
            from config import load_service_config as _lsc_cg
            _svc_cg = _lsc_cg() or {}
            _audio_cg = _svc_cg.get("audio_settings", {}) or {}
            _img_proc = _svc_cg.get("image_processing", {}) or {}
            _parents_cg = set(_img_proc.get("parent_object_list") or [])
            _markers_cg = set(_img_proc.get("marker_object_list") or [])
            # v4.0.136 — enumerate EVERY class the config knows about, not just
            # ones with dots in the current dot_cap slice. Progressive bucket
            # streaming later brings older classes onto the chart; the frontend
            # caches this map and looks up shapes by class name, so a missing
            # entry means the dot renders as `context` (cross) even when the
            # class is really a defect. This was the WIR bug on vteam12: WIR
            # only appeared in the older buckets, so `class_groups` was
            # {TB:defect} at fetch time and WIR drew as a cross.
            _all_classes = set()
            _all_classes.update(k for k in _audio_cg.keys() if k)
            _all_classes.update(_parents_cg)
            _all_classes.update(_markers_cg)
            # Still union with anything actually seen in the current window,
            # so a class that exists on disk / DB without a config entry still
            # gets a group ("context" via the fallback below).
            for _row in (camera_scatter_encoder or []):
                _c = _row.get("cls")
                if _c: _all_classes.add(str(_c))
            for _row in (camera_scatter or []):
                _c = _row.get("cls")
                if _c: _all_classes.add(str(_c))
            for _c in _all_classes:
                # v4.0.146 — read the per-class `role` field on
                # audio_settings[cls] FIRST. The Process tab UI writes the
                # marker/parent tag there (audio_settings.TB.role = 'marker'),
                # not into the legacy image_processing.marker_object_list
                # (which stays None on sites that never used the old bulk
                # editor). Prior to this fix, TB was classified as context on
                # kiancord even though the operator had explicitly set it as
                # a marker via the per-class UI.
                _entry = _audio_cg.get(_c, {}) or {}
                _role = str(_entry.get("role") or "").strip().lower()
                if _role == "parent":
                    class_groups[_c] = "parent"
                    continue
                if _role == "marker":
                    class_groups[_c] = "marker"
                    continue
                # 4.0.180 — orientation-tagged defects. Frontend renders these
                # as narrow rectangles on the scatter (▮ vertical, ▬ horizontal).
                if _role == "vertical_defect":
                    class_groups[_c] = "vertical_defect"
                    continue
                if _role == "horizontal_defect":
                    class_groups[_c] = "horizontal_defect"
                    continue
                # Legacy bulk lists — kept for sites that still use them.
                if _c in _parents_cg:
                    class_groups[_c] = "parent"
                    continue
                if _c in _markers_cg:
                    class_groups[_c] = "marker"
                    continue
                _sev = _entry.get("severity", 0)
                try: _sev = float(_sev or 0)
                except Exception: _sev = 0
                class_groups[_c] = "defect" if _sev > 0 else "context"
        except Exception as _cge:
            logger.debug(f"class_groups compute failed: {_cge}")

        _payload = {
            "shipments": shipments,
            "size_over_time": size_over_time,
            "confidence_over_time": confidence_over_time,
            "confidence_by_class": confidence_by_class,
            "camera_scatter": camera_scatter,
            "camera_scatter_encoder": camera_scatter_encoder,
            "class_groups": class_groups,
            "camera_y_order": camera_y_order,
            "color_heatmap": color_heatmap,
            "color_heatmap_time": color_heatmap_time,
            # v4.0.210 — quality + ejection strips from the SAME endpoint +
            # binning as the scatter/colour. Empty {cells:[]} unless
            # include_strips=1. The frontend renders the strips straight from
            # these so a vertical guideline hits the same data in every layer.
            "quality_heatmap": (locals().get("quality_heatmap") or {"n_bins": 0, "cells": []}),
            "quality_heatmap_time": (locals().get("quality_heatmap_time") or {"n_bins": 0, "cells": []}),
            "ejection_heatmap": (locals().get("ejection_heatmap") or {"n_bins": 0, "cells": []}),
            "ejection_heatmap_time": (locals().get("ejection_heatmap_time") or {"n_bins": 0, "cells": []}),
            # v4.0.217 — shipment lane as a proper bin-strip: dominant shipment per
            # bin, same bounds/bins as the others → the lane streams + syncs like
            # quality/ejection instead of being derived from the sparse dots.
            "shipment_heatmap": (locals().get("shipment_heatmap") or {"n_bins": 0, "cells": []}),
            "shipment_heatmap_time": (locals().get("shipment_heatmap_time") or {"n_bins": 0, "cells": []}),
            # 4.0.162 — per-bucket colour-change strips. Encoder + time variants.
            "color_change_strip": (locals().get("color_change_strip_enc") or {"n_bins": 0, "cells": []}),
            "color_change_strip_time": (locals().get("color_change_strip_time") or {"n_bins": 0, "cells": []}),
            "phases_available": phases_available,
            "phase": _phase,
            # 4.0.161 — same shape as phases_available. Frontend renders
            # one pill per entry in the toolbar and stores the choice in
            # localStorage as `mve_heatmap_parent_class`.
            "parent_classes_available": parent_classes_available,
            "parent_class": _parent,
            # 4.0.161 — surface the "Parent Object Color Check" toggle here so
            # the frontend heatmap plugin doesn't race with loadColorConfig's
            # background fetch. Falls back to False on any read error.
            "parent_color_check_enabled": _color_feature_enabled_snapshot,
            "bins": N_BINS,
            "hidden_by_process": _hidden_by_process,
            "color_baseline": color_baseline,
            "color_baseline_mode": baseline if baseline in color_baseline_modes else "camera",
            "color_baseline_modes": color_baseline_modes,
            # 4.0.32 — surface the picked reference point so the chart can
            # paint a vertical marker showing the operator's chosen baseline
            # spot. Null when nothing's been picked.
            "color_reference_position": (locals().get("_svc") or {}).get("color_reference_position"),
            "window": window,
            "shipment": shipment,
            # v4.0.114 — the strip renderer in charts.js reads this key
            # (introduced in v4.0.111 but never wired into the payload).
            # Empty list when the CTE returns nothing.
            "shipment_spans": shipment_spans,
            # v4.0.132 — {shipment_id: {offset, enc_min, length_span, t_min, t_max}}
            "shipment_length_offsets": shipment_length_offsets,
        }
        _endpoint_cache_put(_cache_key, _payload)
        return JSONResponse(content=_payload)
    except Exception as e:
        logger.warning(f"detection_charts query failed (returning empty): {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.get("/api/shipment_baseline")
def shipment_baseline_endpoint(request: Request, shipment: str = "", window: str = "7d",
                               phase: str = "", parent_class: str = ""):
    """4.0.318b — per-camera colour baseline from the materialized cache (instant table read).

    A picked FINISHED shipment → its own cached stats; all-shipments → pooled over the window's
    finished shipments. `current: true` with empty baselines means the pick is the in-progress
    shipment (not finalized) → the client shows ABSOLUTE values + neighbour-bucket deltas instead
    of a baseline comparison. Replaces the old per-bucket client-side baseline accumulation.
    """
    try:
        from services.shipment_stats import get_cached_baseline
        return JSONResponse(content=get_cached_baseline(
            shipment=shipment, window=window, phase=phase, parent_class=parent_class))
    except Exception as e:
        logger.debug(f"shipment_baseline endpoint failed: {e}")
        return JSONResponse(content={"source": "none", "current": False, "baselines": {}})


@router.get("/api/shipment_scores")
def shipment_scores_endpoint(request: Request, limit: int = 50):
    """4.0.318b — the Score-per-shipment lane, read straight from shipment_summary (latest N).
    No per-shipment recompute — a finished shipment's score was materialized at finalize time."""
    try:
        from services.shipment_stats import get_latest_shipment_scores
        return JSONResponse(content={"shipments": get_latest_shipment_scores(int(limit or 50))})
    except Exception as e:
        logger.debug(f"shipment_scores endpoint failed: {e}")
        return JSONResponse(content={"shipments": []})


@router.get("/api/shipment_spans")
def shipment_spans_endpoint(request: Request, window: str = "1h", shipment: str = ""):
    """v4.0.119 — Standalone lightweight endpoint that ONLY returns the
    shipment_spans list. The Charts-tab dashboard uses `window=24h` on
    `/api/detection_charts`, which on vteam12 takes >30 s to run all
    five CTEs — the browser fetch hangs, `_refreshAdvancedChartsCore`
    never resolves, and the shipment strip never paints even though the
    span data itself is trivial (200 k-row bounded CTE, ~70 ms).

    This endpoint isolates the strip data so the frontend can fetch it
    IMMEDIATELY on page load and paint the strip regardless of what the
    rest of the dashboard is doing.

    4.0.196 — accepts optional `shipment` param. When set, the endpoint
    returns exactly ONE span (the picked shipment) and widens the
    time filter to 90 days so an older shipment still surfaces. Mirrors
    the /api/detection_charts fix so the shipment strip painted by
    the standalone refresher matches what the operator asked for
    instead of continuing to render every shipment in the window.
    """
    _windows_map = {
        "1h":  "1 hour",
        "6h":  "6 hours",
        "24h": "24 hours",
        "7d":  "7 days",
    }
    interval = _windows_map.get(window, "1 hour")
    if shipment:
        interval = "90 days"
    empty = {"shipment_spans": [], "window": window}
    # 4.0.322 — a SPECIFIC finished shipment: serve its authoritative FULL span from the
    # shipment_summary cache (reset-safe enc_min/max, never truncated by the 200k LIMIT,
    # instant, can't drop on the tunnel). The frontend picked-shipment ladder reads
    # enc_min/enc_max from here to PIN the encoder axis; the old live scan of a 2M-row
    # shipment returned only the newest 200k rows' partial span → the pick's dots squished
    # into one axis column at ~2.5M on a stale 93M axis. Cache MISS (current in-progress
    # shipment, not finalized yet) falls through to the live scan below unchanged.
    if shipment:
        try:
            from services.shipment_stats import get_shipment_span
            _c = get_shipment_span(shipment)
            if _c and _c.get("enc_min") is not None and _c.get("enc_max") is not None \
               and _c["enc_max"] > _c["enc_min"]:
                _sp = {
                    "shipment": shipment,
                    "enc_min": _c["enc_min"], "enc_max": _c["enc_max"],
                    "t_min":   _c["t_min"],   "t_max":   _c["t_max"],
                    "n_rows":  _c["n_rows"],
                    "verdict": _c.get("verdict"), "score": _c.get("score"),
                }
                return JSONResponse(content={"shipment_spans": [_sp], "window": window, "cached": True})
        except Exception as _cse:
            logger.debug(f"shipment_spans cache read failed: {_cse}")
    _ship_where = "AND shipment = %s" if shipment else ""
    _ship_params = [interval] + ([shipment] if shipment else [])
    conn = None
    try:
        from services.db import get_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()
        cur.execute("SET LOCAL statement_timeout = 8000")
        cur.execute(
            f"""
            WITH recent AS (
                SELECT shipment, encoder_value, time
                FROM inference_results
                WHERE time > NOW() - INTERVAL %s
                  AND shipment IS NOT NULL
                  AND shipment <> ''
                  {_ship_where}
                ORDER BY time DESC
                LIMIT 200000
            )
            SELECT shipment,
                   MIN(encoder_value)                        AS enc_min,
                   MAX(encoder_value)                        AS enc_max,
                   EXTRACT(EPOCH FROM MIN(time)) * 1000.0    AS t_min,
                   EXTRACT(EPOCH FROM MAX(time)) * 1000.0    AS t_max,
                   COUNT(*)                                  AS n_rows
            FROM recent
            GROUP BY shipment
            ORDER BY MIN(time) ASC
            LIMIT 200
            """,
            tuple(_ship_params),
        )
        spans = []
        for _s, _e_min, _e_max, _t_min, _t_max, _n in cur.fetchall():
            spans.append({
                "shipment": _s,
                "enc_min": int(_e_min) if _e_min is not None else None,
                "enc_max": int(_e_max) if _e_max is not None else None,
                "t_min":   float(_t_min) if _t_min is not None else None,
                "t_max":   float(_t_max) if _t_max is not None else None,
                "n_rows":  int(_n or 0),
            })
        cur.close()
        # v4.0.123 — annotate each span with its RELEASE / RE-INSPECT / HOLD
        # / PENDING verdict so the shipment-strip renderer can color boxes
        # by status instead of by golden-angle hash. Cached per shipment
        # with a 60 s TTL because `_compute_quality_payload` is expensive
        # (does its own scoring pass over inference_results).
        _now = time.time()
        _cache = getattr(shipment_spans_endpoint, "_verdict_cache", None)
        if _cache is None:
            _cache = {}
            shipment_spans_endpoint._verdict_cache = _cache
        for _sp in spans:
            _ship = _sp.get("shipment") or ""
            if not _ship or _ship == "no_shipment":
                _sp["verdict"] = None
                _sp["score"] = None
                continue
            _entry = _cache.get(_ship)
            if _entry and (_now - _entry[0]) < 60.0:
                _sp["verdict"] = _entry[1]
                _sp["score"]   = _entry[2]
                continue
            try:
                _qp = _compute_quality_payload(shipment=_ship, window=window)
                _v = _qp.get("verdict") if _qp else None
                _sc = _qp.get("score")   if _qp else None
                _sp["verdict"] = _v
                _sp["score"]   = _sc
                _cache[_ship]  = (_now, _v, _sc)
            except Exception:
                _sp["verdict"] = None
                _sp["score"]   = None
        return JSONResponse(content={"shipment_spans": spans, "window": window})
    except Exception as e:
        logger.warning(f"shipment_spans standalone endpoint failed: {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.get("/api/shipment_length_offsets")
def shipment_length_offsets_endpoint(request: Request, window: str = "24h"):
    """4.0.327 — sequential length-axis layout. Returns, per finished shipment in the window,
    {offset, enc_min, length_span} ordered by time, where length_span is the RESET-SAFE length_raw
    (Σ forward encoder deltas) and offset is the running cumulative sum. The Length axis lays every
    shipment out back-to-back over one continuous span [0, total] = Σ lengths — no raw-encoder gaps.
    Reads straight from shipment_summary (cheap, indexed); the in-progress shipment (not finalized)
    is omitted until it closes."""
    _map = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days",
            "30d": "30 days", "90d": "90 days"}
    interval = _map.get(str(window or "24h"), "24 hours")
    empty = {"offsets": {}, "total": 0, "window": window}
    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()
        cur.execute("SET LOCAL statement_timeout = 8000")
        cur.execute(
            """SELECT shipment,
                      EXTRACT(EPOCH FROM t_start) * 1000.0,
                      EXTRACT(EPOCH FROM t_stop)  * 1000.0,
                      enc_min, enc_max, length_raw, verdict
               FROM shipment_summary
               WHERE t_stop > NOW() - INTERVAL %s
               ORDER BY t_start ASC NULLS LAST""",
            [interval],
        )
        offsets = {}
        cum = 0
        for _sid, _t0, _t1, _emin, _emax, _lraw, _vrd in cur.fetchall():
            if not _sid:
                continue
            _span = int(_lraw) if (_lraw is not None and _lraw > 0) \
                else max(0, int(_emax or 0) - int(_emin or 0))
            offsets[str(_sid)] = {
                "offset": int(cum),
                "enc_min": int(_emin or 0),
                "enc_max": int(_emax or 0),          # 4.0.328 — raw range, for normalising dots into the segment
                "length_span": int(_span),
                "verdict": _vrd,                       # 4.0.332 — so the lane can be painted from offsets (colour + label)
                "t_min": float(_t0) if _t0 is not None else None,
                "t_max": float(_t1) if _t1 is not None else None,
            }
            cum += _span
        cur.close()
        return JSONResponse(content={"offsets": offsets, "total": int(cum), "window": window})
    except Exception as e:
        logger.warning(f"shipment_length_offsets endpoint failed: {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.get("/api/insight_layout")
def insight_layout_endpoint(request: Request, axis: str = "length", window: str = "24h",
                            shipment: str = "", phase: str = "", parent_class: str = "",
                            bins: int = 48, baseline: str = "camera"):
    """v4.0.334 — the ONE up-front call for the Detection Insights scatter refactor.

    Returns everything the client needs to fix the layout and know the colour baseline BEFORE it
    streams any bucket, so colour can be applied inline (never racing the ladder):
      - `shipments`: the finished shipments in the window (or the one picked), each with its
        cumulative `offset`, reset-safe `length_span`, `enc_min/enc_max`, `verdict`, time bounds,
        and its per-camera colour `baseline` {cam: {avg, median, n}} read from shipment_stats.
      - `domain` {lo, hi} for the chosen axis (+ `total` for length), `bins`, and `camera_y_order`.
    Reads ONLY the materialized caches (shipment_summary + shipment_stats) — cheap, instant,
    reset-safe. Replaces the old scatter's fan-out (fire-and-forget /api/shipment_baseline +
    range probe + /api/shipment_length_offsets) with a single deterministic read.
    """
    _map = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days",
            "30d": "30 days", "90d": "90 days"}
    interval = _map.get(str(window or "24h"), "24 hours")
    try:
        _bins = max(1, int(bins or 48))
    except Exception:
        _bins = 48
    empty = {"axis": axis, "window": window, "shipment": shipment, "bins": _bins,
             "domain": {"lo": 0, "hi": 0}, "total": 0, "shipments": [],
             "camera_y_order": [], "baseline_mode": baseline}
    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()
        cur.execute("SET LOCAL statement_timeout = 8000")

        # --- 1) shipment list: finished shipments in the window, or the one picked ---
        if shipment:
            cur.execute(
                """SELECT shipment, EXTRACT(EPOCH FROM t_start)*1000.0,
                          EXTRACT(EPOCH FROM t_stop)*1000.0, enc_min, enc_max, length_raw, verdict
                   FROM shipment_summary WHERE shipment = %s LIMIT 1""",
                [shipment])
        else:
            cur.execute(
                """SELECT shipment, EXTRACT(EPOCH FROM t_start)*1000.0,
                          EXTRACT(EPOCH FROM t_stop)*1000.0, enc_min, enc_max, length_raw, verdict
                   FROM shipment_summary
                   WHERE t_stop > NOW() - INTERVAL %s
                   ORDER BY t_start ASC NULLS LAST""",
                [interval])
        ships = []
        cum = 0
        t_lo = t_hi = e_lo = e_hi = None
        for _sid, _t0, _t1, _emin, _emax, _lraw, _vrd in cur.fetchall():
            if not _sid:
                continue
            _emin_i = int(_emin or 0)
            _emax_i = int(_emax or 0)
            _span = int(_lraw) if (_lraw is not None and _lraw > 0) else max(0, _emax_i - _emin_i)
            ships.append({
                "sid": str(_sid), "offset": int(cum),
                "enc_min": _emin_i, "enc_max": _emax_i, "length_span": int(_span),
                "verdict": _vrd,
                "t_min": float(_t0) if _t0 is not None else None,
                "t_max": float(_t1) if _t1 is not None else None,
                "baseline": {},
            })
            cum += _span
            if _t0 is not None:
                t_lo = _t0 if t_lo is None else min(t_lo, _t0)
            if _t1 is not None:
                t_hi = _t1 if t_hi is None else max(t_hi, _t1)
            e_lo = _emin_i if e_lo is None else min(e_lo, _emin_i)
            e_hi = _emax_i if e_hi is None else max(e_hi, _emax_i)

        # --- 2) per-(camera, shipment, PARENT OBJECT) colour baseline from shipment_stats ---
        # NO parent_class filter — return EVERY parent object's baseline so the client colours each
        # parent against its OWN normal (operator: "get the base for each parent object separately";
        # different parent objects are different materials with different normal colours, so pooling
        # them corrupts the baseline). Shape: shipments[].baseline = {parent: {cam: {avg,median,n}}}.
        _ph = " AND st.phase = %s" if str(phase or "").strip() else ""
        _ph_a = [str(phase).strip()] if str(phase or "").strip() else []
        by_ship = {s["sid"]: s for s in ships}
        cams = set()
        if by_ship:
            if shipment:
                cur.execute(
                    f"""SELECT st.shipment, st.parent_class, st.cam,
                               SUM(st.avg_e*st.n)/NULLIF(SUM(st.n),0),
                               SUM(st.median_e*st.n)/NULLIF(SUM(st.n),0), SUM(st.n)
                        FROM shipment_stats st
                        WHERE st.shipment = %s {_ph}
                        GROUP BY st.shipment, st.parent_class, st.cam""",
                    [shipment] + _ph_a)
            else:
                cur.execute(
                    f"""SELECT st.shipment, st.parent_class, st.cam,
                               SUM(st.avg_e*st.n)/NULLIF(SUM(st.n),0),
                               SUM(st.median_e*st.n)/NULLIF(SUM(st.n),0), SUM(st.n)
                        FROM shipment_stats st
                        JOIN shipment_summary su ON su.shipment = st.shipment
                        WHERE su.t_stop > NOW() - INTERVAL %s {_ph}
                        GROUP BY st.shipment, st.parent_class, st.cam""",
                    [interval] + _ph_a)
            for _sid, _pc, _cam, _avg, _med, _n in cur.fetchall():
                if _cam is None:
                    continue
                cams.add(int(_cam))
                s = by_ship.get(str(_sid))
                if s is not None:
                    _pkey = str(_pc) if _pc is not None else ""
                    s["baseline"].setdefault(_pkey, {})[str(int(_cam))] = {
                        "avg": round(float(_avg), 3) if _avg is not None else None,
                        "median": round(float(_med), 3) if _med is not None else None,
                        "n": int(_n or 0),
                    }
        cur.close()

        camera_y_order = [str(c) for c in sorted(cams)]
        total = int(cum)
        if axis == "length":
            domain = {"lo": 0, "hi": total}
        elif axis == "encoder":
            domain = {"lo": int(e_lo or 0), "hi": int(e_hi or 0)}
        else:  # time
            domain = {"lo": float(t_lo or 0), "hi": float(t_hi or 0)}

        return JSONResponse(content={
            "axis": axis, "window": window, "shipment": shipment, "bins": _bins,
            "domain": domain, "total": total, "shipments": ships,
            "camera_y_order": camera_y_order, "baseline_mode": baseline,
        })
    except Exception as e:
        logger.warning(f"insight_layout endpoint failed: {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                conn.rollback()  # SELECT-only → release clean, no idle-in-transaction
            except Exception:
                pass
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.get("/api/ejection_stats")
def ejection_stats(request: Request, window: str = "24h", shipment: str = ""):
    """Ejection analytics for the Charts tab (3.17.0).

    Reads the ejection_events table (one row per triggered procedure with
    Store=ON). Scoped to a window + optional shipment, returns:
      - by_procedure: {procedure_name: count}  → distribution bar + doughnut
      - timeline:     [{t, count}]             → ejections over time (number)
      - total:        total ejection events in the window
      - shipments:    distinct shipment ids (for the dropdown)
    Returns a well-formed empty payload on any error (e.g. table not created yet).
    """
    _windows = {
        "1h":  ("1 hour",   "1 minute"),
        "6h":  ("6 hours",  "5 minutes"),
        "24h": ("24 hours", "30 minutes"),
        "7d":  ("7 days",   "6 hours"),
    }
    interval, bucket = _windows.get(window, _windows["24h"])
    # 4.0.196 — specific shipment overrides window (see /api/detection_charts)
    if shipment:
        interval = "90 days"
    empty = {"by_procedure": {}, "timeline": [], "total": 0,
             "shipments": [], "window": window, "shipment": shipment}
    ship_clause = "AND shipment = %s" if shipment else ""
    base_params = [interval] + ([shipment] if shipment else [])

    conn = None
    try:
        from services.db import get_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()

        # distinct shipments seen in ejection_events within the window
        cur.execute(
            "SELECT DISTINCT shipment FROM ejection_events "
            "WHERE time > NOW() - INTERVAL %s AND shipment IS NOT NULL "
            "ORDER BY shipment LIMIT 100",
            (interval,),
        )
        shipments = sorted(set([r[0] for r in cur.fetchall() if r[0]]) | set(_list_shipment_dirs()))

        # count per procedure
        cur.execute(
            f"""SELECT COALESCE(procedure_name, 'Unnamed') AS p, COUNT(*) AS c
                FROM ejection_events
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                GROUP BY p ORDER BY c DESC""",
            base_params,
        )
        by_procedure, total = {}, 0
        for p, c in cur.fetchall():
            by_procedure[p] = int(c)
            total += int(c)

        # ejections over time (bucketed count)
        cur.execute(
            f"""SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt, COUNT(*) AS c
                FROM ejection_events
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                GROUP BY bkt ORDER BY bkt""",
            base_params,
        )
        timeline = [
            {"t": bkt.strftime("%m-%d %H:%M"), "count": int(c)}
            for bkt, c in cur.fetchall()
        ]
        cur.close()

        return JSONResponse(content={
            "by_procedure": by_procedure,
            "timeline": timeline,
            "total": total,
            "shipments": shipments,
            "window": window,
            "shipment": shipment,
        })
    except Exception as e:
        logger.warning(f"ejection_stats query failed (returning empty): {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.get("/api/production_stats")
def production_stats(request: Request, window: str = "24h", shipment: str = ""):
    """Line KPIs from production_metrics (3.18.0).

    OKC/NGC are *cumulative* hardware counters from the PLC (serial KV: OKC, NGC,
    DWS downtime, ENC encoder, PPS→is_moving). We diff consecutive samples per
    bucket and clamp negatives (a restart resets the counter) to get per-bucket
    OK / NG counts → reject-rate, throughput (units), and uptime %. Also returns
    overall p̄ + total units so the frontend can draw a proper SPC p-chart.
    """
    _windows = {
        "1h":  ("1 hour",   "1 minute"),
        "6h":  ("6 hours",  "5 minutes"),
        "24h": ("24 hours", "30 minutes"),
        "7d":  ("7 days",   "6 hours"),
    }
    interval, bucket = _windows.get(window, _windows["24h"])
    # 4.0.196 — specific shipment overrides window (see /api/detection_charts)
    if shipment:
        interval = "90 days"
    bucket_seconds = {"1h": 60, "6h": 300, "24h": 1800, "7d": 21600}.get(window, 1800)
    empty = {"timeline": [], "total_ok": 0, "total_ng": 0, "total_units": 0,
             "reject_rate_overall": 0.0, "eject_over_total": 0.0,
             "availability": 0.0, "performance": 0.0, "quality": 0.0, "oee": 0.0,
             "downtime_total_s": 0.0, "speed_avg": 0.0, "speed_max": 0.0,
             "shipments": [], "window": window, "shipment": shipment}
    ship_clause = "AND shipment = %s" if shipment else ""
    base_params = [interval] + ([shipment] if shipment else [])

    conn = None
    try:
        from services.db import get_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()

        cur.execute(
            "SELECT DISTINCT shipment FROM production_metrics "
            "WHERE time > NOW() - INTERVAL %s AND shipment IS NOT NULL "
            "ORDER BY shipment LIMIT 100",
            (interval,),
        )
        shipments = sorted(set([r[0] for r in cur.fetchall() if r[0]]) | set(_list_shipment_dirs()))

        # OKC/NGC/encoder/downtime are cumulative — diff consecutive samples, clamp
        # resets. Speed = encoder delta / bucket seconds. Downtime = downtime delta.
        cur.execute(
            f"""
            WITH ordered AS (
                SELECT time, is_moving,
                       ok_counter - LAG(ok_counter) OVER (ORDER BY time) AS d_ok,
                       ng_counter - LAG(ng_counter) OVER (ORDER BY time) AS d_ng,
                       encoder_value - LAG(encoder_value) OVER (ORDER BY time) AS d_enc,
                       downtime_seconds - LAG(downtime_seconds) OVER (ORDER BY time) AS d_dt
                FROM production_metrics
                WHERE time > NOW() - INTERVAL %s {ship_clause}
            ),
            b AS (
                SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt,
                       GREATEST(COALESCE(d_ok, 0), 0) AS d_ok,
                       GREATEST(COALESCE(d_ng, 0), 0) AS d_ng,
                       GREATEST(COALESCE(d_enc, 0), 0) AS d_enc,
                       GREATEST(COALESCE(d_dt, 0), 0) AS d_dt,
                       CASE WHEN is_moving THEN 1.0 ELSE 0.0 END AS mv
                FROM ordered
            )
            SELECT bkt, SUM(d_ok) AS ok, SUM(d_ng) AS ng, AVG(mv) * 100 AS uptime,
                   SUM(d_enc) AS enc, SUM(d_dt) AS dt
            FROM b GROUP BY bkt ORDER BY bkt
            """,
            base_params,
        )
        rows = cur.fetchall()
        cur.close()

        # pass 1: per-bucket primitives + peak speed (encoder units/sec) for Performance
        raw, speed_max = [], 0.0
        for bkt, ok, ng, uptime, enc, dt in rows:
            ok = int(ok or 0); ng = int(ng or 0)
            speed = float(enc or 0) / bucket_seconds if bucket_seconds else 0.0
            if speed > speed_max:
                speed_max = speed
            raw.append((bkt, ok, ng, float(uptime or 0), speed, float(dt or 0)))

        # pass 2: build timeline with per-bucket OEE = Availability × Performance × Quality
        timeline, t_ok, t_ng, t_dt = [], 0, 0, 0.0
        speed_sum, speed_n, avail_sum = 0.0, 0, 0.0
        for bkt, ok, ng, uptime, speed, dt in raw:
            units = ok + ng
            t_ok += ok; t_ng += ng; t_dt += dt
            avail_sum += uptime
            if speed > 0:
                speed_sum += speed; speed_n += 1
            availability = uptime / 100.0
            quality = (ok / units) if units else 0.0
            performance = (speed / speed_max) if speed_max else 0.0
            timeline.append({
                "t": bkt.strftime("%m-%d %H:%M"),
                "ok": ok, "ng": ng, "units": units,
                "reject_rate": round(ng / units * 100, 2) if units else 0.0,
                "uptime_pct": round(uptime, 1),
                "speed": round(speed, 2),
                "downtime_s": round(dt, 1),
                "oee": round(availability * performance * quality * 100, 1),
            })

        t_units = t_ok + t_ng
        n_buckets = len(raw) or 1
        availability_o = avail_sum / n_buckets / 100.0
        quality_o = (t_ok / t_units) if t_units else 0.0
        speed_avg = (speed_sum / speed_n) if speed_n else 0.0
        performance_o = (speed_avg / speed_max) if speed_max else 0.0
        return JSONResponse(content={
            "timeline": timeline,
            "total_ok": t_ok, "total_ng": t_ng, "total_units": t_units,
            "reject_rate_overall": round(t_ng / t_units * 100, 2) if t_units else 0.0,
            "eject_over_total": round(t_ng / t_units * 100, 2) if t_units else 0.0,
            "availability": round(availability_o * 100, 1),
            "performance": round(performance_o * 100, 1),
            "quality": round(quality_o * 100, 1),
            "oee": round(availability_o * performance_o * quality_o * 100, 1),
            "downtime_total_s": round(t_dt, 1),
            "speed_avg": round(speed_avg, 2),
            "speed_max": round(speed_max, 2),
            "shipments": shipments, "window": window, "shipment": shipment,
        })
    except Exception as e:
        logger.warning(f"production_stats query failed (returning empty): {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.get("/api/quality_charts")
def quality_charts(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0):
    """Defect diagnostics from inference_results (3.18.0).

    Single expansion pass returns:
      - by_class:   detections per class (drives the Pareto chart)
      - by_camera:  detections per camera id (which station sees most defects)
      - heatmap:    bbox-center density binned into a gw×gh grid (normalized by the
                    max observed x/y) → where on the product defects cluster
      - latency_over_time: per-bucket avg/max inference_time_ms (model/pipeline health)
    Capped at recent rows; returns a well-formed empty payload on any error.
    """
    _windows = {
        "1h":  ("1 hour",   "1 minute"),
        "6h":  ("6 hours",  "5 minutes"),
        "24h": ("24 hours", "30 minutes"),
        "7d":  ("7 days",   "6 hours"),
    }
    interval, bucket = _windows.get(window, _windows["24h"])
    # 4.0.196 — specific shipment overrides window (see /api/detection_charts)
    if shipment:
        interval = "90 days"
    GW, GH = 32, 20
    empty = {"by_class": {}, "by_camera": {},
             "heatmap": {"gw": GW, "gh": GH, "max": 0, "cells": []},
             "latency_over_time": [], "window": window, "shipment": shipment}
    ship_clause = "AND shipment = %s" if shipment else ""
    base_params = [interval] + ([shipment] if shipment else [])

    conn = None
    try:
        from services.db import get_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()
        # v4.0.102 — same 3s-timeout rescue. Also, this endpoint's outer
        # try/except was surfacing "list index out of range" on khoy — that's
        # NOT a timeout, it's `base_params` being shorter than the two %s
        # placeholders when the query was rewritten with `{ship_clause}` but
        # the confidence filter (which needs `min_conf`) wasn't added to
        # `base_params`. Adding it here alongside the timeout raise fixes
        # both bugs at once.
        cur.execute("SET LOCAL statement_timeout = 15000")

        # by_class / by_camera / heatmap centers (one expansion pass, capped)
        cur.execute(
            f"""
            WITH recent AS (
                SELECT detections FROM inference_results
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                ORDER BY time DESC
            )
            SELECT (elem->>'name') AS cls,
                   COALESCE((elem->>'_cam')::int, 0) AS cam,
                   ((elem->>'xmin')::float + (elem->>'xmax')::float) / 2.0 AS cx,
                   ((elem->>'ymin')::float + (elem->>'ymax')::float) / 2.0 AS cy
            FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
            WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
            LIMIT 40000
            """,
            base_params + [float(min_conf or 0.0)],   # v4.0.102 — was missing min_conf
        )
        by_class, by_camera, centers = {}, {}, []
        max_x, max_y = 1.0, 1.0
        for cls, cam, cx, cy in cur.fetchall():
            if cls is not None:
                by_class[cls] = by_class.get(cls, 0) + 1
            ck = str(cam if cam is not None else 0)
            by_camera[ck] = by_camera.get(ck, 0) + 1
            if cx and cy and cx > 0 and cy > 0:
                centers.append((cx, cy))
                if cx > max_x: max_x = cx
                if cy > max_y: max_y = cy

        grid = {}
        for cx, cy in centers:
            gx = min(GW - 1, int(cx / max_x * GW))
            gy = min(GH - 1, int(cy / max_y * GH))
            grid[(gx, gy)] = grid.get((gx, gy), 0) + 1
        max_c = max(grid.values()) if grid else 0
        cells = [{"x": gx, "y": gy, "c": c} for (gx, gy), c in grid.items()]
        heatmap = {"gw": GW, "gh": GH, "max": max_c, "cells": cells}

        # inference latency over time (per row, not per detection)
        cur.execute(
            f"""SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt,
                       AVG(inference_time_ms) AS avg_ms, MAX(inference_time_ms) AS max_ms
                FROM inference_results
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                GROUP BY bkt ORDER BY bkt""",
            base_params,
        )
        latency_over_time = [
            {"t": bkt.strftime("%m-%d %H:%M"),
             "avg": round(float(a or 0), 1), "max": round(float(m or 0), 1)}
            for bkt, a, m in cur.fetchall()
        ]
        cur.close()
        # 4.0.103 — the underlying SELECT is bounded by `LIMIT 40000` (line
        # 3994) so `by_class` + `by_camera` + `heatmap.cells` are all counts
        # over the SAME 40k-detection sample, NOT over the whole window.
        # Surface that fact so the frontend can label the tiles "recent
        # 40k sample" instead of pretending the numbers are the totals.
        # `sample_size` = the actual number of detections the counts came
        # from (may be less than the cap if the window was small). Compare
        # this against `/api/detection_stats.total` for context — that one
        # is the true full-window count.
        sample_size = sum(by_class.values())
        SAMPLE_CAP = 40000
        return JSONResponse(content={
            "by_class": by_class, "by_camera": by_camera, "heatmap": heatmap,
            "latency_over_time": latency_over_time, "window": window, "shipment": shipment,
            "sample": {
                "size": sample_size,
                "cap":  SAMPLE_CAP,
                "capped": sample_size >= SAMPLE_CAP,
                "note": (
                    f"counts over the most-recent {SAMPLE_CAP}-detection sample "
                    f"(hit the cap; the full-window total is exposed by /api/detection_stats)"
                    if sample_size >= SAMPLE_CAP else
                    f"counts over {sample_size} detections in the window (below the {SAMPLE_CAP} cap)"
                ),
            },
        })
    except Exception as e:
        logger.warning(f"quality_charts query failed (returning empty): {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.post("/api/timeline_clear")
def timeline_clear(request: Request):
    """Clear all frames from the timeline buffer."""
    try:
        redis_client = Redis("redis", 6379, db=cfg_module.REDIS_DB)
        # Clear all timeline keys
        timeline_keys = redis_client.keys(f"{TIMELINE_REDIS_PREFIX}*")
        if timeline_keys:
            redis_client.delete(*timeline_keys)
            logger.info(f"Cleared {len(timeline_keys)} timeline buffers")
        return {"success": True, "cleared": len(timeline_keys)}
    except Exception as e:
        logger.error(f"Error clearing timeline: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/timeline_config")
def get_timeline_config(request: Request):
    """Return current in-memory timeline configuration."""
    config = getattr(request.app.state, 'timeline_config', {})
    return JSONResponse(content=config)


@router.post("/api/timeline_config")
async def update_timeline_config(request: Request):
    """Update timeline configuration (bounding boxes, camera order, quality, rows, buffer size)."""
    try:
        data = await request.json()

        # Extract configuration values
        show_bounding_boxes = data.get('show_bounding_boxes', True)  # Global bbox toggle
        camera_order = data.get('camera_order', 'normal')
        custom_camera_order = data.get('custom_camera_order', '')
        image_quality = data.get('image_quality', 85)
        num_rows = data.get('num_rows', 10)
        buffer_size = data.get('buffer_size', 20)  # Total frames to store
        image_rotation = data.get('image_rotation', 0)  # 0, 90, 180, 270
        procedures = data.get('procedures', getattr(request.app.state, 'timeline_config', {}).get('procedures', []))

        # 3.21.22 — object_filters is deprecated. The Process tab's
        # audio_settings.<class>.show is the canonical source of truth for the
        # annotator (see services/draw_filters.py). We preserve any existing
        # object_filters dict on the in-memory state for backwards compatibility
        # with any reader I haven't migrated yet, but we no longer ACCEPT
        # writes to it from the timeline_config POST. If a legacy client still
        # sends `object_filters`, we silently ignore it.
        _prev_tc = getattr(request.app.state, 'timeline_config', {}) or {}
        preserved_object_filters = _prev_tc.get('object_filters', {})
        if 'object_filters' in data:
            logger.info("update_timeline_config: ignoring deprecated `object_filters` field in payload (use POST /api/audio_settings instead)")

        # Store configuration in app state
        request.app.state.timeline_config = {
            'show_bounding_boxes': show_bounding_boxes,
            'camera_order': camera_order,
            'custom_camera_order': custom_camera_order,
            'image_quality': image_quality,
            'num_rows': num_rows,
            'buffer_size': buffer_size,
            'image_rotation': image_rotation,
            'object_filters': preserved_object_filters,
            'procedures': procedures
        }

        # Save to DATA_FILE
        file_data = load_data_file()
        file_data['timeline_config'] = request.app.state.timeline_config
        save_data_file(file_data)

        # Clear timeline buffer so new frames with new quality settings are captured
        redis_client = Redis("redis", 6379, db=cfg_module.REDIS_DB)
        timeline_keys = redis_client.keys(f"{TIMELINE_REDIS_PREFIX}*")
        if timeline_keys:
            redis_client.delete(*timeline_keys)
            logger.info(f"Cleared {len(timeline_keys)} timeline buffers to apply new settings")

        logger.info(f"Timeline config saved: order={camera_order}, quality={image_quality}, rows={num_rows}, buffer={buffer_size}, object_filters={len(preserved_object_filters)} preserved (deprecated)")

        return {
            'success': True,
            'message': 'Timeline configuration updated successfully. Buffer cleared - new frames will use new settings.',
            'config': request.app.state.timeline_config
        }
    except Exception as e:
        logger.error(f"Error updating timeline config: {e}")
        return JSONResponse(
            status_code=500,
            content={'success': False, 'error': str(e)}
        )


@router.get("/api/recent_detections")
def recent_detections(request: Request, window: str = "24h", shipment: str = "",
                            cls: str = "", limit: int = 24):
    """Recent stored detections + image paths for the chart click-through gallery (3.21.0).

    Filters by class (JSONB containment), optional shipment, time window. Returns
    rows from inference_results that contain at least one detection of `cls`, with
    the annotated image path so the UI can show a thumbnail grid. Capped at `limit`.
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
    interval = _windows.get(window, "24 hours")
    limit = max(1, min(100, int(limit or 24)))
    empty = {"items": [], "class": cls, "shipment": shipment, "window": window}

    ship_clause = "AND shipment = %s" if shipment else ""
    cls_clause = "AND detections @> %s::jsonb" if cls else ""
    params = [interval]
    if shipment:
        params.append(shipment)
    if cls:
        import json as _json
        params.append(_json.dumps([{"name": cls}]))
    params.append(limit)

    conn = None
    try:
        from services.db import get_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT EXTRACT(EPOCH FROM time) * 1000 AS t,
                   shipment, image_path, detection_count, detections
            FROM inference_results
            WHERE time > NOW() - INTERVAL %s {ship_clause} {cls_clause}
            ORDER BY time DESC LIMIT %s
            """,
            params,
        )
        items = []
        for t, ship, img, dcount, dets in cur.fetchall():
            # extract a compact summary: the matching class and its confidence
            best_conf = None
            classes_in_frame = []
            try:
                for d in (dets or []):
                    if isinstance(d, dict):
                        nm = d.get("name")
                        if nm:
                            classes_in_frame.append(nm)
                        if (not cls) or nm == cls:
                            c = float(d.get("confidence") or 0)
                            if best_conf is None or c > best_conf:
                                best_conf = c
            except Exception:
                pass
            items.append({
                "t": int(t),
                "shipment": ship,
                "image_path": img,
                "detection_count": int(dcount or 0),
                "best_confidence": round(best_conf or 0, 3),
                "classes": classes_in_frame[:6],
            })
        cur.close()
        return JSONResponse(content={"items": items, "class": cls,
                                     "shipment": shipment, "window": window})
    except Exception as e:
        logger.warning(f"recent_detections query failed (returning empty): {e}")
        return JSONResponse(content=empty)
    finally:
        if conn is not None:
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


# v4.0.152 — CSV export column list (shared between single-CSV and per-
# shipment ZIP paths). Kept here so both paths can't drift.
_CSV_COLUMNS = [
    "time", "shipment", "encoder", "length", "unwind_length", "camera", "class", "confidence",
    "severity", "impact",
    "xmin", "ymin", "xmax", "ymax", "image_url",
    "inference_time_ms", "model_used",
]


def _encoder_calibration():
    """Return (units_per_unit, unit_label) for converting raw encoder counts to
    the operator's calibrated unit. (None, "") when uncalibrated → CSV falls back
    to raw encoder counts with an un-suffixed header."""
    try:
        from config import load_service_config as _lsc
        svc = _lsc() or {}
        unit = str(svc.get("encoder_unit") or "").strip()
        upu = svc.get("encoder_units_per_unit")
        if upu in (None, "", 0):
            upu = svc.get("encoder_units_per_meter")   # legacy key
        upu = float(upu or 0)
    except Exception:
        return None, ""
    if upu <= 0:
        return None, ""
    if not unit or unit == "encoder_unit":
        unit = "meters"   # legacy encoder_units_per_meter implies meters
    return upu, unit


def _csv_columns_with_unit(unit):
    """`_CSV_COLUMNS` with the two length headers suffixed by the calibrated unit,
    e.g. `length (meters)` / `unwind_length (meters)`. Raw when unit is empty."""
    cols = list(_CSV_COLUMNS)
    if unit:
        cols[3] = f"length ({unit})"
        cols[4] = f"unwind_length ({unit})"
    return cols


def _detection_row(t, ship, enc, img, infms, model, d, sev_map, enc_min, enc_max, min_conf_f, img_prefix="", upm=None):
    """Build one detection row (list of column values) or None if the row is
    filtered out by min_conf. Extracted so single-CSV and ZIP paths share it.

    Both length conventions ship as columns (no mode toggle):
      length         = encoder − shipment-start  (start = 0)
      unwind_length  = shipment-end − encoder     (end = 0, "meters left to unwind")
    """
    try:
        conf_f = float(d.get("confidence") or 0.0)
        if conf_f < min_conf_f:
            return None
    except (TypeError, ValueError):
        conf_f = 0.0
    cls = d.get("name", "")
    sev = sev_map.get(cls, 0)
    impact = round((sev / 100.0) * conf_f, 3)
    if enc is None:
        length_val = ""
        unwind_length_val = ""
    else:
        _e = int(enc)
        _raw_len = _e - enc_min
        _raw_unw = enc_max - _e
        if upm and upm > 0:
            length_val = round(_raw_len / upm, 2)
            unwind_length_val = round(_raw_unw / upm, 2)
        else:
            length_val = int(_raw_len)
            unwind_length_val = int(_raw_unw)
    # 4.0.343 — image column is a DIRECT download URL via the /api/raw_image/<rel> serve route
    # (operator: "put the downloadable link so they can download right away"), built on whatever host
    # the operator is using. Falls back to the raw path when no prefix is supplied.
    if img and img_prefix:
        rel = img.split("raw_images/")[-1].lstrip("/") if "raw_images/" in str(img) else str(img).lstrip("/")
        img_out = img_prefix + rel
    else:
        img_out = img or ""
    return [
        t.strftime("%Y-%m-%d %H:%M:%S.%f") if t else "",
        ship or "", enc if enc is not None else "",
        length_val, unwind_length_val,
        d.get("_cam", ""), cls,
        d.get("confidence", ""),
        sev, impact,
        d.get("xmin", ""), d.get("ymin", ""),
        d.get("xmax", ""), d.get("ymax", ""),
        img_out, infms or "", model or "",
    ]


@router.get("/api/export_csv")
def export_csv(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0):
    """Export stored detections for a window.

    v4.0.152 — output format now depends on the `shipment` parameter:
      * `shipment=<id>`  → single CSV (streamed).      Filename: detections_<id>_<window>.csv
      * `shipment=`      → ZIP of one CSV per shipment. Filename: detections_all_<window>.zip

    The ZIP path is what the "CSV" button on Charts triggers when the operator
    hasn't picked a specific shipment. Each entry inside the ZIP is named
    `detections_<shipment>_<window>.csv` so unzipping into a directory gives
    one file per shipment the operator can open in Excel independently.

    Column list (shared):
      time, shipment, encoder, length, unwind_length, camera, class, confidence,
      severity, impact, xmin, ymin, xmax, ymax, image_url, inference_time_ms, model_used

    Both length conventions are always present as columns: `length` = encoder −
    shipment-start, `unwind_length` = shipment-end − encoder (meters left to
    unwind, end = 0). No mode toggle.
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
    interval = _windows.get(window, "24 hours")
    # 4.0.345 — a PICKED shipment is exported by SHIPMENT ID ALONE — NO time filter whatsoever. The
    # whole point of picking a shipment is "give me THIS shipment's rows", regardless of when it ran.
    # The sliding window (and even 4.0.342's 90-day widen) still dropped rows for a shipment outside
    # that span → empty CSV (operator: "when I choose a shipment, only WHERE shipment=X, nothing to
    # do with time"). The all-shipments ZIP path keeps the window — that one genuinely means
    # "everything in this window".
    if shipment:
        where_sql = "shipment = %s"
        where_params = [shipment]
    else:
        where_sql = "time > NOW() - INTERVAL %s"
        where_params = [interval]
    min_conf_f = float(min_conf or 0.0)
    # 4.0.343 — build image download URLs off the host the operator is actually on (request.base_url),
    # via the existing /api/raw_image/<rel> serve route, so the CSV's image column is click-to-download.
    try:
        img_prefix = str(request.base_url).rstrip("/") + "/api/raw_image/"
    except Exception:
        img_prefix = "/api/raw_image/"

    from services.db import get_db_connection, release_db_connection
    conn = get_db_connection()
    if conn is None:
        return Response(content="ERROR: database unavailable\n", status_code=503,
                        media_type="text/plain")

    import csv as _csv
    import io as _io
    import re as _re

    # Shared: severity map + window-wide enc_min/enc_max for length column.
    try:
        from config import load_service_config as _lsc
        _svc = _lsc() or {}
        _audio = _svc.get("audio_settings", {}) or {}
        sev_map = {k: int(v.get("severity", 0) or 0) for k, v in _audio.items() if isinstance(v, dict)}
    except Exception:
        sev_map = {}
    _upm, _unit = _encoder_calibration()   # convert length columns to real units

    # enc_min/enc_max feed ONLY the derived "length" column (encoder − start, or max − encoder for
    # unwind). The raw rows don't need it.
    enc_min_global = 0
    enc_max_global = 0
    if shipment:
        # 4.0.346 — a PICKED shipment's encoder range comes from the shipment_summary CACHE (instant),
        # NOT a MIN/MAX scan of every row. That aggregate had NO time bound, so on a big shipment
        # (e.g. 196k rows) it blew past the DB's 9s statement_timeout, was cancelled, and left the
        # connection in an ABORTED transaction — so the actual CSV stream failed with "current
        # transaction is aborted" and returned a 2-line error file (operator: real shipment, CSV empty).
        # A cache miss just leaves start=0 (length column = raw encoder); the rows still export fine.
        try:
            from services.shipment_stats import get_shipment_span
            _sp = get_shipment_span(shipment)
            if _sp:
                enc_min_global = int(_sp.get("enc_min") or 0)
                enc_max_global = int(_sp.get("enc_max") or 0)
        except Exception as _me:
            logger.debug(f"export_csv enc-span cache read failed: {_me}")
    else:
        # All-shipments: the window (idx on time) bounds the scan, so MIN/MAX is cheap.
        try:
            with conn.cursor() as _mcur:
                _mcur.execute(
                    f"""SELECT MIN(encoder_value), MAX(encoder_value)
                        FROM inference_results
                        WHERE {where_sql}""",
                    where_params,
                )
                _row = _mcur.fetchone()
                if _row:
                    enc_min_global = int(_row[0]) if _row[0] is not None else 0
                    enc_max_global = int(_row[1]) if _row[1] is not None else 0
            conn.commit()
        except Exception as _me:
            logger.debug(f"export_csv min/max encoder probe failed: {_me}")
            try:
                conn.rollback()   # clear any aborted txn so the export below can run
            except Exception:
                pass

    _dir_tag = ""  # unwind is now a CSV column, not a filename variant

    # ------------------------------------------------------------------
    # Single-CSV path (operator picked a specific shipment)
    # ------------------------------------------------------------------
    if shipment:
        def gen_single():
            try:
                cur = conn.cursor(name="export_csv_single_cursor")
                cur.itersize = 500
                cur.execute(
                    f"""SELECT time, shipment, encoder_value, image_path,
                               inference_time_ms, model_used, detections
                        FROM inference_results
                        WHERE {where_sql}
                        ORDER BY time ASC""",
                    where_params,
                )
                buf = _io.StringIO()
                writer = _csv.writer(buf)
                writer.writerow(_csv_columns_with_unit(_unit))
                yield buf.getvalue(); buf.seek(0); buf.truncate(0)
                for t, ship, enc, img, infms, model, dets in cur:
                    if not dets:
                        continue
                    for d in dets:
                        if not isinstance(d, dict):
                            continue
                        row = _detection_row(t, ship, enc, img, infms, model, d,
                                             sev_map, enc_min_global, enc_max_global,
                                             min_conf_f, img_prefix, upm=_upm)
                        if row is None:
                            continue
                        writer.writerow(row)
                        yield buf.getvalue(); buf.seek(0); buf.truncate(0)
                cur.close()
            except Exception as e:
                logger.warning(f"export_csv single stream failed: {e}")
                yield f"# ERROR: {e}\n"
            finally:
                try: release_db_connection(conn)
                except Exception: pass

        safe_ship = _re.sub(r"[^A-Za-z0-9_-]+", "_", shipment)
        fname = f"detections_{safe_ship}_{window}{_dir_tag}.csv"
        return StreamingResponse(gen_single(), media_type="text/csv; charset=utf-8",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    # ------------------------------------------------------------------
    # ZIP path (all shipments in the window, one CSV per shipment inside)
    # ------------------------------------------------------------------
    # Group detections by shipment in memory. The rows are grouped as we scan,
    # so peak memory holds only the built rows (not the raw jsonb detections).
    # For a 24 h window on the biggest current site that's roughly a few MB.
    try:
        from collections import defaultdict
        import zipfile as _zip

        per_shipment_rows = defaultdict(list)
        cur = conn.cursor(name="export_csv_zip_cursor")
        cur.itersize = 500
        cur.execute(
            f"""SELECT time, shipment, encoder_value, image_path,
                       inference_time_ms, model_used, detections
                FROM inference_results
                WHERE time > NOW() - INTERVAL %s
                ORDER BY time ASC""",
            [interval],
        )
        for t, ship, enc, img, infms, model, dets in cur:
            if not dets or not ship:
                continue
            for d in dets:
                if not isinstance(d, dict):
                    continue
                row = _detection_row(t, ship, enc, img, infms, model, d,
                                     sev_map, enc_min_global, enc_max_global,
                                     min_conf_f, img_prefix, upm=_upm)
                if row is None:
                    continue
                per_shipment_rows[ship].append(row)
        cur.close()

        # Build the ZIP in memory. One CSV entry per shipment id, sorted so the
        # ZIP has a stable file order (helps diffs and human scanning).
        buf = _io.BytesIO()
        with _zip.ZipFile(buf, "w", compression=_zip.ZIP_DEFLATED) as zf:
            if not per_shipment_rows:
                # Empty result — still return a valid ZIP with a README so the
                # operator sees something meaningful instead of a 0-byte
                # download that looks like a bug.
                zf.writestr(
                    "README.txt",
                    f"No detections in window={window}.\n"
                    f"Filter used: min_conf={min_conf_f}\n",
                )
            for ship_id in sorted(per_shipment_rows.keys()):
                sbuf = _io.StringIO()
                w = _csv.writer(sbuf)
                w.writerow(_csv_columns_with_unit(_unit))
                for r in per_shipment_rows[ship_id]:
                    w.writerow(r)
                safe_ship = _re.sub(r"[^A-Za-z0-9_-]+", "_", str(ship_id))
                zf.writestr(f"detections_{safe_ship}_{window}{_dir_tag}.csv",
                            sbuf.getvalue())
        buf.seek(0)
        zip_bytes = buf.getvalue()
        fname = f"detections_all_{window}{_dir_tag}.zip"
        return Response(content=zip_bytes,
                        media_type="application/zip",
                        headers={
                            "Content-Disposition": f'attachment; filename="{fname}"',
                            "Content-Length": str(len(zip_bytes)),
                        })
    except Exception as e:
        logger.warning(f"export_csv zip build failed: {e}")
        return Response(content=f"ERROR: {e}\n", status_code=500,
                        media_type="text/plain")
    finally:
        try: release_db_connection(conn)
        except Exception: pass


# =============================================================================
# 4.0.348 — bulk image download for a shipment / window as size-capped ZIP PARTS.
# A shipment can be 196k+ frames (~tens of GB); a single archive is un-resumable
# over a flaky link and pointless to build on disk. So:
#   * /api/export_images/plan            → {count, est size, images_per_part, parts}
#   * /api/export_images?offset=&limit=  → STREAMED ZIP of just that slice
# The ZIP is generated on the fly (ZIP_STORED, one image buffered at a time) and
# flushed straight to the socket — nothing is written to server disk, peak RAM is
# ~one image, and there is no temp archive to clean up. `annotated=true` draws the
# stored boxes on each frame (reusing the render_detected renderer + its Redis
# cache); default is the original raw JPEGs (copied byte-for-byte, no re-encode).
# =============================================================================
_IMG_PART_TARGET_BYTES = 1024 * 1024 * 1024   # ~1 GiB per part
_IMG_WINDOWS = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}


def _img_where(shipment, window):
    """(where_sql, params) — a picked shipment filters by id ALONE (no time
    bound, same rule as the CSV button); otherwise by the time window."""
    if shipment:
        return "shipment = %s", [shipment]
    return "time > NOW() - INTERVAL %s", [_IMG_WINDOWS.get(window, "24 hours")]


def _img_conf_clause(min_conf_f):
    """min_conf>0 → only frames with at least one detection at/above threshold."""
    if min_conf_f and min_conf_f > 0:
        return (" AND EXISTS (SELECT 1 FROM jsonb_array_elements(detections) e "
                "WHERE (e->>'confidence') ~ '^[0-9]*\\.?[0-9]+$' "
                "AND (e->>'confidence')::float >= %s)")
    return ""


def _img_select(cols, where_sql, conf_clause):
    # Stable order so a given (offset,limit) part always contains the same frames.
    return (f"SELECT {cols} FROM inference_results "
            f"WHERE {where_sql} AND image_path IS NOT NULL AND image_path <> ''"
            f"{conf_clause} ORDER BY time ASC, image_path ASC")


def _img_resolve(img):
    """DB image_path → (abspath, arcname under raw_images/) or (None, None).
    arcname keeps the `<shipment>/<frame>.jpg` shape so the ZIP unzips into a
    folder per shipment. Path-traversal guarded exactly like serve_raw_image."""
    s = str(img)
    rel = (s.split("raw_images/")[-1] if "raw_images/" in s else s).lstrip("/\\")
    try:
        rp = (pathlib.Path("raw_images") / rel).resolve()
        if not str(rp).startswith(str(_RAW_IMAGES_ROOT)) or not rp.exists():
            return None, None
    except Exception:
        return None, None
    return str(rp), rel


def _img_annotate_bytes(abspath, dets, r):
    """Raw frame at abspath with its stored boxes drawn → JPEG bytes. Shares the
    render_detected Redis cache (key = path|<show="">|mtime) so a frame the
    operator already hovered is a cache hit, and this export warms it for later."""
    import os as _os, hashlib as _hl
    try:
        mtime_ns = _os.stat(abspath).st_mtime_ns
    except Exception:
        mtime_ns = 0
    cache_key = "render:" + _hl.sha1(f"{abspath}||{mtime_ns}".encode("utf-8")).hexdigest()
    if r is not None:
        try:
            cached = r.get(cache_key)
            if cached:
                return cached
        except Exception:
            pass
    img = cv2.imread(abspath)
    if img is None:
        raise RuntimeError("cv2 cannot read frame")
    try:
        from services.render import draw_detection_on as _draw
        kv_y = 4
        for det in dets:
            if isinstance(det, dict):
                try:
                    kv_y = _draw(img, det, kv_y=kv_y, bbox_thickness=3)
                except Exception:
                    pass
    except Exception as _e:
        logger.debug(f"export_images annotate draw failed: {_e}")
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    data = bytes(enc)
    if r is not None:
        try:
            r.setex(cache_key, 3600, data)
        except Exception:
            pass
    return data


def _stream_shipment_csv_entry(zf, sink, shipment, window, min_conf_f, img_prefix):
    """Write the shipment's detections CSV as ONE zip entry, STREAMED (deflated,
    row-batched) so a multi-million-row shipment neither blocks first-byte nor
    builds the whole CSV in memory. Same columns/logic as export_csv. Yields
    drained ZIP bytes as it goes. `zf` must be a ZipFile over the non-seekable
    `sink`; nothing else may write to `zf` until this generator is exhausted."""
    import csv as _csv
    import io as _io
    import re as _re2
    import zipfile as _zip2
    from services.db import get_db_connection, release_db_connection
    try:
        from config import load_service_config as _lsc
        _svc = _lsc() or {}
        _audio = _svc.get("audio_settings", {}) or {}
        sev_map = {k: int(v.get("severity", 0) or 0) for k, v in _audio.items() if isinstance(v, dict)}
    except Exception:
        sev_map = {}
    _upm, _unit = _encoder_calibration()
    enc_min = enc_max = 0
    try:
        from services.shipment_stats import get_shipment_span
        _sp = get_shipment_span(shipment)
        if _sp:
            enc_min = int(_sp.get("enc_min") or 0)
            enc_max = int(_sp.get("enc_max") or 0)
    except Exception:
        pass
    conn = get_db_connection()
    if conn is None:
        return
    safe = _re2.sub(r"[^A-Za-z0-9_-]+", "_", str(shipment))
    zi = _zip2.ZipInfo(f"detections_{safe}_{window}.csv")
    zi.compress_type = _zip2.ZIP_DEFLATED   # text compresses ~10x; images stay STORED
    cf = zf.open(zi, "w")
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(_csv_columns_with_unit(_unit))
    try:
        cur = conn.cursor(name="export_images_csv_cursor")
        cur.itersize = 1000
        cur.execute(
            """SELECT time, shipment, encoder_value, image_path,
                      inference_time_ms, model_used, detections
                 FROM inference_results
                WHERE shipment = %s
                ORDER BY time ASC""",
            [shipment],
        )
        n = 0
        for t, ship, enc, img, infms, model, dets in cur:
            if dets:
                for d in dets:
                    if isinstance(d, dict):
                        row = _detection_row(t, ship, enc, img, infms, model, d, sev_map,
                                             enc_min, enc_max, min_conf_f, img_prefix, upm=_upm)
                        if row is not None:
                            w.writerow(row)
            n += 1
            if n % 1000 == 0:
                cf.write(buf.getvalue().encode("utf-8"))
                buf.seek(0); buf.truncate(0)
                chunk = sink.drain()
                if chunk:
                    yield chunk
        cf.write(buf.getvalue().encode("utf-8"))
        cur.close()
    except Exception as e:
        logger.warning(f"export_images CSV stream failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        try:
            cf.close()
        except Exception:
            pass
        try:
            release_db_connection(conn)
        except Exception:
            pass
    chunk = sink.drain()
    if chunk:
        yield chunk


@router.get("/api/export_images/plan")
def export_images_plan(request: Request, window: str = "24h", shipment: str = "",
                       min_conf: float = 0.0):
    """Count the images for a selection and propose ~1 GiB ZIP parts. Per-part
    size is ESTIMATED from a small sample of real files (stat-ing every file
    up-front would be slow), so parts are a fixed images-per-part derived from
    that estimate — actual part size varies a little."""
    import os as _os
    min_conf_f = float(min_conf or 0.0)
    where_sql, where_params = _img_where(shipment, window)
    conf_clause = _img_conf_clause(min_conf_f)
    params = list(where_params) + ([min_conf_f] if min_conf_f > 0 else [])

    from services.db import get_db_connection, release_db_connection
    conn = get_db_connection()
    if conn is None:
        return JSONResponse({"error": "database unavailable"}, status_code=503)
    count = 0
    sample = []
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 30000")
            cur.execute(f"SELECT COUNT(*) FROM inference_results WHERE {where_sql} "
                        f"AND image_path IS NOT NULL AND image_path <> ''{conf_clause}", params)
            count = int(cur.fetchone()[0] or 0)
            cur.execute(_img_select("image_path", where_sql, conf_clause) + " LIMIT 200", params)
            sample = [row[0] for row in cur.fetchall() if row and row[0]]
        conn.commit()
    except Exception as e:
        logger.warning(f"export_images_plan failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            release_db_connection(conn)
        except Exception:
            pass

    sizes = []
    for p in sample:
        ap, _ = _img_resolve(p)
        if ap:
            try:
                sizes.append(_os.path.getsize(ap))
            except Exception:
                pass
    avg_bytes = int(sum(sizes) / len(sizes)) if sizes else 150 * 1024   # 150 KB fallback
    # Every frame ships TWICE (raw + annotated), so budget ~2× per frame when
    # sizing the ~1 GiB parts and estimating the total.
    per_frame_bytes = max(1, avg_bytes * 2)
    per_part = max(1, round(_IMG_PART_TARGET_BYTES / per_frame_bytes))
    parts = (count + per_part - 1) // per_part if count else 0
    return JSONResponse({
        "shipment": shipment, "window": window, "min_conf": min_conf_f,
        "count": count, "avg_bytes": avg_bytes, "est_total_bytes": per_frame_bytes * count,
        "images_per_part": per_part, "parts": parts,
        "target_part_bytes": _IMG_PART_TARGET_BYTES,
    })


@router.get("/api/export_images")
def export_images(request: Request, window: str = "24h", shipment: str = "",
                  min_conf: float = 0.0, offset: int = 0, limit: int = 0,
                  part: int = 0, parts: int = 0):
    """Stream a ZIP of frame images for a selection (or one PART of it).

    Selection mirrors the CSV button (shipment OR window + min_conf). `offset`
    /`limit` carve out one ~1 GiB part (see /plan) so each is an independent,
    resumable download; offset=limit=0 streams the whole selection.

    Every frame is written TWICE: the original JPEG under `raw/<shipment>/…` and
    the same frame with its stored boxes drawn under `annotated/<shipment>/…`
    (the annotated copy reuses the render_detected renderer + its Redis cache).

    Streamed on the fly, one image buffered at a time: no temp file on disk, flat
    memory. part/parts only affect the download filename."""
    min_conf_f = float(min_conf or 0.0)
    where_sql, where_params = _img_where(shipment, window)
    conf_clause = _img_conf_clause(min_conf_f)
    cols = "image_path, detections"

    # Embed the shipment's detections CSV as the FIRST entry of the FIRST part
    # (operator: "add the csv of that shipment to the first zip"). Only for a
    # picked shipment (the button is shipment-only) and only on part 1 / a single
    # zip, so it isn't duplicated into every part.
    include_csv = bool(shipment) and (int(part or 0) <= 1)
    try:
        img_prefix = str(request.base_url).rstrip("/") + "/api/raw_image/"
    except Exception:
        img_prefix = "/api/raw_image/"

    from services.db import get_db_connection, release_db_connection
    conn = get_db_connection()
    if conn is None:
        return Response(content="ERROR: database unavailable\n", status_code=503,
                        media_type="text/plain")
    rows = []
    try:
        sql = _img_select(cols, where_sql, conf_clause)
        params = list(where_params) + ([min_conf_f] if min_conf_f > 0 else [])
        if limit and limit > 0:
            sql += " LIMIT %s OFFSET %s"
            params += [int(limit), int(offset or 0)]
        with conn.cursor() as cur:
            # Bulk export, not a hot path — a large OFFSET on the last parts of a
            # huge shipment can take a few seconds, so allow 60s (the slice is
            # fetched here up-front, then the DB connection is released before the
            # slow disk/network-bound ZIP stream begins).
            cur.execute("SET LOCAL statement_timeout = 60000")
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.commit()
    except Exception as e:
        logger.warning(f"export_images path query failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return Response(content=f"ERROR: {e}\n", status_code=500, media_type="text/plain")
    finally:
        try:
            release_db_connection(conn)
        except Exception:
            pass

    import re as _re
    import zipfile as _zip

    # Non-seekable sink → zipfile switches to streaming mode (data descriptors),
    # so we never need to seek back to patch headers. drain() hands each file's
    # bytes to the generator and clears the buffer → peak RAM ≈ one image.
    class _Sink:
        def __init__(self):
            self._b = bytearray()

        def write(self, b):
            self._b.extend(b)
            return len(b)

        def flush(self):
            pass

        def drain(self):
            out = bytes(self._b)
            del self._b[:]
            return out

    def gen():
        sink = _Sink()
        zf = _zip.ZipFile(sink, "w", compression=_zip.ZIP_STORED, allowZip64=True)
        written = missing = 0
        r = None
        try:
            r = Redis("redis", 6379, db=cfg_module.REDIS_DB)
        except Exception:
            r = None
        try:
            if include_csv:
                try:
                    for _c in _stream_shipment_csv_entry(zf, sink, shipment, window,
                                                         min_conf_f, img_prefix):
                        if _c:
                            yield _c
                except Exception as _ce:
                    logger.warning(f"export_images: embed CSV failed: {_ce}")
            for rec in rows:
                img = rec[0]
                ap, arc = _img_resolve(img)
                if not ap:
                    missing += 1
                    continue
                dets = rec[1] if len(rec) > 1 and isinstance(rec[1], list) else []
                # A frame earns an annotated copy only if it has a REAL, drawable
                # detection — not just metadata entries like `_color`/`_absolute`
                # (which draw nothing, so the annotated copy would merely duplicate
                # the raw). Operator: "why generate a blank annotated image if there
                # is no annotation." raw/ still gets every frame.
                has_box = any(
                    isinstance(d, dict) and str(d.get("name", "")).strip()
                    and not str(d.get("name", "")).startswith("_")
                    for d in dets
                )
                wrote_any = False
                # Original frame, copied byte-for-byte, under raw/<shipment>/…
                try:
                    zf.write(ap, "raw/" + arc)
                    written += 1
                    wrote_any = True
                except Exception as we:
                    logger.debug(f"export_images raw skip {img}: {we}")
                # Same frame with its stored boxes drawn, under annotated/<shipment>/…
                if has_box:
                    try:
                        zf.writestr("annotated/" + arc, _img_annotate_bytes(ap, dets, r))
                        written += 1
                        wrote_any = True
                    except Exception as we:
                        logger.debug(f"export_images annotated skip {img}: {we}")
                if not wrote_any:
                    missing += 1
                    continue
                chunk = sink.drain()
                if chunk:
                    yield chunk
            if written == 0:
                zf.writestr("README.txt",
                            f"No image files were found on disk for this selection.\n"
                            f"(matched {len(rows)} rows, {missing} files missing)\n")
            zf.close()
            yield sink.drain()
            logger.info(f"export_images: ship={shipment or '(win '+window+')'} "
                        f"annotated={annotated} offset={offset} limit={limit} "
                        f"wrote={written} missing={missing}")
        except Exception as e:
            logger.warning(f"export_images stream aborted: {e}")
            try:
                zf.close()
            except Exception:
                pass
            tail = sink.drain()
            if tail:
                yield tail

    tag = _re.sub(r"[^A-Za-z0-9_-]+", "_", shipment) if shipment else f"all_{window}"
    if parts and part:
        fname = f"images_{tag}_p{int(part):02d}of{int(parts):02d}.zip"
    else:
        fname = f"images_{tag}.zip"
    return StreamingResponse(gen(), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.get("/api/raw_image/{path:path}")
def serve_raw_image(path: str):
    """Serve a raw image file from the raw_images directory."""
    safe_path = pathlib.Path("raw_images") / path
    try:
        resolved = safe_path.resolve()
        if not str(resolved).startswith(str(_RAW_IMAGES_ROOT)):
            raise HTTPException(status_code=403, detail="Access denied")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(resolved), media_type="image/jpeg")


# =============================================================================
# 4.0.0 — render-on-demand annotated frames.
#
# Replaces the `<frame>_DETECTED.jpg` files that the detection pipeline used to
# write to disk for every detected frame. Now:
#   - storage = raw JPG only (one file per frame instead of two)
#   - viewing = this endpoint reads the raw + the `detections` JSON from the
#               inference_results row + draws bboxes with OpenCV on-the-fly
#   - download = same endpoint with ?download=1 (Content-Disposition: attachment)
#
# Redis cache: rendered bytes are cached for 1 hour keyed by (path, show, mtime).
# First view after restart pays ~20 ms of OpenCV work; every subsequent hit is
# a Redis lookup (<1 ms). The cache invalidates automatically when the raw file
# changes (mtime in key) or when a class is hidden/shown (show in key).
# =============================================================================
@router.get("/api/render_detected/{path:path}")
def render_detected(path: str, show: str = "", download: int = 0):
    """Render the raw frame at `path` with its stored bounding boxes drawn on top.

    Query:
      show     — comma-separated allowlist of class names; empty = use the
                 current Show-toggle state from audio_settings (default).
      download — when 1, sets Content-Disposition: attachment so the browser
                 downloads instead of inlining.
    """
    import hashlib, io
    safe_path = pathlib.Path("raw_images") / path
    try:
        resolved = safe_path.resolve()
        if not str(resolved).startswith(str(_RAW_IMAGES_ROOT)):
            raise HTTPException(status_code=403, detail="Access denied")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid path")
    if not resolved.exists():
        # Accept the legacy `_DETECTED.jpg` URL form by stripping the suffix so
        # any cached link from the pre-4.0 era still works.
        if str(resolved).endswith("_DETECTED.jpg"):
            resolved = pathlib.Path(str(resolved)[:-len("_DETECTED.jpg")] + ".jpg")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="Raw image not found")

    # Build cache key: hash(path + show + mtime) so any file/visibility change
    # invalidates automatically.
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except Exception:
        mtime_ns = 0
    key_seed = f"{resolved}|{show}|{mtime_ns}".encode("utf-8")
    cache_key = "render:" + hashlib.sha1(key_seed).hexdigest()

    cached = None
    try:
        r = Redis("redis", 6379, db=cfg_module.REDIS_DB)
        cached = r.get(cache_key)
    except Exception:
        r = None
    if cached:
        headers = {}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{resolved.stem}_DETECTED.jpg"'
        return Response(content=cached, media_type="image/jpeg", headers=headers)

    # 4.0.22 — resolve the inference_results row by EXACT image_path first.
    # The previous LIKE '%<stem>%' query was prone to picking the wrong row
    # when two rows shared the same `<timestamp>_<cam>` stem (one under raw
    # shipment-A path + one under shipment-B, one raw row + one pre-4.0.0
    # _DETECTED row, or any other path sharing the substring). That meant
    # the hover preview could show wrong boxes on the right image.
    #
    # New lookup order, each one EXACT:
    #   1. raw_images/<rel>                     (post-4.0.10 rows)
    #   2. raw_images/<rel-stem>_DETECTED.jpg   (legacy pre-4.0.0 rows)
    #   3. stem LIKE %                          (final fallback for paths
    #      stored without the `raw_images/` prefix or in other legacy shapes)
    detections = []
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is not None:
            try:
                # `path` is the URL parameter (e.g. `no_shipment/.../<stem>.jpg`).
                # Stored image_path includes the `raw_images/` prefix.
                exact_raw = "raw_images/" + path.lstrip("/")
                # The DETECTED-form sibling (legacy rows).
                if exact_raw.endswith(".jpg"):
                    exact_det = exact_raw[:-4] + "_DETECTED.jpg"
                else:
                    exact_det = exact_raw + "_DETECTED.jpg"
                stem = resolved.stem
                # v4.0.112 — parse shipment + hour + capture-timestamp from the
                # URL path so each SQL below can filter by shipment + time
                # bounds. Previously the queries were `WHERE image_path = %s
                # ORDER BY time DESC LIMIT 1` with NO time predicate; Postgres
                # scanned back from newest through millions of rows to find
                # the match, timing out on older images (2026-07-12 hovers
                # on khoy). Path format:
                #   <shipment>/<YYYY-MM-DD_HH>/<YYYY-MM-DD-HH-MM-SS-uuuuuu>_...jpg
                # Both segments are optional guards — if parsing fails we just
                # fall through to the un-bounded query (worst case = previous
                # behaviour + statement_timeout still capped).
                import re as _re
                _parts = [p for p in path.strip("/").split("/") if p]
                _ship_guess = _parts[0] if _parts else ""
                _t_from = _t_to = None
                # Prefer the exact capture timestamp encoded in the filename.
                # v4.0.113 — MUST tag the parsed datetime as UTC. Frame filenames
                # are written with UTC components (see services/watcher.py
                # frame-write path), so the raw digits are UTC. If we build a
                # naive `datetime` psycopg2 forwards it as server-local time,
                # and PG then converts local → UTC using the server's zone
                # (Iran, UTC+3:30 on khoy). Net effect: the BETWEEN window
                # shifts by 3.5 h and matches ZERO rows — so v4.0.112
                # returned an image with no boxes drawn even when detections
                # existed. Constructing with `tzinfo=UTC` fixes the shift.
                _fn = _parts[-1] if _parts else ""
                _tm = _re.match(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", _fn)
                if _tm:
                    from datetime import datetime as _dt, timedelta as _td2, timezone as _tz2
                    _cap = _dt(int(_tm.group(1)), int(_tm.group(2)), int(_tm.group(3)),
                                int(_tm.group(4)), int(_tm.group(5)), int(_tm.group(6)),
                                tzinfo=_tz2.utc)
                    # Widen window from ±2 s to ±5 s so a small clock drift
                    # between capture and DB insert (~ms in normal flow, but
                    # could be seconds if the DB writer queue was backed up)
                    # still resolves to the right row.
                    _t_from = _cap - _td2(seconds=5)
                    _t_to   = _cap + _td2(seconds=5)
                # SET LOCAL statement_timeout so a slow-path fallback can't
                # camp on a pool connection for the whole 9 s pool default.
                cur = conn.cursor()
                cur.execute("SET LOCAL statement_timeout = 3000")

                _bounded_where = ""
                _bounded_args_prefix = []
                if _ship_guess and _t_from and _t_to:
                    _bounded_where = " AND shipment = %s AND time BETWEEN %s AND %s"
                    _bounded_args_prefix = [_ship_guess, _t_from, _t_to]

                # 1: exact raw (with shipment + time bounds if parseable)
                cur.execute(
                    "SELECT detections FROM inference_results "
                    "WHERE image_path = %s" + _bounded_where +
                    " ORDER BY time DESC LIMIT 1",
                    (exact_raw, *_bounded_args_prefix),
                )
                row = cur.fetchone()
                # 2: exact DETECTED sibling (same bounds)
                if not row:
                    cur.execute(
                        "SELECT detections FROM inference_results "
                        "WHERE image_path = %s" + _bounded_where +
                        " ORDER BY time DESC LIMIT 1",
                        (exact_det, *_bounded_args_prefix),
                    )
                    row = cur.fetchone()
                # 3: legacy stem LIKE (kept for any out-of-pattern stored paths,
                # but BOTH endpoints in this row pair are now warned-logged
                # so we can find and clean them up). Still bounded by shipment +
                # time when parseable so LIKE runs over a tiny slice, not the
                # whole hypertable.
                if not row:
                    cur.execute(
                        "SELECT detections FROM inference_results "
                        "WHERE image_path LIKE %s" + _bounded_where +
                        " ORDER BY time DESC LIMIT 1",
                        ("%" + stem + "%", *_bounded_args_prefix),
                    )
                    row = cur.fetchone()
                    if row:
                        logger.warning(
                            f"render_detected: matched via legacy LIKE fallback "
                            f"for stem={stem!r} (request path={path!r}). Row "
                            f"image_path does NOT equal either exact form — check "
                            f"for path-format drift in detection.py."
                        )
                if row and row[0]:
                    d = row[0]
                    if isinstance(d, list):
                        detections = d
                cur.close()
            finally:
                try: release_db_connection(conn)
                except Exception: pass
    except Exception as e:
        logger.warning(f"render_detected: detection lookup failed: {e}")

    # Optional class-allowlist filter (`show=cls1,cls2`). When unset, the
    # draw_filters module decides per-class visibility from audio_settings.
    if show:
        allow = {c.strip() for c in show.split(",") if c.strip()}
        detections = [d for d in detections if str(d.get("name", "")) in allow]

    # Draw boxes via the existing per-detection renderer.
    try:
        img = cv2.imread(str(resolved))
        if img is None:
            raise HTTPException(status_code=500, detail="OpenCV cannot read raw frame")
        try:
            from services.render import draw_detection_on as _draw
            kv_y = 4
            for det in detections:
                if isinstance(det, dict):
                    try:
                        kv_y = _draw(img, det, kv_y=kv_y, bbox_thickness=3)
                    except Exception as _e:
                        logger.debug(f"render_detected: skipped one det: {_e}")
        except Exception as _e:
            logger.warning(f"render_detected: draw loop failed: {_e}")

        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise HTTPException(status_code=500, detail="JPEG encode failed")
        rendered = bytes(encoded)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"render_detected: render failed: {e}")
        raise HTTPException(status_code=500, detail=f"render failed: {e}")

    # Populate cache (best-effort; never block the response on Redis failure).
    if r is not None:
        try:
            r.setex(cache_key, 3600, rendered)
        except Exception as _e:
            logger.debug(f"render cache write failed: {_e}")

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{resolved.stem}_DETECTED.jpg"'
    return Response(content=rendered, media_type="image/jpeg", headers=headers)


@router.get("/api/timeline_frame")
def timeline_frame(request: Request, cam: int = 1, col: int = 0, page: int = 0, path: str = ""):
    """Serve full-res raw image with bboxes drawn from timeline detection data.

    If `path` is provided, look up the exact frame by d_path (stable across
    timeline updates). Otherwise fall back to column-index lookup.
    """
    try:
        tl_config = getattr(request.app.state, 'timeline_config', {})
        frames_per_page = tl_config.get('num_rows', 10)
        image_rotation = tl_config.get('image_rotation', 0)
        obj_filters = tl_config.get('object_filters', {})
        show_bbox = tl_config.get('show_bounding_boxes', True)

        redis_client = Redis("redis", 6379, db=cfg_module.REDIS_DB)
        redis_key = f"{TIMELINE_REDIS_PREFIX}{cam}"
        frames_raw = redis_client.lrange(redis_key, 0, -1)
        if not frames_raw:
            raise HTTPException(status_code=404, detail="No frames")

        import pickle as _pkl
        all_frames = []
        for fd in frames_raw:
            try:
                ts, jpeg_bytes, detections, meta = _pkl.loads(fd)
                all_frames.append((ts, jpeg_bytes, detections, meta))
            except Exception:
                pass
        all_frames.sort(key=lambda x: x[0])

        # Stable lookup by d_path (preferred — immune to timeline shifts)
        matched = None
        if path:
            for frame in all_frames:
                m = frame[3]
                if m and m.get('d_path') == path:
                    matched = frame
                    break

        # Fallback to column-index lookup
        if not matched:
            total = len(all_frames)
            end_index = total - page * frames_per_page
            start_index = max(0, end_index - frames_per_page)
            page_slice = all_frames[start_index:end_index] if end_index > 0 else []
            if col < 0 or col >= len(page_slice):
                raise HTTPException(status_code=404, detail="Column out of range")
            matched = page_slice[col]

        ts, jpeg_bytes, detections, meta = matched
        d_path = meta.get('d_path') if meta else None

        # Try to load full-res image from disk
        image = None
        is_full_res = False
        if d_path:
            raw_path = pathlib.Path("raw_images") / f"{d_path}.jpg"
            if raw_path.exists():
                image = cv2.imread(str(raw_path))
                if image is not None:
                    is_full_res = True

        # Fallback to timeline thumbnail if disk file missing
        if image is None:
            image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Draw bboxes from timeline detection data — single shared helper
        if show_bbox and detections:
            ih, iw = image.shape[:2]
            orig_h = meta.get('orig_h', ih) if meta else ih
            orig_w = meta.get('orig_w', iw) if meta else iw
            if is_full_res:
                sx, sy = 1.0, 1.0
            else:
                sx = iw / orig_w if orig_w else 1
                sy = ih / orig_h if orig_h else 1
            kv_y = 4
            for det in detections:
                kv_y = draw_detection_on(
                    image, det, sx=sx, sy=sy, kv_y=kv_y,
                    bbox_thickness=3, obj_filters=obj_filters,
                )

        # Apply rotation
        if image_rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif image_rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif image_rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        _, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving timeline frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/timeline_meta")
def timeline_meta(request: Request, page: int = 0):
    """Get metadata for each column in the timeline composite (used by frontend click handler)."""
    try:
        try:
            tl_config = request.app.state.timeline_config
            frames_per_page = tl_config.get('num_rows', 10)
            image_rotation = tl_config.get('image_rotation', 0)
            procedures = tl_config.get('procedures', [])
        except Exception:
            frames_per_page = 10
            image_rotation = 0
            procedures = []

        redis_client = Redis("redis", 6379, db=cfg_module.REDIS_DB)
        all_keys = redis_client.keys(f"{TIMELINE_REDIS_PREFIX}*")
        if not all_keys:
            return {"type": "timeline_meta", "columns": [], "thumb_width": 0, "thumb_height": 240,
                    "header_height": _HEADER_HEIGHT, "num_cameras": 0, "page": page, "total_pages": 0}

        def extract_cam_id(key):
            k = key.decode() if isinstance(key, bytes) else key
            try:
                return int(k.split(":")[-1])
            except ValueError:
                return 0
        all_keys.sort(key=extract_cam_id)

        camera_frames_raw = {}
        max_total_frames = 0
        for key in all_keys:
            frames_raw = redis_client.lrange(key, 0, -1)
            cam_id = extract_cam_id(key)
            camera_frames_raw[cam_id] = frames_raw if frames_raw else []
            max_total_frames = max(max_total_frames, len(camera_frames_raw[cam_id]))

        total_pages = max(1, (max_total_frames + frames_per_page - 1) // frames_per_page)
        if page < 0 or page >= total_pages:
            page = 0

        sorted_cam_ids = sorted(camera_frames_raw.keys())
        cam_page_slices = {}
        num_columns = 0

        for cam_id in sorted_cam_ids:
            frames_raw = camera_frames_raw[cam_id]
            all_frames = []
            for frame_data in frames_raw:
                ts, jpeg_bytes, detections, meta = _unpack_timeline_entry(frame_data)
                all_frames.append((ts, detections, meta))
            all_frames.sort(key=lambda x: x[0])

            total = len(all_frames)
            end_index = total - (page * frames_per_page)
            start_index = max(0, end_index - frames_per_page)
            ps = all_frames[start_index:end_index] if end_index > 0 else []
            cam_page_slices[cam_id] = ps
            num_columns = max(num_columns, len(ps))

        # Determine thumb dimensions from first available frame
        thumb_width = 0
        thumb_height = 240
        for cam_id in sorted_cam_ids:
            frames_raw = camera_frames_raw[cam_id]
            if frames_raw:
                _, jpeg_bytes, _, _ = _unpack_timeline_entry(frames_raw[0])
                thumb = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if thumb is not None:
                    if image_rotation in (90, 270):
                        thumb_width, thumb_height = thumb.shape[:2]
                    else:
                        thumb_height, thumb_width = thumb.shape[:2]
                    break

        # Build per-column metadata
        columns = []
        for col_idx in range(num_columns):
            col_all_dets = []
            col_d_paths = {}
            col_encoder = None
            col_ts = None
            for cam_id in sorted_cam_ids:
                ps = cam_page_slices.get(cam_id, [])
                if col_idx < len(ps):
                    ts, detections, meta = ps[col_idx]
                    if detections:
                        col_all_dets.extend(detections)
                    col_d_paths[str(cam_id)] = meta.get('d_path') if meta else None
                    if meta and meta.get('encoder') is not None:
                        col_encoder = meta['encoder']
                    if col_ts is None:
                        col_ts = ts

            should_eject = False
            eject_reasons = []
            if procedures and col_all_dets:
                should_eject, eject_reasons = evaluate_eject_from_detections(col_all_dets, procedures)

            columns.append({
                "index": col_idx,
                "ts": col_ts,
                "encoder": col_encoder,
                "should_eject": should_eject,
                "eject_reasons": eject_reasons,
                "d_paths": col_d_paths,
            })

        return {
            "type": "timeline_meta",
            "columns": columns,
            "thumb_width": thumb_width,
            "thumb_height": thumb_height,
            "header_height": _HEADER_HEIGHT,
            "num_cameras": len(sorted_cam_ids),
            "cam_ids": sorted_cam_ids,
            "page": page,
            "total_pages": total_pages,
        }

    except Exception as e:
        logger.error(f"Error getting timeline metadata: {e}")
        return {"type": "timeline_meta", "columns": [], "error": str(e)}


@router.get("/timeline_slideshow")
def timeline_slideshow():
    """Timeline slideshow page with pan/zoom controls (inspired by FabriQC slideshow service)."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Camera Timeline Slideshow</title>
  <meta http-equiv="refresh" content="5">
  <style>
    body {
      margin: 0;
      background: #000;
      color: #fff;
      width: 100%;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    #app {
      position: relative;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    #slide-container {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: auto;
      background: #111;
    }
    #slide {
      max-width: 100%;
      max-height: 100%;
      user-select: none;
      transform-origin: center center;
      display: block;
    }
    #footer {
      height: 50px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.8);
      gap: 10px;
      padding: 0 20px;
    }
    button {
      font-size: 1.2em;
      font-weight: bold;
      padding: 6px 12px;
      cursor: pointer;
      background: #222;
      color: #fff;
      border: 1px solid #555;
      border-radius: 4px;
    }
    button:hover { background: #444; }
    #info {
      color: #aaa;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div id="app">
    <div id="slide-container">
      <img id="slide" src="/timeline_image" alt="Camera Timeline" draggable="false" />
    </div>
    <div id="footer">
      <button id="zoom-in" onclick="zoomIn()">+</button>
      <button id="reset-zoom" onclick="resetZoom()">Reset</button>
      <button id="zoom-out" onclick="zoomOut()">\u2013</button>
      <button id="fullscreen" onclick="toggleFullscreen()">\u26F6 Fullscreen</button>
      <span id="info">Auto-refresh every 5 seconds | Columns = Cameras | Rows = Time (newest at top)</span>
    </div>
  </div>

  <!-- Panzoom library (embedded for offline use) -->
  <script>((t,e)=>{"object"==typeof exports&&"undefined"!=typeof module?module.exports=e():"function"==typeof define&&define.amd?define(e):(t="undefined"!=typeof globalThis?globalThis:t||self).Panzoom=e()})(this,function(){var a,X=function(){return(X=Object.assign||function(t){for(var e,n=1,o=arguments.length;n<o;n++)for(var r in e=arguments[n])Object.prototype.hasOwnProperty.call(e,r)&&(t[r]=e[r]);return t}).apply(this,arguments)},i=("undefined"!=typeof window&&(window.NodeList&&!NodeList.prototype.forEach&&(NodeList.prototype.forEach=Array.prototype.forEach),"function"!=typeof window.CustomEvent)&&(window.CustomEvent=function(t,e){e=e||{bubbles:!1,cancelable:!1,detail:null};var n=document.createEvent("CustomEvent");return n.initCustomEvent(t,e.bubbles,e.cancelable,e.detail),n}),"undefined"!=typeof document&&!!document.documentMode);var c=["webkit","moz","ms"],l={};function Y(t){if(l[t])return l[t];var e=a=a||document.createElement("div").style;if(t in e)return l[t]=t;for(var n=t[0].toUpperCase()+t.slice(1),o=c.length;o--;){var r="".concat(c[o]).concat(n);if(r in e)return l[t]=r}}function o(t,e){return parseFloat(e[Y(t)])||0}function s(t,e,n){void 0===n&&(n=window.getComputedStyle(t));t="border"===e?"Width":"";return{left:o("".concat(e,"Left").concat(t),n),right:o("".concat(e,"Right").concat(t),n),top:o("".concat(e,"Top").concat(t),n),bottom:o("".concat(e,"Bottom").concat(t),n)}}function C(t,e,n){t.style[Y(e)]=n}function N(t){var e=t.parentNode,n=window.getComputedStyle(t),o=window.getComputedStyle(e),r=t.getBoundingClientRect(),a=e.getBoundingClientRect();return{elem:{style:n,width:r.width,height:r.height,top:r.top,bottom:r.bottom,left:r.left,right:r.right,margin:s(t,"margin",n),border:s(t,"border",n)},parent:{style:o,width:a.width,height:a.height,top:a.top,bottom:a.bottom,left:a.left,right:a.right,padding:s(e,"padding",o),border:s(e,"border",o)}}}var T={down:"mousedown",move:"mousemove",up:"mouseup mouseleave"};function L(t,e,n,o){T[t].split(" ").forEach(function(t){e.addEventListener(t,n,o)})}function V(t,e,n){T[t].split(" ").forEach(function(t){e.removeEventListener(t,n)})}function G(t,e){for(var n=t.length;n--;)if(t[n].pointerId===e.pointerId)return n;return-1}function I(t,e){if(e.touches)for(var n=0,o=0,r=e.touches;o<r.length;o++){var a=r[o];a.pointerId=n++,I(t,a)}else-1<(n=G(t,e))&&t.splice(n,1),t.push(e)}function R(t){for(var e,n=(t=t.slice(0)).pop();e=t.pop();)n={clientX:(e.clientX-n.clientX)/2+n.clientX,clientY:(e.clientY-n.clientY)/2+n.clientY};return n}function W(t){var e;return t.length<2?0:(e=t[0],t=t[1],Math.sqrt(Math.pow(Math.abs(t.clientX-e.clientX),2)+Math.pow(Math.abs(t.clientY-e.clientY),2)))}"undefined"!=typeof window&&("function"==typeof window.PointerEvent?T={down:"pointerdown",move:"pointermove",up:"pointerup pointerleave pointercancel"}:"function"==typeof window.TouchEvent&&(T={down:"touchstart",move:"touchmove",up:"touchend touchcancel"}));var Z=/^http:[\w\.\/]+svg$/;var q={animate:!1,canvas:!1,cursor:"move",disablePan:!1,disableZoom:!1,disableXAxis:!1,disableYAxis:!1,duration:200,easing:"ease-in-out",exclude:[],excludeClass:"panzoom-exclude",handleStartEvent:function(t){t.preventDefault(),t.stopPropagation()},maxScale:40,minScale:.125,overflow:"hidden",panOnlyWhenZoomed:!1,pinchAndPan:!1,relative:!1,setTransform:function(t,e,n){var o=e.x,r=e.y,a=e.isSVG;C(t,"transform","scale(".concat(e.scale,") translate(").concat(o,"px, ").concat(r,"px)")),a&&i&&(e=window.getComputedStyle(t).getPropertyValue("transform"),t.setAttribute("transform",e))},startX:0,startY:0,startScale:1,step:.3,touchAction:"none"};function t(u,f){if(!u)throw new Error("Panzoom requires an element as an argument");if(1!==u.nodeType)throw new Error("Panzoom requires an element with a nodeType of 1");if(!(t=>{for(var e=t;e&&e.parentNode;){if(e.parentNode===document)return 1;e=e.parentNode instanceof ShadowRoot?e.parentNode.host:e.parentNode}})(u))throw new Error("Panzoom should be called on elements that have been attached to the DOM");f=X(X({},q),f);t=u;var t,l=Z.test(t.namespaceURI)&&"svg"!==t.nodeName.toLowerCase(),n=u.parentNode;n.style.overflow=f.overflow,n.style.userSelect="none",n.style.touchAction=f.touchAction,(f.canvas?n:u).style.cursor=f.cursor,u.style.userSelect="none",u.style.touchAction=f.touchAction,C(u,"transformOrigin","string"==typeof f.origin?f.origin:l?"0 0":"50% 50%");var r,a,i,c,s,d,m=0,h=0,v=1,p=!1;function g(t,e,n){n.silent||(n=new CustomEvent(t,{detail:e}),u.dispatchEvent(n))}function y(o,r,t){var a={x:m,y:h,scale:v,isSVG:l,originalEvent:t};return requestAnimationFrame(function(){var t,e,n;"boolean"==typeof r.animate&&(r.animate?(t=u,e=r,n=Y("transform"),C(t,"transition","".concat(n," ").concat(e.duration,"ms ").concat(e.easing))):C(u,"transition","none")),r.setTransform(u,a,r),g(o,a,r),g("panzoomchange",a,r)}),a}function w(t,e,n,o){var r,a,i,c,l,s,d,o=X(X({},f),o),p={x:m,y:h,opts:o};return!o.force&&(o.disablePan||o.panOnlyWhenZoomed&&v===o.startScale)||(t=parseFloat(t),e=parseFloat(e),o.disableXAxis||(p.x=(o.relative?m:0)+t),o.disableYAxis||(p.y=(o.relative?h:0)+e),o.contain&&(e=((r=(e=(t=N(u)).elem.width/v)*n)-e)/2,i=((a=(i=t.elem.height/v)*n)-i)/2,"inside"===o.contain?(c=(-t.elem.margin.left-t.parent.padding.left+e)/n,l=(t.parent.width-r-t.parent.padding.left-t.elem.margin.left-t.parent.border.left-t.parent.border.right+e)/n,p.x=Math.max(Math.min(p.x,l),c),s=(-t.elem.margin.top-t.parent.padding.top+i)/n,d=(t.parent.height-a-t.parent.padding.top-t.elem.margin.top-t.parent.border.top-t.parent.border.bottom+i)/n,p.y=Math.max(Math.min(p.y,d),s)):"outside"===o.contain&&(c=(-(r-t.parent.width)-t.parent.padding.left-t.parent.border.left-t.parent.border.right+e)/n,l=(e-t.parent.padding.left)/n,p.x=Math.max(Math.min(p.x,l),c),s=(-(a-t.parent.height)-t.parent.padding.top-t.parent.border.top-t.parent.border.bottom+i)/n,d=(i-t.parent.padding.top)/n,p.y=Math.max(Math.min(p.y,d),s))),o.roundPixels&&(p.x=Math.round(p.x),p.y=Math.round(p.y))),p}function b(t,e){var n,o,r,a,e=X(X({},f),e),i={scale:v,opts:e};return!e.force&&e.disableZoom||(n=f.minScale,o=f.maxScale,e.contain&&(a=(e=N(u)).elem.width/v,r=e.elem.height/v,1<a)&&1<r&&(a=(e.parent.width-e.parent.border.left-e.parent.border.right)/a,e=(e.parent.height-e.parent.border.top-e.parent.border.bottom)/r,"inside"===f.contain?o=Math.min(o,a,e):"outside"===f.contain&&(n=Math.max(n,a,e))),i.scale=Math.min(Math.max(t,n),o)),i}function x(t,e,n,o){var r,a,e=X(X({},f),e),p={x:m,y:h,opts:e};return!e.force&&(e.disablePan||e.panOnlyWhenZoomed&&v===e.startScale)||(t=parseFloat(t),e=parseFloat(e),e.disableXAxis||(p.x=(e.relative?m:0)+t),e.disableYAxis||(p.y=(e.relative?h:0)+e)),m!==p.x||h!==p.y?(m=p.x,h=p.y,y("panzoompan",p.opts,o)):{x:m,y:h,scale:v,isSVG:l,originalEvent:o}}function S(t,e,n){var o,r,e=b(t,e),a=e.opts;if(a.force||!a.disableZoom)return t=e.scale,e=m,o=h,a.focal&&(e=((r=a.focal).x/t-r.x/v+m*t)/t,o=(r.y/t-r.y/v+h*t)/t),r=w(e,o,t,{relative:!1,force:!0}),m=r.x,h=r.y,v=t,y("panzoomzoom",a,n)}function e(t,e){e=X(X(X({},f),{animate:!0}),e);return S(v*Math.exp((t?1:-1)*e.step),e)}function E(t,e,n,o){var r=N(u),a=r.parent.width-r.parent.padding.left-r.parent.padding.right-r.parent.border.left-r.parent.border.right,i=r.parent.height-r.parent.padding.top-r.parent.padding.bottom-r.parent.border.top-r.parent.border.bottom,c=e.clientX-r.parent.left-r.parent.padding.left-r.parent.border.left-r.elem.margin.left,e=e.clientY-r.parent.top-r.parent.padding.top-r.parent.border.top-r.elem.margin.top,r=(l||(c-=r.elem.width/v/2,e-=r.elem.height/v/2),{x:c/a*(a*t),y:e/i*(i*t)});return S(t,X(X({},n),{animate:!1,focal:r}),o)}S(f.startScale,{animate:!1,force:!0}),setTimeout(function(){x(f.startX,f.startY,{animate:!1,force:!0})});var M=[];function o(t){((t,e)=>{for(var n,o,r=t;null!=r;r=r.parentNode)if(n=r,o=e.excludeClass,1===n.nodeType&&-1<" ".concat((n.getAttribute("class")||"").trim()," ").indexOf(" ".concat(o," "))||-1<e.exclude.indexOf(r))return 1})(t.target,f)||(I(M,t),p=!0,f.handleStartEvent(t),g("panzoomstart",{x:r=m,y:a=h,scale:v,isSVG:l,originalEvent:t},f),t=R(M),i=t.clientX,c=t.clientY,s=v,d=W(M))}function A(t){var e,n,o;p&&void 0!==r&&void 0!==a&&void 0!==i&&void 0!==c&&(I(M,t),e=R(M),n=1<M.length,o=v,n&&(0===d&&(d=W(M)),E(o=b((W(M)-d)*f.step/80+s).scale,e,{animate:!1},t)),n&&!f.pinchAndPan||x(r+(e.clientX-i)/o,a+(e.clientY-c)/o,{animate:!1},t))}function P(t){1===M.length&&g("panzoomend",{x:m,y:h,scale:v,isSVG:l,originalEvent:t},f);var e=M;if(t.touches)for(;e.length;)e.pop();else{t=G(e,t);-1<t&&e.splice(t,1)}p&&(p=!1,r=a=i=c=void 0)}var O=!1;function z(){O||(O=!0,L("down",f.canvas?n:u,o),L("move",document,A,{passive:!0}),L("up",document,P,{passive:!0}))}return f.noBind||z(),{bind:z,destroy:function(){O=!1,V("down",f.canvas?n:u,o),V("move",document,A),V("up",document,P)},eventNames:T,getPan:function(){return{x:m,y:h}},getScale:function(){return v},getOptions:function(){var t,e=f,n={};for(t in e)e.hasOwnProperty(t)&&(n[t]=e[t]);return n},handleDown:o,handleMove:A,handleUp:P,pan:x,reset:function(t){var t=X(X(X({},f),{animate:!0,force:!0}),t),e=(v=b(t.startScale,t).scale,w(t.startX,t.startY,v,t));return m=e.x,h=e.y,y("panzoomreset",t)},resetStyle:function(){n.style.overflow="",n.style.userSelect="",n.style.touchAction="",n.style.cursor="",u.style.cursor="",u.style.userSelect="",u.style.touchAction="",C(u,"transformOrigin","")},setOptions:function(t){for(var e in t=void 0===t?{}:t)t.hasOwnProperty(e)&&(f[e]=t[e]);(t.hasOwnProperty("cursor")||t.hasOwnProperty("canvas"))&&(n.style.cursor=u.style.cursor="",(f.canvas?n:u).style.cursor=f.cursor),t.hasOwnProperty("overflow")&&(n.style.overflow=t.overflow),t.hasOwnProperty("touchAction")&&(n.style.touchAction=t.touchAction,u.style.touchAction=t.touchAction)},setStyle:function(t,e){return C(u,t,e)},zoom:S,zoomIn:function(t){return e(!0,t)},zoomOut:function(t){return e(!1,t)},zoomToPoint:E,zoomWithWheel:function(t,e){t.preventDefault();var e=X(X(X({},f),e),{animate:!1}),n=0===t.deltaY&&t.deltaX?t.deltaX:t.deltaY;return E(b(v*Math.exp((n<0?1:-1)*e.step/3),e).scale,t,e,t)}}}return t.defaultOptions=q,t});</script>

  <script>
    const slide = document.getElementById("slide");
    const container = document.getElementById("slide-container");

    const panzoom = Panzoom(slide, { maxScale: 40 });

    function zoomIn() { panzoom.zoomIn(); }
    function zoomOut() { panzoom.zoomOut(); }
    function resetZoom() { panzoom.reset(); }
    function toggleFullscreen() {
      if (!document.fullscreenElement) document.getElementById('app').requestFullscreen();
      else document.exitFullscreen();
    }

    container.addEventListener("wheel", panzoom.zoomWithWheel);
    slide.addEventListener("dblclick", () => {
      panzoom.zoom(panzoom.getScale() === 1 ? 2 : 1);
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", e => {
      if (e.key === "+") zoomIn();
      if (e.key === "-") zoomOut();
      if (e.ctrlKey && e.key === "0") resetZoom();
      if (e.key === "f") toggleFullscreen();
    });
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@router.get("/latest_detection_image")
def latest_detection_image():
    """Serve the latest image with detection bounding boxes."""
    try:
        import glob
        # Find all _DETECTED.jpg files in raw_images directory (including subdirectories)
        pattern = os.path.join("raw_images", "**", "*_DETECTED.jpg")
        detected_files = glob.glob(pattern, recursive=True)

        if detected_files:
            # Get the most recent file by modification time
            latest_file = max(detected_files, key=os.path.getmtime)

            # Read and return the image
            with open(latest_file, 'rb') as f:
                image_data = f.read()

            return Response(content=image_data, media_type="image/jpeg")
        else:
            # Return a placeholder image or 404
            raise HTTPException(status_code=404, detail="No detection images found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving latest detection image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detection_stream")
def detection_stream():
    """Serve a continuous MJPEG stream of detection results."""
    return StreamingResponse(
        generate_detection_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
