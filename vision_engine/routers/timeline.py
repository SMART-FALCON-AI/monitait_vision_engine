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
QUALITY_RELEASE_SCORE = 85
QUALITY_REINSPECT_SCORE = 60


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
    ship_clause = "AND shipment = %s" if shipment else ""
    base_params = [interval] + ([shipment] if shipment else [])

    empty = {
        "shipment": shipment, "window": window,
        "score": None, "verdict": "NO_DATA",
        "impact_total": 0.0, "impact_by_class": {},
        "total_detections": 0, "top_defects": [],
        "encoder_min": None, "encoder_max": None, "encoder_span": 0,
        "encoder_unit": "encoder_unit", "encoder_units_per_meter": None,
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
        try:
            encoder_units_per_meter = float(_svc.get("encoder_units_per_meter") or 0)
        except (TypeError, ValueError):
            encoder_units_per_meter = 0.0

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT (det->>'name') AS cls,
                   COUNT(*)       AS n,
                   SUM((det->>'confidence')::float) AS conf_sum
            FROM inference_results, LATERAL jsonb_array_elements(detections) det
            WHERE time > NOW() - INTERVAL %s {ship_clause}
              AND (det->>'confidence') IS NOT NULL
            GROUP BY (det->>'name')
            """,
            base_params,
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

        if score >= QUALITY_RELEASE_SCORE:
            verdict = "RELEASE"
        elif score >= QUALITY_REINSPECT_SCORE:
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
            "thresholds": {"release": QUALITY_RELEASE_SCORE, "reinspect": QUALITY_REINSPECT_SCORE},
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
    upm = payload.get("encoder_units_per_meter")
    length_txt = f"{span:,} {unit}"
    if upm and span:
        length_txt += f"  (≈ {span / upm:,.1f} m)"
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
def quality_shipments(request: Request, n: int = 30, window: str = "30d"):
    """3.25.4 — recent shipments + their quality scores for a per-shipment bar chart.

    Returns up to `n` shipments started within `window`, each with its
    quality score + verdict + detection count. Order is most-recent-first.
    Chart shows one bar per shipment: bar height = score, color = verdict.
    """
    _windows = {"24h": "24 hours", "7d": "7 days", "30d": "30 days", "90d": "90 days"}
    interval = _windows.get(window, "30 days")
    out: list = []
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content={"shipments": []})
        try:
            cur = conn.cursor()
            # 3.25.4 hotfix — image_path is `raw_images/<shipment>/<hour>/<frame>.jpg`,
            # so the shipment label lives in segment 2.
            cur.execute(
                f"""
                SELECT SPLIT_PART(image_path, '/', 2) AS ship,
                       MIN(time) AS first_t,
                       MAX(time) AS last_t,
                       COUNT(*)   AS n_rows
                FROM inference_results
                WHERE time > NOW() - INTERVAL %s
                  AND image_path IS NOT NULL
                  AND SPLIT_PART(image_path, '/', 2) NOT IN ('', 'no_shipment')
                GROUP BY 1
                ORDER BY first_t DESC
                LIMIT %s
                """,
                (interval, int(n)),
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            try: release_db_connection(conn)
            except Exception: pass

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
        for ship, first_t, last_t, n_rows in rows:
            try:
                qp = _compute_quality_payload(shipment=ship, window=window)
                out.append({
                    "shipment": ship,
                    "score":            qp.get("score") if qp else None,
                    "score_absolute":   qp.get("score_absolute") if qp else None,
                    "score_relative":   qp.get("score_relative") if qp else None,
                    "impact_per_unit":  qp.get("impact_per_unit") if qp else None,
                    "verdict":          qp.get("verdict") if qp else None,
                    "rows":             int(n_rows or 0),
                    "first_t": first_t.isoformat() if first_t else None,
                    "last_t":  last_t.isoformat()  if last_t  else None,
                    "top_defects": [d.get("class") for d in (qp.get("top_defects") or [])][:3] if qp else [],
                })
            except Exception:
                out.append({
                    "shipment": ship, "score": None, "score_absolute": None,
                    "score_relative": None, "impact_per_unit": None,
                    "verdict": None, "rows": int(n_rows or 0), "top_defects": [],
                })
    except Exception as e:
        logger.warning(f"quality/shipments failed: {e}")
    return JSONResponse(content={"shipments": out})


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
    axis: str = "time",
    window: str = "24h",
    buckets: int = 48,
    shipment: str = "",
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
    # 4.0.39 — raise cap from 120 to 192 so the strip can match the colour
    # heatmap's max (N_BINS=192 in detection_charts). At 120 the heatmap had
    # more cells than the quality / ejection strip below it and the columns
    # didn't line up at high bucket counts.
    buckets = max(8, min(192, int(buckets)))
    axis_l = (axis or "time").lower()

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
                        {ship_clause}
                    ) src
                    GROUP BY bucket, procedure_name
                    """,
                    [buckets, enc_min, width, interval, *params],
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
            cur.execute("SELECT NOW() - INTERVAL %s, NOW()", [interval])
            t_min, t_max = cur.fetchone() or (None, None)
            if not t_min or not t_max:
                cur.close()
                return JSONResponse(content={
                    "axis": "time", "buckets": [],
                    "shipment": shipment, "note": "no time window",
                })

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
                    {ship_clause}
                ) src
                GROUP BY bucket, procedure_name
                """,
                [buckets, t_min, t_max, t_min, buckets, interval, *params],
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
    axis: str = "time",
    window: str = "24h",
    buckets: int = 48,
    shipment: str = "",
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
    # 4.0.39 — raise cap from 120 to 192 so the strip can match the colour
    # heatmap's max (N_BINS=192 in detection_charts). At 120 the heatmap had
    # more cells than the quality / ejection strip below it and the columns
    # didn't line up at high bucket counts.
    buckets = max(8, min(192, int(buckets)))
    axis_l = (axis or "time").lower()

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
    except Exception:
        sev_map = {}
        _heatmap_scale = 1.0

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
                        AND (det->>'name') IS NOT NULL
                        {ship_clause}
                    ) src
                    GROUP BY bucket, name
                    """,
                    [buckets, enc_min, width, interval, *params],
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
            cur.execute("SELECT NOW() - INTERVAL %s, NOW()", [interval])
            t_min, t_max = cur.fetchone() or (None, None)
            if not t_min or not t_max:
                cur.close()
                return JSONResponse(content={"axis": "time", "buckets": [], "shipment": shipment})

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
                    AND (det->>'name') IS NOT NULL
                    {ship_clause}
                ) src
                GROUP BY bucket, name
                """,
                [buckets, t_min, t_max, t_min, buckets, interval, *params],
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
def detection_stats(request: Request, window: str = "1h", min_conf: float = 0.0):
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
    _windows = {
        "1h":  ("1 hour",   "1 minute"),
        "6h":  ("6 hours",  "10 minutes"),
        "24h": ("24 hours", "1 hour"),
        "7d":  ("7 days",   "6 hours"),
    }
    interval, bucket = _windows.get(window, _windows["1h"])
    empty = {"by_class": {}, "timeline": [], "total": 0, "window": window, "persisted": False}

    conn = None
    try:
        from services.db import get_db_connection, release_db_connection
        conn = get_db_connection()
        if conn is None:
            return JSONResponse(content=empty)
        cur = conn.cursor()

        # Class distribution over the window. Prefer the human-readable `name`
        # (e.g. "DIE_LINE", "mean_L") over the numeric `class` index. Cap the rows
        # scanned so a huge hypertable (millions of rows) can't make this query
        # run for many seconds — a recent sample is representative for the panel.
        cur.execute(
            """
            SELECT COALESCE(elem->>'name', elem->>'class') AS cls, COUNT(*) AS n
            FROM (
                SELECT detections FROM inference_results
                WHERE time > NOW() - INTERVAL %s
                ORDER BY time DESC
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
            (interval, float(min_conf or 0.0)),
        )
        by_class = {}
        for cls, n in cur.fetchall():
            if cls is not None:
                by_class[str(cls)] = int(n)

        # Detections-per-bucket timeline
        cur.execute(
            """
            SELECT time_bucket(INTERVAL %s, time) AS bkt,
                   COALESCE(SUM(detection_count), 0) AS n
            FROM inference_results
            WHERE time > NOW() - INTERVAL %s
            GROUP BY bkt
            ORDER BY bkt
            """,
            (bucket, interval),
        )
        timeline = [{"t": b.strftime("%m-%d %H:%M"), "count": int(n)} for b, n in cur.fetchall()]
        cur.close()

        # 3.21.12 — impact score per class (severity × confidence, summed per class).
        # area_factor is currently 1 (no normalisation yet); will be added when we
        # have a reliable per-class typical-area baseline.
        from config import load_service_config as _lsc
        _svc = _lsc() or {}
        _audio = _svc.get("audio_settings", {}) or {}
        _severities = {k: (v.get("severity", 0) or 0) / 100.0 for k, v in _audio.items() if isinstance(v, dict)}
        impact_by_class = {}
        if any(s > 0 for s in _severities.values()):
            cur2 = conn.cursor()
            cur2.execute(
                """
                SELECT (det->>'name') AS cls,
                       SUM((det->>'confidence')::float) AS conf_sum
                FROM inference_results, LATERAL jsonb_array_elements(detections) det
                WHERE time > NOW() - INTERVAL %s
                  AND COALESCE((det->>'confidence')::float, 0) >= %s
                  -- 4.0.38: skip synthetic (`_color` etc.) from impact-by-class too.
                  AND COALESCE(det->>'name', '') !~ '^_'
                GROUP BY (det->>'name')
                """,
                (interval, float(min_conf or 0.0)),
            )
            for cls, conf_sum in cur2.fetchall():
                if cls and _severities.get(cls, 0) > 0:
                    impact_by_class[str(cls)] = round(_severities[cls] * float(conf_sum or 0.0), 2)
            cur2.close()
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
            try:
                from services.db import release_db_connection
                release_db_connection(conn)
            except Exception:
                pass


@router.get("/api/detection_charts")
def detection_charts(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0, baseline: str = "camera", phase: str = "", bins: int = 32, since_ms: int = 0, until_ms: int = 0, scatter_only: int = 0):
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
    # v4.0.75 — TTL cache lookup. All params contribute to the key so
    # different dashboards with different filters get different cached
    # results but the same dashboard polling on a 5-s interval only hits
    # the DB once per 20-s window instead of every poll.
    _cache_key = _endpoint_cache_key(
        "dc", window, shipment, min_conf, baseline, phase, bins,
        since_ms, until_ms, scatter_only,
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
                SELECT time, shipment, detections, image_path FROM inference_results
                WHERE { 'time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)' if _use_slice else 'time > NOW() - INTERVAL %s' } {ship_clause}
                ORDER BY time DESC
                LIMIT 50000  -- v4.0.78: bound the CTE so full-time-range scan can't blow the query
            ),
            exploded AS (
                SELECT EXTRACT(EPOCH FROM time) * 1000 AS x_ms,
                       time AS t,
                       COALESCE((elem->>'_cam')::int, 0) AS cam,
                       (elem->>'name') AS cls,
                       (elem->>'confidence')::float AS conf,
                       image_path AS img,
                       shipment AS ship
                FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
                  AND COALESCE(elem->>'name', '') !~ '^_'  -- 4.0.29: skip synthetic (_color etc.)
                  {_parent_filter_sql}
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY cls ORDER BY t DESC) AS rn
                FROM exploded
            )
            SELECT x_ms, cam, cls, conf, img, ship FROM ranked
            WHERE rn <= 750
            ORDER BY t DESC
            LIMIT 6000
            """,
            (([int(since_ms), int(until_ms)] if _use_slice else [interval]) + ([shipment] if shipment else []) + [float(min_conf or 0.0)] + _parent_filter_args),
        )
        camera_scatter = [
            {"x": int(x), "y": cam, "cls": cls, "r": round((conf or 0), 3),
             "img": img, "ship": ship}
            for x, cam, cls, conf, img, ship in cur.fetchall()
        ]

        # --- camera × ENCODER scatter (3.21.0): defect map by roll position ---
        # Stratified per-class (same as camera-time scatter above) so rare
        # classes survive next to weft_up's flood.
        cur.execute(
            f"""
            WITH recent AS (
                SELECT time, shipment, detections, image_path, encoder_value FROM inference_results
                WHERE { 'time >= to_timestamp(%s / 1000.0) AND time < to_timestamp(%s / 1000.0)' if _use_slice else 'time > NOW() - INTERVAL %s' } {ship_clause}
                  AND encoder_value IS NOT NULL
                ORDER BY time DESC
                LIMIT 50000  -- v4.0.78: bound the CTE so full-time-range scan can't blow the query
            ),
            exploded AS (
                SELECT encoder_value AS enc,
                       time AS t,
                       COALESCE((elem->>'_cam')::int, 0) AS cam,
                       (elem->>'name') AS cls,
                       (elem->>'confidence')::float AS conf,
                       image_path AS img,
                       shipment AS ship
                FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
                  AND COALESCE(elem->>'name', '') !~ '^_'  -- 4.0.29: skip synthetic (_color etc.)
                  {_parent_filter_sql}
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY cls ORDER BY t DESC) AS rn
                FROM exploded
            )
            SELECT enc, cam, cls, conf, img, ship FROM ranked
            WHERE rn <= 750
            LIMIT 6000
            """,
            (([int(since_ms), int(until_ms)] if _use_slice else [interval]) + ([shipment] if shipment else []) + [float(min_conf or 0.0)] + _parent_filter_args),
        )
        camera_scatter_encoder = [
            {"x": int(enc or 0), "y": cam, "cls": cls,
             "r": round((conf or 0), 3), "img": img, "ship": ship}
            for enc, cam, cls, conf, img, ship in cur.fetchall()
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
            _hm_cur.execute(
                f"""
                WITH color_rows AS (
                    SELECT
                        encoder_value AS enc,
                        (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                        (elem->>'L')::float AS L,
                        (elem->>'E')::float AS E,
                        (elem->>'phase') AS phase
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND elem->>'name' = '_color'
                      AND encoder_value IS NOT NULL
                      AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                      {_phase_clause}
                ),
                bounds AS (SELECT MIN(enc) AS lo, MAX(enc) AS hi FROM color_rows),
                binned AS (
                    SELECT cr.cam,
                           CASE WHEN b.hi > b.lo
                                THEN LEAST({N_BINS} - 1, GREATEST(0,
                                     FLOOR(((cr.enc - b.lo)::numeric * {N_BINS}) / (b.hi - b.lo))))::int
                                ELSE 0 END AS bin,
                           cr.L, cr.E
                    FROM color_rows cr CROSS JOIN bounds b
                ),
                cam_baselines AS (
                    SELECT cam, percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                    FROM binned GROUP BY cam
                )
                SELECT b.cam, b.bin, AVG(b.L)::float, AVG(b.E)::float, COUNT(*),
                       AVG(b.E)::float - MAX(cb.base_E)::float
                FROM binned b JOIN cam_baselines cb ON cb.cam = b.cam
                GROUP BY b.cam, b.bin ORDER BY b.cam, b.bin
                """,
                [interval] + ([shipment] if shipment else []) + _phase_args,
            )
            cells = []
            for cam, bin_idx, mL, mE, n, de in _hm_cur.fetchall():
                cells.append({"cam": int(cam or 0), "bin": int(bin_idx or 0),
                              "L": round(mL or 0, 2), "E": round(mE or 0, 2),
                              "delta_e": round(de or 0, 2), "n": int(n or 0)})
            _hm_cur.execute(
                f"""SELECT MIN(encoder_value), MAX(encoder_value)
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND elem->>'name' = '_color' AND encoder_value IS NOT NULL
                      AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                      {_phase_clause}""",
                [interval] + ([shipment] if shipment else []) + _phase_args,
            )
            row = _hm_cur.fetchone()
            color_heatmap = {
                "enc_min": int(row[0]) if row and row[0] is not None else None,
                "enc_max": int(row[1]) if row and row[1] is not None else None,
                "n_bins": N_BINS, "cells": cells,
            }
            # Time-binned twin
            _hm_cur.execute(
                f"""
                WITH color_rows AS (
                    SELECT EXTRACT(EPOCH FROM time) * 1000.0 AS t_ms,
                        (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                        (elem->>'L')::float AS L, (elem->>'E')::float AS E
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND elem->>'name' = '_color'
                      AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                      {_phase_clause}
                ),
                bounds AS (SELECT MIN(t_ms) AS lo, MAX(t_ms) AS hi FROM color_rows),
                binned AS (
                    SELECT cr.cam,
                           CASE WHEN b.hi > b.lo
                                THEN LEAST({N_BINS} - 1, GREATEST(0,
                                     FLOOR(((cr.t_ms - b.lo) * {N_BINS}) / (b.hi - b.lo))))::int
                                ELSE 0 END AS bin,
                           cr.L, cr.E
                    FROM color_rows cr CROSS JOIN bounds b
                ),
                cam_baselines AS (
                    SELECT cam, percentile_cont(0.5) WITHIN GROUP (ORDER BY E) AS base_E
                    FROM binned GROUP BY cam
                )
                SELECT b.cam, b.bin, AVG(b.L)::float, AVG(b.E)::float, COUNT(*),
                       AVG(b.E)::float - MAX(cb.base_E)::float
                FROM binned b JOIN cam_baselines cb ON cb.cam = b.cam
                GROUP BY b.cam, b.bin ORDER BY b.cam, b.bin
                """,
                [interval] + ([shipment] if shipment else []) + _phase_args,
            )
            tcells = []
            for cam, bin_idx, mL, mE, n, de in _hm_cur.fetchall():
                tcells.append({"cam": int(cam or 0), "bin": int(bin_idx or 0),
                               "L": round(mL or 0, 2), "E": round(mE or 0, 2),
                               "delta_e": round(de or 0, 2), "n": int(n or 0)})
            _hm_cur.execute(
                f"""SELECT MIN(EXTRACT(EPOCH FROM time) * 1000.0),
                           MAX(EXTRACT(EPOCH FROM time) * 1000.0)
                    FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                      AND elem->>'name' = '_color'
                      AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
                      {_phase_clause}""",
                [interval] + ([shipment] if shipment else []) + _phase_args,
            )
            row = _hm_cur.fetchone()
            color_heatmap_time = {
                "t_min": float(row[0]) if row and row[0] is not None else None,
                "t_max": float(row[1]) if row and row[1] is not None else None,
                "n_bins": N_BINS, "cells": tcells,
            }
            _hm_cur.close()
        except Exception as _he:
            logger.debug(f"color heatmap pre-aggregation failed: {_he}")

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
            color_baseline_modes = ["camera"]   # always available when there's any _color data
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

        _payload = {
            "shipments": shipments,
            "size_over_time": size_over_time,
            "confidence_over_time": confidence_over_time,
            "confidence_by_class": confidence_by_class,
            "camera_scatter": camera_scatter,
            "camera_scatter_encoder": camera_scatter_encoder,
            "camera_y_order": camera_y_order,
            "color_heatmap": color_heatmap,
            "color_heatmap_time": color_heatmap_time,
            "phases_available": phases_available,
            "phase": _phase,
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
            base_params,
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
        return JSONResponse(content={
            "by_class": by_class, "by_camera": by_camera, "heatmap": heatmap,
            "latency_over_time": latency_over_time, "window": window, "shipment": shipment,
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


@router.get("/api/export_csv")
def export_csv(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0, unwind: bool = False):
    """Stream a CSV export of stored detections for a shipment+window (3.21.3).

    One row per detection (inference_results expanded). Honors the Charts tab's
    window + shipment + min_conf selectors (3.21.10). Filename: detections_<shipment>_<window>.csv.

    4.0.59:
      * New ``length`` column = ``encoder_value - min(encoder_value in this shipment)``
        so each row shows how far the belt had moved when that detection was
        captured, in encoder counts.
      * New ``unwind`` query flag: for roll-unwinding lines the operator
        thinks of the SHIPMENT-END as position 0 and earlier detections as
        further along the roll. Setting ``unwind=true`` reports
        ``length = max(encoder) - encoder`` per row instead — same magnitude,
        inverted direction.
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
    interval = _windows.get(window, "24 hours")
    ship_clause = "AND shipment = %s" if shipment else ""
    params = [interval] + ([shipment] if shipment else [])
    min_conf_f = float(min_conf or 0.0)

    from services.db import get_db_connection, release_db_connection
    conn = get_db_connection()
    if conn is None:
        return Response(content="ERROR: database unavailable\n", status_code=503,
                        media_type="text/plain")

    import csv as _csv
    import io as _io

    def gen():
        try:
            # 3.21.12 — load per-class severity (0-100) for impact column.
            from config import load_service_config as _lsc
            _svc = _lsc() or {}
            _audio = _svc.get("audio_settings", {}) or {}
            sev_map = {k: int(v.get("severity", 0) or 0) for k, v in _audio.items() if isinstance(v, dict)}

            # 4.0.59 — pre-query min/max encoder for the selected window so the
            # length column can be computed per row without holding all rows in
            # memory. The MIN is the shipment-start anchor (equivalent to
            # watcher.shipment_start_encoder for the CURRENT shipment, but this
            # works for HISTORICAL shipments too because it's derived from the
            # data). MAX is only used in unwind mode. Runs on the same cursor
            # connection but a separate short-lived cursor so it doesn't
            # interfere with the streaming server-side cursor below.
            enc_min = 0
            enc_max = 0
            try:
                with conn.cursor() as _mcur:
                    _mcur.execute(
                        f"""SELECT MIN(encoder_value), MAX(encoder_value)
                            FROM inference_results
                            WHERE time > NOW() - INTERVAL %s {ship_clause}""",
                        params,
                    )
                    _row = _mcur.fetchone()
                    if _row:
                        enc_min = int(_row[0]) if _row[0] is not None else 0
                        enc_max = int(_row[1]) if _row[1] is not None else 0
            except Exception as _me:
                logger.debug(f"export_csv min/max encoder probe failed: {_me}")

            cur = conn.cursor(name="export_csv_cursor")  # server-side cursor (streaming, low RAM)
            cur.itersize = 500
            cur.execute(
                f"""SELECT time, shipment, encoder_value, image_path,
                           inference_time_ms, model_used, detections
                    FROM inference_results
                    WHERE time > NOW() - INTERVAL %s {ship_clause}
                    ORDER BY time ASC""",
                params,
            )
            buf = _io.StringIO()
            writer = _csv.writer(buf)
            # 4.0.59 — added "length" between encoder and camera.
            writer.writerow([
                "time", "shipment", "encoder", "length", "camera", "class", "confidence",
                "severity", "impact",                                  # 3.21.12
                "xmin", "ymin", "xmax", "ymax", "image_path",
                "inference_time_ms", "model_used",
            ])
            yield buf.getvalue(); buf.seek(0); buf.truncate(0)

            for t, ship, enc, img, infms, model, dets in cur:
                ts = t.strftime("%Y-%m-%d %H:%M:%S.%f") if t else ""
                if not dets:
                    continue
                # 4.0.59 — compute length per row. Wind (default) = distance
                # travelled from the shipment start (min encoder). Unwind =
                # distance remaining before the end (max encoder − now), the
                # right mental model for a roll being unwound from full.
                if enc is None:
                    length_val = ""
                elif unwind:
                    length_val = int(enc_max - int(enc))
                else:
                    length_val = int(int(enc) - enc_min)
                for d in dets:
                    if not isinstance(d, dict):
                        continue
                    # 3.21.10: filter rows by min_conf so CSV matches what the
                    # charts show.
                    try:
                        conf_f = float(d.get("confidence") or 0.0)
                        if conf_f < min_conf_f:
                            continue
                    except (TypeError, ValueError):
                        conf_f = 0.0
                    cls = d.get("name", "")
                    sev = sev_map.get(cls, 0)
                    impact = round((sev / 100.0) * conf_f, 3)  # 3.21.12 — defect impact
                    writer.writerow([
                        ts, ship or "", enc if enc is not None else "",
                        length_val,
                        d.get("_cam", ""), cls,
                        d.get("confidence", ""),
                        sev, impact,
                        d.get("xmin", ""), d.get("ymin", ""),
                        d.get("xmax", ""), d.get("ymax", ""),
                        img or "", infms or "", model or "",
                    ])
                    yield buf.getvalue(); buf.seek(0); buf.truncate(0)
            cur.close()
        except Exception as e:
            logger.warning(f"export_csv stream failed: {e}")
            yield f"# ERROR: {e}\n"
        finally:
            try: release_db_connection(conn)
            except Exception: pass

    import re as _re
    safe_ship = _re.sub(r"[^A-Za-z0-9_-]+", "_", shipment) if shipment else "all"
    # 4.0.59 — include direction hint in filename so unwind exports are
    # visibly distinct from wind exports on disk.
    _dir_tag = "_unwind" if unwind else ""
    fname = f"detections_{safe_ship}_{window}{_dir_tag}.csv"
    return StreamingResponse(gen(), media_type="text/csv; charset=utf-8",
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
                cur = conn.cursor()
                # 1: exact raw
                cur.execute(
                    "SELECT detections FROM inference_results WHERE image_path = %s "
                    "ORDER BY time DESC LIMIT 1",
                    (exact_raw,),
                )
                row = cur.fetchone()
                # 2: exact DETECTED sibling
                if not row:
                    cur.execute(
                        "SELECT detections FROM inference_results WHERE image_path = %s "
                        "ORDER BY time DESC LIMIT 1",
                        (exact_det,),
                    )
                    row = cur.fetchone()
                # 3: legacy stem LIKE (kept for any out-of-pattern stored paths,
                # but BOTH endpoints in this row pair are now warned-logged
                # so we can find and clean them up).
                if not row:
                    cur.execute(
                        "SELECT detections FROM inference_results WHERE image_path LIKE %s "
                        "ORDER BY time DESC LIMIT 1",
                        ("%" + stem + "%",),
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
