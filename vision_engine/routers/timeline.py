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


@router.get("/api/live_channels")
async def live_channels(request: Request, max_age_sec: int = 30):
    """3.21.18 — Live per-class value strip for the Status page (Dashboard tab).

    Reads the most recent detection events from Redis (no DB) and returns
    the latest value seen per class — yolo classes get confidence (0..1),
    math channels get their raw metric value (often > 1). Filtered to
    classes the operator has marked `show=true` in audio_settings; classes
    with `show=false` are dropped from the response entirely.

    This is *display-only* and *non-persistent*: nothing in this path
    touches `inference_results`, so operators can leave Store=OFF on math
    channels and still see live values on the dashboard.
    """
    try:
        watcher = getattr(request.app.state, "watcher_instance", None)
        if not watcher or not watcher.redis_connection or not watcher.redis_connection.redis_connection:
            return {"channels": {}, "max_age_sec": max_age_sec, "now": time.time()}
        r = watcher.redis_connection.redis_connection

        # Walk the last ~50 events so we have enough history to cover the
        # slowest math channel's stride. Most lines fire something every
        # frame, so this is rarely more than a few seconds of history.
        raw_events = r.lrange(DETECTION_EVENTS_REDIS_KEY, 0, 49) or []
        now = time.time()

        # Audio settings drives Show gating.
        from config import load_service_config as _lsc
        _svc = _lsc() or {}
        _audio = _svc.get("audio_settings", {}) or {}
        # Default: shown. Only drop classes the operator explicitly set show=false.
        hidden = {k for k, v in _audio.items() if isinstance(v, dict) and v.get("show") is False}

        # latest per class: pick the youngest event that mentions the class
        latest = {}
        for raw in raw_events:
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            ts = float(ev.get("timestamp") or 0)
            if not ts or (now - ts) > max_age_sec:
                continue
            details = ev.get("details") or {}
            dets = details.get("detections") or []
            cam = details.get("_cam") or ev.get("_cam") or ""
            for d in dets:
                if not isinstance(d, dict):
                    continue
                name = d.get("name") or d.get("class") or ""
                if not name or name in hidden:
                    continue
                # Skip if we already have a newer entry
                prev = latest.get(name)
                if prev and prev["ts"] >= ts:
                    continue
                cv = d.get("confidence")
                try:
                    cv_f = float(cv) if cv is not None else None
                except (TypeError, ValueError):
                    cv_f = None
                is_math = (cv_f is not None and cv_f > 1.0)
                latest[name] = {
                    "value": cv_f,
                    "is_math": is_math,
                    "ts": ts,
                    "age_sec": round(now - ts, 2),
                    "camera": cam or (d.get("_cam") or ""),
                    "severity": int((_audio.get(name, {}) or {}).get("severity", 0) or 0),
                }
        return {
            "channels": latest,
            "max_age_sec": max_age_sec,
            "now": now,
            "shown_count": len(latest),
        }
    except Exception as e:
        logger.warning(f"live_channels failed: {e}")
        return {"channels": {}, "max_age_sec": max_age_sec, "now": time.time(), "error": str(e)}


@router.get("/api/latest_detections")
async def get_latest_detections(request: Request):
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
async def timeline_feed(request: Request):
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
async def timeline_image(request: Request, page: int = 0):
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
async def timeline_count(request: Request):
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
    """Per-class p50/p95/n of confidence over last 7 days of stored detections."""
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
        cur.execute(
            """
            SELECT (det->>'name') AS cls,
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
        for cls, p50, p95, n in cur.fetchall():
            if cls:
                out[str(cls)] = {
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
async def get_conf_baselines():
    """Per-class confidence baselines (p50, p95, n) from last 7 days of stored
    detections. Cached for 1 hour. Used by the Process tab to display a
    read-only badge under each class card showing 'normal range'."""
    return JSONResponse(content={"baselines": _compute_conf_baselines()})


@router.post("/api/conf_baselines/recompute")
async def recompute_conf_baselines():
    """Force-invalidate the baseline cache (used by admin/dev tools)."""
    _baseline_cache["computed_at"] = 0.0
    return JSONResponse(content={"baselines": _compute_conf_baselines()})


def _compute_quality_payload(shipment: str = "", window: str = "24h") -> dict:
    """Shared computation for the score endpoint and PDF report.

    Returns the full payload dict (same shape the /api/shipment_quality_score
    endpoint serializes). Returns the `empty` payload if the DB is unreachable
    or has no rows for the requested window/shipment.
    """
    _windows = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days"}
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

        SCALE = 1.0
        score = max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, impact_per_unit * SCALE))))

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
            top_defects.append({
                "class":         cls,
                "impact":        round(imp, 2),
                "impact_per_unit": round(imp / denom, 4) if denom > 0 else 0.0,
                "count":         n,
                "count_per_unit": round(n / denom, 4) if denom > 0 else 0.0,
                "severity":      severities.get(cls, 0),
            })

        return {
            "shipment": shipment, "window": window,
            "score": round(score, 1),
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


@router.get("/api/shipment_quality_score")
async def shipment_quality_score(request: Request, shipment: str = "", window: str = "24h"):
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
async def shipment_quality_score_report(request: Request, shipment: str = "", window: str = "24h"):
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
async def shipment_quality_score_trend(
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


@router.get("/api/detection_stats")
async def detection_stats(request: Request, window: str = "1h", min_conf: float = 0.0):
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
async def detection_charts(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0):
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

    All aggregation runs over a capped recent slice (LIMIT 20000 rows) so a huge
    hypertable can't make this slow. Reads from inference_results, so it reflects
    only Store=ON classes. Returns a well-formed empty payload on any error.
    """
    _windows = {
        "1h":  ("1 hour",   "1 minute"),
        "6h":  ("6 hours",  "5 minutes"),
        "24h": ("24 hours", "30 minutes"),
        "7d":  ("7 days",   "6 hours"),
    }
    interval, bucket = _windows.get(window, _windows["24h"])
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

        # --- camera scatter: stratified per-class so rare classes (spot/warp/stitch)
        # don't get swamped by dominant ones (weft_up). Up to 750 newest dots PER
        # CLASS, capped at 6000 total across classes.
        cur.execute(
            f"""
            WITH recent AS (
                SELECT time, shipment, detections, image_path FROM inference_results
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                ORDER BY time DESC
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
            base_params,
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
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                  AND encoder_value IS NOT NULL
                ORDER BY time DESC
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
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY cls ORDER BY t DESC) AS rn
                FROM exploded
            )
            SELECT enc, cam, cls, conf, img, ship FROM ranked
            WHERE rn <= 750
            LIMIT 6000
            """,
            base_params,
        )
        camera_scatter_encoder = [
            {"x": int(enc or 0), "y": cam, "cls": cls,
             "r": round((conf or 0), 3), "img": img, "ship": ship}
            for enc, cam, cls, conf, img, ship in cur.fetchall()
        ]

        # --- confidence by class over time (one line per class) ---
        cur.execute(
            f"""
            WITH recent AS (
                SELECT time, detections FROM inference_results
                WHERE time > NOW() - INTERVAL %s {ship_clause}
                ORDER BY time DESC
            ),
            expanded AS (
                SELECT time_bucket(INTERVAL '{bucket}', time) AS bkt,
                       (elem->>'name') AS cls,
                       (elem->>'confidence')::float AS conf
                FROM recent, LATERAL jsonb_array_elements(recent.detections) elem
                WHERE COALESCE((elem->>'confidence')::float, 0) >= %s
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

        return JSONResponse(content={
            "shipments": shipments,
            "size_over_time": size_over_time,
            "confidence_over_time": confidence_over_time,
            "confidence_by_class": confidence_by_class,
            "camera_scatter": camera_scatter,
            "camera_scatter_encoder": camera_scatter_encoder,
            "window": window,
            "shipment": shipment,
        })
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
async def ejection_stats(request: Request, window: str = "24h", shipment: str = ""):
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
async def production_stats(request: Request, window: str = "24h", shipment: str = ""):
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
async def quality_charts(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0):
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
async def timeline_clear(request: Request):
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
async def get_timeline_config(request: Request):
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
        object_filters = data.get('object_filters', {})  # {name: {show: bool, min_confidence: float}}
        procedures = data.get('procedures', getattr(request.app.state, 'timeline_config', {}).get('procedures', []))

        # Strip legacy eject fields from object_filters (handled by procedures now)
        for obj_cfg in object_filters.values():
            obj_cfg.pop('eject', None)
            obj_cfg.pop('eject_condition', None)

        # Store configuration in app state
        request.app.state.timeline_config = {
            'show_bounding_boxes': show_bounding_boxes,
            'camera_order': camera_order,
            'custom_camera_order': custom_camera_order,
            'image_quality': image_quality,
            'num_rows': num_rows,
            'buffer_size': buffer_size,
            'image_rotation': image_rotation,
            'object_filters': object_filters,
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

        logger.info(f"Timeline config saved: order={camera_order}, quality={image_quality}, rows={num_rows}, buffer={buffer_size}, object_filters={len(object_filters)} objects")

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
async def recent_detections(request: Request, window: str = "24h", shipment: str = "",
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
async def export_csv(request: Request, window: str = "24h", shipment: str = "", min_conf: float = 0.0):
    """Stream a CSV export of stored detections for a shipment+window (3.21.3).

    One row per detection (inference_results expanded). Honors the Charts tab's
    window + shipment + min_conf selectors (3.21.10). Filename: detections_<shipment>_<window>.csv.
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
            writer.writerow([
                "time", "shipment", "encoder", "camera", "class", "confidence",
                "severity", "impact",                                  # 3.21.12
                "xmin", "ymin", "xmax", "ymax", "image_path",
                "inference_time_ms", "model_used",
            ])
            yield buf.getvalue(); buf.seek(0); buf.truncate(0)

            for t, ship, enc, img, infms, model, dets in cur:
                ts = t.strftime("%Y-%m-%d %H:%M:%S.%f") if t else ""
                if not dets:
                    continue
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
    fname = f"detections_{safe_ship}_{window}.csv"
    return StreamingResponse(gen(), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.get("/api/raw_image/{path:path}")
async def serve_raw_image(path: str):
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


@router.get("/api/timeline_frame")
async def timeline_frame(request: Request, cam: int = 1, col: int = 0, page: int = 0, path: str = ""):
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
async def timeline_meta(request: Request, page: int = 0):
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
async def timeline_slideshow():
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
async def latest_detection_image():
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
async def detection_stream():
    """Serve a continuous MJPEG stream of detection results."""
    return StreamingResponse(
        generate_detection_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
