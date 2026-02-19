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

logger = logging.getLogger(__name__)
router = APIRouter()

# Absolute path to raw_images for path traversal protection
_RAW_IMAGES_ROOT = pathlib.Path("raw_images").resolve()

# Header strip height in pixels
_HEADER_HEIGHT = 28


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
        redis_client = Redis("redis", 6379, db=0)

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
        redis_client = Redis("redis", 6379, db=0)

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
                        for det in detections:
                            try:
                                name = det.get('name', '')
                                confidence = det.get('confidence', 0)
                                of = obj_filters.get(name, {})
                                if of.get('show') is False:
                                    continue
                                if confidence < of.get('min_confidence', 0.01):
                                    continue
                                x1 = int(det.get('xmin', det.get('x1', 0)))
                                y1 = int(det.get('ymin', det.get('y1', 0)))
                                x2 = int(det.get('xmax', det.get('x2', 0)))
                                y2 = int(det.get('ymax', det.get('y2', 0)))
                                cv2.rectangle(thumb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{name} {confidence:.0%}"
                                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                                cv2.rectangle(thumb, (x1, y1 - lh - 4), (x1 + lw + 4, y1), (0, 255, 0), -1)
                                cv2.putText(thumb, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
                            except Exception:
                                pass

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

        redis_client = Redis("redis", 6379, db=0)
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


@router.post("/api/timeline_clear")
async def timeline_clear(request: Request):
    """Clear all frames from the timeline buffer."""
    try:
        redis_client = Redis("redis", 6379, db=0)
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
        redis_client = Redis("redis", 6379, db=0)
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
async def timeline_frame(request: Request, cam: int = 1, col: int = 0, page: int = 0):
    """Serve full-res raw image with bboxes drawn from timeline detection data.

    This ensures the popup shows the same detections as the stitched thumbnail,
    even when consecutive captures share the same underlying camera frame.
    """
    try:
        tl_config = getattr(request.app.state, 'timeline_config', {})
        frames_per_page = tl_config.get('num_rows', 10)
        image_rotation = tl_config.get('image_rotation', 0)
        obj_filters = tl_config.get('object_filters', {})
        show_bbox = tl_config.get('show_bounding_boxes', True)

        redis_client = Redis("redis", 6379, db=0)
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

        total = len(all_frames)
        end_index = total - page * frames_per_page
        start_index = max(0, end_index - frames_per_page)
        page_slice = all_frames[start_index:end_index] if end_index > 0 else []

        if col < 0 or col >= len(page_slice):
            raise HTTPException(status_code=404, detail="Column out of range")

        ts, jpeg_bytes, detections, meta = page_slice[col]
        d_path = meta.get('d_path') if meta else None

        # Try to load full-res image from disk
        image = None
        if d_path:
            raw_path = pathlib.Path("raw_images") / f"{d_path}.jpg"
            if raw_path.exists():
                image = cv2.imread(str(raw_path))

        # Fallback to timeline thumbnail if disk file missing
        if image is None:
            image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Draw bboxes from timeline detection data
        if show_bbox and detections:
            for det in detections:
                try:
                    name = det.get('name', '')
                    confidence = det.get('confidence', 0)
                    of = obj_filters.get(name, {})
                    if of.get('show') is False:
                        continue
                    if confidence < of.get('min_confidence', 0.01):
                        continue
                    x1 = int(det.get('xmin', det.get('x1', 0)))
                    y1 = int(det.get('ymin', det.get('y1', 0)))
                    x2 = int(det.get('xmax', det.get('x2', 0)))
                    y2 = int(det.get('ymax', det.get('y2', 0)))
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"{name} {confidence:.0%}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(image, (x1, y1 - lh - 10), (x1 + lw + 10, y1), (0, 255, 0), -1)
                    cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                except Exception:
                    pass

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

        redis_client = Redis("redis", 6379, db=0)
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
