"""WebSocket endpoints for real-time dashboard streaming.

Provides two WebSocket endpoints:
- /ws/timeline: Event-driven timeline composite push (via Redis Pub/Sub)
- /ws/camera/{camera_id}: Throttled live camera feed (~3 FPS)
"""

import asyncio
import cv2
import logging
import numpy as np
import pickle
from typing import Dict, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from redis import Redis

import config as cfg
from config import TIMELINE_REDIS_PREFIX

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    """Tracks WebSocket clients per channel, broadcasts binary data."""

    def __init__(self):
        self._channels: Dict[str, List[WebSocket]] = {}

    async def connect(self, channel: str, ws: WebSocket):
        await ws.accept()
        self._channels.setdefault(channel, []).append(ws)
        logger.info(f"WS connect: {channel} ({len(self._channels[channel])} clients)")

    def disconnect(self, channel: str, ws: WebSocket):
        clients = self._channels.get(channel, [])
        if ws in clients:
            clients.remove(ws)
        logger.info(f"WS disconnect: {channel} ({len(clients)} clients)")

    async def broadcast_binary(self, channel: str, data: bytes):
        dead: List[WebSocket] = []
        for ws in self._channels.get(channel, []):
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(channel, ws)

    async def send_binary(self, ws: WebSocket, data: bytes):
        try:
            await ws.send_bytes(data)
        except Exception:
            pass


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Timeline composite cache
# ---------------------------------------------------------------------------
_timeline_cache_version = -1
_timeline_cache_bytes: bytes = b""
_timeline_cache_page: int = 0


def _build_timeline_composite(page: int, app_state) -> bytes | None:
    """Build the timeline composite JPEG (runs in executor thread).
    Reuses the same logic as timeline.py /timeline_image endpoint."""
    try:
        # Get config
        try:
            tl_config = app_state.timeline_config
            quality = tl_config.get('image_quality', 85)
            frames_per_page = tl_config.get('num_rows', 10)
            image_rotation = tl_config.get('image_rotation', 0)
            show_bbox = tl_config.get('show_bounding_boxes', True)
            obj_filters = tl_config.get('object_filters', {})
        except Exception:
            quality = 85
            frames_per_page = 10
            image_rotation = 0
            show_bbox = True
            obj_filters = {}

        redis_client = Redis("redis", 6379, db=0)
        all_keys = redis_client.keys(f"{TIMELINE_REDIS_PREFIX}*")
        if not all_keys:
            return None

        # Sort keys by camera ID
        def extract_cam_id(key):
            k = key.decode() if isinstance(key, bytes) else key
            try:
                return int(k.split(":")[-1])
            except ValueError:
                return 0
        all_keys.sort(key=extract_cam_id)

        # Collect frames per camera
        camera_frames_raw = {}
        max_total_frames = 0
        for key in all_keys:
            frames_raw = redis_client.lrange(key, 0, -1)
            cam_id = extract_cam_id(key)
            camera_frames_raw[cam_id] = frames_raw if frames_raw else []
            max_total_frames = max(max_total_frames, len(camera_frames_raw[cam_id]))

        total_pages = max(1, (max_total_frames + frames_per_page - 1) // frames_per_page)
        safe_page = page if 0 <= page < total_pages else 0

        camera_rows = []
        for cam_id in sorted(camera_frames_raw.keys()):
            frames_raw = camera_frames_raw[cam_id]
            all_frames = []
            for frame_data in frames_raw:
                unpacked = pickle.loads(frame_data)
                if len(unpacked) == 3:
                    ts, jpeg_bytes, detections = unpacked
                else:
                    ts, jpeg_bytes = unpacked
                    detections = None
                all_frames.append((ts, jpeg_bytes, detections))
            all_frames.sort(key=lambda x: x[0])

            total = len(all_frames)
            end_index = total - (safe_page * frames_per_page)
            start_index = max(0, end_index - frames_per_page)
            page_slice = all_frames[start_index:end_index] if end_index > 0 else []

            row_frames = []
            for ts, jpeg_bytes, detections in page_slice:
                thumb = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if thumb is not None:
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
            return None

        max_width = max(row.shape[1] for row in camera_rows)
        padded_rows = []
        for row in camera_rows:
            if row.shape[1] < max_width:
                pad = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                row = np.hstack([row, pad])
            padded_rows.append(row)

        composite = np.vstack(padded_rows)
        _, jpeg = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

    except Exception as e:
        logger.error(f"WS timeline composite error: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# /ws/timeline — event-driven timeline push
# ---------------------------------------------------------------------------
@router.websocket("/ws/timeline")
async def ws_timeline(websocket: WebSocket):
    global _timeline_cache_version, _timeline_cache_bytes, _timeline_cache_page

    app_state = websocket.app.state
    await manager.connect("timeline", websocket)
    client_page = 0

    try:
        # Send current composite immediately on connect
        loop = asyncio.get_event_loop()
        jpeg = await loop.run_in_executor(None, _build_timeline_composite, client_page, app_state)
        if jpeg:
            _timeline_cache_bytes = jpeg
            _timeline_cache_version = cfg.timeline_frame_counter
            _timeline_cache_page = client_page
            await manager.send_binary(websocket, jpeg)

        # Start two concurrent tasks: listen for client messages + subscribe to Redis updates
        async def _listen_client():
            """Receive page navigation from client."""
            nonlocal client_page
            while True:
                try:
                    msg = await websocket.receive_text()
                    client_page = int(msg)
                    # Immediately rebuild for this client's new page
                    jpeg = await loop.run_in_executor(
                        None, _build_timeline_composite, client_page, app_state
                    )
                    if jpeg:
                        await manager.send_binary(websocket, jpeg)
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception:
                    pass

        async def _listen_redis():
            """Subscribe to timeline updates from inference workers via Redis Pub/Sub."""
            nonlocal client_page
            global _timeline_cache_version, _timeline_cache_bytes, _timeline_cache_page

            def _wait_for_message(pubsub, timeout=1.0):
                """Blocking Redis get_message — runs in executor thread."""
                return pubsub.get_message(timeout=timeout)

            r = Redis("redis", 6379, db=0)
            pubsub = r.pubsub()
            pubsub.subscribe("ws:timeline_update")

            try:
                while True:
                    # Wait for Redis notification (blocking in executor)
                    msg = await loop.run_in_executor(None, _wait_for_message, pubsub, 2.0)
                    if msg and msg["type"] == "message":
                        current_version = cfg.timeline_frame_counter
                        # Only rebuild if version changed or page differs
                        if current_version != _timeline_cache_version or client_page != _timeline_cache_page:
                            jpeg = await loop.run_in_executor(
                                None, _build_timeline_composite, client_page, app_state
                            )
                            if jpeg:
                                _timeline_cache_version = current_version
                                _timeline_cache_bytes = jpeg
                                _timeline_cache_page = client_page
                                await manager.broadcast_binary("timeline", jpeg)
            except (WebSocketDisconnect, RuntimeError):
                pass
            except Exception as e:
                logger.debug(f"WS timeline redis loop error: {e}")
            finally:
                pubsub.unsubscribe()
                pubsub.close()

        # Run both concurrently; when one exits, cancel the other
        listen_client_task = asyncio.create_task(_listen_client())
        listen_redis_task = asyncio.create_task(_listen_redis())
        done, pending = await asyncio.wait(
            [listen_client_task, listen_redis_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS timeline error: {e}")
    finally:
        manager.disconnect("timeline", websocket)


# ---------------------------------------------------------------------------
# /ws/camera/{camera_id} — throttled live camera feed
# ---------------------------------------------------------------------------
@router.websocket("/ws/camera/{camera_id}")
async def ws_camera(websocket: WebSocket, camera_id: int):
    app_state = websocket.app.state
    watcher = getattr(app_state, 'watcher_instance', None)
    if watcher is None or not watcher.cameras:
        await websocket.accept()
        await websocket.close(code=1011, reason="Watcher not initialized")
        return

    cam = watcher.cameras.get(camera_id)
    if cam is None:
        await websocket.accept()
        await websocket.close(code=1011, reason=f"Camera {camera_id} not found")
        return

    channel = f"camera:{camera_id}"
    await manager.connect(channel, websocket)

    try:
        loop = asyncio.get_event_loop()
        while True:
            # Read frame in executor to avoid blocking asyncio
            frame = await loop.run_in_executor(None, cam.read)
            if frame is not None:
                # Encode JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                await manager.send_binary(websocket, jpeg.tobytes())
            await asyncio.sleep(0.33)  # ~3 FPS
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS camera {camera_id} error: {e}")
    finally:
        manager.disconnect(channel, websocket)
