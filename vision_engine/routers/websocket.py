"""WebSocket endpoints for real-time dashboard streaming.

Provides two WebSocket endpoints:
- /ws/timeline: Event-driven timeline composite push (via Redis Pub/Sub)
- /ws/camera/{camera_id}: Throttled live camera feed (~3 FPS)
"""

import asyncio
import cv2
import json
import logging
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from redis import Redis

import config as cfg
from config import TIMELINE_REDIS_PREFIX
from services.detection import evaluate_eject_from_detections
from services.render import draw_detection_on
from routers.timeline import _unpack_timeline_entry, _build_header_strip, _HEADER_HEIGHT

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

    async def send_text(self, ws: WebSocket, data: str):
        try:
            await ws.send_text(data)
        except Exception:
            pass

    async def broadcast_text(self, channel: str, data: str):
        dead: List[WebSocket] = []
        for ws in self._channels.get(channel, []):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(channel, ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Timeline composite cache
# ---------------------------------------------------------------------------
# 4.0.377 — the shared global timeline cache was removed: it broadcast one client's
# render (with that client's mark state) to every timeline connection, which made the
# red box flicker on/off across tabs/reconnects. Each connection now renders + caches
# its own last-sent state locally (see ws_timeline). No module-level cache remains.


def _build_timeline_composite(page: int, app_state, mark_ejects: bool = False, mark_thickness: int = 3) -> Optional[Tuple[bytes, dict]]:
    """Build the timeline composite JPEG + metadata (runs in executor thread).
    Returns (jpeg_bytes, metadata_dict) or None."""
    try:
        # Get config
        try:
            tl_config = app_state.timeline_config
            quality = tl_config.get('image_quality', 85)
            frames_per_page = tl_config.get('num_rows', 10)
            image_rotation = tl_config.get('image_rotation', 0)
            show_bbox = tl_config.get('show_bounding_boxes', True)
            obj_filters = tl_config.get('object_filters', {})
            procedures = tl_config.get('procedures', [])
        except Exception:
            tl_config = None
            quality = 85
            frames_per_page = 10
            image_rotation = 0
            show_bbox = True
            obj_filters = {}
            procedures = []
        # 3.21.26 — draw_filters loads audio_settings itself; nothing for us to do here.

        # Get current encoder for ejector marker
        try:
            watcher_inst = app_state.watcher_instance
            current_encoder = getattr(watcher_inst, 'encoder_value', None)
        except Exception:
            current_encoder = None
        ejector_offset = int(os.environ.get("EJECTOR_OFFSET", 0))

        redis_client = Redis("redis", 6379, db=cfg.REDIS_DB)
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

        camera_order = tl_config.get('camera_order', 'normal') if tl_config else 'normal'
        if camera_order == 'custom':
            custom_order_str = tl_config.get('custom_camera_order', '') if tl_config else ''
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
        cam_page_slices = {}
        num_columns = 0

        for cam_id in sorted_cam_ids:
            frames_raw = camera_frames_raw[cam_id]
            all_frames = []
            for frame_data in frames_raw:
                ts, jpeg_bytes, detections, meta = _unpack_timeline_entry(frame_data)
                all_frames.append((ts, jpeg_bytes, detections, meta))
            all_frames.sort(key=lambda x: x[0])

            total = len(all_frames)
            end_index = total - (safe_page * frames_per_page)
            start_index = max(0, end_index - frames_per_page)
            page_slice = all_frames[start_index:end_index] if end_index > 0 else []

            cam_page_slices[cam_id] = page_slice
            num_columns = max(num_columns, len(page_slice))

        # Build per-column metadata
        columns_meta = []
        for col_idx in range(num_columns):
            col_all_dets = []
            col_per_cam = {}   # 4.0.376 — per-camera dets, for per-camera eject attribution
            col_d_paths = {}
            col_encoder = None
            col_ts = None
            for cam_id in sorted_cam_ids:
                ps = cam_page_slices.get(cam_id, [])
                if col_idx < len(ps):
                    ts, jpeg_bytes, detections, meta = ps[col_idx]
                    if detections:
                        for _d in detections:
                            if isinstance(_d, dict):
                                _d.setdefault('_cam', cam_id)
                        col_all_dets.extend(detections)
                        col_per_cam[cam_id] = detections
                    col_d_paths[str(cam_id)] = meta.get('d_path') if meta else None
                    if meta and meta.get('encoder') is not None:
                        col_encoder = meta['encoder']
                    if col_ts is None:
                        col_ts = ts
            columns_meta.append({
                'all_detections': col_all_dets,
                'per_cam_dets': col_per_cam,
                'd_paths': col_d_paths,
                'encoder': col_encoder,
                'ts': col_ts,
                'is_capture': (safe_page == 0 and col_idx == num_columns - 1),
            })

        # Render camera rows
        camera_rows = []
        row_cam_ids = []   # 4.0.376 — cam_id for each camera_row (parallel list)
        thumb_width = 0
        thumb_height = 240

        for cam_id in sorted_cam_ids:
            page_slice = cam_page_slices.get(cam_id, [])
            row_frames = []
            for ts, jpeg_bytes, detections, meta in page_slice:
                thumb = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if thumb is not None:
                    if show_bbox and detections:
                        th, tw = thumb.shape[:2]
                        orig_h = meta.get('orig_h', th) if meta else th
                        orig_w = meta.get('orig_w', tw) if meta else tw
                        sx = tw / orig_w if orig_w else 1
                        sy = th / orig_h if orig_h else 1
                        kv_y = 4
                        for det in detections:
                            kv_y = draw_detection_on(
                                thumb, det, sx=sx, sy=sy, kv_y=kv_y, bbox_thickness=2,
                            )
                    if image_rotation == 90:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                    elif image_rotation == 180:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_180)
                    elif image_rotation == 270:
                        thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    row_frames.append(thumb)
                    if thumb_width == 0:
                        thumb_height, thumb_width = thumb.shape[:2]

            if row_frames:
                camera_rows.append(np.hstack(row_frames))
                row_cam_ids.append(cam_id)

        if not camera_rows:
            return None

        # Build header strip
        header = _build_header_strip(columns_meta, thumb_width, num_columns, procedures, current_encoder, ejector_offset)

        max_width = max(row.shape[1] for row in camera_rows)
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
        # 4.0.378 — "mark ejected frames" is a SERVER-SIDE timeline_config setting read in
        # the same pass that draws the red/green header strip, so the box appears at the
        # same moment the strip goes red — no dependency on a client WS flag that a
        # reconnect could drop (the "box appears after a while" bug). Legacy client flag
        # still forces it on. Mirrors /timeline_image.
        _tlc = tl_config or {}
        mark_on = bool(mark_ejects) or bool(_tlc.get('mark_ejects_enabled', False))
        if mark_on and thumb_width > 0:
            comp_h, comp_w = composite.shape[:2]
            box_th = min(12, max(1, int(_tlc.get('mark_ejects_thickness', mark_thickness) or 3)))
            # 4.0.376 — box the EXACT camera frame that fired the ejection (dets carry
            # `_cam`), not the whole column. Fall back to a full-column box only when no
            # single camera triggers (aggregate-count rule). Mirrors timeline.py.
            row_bands = []
            _y = _HEADER_HEIGHT
            for _idx, _r in enumerate(camera_rows):
                _cid = row_cam_ids[_idx] if _idx < len(row_cam_ids) else None
                row_bands.append((_cid, _y, _r.shape[0]))
                _y += _r.shape[0]
            for col_idx, col in enumerate(columns_meta):
                if col.get('should_eject') is not True:
                    continue
                x0 = col_idx * thumb_width + 1
                x1 = min((col_idx + 1) * thumb_width - 1, comp_w - 1)
                if x1 <= x0:
                    continue
                per_cam = col.get('per_cam_dets') or {}
                triggering = set()
                for _cid, _dets in per_cam.items():
                    try:
                        _ej, _ = evaluate_eject_from_detections(_dets, procedures)
                    except Exception:
                        _ej = False
                    if _ej:
                        triggering.add(_cid)
                if triggering:
                    for (_cid, ry, rh) in row_bands:
                        if _cid in triggering:
                            y0 = ry + 1
                            y1 = min(ry + rh - 2, comp_h - 1)
                            if y1 > y0:
                                cv2.rectangle(composite, (x0, y0), (x1, y1), (0, 0, 255), box_th)
                else:
                    cv2.rectangle(composite, (x0, _HEADER_HEIGHT + 1),
                                  (x1, comp_h - 2), (0, 0, 255), box_th)
        _, jpeg = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # Build metadata for frontend
        meta_columns = []
        for i, cm in enumerate(columns_meta):
            should_eject = False
            eject_reasons = []
            if procedures and cm['all_detections']:
                should_eject, eject_reasons = evaluate_eject_from_detections(cm['all_detections'], procedures)
            meta_columns.append({
                "index": i,
                "ts": cm['ts'],
                "encoder": cm['encoder'],
                "should_eject": should_eject,
                "eject_reasons": eject_reasons,
                "d_paths": cm['d_paths'],
            })

        metadata = {
            "type": "timeline_meta",
            "columns": meta_columns,
            "thumb_width": thumb_width,
            "thumb_height": thumb_height,
            "header_height": _HEADER_HEIGHT,
            "num_cameras": len(sorted_cam_ids),
            "cam_ids": sorted_cam_ids,
            "page": safe_page,
            "total_pages": total_pages,
        }

        return jpeg.tobytes(), metadata

    except Exception as e:
        logger.error(f"WS timeline composite error: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# /ws/timeline — event-driven timeline push
# ---------------------------------------------------------------------------
@router.websocket("/ws/timeline")
async def ws_timeline(websocket: WebSocket):
    app_state = websocket.app.state
    await manager.connect("timeline", websocket)
    client_page = 0
    # 4.0.377 — seed mark/thickness from the connect query string so the FIRST frame
    # after any (re)connect already carries the right box state — no no-box flash.
    try:
        _qp = websocket.query_params
        client_mark = (_qp.get('mark') == '1')
        client_thick = min(12, max(1, int(_qp.get('thick', 3))))
    except Exception:
        client_mark = False
        client_thick = 3

    loop = asyncio.get_event_loop()

    # 4.0.377 — the red box "flickered" on/off because every connection's Redis loop
    # BROADCAST its render (built with ITS OWN mark state) to ALL timeline clients via
    # a shared global cache: a second tab or a reconnect-overlap with mark=OFF would
    # broadcast a no-box frame to everyone, then the mark=ON loop re-broadcast boxes,
    # alternating at the Redis tick rate. Now each connection renders with its OWN
    # mark/thickness and sends ONLY to itself, tracking its own last-sent state — no
    # shared cache, no cross-connection broadcast, no flicker.
    last = {"version": -1, "page": None, "mark": None, "thick": None}

    async def _render_and_send_self():
        result = await loop.run_in_executor(
            None, _build_timeline_composite, client_page, app_state, client_mark, client_thick
        )
        if result:
            jpeg, meta = result
            last["version"] = cfg.timeline_frame_counter
            last["page"] = client_page
            last["mark"] = client_mark
            last["thick"] = client_thick
            await manager.send_text(websocket, json.dumps(meta))
            await manager.send_binary(websocket, jpeg)

    try:
        # Immediate frame on connect (mark/thickness already known from the query string)
        await _render_and_send_self()

        # Start two concurrent tasks: listen for client messages + subscribe to Redis updates
        async def _listen_client():
            """Receive page navigation + mark-ejects toggle (msg = "page" or "page|mark|thickness")."""
            nonlocal client_page, client_mark, client_thick
            while True:
                try:
                    msg = await websocket.receive_text()
                    parts = str(msg).split('|')
                    client_page = int(parts[0])
                    client_mark = (len(parts) > 1 and parts[1] == '1')
                    if len(parts) > 2:
                        try:
                            client_thick = min(12, max(1, int(parts[2])))
                        except (ValueError, TypeError):
                            client_thick = 3
                    await _render_and_send_self()
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception:
                    pass

        async def _listen_redis():
            """Subscribe to timeline updates from inference workers via Redis Pub/Sub."""
            nonlocal client_page, client_mark, client_thick

            def _wait_for_message(pubsub, timeout=1.0):
                """Blocking Redis get_message — runs in executor thread."""
                return pubsub.get_message(timeout=timeout)

            r = Redis("redis", 6379, db=cfg.REDIS_DB)
            pubsub = r.pubsub()
            pubsub.subscribe("ws:timeline_update")

            try:
                while True:
                    # Wait for Redis notification (blocking in executor)
                    msg = await loop.run_in_executor(None, _wait_for_message, pubsub, 2.0)
                    if msg and msg["type"] == "message":
                        current_version = cfg.timeline_frame_counter
                        # Rebuild for THIS connection only when its own view changed.
                        if (current_version != last["version"]
                                or client_page != last["page"]
                                or client_mark != last["mark"]
                                or client_thick != last["thick"]):
                            await _render_and_send_self()
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
