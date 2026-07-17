"""ArduinoSocket — central coordinator for serial, camera, and Redis communication.

Manages:
- Serial communication with Arduino/watcher hardware
- Camera capture and frame management
- Redis-based frame queue for inference pipeline
- Ejector control based on encoder position
- Stream results visualization
- Production metrics to TimescaleDB
- Barcode scanner input
"""

import cv2
import time
import json
import os
import shutil
import queue
import numpy as np
import serial
import requests
import threading
import logging
from datetime import datetime
from typing import Dict, Any, List

import config as cfg
from services.camera import (
    CameraBuffer, DETECTED_CAMERAS, detect_video_devices,
    CAM_1_PATH, CAM_2_PATH, CAM_3_PATH, CAM_4_PATH,
    apply_camera_config_from_saved,
)
from redis import Redis as DirectRedis
from services.redis_service import RedisConnection
from services.db import write_production_metrics_to_db
from services.detection import create_dynamic_images

logger = logging.getLogger(__name__)

# Module-level runtime references (set from main.py after initialization)
_state_manager = None
_app = None

# Background disk write queue — cv2.imwrite runs here to unblock capture loop
# Each item is a raw numpy array from cv2 (~3 MB for 1280x720 USB, ~6.9 MB
# for Basler acA1920 1920x1200 uint8 BGR).
#
# v4.0.123 — size from AVAILABLE RAM, not TOTAL, matching the v4.0.83 hot-
# queue lesson (containers on a shared box already consume a big chunk of
# TOTAL). Per-item budget and RAM percentage are env-driven so 4K / large-
# sensor sites can shrink the slot count without losing bytes-of-budget.
# FLOOR 500 (up from 200) gives usable burst headroom even on tiny boxes;
# CEILING 4000 prevents an oversized queue on a 128+ GB host.
import psutil as _psutil
_total_ram_gb = _psutil.virtual_memory().total / (1024 ** 3)
_avail_ram_gb = _psutil.virtual_memory().available / (1024 ** 3)
_DISK_QUEUE_PCT_AVAIL = float(os.environ.get("MVE_DISK_QUEUE_PCT_AVAIL", "20"))
_DISK_QUEUE_ITEM_MB   = float(os.environ.get("MVE_DISK_QUEUE_ITEM_MB",   "3"))
_DISK_QUEUE_FLOOR     = int(os.environ.get("MVE_DISK_QUEUE_FLOOR",       "500"))
_DISK_QUEUE_CEILING   = int(os.environ.get("MVE_DISK_QUEUE_CEILING",     "4000"))
_disk_queue_ideal = int(_avail_ram_gb * 1024 * _DISK_QUEUE_PCT_AVAIL / 100 / max(0.1, _DISK_QUEUE_ITEM_MB))
_disk_queue = queue.Queue(maxsize=max(_DISK_QUEUE_FLOOR, min(_DISK_QUEUE_CEILING, _disk_queue_ideal)))

# ── Two-tier inference queue: Hot (RAM) + Cold (Disk) ──
#
# Hot queue (RAM, LIFO): sized by RAM budget. When full, frames spill to cold.
#   Workers pop newest first so ejector gets the freshest decision.
# Cold queue (Disk, FIFO): practically unlimited. Frames spill here when hot is full
#   or when YOLO fails (after fail-safe eject). Processed oldest-first for historical data.
# No frame is ever dropped.

import collections
import pickle
import glob as _glob_mod

# v4.0.83 — Auto-size the hot RAM queue based on AVAILABLE memory rather than total.
# Old formula (fixed 5% of TOTAL RAM @ 500KB/frame) gave 3190 slots on a 31GB box —
# too small for high-FPS bursts (Cap FPS throttled to 39 fps once queue filled, then
# capture blocked on hot_q.put() until inference workers drained the backlog).
#
# New formula scales with what the OS actually has free at boot time. Reserves a
# headroom the operator controls (default 60% of available RAM stays free — i.e.
# the queue consumes AT MOST 40%). Frame size and both floor/ceiling are tunable
# per site via env.
#
# Env knobs:
#   MVE_HOT_QUEUE_RAM_PCT  — % of AVAILABLE RAM the queue may consume (default 40)
#   MVE_AVG_FRAME_KB       — planning size per frame (default 300)
#   MVE_HOT_QUEUE_FLOOR    — minimum slots regardless of RAM (default 1000)
#   MVE_HOT_QUEUE_CEILING  — maximum slots regardless of RAM (default 60000)
#
# On a 31 GB box with ~16 GB free: 16 * 1024 * 0.40 / 0.3 ≈ 21,800 slots
# (vs ~3,190 slots under the old fixed formula). Cap FPS bursts of ~100 fps have
# ~4 minutes of headroom before the queue caps out — plenty for the drain to
# catch up under real production loads.
_avail_ram_bytes = _psutil.virtual_memory().available
_pct = float(os.environ.get('MVE_HOT_QUEUE_RAM_PCT', '40'))
_frame_kb = float(os.environ.get('MVE_AVG_FRAME_KB', '300'))
_floor = int(os.environ.get('MVE_HOT_QUEUE_FLOOR', '1000'))
_ceiling = int(os.environ.get('MVE_HOT_QUEUE_CEILING', '60000'))
_ram_budget_raw = int(_avail_ram_bytes * _pct / 100.0 / (_frame_kb * 1024))
_ram_budget = max(_floor, min(_ceiling, _ram_budget_raw))
logger.info(
    f"hot_queue sizing: avail_ram={_avail_ram_bytes/(1024**3):.1f}GB * {_pct}% "
    f"/ {_frame_kb}KB/frame = {_ram_budget_raw} raw, clamped to {_ram_budget} "
    f"(floor={_floor}, ceiling={_ceiling})"
)


class HotQueue:
    """RAM-based LIFO queue for ejector-critical frames."""

    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self._deque = collections.deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Event()

    def put(self, item):
        """Add frame to hot queue. Returns False if full (caller should spill to cold)."""
        with self._lock:
            if len(self._deque) >= self.maxsize:
                return False
            item['_enqueue_t'] = time.time()
            self._deque.append(item)
            self._not_empty.set()
            return True

    def get(self, timeout=None):
        """LIFO pop (newest first). Raises queue.Empty on timeout."""
        deadline = time.time() + timeout if timeout else None
        while True:
            with self._lock:
                if self._deque:
                    item = self._deque.pop()  # LIFO — newest first for ejector
                    if not self._deque:
                        self._not_empty.clear()
                    return item
            if deadline and time.time() >= deadline:
                raise queue.Empty()
            self._not_empty.wait(timeout=0.05)

    def qsize(self):
        return len(self._deque)


class ColdDiskQueue:
    """Disk-based FIFO queue for historical frame processing. No frame is ever dropped.

    Uses an in-memory sorted index (SortedList) for O(log n) put/get instead of
    globbing the entire directory on every get() call.  Thread-safe: the lock
    protects both the index and the filesystem read/delete so no two threads can
    race for the same file.

    4.0.64 — SIZE-BOUNDED. Previously the queue was unbounded on disk; when the
    inference pipeline couldn't keep up, this directory could grow into the
    tens or hundreds of GBs and eventually take the whole disk down. On
    vteam12 this ate 97GB and killed the container. Now the queue tracks its
    on-disk byte total and evicts oldest-first whenever it exceeds the cap
    (COLD_QUEUE_MAX_BYTES env var, default 5 GB). Periodic staleness sweep
    (COLD_QUEUE_MAX_AGE_SECONDS env var, default 600s) runs from a background
    thread every 5 min so stale frames don't accumulate even below the cap.
    """

    def __init__(self, directory="cold_queue"):
        self._dir = directory
        os.makedirs(self._dir, exist_ok=True)
        self._lock = threading.Lock()
        # Build in-memory index from any leftover files (sorted = FIFO order)
        existing = sorted(_glob_mod.glob(os.path.join(self._dir, '*.pkl')))
        self._index = collections.deque(existing)   # deque of full paths, FIFO order
        # 4.0.64 — track on-disk bytes for the size cap. Compute once at boot
        # from leftover files then maintain incrementally on put/get/evict.
        self._bytes = 0
        for p in existing:
            try:
                self._bytes += os.path.getsize(p)
            except OSError:
                pass
        try:
            self._max_bytes = int(os.environ.get("COLD_QUEUE_MAX_BYTES", 5 * 1024 * 1024 * 1024))
        except (TypeError, ValueError):
            self._max_bytes = 5 * 1024 * 1024 * 1024   # 5 GB
        try:
            self._max_age_s = int(os.environ.get("COLD_QUEUE_MAX_AGE_SECONDS", 600))
        except (TypeError, ValueError):
            self._max_age_s = 600
        if self._index:
            logger.info(
                f"Cold queue: {len(self._index)} leftover frames "
                f"({self._bytes / (1024**3):.2f} GB) from previous run"
                f"; cap={self._max_bytes / (1024**3):.1f} GB, "
                f"stale-age={self._max_age_s}s"
            )

    def _evict_to_cap_locked(self):
        """Called with self._lock held. Drops oldest files until under cap."""
        evicted = 0
        while self._bytes > self._max_bytes and self._index:
            oldest = self._index.popleft()
            try:
                self._bytes -= os.path.getsize(oldest)
            except OSError:
                pass
            try:
                os.remove(oldest)
            except OSError:
                pass
            evicted += 1
        if evicted:
            logger.warning(
                f"Cold queue: evicted {evicted} oldest frame batches to stay "
                f"under {self._max_bytes / (1024**3):.1f} GB cap "
                f"(now {self._bytes / (1024**3):.2f} GB, {len(self._index)} batches)"
            )

    def put(self, item):
        """Serialize frame batch to disk. Never fails (unless disk is full)."""
        item['_enqueue_t'] = item.get('_enqueue_t', time.time())
        fname = f"{item['_enqueue_t']:.6f}_{item.get('encoder', 0)}.pkl"
        path = os.path.join(self._dir, fname)
        try:
            with open(path, 'wb') as f:
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
            with self._lock:
                self._index.append(path)
                try:
                    self._bytes += os.path.getsize(path)
                except OSError:
                    pass
                # 4.0.64 — enforce cap on every put so we never overshoot.
                self._evict_to_cap_locked()
        except Exception as e:
            logger.error(f"Cold queue write failed: {e}")

    def get(self):
        """FIFO: pop oldest path from index, deserialize, delete. Returns None if empty.

        Thread-safe: only one thread can claim a file at a time.
        """
        with self._lock:
            if not self._index:
                return None
            path = self._index.popleft()
            # 4.0.64 — decrement byte accounting on successful pop. Even if
            # the read below fails, the file gets removed, so we account here.
            try:
                self._bytes -= os.path.getsize(path)
                if self._bytes < 0:
                    self._bytes = 0
            except OSError:
                pass

        # Read + delete outside the lock (I/O can be slow)
        try:
            with open(path, 'rb') as f:
                item = pickle.load(f)
            os.remove(path)
            return item
        except Exception as e:
            logger.error(f"Cold queue read failed ({path}): {e}")
            # Remove corrupt file if it still exists
            try:
                os.remove(path)
            except Exception:
                pass
            return None

    def qsize(self):
        with self._lock:
            return len(self._index)

    def size_bytes(self):
        """4.0.64 — expose on-disk byte total so /api/status can show it."""
        with self._lock:
            return int(self._bytes)

    def flush_stale(self, max_age_seconds=None):
        """Remove all files older than max_age_seconds from the queue.

        Called at startup AND periodically from a background sweep thread.
        Defaults to the instance's max_age_s (env-configurable).

        4.0.66 — SNAPSHOT the index under the lock, then do all disk I/O
        (getsize + remove) OUTSIDE the lock. Previously we held _lock for
        the whole scan, which meant every put() call from detection worker
        threads at 100 Hz blocked for as long as flush took — on vteam12
        with thousands of stale files, that was several seconds. Meanwhile
        anyone else calling size_bytes() / qsize() (including /api/cold_queue)
        also blocked, freezing uvicorn's event loop. Same pattern as get()
        (I/O outside the lock).
        """
        if max_age_seconds is None:
            max_age_seconds = self._max_age_s
        now = time.time()
        flushed = 0
        freed_bytes = 0
        with self._lock:
            paths_snapshot = list(self._index)
        stale_paths = set()
        remaining = collections.deque()
        for path in paths_snapshot:
            try:
                fname = os.path.basename(path)
                ts = float(fname.split('_')[0])
                if now - ts > max_age_seconds:
                    try:
                        freed_bytes += os.path.getsize(path)
                    except OSError:
                        pass
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                    stale_paths.add(path)
                    flushed += 1
                    continue
            except (ValueError, IndexError):
                pass
            remaining.append(path)
        # Re-acquire the lock briefly to reconcile with any concurrent put()
        # calls that ran while we were doing disk I/O. Preserve every entry
        # they added; drop the stale entries we identified above.
        with self._lock:
            snapshot_set = set(paths_snapshot)
            live_new = [p for p in self._index if p not in snapshot_set]
            self._index = remaining
            for p in live_new:
                self._index.append(p)
            self._bytes -= freed_bytes
            if self._bytes < 0:
                self._bytes = 0
        if flushed:
            logger.info(
                f"Cold queue: flushed {flushed} stale frames "
                f"(>{max_age_seconds}s old, freed {freed_bytes / (1024**2):.1f} MB)"
            )
        return flushed

    def start_janitor(self, interval_s=300):
        """4.0.64 — start a background thread that runs flush_stale() every
        `interval_s` seconds so stale frames don't accumulate between reboots.
        No-op if already started. Daemon thread, safe to leak on shutdown."""
        if getattr(self, "_janitor_started", False):
            return
        self._janitor_started = True
        def _loop():
            while True:
                try:
                    time.sleep(interval_s)
                    self.flush_stale()
                except Exception as _e:
                    logger.error(f"Cold queue janitor: {_e}")
        threading.Thread(target=_loop, daemon=True, name="cold-queue-janitor").start()
        logger.info(f"Cold queue janitor started (interval {interval_s}s)")


class InferenceQueueFacade:
    """Unified producer interface: tries hot first, spills to cold. Never drops."""

    def __init__(self, hot, cold):
        self.hot = hot
        self.cold = cold

    def put(self, item, timeout=None):
        """Add frame — hot queue first, spill to cold disk if hot is full."""
        # v4.0.90 — temporary INFO-level log so we can see the spill rate.
        # Was DEBUG (hidden by default log level). Revert to DEBUG once we
        # understand why cold grows while hot=0.
        if not self.hot.put(item):
            logger.warning(f"Hot queue full ({self.hot.qsize()}/{self.hot.maxsize}) — spilling to cold disk queue")
            self.cold.put(item)

    def stats(self):
        """Return (hot_count, cold_count) for dashboard."""
        return self.hot.qsize(), self.cold.qsize()

    @property
    def maxsize(self):
        return self.hot.maxsize

    def qsize(self):
        return self.hot.qsize() + self.cold.qsize()


# Initialize with defaults immediately so capture threads don't hit None.
# init_queues() re-initializes with correct sizing after cameras + ejector config are known.
_hot_queue = HotQueue(maxsize=max(10, _ram_budget))
_cold_queue_disk = ColdDiskQueue(directory="/tmp/cold_queue")
_inference_queue = InferenceQueueFacade(_hot_queue, _cold_queue_disk)


def init_queues(num_cameras, ejector_offset, ejector_enabled):
    """Initialize hot (RAM) + cold (disk) inference queues.

    Hot queue uses the full RAM budget. When hot fills up, frames spill to cold (disk).
    """
    global _hot_queue, _cold_queue_disk, _inference_queue
    hot_max = max(10, _ram_budget)
    _hot_queue = HotQueue(maxsize=hot_max)
    _cold_queue_disk = ColdDiskQueue(directory="/tmp/cold_queue")
    _inference_queue = InferenceQueueFacade(_hot_queue, _cold_queue_disk)
    logger.info(
        f"Inference queues initialized: hot={hot_max} frames (RAM), "
        f"cold=disk (/tmp/cold_queue/), ejector={'ON' if ejector_enabled else 'OFF'}, "
        f"cameras={num_cameras}, offset={ejector_offset}"
    )
_disk_writers_count = 0
_disk_lock = threading.Lock()

# v4.0.103 / v4.0.105 — env-driven pressure-based retention. When the host
# disk crosses `_DISK_MAX_PCT`, `_ensure_disk_space` deletes oldest hourly
# chunks (skipping the current hour) until back under it. Historical
# hardcode was 75; now tuneable via env.
#
# v4.0.105 CORRECTION: the age-based janitor added in v4.0.103 was WRONG
# design. Files retention MUST be disk-usage based only, not time-based.
# Reason: on smaller/slower factories a shipment can legitimately span
# weeks; deleting its raw_images at 30 days by clock would destroy
# audit evidence the operator still needed. Pressure-based cleanup
# handles capacity correctly regardless of age. The DB gets the same
# treatment via `_db_disk_pressure_janitor_loop` in services/db.py.
_DISK_MAX_PCT = int(os.environ.get("RAW_IMAGES_MAX_DISK_PCT", "75"))
# v4.0.121 — proactive margin below the hard cap. Janitor thread starts
# evicting at (_DISK_MAX_PCT - _DISK_EVICT_MARGIN_PCT) so writers ALWAYS
# have runway. Default 5 → evict starts at 70 %, cap remains 75 %. This
# is the buffer that guarantees writers never see ENOSPC in practice.
_DISK_EVICT_MARGIN_PCT = int(os.environ.get("DISK_EVICT_MARGIN_PCT", "5"))
_last_disk_ok = True
_last_disk_check_t = 0
_raw_root = None                # Resolved once on first write

# v4.0.121 — event the writer can set to WAKE the janitor immediately
# (used on a rare ENOSPC retry path in _disk_writer_loop).
_disk_janitor_event = threading.Event()
_disk_janitor_started = False
_disk_janitor_start_lock = threading.Lock()

def _resolve_raw_root(some_path):
    """Walk up from a write path to find the raw_images root."""
    global _raw_root
    if _raw_root:
        return _raw_root
    p = some_path
    while p and os.path.basename(p) != "raw_images":
        p = os.path.dirname(p)
    _raw_root = p or "raw_images"
    return _raw_root

def _sorted_chunks():
    """Return all hourly chunk dirs across all shipments, oldest first."""
    root = _raw_root or "raw_images"
    chunks = []
    try:
        for shipment in os.listdir(root):
            sp = os.path.join(root, shipment)
            if not os.path.isdir(sp):
                continue
            for chunk in os.listdir(sp):
                cp = os.path.join(sp, chunk)
                if os.path.isdir(cp) and len(chunk) >= 13 and chunk[4] == '-' and '_' in chunk:
                    chunks.append((chunk, cp))
    except OSError:
        pass
    chunks.sort(key=lambda x: x[0])
    return chunks

def _ensure_disk_space(write_dir):
    """DVR-style ring buffer: if disk > 75%, delete oldest hour chunk. Never stop writing."""
    global _last_disk_ok, _last_disk_check_t
    _resolve_raw_root(write_dir)

    now = time.time()
    if now - _last_disk_check_t < 2 and _last_disk_ok:
        return

    try:
        usage = shutil.disk_usage(write_dir)
        # Use (used + free) as denominator to match df's view (excludes ext4's
        # 5%-reserved-for-root blocks). Using usage.total here understates pct
        # by ~5–10 points and made DVR cleanup silently never trigger when
        # df-disk was already at 80% (bug found 2026-05-29 on vteam19).
        pct = usage.used * 100 // max(1, usage.used + usage.free)
        _last_disk_check_t = now
    except OSError:
        return

    if pct <= _DISK_MAX_PCT:
        _last_disk_ok = True
        return

    # Over limit — delete oldest chunks until under 75%
    _last_disk_ok = False
    current_hour = datetime.now().strftime("%Y-%m-%d_%H")
    for chunk_name, chunk_path in _sorted_chunks():
        if chunk_name >= current_hour:
            break  # Never delete current hour
        shutil.rmtree(chunk_path, ignore_errors=True)
        logger.info(f"DVR cleanup: deleted {chunk_path}")
        try:
            # v4.0.108 — MUST use the same (used + free) denominator as the
            # entry check on line 458. Previously this used `.total` which
            # includes ext4's 5%-reserved-for-root blocks, silently under-
            # reporting pct by ~5 pts. Symptom on khoy (SSD-RESERVE 220 GB):
            # entry sees 79% -> evict; loop check sees 74% -> stops; disk
            # regrows to 90% before next cycle. Eviction thrashed 159x in
            # 3 min without holding the disk anywhere near the threshold.
            _u = shutil.disk_usage(write_dir)
            pct = _u.used * 100 // max(1, _u.used + _u.free)
            if pct <= _DISK_MAX_PCT:
                _last_disk_ok = True
                logger.info(f"DVR cleanup done — disk at {pct}%")
                break
        except OSError:
            break


# v4.0.105 — REVERTED. The age-based raw_images janitor added in v4.0.103
# was the wrong design. Files retention is DISK-USAGE based (the reactive
# `_ensure_disk_space` at `_DISK_MAX_PCT`), never time based, because on
# smaller/slower factories one shipment can legitimately span weeks and
# deleting its raw_images by clock would destroy audit evidence.
# The corresponding DB retention now also uses disk pressure — see the
# `_db_disk_pressure_janitor_loop` in services/db.py.


# v4.0.121 — disk-pressure JANITOR THREAD. Runs continuously in the
# background so that `_ensure_disk_space` no longer needs to fire from
# the disk-writer hot path. Keeps disk under `_DISK_MAX_PCT` by evicting
# oldest hourly chunks, and starts eviction PROACTIVELY at
# `_DISK_MAX_PCT - _DISK_EVICT_MARGIN_PCT` (default 70 %) so writers
# always have runway — a full disk should never actually happen.
#
# Why this design change: v3.2.0 put `_ensure_disk_space` inside
# `_disk_writer_loop` on the theory that the check is "cheap when
# healthy" (2-second cache short-circuits below the threshold). That
# assumption breaks once disk crosses the threshold: `_last_disk_ok`
# flips False and the cache is bypassed, so every write pays the full
# scan + rmtree cost. Worse, all N writer threads race into
# `_sorted_chunks()` and `shutil.rmtree` simultaneously with no lock.
# On khoy at 86 % this produced 3,738 queue-full events / 30 min.
#
# Contract preserved: DVR "never stop writing", same `_DISK_MAX_PCT`
# threshold semantics, same eviction algorithm (oldest hourly chunk,
# skip current hour). Just moved off the writer hot path.
def _disk_pressure_janitor_loop():
    """Runs forever. Adaptive tick: 500 ms while under pressure, 2 s while idle."""
    global _last_disk_ok, _raw_root
    logger.info(
        f"disk-pressure janitor started "
        f"(evict_at={_DISK_MAX_PCT - _DISK_EVICT_MARGIN_PCT}% target={_DISK_MAX_PCT}%)"
    )
    while True:
        try:
            root = _raw_root or "raw_images"
            # Only measure once we have a real raw_root — resolved by
            # the first _resolve_raw_root call from a writer. Until then
            # sleep and re-check.
            if not os.path.isdir(root):
                _disk_janitor_event.wait(timeout=2.0)
                _disk_janitor_event.clear()
                continue
            try:
                u = shutil.disk_usage(root)
                pct = u.used * 100 // max(1, u.used + u.free)
            except OSError:
                _disk_janitor_event.wait(timeout=2.0)
                _disk_janitor_event.clear()
                continue
            evict_start = max(1, _DISK_MAX_PCT - _DISK_EVICT_MARGIN_PCT)
            if pct <= evict_start:
                _last_disk_ok = True
                _disk_janitor_event.wait(timeout=2.0)
                _disk_janitor_event.clear()
                continue
            # Under pressure. Evict oldest chunks until back under evict_start.
            _last_disk_ok = False
            current_hour = datetime.now().strftime("%Y-%m-%d_%H")
            for chunk_name, chunk_path in _sorted_chunks():
                if chunk_name >= current_hour:
                    break  # never touch the current hour
                shutil.rmtree(chunk_path, ignore_errors=True)
                logger.info(f"DVR cleanup (janitor): deleted {chunk_path}")
                try:
                    u2 = shutil.disk_usage(root)
                    pct = u2.used * 100 // max(1, u2.used + u2.free)
                    if pct <= evict_start:
                        _last_disk_ok = True
                        logger.info(f"DVR cleanup done — disk at {pct}%")
                        break
                except OSError:
                    break
            # Fast tick under continued pressure, otherwise back off.
            _disk_janitor_event.wait(timeout=0.5 if not _last_disk_ok else 2.0)
            _disk_janitor_event.clear()
        except Exception as _e:
            logger.error(f"disk-pressure janitor error: {_e}")
            time.sleep(2.0)


def start_disk_pressure_janitor():
    """Idempotent — safe to call from main.py startup."""
    global _disk_janitor_started
    with _disk_janitor_start_lock:
        if _disk_janitor_started:
            return
        threading.Thread(
            target=_disk_pressure_janitor_loop,
            daemon=True,
            name="disk-pressure-janitor",
        ).start()
        _disk_janitor_started = True


def _disk_writer_loop():
    """Background thread: checks disk space (NVR-style), then writes image.

    4.0.52 — uses services.jpeg_codec.imwrite_jpeg, which encodes via
    libjpeg-turbo when available (~3× faster on modern x86). Falls back
    to cv2.imwrite when the library is missing. The encode is the CPU
    hotspot on this thread pool — replacing it lets the same # of workers
    absorb 3× the framerate before the queue backs up.
    """
    from services.jpeg_codec import imwrite_jpeg
    _q = cfg.RAW_IMAGE_JPEG_QUALITY if hasattr(cfg, 'RAW_IMAGE_JPEG_QUALITY') else 85
    while True:
        try:
            path, frame = _disk_queue.get()
            # v4.0.121 — no more per-write _ensure_disk_space. The
            # disk-pressure janitor thread evicts oldest chunks in the
            # background at (75% - 5%) = 70% so writers always have
            # runway. Ensure raw_root is resolved (janitor needs it).
            _resolve_raw_root(os.path.dirname(path))
            if not imwrite_jpeg(path, frame, quality=_q):
                # Write failed — most likely ENOSPC. Wake the janitor
                # so it evicts immediately, then retry the frame once.
                _disk_janitor_event.set()
                time.sleep(0.1)
                if not imwrite_jpeg(path, frame, quality=_q):
                    logger.warning(f"jpeg_codec.imwrite_jpeg failed twice for {path}")
        except Exception as e:
            logger.error(f"Disk write error: {e}")

def add_disk_writers(count: int):
    """Add disk writer threads. Thread-safe, callable at any time by autoscaler.

    Args:
        count: Number of NEW threads to add (not total target).
    """
    global _disk_writers_count
    if count <= 0:
        return
    with _disk_lock:
        start_idx = _disk_writers_count
        for i in range(count):
            threading.Thread(
                target=_disk_writer_loop, daemon=True,
                name=f"disk-writer-{start_idx + i + 1}"
            ).start()
        _disk_writers_count += count
        logger.info(f"[DiskWriters] +{count} threads, total now {_disk_writers_count}")


# Persistent Redis connection for capture FPS timestamps (db=cfg.REDIS_DB, reused across captures)
_cap_redis = None

def _get_cap_redis():
    """Get or create persistent Redis connection for capture timestamps."""
    global _cap_redis
    if _cap_redis is None:
        _cap_redis = DirectRedis("redis", 6379, db=cfg.REDIS_DB)
    return _cap_redis


def set_state_manager(sm):
    """Set the state manager reference (called from main.py after both are created)."""
    global _state_manager
    _state_manager = sm


def set_app(app):
    """Set the FastAPI app reference (called from main.py)."""
    global _app
    _app = app


class ArduinoSocket:
    def __init__(self, camera_paths: List[str] = None, camera_configs: dict = None,
                 serial_port=None, serial_baudrate=None):
        """Initialize ArduinoSocket with dynamic camera support.

        Args:
            camera_paths: List of video device paths to use. If None, uses auto-detected cameras.
            camera_configs: Dict of saved per-camera settings from config file, keyed by cam_id string.
            serial_port: Serial port path (default from env)
            serial_baudrate: Serial baud rate (default from env)
        """
        # Use configurable serial settings
        self.serial_port = serial_port or cfg.WATCHER_USB
        self.serial_baudrate = serial_baudrate or cfg.SERIAL_BAUDRATE
        self.serial_mode = cfg.SERIAL_MODE
        self.serial_available = False

        # Try to initialize serial, but continue without it if unavailable
        try:
            self.serial = serial.Serial(self.serial_port, self.serial_baudrate, 8, 'N', 1, timeout=1)
            self.serial.flushInput()  # clear input serial buffer
            self.serial.flushOutput()  # clear output serial buffer
            self.serial_available = True
            logger.info(f"Serial connected: {self.serial_port} @ {self.serial_baudrate}")
        except Exception as e:
            self.serial = None
            logger.warning(f"Serial not available ({self.serial_port}): {e}. Running in camera-only mode.")

        self.encoder = 0
        self.stop_thread = False
        self.redis_connection = RedisConnection(cfg.REDIS_HOST, cfg.REDIS_PORT)
        self.last_encoder_value = 0
        self.encoder_value = 0
        # 4.0.54 — Length tracking. Snapshot of encoder_value at the moment a
        # new shipment starts. Dashboard "Length" tile shows the delta
        # (encoder_value - shipment_start_encoder), i.e. how far the conveyor
        # has moved since this shipment began. Persisted alongside
        # current_shipment so it survives restart. Zero on "no_shipment".
        self.shipment_start_encoder = 0
        # 4.0.98 — Length persistence for encoder-reset resilience. Every N
        # seconds we snapshot `(shipment, shipment_start_encoder, last_seen_length)`
        # to a small JSON file (bind-mounted alongside .env.prepared_query_data).
        # On boot / after a config restore, we check: if current_encoder <
        # persisted_shipment_start (encoder was reset by PLC reboot / power cut /
        # MVE recreate), we rebuild shipment_start_encoder so length continues
        # from where it left off — no more negative length spikes. Guarded by
        # `_length_state_lock` so the periodic writer and the shipment-change
        # writer can't race on the file.
        self._length_state_path = "/code/.env.length_state.json"
        self._length_state_lock = threading.Lock()
        self._length_state_last_saved = 0.0
        self.health_check = False
        self.is_moving = False
        self.step = 15
        # Extended watcher-style metrics (used in new serial mode)
        self.pulses_per_second = 0
        self.pulses_per_minute = 0
        self.downtime_seconds = 0
        self.ok_counter = 0
        self.ng_counter = 0
        self.eject_ok_counter = 0   # Software: frames that passed (no eject)
        self.eject_ng_counter = 0   # Software: frames that triggered eject
        self.status_value = 0
        self.u_status = False
        self.b_status = False
        self.warning_status = False
        # Extra analog / power metrics (ANG / PWR in watcher)
        self.analog_value = 0
        self.power_value = 0
        # Verbose configuration values (mirror watcher verbose data)
        self.ok_offset_delay = 0      # OOD
        self.ok_duration_pulses = 0   # ODP
        self.ok_duration_percent = 0  # ODL
        self.ok_encoder_factor = 0    # OEF
        self.ng_offset_delay = 0      # NOD
        self.ng_duration_pulses = 0   # NDP
        self.ng_duration_percent = 0  # NDL
        self.ng_encoder_factor = 0    # NEF
        self.external_reset = 0       # EXT
        self.baud_rate = self.serial_baudrate  # BUD
        self.downtime_threshold = 0   # DWT
        self.last_speed_point = 0
        self.last_speed_time = time.time()
        self.last_capture_encoder = self.step * -1
        self.last_encoder_value_captured = 0
        self.take = False
        self.d_or_k = 0
        self.last_d_or_k = time.time()
        self.on_or_off_enc = True
        self.last_move_time = time.time()
        self.ejector_start_ts = time.time()
        self.ejector_running = False
        self.ejection_queue = []  # Queue of encoder targets for ejection
        self._ejector_pending = []  # Entries waiting for EJECTOR_DELAY before firing
        self.shipment="no_shipment"
        self.stream_histogram_data = []  # Histogram data for detected objects
        self.old_shipment = "no_shipment"
        # Minimal data snapshot coming from watcher (encoder + movement info)
        self.data = {"encoder_value": 0}

        # Dynamic camera initialization
        # Use provided paths, or auto-detected, or fallback to legacy env vars
        if camera_paths is None:
            if DETECTED_CAMERAS:
                camera_paths = DETECTED_CAMERAS
            else:
                # Only include non-empty paths from env vars
                camera_paths = [p for p in [CAM_1_PATH, CAM_2_PATH, CAM_3_PATH, CAM_4_PATH] if p]

        # Store camera paths for reference
        self.camera_paths = camera_paths
        self.cameras: Dict[int, CameraBuffer] = {}  # Dynamic camera storage {1: cam, 2: cam, ...}
        self.camera_metadata: Dict[int, Dict[str, Any]] = {}  # Metadata for each camera

        if camera_paths:
            logger.info(f"Initializing {len(camera_paths)} camera(s) from paths: {camera_paths}")
        else:
            logger.info("No USB cameras configured. Use IP Camera Discovery to add IP cameras.")

        # Initialize cameras dynamically (1-indexed for backward compatibility)
        _cam_cfgs = camera_configs or {}
        for idx, cam_path in enumerate(camera_paths, start=1):
            try:
                cc = _cam_cfgs.get(str(idx), {})
                cam = CameraBuffer(
                    cam_path,
                    exposure=cc.get('exposure', 100),
                    gain=cc.get('gain', 100),
                    brightness=cc.get('brightness', 100),
                    contrast=cc.get('contrast', 0),
                    saturation=cc.get('saturation', 50),
                    fps=cc.get('fps', 10),
                    roi_config=cc if cc.get('roi_enabled') else None,
                    auto_exposure=cc.get('auto_exposure', False),
                )
                # 4.0.15 — restore per-camera px/mm calibration from saved
                # config. Stored as a plain attribute so callers can read
                # `cam.px_per_mm` without touching the CameraBuffer constructor.
                try:
                    _ppm = cc.get('px_per_mm')
                    cam.px_per_mm = float(_ppm) if _ppm not in (None, "") else None
                except (TypeError, ValueError):
                    cam.px_per_mm = None
                self.cameras[idx] = cam
                logger.info(f"Camera {idx} initialized: {cam_path} (success={cam.success})")

                # 4.0.50 — infer type + a smart default name from the source
                # URI. UVC = /dev/video*, IP = rtsp/http, pro = basler://.
                # apply_saved_config_at_startup will later override the name
                # with whatever the operator persisted, so this default is
                # only visible until the config-restore step runs.
                if isinstance(cam_path, str) and cam_path.startswith("basler://"):
                    _t = "pro"
                    _default_name = f"Basler Camera {idx}"
                elif isinstance(cam_path, str) and cam_path.startswith(("rtsp://", "http://", "https://")):
                    _t = "ip"
                    _default_name = f"IP Camera {idx}"
                else:
                    _t = "usb"
                    _default_name = f"USB Camera {idx}"
                self.camera_metadata[idx] = {
                    "name": _default_name,
                    "type": _t,
                    "path": cam_path,
                    "source": cam_path
                }

                time.sleep(1)
            except Exception as e:
                logger.warning(f"Failed to initialize camera {idx} at {cam_path}: {e}")
                # Create a dummy camera object with success=False
                class DummyCamera:
                    success = False
                    def read(self): return None
                self.cameras[idx] = DummyCamera()
                # Still store metadata even for failed cameras
                self.camera_metadata[idx] = {
                    "name": f"USB Camera {idx} (failed)",
                    "type": "usb",
                    "path": cam_path,
                    "source": cam_path,
                    "error": str(e)
                }

        # Legacy attribute compatibility (cam_1, cam_2, cam_3, cam_4)
        self._sync_legacy_cam_attrs()

        # Note: Service config is now loaded via apply_saved_config_at_startup() after watcher init

        # 4.0.51 — REVERTED 4.0.50's aggressive initial count. That regression
        # caused a 362-drops-in-60s spike on khoy where each cv2.imwrite
        # thread contended for the same CPU. Kiancord shared the fate.
        # Rationale for the small-and-adaptive default:
        #   - "cams × 2" ignored the actual CPU capacity of the host. A
        #     3-cam machine on a 4-core box got 8 threads instantly,
        #     saturating cores before a single frame arrived.
        #   - Threads are cheap when IDLE but expensive when CONTENDING.
        #     Boot-time contention is pure loss.
        # Start conservatively; the autoscaler (in main.py) ramps only when
        # the queue actually backs up AND CPU headroom permits (4.0.51 new
        # gate — see main.py `_autoscaler`).
        add_disk_writers(max(2, len(self.cameras)))

        # Turn off lights at startup to ensure known state
        self.off()
        # Turn off ejector at startup
        self.motor_stop()

        threading.Thread(target=self.capture_frames).start()
        time.sleep(0.5)
        threading.Thread(target=self.run).start()
        time.sleep(0.5)
        threading.Thread(target=self.run_ejector).start()
        time.sleep(0.5)
        threading.Thread(target=self.stream_results).start()
        time.sleep(0.5)
        threading.Thread(target=self.write_production_metrics_loop).start()
        time.sleep(0.5)
        threading.Thread(target=self.run_barcode_scanner, daemon=True).start()
        # 4.0.98 — periodic length persistence, so if MVE crashes or the encoder
        # resets between shipment-change writes, we still have a fresh snapshot
        # to restore from on next boot. Runs every 5 s inside persist_length_state,
        # writes only when the actual encoder/length changed.
        threading.Thread(target=self._length_state_writer_loop, daemon=True, name="length-state-writer").start()

    def _length_state_writer_loop(self):
        """v4.0.98 — Periodic length-state snapshot. Sleeps 5 s between attempts;
        `persist_length_state` throttles further to at most one file write every
        5 s, so real cadence is ~5-10 s. Cheap: one small JSON file write."""
        while not self.stop_thread:
            try:
                self.persist_length_state()
            except Exception as e:
                logger.debug(f"length_state_writer_loop: {e}")
            time.sleep(5.0)


    def _sync_legacy_cam_attrs(self):
        """Keep cam_1..cam_4 attributes in sync with dynamic cameras dict."""
        self.cam_1 = self.cameras.get(1)
        self.cam_2 = self.cameras.get(2)
        self.cam_3 = self.cameras.get(3)
        self.cam_4 = self.cameras.get(4)

    def rescan_cameras(self):
        """Re-detect USB cameras and update the cameras dict.

        - New devices that appeared get a CameraBuffer created
        - Existing cameras whose device node disappeared get stopped and removed
        - Returns summary of changes: {added: [...], removed: [...], unchanged: [...]}
        """
        current_devices = detect_video_devices()  # e.g. ['/dev/video0', '/dev/video2', '/dev/video4']
        # Filter to devices that actually exist right now
        current_devices = [d for d in current_devices if os.path.exists(d)]

        # Build reverse map: device_path → cam_id for existing cameras
        existing_paths = {}
        for cam_id, cam in list(self.cameras.items()):
            path = self.camera_paths[cam_id - 1] if cam_id <= len(self.camera_paths) else None
            if path:
                existing_paths[path] = cam_id

        added = []
        removed = []
        unchanged = []

        # Check for removed cameras (device node gone)
        # 4.0.50 — rescan is a UVC (V4L2) hot-plug helper. It MUST NOT touch
        # cameras that live outside /dev/video* — that includes IP cameras
        # (rtsp://, http(s)://) and "pro" industrial cameras (basler://).
        # Before 4.0.50 this loop happily deleted the Basler entry on every
        # UI refresh because "basler://…" isn't in the V4L2 device list, and
        # then the persisted config re-loaded it minutes later — cycling
        # forever.
        NON_UVC_PREFIXES = ("basler://", "rtsp://", "http://", "https://")
        for path, cam_id in list(existing_paths.items()):
            # Skip anything that isn't a /dev/video* path. IP / pro cameras
            # have their own reconnect logic inside CameraBuffer /
            # BaslerBuffer.
            if isinstance(path, str) and path.startswith(NON_UVC_PREFIXES):
                unchanged.append({"id": cam_id, "path": path})
                continue
            if path not in current_devices:
                # Device disappeared — stop and remove
                cam = self.cameras.get(cam_id)
                if cam and hasattr(cam, 'stop'):
                    cam.stop = True
                    if hasattr(cam, 'camera'):
                        try:
                            cam.camera.release()
                        except Exception:
                            pass
                del self.cameras[cam_id]
                if cam_id in self.camera_metadata:
                    del self.camera_metadata[cam_id]
                removed.append({"id": cam_id, "path": path})
                logger.info(f"Camera {cam_id} removed (device gone): {path}")
            else:
                # Device still exists — check if cam.success is valid
                cam = self.cameras.get(cam_id)
                if cam and not getattr(cam, 'success', False):
                    # Camera object exists but failing — try restart
                    try:
                        cam.restart_camera()
                        logger.info(f"Camera {cam_id} restarted at {path}")
                    except Exception as e:
                        logger.warning(f"Camera {cam_id} restart failed: {e}")
                unchanged.append({"id": cam_id, "path": path})

        # Check for new cameras (device node appeared but not tracked)
        existing_device_set = set(existing_paths.keys())
        for dev_path in current_devices:
            if dev_path not in existing_device_set:
                # New device — assign next available cam_id
                next_id = max(self.cameras.keys()) + 1 if self.cameras else 1
                try:
                    cam = CameraBuffer(dev_path, exposure=100, gain=100)
                    self.cameras[next_id] = cam
                    # Extend camera_paths list if needed
                    while len(self.camera_paths) < next_id:
                        self.camera_paths.append(None)
                    self.camera_paths[next_id - 1] = dev_path
                    self.camera_metadata[next_id] = {
                        "name": f"USB Camera {next_id}",
                        "type": "usb",
                        "path": dev_path,
                        "source": dev_path,
                    }
                    added.append({"id": next_id, "path": dev_path})
                    logger.info(f"Camera {next_id} added (hot-plug): {dev_path}")
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Failed to init new camera at {dev_path}: {e}")

        # Sync legacy attributes
        self._sync_legacy_cam_attrs()

        return {"added": added, "removed": removed, "unchanged": unchanged}

    def on_up_down_off(self):
        self._send_message('1\n')

    def on_down_up_off(self):
        self._send_message('2\n')

    def on_up_down_on(self):
        """Both U and B lights ON."""
        self._send_message('9\n')

    def off(self):
        self._send_message('8\n')

    def _get_current_light_mode(self) -> str:
        """Get current light mode based on serial status feedback."""
        if self.u_status and not self.b_status:
            return "U_ON_B_OFF"
        elif not self.u_status and self.b_status:
            return "U_OFF_B_ON"
        elif self.u_status and self.b_status:
            return "U_ON_B_ON"
        else:
            return "U_OFF_B_OFF"

    def _set_light_mode(self, mode: str, force: bool = False):
        """Set light mode based on state configuration.

        If light_status_check is True on the current state (closed-loop):
            Checks actual serial status before sending command.
            Only sends command if current status differs from requested mode.

        If light_status_check is False (open-loop, default):
            Always sends command without checking serial status.

        Supported modes:
        - U_ON_B_OFF: U light on, B light off (command 1)
        - U_OFF_B_ON: U light off, B light on (command 2)
        - U_ON_B_ON: Both lights on (command 9)
        - U_OFF_B_OFF: Both lights off (command 8)

        Args:
            mode: Target light mode
            force: If True, send command regardless of current status
        """
        # Check if current status already matches requested mode (closed-loop mode)
        current_state = _state_manager.current_state if _state_manager else None
        light_check = getattr(current_state, 'light_status_check', False)
        if not force and light_check and self.serial_available:
            current_mode = self._get_current_light_mode()
            if current_mode == mode:
                logger.debug(f"Light mode already {mode} (verified via serial), skipping command")
                return

        mode_commands = {
            "U_ON_B_OFF": '1\n',      # on_up_down_off
            "U_OFF_B_ON": '2\n',      # on_down_up_off
            "U_ON_B_ON": '9\n',       # both on
            "U_OFF_B_OFF": '8\n',     # off
        }
        command = mode_commands.get(mode, '1\n')  # Default to U_ON_B_OFF
        self._send_message(command)
        logger.debug(f"Set light mode: {mode} (command: {command.strip()})")

    def persist_length_state(self, force=False):
        """v4.0.98 — Save (shipment, shipment_start_encoder, last_seen_length)
        to a small JSON file so an encoder reset (PLC reboot, MVE recreate,
        power outage) doesn't cause length to go negative. Throttled to at
        most one write every 5 s unless `force=True`.

        v4.0.114 — encoder-wrap guard. If we observe enc drop far below
        start mid-shipment (PLC power cycle while MVE stays up), the raw
        delta goes catastrophically negative and — worse — the previous
        version PERSISTED that negative delta to disk. Restart then read
        back the poisoned state and rebuilt shipment_start_encoder from
        it, so length stayed negative forever. On khoy 2026-07-13 the
        operator saw length=-2,149,574 for hours because of this loop.
        Fix: on downward jumps >REBASE_THRESHOLD, rebase start_encoder
        to the current encoder in-memory so length restarts at 0. We do
        NOT persist a negative length — ever.
        """
        try:
            with self._length_state_lock:
                now = time.time()
                if not force and (now - self._length_state_last_saved) < 5.0:
                    return
                ship = str(getattr(self, "shipment", "no_shipment") or "no_shipment")
                if ship in ("", "no_shipment"):
                    return
                enc = int(getattr(self, "encoder_value", 0) or 0)
                start = int(getattr(self, "shipment_start_encoder", 0) or 0)
                REBASE_THRESHOLD = 10000  # < 10k = belt reversal, keep the signed delta
                if start > 0 and (start - enc) > REBASE_THRESHOLD:
                    # Real encoder reset. Rebase so length restarts at 0.
                    logger.warning(
                        f"length wrap-guard: encoder reset detected mid-shipment "
                        f"(start={start} enc={enc} drop={start-enc}) — "
                        f"rebasing shipment_start_encoder={enc}"
                    )
                    self.shipment_start_encoder = int(enc)
                    start = int(enc)
                length_delta = enc - start
                state = {
                    "shipment":               ship,
                    "shipment_start_encoder": start,
                    "last_seen_encoder":      enc,
                    "last_seen_length":       length_delta,
                    "saved_at":               now,
                }
                tmp = self._length_state_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(state, f)
                os.replace(tmp, self._length_state_path)
                self._length_state_last_saved = now
        except Exception as e:
            logger.debug(f"persist_length_state failed: {e}")

    def restore_length_state(self):
        """v4.0.98 — On boot / after config restore, check the persisted length
        file. If the persisted shipment matches the current shipment AND the
        current encoder is BELOW the persisted shipment_start_encoder (indicates
        encoder was reset while shipment stayed active), rebuild
        shipment_start_encoder so the length delta continues from where it
        left off. Otherwise no-op. Called ONCE from main.py at startup."""
        try:
            with self._length_state_lock:
                if not os.path.exists(self._length_state_path):
                    return
                with open(self._length_state_path) as f:
                    state = json.load(f)
            persisted_ship = str(state.get("shipment", "") or "")
            current_ship   = str(getattr(self, "shipment", "no_shipment") or "no_shipment")
            if persisted_ship != current_ship or current_ship in ("", "no_shipment"):
                return  # different shipment; no restore
            persisted_start  = int(state.get("shipment_start_encoder", 0) or 0)
            persisted_length = int(state.get("last_seen_length", 0) or 0)
            current_encoder  = int(getattr(self, "encoder_value", 0) or 0)
            # v4.0.114 — reject poisoned state. Prior versions saved
            # negative persisted_length when the encoder was reset
            # mid-shipment (see persist_length_state wrap-guard). Reading
            # that back and rebuilding start_encoder from a negative
            # length just carries the bad state forward across restarts.
            if persisted_length < 0:
                logger.warning(
                    f"length restore: persisted_length={persisted_length} < 0 "
                    f"(poisoned state from pre-4.0.114) — anchoring "
                    f"shipment_start_encoder to current_encoder={current_encoder}"
                )
                self.shipment_start_encoder = int(current_encoder)
                return
            if current_encoder < persisted_start:
                # Encoder was reset (PLC reboot / power cut). Rebuild the start
                # so `encoder_value - shipment_start_encoder == persisted_length`
                # → length continues from where it left off instead of going
                # deeply negative like the operator saw on 2026-07-11.
                new_start = current_encoder - persisted_length
                self.shipment_start_encoder = int(new_start)
                logger.info(
                    f"length restore: encoder reset detected for {current_ship} "
                    f"(persisted_start={persisted_start} current_encoder={current_encoder} "
                    f"persisted_length={persisted_length}) — "
                    f"rebuilt shipment_start_encoder={self.shipment_start_encoder}"
                )
        except Exception as e:
            logger.warning(f"restore_length_state failed: {e}")

    def reset_encoder(self):
        if self.serial_available and self.serial:
            self.serial.write('3\n'.encode('utf-8'))
        self.last_encoder_value = 0
        self.last_capture_encoder = self.step * -1
        self.encoder_value = 0

    def close(self):
        if self.serial_available and self.serial:
            self.serial.close()

    def _send_message(self, msg):
        if self.serial_available and self.serial:
            self.serial.write(msg.encode('utf-8'))
            logger.debug(f"[SERIAL_TX] Sent: {msg.strip()}")
        else:
            logger.warning(f"[SERIAL_TX] Cannot send '{msg.strip()}': serial_available={self.serial_available}, serial={self.serial is not None}")

    def set_PWM_backlight(self, pwm):
        if self.serial_available and self.serial:
            self.serial.write(f"5,{pwm}\n".encode('utf-8'))

    def set_PWM_uplight(self, pwm):
        if self.serial_available and self.serial:
            self.serial.write(f"4,{pwm}\n".encode('utf-8'))

    def motor_start(self):
        self._send_message('6\n')
    
    def motor_stop(self):
        self._send_message('7\n')

    def set_step(self, step):
        self.step = step

    def keep(self):
        self.d_or_k = 1
        self.last_d_or_k = time.time()

    def on_or_off_encoder(self, on_or_off):
        self.on_or_off_enc = on_or_off

    def buzzer(self, t):
        ...

    def led(self, t):
        ...

    def find_barcode_scanner(self):
        """Dynamically find barcode scanner device."""
        import glob
        import subprocess

        # First try /dev/input/by-id/ (works on host or if mounted)
        scanner_patterns = [
            "/dev/input/by-id/*[Bb]ar[Cc]ode*event*",
            "/dev/input/by-id/*[Ss]canner*event*",
        ]
        for pattern in scanner_patterns:
            devices = glob.glob(pattern)
            if devices:
                device_path = os.path.realpath(devices[0])
                return device_path

        # Fallback: scan all event devices and check their names
        try:
            for event_file in sorted(glob.glob("/dev/input/event*")):
                try:
                    # Try to read device name from /sys
                    event_num = event_file.split("event")[-1]
                    name_path = f"/sys/class/input/event{event_num}/device/name"
                    if os.path.exists(name_path):
                        with open(name_path, 'r') as f:
                            name = f.read().strip().lower()
                            if 'barcode' in name or 'scanner' in name or '2d' in name:
                                logger.info(f"Found barcode scanner: {event_file} ({name})")
                                return event_file
                except:
                    pass
        except Exception as e:
            logger.debug(f"Error scanning input devices: {e}")

        return None

    def run_barcode_scanner(self):
        """Thread to read barcode scanner input and set shipment ID."""
        import struct
        import select

        # Key code to character mapping (US keyboard layout)
        KEY_MAP = {
            2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: '0',
            16: 'q', 17: 'w', 18: 'e', 19: 'r', 20: 't', 21: 'y', 22: 'u', 23: 'i', 24: 'o', 25: 'p',
            30: 'a', 31: 's', 32: 'd', 33: 'f', 34: 'g', 35: 'h', 36: 'j', 37: 'k', 38: 'l',
            44: 'z', 45: 'x', 46: 'c', 47: 'v', 48: 'b', 49: 'n', 50: 'm',
            12: '-', 13: '=', 26: '[', 27: ']', 39: ';', 40: "'", 41: '`',
            43: '\\', 51: ',', 52: '.', 53: '/',
            57: ' ',  # Space
        }
        KEY_MAP_SHIFT = {
            2: '!', 3: '@', 4: '#', 5: '$', 6: '%', 7: '^', 8: '&', 9: '*', 10: '(', 11: ')',
            12: '_', 13: '+',
        }

        EVENT_SIZE = struct.calcsize('llHHI')
        EV_KEY = 0x01
        KEY_ENTER = 28

        barcode_buffer = ""
        scanner_device = None
        scanner_fd = None
        last_scan_check = 0

        while not self.stop_thread:
            try:
                # Periodically check for scanner device (every 5 seconds)
                if scanner_fd is None or time.time() - last_scan_check > 5:
                    last_scan_check = time.time()
                    new_device = self.find_barcode_scanner()

                    if new_device and new_device != scanner_device:
                        # Close old device if open
                        if scanner_fd:
                            try:
                                os.close(scanner_fd)
                            except:
                                pass

                        # Open new device
                        try:
                            scanner_fd = os.open(new_device, os.O_RDONLY | os.O_NONBLOCK)
                            scanner_device = new_device
                            logger.info(f"Barcode scanner connected: {scanner_device}")
                        except Exception as e:
                            logger.warning(f"Cannot open barcode scanner {new_device}: {e}")
                            scanner_fd = None
                            scanner_device = None
                    elif not new_device and scanner_fd:
                        # Scanner disconnected
                        try:
                            os.close(scanner_fd)
                        except:
                            pass
                        scanner_fd = None
                        scanner_device = None
                        logger.info("Barcode scanner disconnected")

                if scanner_fd is None:
                    time.sleep(1)
                    continue

                # Use select to wait for data with timeout
                readable, _, _ = select.select([scanner_fd], [], [], 0.5)
                if not readable:
                    continue

                # Read events
                try:
                    data = os.read(scanner_fd, EVENT_SIZE * 10)
                except OSError as e:
                    if e.errno == 11:  # EAGAIN - no data available
                        continue
                    raise

                # Process events
                for i in range(0, len(data), EVENT_SIZE):
                    if i + EVENT_SIZE > len(data):
                        break

                    _, _, ev_type, code, value = struct.unpack('llHHI', data[i:i+EVENT_SIZE])

                    if ev_type != EV_KEY or value != 1:  # Only key press events
                        continue

                    if code == KEY_ENTER:
                        # Barcode complete - set shipment ID
                        if barcode_buffer.strip():
                            new_shipment = barcode_buffer.strip()
                            logger.info(f"Barcode scanned: {new_shipment}")

                            # 4.0.54 — snapshot current encoder as the shipment's
                            # start position so the dashboard Length tile can
                            # show belt-travelled-since-shipment-began. Only
                            # captured when this is a NEW shipment.
                            #
                            # 4.0.60 — REORDERED to match the /api/config path:
                            # (1) mutate in-memory + Redis FIRST so the very next
                            #     captured frame sees the new shipment,
                            # (2) persist to disk (save_service_config → full
                            #     JSON read + DB write + fsync + os.replace) on
                            #     a background thread so the scanner select()
                            #     loop resumes immediately and a fast follow-up
                            #     scan can't miss key events.
                            # Additionally the persistence now includes
                            # `current_shipment` — previously the barcode path
                            # only saved shipment_start_encoder, so a reboot
                            # after a barcode scan would silently revert to
                            # whatever the last API-set shipment was. This
                            # matches the API path exactly for cross-restart
                            # durability.
                            is_new_shipment = new_shipment != self.shipment
                            if is_new_shipment:
                                self.shipment_start_encoder = int(self.encoder_value or 0)
                                logger.info(
                                    f"Shipment start encoder: {self.shipment_start_encoder} "
                                    f"(barcode scan → {new_shipment})"
                                )
                                # 4.0.98 — new shipment → snapshot to disk immediately so
                                # a crash between now and the next periodic tick doesn't
                                # lose the start_encoder value.
                                try:
                                    self.shipment = new_shipment  # briefly set for persist to see
                                    self.persist_length_state(force=True)
                                except Exception:
                                    pass

                            # (1) live in-memory FIRST — detection loop + status
                            self.shipment = new_shipment
                            if self.redis_connection:
                                try:
                                    self.redis_connection.redis_connection.set("shipment", new_shipment)
                                except Exception as e:
                                    logger.warning(f"Failed to set shipment in Redis: {e}")

                            # (2) persistence off the scanner thread so the
                            # select() loop returns to reading key events
                            # immediately. Daemon thread — process shutdown
                            # doesn't wait for it. Snapshot the values into
                            # locals so the thread sees the correct captures
                            # even if a follow-up scan mutates them.
                            _snap_ship = new_shipment
                            _snap_sse = self.shipment_start_encoder if is_new_shipment else None
                            def _persist_barcode_shipment(sid, sse):
                                try:
                                    from config import load_service_config, save_service_config
                                    _svc = load_service_config() or {}
                                    _svc["current_shipment"] = sid
                                    if sse is not None:
                                        _svc["shipment_start_encoder"] = sse
                                    save_service_config(_svc)
                                except Exception as _pe:
                                    logger.warning(f"Failed to persist barcode-scanned shipment: {_pe}")
                                try:
                                    os.makedirs(f"raw_images/{sid}", exist_ok=True)
                                except Exception as _me:
                                    logger.warning(f"Failed to create shipment directory: {_me}")
                            threading.Thread(
                                target=_persist_barcode_shipment,
                                args=(_snap_ship, _snap_sse),
                                daemon=True,
                                name=f"barcode-persist-{_snap_ship}",
                            ).start()

                        barcode_buffer = ""
                    elif code in KEY_MAP:
                        barcode_buffer += KEY_MAP[code]

            except Exception as e:
                logger.error(f"Barcode scanner error: {e}")
                if scanner_fd:
                    try:
                        os.close(scanner_fd)
                    except:
                        pass
                scanner_fd = None
                scanner_device = None
                time.sleep(2)

    def signal_captured(self):
        """Check if capture should be triggered based on encoder change and state thresholds.

        Uses first phase's thresholds if set, otherwise falls back to state-level thresholds.
        steps=-1 means infinite loop (always capture, no encoder check).
        steps=1 means capture on every 1 step change (default).
        steps=N means capture on every N step changes.
        analog=-1 means analog threshold is disabled.
        """
        # Get thresholds from current state/phase
        current_state = _state_manager.current_state if _state_manager else None

        # Prefer first phase thresholds, fallback to state-level
        steps_threshold = 1  # Default: capture on every step change
        analog_threshold = -1
        if current_state:
            if current_state.phases and len(current_state.phases) > 0:
                first_phase = current_state.phases[0]
                steps_threshold = first_phase.steps
                analog_threshold = first_phase.analog
            # Fallback to state-level if phase thresholds are default
            if steps_threshold == 1 and current_state.steps != 1:
                steps_threshold = current_state.steps
            if analog_threshold == -1 and current_state.analog != -1:
                analog_threshold = current_state.analog

        # If steps threshold is -1, always trigger (infinite loop, no encoder check)
        if steps_threshold < 0:
            # Also check analog threshold if enabled (>= 0)
            if analog_threshold >= 0:
                return self.analog_value >= analog_threshold
            return True

        # Calculate encoder steps since last capture
        encoder_diff = abs(self.encoder_value - self.last_encoder_value_captured)

        # Check if encoder has moved enough steps
        if encoder_diff >= steps_threshold:
            # Also check analog threshold if enabled (>= 0)
            if analog_threshold >= 0:
                return self.analog_value >= analog_threshold
            return True
        return False

    def clear_signal(self):
        self.last_encoder_value_captured = self.encoder_value

    def queue_ejection(self, dm=None):
        """Add an ejection target to the queue based on current encoder + offset"""
        if not cfg.EJECTOR_ENABLED:
            logger.info(f"Ejector disabled, ignoring ejection request for dm={dm}")
            return
        target_encoder = self.encoder_value + cfg.EJECTOR_OFFSET
        self.ejection_queue.append({
            "target": target_encoder,
            "dm": dm,
            "queued_at": self.encoder_value
        })

    def run(self):
        """
        Read and parse serial data from the watcher device.

        Supports two modes (configured via cfg.SERIAL_MODE env variable):
            - 'legacy':  Arduino prints lines like
                'Encoder:123,Red:0,Green:0,Blue:0,Color:0,'
            - 'new':     Firmware prints CSV key/value pairs like
                'ENC:123,OKC:0,NGC:0,DWS:0,PPS:10,PPM:600,...'
        """
        buffer = ""
        MAX_BUFFER_SIZE = 1024

        while not self.stop_thread:
            try:
                # If serial is not available, just sleep and continue (camera-only mode)
                if not self.serial_available or not self.serial:
                    time.sleep(0.1)
                    continue

                if self.serial.inWaiting():
                    raw_bytes = self.serial.read(self.serial.inWaiting())
                    try:
                        chunk = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        # Drop undecodable garbage and continue
                        buffer = ""
                        time.sleep(0.03)
                        continue

                    buffer += chunk

                    # Avoid unbounded growth
                    if len(buffer) > MAX_BUFFER_SIZE:
                        logger.warning("Serial buffer exceeded max size, clearing")
                        buffer = ""

                    # Process complete lines
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        parsed_legacy = False

                        # Legacy format: starts with/contains 'Encoder:' prefix
                        if "Encoder:" in line:
                            try:
                                self.last_encoder_value = self.encoder_value
                                encoder_first_string_index = line.index("Encoder:")
                                # Legacy lines look like:
                                # 'Encoder:123,Red:0,Green:0,Blue:0,Color:0,'
                                # We only care about the encoder number.
                                encoder_last_string_index = line.find(",", encoder_first_string_index)
                                if encoder_last_string_index == -1:
                                    encoder_last_string_index = len(line)
                                self.encoder_value = int(
                                    line[encoder_first_string_index + len("Encoder:"):encoder_last_string_index]
                                )

                                self.data = {
                                    "encoder_value": self.encoder_value,
                                }
                                parsed_legacy = True
                            except Exception as parse_err:
                                logger.warning(f"Failed to parse legacy line '{line}': {parse_err}")
                                continue

                        # New format: contains ENC/OKC/NGC/DWS style keys
                        elif any(key in line for key in ("ENC:", "OKC:", "NGC:", "DWS:")):
                            try:
                                # Remove trailing comma and split
                                clean = line.rstrip(",")
                                pairs = clean.split(",")
                                kv = {}
                                for pair in pairs:
                                    if ":" in pair:
                                        k, v = pair.split(":", 1)
                                        k = k.strip()
                                        v = v.strip()
                                        if not v or not v.replace("-", "").isdigit():
                                            continue
                                        kv[k] = int(v)

                                if "ENC" not in kv:
                                    continue

                                self.last_encoder_value = self.encoder_value
                                self.encoder_value = kv["ENC"]

                                # Extended counters and metrics (if provided)
                                self.ok_counter = kv.get("OKC", self.ok_counter)
                                self.ng_counter = kv.get("NGC", self.ng_counter)
                                self.downtime_seconds = kv.get("DWS", self.downtime_seconds)
                                self.pulses_per_second = kv.get("PPS", self.pulses_per_second)
                                self.pulses_per_minute = kv.get("PPM", self.pulses_per_minute)
                                # Extra metrics
                                self.analog_value = kv.get("ANG", self.analog_value)
                                self.power_value = kv.get("PWR", self.power_value)

                                # Status bits (STS) – same semantics as watcher service
                                if "STS" in kv:
                                    self.status_value = kv["STS"]
                                    self.u_status = bool(self.status_value & 0x01)   # Bit 0: U
                                    self.b_status = bool(self.status_value & 0x02)   # Bit 1: B
                                    # Bit 2: WARNING (active-low in watcher code)
                                    self.warning_status = not bool(self.status_value & 0x04)

                                # Verbose configuration values (if present)
                                # OK configuration
                                self.ok_offset_delay = kv.get("OOD", self.ok_offset_delay)
                                self.ok_duration_pulses = kv.get("ODP", self.ok_duration_pulses)
                                self.ok_duration_percent = kv.get("ODL", self.ok_duration_percent)
                                self.ok_encoder_factor = kv.get("OEF", self.ok_encoder_factor)
                                # NG configuration
                                self.ng_offset_delay = kv.get("NOD", self.ng_offset_delay)
                                self.ng_duration_pulses = kv.get("NDP", self.ng_duration_pulses)
                                self.ng_duration_percent = kv.get("NDL", self.ng_duration_percent)
                                self.ng_encoder_factor = kv.get("NEF", self.ng_encoder_factor)
                                # System configuration
                                self.external_reset = kv.get("EXT", self.external_reset)
                                self.baud_rate = kv.get("BUD", self.baud_rate)
                                self.downtime_threshold = kv.get("DWT", self.downtime_threshold)

                                # Speed & movement (use PPS/PPM if available)
                                pps = self.pulses_per_second
                                self.is_moving = pps != 0
                                if self.is_moving:
                                    self.last_move_time = time.time()

                                self.data = {
                                    "encoder_value": self.encoder_value,
                                }
                            except Exception as parse_err:
                                logger.warning(f"Failed to parse new-mode line '{line}': {parse_err}")
                                continue
                        else:
                            # Unknown format, skip line
                            continue

                        # Common post-parse state updates
                        # Movement based on PPS (pulses per second) - more reliable than encoder comparison
                        if self.pulses_per_second > 0:
                            self.is_moving = True
                            self.last_move_time = time.time()
                        else:
                            self.is_moving = False

                        if self.encoder_value - self.last_capture_encoder > self.step:
                            self.take = True
                        else:
                            self.take = False

                        self.health_check = True
                        tmp_time = time.time()

                        if tmp_time - self.last_d_or_k > 1:
                            self.d_or_k = 0
                            self.last_d_or_k = tmp_time

                time.sleep(0.03)
            except Exception as e:
                self.health_check = False
                # 4.0.47 — universal serial reconnect with backoff.
                #
                # The previous code only attempted reconnect on "Input/output"
                # or "fileno" errors, AND short-circuited entirely if serial
                # was never available initially. Both gates failed on the
                # 2026-06-20 incident: the `.state` NoneType error didn't
                # match either substring AND the encoder USB drop set
                # serial_available=False which then prevented all future
                # reconnect attempts. Result: watcher loop kept calling
                # `time.sleep(0.1) ; continue` for ~12 hours with no recovery
                # while the inference pipeline stayed at hot=0 cold=0.
                #
                # New behaviour: ALWAYS attempt to reopen the serial port on
                # any exception, regardless of error type or prior state.
                # Backoff between attempts (capped at ~5s) so a hard-down
                # device doesn't flood logs.
                now = time.time()
                last = getattr(self, "_serial_reconnect_last", 0.0)
                backoff = getattr(self, "_serial_reconnect_backoff", 0.5)
                if now - last >= backoff:
                    self._serial_reconnect_last = now
                    try:
                        if self.serial:
                            try: self.serial.close()
                            except Exception: pass
                        self.serial = serial.Serial(self.serial_port, self.serial_baudrate, 8, 'N', 1, timeout=1)
                        self.serial.flushInput()
                        self.serial.flushOutput()
                        if not self.serial_available:
                            logger.warning(f"Serial reconnected on {self.serial_port} after error: {e}")
                        self.serial_available = True
                        self._serial_reconnect_backoff = 0.5
                    except Exception as er:
                        was_available = self.serial_available
                        self.serial_available = False
                        # Cap the backoff at ~5s so dead device doesn't flood logs.
                        self._serial_reconnect_backoff = min(backoff * 2, 5.0)
                        # Only log a state CHANGE (available -> unavailable), or
                        # one error per backoff window, to keep the log readable.
                        if was_available:
                            logger.error(f"Serial reconnection failed (lost device): {er}")
                logger.debug(f"Serial run error (will retry): {e}")
                time.sleep(0.1)

    def run_ejector(self):
        """
        Encoder-based ejector using Redis list queue.

        External service pushes to 'ejector_queue' with format:
            {"encoder": 150, "dm": "ABC123"}  (JSON string)

        This method pops from queue, calculates target (encoder + offset),
        and ejects when current encoder reaches target.
        """
        while not self.stop_thread:
            try:
                # If ejector is globally disabled, ensure it's stopped and ignore any queues
                if not cfg.EJECTOR_ENABLED:
                    if self.ejector_running:
                        self._send_message('7\n')
                        self.ejector_running = False
                    # Clear any pending ejection requests
                    self.ejection_queue.clear()
                    self._ejector_pending.clear()
                    time.sleep(cfg.EJECTOR_POLL_INTERVAL)
                    continue

                # v4.0.101 — swapped 200 Hz polling (`lpop` + `sleep 0.005`) for
                # a wake-on-data BLPOP. When there ARE pending items in the
                # queue we still need to loop fast to process the encoder-based
                # state machine below, so the timeout adapts:
                #   - queue drained → BLPOP with 1.0 s timeout (thread parks in
                #     Redis; wakes instantly on new data or after 1 s to re-check
                #     encoder progression)
                #   - queue backing up → very short timeout (5 ms) so the
                #     state-machine loop runs at full speed and drains it
                # Result: ~200× fewer wakeups when idle, no throughput loss when
                # busy. Redis-side BLPOP cost is a park on a condition variable,
                # so idle CPU cost drops to ~zero here.
                _blpop_timeout = 0.005 if self.ejection_queue or self._ejector_pending else 1.0
                blpop_result = self.redis_connection.pop_queue_blocking(
                    stream_name="ejector_queue", timeout=_blpop_timeout
                )
                raw_ejector_data = blpop_result[1] if blpop_result else None
                if raw_ejector_data:
                    try:
                        ejector_data = json.loads(raw_ejector_data.decode('utf-8'))
                        capture_encoder = ejector_data.get("encoder", self.encoder_value)
                        dm = ejector_data.get("dm", None)
                        target_encoder = capture_encoder + cfg.EJECTOR_OFFSET
                        self.ejection_queue.append({
                            "target": target_encoder,
                            "dm": dm,
                            "queued_at": capture_encoder
                        })
                        logger.info(f"[EJ_QUEUE] Added: capture_enc={capture_encoder}, target_enc={target_encoder}, offset={cfg.EJECTOR_OFFSET}, queue_size={len(self.ejection_queue)}, current_enc={self.encoder_value}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid ejector queue data: {e}")

                # Process ejection queue based on encoder position
                # Move entries whose target the encoder has reached into "waiting" state
                now = time.time()
                while self.ejection_queue and self.encoder_value >= self.ejection_queue[0]["target"]:
                    entry = self.ejection_queue.pop(0)
                    entry["ready_at"] = now
                    logger.info(f"[EJ_QUEUE] Ready: target={entry['target']}, current_enc={self.encoder_value}, delay={cfg.EJECTOR_DELAY}s, remaining={len(self.ejection_queue)}")
                    self._ejector_pending.append(entry)

                # Fire pending entries after EJECTOR_DELAY has elapsed
                if hasattr(self, '_ejector_pending') and self._ejector_pending:
                    while self._ejector_pending and (now - self._ejector_pending[0]["ready_at"]) >= cfg.EJECTOR_DELAY:
                        entry = self._ejector_pending.pop(0)
                        if not self.ejector_running:
                            logger.info(f"[EJ_FIRE] Sending '6' (ON) | target={entry['target']}, enc={self.encoder_value}, delay={cfg.EJECTOR_DELAY}s")
                            self._send_message('6\n')
                            self.ejector_running = True
                            self.ejector_start_ts = now
                            self.redis_connection.update_queue_messages_redis("Eject", stream_name="speaker")
                        else:
                            logger.info(f"[EJ_QUEUE] Discarded stale entry (ejector already running)")

                # Stop ejector after cfg.EJECTOR_DURATION
                if self.ejector_running and (now - self.ejector_start_ts > cfg.EJECTOR_DURATION):
                    logger.info(f"[EJ_FIRE] Sending '7' (OFF) | duration={cfg.EJECTOR_DURATION}s elapsed")
                    self._send_message('7\n')
                    self.ejector_running = False

                # v4.0.101 — trailing sleep removed. Pacing is now controlled
                # by BLPOP's `timeout` at the top of the loop (1.0 s when idle,
                # 5 ms when the local queue has entries needing encoder-based
                # progression checks). `EJECTOR_POLL_INTERVAL` is still honored
                # on the disabled path above (line ~1558).

            except Exception as e:
                logger.error(f"Run ejector failed: {e}")
    
    def capture_frames(self):
        """Capture frames when a signal is detected."""

        light_sleep = True
        while True:
            try:
                if self.signal_captured():
                    self.clear_signal()
                    capture_timestamp = time.time()
                    d = str(datetime.now()).replace('.', "-").replace(':', "-").replace(' ', "-")
                    frames = []
                    # 4.0.49 — TRUST self.shipment as the source of truth.
                    # The previous code re-read shipment from Redis on EVERY
                    # frame and reset to "no_shipment" if Redis returned None
                    # OR any exception fired. Combined with redis.conf
                    # `--maxmemory-policy allkeys-lru` (set in docker-compose),
                    # the `shipment` key got LRU-evicted under memory pressure
                    # (`dms` list alone holds 38k items, and INFO stats
                    # confirms evicted_keys>0 in production). The watcher then
                    # silently flipped self.shipment to "no_shipment", frames
                    # landed in raw_images/no_shipment/..., and the
                    # persistence file was never re-written — so the operator
                    # would lose the shipment without any visible cause until
                    # they noticed and re-set it.
                    #
                    # self.shipment is already kept in sync by:
                    #   - main.py [SHIPMENT-RESTORE] on boot from persistence
                    #   - routers/config_routes.py update_config which sets
                    #     watcher.shipment = shipment_id when the operator
                    #     POSTs /api/config
                    # so there is no reason to re-read Redis here. Drop it.
                    try:
                        hour_chunk = datetime.now().strftime("%Y-%m-%d_%H")
                        chunk_dir = os.path.join("raw_images", self.shipment, hour_chunk)
                        os.makedirs(chunk_dir, exist_ok=True)
                        if self.shipment != self.old_shipment:
                            self.old_shipment = self.shipment
                    except Exception as e:
                        # Don't reset self.shipment on failure here — that's
                        # exactly the bug we just removed. Log and continue
                        # with whatever shipment we already had.
                        logger.warning(
                            f"capture chunk_dir prep failed "
                            f"(shipment={self.shipment!r}): {e}"
                        )
                        hour_chunk = datetime.now().strftime("%Y-%m-%d_%H")

                    # Execute capture based on StateManager configuration
                    grabbed_frames = []
                    start_save = time.time()

                    # Get current state from _state_manager
                    current_state = _state_manager.current_state if _state_manager else None

                    if current_state:
                        # Track last light mode for open-loop optimization
                        last_light_mode = None
                        num_phases = len(current_state.phases)

                        # Execute each capture phase from state configuration
                        for phase_idx, phase in enumerate(current_state.phases):
                            # Only send command if light mode changed between phases
                            if phase.light_mode != last_light_mode:
                                self._set_light_mode(phase.light_mode)
                                logger.info(f"Phase {phase_idx+1}/{num_phases}: {phase.light_mode}")
                                last_light_mode = phase.light_mode

                            # Wait for configured delay
                            if phase.delay > 0:
                                time.sleep(phase.delay)

                            # Clear signal before capture
                            self.clear_signal()

                            # Capture cameras specified in this phase.
                            # 4.0.23 — carry phase_idx so the encode/save loop
                            # below can put the phase into the filename. Without
                            # this, phase-2 frames for a given camera OVERWRITE
                            # phase-1's file (same `<d>_<cam>.jpg` path), but
                            # phase-1's inference row survives the Redis dedupe,
                            # producing the "bbox from phase-1 image drawn on
                            # phase-2 image" visual mismatch operators have been
                            # reporting on khoy + razin.
                            for cam_id in phase.cameras:
                                cam = self.cameras.get(cam_id)
                                if cam and cam.success:
                                    capture_ts = time.time()
                                    frame = cam.read()
                                    grabbed_frames.append((cam_id, frame, phase_idx, phase.light_mode))

                                    # Track capture FPS via persistent Redis connection (db=cfg.REDIS_DB)
                                    try:
                                        cr = _get_cap_redis()
                                        cr.lpush("capture_timestamps", str(capture_ts))
                                        cr.ltrim("capture_timestamps", 0, 9)
                                        # Atomic counter: total captured frames (all cameras)
                                        cr.lpush("cap_frame_timestamps", str(capture_ts))
                                        cr.ltrim("cap_frame_timestamps", 0, 1999)  # Keep last 2000
                                    except Exception:
                                        globals()['_cap_redis'] = None  # Reset, will reconnect
                    else:
                        # No fallback - StateManager must be properly configured
                        logger.error("StateManager not available or state disabled - no capture performed")

                    # Second loop - encode frames in-memory and queue for inference + disk archival
                    # 4.0.23 — `p<phase_idx>` token inserted BEFORE the camera
                    # index makes the path unique per (timestamp, phase, camera)
                    # while keeping the camera ID at the END of the stem. That
                    # last-token-is-cam-id contract is relied on by
                    # `int(frame_id.rsplit('_', 1)[-1])` in detection.py:1197 /
                    # 1416 and `int(stream_frame[0].split('_')[-1])` in
                    # watcher.py:1487 — putting phase AFTER cam would have
                    # turned those parses into `int("p0")` and broken inference.
                    # Multi-phase states (e.g. Phase 0=U_OFF_B_ON,
                    # Phase 1=U_ON_B_OFF) used to collide on the same
                    # `<d>_<cam>.jpg` path, with the second phase's file
                    # overwriting the first on disk while two inference rows
                    # existed for the same path. That was the source of the
                    # wrong-bbox-on-right-image symptom reported on khoy + razin.
                    for camera_index, grabbed, phase_idx, light_mode in grabbed_frames:
                        d_path = f"{self.shipment}/{hour_chunk}/{d}_p{phase_idx}_{camera_index}"
                        name = os.path.join("raw_images", f"{d_path}.jpg")

                        # 4.0.52 — Encode via services.jpeg_codec (libjpeg-turbo
                        # when available). Same bytes, ~3× faster than cv2.imencode.
                        # This is the highest-frequency encode call in MVE — runs
                        # once per frame per camera, so replacing it is the biggest
                        # single CPU win the 4.0.52 change delivers.
                        from services.jpeg_codec import encode_jpeg as _enc_jpeg
                        jpeg_bytes = _enc_jpeg(grabbed, quality=85)

                        # Queue raw numpy for async disk archival (non-blocking, drop if full)
                        try:
                            _disk_queue.put_nowait((name, grabbed))
                        except queue.Full:
                            logger.warning("Disk write queue full — dropping disk write (inference continues)")

                        # Carry JPEG bytes alongside path for in-memory inference
                        frames.append([d_path, jpeg_bytes])

                    # Push every captured batch to inference (no throttle gate)
                    frames_data = {
                        "frames": frames,
                        "encoder": self.encoder_value,
                        "shipment": self.shipment,
                        "capture_t": capture_timestamp,
                    }
                    _inference_queue.put(frames_data)  # Hot → cold spill. Never drops.
            except Exception as e:
                logger.error(f"Capture error: {e}")
                # Restart all cameras dynamically
                for cam_id, cam in self.cameras.items():
                    if cam and hasattr(cam, 'restart_camera'):
                        try:
                            cam.restart_camera()
                        except Exception as restart_err:
                            logger.warning(f"Failed to restart camera {cam_id}: {restart_err}")
                time.sleep(0.01)


                
    def off_if_not_moving(self):
        if abs(time.time() - self.last_move_time) > 1:
            self.off()


    def update_data_on_redis(self, frame_number, mode,buffer, gap, data_gathering):
        data = self.data
        data['light_mode'] = mode
        data['frame_number'] = frame_number
        data['buffer'] = buffer
        data['gap'] = gap
        data['d_or_k'] = self.d_or_k
        data['data_gathering'] = data_gathering
        self.redis_connection.update_encoder_redis(data)
    
    def send_queue_messages(self, queue_messages):
        self.redis_connection.update_queue_messages_redis(queue_messages)
    
    def get_queue_messages(self, stream_name):
        return self.redis_connection.pop_queue_messages_redis(stream_name)

    def stream_results(self):
        # Target height for stream concatenation (all frames will be resized to this height)
        TARGET_STREAM_HEIGHT = 720
        # Pre-create green/red separator images once (same height every time)
        _, _green_img, _red_img = create_dynamic_images(TARGET_STREAM_HEIGHT)

        while not self.stop_thread:
            try:
                # Fetch all frames from Redis queue
                self.raw_stream_queue = self.get_queue_messages(stream_name="stream_queue")
                if self.raw_stream_queue:
                    self.stream_data = json.loads(self.raw_stream_queue)
                    stream_image = None
                    self.stream_histogram_data = []
                    remove_raw_image = cfg.remove_raw_image_when_dm_decoded
                    for stream_frame in self.stream_data:
                        frame_histogram_data = []
                        stream_path = os.path.join("raw_images", f"{stream_frame[0]}.jpg")
                        # 4.0.47 — guard cv2.imread against missing files. The
                        # disk_cleanup_loop deletes hourly chunks under
                        # raw_images/<shipment>/<YYYY-MM-DD_HH>/ when disk is
                        # >75%; if the stream_queue still holds a path inside
                        # a deleted chunk, imread returns None and the next
                        # access (.shape, cv2.rectangle, etc.) crashes the
                        # stream loop with `'NoneType' has no attribute 'shape'`.
                        # 603 of these occurred on khoy on 2026-06-20 23:46.
                        # Skip this frame cleanly when the JPG is gone.
                        if not os.path.exists(stream_path):
                            continue
                        stream_frame[1] = cv2.imread(stream_path)
                        if stream_frame[1] is None:
                            # Path exists but cv2 couldn't decode it (corrupt write
                            # or partial flush). Same skip — log once at debug to
                            # avoid log flooding when many frames in a row fail.
                            logger.debug(f"stream_results: cv2.imread returned None for {stream_path}")
                            continue
                        # 3.21.26 — single source of truth: draw_filters owns
                        # the audio_settings read. Just ask `should_draw_class(name)`.
                        # 4.0.47 — guard `_app` against being None. main.py calls
                        # set_app(app) once at startup; if stream_results() runs
                        # before that completes (race window), _app.state raises
                        # `'NoneType' has no attribute 'state'`. The 32 errors at
                        # 2026-06-20 11:33 on khoy were exactly this.
                        _tl_cfg = getattr(_app, 'state', None)
                        _tl_cfg = getattr(_tl_cfg, 'timeline_config', {}) if _tl_cfg else {}
                        _skip_bbox = not _tl_cfg.get('show_bounding_boxes', True)
                        from services.draw_filters import should_draw_class, min_confidence_for
                        for idx, res in enumerate(stream_frame[2]):
                            if _skip_bbox:
                                break
                            _nm = res.get('name', '')
                            if not should_draw_class(_nm):
                                continue
                            if res.get('confidence', 0) < min_confidence_for(_nm):
                                continue
                            cv2.rectangle(stream_frame[1], (int(res['xmin']), int(res['ymin'])),(int(res['xmax']), int(res['ymax'])), (100, 150, 250), 4)
                            text = f"{res['name']} {res['confidence']:.2f}"
                            font = cv2.FONT_HERSHEY_COMPLEX
                            font_scale = 1
                            font_color = (255, 255, 255)
                            thickness = 1
                            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                            text_w, text_h = text_size
                            x, y = int(res['xmin']), int(res['ymin'])
                            # Clamp label inside the visible frame. When the bbox top sits at
                            # y=0 (full-frame detections from the math module) the original
                            # `y - text_h` puts the label above the canvas → invisible. Draw
                            # inside-the-bbox in that case; also clamp x against the right edge.
                            frame_h, frame_w = stream_frame[1].shape[:2]
                            label_inside = (y - text_h - 4) < 0
                            if label_inside:
                                label_top = min(y + 2, max(0, frame_h - text_h - 6))
                            else:
                                label_top = y - text_h - 4
                            label_x = min(max(x, 0), max(0, frame_w - text_w - 4))
                            cv2.rectangle(
                                stream_frame[1],
                                (label_x, label_top),
                                (label_x + text_w + 4, label_top + text_h + 4),
                                (120, 120, 120), -1,
                            )
                            cv2.putText(
                                stream_frame[1], text,
                                (label_x + 2, label_top + text_h + 1),
                                font, font_scale, font_color, thickness,
                            )

                            # Calculate and save histogram for each detected object (if enabled)
                            if cfg.HISTOGRAM_ENABLED:
                                roi = stream_frame[1][int(res['ymin']):int(res['ymax']), int(res['xmin']):int(res['xmax'])]
                                if roi.size > 0:
                                    hist_r = cv2.calcHist([roi], [2], None, [256], [0, 256])
                                    hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
                                    hist_b = cv2.calcHist([roi], [0], None, [256], [0, 256])

                                    # Create and save histogram visualization image (if enabled)
                                    if cfg.HISTOGRAM_SAVE_IMAGE:
                                        hist_height = 200
                                        hist_width = 256
                                        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
                                        hist_r_norm = hist_r.copy()
                                        hist_g_norm = hist_g.copy()
                                        hist_b_norm = hist_b.copy()
                                        cv2.normalize(hist_r_norm, hist_r_norm, 0, hist_height, cv2.NORM_MINMAX)
                                        cv2.normalize(hist_g_norm, hist_g_norm, 0, hist_height, cv2.NORM_MINMAX)
                                        cv2.normalize(hist_b_norm, hist_b_norm, 0, hist_height, cv2.NORM_MINMAX)
                                        for i in range(1, 256):
                                            cv2.line(hist_image, (i-1, hist_height - int(hist_r_norm[i-1])),
                                                    (i, hist_height - int(hist_r_norm[i])), (0, 0, 255), 1)
                                            cv2.line(hist_image, (i-1, hist_height - int(hist_g_norm[i-1])),
                                                    (i, hist_height - int(hist_g_norm[i])), (0, 255, 0), 1)
                                            cv2.line(hist_image, (i-1, hist_height - int(hist_b_norm[i-1])),
                                                    (i, hist_height - int(hist_b_norm[i])), (255, 0, 0), 1)

                                        # Save histogram image
                                        name_hist_obj = os.path.join("raw_images", f"{stream_frame[0]}_obj{idx}_{res['name']}_hist.jpg")
                                        cv2.imwrite(name_hist_obj, hist_image)

                                    # Convert histograms to list of 256 ints for JSON serialization
                                    hist_r_list = [int(x) for x in hist_r.flatten()]
                                    hist_g_list = [int(x) for x in hist_g.flatten()]
                                    hist_b_list = [int(x) for x in hist_b.flatten()]

                                    frame_histogram_data.append({
                                        "frame": stream_frame[0],
                                        "id": int(stream_frame[0].split('_')[-1]) if '_' in stream_frame[0] else 0,
                                        "obj": int(idx),
                                        "name": res['name'],
                                        "bbox": {
                                            "ymin": int(res['ymin']),
                                            "ymax": int(res['ymax']),
                                            "xmin": int(res['xmin']),
                                            "xmax": int(res['xmax'])
                                        },
                                        "histogram": {
                                            "r": hist_r_list,
                                            "g": hist_g_list,
                                            "b": hist_b_list
                                        }
                                    })

                        self.stream_histogram_data.append(frame_histogram_data)

                        # Resize frame to target height for consistent concatenation (handles different ROI sizes)
                        if stream_frame[1] is not None:
                            orig_h, orig_w = stream_frame[1].shape[:2]
                            if orig_h != TARGET_STREAM_HEIGHT:
                                scale = TARGET_STREAM_HEIGHT / orig_h
                                new_w = int(orig_w * scale)
                                stream_frame[1] = cv2.resize(stream_frame[1], (new_w, TARGET_STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)

                        frame_green_image, frame_red_image = _green_img, _red_img

                        for k in range(len(stream_frame) - 4):
                            if stream_frame[4 + k] is not None:
                                remove_raw_image = cfg.remove_raw_image_when_dm_decoded
                                # Concatenate green image
                                stream_image = np.concatenate((stream_image, frame_green_image), axis=1) if stream_image is not None else frame_green_image
                                # Draw a semi-transparent rectangle behind the text
                                text = stream_frame[4 + k]
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                fontScale = 1.5  # Larger font size
                                fontColor = (255, 255, 255)  # White color
                                thickness = 2
                                lineType = cv2.LINE_AA

                                # Calculate text size
                                text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
                                text_x, text_y = 10, 40  # Starting position
                                box_coords = ((text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5))

                                # Draw the rectangle
                                cv2.rectangle(stream_frame[1], box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

                                # Add text with a stroke (optional for better visibility)
                                cv2.putText(stream_frame[1], text, (text_x, text_y), font, fontScale, (0, 0, 0), thickness + 2, lineType)  # Black border
                                cv2.putText(stream_frame[1], text, (text_x, text_y), font, fontScale, fontColor, thickness, lineType)  # Inner white text

                            else:
                                remove_raw_image = False
                                # v4.0.124 — removed the `_nd.jpg` write.
                                # Grepped the whole repo: nothing ever reads
                                # `_nd.jpg` files (write-only dead code).
                                # It was also going to the wrong path — the
                                # `os.path.join("raw_images", stream_path…)`
                                # form produced a nested `raw_images/
                                # raw_images/…` tree the disk-pressure
                                # janitor never walked, so it grew forever.
                                # The raw frame is already on disk at
                                # `stream_path.jpg`; no need for a `_nd`
                                # duplicate. Operator asked: "don't we only
                                # store the raw image?" — yes, correct.
                                stream_image = np.concatenate((stream_image, frame_red_image), axis=1) if stream_image is not None else frame_red_image

                        # Check if we should remove the raw image
                        # 4.0.57 — tolerate a missing file. Earlier code would
                        # crash the whole stream_results loop with
                        # FileNotFoundError if the raw jpg was already removed
                        # (concurrent cleanup, failed prior write, log-rotation).
                        if remove_raw_image:
                            try:
                                os.remove(os.path.join("raw_images", f"{stream_path}.jpg"))
                            except FileNotFoundError:
                                pass  # already gone — no-op, don't kill the loop
                            except OSError as _rm_err:
                                logger.warning(f"Failed to remove raw image {stream_path}.jpg: {_rm_err}")

                        stream_image = np.concatenate((stream_image, stream_frame[1]), axis=1) if stream_image is not None else stream_frame[1]

                    # 3.21.22 — removed the `output.jpg` write + `http://stream:5000`
                    # POST. The `stream` service was removed from docker-compose long
                    # ago, but the unconditional call here was still spamming the log
                    # ~once per frame with "Temporary failure in name resolution".
                    # The intermediate `stream_image` mosaic is no longer consumed
                    # by anything, so the concat above could also be deleted later.


                try:
                    self.raw_mismatch_queue = self.get_queue_messages(stream_name="dms_mismatch")
                    if self.raw_mismatch_queue:
                        self.mismatch_data = json.loads(self.raw_mismatch_queue)
                        if not isinstance(self.mismatch_data, list):
                            continue

                        for mismatch_data in self.mismatch_data:
                            if not isinstance(mismatch_data, list):
                                continue

                            for mismatch_frame in mismatch_data:
                                if not isinstance(mismatch_frame, dict) or "shipment" not in mismatch_frame or "ts" not in mismatch_frame:
                                    continue

                                shipment = mismatch_frame["shipment"]
                                ts = mismatch_frame["ts"]
                                src_folder = os.path.join("raw_images", shipment)
                                dest_folder = os.path.join("raw_images", shipment, "mismatch")

                                if not os.path.exists(src_folder):
                                    continue

                                if not os.path.exists(dest_folder):
                                    os.makedirs(dest_folder, exist_ok=True)

                                # Search recursively (files may be in hourly time-chunk subdirs)
                                for root, dirs, files in os.walk(src_folder):
                                    if "mismatch" in root:
                                        continue
                                    for file in files:
                                        if file.startswith(ts) and file.endswith(".jpg"):
                                            src_path = os.path.join(root, file)
                                            dest_path = os.path.join(dest_folder, file)
                                            try:
                                                os.rename(src_path, dest_path)
                                            except OSError:
                                                pass

                except Exception as e:
                    logger.error(f"Error on dms_mismatch: {e}")

            except Exception as e:
                logger.error(f"Error streaming frame: {e}")

    def write_production_metrics_loop(self):
        """Background thread that periodically writes production metrics to TimescaleDB."""
        logger.info("Production metrics database writer started")
        write_interval = 60  # Write every 60 seconds

        # 4.0.57 — chunked interruptible sleep. Original code slept 60s in one
        # call, so shutdown (self.stop_thread = True) blocked up to 60s
        # waiting for this thread to exit — user-visible shutdown latency on
        # every restart. The 1s check keeps the interval semantics identical
        # (writes still happen every 60s) but shortens worst-case shutdown
        # wait from 60s to ~1s.
        def _interruptible_sleep(total: int) -> bool:
            """Sleep up to `total` seconds, returning early if stop_thread flips.
            Returns True iff we exited due to shutdown."""
            for _ in range(total):
                if self.stop_thread:
                    return True
                time.sleep(1)
            return self.stop_thread

        while not self.stop_thread:
            try:
                if _interruptible_sleep(write_interval):
                    break

                # Write current production metrics to database
                write_production_metrics_to_db(
                    encoder_value=self.encoder_value,
                    ok_counter=self.ok_counter,
                    ng_counter=self.ng_counter,
                    shipment=self.shipment or "no_shipment",
                    is_moving=self.is_moving,
                    downtime_seconds=self.downtime_seconds
                )

            except Exception as e:
                logger.error(f"Error writing production metrics to database: {e}")
                if _interruptible_sleep(5):
                    break
