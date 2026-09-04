"""Global configuration and data file management for MonitaQC Vision Engine.

All environment-based constants, mutable runtime state, and data file I/O live here.
Imported via `from config import *` in main.py as a temporary bridge during refactoring.
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration from environment variables
# =============================================================================

# Ejector configuration
EJECTOR_ENABLED = os.environ.get("EJECTOR_ENABLED", "true").lower() == "true"
EJECTOR_OFFSET = int(os.environ.get("EJECTOR_OFFSET", 1))
EJECTOR_DELAY = float(os.environ.get("EJECTOR_DELAY", 0))
EJECTOR_DURATION = float(os.environ.get("EJECTOR_DURATION", 0.01))
EJECTOR_POLL_INTERVAL = float(os.environ.get("EJECTOR_POLL_INTERVAL", 0.005))

# Capture configuration
CAPTURE_MODE = os.environ.get("CAPTURE_MODE", "single")

# Image processing configuration
REMOVE_RAW_IMAGE_WHEN_DM_DECODED = os.environ.get("REMOVE_RAW_IMAGE_WHEN_DM_DECODED", "false").lower() == "true"
PARENT_OBJECT_LIST = [x.strip() for x in os.environ.get("PARENT_OBJECT_LIST", "_root,box,pack,DP-105,jean,knit").split(",") if x.strip()]

# DataMatrix configuration
DM_CHARS_SIZES = [int(x) for x in os.environ.get("DM_CHARS_SIZES", "13,19,26").split(",")]
DM_CONFIDENCE_THRESHOLD = float(os.environ.get("DM_CONFIDENCE_THRESHOLD", 0.8))
DM_OVERLAP_THRESHOLD = float(os.environ.get("DM_OVERLAP_THRESHOLD", 0.2))

# Object count checking configuration
CHECK_CLASS_COUNTS_ENABLED = os.environ.get("CHECK_CLASS_COUNTS_ENABLED", "true").lower() == "true"
CHECK_CLASS_COUNTS_CLASSES = [x.strip() for x in os.environ.get("CHECK_CLASS_COUNTS_CLASSES", "socket,nozzle").split(",") if x.strip()]
CHECK_CLASS_COUNTS_CONFIDENCE = float(os.environ.get("CHECK_CLASS_COUNTS_CONFIDENCE", "0.5"))

# Light control configuration
LIGHT_STATUS_CHECK_ENABLED = os.environ.get("LIGHT_STATUS_CHECK_ENABLED", "false").lower() == "true"

# Histogram feature configuration
HISTOGRAM_ENABLED = os.environ.get("HISTOGRAM_ENABLED", "true").lower() == "true"
HISTOGRAM_SAVE_IMAGE = os.environ.get("HISTOGRAM_SAVE_IMAGE", "true").lower() == "true"

# Store annotation to PostgreSQL configuration
STORE_ANNOTATION_ENABLED = os.environ.get("STORE_ANNOTATION_ENABLED", "false").lower() == "true"
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "timescaledb")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "monitaqc")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "monitaqc")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "monitaqc2024")

# Data file configuration
DATA_FILE = os.environ.get("DATA_FILE", ".env.prepared_query_data")

# Parent object enforcement configuration
ENFORCE_PARENT_OBJECT = os.environ.get("ENFORCE_PARENT_OBJECT", "true").lower() == "true"

# Gradio/YOLO API configuration (defaults - will be overridden by DATA_FILE config)
YOLO_INFERENCE_URL = os.environ.get("YOLO_INFERENCE_URL", "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/")
GRADIO_MODEL = "Data Matrix"
GRADIO_CONFIDENCE_THRESHOLD = 0.25

# Global cache for latest detections
latest_detections = []
latest_detections_timestamp = 0.0

# Timeline buffer constants
TIMELINE_THUMBNAIL_WIDTH = 120
TIMELINE_THUMBNAIL_HEIGHT = 90
timeline_frame_counter = 0

# Redis keys for timeline storage
TIMELINE_REDIS_PREFIX = "timeline:"

# Detection events queue for audio notifications
DETECTION_EVENTS_REDIS_KEY = "detection_events"
DETECTION_EVENTS_MAX_SIZE = 100

# Inference performance tracking
inference_times = []
frame_intervals = []
last_inference_timestamp = 0.0
max_inference_samples = 10

# Camera capture FPS tracking
capture_timestamps = []
max_capture_samples = 10

# Lowercase aliases (used by ArduinoSocket)
capture_mode = CAPTURE_MODE
remove_raw_image_when_dm_decoded = REMOVE_RAW_IMAGE_WHEN_DM_DECODED
parent_object_list = PARENT_OBJECT_LIST

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB   = int(os.environ.get("REDIS_DB",   0))  # single db; historically code mixed 0 and 3, causing cross-store key invisibility

# Serial/Watcher configuration
WATCHER_USB = os.environ.get("WATCHER_USB", "/dev/ttyUSB0")
SERIAL_BAUDRATE = int(os.environ.get("BAUD_RATE", 57600))
SERIAL_MODE = os.environ.get("SERIAL_MODE", "new")

# Web server configuration
WEB_SERVER_PORT = int(os.environ.get("WEB_SERVER_PORT", 5050))
WEB_SERVER_HOST = os.environ.get("WEB_SERVER_HOST", "0.0.0.0")

# Command mapping for user-friendly names
USER_COMMANDS = {
    # Light control commands
    "U_ON_B_OFF": os.environ.get("WATCHER_CMD_U_ON_B_OFF", "1"),
    "B_ON_U_OFF": os.environ.get("WATCHER_CMD_B_ON_U_OFF", "2"),
    "B_OFF_U_OFF": os.environ.get("WATCHER_CMD_B_OFF_U_OFF", "8"),
    "U_ON_B_ON": os.environ.get("WATCHER_CMD_U_ON_B_ON", "9"),
    "WARNING_ON": os.environ.get("WATCHER_CMD_WARNING_ON", "6"),
    "WARNING_OFF": os.environ.get("WATCHER_CMD_WARNING_OFF", "7"),
    "RESET_ENCODER": os.environ.get("WATCHER_CMD_RST_ENCODER", "3"),
    "U_SET_PWM": os.environ.get("WATCHER_CMD_U_SET_PWM", "4"),
    "B_SET_PWM": os.environ.get("WATCHER_CMD_B_SET_PWM", "5"),
    # Counter adjustment commands
    "OK_ADJUST": os.environ.get("WATCHER_CMD_OK_ADJUST", "a"),
    "NG_ADJUST": os.environ.get("WATCHER_CMD_NG_ADJUST", "b"),
    "SET_DOWNTIME": os.environ.get("WATCHER_CMD_SET_DOWNTIME", "d"),
    # System configuration commands
    "SET_BAUD_RATE": os.environ.get("WATCHER_CMD_SET_BAUD_RATE", "s"),
    "SET_EXTERNAL_RESET": os.environ.get("WATCHER_CMD_SET_EXTERNAL_RESET", "e"),
    "SET_VERBOSE": os.environ.get("WATCHER_CMD_SET_VERBOSE", "v"),
    "SET_LEGACY": os.environ.get("WATCHER_CMD_SET_LEGACY", "l"),
    # OK configuration commands
    "OK_OFFSET_DELAY": os.environ.get("WATCHER_CMD_OK_OFFSET_DELAY", "o,od"),
    "OK_DURATION_PULSES": os.environ.get("WATCHER_CMD_OK_DURATION_PULSES", "o,dp"),
    "OK_DURATION_PERCENT": os.environ.get("WATCHER_CMD_OK_DURATION_PERCENT", "o,dl"),
    "OK_ENCODER_FACTOR": os.environ.get("WATCHER_CMD_OK_ENCODER_FACTOR", "o,ef"),
    # NG configuration commands
    "NG_OFFSET_DELAY": os.environ.get("WATCHER_CMD_NG_OFFSET_DELAY", "n,od"),
    "NG_DURATION_PULSES": os.environ.get("WATCHER_CMD_NG_DURATION_PULSES", "n,dp"),
    "NG_DURATION_PERCENT": os.environ.get("WATCHER_CMD_NG_DURATION_PERCENT", "n,dl"),
    "NG_ENCODER_FACTOR": os.environ.get("WATCHER_CMD_NG_ENCODER_FACTOR", "n,ef"),
}

# Commands that require a value parameter
COMMANDS_WITH_VALUE = [
    "U_SET_PWM", "B_SET_PWM",
    "OK_ADJUST", "NG_ADJUST", "SET_DOWNTIME",
    "SET_BAUD_RATE", "SET_EXTERNAL_RESET", "SET_VERBOSE", "SET_LEGACY",
    "OK_OFFSET_DELAY", "OK_DURATION_PULSES", "OK_DURATION_PERCENT", "OK_ENCODER_FACTOR",
    "NG_OFFSET_DELAY", "NG_DURATION_PULSES", "NG_DURATION_PERCENT", "NG_ENCODER_FACTOR",
]


# =============================================================================
# Data file management
# =============================================================================

# 4.0.321 CONFIG-CACHE — a short TTL cache in front of load_data_file(). The config hot paths
# (the SSE stream and /api/status each read it twice, the encoder tick reads it per serial line)
# each triggered a full DB round-trip — CREATE TABLE IF NOT EXISTS + COMMIT + a whole-config
# SELECT + JSONB parse — many times per second, the dominant steady-state DB/CPU load on the box.
# We now cache the loaded config as a JSON STRING and re-parse it on each read, so every caller
# gets an INDEPENDENT copy (a reader that mutates its result can't corrupt the cache), and the DB
# round-trip is skipped for the TTL window. Any save invalidates the cache immediately.
import time as _time
import threading as _threading
_CFG_CACHE_TTL = 2.0            # seconds — config changes rarely; 2s staleness is harmless
_cfg_cache_str = None
_cfg_cache_ts = 0.0
_cfg_cache_lock = _threading.Lock()

# 4.0.355 PER-KEY-WRITES — serialize ALL config persistence. Every writer used to
# do load-whole-blob → change one key → save-whole-blob with NO lock, so two
# overlapping writers (a frequent service_config save vs. a rare timeline_config /
# rules edit) clobbered each other's key — the "my rules vanished while running"
# bug, and, via the DB-upsert-vs-disk-replace split, the DB↔disk divergence. This
# RLock makes each persist atomic; save_config_key() writes only the ONE key that
# changed, so unrelated keys are never carried (nor clobbered) by another writer's
# stale snapshot.
_CONFIG_WRITE_LOCK = _threading.RLock()


def invalidate_config_cache():
    """Drop the cached config so the next load_data_file() re-reads the source. Called on save."""
    global _cfg_cache_str
    with _cfg_cache_lock:
        _cfg_cache_str = None


# 4.0.352 TIMELINE-PERSIST — provenance of the most recent *fresh* (uncached) config
# load. The boot path reads this to decide whether its conditional re-saves
# (camera-normalize, legacy→procedures migration) are safe: if the DB was DOWN at
# load time we served a stale disk snapshot, and writing it back would clobber the
# good DB row the moment the DB recovers. Values:
#   'db'       rows returned            — authoritative
#   'db-empty' reachable, zero rows     — disk is the correct bootstrap source
#   'db-down'  unreachable              — stale disk fallback; DEFER boot writes
#   'disk'     no DB layer at all
_last_load_source = 'disk'


def _load_data_file_uncached():
    """The real loader: DB-first (3.22.0), file fallback. Returns a dict (possibly empty)."""
    global _last_load_source
    # 3.22.0 — DB first
    try:
        from services.config_db import load_all as _db_load
        db_data = _db_load()
        if db_data:            # non-empty dict — authoritative
            _last_load_source = 'db'
            return db_data
        if db_data is None:    # unreachable — DB is DOWN, not merely empty
            _last_load_source = 'db-down'
        else:                  # {} — reachable but empty; disk is the bootstrap source
            _last_load_source = 'db-empty'
    except Exception as e:
        logger.debug(f"DB load skipped, using file: {e}")
        _last_load_source = 'db-down'

    # File fallback (legacy path — also the bootstrap source for the first DB save)
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                content = json.load(f)
                if isinstance(content, list):
                    return {"prepared_query_data": content}
                return content
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
    return {}


def config_db_was_down_at_load():
    """True only if the most recent fresh config load fell back to disk because the
    DB was UNREACHABLE (not merely empty). Boot skips its clobber-prone re-saves in
    this case so a temporarily-down DB isn't overwritten with stale disk data."""
    return _last_load_source == 'db-down'


def wait_for_config_db(timeout=30.0, interval=0.5):
    """Block (up to `timeout`s) until the config DB answers, so the boot config read
    is DB-first rather than a stale-disk fallback that a later boot-save would clobber
    the good DB row with. Returns True once the DB is reachable (even if empty), or
    False on timeout (boot then proceeds on the disk fallback, with writes deferred).
    Returns the instant the DB responds — a healthy boot barely waits.
    """
    import time as _t
    deadline = _t.time() + max(0.0, float(timeout))
    attempt = 0
    while True:
        attempt += 1
        try:
            from services.config_db import load_all as _db_load
            if _db_load() is not None:   # None == unreachable; {} == reachable-but-empty
                if attempt > 1:
                    logger.info(f"[BOOT-DB] config DB reachable after {attempt} attempt(s)")
                return True
        except Exception:
            pass
        if _t.time() >= deadline:
            logger.warning(
                f"[BOOT-DB] config DB not reachable after {timeout:.0f}s — booting on disk "
                f"fallback; boot-time config writes deferred to avoid clobbering the DB")
            return False
        _t.sleep(interval)


def load_data_file():
    """Load the MVE configuration (DB-first, file fallback), served from a 2s TTL cache.

    Each call returns a fresh independent copy (parsed from the cached JSON string), so a caller
    that mutates the result — e.g. save_service_config — can never corrupt the cache.
    """
    global _cfg_cache_str, _cfg_cache_ts
    now = _time.time()
    with _cfg_cache_lock:
        if _cfg_cache_str is not None and (now - _cfg_cache_ts) < _CFG_CACHE_TTL:
            try:
                return json.loads(_cfg_cache_str)
            except Exception:
                pass  # corrupt cache entry → fall through to a fresh load
    fresh = _load_data_file_uncached()
    try:
        _s = json.dumps(fresh)
        with _cfg_cache_lock:
            _cfg_cache_str = _s
            _cfg_cache_ts = _time.time()
    except Exception:
        pass  # non-JSON-serializable (shouldn't happen for config) → just don't cache
    return fresh

def _read_disk_dict():
    """Read the on-disk config snapshot as a dict (empty dict if missing/corrupt).
    A bare list (legacy prepared_query_data) is wrapped so a merge never loses it."""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                c = json.load(f)
            if isinstance(c, dict):
                return c
            if isinstance(c, list):
                return {"prepared_query_data": c}
    except Exception as e:
        logger.error(f"Error reading data file for merge: {e}")
    return {}


def _atomic_write_disk(data):
    """Atomically replace DATA_FILE with `data` (tmp + fsync + os.replace), keeping
    a .backup of the prior file. Returns True on success.

    4.0.57 rationale — a plain open('w')+dump truncates first, so a crash mid-write
    leaves an empty/partial file and the next boot falls back to defaults. Writing a
    sibling temp, fsync'ing it, then os.replace() (atomic on one filesystem) means a
    reader/next-boot always sees either the OLD file whole or the NEW file whole."""
    try:
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    _prev = f.read()
                with open(f"{DATA_FILE}.backup", 'w') as f:
                    f.write(_prev)
            except Exception:
                pass  # backup is best-effort; never block the write
        tmp_file = f"{DATA_FILE}.tmp"
        with open(tmp_file, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # fsync unsupported on some filesystems — best-effort
        os.replace(tmp_file, DATA_FILE)
        return True
    except Exception as e:
        logger.error(f"Error writing data file: {e}")
        try:
            if os.path.exists(f"{DATA_FILE}.tmp"):
                os.remove(f"{DATA_FILE}.tmp")
        except Exception:
            pass
        return False


# 4.0.391 — set when a persist could not reach the DB; cleared once re-pushed.
_db_write_pending = False


def _merge_and_persist(partial, updated_by="mve"):
    """Merge `partial` into the current config and write the SAME object to BOTH stores.

    4.0.391 ONE-KEY / ONE-WRITE. The old version built the two writes from DIFFERENT
    base states — the DB got a per-key upsert of `partial`, while the disk snapshot
    was rebuilt from whatever was already ON DISK. Any moment the disk copy was stale
    (a write while the DB was down, a restore, a hand-edit) the two stores diverged
    permanently, and the disk fallback then loaded the drifted copy. Now there is one
    authoritative base (DB first, disk only if the DB is unreachable), one merged
    object, and both stores receive it verbatim — so they cannot disagree.

    Returns True if at least one store was written.
    """
    global _db_write_pending
    if not isinstance(partial, dict):
        return False
    with _CONFIG_WRITE_LOCK:
        # 1. authoritative base: the DB, falling back to disk only when it is DOWN.
        base = None
        try:
            from services.config_db import load_all as _db_load
            base = _db_load()               # None == unreachable, {} == empty
        except Exception as e:
            logger.debug(f"DB read before persist skipped: {e}")
        db_reachable = base is not None
        if not db_reachable:
            base = _read_disk_dict()

        merged = dict(base or {})
        merged.update(partial)

        # 2. write the SAME object to both stores.
        db_ok = False
        if db_reachable:
            try:
                from services.config_db import save_all as _db_save
                db_ok = _db_save(merged, updated_by=updated_by)
            except Exception as e:
                logger.debug(f"DB save skipped: {e}")
        file_ok = _atomic_write_disk(merged)

        # 3. if the DB could not take it, remember to push disk up when it returns —
        #    otherwise a DB that comes back with older data silently wins on next boot.
        if not db_ok:
            _db_write_pending = True
            logger.warning(
                "[ONE-KEY] config written to DISK ONLY (DB unreachable) — will re-push "
                "to the database as soon as it answers")
        else:
            _db_write_pending = False

        if file_ok or db_ok:
            logger.info(
                f"Config persisted ({','.join(map(str, partial.keys()))}) "
                f"[{'DB+disk' if db_ok else 'disk only — DB unreachable'}]"
            )
        try:
            invalidate_config_cache()
        except Exception:
            pass
    return db_ok or file_ok


def reconcile_db_from_disk():
    """Push the disk bundle into the DB after a disk-only write.

    Called periodically once the DB is back. Without this, a write made while the DB
    was down lives only on disk, and the next boot — which prefers the DB — quietly
    serves the older database copy instead. Returns True when a push happened.
    """
    global _db_write_pending
    if not _db_write_pending:
        return False
    with _CONFIG_WRITE_LOCK:
        if not _db_write_pending:
            return False
        disk = _read_disk_dict()
        if not disk:
            return False
        try:
            from services.config_db import save_all as _db_save
            if _db_save(disk, updated_by="reconcile:disk->db"):
                _db_write_pending = False
                logger.warning("[ONE-KEY] re-pushed the disk config into the database "
                               "after a disk-only write")
                try:
                    invalidate_config_cache()
                except Exception:
                    pass
                return True
        except Exception as e:
            logger.debug(f"reconcile_db_from_disk failed: {e}")
    return False


def save_config_key(key, value, updated_by="mve"):
    """Persist ONE top-level config key, touching nothing else in either store.

    This is the SAFE writer: a `service_config` save can never drop `timeline_config`
    (your ejection / shipment-close rules), and vice-versa, because only the one key
    is carried. Use this instead of load-whole → modify one key → save_data_file(whole),
    which loses concurrent updates to every OTHER key (the lost-update race)."""
    return _merge_and_persist({key: value}, updated_by=updated_by)


def save_data_file(data):
    """Persist the MVE configuration (dual-write DB + disk).

    4.0.355 — MERGE semantics on BOTH stores now: the DB upserts each key in `data`
    and the disk snapshot is MERGED (not whole-file-replaced), so a partial payload
    can no longer trim keys off disk while the DB keeps them. Serialized by
    _CONFIG_WRITE_LOCK. Whole-blob callers (full restore, boot) are unaffected —
    merging a full blob equals replacing it. For single-key writes prefer
    save_config_key(), which additionally sidesteps the lost-update race.

    Returns True if at least one store was written."""
    return _merge_and_persist(data if isinstance(data, dict) else {})

def load_service_config():
    """Load service configuration from data file (under 'service_config' key)."""
    data = load_data_file()
    return data.get("service_config", {})

def save_service_config(config):
    """Save service configuration (under 'service_config').

    4.0.355 — writes ONLY the service_config key (save_config_key) instead of
    load-whole-blob → set one key → save-whole-blob. This is what stops the ~40
    service_config writers (pipeline activate, camera tweaks, scheduler, watcher,
    anomaly…) from clobbering timeline_config/rules a concurrent writer just changed."""
    config["saved_at"] = datetime.now().isoformat()
    return save_config_key("service_config", config)

# Aliases for backwards compatibility
load_camera_config = load_service_config
save_camera_config = save_service_config
