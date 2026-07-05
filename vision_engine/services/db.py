import logging
import threading
import queue
import psycopg2
from psycopg2.extras import Json
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = logging.getLogger(__name__)

# Database connection pool
db_connection_pool = None

# Background write queue — all DB writes go through this to avoid blocking callers
# DB tasks are lightweight metadata (~1KB each); scale with RAM
import psutil as _psutil
_ram_gb = _psutil.virtual_memory().total / (1024 ** 3)
_db_queue = queue.Queue(maxsize=max(500, int(_ram_gb * 100)))


def get_db_connection():
    """Get PostgreSQL connection from pool."""
    global db_connection_pool
    if db_connection_pool is None:
        try:
            from psycopg2 import pool
            # 4.0.50 — pool ceiling raised from 10 → 20 to accommodate the
            # multi-thread DB writer pool + concurrent chart/API reads
            # without any one path starving another for a connection.
            db_connection_pool = pool.SimpleConnectionPool(
                1, 20,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return None
    try:
        return db_connection_pool.getconn()
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        return None

def release_db_connection(conn):
    """Release connection back to pool."""
    if db_connection_pool and conn:
        db_connection_pool.putconn(conn)


def _db_writer_loop():
    """Background thread that drains the write queue and executes DB inserts."""
    while True:
        try:
            task = _db_queue.get()
            if task is None:
                break
            sql, params = task
            conn = get_db_connection()
            if not conn:
                continue
            try:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                cursor.close()
            except Exception as e:
                logger.error(f"DB write error: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                release_db_connection(conn)
        except Exception as e:
            logger.error(f"DB writer loop error: {e}")


# 4.0.51 — DB writer pool, adaptive.
#
# History: 4.0.50 hard-started 6 threads at module import time. Combined
# with the disk-writer overshoot, this was one of the two regressions
# that caused the drop storm on khoy — 6 threads competing at boot with
# capture/inference for a small connection pool + CPU. Rolled back.
#
# New shape: start with 2 threads (matches the pre-4.0.50 behaviour of
# effectively 1 producer + 1 committer worth of throughput) and expose
# `add_db_writers(count)` so the autoscaler in main.py can add more
# ONLY when both signals fire:
#     - DB queue depth over threshold (real backpressure)
#     - CPU headroom available (psutil.cpu_percent < CAP)
# That guardrail (added in main.py `_autoscaler`) is what makes this
# non-regressive on small-core hosts.
_db_writers_count = 0
_db_writers_lock = threading.Lock()


def add_db_writers(count: int):
    """Add DB writer threads. Thread-safe, callable at any time by the
    main.py autoscaler. `count` is the NUMBER TO ADD, not the target total.

    Capped internally at 12 total to avoid exhausting the Postgres pool
    (psycopg2 SimpleConnectionPool has maxconn=20 in 4.0.50; leave 8
    headroom for API-side query paths).
    """
    global _db_writers_count
    if count <= 0:
        return
    with _db_writers_lock:
        current = _db_writers_count
        max_total = 12
        add = min(count, max_total - current)
        if add <= 0:
            return
        for _i in range(add):
            _t = threading.Thread(
                target=_db_writer_loop,
                daemon=True,
                name=f"db-writer-{current + _i + 1}",
            )
            _t.start()
        _db_writers_count += add
        logger.info(f"[DBWriters] +{add} threads, total now {_db_writers_count}")


# Bootstrap: start conservative. Autoscaler decides the rest.
add_db_writers(2)


def write_inference_to_db(shipment, image_path, detections, inference_time_ms,
                          model_used="yolov8", encoder_value=None):
    """Queue inference result for async write to TimescaleDB.

    encoder_value (3.21.0): capture-time encoder position, so charts can plot
    defects by roll position (camera × encoder), not just by timestamp.
    """
    try:
        _db_queue.put_nowait((
            """INSERT INTO inference_results
               (time, shipment, image_path, detections, detection_count,
                inference_time_ms, model_used, encoder_value)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)""",
            (shipment, image_path, Json(detections), len(detections),
             inference_time_ms, model_used,
             int(encoder_value) if encoder_value is not None else None)
        ))
    except queue.Full:
        logger.warning("DB write queue full — dropping inference result")

def write_production_metrics_to_db(encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds):
    """Queue production metrics for async write to TimescaleDB."""
    try:
        _db_queue.put_nowait((
            """INSERT INTO production_metrics
               (time, encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s)""",
            (encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds)
        ))
    except queue.Full:
        logger.warning("DB write queue full — dropping production metrics")


# --- Ejection events (3.17.0) -------------------------------------------------
# One row per *triggered* procedure with Store=ON, written when an eject fires.
# Powers the Ejection Insights charts (by procedure, distribution, over time).
_ejection_table_ready = False

def ensure_ejection_events_table():
    """Create the ejection_events hypertable if missing. Idempotent.

    Runs at startup (retry thread) so existing DBs that predate this table —
    init.sql only executes on a fresh data dir — get it without a manual migration.
    """
    global _ejection_table_ready
    if _ejection_table_ready:
        return True
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS ejection_events (
                   time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                   shipment TEXT,
                   procedure_name TEXT,
                   reason TEXT,
                   encoder_value BIGINT
               )"""
        )
        conn.commit()
        try:
            cur.execute("SELECT create_hypertable('ejection_events', 'time', if_not_exists => TRUE)")
            conn.commit()
        except Exception:
            conn.rollback()  # timescaledb missing → plain table is fine
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ejection_time ON ejection_events (time DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ejection_proc ON ejection_events (procedure_name, time DESC)")
        conn.commit()
        cur.close()
        _ejection_table_ready = True
        logger.info("ejection_events table ready")
        return True
    except Exception as e:
        logger.warning(f"ensure_ejection_events_table failed (will retry): {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        release_db_connection(conn)


def write_ejection_event_to_db(procedure_name, reason, shipment, encoder_value):
    """Queue a single ejection event for async write. Gated by per-procedure Store."""
    try:
        _db_queue.put_nowait((
            """INSERT INTO ejection_events (time, shipment, procedure_name, reason, encoder_value)
               VALUES (NOW(), %s, %s, %s, %s)""",
            (shipment, procedure_name, reason, int(encoder_value) if encoder_value is not None else None)
        ))
    except queue.Full:
        logger.warning("DB write queue full — dropping ejection event")


_inference_encoder_col_ready = False

def ensure_inference_encoder_column():
    """Add inference_results.encoder_value column if missing (3.21.0).

    Lets older DBs (from before 3.21.0) pick up the new column without re-init,
    so chart.js's camera×encoder scatter has data to plot.
    """
    global _inference_encoder_col_ready
    if _inference_encoder_col_ready:
        return True
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("ALTER TABLE inference_results ADD COLUMN IF NOT EXISTS encoder_value BIGINT;")
        conn.commit()
        cur.close()
        _inference_encoder_col_ready = True
        logger.info("inference_results.encoder_value column ready")
        return True
    except Exception as e:
        logger.warning(f"ensure_inference_encoder_column failed (will retry): {e}")
        try: conn.rollback()
        except Exception: pass
        return False
    finally:
        release_db_connection(conn)


def _ensure_tables_with_retry():
    """Best-effort startup migration: keep trying until the DB is up."""
    import time as _t
    for _ in range(20):
        ok_ej = ensure_ejection_events_table()
        ok_enc = ensure_inference_encoder_column()
        if ok_ej and ok_enc:
            return
        _t.sleep(3)

threading.Thread(target=_ensure_tables_with_retry, daemon=True, name="ensure-tables").start()
