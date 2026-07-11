import logging
import os
import threading
import queue
import psycopg2
from psycopg2.extras import Json
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = logging.getLogger(__name__)

# Database connection pool
db_connection_pool = None

# 4.0.100 — the three DB knobs the operator needs to tune per-site are now
# env-driven so an autoscaler (or a human) can bump them without a code push.
# Defaults preserve prior v4.0.79 behaviour on any site where the env is
# unset — nothing changes for existing deployments unless the operator opts in.
POSTGRES_POOL_MAX   = int(os.environ.get("POSTGRES_POOL_MAX", "40"))
MVE_DB_WRITERS_MAX  = int(os.environ.get("MVE_DB_WRITERS_MAX", "12"))

# Background write queue — all DB writes go through this to avoid blocking callers.
# DB tasks are lightweight metadata (~1KB each); default scales with RAM but the
# operator can pin an explicit size via MVE_DB_QUEUE_MAX (overrides the RAM heuristic).
import psutil as _psutil
_ram_gb = _psutil.virtual_memory().total / (1024 ** 3)
_MVE_DB_QUEUE_MAX_ENV = os.environ.get("MVE_DB_QUEUE_MAX")
if _MVE_DB_QUEUE_MAX_ENV:
    try:
        _db_queue_max = max(100, int(_MVE_DB_QUEUE_MAX_ENV))
    except ValueError:
        _db_queue_max = max(500, int(_ram_gb * 100))
else:
    _db_queue_max = max(500, int(_ram_gb * 100))
_db_queue = queue.Queue(maxsize=_db_queue_max)
logger.info(
    f"[DB] pool_max={POSTGRES_POOL_MAX} writers_max={MVE_DB_WRITERS_MAX} "
    f"queue_max={_db_queue_max} (ram={_ram_gb:.1f} GB)"
)


def get_db_connection():
    """Get PostgreSQL connection from pool."""
    global db_connection_pool
    if db_connection_pool is None:
        try:
            from psycopg2 import pool
            # 4.0.50 — pool ceiling raised from 10 → 20 to accommodate the
            # multi-thread DB writer pool + concurrent chart/API reads
            # without any one path starving another for a connection.
            # 4.0.74 — raised again 20 → 40. After v4.0.72+73 moved 116 endpoints
            # off the async event loop into FastAPI's threadpool, many more
            # queries can now run truly concurrently — a dashboard with several
            # tabs open plus the DB writer pool (cap=12) plus the SSE stream's
            # 30-s DB probe can easily peg 20 connections. Symptom the operator
            # saw: `Failed to get database connection: connection pool
            # exhausted` in the logs, then a browser `Failed to fetch` alert
            # on the shipment-save POST because save_data_file couldn't reach
            # the DB to persist the change (fell back to file-only write, but
            # under enough contention the whole POST timed out at the browser).
            # Postgres default max_connections is 100 so 40 leaves plenty of
            # headroom for other services + external tooling.
            # v4.0.79 — session-level statement_timeout=3s applied to EVERY
            # pooled connection via the Postgres `options` parameter. Under
            # heavy load (13+ hours of accumulated math_inference row spew
            # blowing up the inference_results hypertable), the analytical
            # endpoints (detection_charts, quality/shipments, quality/heatmap,
            # detection_stats, shipment_quality_score, area_stats) all scan
            # jsonb_array_elements and can hang 12–60 s per call — which
            # cascades into the browser's 6-concurrent-request-per-origin
            # limit filling up with pending fetches, so EVERY panel on the
            # Charts tab looks stuck (Score per shipment, Insights, Trend,
            # heatmap, scatter — all waiting for socket slots). Failing fast
            # (500 in ~3 s instead of hanging indefinitely) frees the
            # browser slot, lets the fast panels render on their own data,
            # and produces an actionable error the operator can see.
            #
            # DB writer thread (services/db.py::_db_writer_loop) already
            # catches exceptions and rolls back per row, so a 3 s cap on
            # writes is safe — under overload we drop that row and log a
            # warning, which is the exact behaviour v4.0.50's autoscaler
            # already assumes. Existing db-queue-full backpressure kicks
            # in normally.
            db_connection_pool = pool.SimpleConnectionPool(
                1, POSTGRES_POOL_MAX,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                options='-c statement_timeout=3000',
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
    """Background thread that drains the write queue and executes DB inserts.

    v4.0.80 — batched writes. Coalesces up to 200 rows per INSERT via
    psycopg2.extras.execute_values, grouped by exact SQL string so multiple
    distinct INSERT statements queued in parallel each get their own batch.
    Flushes when EITHER any bucket reaches 200 rows OR 100 ms have elapsed
    since the first item of the current cycle was pulled — so latency stays
    bounded when the queue is idle. Preserves the v4.0.79 SET LOCAL
    statement_timeout = 0 pattern (writes must never be dropped by the
    pool-wide 3-s cap) and the existing rollback+log+continue error semantics
    (the writer thread never dies).

    Target throughput at 240 FPS × ~15 detections/frame = 3,600 rows/sec:
    with batch=200 and flush=100ms, the writer does ~18 COMMITs/sec instead
    of 3,600. Combined with v4.0.80's synchronous_commit=off + wal_writer_delay=20ms
    postgres tuning, the fsync path is amortised over the batch and no longer
    the bottleneck.
    """
    import time as _time
    from psycopg2.extras import execute_values as _execute_values
    import re as _re

    MAX_BATCH = 200
    MAX_LATENCY_S = 0.100  # 100 ms

    def _to_values_template(sql):
        """Rewrite a single-row `... VALUES (...)` INSERT into an
        execute_values-compatible `... VALUES %s` template. Preserves the row
        tuple shape via the captured parenthesised group. Returns (None, None)
        when the SQL doesn't match the ...VALUES(...) tail — fallback path
        then runs per-row inside the same transaction (no data loss, no batch
        speedup for that odd SQL)."""
        m = _re.search(r"(?is)\bVALUES\s*(\([^)]*\))\s*$", sql.strip())
        if not m:
            return None, None
        template = m.group(1)  # e.g. "(NOW(), %s, %s, ...)"
        prefix = sql[:m.start()] + "VALUES %s"
        return prefix, template

    # Cache the rewrite per distinct SQL so we don't regex on every row.
    _tmpl_cache = {}

    def _flush(buckets):
        for sql, rows in buckets.items():
            if not rows:
                continue
            cached = _tmpl_cache.get(sql)
            if cached is None:
                cached = _to_values_template(sql)
                _tmpl_cache[sql] = cached
            prefix, template = cached

            conn = get_db_connection()
            if not conn:
                logger.warning(f"DB write: no connection, dropping batch of {len(rows)} rows")
                continue
            try:
                cursor = conn.cursor()
                cursor.execute("BEGIN")
                cursor.execute("SET LOCAL statement_timeout = 0")
                if prefix is not None and template is not None:
                    _execute_values(cursor, prefix, rows, template=template, page_size=MAX_BATCH)
                else:
                    for params in rows:
                        cursor.execute(sql, params)
                conn.commit()
                cursor.close()
            except Exception as e:
                logger.error(f"DB write error (batch of {len(rows)}): {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                release_db_connection(conn)

    while True:
        try:
            first = _db_queue.get()
            if first is None:
                break

            buckets = {}
            deadline = _time.monotonic() + MAX_LATENCY_S
            total = [0]

            def _add(task):
                sql, params = task
                bucket = buckets.get(sql)
                if bucket is None:
                    bucket = []
                    buckets[sql] = bucket
                bucket.append(params)
                total[0] += 1

            _add(first)

            shutdown = False
            while total[0] < MAX_BATCH:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                try:
                    task = _db_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if task is None:
                    shutdown = True
                    break
                _add(task)
                # Also flush early if any single bucket hit the cap — keeps
                # one hot SQL from starving another that's been waiting.
                if len(buckets[task[0]]) >= MAX_BATCH:
                    break

            _flush(buckets)

            if shutdown:
                break
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
        max_total = MVE_DB_WRITERS_MAX     # 4.0.100 env-driven (was hardcoded 12)
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
        # v4.0.79 — same reasoning as ensure_inference_encoder_column: DDL can
        # legitimately need to wait longer than the pool-wide 3-s statement
        # timeout when a heavy analytical query is holding a share lock or a
        # writer is holding row locks. SET LOCAL scopes the disable to this
        # transaction only.
        cur.execute("BEGIN")
        cur.execute("SET LOCAL statement_timeout = 0")
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
        # v4.0.79 — waive the pool-wide statement_timeout for this DDL. Under
        # heavy DB load (the same load that made me add the 3-s timeout in the
        # first place), the ALTER TABLE ... IF NOT EXISTS waits for an ACCESS
        # EXCLUSIVE lock on inference_results while writers hold it, then hits
        # the 3-s cap and gets cancelled — retry-loops for hours, spamming
        # `canceling statement due to statement timeout` and keeping the
        # column flagged as "not ready" so the encoder-axis scatter can never
        # populate. Set timeout=0 for THIS statement only via SET LOCAL, which
        # is scoped to the current transaction and reverts on commit/rollback.
        cur.execute("BEGIN")
        cur.execute("SET LOCAL statement_timeout = 0")
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
