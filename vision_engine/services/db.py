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

# 4.0.102 — pool-wide statement_timeout is env-driven and DEFAULT RAISED
# from 3 s (v4.0.79) → 9 s. Reason: the customer-ship endpoint audit on
# khoy (900 k+ inference rows in 24 h) showed FIVE separate analytical
# endpoints silently timing out and returning empty payloads —
# `detection_stats`, `area_stats`, `color_drift`, `quality_charts`, and
# the `detection_charts` scatter fetch. The 3 s cap was for pool-exhaustion
# protection under v4.0.79's tight pool=40. With v4.0.100's pool=80 +
# writers=24 the pool has plenty of headroom, so 9 s gives normal
# dashboards enough time without letting a truly runaway query camp on
# a connection indefinitely. The five heaviest analytical endpoints ALSO
# call `SET LOCAL statement_timeout = 15000` for extra buffer.
POSTGRES_STATEMENT_TIMEOUT_MS = int(os.environ.get("POSTGRES_STATEMENT_TIMEOUT_MS", "9000"))

# v4.0.105 — DB retention is now DISK-BUDGET based. Different semantics
# from raw_images (which uses host disk usage %):
#
#   raw_images (`_DISK_MAX_PCT`, default 75)  — reactive on HOST-disk %
#   DB         (`DB_MAX_PCT_OF_DISK`, 10)     — proactive on DB-SIZE-as-%-of-DISK
#
# Rationale: raw_images is many small files that come and go with production
# rhythm — using host disk % lets it grow to fill available space and only
# prunes when we're getting tight. The DB, by contrast, is a single logical
# store whose growth is unbounded unless we cap it — so we give it a fixed
# fraction of the disk (default 10 %) and the janitor drops oldest
# TimescaleDB chunks whenever `pg_database_size` exceeds that budget.
#
# On a 200 GB machine: raw_images can grow to 150 GB (75 %), DB caps at
# 20 GB (10 %). Combined budget 85 % leaves 15 % for OS + logs + safety.
#
# Safety valve: never drop more than `DB_MAX_CHUNKS_PER_RUN` per invocation
# so a bogus disk-total read can't nuke the whole hypertable in one pass.
DB_MAX_PCT_OF_DISK        = int(os.environ.get("DB_MAX_PCT_OF_DISK", "10"))
DB_JANITOR_INTERVAL_S     = int(os.environ.get("DB_JANITOR_INTERVAL_S", "300"))
DB_MAX_CHUNKS_PER_RUN     = int(os.environ.get("DB_MAX_CHUNKS_PER_RUN", "5"))
# Path to check for disk TOTAL. Should point at whatever mount actually
# holds the DB data. Defaults to `/` which is the correct volume on all
# three sites (khoy, kiancord, hashtgerd) — the DB container's docker
# volume is on the root fs. Override via env if your DB lives elsewhere.
DB_DISK_CHECK_PATH        = os.environ.get("DB_DISK_CHECK_PATH", "/")

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
    f"queue_max={_db_queue_max} statement_timeout={POSTGRES_STATEMENT_TIMEOUT_MS}ms "
    f"(ram={_ram_gb:.1f} GB)"
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
                options=f'-c statement_timeout={POSTGRES_STATEMENT_TIMEOUT_MS}',
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
        ok_ret = _reset_inference_retention_policy()
        if ok_ej and ok_enc and ok_ret:
            return
        _t.sleep(3)


# v4.0.105 — one-shot retention-policy migration. Since v3.x TimescaleDB
# has a policy on `inference_results` with `drop_after: 30 days`. The
# operator wants the audit horizon extended to 1 YEAR, and the disk-budget
# janitor (below) handles the "actually running out of room" case with
# no time dependency. Net effect after this migration:
#
#   - System A (TimescaleDB): drops chunks whose data ends >1 YEAR ago
#     (was 30 days). Purely a legal/audit floor now — should almost never
#     fire on medium+ volume sites because System B kicks in first.
#   - System A (compression): unchanged — still compresses chunks >1 day
#     old for the 5-10x disk savings.
#   - System B (janitor): drops OLDEST chunk when pg_database_size exceeds
#     DB_MAX_PCT_OF_DISK % of the DB volume. Runs every 5 min.
#
# The DB_RETENTION_INTERVAL env override lets the operator pin a different
# time floor per site if regulator requirements change without a code
# push. Set to "0" to REMOVE the time policy entirely (System B only).
DB_RETENTION_INTERVAL = os.environ.get("DB_RETENTION_INTERVAL", "1 year")


def _reset_inference_retention_policy():
    """Idempotent: set the drop-chunks policy on inference_results to
    `DB_RETENTION_INTERVAL`. Runs at every MVE boot but is a no-op if the
    policy is already correct."""
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        want_disabled = DB_RETENTION_INTERVAL.strip() in ("0", "off", "false", "none", "")
        # Read the current policy (if any) so we don't churn on unchanged.
        cur.execute("""
            SELECT config->>'drop_after'
            FROM timescaledb_information.jobs
            WHERE hypertable_name = 'inference_results'
              AND proc_name = 'policy_retention'
            LIMIT 1
        """)
        row = cur.fetchone()
        current = row[0] if row else None

        if want_disabled:
            if current is not None:
                cur.execute("SELECT remove_retention_policy('inference_results', if_exists => TRUE)")
                conn.commit()
                logger.info("[DB retention] time-based drop policy REMOVED (disk-budget janitor only)")
            cur.close()
            return True

        # Normalise "1 year" ⇔ "1 year" but PG stores as "1 year 00:00:00" style.
        # Cheapest correctness check: interval equality via PG.
        cur.execute("SELECT %s::interval = COALESCE(%s, '0')::interval",
                    (DB_RETENTION_INTERVAL, current))
        same = cur.fetchone()[0]
        if same:
            cur.close()
            return True

        # Change needed — drop existing, add new.
        cur.execute("SELECT remove_retention_policy('inference_results', if_exists => TRUE)")
        cur.execute("SELECT add_retention_policy('inference_results', %s::interval)",
                    (DB_RETENTION_INTERVAL,))
        conn.commit()
        logger.info(f"[DB retention] time-based drop policy set to {DB_RETENTION_INTERVAL} "
                    f"(was {current or 'unset'})")
        cur.close()
        return True
    except Exception as e:
        logger.warning(f"reset_inference_retention_policy failed (will retry): {e}")
        try: conn.rollback()
        except Exception: pass
        return False
    finally:
        release_db_connection(conn)


threading.Thread(target=_ensure_tables_with_retry, daemon=True, name="ensure-tables").start()


# ---------------------------------------------------------------------------
# v4.0.105 — DB disk-budget janitor.
# ---------------------------------------------------------------------------
def _get_disk_total_bytes(path):
    """Return the total capacity of the filesystem at `path` in bytes.
    Uses (used + free) as the denominator to match `df`'s view — ext4
    reserves ~5 % for root by default, and `shutil.disk_usage().total`
    includes that reserve, which would let the janitor's budget silently
    creep over its intended cap by 5 %."""
    import shutil
    du = shutil.disk_usage(path)
    return du.used + du.free


def _get_db_size_bytes(conn):
    cur = conn.cursor()
    cur.execute("SELECT pg_database_size(current_database())")
    n = cur.fetchone()[0]
    cur.close()
    return int(n or 0)


def _drop_oldest_inference_chunk(conn):
    """Drop the OLDEST TimescaleDB chunk of `inference_results`. Returns
    the range_end of the dropped chunk, or None if nothing was dropped
    (empty hypertable, error, etc.)."""
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT range_end "
            "FROM timescaledb_information.chunks "
            "WHERE hypertable_name = 'inference_results' "
            "ORDER BY range_start ASC "
            "LIMIT 1"
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        range_end = row[0]
        # drop_chunks drops every chunk WHOLLY-BEFORE the given cutoff. Passing
        # the OLDEST chunk's range_end drops that one chunk (and any other
        # chunks strictly older, though there won't be any if we ordered right).
        cur.execute(
            "SELECT drop_chunks('inference_results', older_than => %s)",
            (range_end,),
        )
        dropped = cur.fetchall()
        conn.commit()
        return range_end if dropped else None
    except Exception as e:
        logger.warning(f"drop_oldest_inference_chunk failed: {e}")
        try: conn.rollback()
        except Exception: pass
        return None
    finally:
        try: cur.close()
        except Exception: pass


def _db_disk_pressure_janitor_loop():
    """Every DB_JANITOR_INTERVAL_S seconds, compare pg_database_size()
    against DB_MAX_PCT_OF_DISK % of the DB volume's TOTAL. When over,
    drop up to DB_MAX_CHUNKS_PER_RUN oldest chunks and re-check.

    NEVER runs on the first invocation without waiting the interval — so
    if the machine boots WITH a huge legacy DB, the operator has a chance
    to inspect + tune the env before the janitor starts pruning.
    """
    import time as _t
    logger.info(
        f"DB disk-budget janitor started "
        f"(budget={DB_MAX_PCT_OF_DISK}% of {DB_DISK_CHECK_PATH}, "
        f"interval={DB_JANITOR_INTERVAL_S}s, max_drops_per_run={DB_MAX_CHUNKS_PER_RUN})"
    )
    while True:
        try:
            _t.sleep(DB_JANITOR_INTERVAL_S)
            total_disk = _get_disk_total_bytes(DB_DISK_CHECK_PATH)
            budget = int(total_disk * DB_MAX_PCT_OF_DISK / 100)

            conn = get_db_connection()
            if conn is None:
                logger.warning("DB janitor: no DB connection, skipping this tick")
                continue
            try:
                db_size = _get_db_size_bytes(conn)
                if db_size <= budget:
                    # Log once per hour so ops can see the janitor IS running.
                    if int(_t.time()) % 3600 < DB_JANITOR_INTERVAL_S:
                        logger.info(
                            f"[DB Janitor] under budget: "
                            f"{db_size/1e9:.2f} GB / {budget/1e9:.2f} GB "
                            f"({100*db_size/max(1,budget):.0f}%)"
                        )
                    continue

                # Over budget — drop up to N oldest chunks
                drops = 0
                dropped_ranges = []
                while db_size > budget and drops < DB_MAX_CHUNKS_PER_RUN:
                    dropped_end = _drop_oldest_inference_chunk(conn)
                    if not dropped_end:
                        break
                    drops += 1
                    dropped_ranges.append(str(dropped_end))
                    db_size = _get_db_size_bytes(conn)

                if drops:
                    logger.info(
                        f"[DB Janitor] dropped {drops} chunk(s) "
                        f"(oldest range_end: {dropped_ranges[0]}); "
                        f"DB now {db_size/1e9:.2f} GB / budget {budget/1e9:.2f} GB "
                        f"({100*db_size/max(1,budget):.0f}%)"
                    )
                    if db_size > budget:
                        logger.warning(
                            f"[DB Janitor] still over budget after {drops} drops — "
                            f"safety cap reached; next run in {DB_JANITOR_INTERVAL_S}s"
                        )
            finally:
                release_db_connection(conn)
        except Exception as e:
            logger.error(f"DB disk-budget janitor loop error: {e}")


def start_db_disk_pressure_janitor():
    """Idempotent starter — safe to call from bootstrap. No-op on re-entry."""
    for t in threading.enumerate():
        if t.name == "db-disk-budget-janitor":
            return
    threading.Thread(
        target=_db_disk_pressure_janitor_loop,
        daemon=True,
        name="db-disk-budget-janitor",
    ).start()
