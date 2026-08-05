"""4.0.318 — materialized per-shipment summary + colour stats.

A finished shipment is IMMUTABLE, so we compute its stats ONCE (at finalize, or a
one-time historical backfill) and store them in two small tables. Every future query
for a finished shipment's baseline / score then reads a tiny table instead of
re-expanding millions of `detections` jsonb rows.

  shipment_summary  — one row per shipment: start/stop time, start/stop encoder,
                      reset-safe cumulative length, score, verdict, row count.
  shipment_stats    — per (shipment, camera, phase, parent_class): avg/median of the
                      colour ΔE ("E") and L, plus the sample count. This is the colour
                      BASELINE a finished shipment's cells compare against.

The CURRENT (in-progress) shipment is deliberately NOT materialized — it isn't final,
so the Charts tab shows its absolute per-bucket values + neighbour-bucket deltas instead.
"""
import logging
import threading as _threading

logger = logging.getLogger(__name__)

from services.db import get_db_connection, release_db_connection

_shipment_stats_ready = False


def ensure_shipment_stats_tables():
    """Create shipment_summary + shipment_stats if missing. Mirrors ensure_ejection_events_table."""
    global _shipment_stats_ready
    if _shipment_stats_ready:
        return True
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("BEGIN")
        cur.execute("SET LOCAL statement_timeout = 0")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shipment_summary (
                shipment        TEXT PRIMARY KEY,
                t_start         TIMESTAMPTZ,
                t_stop          TIMESTAMPTZ,
                enc_min         BIGINT,
                enc_max         BIGINT,
                enc_span        BIGINT,
                length_raw      DOUBLE PRECISION,   -- reset-safe sum of forward encoder deltas
                length_units    DOUBLE PRECISION,   -- length_raw / encoder_units_per_unit
                length_unit     TEXT,               -- label, e.g. "m"
                score           DOUBLE PRECISION,
                score_absolute  DOUBLE PRECISION,
                verdict         TEXT,
                n_total         BIGINT,
                finalized_at    TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shipment_stats (
                shipment      TEXT NOT NULL,
                cam           INT  NOT NULL,
                phase         TEXT NOT NULL DEFAULT '',
                parent_class  TEXT NOT NULL DEFAULT '',
                avg_e         DOUBLE PRECISION,
                median_e      DOUBLE PRECISION,
                avg_l         DOUBLE PRECISION,
                median_l      DOUBLE PRECISION,
                n             BIGINT,
                PRIMARY KEY (shipment, cam, phase, parent_class)
            )
            """
        )
        conn.commit()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_shipment_summary_finalized ON shipment_summary (finalized_at DESC)"
        )
        conn.commit()
        cur.close()
        _shipment_stats_ready = True
        logger.info("shipment_summary + shipment_stats tables ready")
        return True
    except Exception as e:
        logger.warning(f"ensure_shipment_stats_tables failed (will retry): {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        release_db_connection(conn)


def _units_per_unit():
    """(encoder_units_per_unit, unit_label) from config — new field wins, legacy alias fallback."""
    try:
        from config import load_service_config
        svc = load_service_config() or {}
        upu = svc.get("encoder_units_per_unit")
        if not upu:
            upu = svc.get("encoder_units_per_meter")
        unit = svc.get("encoder_unit") or "units"
        return (float(upu) if upu else 0.0), str(unit)
    except Exception:
        return 0.0, "units"


def compute_and_store_shipment_stats(shipment, skip_if_exists=False):
    """Compute ONE finished shipment's summary + per-(cam,phase,parent) colour stats and upsert.

    Bounded to a single shipment's rows (small), so it's cheap — safe to call at finalize
    time and in the backfill loop. Idempotent: re-running overwrites the same rows.
    """
    if not shipment or shipment in ("", "no_shipment"):
        return False
    # 4.0.321 STATS-2CONN — compute the score FIRST, on _compute_quality_payload's OWN connection,
    # BEFORE we take the connection for our reads+writes. The old order held an idle-in-transaction
    # connection through the two heavy jsonb scans and THEN called _compute_quality_payload (which
    # grabs a SECOND pool connection) — so each compute pinned 2 connections, one holding a snapshot.
    # A short-lived connection does the cheap skip check so we don't hold one during the score either.
    if skip_if_exists:
        _c0 = get_db_connection()
        if _c0 is not None:
            _exists = False
            try:
                _x0 = _c0.cursor()
                _x0.execute("SELECT 1 FROM shipment_summary WHERE shipment = %s", [shipment])
                _exists = _x0.fetchone() is not None
                _x0.close()
            except Exception:
                pass
            finally:
                release_db_connection(_c0)
            if _exists:
                return True

    score = score_abs = verdict = None
    try:
        from routers.timeline import _compute_quality_payload
        qp = _compute_quality_payload(shipment, "90d") or {}
        score = qp.get("score")
        score_abs = qp.get("score_absolute")
        verdict = qp.get("verdict")
    except Exception as _qe:
        logger.debug(f"score compute for {shipment} failed: {_qe}")

    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        # 4.0.321 STATS-2CONN — bounded, not 0: a hung scan can no longer camp a pooled connection
        # forever. 4.0.321 STATS-RACE — a just-closed shipment can be in BOTH the startup backfill
        # list and a finalize-hook call → two threads computing it at once. This transaction advisory
        # lock serializes same-shipment computes, so the DELETE+INSERT of its stat rows can't race
        # into a unique-violation/deadlock.
        cur.execute("SET LOCAL statement_timeout = 300000")
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", [shipment])

        # --- summary: time + encoder extent, reset-safe cumulative length, row count ---
        # length_raw sums only FORWARD encoder deltas so a per-roll encoder reset (a big
        # negative jump) contributes 0 instead of corrupting the total.
        cur.execute(
            """
            WITH rows AS (
                SELECT time, encoder_value,
                       encoder_value - LAG(encoder_value) OVER (ORDER BY time) AS d
                FROM inference_results
                WHERE shipment = %s AND encoder_value IS NOT NULL
            )
            SELECT EXTRACT(EPOCH FROM MIN(time)) * 1000.0,
                   EXTRACT(EPOCH FROM MAX(time)) * 1000.0,
                   MIN(encoder_value), MAX(encoder_value),
                   COALESCE(SUM(GREATEST(0, d)), 0),
                   COUNT(*)
            FROM rows
            """,
            [shipment],
        )
        r = cur.fetchone() or (None, None, None, None, 0, 0)
        t_start_ms, t_stop_ms, enc_min, enc_max, length_raw, n_total = r
        upu, unit = _units_per_unit()
        length_units = (float(length_raw) / upu) if (upu and length_raw is not None) else None
        enc_span = (int(enc_max) - int(enc_min)) if (enc_min is not None and enc_max is not None) else None

        # --- per (cam, phase, parent_class) colour stats over the shipment's _color rows ---
        cur.execute(
            """
            WITH cr AS (
                SELECT (regexp_match(image_path, '_p[0-9]+_([0-9]+)\\.jpg$'))[1]::int AS cam,
                       COALESCE(elem->>'phase', '')        AS phase,
                       COALESCE(elem->>'parent_class', '') AS parent_class,
                       (elem->>'E')::float AS E,
                       (elem->>'L')::float AS L
                FROM inference_results, LATERAL jsonb_array_elements(detections) elem
                WHERE shipment = %s
                  AND elem->>'name' = '_color'
                  AND image_path ~ '_p[0-9]+_[0-9]+\\.jpg$'
            )
            SELECT cam, phase, parent_class,
                   AVG(E)::float, percentile_cont(0.5) WITHIN GROUP (ORDER BY E)::float,
                   AVG(L)::float, percentile_cont(0.5) WITHIN GROUP (ORDER BY L)::float,
                   COUNT(*)
            FROM cr
            WHERE cam IS NOT NULL
            GROUP BY cam, phase, parent_class
            """,
            [shipment],
        )
        stat_rows = cur.fetchall()
        # (score/verdict were computed above, before this connection was taken — STATS-2CONN)

        # --- upsert summary ---
        cur.execute(
            """
            INSERT INTO shipment_summary
              (shipment, t_start, t_stop, enc_min, enc_max, enc_span,
               length_raw, length_units, length_unit, score, score_absolute, verdict, n_total, finalized_at)
            VALUES (%s, to_timestamp(%s / 1000.0), to_timestamp(%s / 1000.0), %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (shipment) DO UPDATE SET
              t_start=EXCLUDED.t_start, t_stop=EXCLUDED.t_stop,
              enc_min=EXCLUDED.enc_min, enc_max=EXCLUDED.enc_max, enc_span=EXCLUDED.enc_span,
              length_raw=EXCLUDED.length_raw, length_units=EXCLUDED.length_units, length_unit=EXCLUDED.length_unit,
              score=EXCLUDED.score, score_absolute=EXCLUDED.score_absolute, verdict=EXCLUDED.verdict,
              n_total=EXCLUDED.n_total, finalized_at=NOW()
            """,
            [shipment, float(t_start_ms or 0), float(t_stop_ms or 0),
             enc_min, enc_max, enc_span, length_raw, length_units, unit,
             score, score_abs, verdict, int(n_total or 0)],
        )

        # --- replace the shipment's per-cam/phase/parent stat rows ---
        cur.execute("DELETE FROM shipment_stats WHERE shipment = %s", [shipment])
        for cam, phase, parent, avg_e, med_e, avg_l, med_l, n in stat_rows:
            cur.execute(
                """
                INSERT INTO shipment_stats
                  (shipment, cam, phase, parent_class, avg_e, median_e, avg_l, median_l, n)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [shipment, int(cam or 0), phase or '', parent or '',
                 avg_e, med_e, avg_l, med_l, int(n or 0)],
            )
        conn.commit()
        cur.close()
        logger.info(
            f"shipment_stats cached: {shipment} (n={n_total}, stat_rows={len(stat_rows)}, "
            f"length={length_units} {unit}, score={score}, verdict={verdict})"
        )
        return True
    except Exception as e:
        logger.warning(f"compute_and_store_shipment_stats({shipment}) failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        release_db_connection(conn)


def finalize_shipment_async(shipment):
    """Fire-and-forget compute for a shipment that just finalized (called from the scheduler)."""
    if not shipment or shipment in ("", "no_shipment"):
        return
    try:
        _threading.Thread(
            target=compute_and_store_shipment_stats, args=(shipment,),
            daemon=True, name="ship-stats",
        ).start()
    except Exception as e:
        logger.debug(f"finalize_shipment_async({shipment}) failed: {e}")


def backfill_shipment_stats(current_shipment="", max_shipments=600, pace_sec=0.15):
    """One-time backfill: cache the RECENT finished shipments not already stored.

    Bounded + paced so it's safe on a busy PRODUCTION box: it only walks shipments seen
    in a recent slice (the Charts never looks past its window anyway), caps the count, and
    sleeps briefly between computes so it can't monopolize the DB. Skips the in-progress
    shipment. Idempotent (skip_if_exists) → after the first pass every later boot is a no-op.
    """
    import time as _t
    if not ensure_shipment_stats_tables():
        return
    conn = get_db_connection()
    if not conn:
        return
    ships = []
    try:
        cur = conn.cursor()
        cur.execute("SET LOCAL statement_timeout = 0")
        # Bounded-CTE enumeration (same trick as shipment_spans): the newest ~500k rows cover
        # every recently-active shipment cheaply via the time index — no full-table DISTINCT.
        cur.execute(
            """
            WITH recent AS (
                SELECT shipment, time FROM inference_results
                WHERE shipment IS NOT NULL AND shipment NOT IN ('', 'no_shipment')
                ORDER BY time DESC LIMIT 500000
            )
            SELECT shipment, MAX(time) AS mt FROM recent GROUP BY shipment ORDER BY mt DESC LIMIT %s
            """,
            [int(max_shipments)],
        )
        ships = [r[0] for r in cur.fetchall() if r and r[0]]
        cur.close()
    except Exception as e:
        logger.warning(f"shipment_stats backfill enumerate failed: {e}")
    finally:
        release_db_connection(conn)

    done = 0
    for sh in ships:
        if sh == current_shipment:
            continue  # in-progress → computed live in absolute mode, not materialized
        if compute_and_store_shipment_stats(sh, skip_if_exists=True):
            done += 1
        try:
            _t.sleep(max(0.0, float(pace_sec)))   # be gentle on a production DB
        except Exception:
            pass
    logger.info(f"shipment_stats backfill complete: {done}/{len(ships)} recent shipments cached")


def start_backfill_async(current_shipment=""):
    try:
        _threading.Thread(
            target=backfill_shipment_stats, args=(current_shipment,),
            daemon=True, name="ship-stats-backfill",
        ).start()
    except Exception as e:
        logger.debug(f"start_backfill_async failed: {e}")


# ---------------------------------------------------------------------------
# 4.0.318b — READ path. Serve the cached baseline + scores instead of re-scanning.
# ---------------------------------------------------------------------------

def get_cached_baseline(shipment="", window="7d", phase="", parent_class=""):
    """Return the per-camera colour baseline {cam: {avg, median, n}} from the cache.

    - A picked FINISHED shipment → its own row(s) in shipment_stats (instant).
    - All-shipments → weighted average across every finished shipment that overlaps the
      window (Σ(avg·n)/Σn per camera; median is n-weighted-averaged as a close proxy).
    Returns {"source": "cache"|"none", "current": bool, "baselines": {cam: {...}}}.
    `current=True` + empty baselines means the pick is the in-progress shipment (not
    finalized) → the caller shows ABSOLUTE values + neighbour-bucket deltas instead.
    """
    out = {"source": "none", "current": False, "baselines": {}}
    conn = get_db_connection()
    if not conn:
        return out
    try:
        cur = conn.cursor()
        _ph = " AND phase = %s" if str(phase or "").strip() else ""
        _ph_a = [str(phase).strip()] if str(phase or "").strip() else []
        _pc = " AND parent_class = %s" if str(parent_class or "").strip() else ""
        _pc_a = [str(parent_class).strip()] if str(parent_class or "").strip() else []
        if shipment:
            # is this shipment finalized (in the cache)?
            cur.execute("SELECT 1 FROM shipment_summary WHERE shipment = %s", [shipment])
            if not cur.fetchone():
                cur.close()
                out["current"] = True   # not finalized → absolute mode
                return out
            cur.execute(
                f"""SELECT cam,
                           SUM(avg_e*n)/NULLIF(SUM(n),0),
                           SUM(median_e*n)/NULLIF(SUM(n),0),
                           SUM(n)
                    FROM shipment_stats
                    WHERE shipment = %s {_ph}{_pc}
                    GROUP BY cam""",
                [shipment] + _ph_a + _pc_a,
            )
        else:
            # all-shipments: pool the finished shipments whose window overlaps.
            _iv = _window_to_interval(window)
            cur.execute(
                f"""SELECT st.cam,
                           SUM(st.avg_e*st.n)/NULLIF(SUM(st.n),0),
                           SUM(st.median_e*st.n)/NULLIF(SUM(st.n),0),
                           SUM(st.n)
                    FROM shipment_stats st
                    JOIN shipment_summary su ON su.shipment = st.shipment
                    WHERE su.t_stop > NOW() - INTERVAL %s {_ph}{_pc}
                    GROUP BY st.cam""",
                [_iv] + _ph_a + _pc_a,
            )
        for cam, avg_e, med_e, n in cur.fetchall():
            if cam is None:
                continue
            out["baselines"][str(int(cam))] = {
                "avg": round(float(avg_e), 3) if avg_e is not None else None,
                "median": round(float(med_e), 3) if med_e is not None else None,
                "n": int(n or 0),
            }
        cur.close()
        if out["baselines"]:
            out["source"] = "cache"
        return out
    except Exception as e:
        logger.debug(f"get_cached_baseline failed: {e}")
        return out
    finally:
        release_db_connection(conn)


def get_latest_shipment_scores(n=50):
    """Latest N finished shipments from shipment_summary for the Score-per-shipment lane."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT shipment, EXTRACT(EPOCH FROM t_start)*1000.0, EXTRACT(EPOCH FROM t_stop)*1000.0,
                      enc_min, enc_max, enc_span, length_units, length_unit,
                      score, score_absolute, verdict, n_total
               FROM shipment_summary
               ORDER BY t_stop DESC NULLS LAST
               LIMIT %s""",
            [int(n)],
        )
        rows = []
        for r in cur.fetchall():
            rows.append({
                "shipment": r[0], "t_start": r[1], "t_stop": r[2],
                "enc_min": r[3], "enc_max": r[4], "enc_span": r[5],
                "length_units": r[6], "length_unit": r[7],
                "score": r[8], "score_absolute": r[9], "verdict": r[10], "n_total": r[11],
            })
        cur.close()
        return rows
    except Exception as e:
        logger.debug(f"get_latest_shipment_scores failed: {e}")
        return []
    finally:
        release_db_connection(conn)


def _window_to_interval(window):
    _map = {"1h": "1 hour", "6h": "6 hours", "24h": "24 hours", "7d": "7 days",
            "30d": "30 days", "90d": "90 days"}
    return _map.get(str(window or "7d"), "7 days")
