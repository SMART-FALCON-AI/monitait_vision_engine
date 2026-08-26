"""External machine metrics ingestion (Watcher Jet & compatible) — 4.0.359.

Many factory machines (Monitait Watcher Jet devices — Raspberry-Pi units that read
OK/NG digital signals + a 4-20mA analog signal) normally POST their production counts
and telemetry to the Monitait cloud. This module lets those SAME devices point at the
LOCAL MVE instead, so their metrics are stored on-prem (TimescaleDB) and charted in the
Charts tab next to the vision data.

Payload (per Watcher Jet firmware, POST JSON):

    { "register_id",              # unique per device — the machine key
      "quantity",                 # OK count
      "defect_quantity",          # NG count
      "extra_info": {..},         # arbitrary analog metrics (temp_a/temp_b/C/…)
      "product_id", "lot_info",
      "timestamp": ISO8601,
      "product_info" }

Storage: `watcher_metrics` hypertable + `watcher_registry` (friendly names / last-seen),
both self-bootstrapping. Ingestion is best-effort and must never raise into the request
path in a way that makes a device retry-storm — the router wraps it.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.db import get_db_connection, release_db_connection

logger = logging.getLogger(__name__)

_schema_ready = False


# --------------------------------------------------------------------------- #
# tiny DB helper (tuple cursors — MVE's pool hands back raw connections)
# --------------------------------------------------------------------------- #
def _exec(sql: str, params: tuple = (), fetch: Optional[str] = None):
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("db unavailable")
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        out = cur.fetchone() if fetch == "one" else cur.fetchall() if fetch == "all" else None
        conn.commit()
        cur.close()
        return out
    finally:
        release_db_connection(conn)


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    _exec("""
        CREATE TABLE IF NOT EXISTS watcher_metrics (
            id              BIGSERIAL,
            ts              TIMESTAMPTZ NOT NULL,
            register_id     TEXT NOT NULL,
            quantity        DOUBLE PRECISION,
            defect_quantity DOUBLE PRECISION,
            extra_info      JSONB,
            product_id      TEXT,
            lot_info        TEXT,
            received_at     TIMESTAMPTZ DEFAULT NOW()
        );""")
    # Make it a hypertable if TimescaleDB is present; a plain table works otherwise.
    try:
        _exec("SELECT create_hypertable('watcher_metrics', 'ts', if_not_exists => TRUE);")
    except Exception as e:
        logger.info("watcher_metrics: hypertable not created (%s) — using plain table", e)
    _exec("CREATE INDEX IF NOT EXISTS ix_watcher_metrics_reg_ts "
          "ON watcher_metrics(register_id, ts DESC);")
    _exec("""
        CREATE TABLE IF NOT EXISTS watcher_registry (
            register_id TEXT PRIMARY KEY,
            name        TEXT,
            first_seen  TIMESTAMPTZ DEFAULT NOW(),
            last_seen   TIMESTAMPTZ
        );""")
    # 4.0.361 — registration fields for fleet management + OEE (added idempotently so
    # an already-deployed watcher_registry upgrades in place).
    for col, ddl in (
        ("line",       "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS line TEXT"),
        ("sensor",     "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS sensor TEXT"),
        ("wtype",      "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS wtype TEXT"),
        ("ideal_rate", "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS ideal_rate DOUBLE PRECISION"),
    ):
        try:
            _exec(ddl)
        except Exception as e:
            logger.info("watcher_registry: add col %s skipped (%s)", col, e)
    _schema_ready = True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_ts(ts: Any) -> datetime:
    """Best-effort parse of the device timestamp; default = now (UTC)."""
    if ts is None or ts == "":
        return datetime.now(timezone.utc)
    if isinstance(ts, (int, float)):
        # seconds vs milliseconds
        return datetime.fromtimestamp(ts / 1000.0 if ts > 1e12 else ts, timezone.utc)
    s = str(ts).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    base = s.split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(base, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _num(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _as_dict(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if not v:
        return {}
    try:
        d = json.loads(v)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Store one device report. Returns {ok, id, register_id}. Raises ValueError
    only for a missing register_id (a 400); everything else is stored leniently."""
    _ensure_schema()
    reg = str(payload.get("register_id") or "").strip()
    if not reg:
        raise ValueError("register_id required")

    ts = _parse_ts(payload.get("timestamp"))
    extra = _as_dict(payload.get("extra_info"))
    lot = payload.get("lot_info")
    lot_txt = (json.dumps(lot) if isinstance(lot, (dict, list))
               else (str(lot) if lot not in (None, "") else None))

    row = _exec(
        "INSERT INTO watcher_metrics "
        "(ts, register_id, quantity, defect_quantity, extra_info, product_id, lot_info) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (ts, reg, _num(payload.get("quantity")), _num(payload.get("defect_quantity")),
         json.dumps(extra), (str(payload.get("product_id")) if payload.get("product_id") not in (None, "") else None),
         lot_txt),
        fetch="one")

    _exec("INSERT INTO watcher_registry (register_id, last_seen) VALUES (%s,%s) "
          "ON CONFLICT (register_id) DO UPDATE SET last_seen=EXCLUDED.last_seen",
          (reg, ts))

    return {"ok": True, "id": int(row[0]) if row else None, "register_id": reg}


# --------------------------------------------------------------------------- #
# read (for the Charts tab)
# --------------------------------------------------------------------------- #
ONLINE_THRESHOLD_SEC = 600   # a machine is "online" if it reported within 10 min


def list_watchers() -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("""
        SELECT r.register_id, r.name, r.last_seen, r.line, r.sensor, r.wtype, r.ideal_rate,
               (SELECT quantity        FROM watcher_metrics m WHERE m.register_id=r.register_id ORDER BY ts DESC LIMIT 1),
               (SELECT defect_quantity FROM watcher_metrics m WHERE m.register_id=r.register_id ORDER BY ts DESC LIMIT 1),
               (SELECT COUNT(*)        FROM watcher_metrics m WHERE m.register_id=r.register_id),
               EXTRACT(EPOCH FROM (NOW() - r.last_seen))
        FROM watcher_registry r
        ORDER BY r.line NULLS LAST, r.last_seen DESC NULLS LAST, r.register_id
    """, fetch="all") or []
    out = []
    for x in rows:
        age = x[10]
        out.append({
            "register_id": x[0], "name": x[1] or x[0],
            "last_seen": str(x[2]) if x[2] else None,
            "line": x[3], "sensor": x[4], "type": x[5], "ideal_rate": x[6],
            "last_quantity": x[7], "last_defect": x[8], "samples": int(x[9] or 0),
            "online": (age is not None and float(age) <= ONLINE_THRESHOLD_SEC),
            "age_sec": (int(float(age)) if age is not None else None),
        })
    return out


def register_watcher(register_id: str, name: Optional[str] = None,
                     line: Optional[str] = None, sensor: Optional[str] = None,
                     wtype: Optional[str] = None,
                     ideal_rate: Optional[float] = None) -> Dict[str, Any]:
    """Create/update a machine's registration (name, line/group, sensor, type,
    ideal production rate for OEE). Only provided fields are changed."""
    _ensure_schema()
    reg = str(register_id or "").strip()
    if not reg:
        raise ValueError("register_id required")
    _exec("INSERT INTO watcher_registry (register_id) VALUES (%s) ON CONFLICT DO NOTHING", (reg,))
    sets, params = [], []
    for col, val in (("name", name), ("line", line), ("sensor", sensor), ("wtype", wtype)):
        if val is not None:
            sets.append(f"{col}=%s")
            params.append(str(val).strip() or None)
    if ideal_rate is not None:
        sets.append("ideal_rate=%s")
        params.append(_num(ideal_rate))
    if sets:
        _exec(f"UPDATE watcher_registry SET {', '.join(sets)} WHERE register_id=%s", tuple(params) + (reg,))
    return {"ok": True, "register_id": reg}


def metrics(register_id: str, since_ms: Optional[int] = None,
            until_ms: Optional[int] = None, limit: int = 5000) -> Dict[str, Any]:
    """Time-series for one machine. Also surfaces the numeric keys present in
    extra_info so the chart can offer them as selectable analog series."""
    _ensure_schema()
    conds = ["register_id=%s"]
    params: list = [register_id]
    if since_ms:
        conds.append("ts >= to_timestamp(%s/1000.0)")
        params.append(int(since_ms))
    if until_ms:
        conds.append("ts <= to_timestamp(%s/1000.0)")
        params.append(int(until_ms))
    where = " AND ".join(conds)
    rows = _exec(
        f"SELECT ts, quantity, defect_quantity, extra_info FROM watcher_metrics "
        f"WHERE {where} ORDER BY ts ASC LIMIT %s",
        tuple(params) + (int(limit),), fetch="all") or []

    points: List[Dict[str, Any]] = []
    keys: set = set()
    for ts, q, d, ex in rows:
        exd = _as_dict(ex)
        for k, v in exd.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
        points.append({
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "quantity": q, "defect_quantity": d, "extra": exd,
        })
    return {"register_id": register_id, "points": points,
            "metric_keys": sorted(keys), "count": len(points)}


def set_name(register_id: str, name: str) -> None:
    _ensure_schema()
    reg = str(register_id or "").strip()
    if not reg:
        raise ValueError("register_id required")
    _exec("INSERT INTO watcher_registry (register_id, name) VALUES (%s,%s) "
          "ON CONFLICT (register_id) DO UPDATE SET name=EXCLUDED.name",
          (reg, str(name or "").strip() or None))


def compute_oee(register_id: str, since_ms: Optional[int] = None,
                until_ms: Optional[int] = None) -> Dict[str, Any]:
    """OEE = Availability × Performance × Quality — the same decomposition the
    Monitait console uses (efficiency × quality × (1 − downtime)).

      Quality      = good / (good + defect)        — exact, from the OK/NG counters
      Performance  = actual_rate / ideal_rate      — needs the machine's ideal_rate
                                                      (units/hr); 1.0 until it's set
      Availability = 1 − downtime_fraction         — from extra_info.downtime_percent
                                                      if the device sends it; else 1.0

    Counter totals are summed over POSITIVE deltas so a cumulative counter that
    resets each shift is handled correctly. Every factor is returned so the UI can
    show the breakdown and the operator sees which inputs are still missing."""
    _ensure_schema()
    reg = str(register_id or "").strip()
    conds = ["register_id=%s"]
    params: list = [reg]
    if since_ms:
        conds.append("ts >= to_timestamp(%s/1000.0)")
        params.append(int(since_ms))
    if until_ms:
        conds.append("ts <= to_timestamp(%s/1000.0)")
        params.append(int(until_ms))
    where = " AND ".join(conds)
    rows = _exec(
        f"SELECT ts, quantity, defect_quantity, extra_info FROM watcher_metrics "
        f"WHERE {where} ORDER BY ts ASC", tuple(params), fetch="all") or []
    ideal_row = _exec("SELECT ideal_rate FROM watcher_registry WHERE register_id=%s",
                      (reg,), fetch="one")
    ideal_rate = float(ideal_row[0]) if ideal_row and ideal_row[0] else None

    if len(rows) < 2:
        return {"register_id": reg, "samples": len(rows), "oee": None,
                "reason": "need at least 2 samples in the window"}

    good_total = 0.0
    defect_total = 0.0
    prev_q = prev_d = None
    downtime_fracs: List[float] = []
    for ts, q, d, ex in rows:
        if q is not None:
            if prev_q is not None and q >= prev_q:
                good_total += (q - prev_q)
            prev_q = q
        if d is not None:
            if prev_d is not None and d >= prev_d:
                defect_total += (d - prev_d)
            prev_d = d
        dp = _as_dict(ex).get("downtime_percent")
        if isinstance(dp, (int, float)) and not isinstance(dp, bool):
            downtime_fracs.append(dp / 100.0 if dp > 1 else float(dp))

    total = good_total + defect_total
    quality = (good_total / total) if total > 0 else None
    availability = (1.0 - (sum(downtime_fracs) / len(downtime_fracs))) if downtime_fracs else 1.0
    availability = max(0.0, min(1.0, availability))

    t0 = rows[0][0]
    t1 = rows[-1][0]
    hours = 0.0
    try:
        hours = max(0.0, (t1 - t0).total_seconds() / 3600.0)
    except Exception:
        hours = 0.0
    actual_rate = (good_total / hours) if hours > 0 else None
    if ideal_rate and actual_rate is not None:
        performance = max(0.0, min(1.0, actual_rate / ideal_rate))
    else:
        performance = 1.0

    oee = (availability * performance * quality) if quality is not None else None
    return {
        "register_id": reg, "samples": len(rows),
        "good": round(good_total, 2), "defect": round(defect_total, 2),
        "hours": round(hours, 3),
        "quality": round(quality, 4) if quality is not None else None,
        "availability": round(availability, 4),
        "performance": round(performance, 4),
        "ideal_rate": ideal_rate,
        "actual_rate": round(actual_rate, 3) if actual_rate is not None else None,
        "oee": round(oee, 4) if oee is not None else None,
        "missing": [k for k, v in (("ideal_rate", ideal_rate),
                                   ("downtime_percent", downtime_fracs or None)) if not v],
    }
