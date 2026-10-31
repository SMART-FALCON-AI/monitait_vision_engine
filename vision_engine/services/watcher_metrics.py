"""External machine metrics ingestion (Watcher Jet & compatible) — 4.0.359.

Many factory machines (Monitait Watcher Jet devices — Raspberry-Pi units that read
OK/NG digital signals + a 4-20mA analog signal) normally POST their production counts
and telemetry to the Monitait cloud. This module lets those SAME devices point at the
LOCAL MVE instead, so their metrics are stored on-prem (TimescaleDB) and charted in the
Charts tab next to the vision data.

Payload (per Watcher Jet firmware, POST JSON):

    { "register_id",              # unique per device — the machine key
      "quantity",                 # OK count SINCE THE PREVIOUS REPORT (not cumulative)
      "defect_quantity",          # NG count since the previous report
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
import re
import threading
import time
from datetime import datetime, timedelta, timezone
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
        # 4.0.417 — Hardware-tab station setup: the station this one feeds (line order),
        # separate OK / NG multiplication factors (units per counted pulse), and whether
        # the operator has set the watcher up yet (a new device shows as "NEW").
        ("next_watcher", "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS next_watcher TEXT"),
        ("ok_factor",    "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS ok_factor DOUBLE PRECISION"),
        ("ng_factor",    "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS ng_factor DOUBLE PRECISION"),
        ("assigned",     "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS assigned BOOLEAN DEFAULT FALSE"),
        # 4.0.419 — camera stations: MVE itself reports the station's OK/NG from the
        # detections of chosen classes on chosen cameras (cam_config); cam_last_ts =
        # how far the detections have been counted.
        ("cam_config",   "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS cam_config JSONB"),
        ("cam_last_ts",  "ALTER TABLE watcher_registry ADD COLUMN IF NOT EXISTS cam_last_ts TIMESTAMPTZ"),
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


def _factor(v: Any) -> float:
    """A stored OK/NG multiplication factor; unset (or invalid) means 1."""
    f = _num(v)
    return f if f is not None and f > 0 else 1.0


def _factors(reg: str) -> tuple:
    r = _exec("SELECT ok_factor, ng_factor FROM watcher_registry WHERE register_id=%s",
              (reg,), fetch="one")
    return (_factor(r[0]), _factor(r[1])) if r else (1.0, 1.0)


def list_watchers() -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("""
        SELECT r.register_id, r.name, r.last_seen, r.line, r.sensor, r.wtype, r.ideal_rate,
               (SELECT quantity        FROM watcher_metrics m WHERE m.register_id=r.register_id ORDER BY ts DESC LIMIT 1),
               (SELECT defect_quantity FROM watcher_metrics m WHERE m.register_id=r.register_id ORDER BY ts DESC LIMIT 1),
               (SELECT COUNT(*)        FROM watcher_metrics m WHERE m.register_id=r.register_id),
               EXTRACT(EPOCH FROM (NOW() - r.last_seen)),
               r.next_watcher, r.ok_factor, r.ng_factor, r.assigned, r.cam_config
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
            # 4.0.417 — station setup (Hardware tab). A watcher named before 4.0.417
            # (Charts-tab registration) counts as already set up.
            "station": x[1], "next_watcher": x[11],
            "ok_factor": _factor(x[12]), "ng_factor": _factor(x[13]),
            "assigned": bool(x[14]) or bool(x[1]),
            # 4.0.419 — "camera" = counted by MVE from detections (see camera_config)
            "kind": "camera" if x[15] is not None else "watcher",
            "camera_config": _as_dict(x[15]) if x[15] is not None else None,
        })
    return out


def _validate_next(reg: str, nxt: str) -> None:
    """The next station must be another known watcher, and following the chain from
    it must never come back to this one (A → B → A would make the line a loop)."""
    if nxt == reg:
        raise ValueError("a station cannot be its own next station")
    links = dict(_exec("SELECT register_id, next_watcher FROM watcher_registry",
                       fetch="all") or [])
    if nxt not in links:
        raise ValueError(f"unknown next station '{nxt}'")
    links[reg] = nxt
    cur, seen = nxt, set()
    while cur and cur not in seen:
        if cur == reg:
            raise ValueError("this next station would make the line a loop")
        seen.add(cur)
        cur = links.get(cur)


def register_watcher(register_id: str, name: Optional[str] = None,
                     line: Optional[str] = None, sensor: Optional[str] = None,
                     wtype: Optional[str] = None,
                     ideal_rate: Optional[float] = None,
                     next_watcher: Optional[str] = None,
                     ok_factor: Optional[Any] = None,
                     ng_factor: Optional[Any] = None,
                     assigned: Optional[bool] = None) -> Dict[str, Any]:
    """Create/update a machine's registration (name, line/group, sensor, type,
    ideal production rate for OEE; 4.0.417: next station + OK/NG multiplication
    factors). Only provided fields are changed; "" clears a text field / resets a
    factor to 1."""
    _ensure_schema()
    reg = str(register_id or "").strip()
    if not reg:
        raise ValueError("register_id required")
    sets, params = [], []
    if next_watcher is not None:
        nxt = str(next_watcher).strip()
        if nxt:
            _validate_next(reg, nxt)
        sets.append("next_watcher=%s")
        params.append(nxt or None)
    for col, val in (("ok_factor", ok_factor), ("ng_factor", ng_factor)):
        if val is not None:
            f = _num(val)
            if val != "" and (f is None or f <= 0):
                raise ValueError(f"{col.replace('_', ' ')} must be a number above 0")
            sets.append(f"{col}=%s")
            params.append(f)
    _exec("INSERT INTO watcher_registry (register_id) VALUES (%s) ON CONFLICT DO NOTHING", (reg,))
    for col, val in (("name", name), ("line", line), ("sensor", sensor), ("wtype", wtype)):
        if val is not None:
            sets.append(f"{col}=%s")
            params.append(str(val).strip() or None)
    if ideal_rate is not None:
        sets.append("ideal_rate=%s")
        params.append(_num(ideal_rate))
    if assigned is not None:
        sets.append("assigned=%s")
        params.append(bool(assigned))
    if sets:
        _exec(f"UPDATE watcher_registry SET {', '.join(sets)} WHERE register_id=%s", tuple(params) + (reg,))
    return {"ok": True, "register_id": reg}


def delete_watcher(register_id: str) -> Dict[str, Any]:
    """Remove a watcher from the station list (its history rows are kept). If the
    device reports again it shows up as NEW."""
    _ensure_schema()
    reg = str(register_id or "").strip()
    if not reg:
        raise ValueError("register_id required")
    _exec("UPDATE watcher_registry SET next_watcher=NULL WHERE next_watcher=%s", (reg,))
    _exec("DELETE FROM watcher_registry WHERE register_id=%s", (reg,))
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

    okf, ngf = _factors(register_id)
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
            # 4.0.417 — units after the station's OK / NG multiplication factors
            "ok": (q * okf if q is not None else None),
            "ng": (d * ngf if d is not None else None),
        })
    return {"register_id": register_id, "points": points, "ok_factor": okf, "ng_factor": ngf,
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

    Each report is an INCREMENT (4.0.414): every Watcher Jet firmware zeroes its
    counters after a successful send (main.py minus(), main_v2.py temp_a = 0), and
    offline-buffered reports are replayed one by one — so totals are plain sums.
    Every factor is returned so the UI can show the breakdown and the operator sees
    which inputs are still missing.

    4.0.417 — good/defect are in UNITS: OK pulses × ok_factor, NG pulses × ng_factor
    (e.g. one OK pulse per 12-bottle case, one NG pulse per bottle). The raw pulse
    totals are returned too."""
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

    good_total = 0.0
    defect_total = 0.0
    downtime_fracs: List[float] = []
    for ts, q, d, ex in rows:
        if q is not None and q > 0:
            good_total += q
        if d is not None and d > 0:
            defect_total += d
        dp = _as_dict(ex).get("downtime_percent")
        if isinstance(dp, (int, float)) and not isinstance(dp, bool):
            downtime_fracs.append(dp / 100.0 if dp > 1 else float(dp))

    okf, ngf = _factors(reg)
    good_raw, defect_raw = good_total, defect_total
    good_total *= okf
    defect_total *= ngf
    if len(rows) < 2:
        # 4.0.418 — the unit totals are still returned (the line OEE needs them)
        return {"register_id": reg, "samples": len(rows), "oee": None,
                "good": round(good_total, 2), "defect": round(defect_total, 2),
                "good_raw": round(good_raw, 2), "defect_raw": round(defect_raw, 2),
                "ok_factor": okf, "ng_factor": ngf,
                "reason": "need at least 2 samples in the window"}
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
        "good_raw": round(good_raw, 2), "defect_raw": round(defect_raw, 2),
        "ok_factor": okf, "ng_factor": ngf,
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


def compute_lines(since_ms: Optional[int] = None,
                  until_ms: Optional[int] = None) -> Dict[str, Any]:
    """4.0.418 — OEE of every LINE: a chain of stations linked by next_watcher (set in
    the Hardware tab), ending at a station with no next station. Several stations may
    feed one (e.g. two IS machines → one Lehr). All counts are units (pulses × the
    station's OK / NG factor).

    Per station:  input = OK + NG;  quality = OK / input;
                  gap   = OK of the station(s) feeding it − its own input
                          (> 0: units that left upstream good but never arrived —
                           broken / removed on the way; < 0: it counted more than it
                           was sent, e.g. stock fed in by hand)
    Per line:     input  = input of the first station(s);  output = OK of the last;
                  Quality      = output / input  (every NG and every gap on the way)
                  Availability = the least available station (a stop anywhere stops
                                 the line; min, not a product, because one stop makes
                                 the stations around it idle too)
                  Performance  = the first station(s) — they set the line's pace
                  OEE          = A × P × Q
    Units still travelling between stations (e.g. inside a Lehr) look like a gap in a
    short window — use a window much longer than the line's transit time."""
    _ensure_schema()
    reg_rows = _exec("SELECT register_id, name, next_watcher FROM watcher_registry",
                     fetch="all") or []
    name = {r: (n or r) for r, n, _ in reg_rows}
    nxt = {r: x for r, _, x in reg_rows if x and x in name and x != r}
    feeders: Dict[str, List[str]] = {}
    for a, b in nxt.items():
        feeders.setdefault(b, []).append(a)

    def _terminal(r: str) -> str:
        seen = set()
        while r in nxt and r not in seen:
            seen.add(r)
            r = nxt[r]
        return r

    groups: Dict[str, List[str]] = {}
    for r in name:
        if r in nxt or r in feeders:
            groups.setdefault(_terminal(r), []).append(r)

    lines = []
    for term, members in groups.items():
        depth: Dict[str, int] = {}

        def _depth(r: str, trail=()) -> int:
            if r not in depth:
                depth[r] = max((_depth(f, trail + (r,)) + 1 for f in feeders.get(r, [])
                                if f not in trail), default=0)
            return depth[r]

        for r in members:
            _depth(r)
        order = sorted(members, key=lambda r: (depth[r], name[r].lower()))
        st = {r: compute_oee(r, since_ms, until_ms) for r in order}
        good = {r: float(st[r].get("good") or 0) for r in order}
        bad = {r: float(st[r].get("defect") or 0) for r in order}

        stations, missing = [], set()
        for r in order:
            o = st[r]
            inp = good[r] + bad[r]
            fed = feeders.get(r, [])
            gap = (sum(good[f] for f in fed) - inp) if fed else None
            if not o.get("samples"):
                missing.add(f"no data from {name[r]} in this window")
            missing.update(o.get("missing") or [])
            stations.append({
                "register_id": r, "name": name[r], "depth": depth[r],
                "next_watcher": nxt.get(r), "feeders": fed,
                "ok": round(good[r], 2), "ng": round(bad[r], 2), "input": round(inp, 2),
                "quality": round(good[r] / inp, 4) if inp > 0 else None,
                "gap": round(gap, 2) if gap is not None else None,
                "availability": o.get("availability"), "performance": o.get("performance"),
                "oee": o.get("oee"), "samples": o.get("samples", 0),
                "ok_factor": o.get("ok_factor"), "ng_factor": o.get("ng_factor"),
            })

        heads = [r for r in order if depth[r] == 0]
        line_in = sum(good[r] + bad[r] for r in heads)
        line_out = good[term]
        gaps = [s["gap"] for s in stations if s["gap"] is not None]
        quality = max(0.0, min(1.0, line_out / line_in)) if line_in > 0 else None
        avs = [s["availability"] for s in stations if s["availability"] is not None]
        availability = min(avs) if avs else 1.0
        pw = [(s["performance"], s["input"]) for s in stations
              if s["depth"] == 0 and s["performance"] is not None]
        wsum = sum(w for _, w in pw)
        performance = (sum(p * w for p, w in pw) / wsum) if wsum > 0 else (pw[0][0] if pw else 1.0)
        oee = availability * performance * quality if quality is not None else None

        levels: Dict[int, List[str]] = {}
        for r in order:
            levels.setdefault(depth[r], []).append(name[r])
        lines.append({
            "name": " → ".join(" + ".join(v) for _, v in sorted(levels.items())),
            "terminal": term, "stations": stations,
            "input": round(line_in, 2), "output": round(line_out, 2),
            "ng": round(sum(bad.values()), 2),
            "lost_between": round(sum(g for g in gaps if g > 0), 2),
            "extra_between": round(-sum(g for g in gaps if g < 0), 2),
            "quality": round(quality, 4) if quality is not None else None,
            "availability": round(availability, 4), "performance": round(performance, 4),
            "oee": round(oee, 4) if oee is not None else None,
            "missing": sorted(missing),
        })
    lines.sort(key=lambda x: x["name"].lower())
    return {"since_ms": since_ms, "until_ms": until_ms, "lines": lines}


# --------------------------------------------------------------------------- #
# camera stations (4.0.419)
# --------------------------------------------------------------------------- #
# A camera station is a station whose OK / NG come from MVE's own detections: OK =
# detections of the chosen OK object(s), NG = detections of the chosen NG object(s),
# on the chosen cameras (none chosen = all). Once a minute MVE counts what is new and
# files it as an ordinary watcher report, so OEE, lines, gaps, charts and the online
# dot treat it exactly like a Watcher Jet. Only detections that are stored count
# (class Store ON, at or above its min confidence), and every detection counts — MVE
# does not track objects, so one object seen in two frames counts twice: a true unit
# count only when each object is captured once (triggered capture); with periodic
# capture the counts are a sample (Quality is still right).
CAMERA_COUNT_INTERVAL_SEC = 60
CAMERA_COUNT_LAG_SEC = 60                    # detections reach the DB via a batch writer
CAMERA_COUNT_MAX_CATCHUP = timedelta(hours=1)
_counter_started = False
_count_lock = threading.Lock()               # one counting pass at a time


def _clean_list(v: Any) -> List[str]:
    out: List[str] = []
    for x in (v or []):
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def camera_options() -> Dict[str, Any]:
    """Cameras and object classes to build a camera station from. A class is
    `stored` when its Store switch is ON — only stored detections can be counted."""
    _ensure_schema()
    from config import load_service_config
    svc = load_service_config() or {}
    cameras = sorted(({"id": int(k), "name": (c or {}).get("name") or f"Camera {k}"}
                      for k, c in (svc.get("cameras") or {}).items() if str(k).isdigit()),
                     key=lambda c: c["id"])
    store = svc.get("store_objects") or {}
    seen = {r[0] for r in (_exec("""
        SELECT DISTINCT d->>'name' FROM
          (SELECT detections FROM inference_results ORDER BY time DESC LIMIT 2000) r
        CROSS JOIN LATERAL jsonb_array_elements(
          CASE WHEN jsonb_typeof(r.detections) = 'array' THEN r.detections ELSE '[]'::jsonb END) d
    """, fetch="all") or []) if r[0]}
    names = sorted((set(store) | seen) - {"_color"}, key=str.lower)
    return {"cameras": cameras,
            "classes": [{"name": n, "stored": bool(store.get(n, False)), "seen_recently": n in seen}
                        for n in names]}


def save_camera_station(register_id: Optional[str], name: str, cameras: Any,
                        ok_classes: Any, ng_classes: Any) -> Dict[str, Any]:
    """Create (no register_id) or update a camera station."""
    _ensure_schema()
    name = str(name or "").strip()
    if not name:
        raise ValueError("station name required")
    ok, ng = _clean_list(ok_classes), _clean_list(ng_classes)
    if not ok and not ng:
        raise ValueError("choose at least one OK or NG object")
    both = sorted(set(ok) & set(ng))
    if both:
        raise ValueError(f"{', '.join(both)} cannot be both OK and NG")
    try:
        cams = sorted({int(c) for c in (cameras or [])})
    except (TypeError, ValueError):
        raise ValueError("cameras must be camera ids")
    conf = {"cameras": cams, "ok_classes": ok, "ng_classes": ng}
    reg = str(register_id or "").strip()
    if reg:
        row = _exec("SELECT cam_config FROM watcher_registry WHERE register_id=%s", (reg,), fetch="one")
        if not row or row[0] is None:
            raise ValueError(f"'{reg}' is not a camera station")
    else:
        base = "CAM-" + (re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").upper() or "STATION")[:40]
        reg, n = base, 2
        while _exec("SELECT 1 FROM watcher_registry WHERE register_id=%s", (reg,), fetch="one"):
            reg, n = f"{base}-{n}", n + 1
        _exec("INSERT INTO watcher_registry (register_id) VALUES (%s)", (reg,))
    _exec("UPDATE watcher_registry SET name=%s, cam_config=%s, wtype='camera', assigned=TRUE "
          "WHERE register_id=%s", (name, json.dumps(conf), reg))
    return {"ok": True, "register_id": reg, "camera_config": conf}


def _count_detections(conf: Dict[str, Any], t0: datetime, t1: datetime) -> tuple:
    cams = [str(c) for c in (conf.get("cameras") or [])]
    r = _exec("""
        SELECT COALESCE(SUM(CASE WHEN d->>'name' = ANY(%s::text[]) THEN 1 ELSE 0 END), 0),
               COALESCE(SUM(CASE WHEN d->>'name' = ANY(%s::text[]) THEN 1 ELSE 0 END), 0)
        FROM inference_results r
        CROSS JOIN LATERAL jsonb_array_elements(
          CASE WHEN jsonb_typeof(r.detections) = 'array' THEN r.detections ELSE '[]'::jsonb END) d
        WHERE r.time > %s AND r.time <= %s
          AND (%s OR substring(r.image_path from '_p[0-9]+_([0-9]+)\\.jpg$') = ANY(%s::text[]))
    """, (list(conf.get("ok_classes") or []), list(conf.get("ng_classes") or []),
          t0, t1, not cams, cams), fetch="one")
    return (int(r[0]), int(r[1])) if r else (0, 0)


def count_camera_stations() -> List[Dict[str, Any]]:
    """One counting pass: for every camera station, count the detections stored since
    the last pass (up to 1 min ago — the DB writer is asynchronous) and file them as a
    report. A new station starts from now (no back-fill); after a restart it resumes
    where it stopped, at most 1 h back."""
    with _count_lock:
        return _count_camera_stations()


def _count_camera_stations() -> List[Dict[str, Any]]:
    _ensure_schema()
    rows = _exec("SELECT register_id, cam_config, cam_last_ts FROM watcher_registry "
                 "WHERE cam_config IS NOT NULL", fetch="all") or []
    if not rows:
        return []
    t1 = _exec("SELECT NOW() - make_interval(secs => %s)", (CAMERA_COUNT_LAG_SEC,), fetch="one")[0]
    out = []
    for reg, conf, last in rows:
        conf = _as_dict(conf)
        if last is None:
            t0 = t1                                   # baseline: count from now on
        else:
            t0 = max(last, t1 - CAMERA_COUNT_MAX_CATCHUP)
        if t0 > t1:
            continue
        ok, ng = _count_detections(conf, t0, t1) if t0 < t1 else (0, 0)
        # claim the window first; if the station was deleted meanwhile, file nothing
        if not _exec("UPDATE watcher_registry SET cam_last_ts=%s WHERE register_id=%s "
                     "AND cam_config IS NOT NULL RETURNING register_id", (t1, reg), fetch="one"):
            continue
        ingest({"register_id": reg, "quantity": ok, "defect_quantity": ng,
                "timestamp": t1.isoformat(), "extra_info": {"source": "camera"}})
        out.append({"register_id": reg, "ok": ok, "ng": ng,
                    "from": t0.isoformat(), "to": t1.isoformat()})
    return out


def start_camera_counter() -> None:
    """Start the once-a-minute camera-station counter (idempotent, daemon thread)."""
    global _counter_started
    if _counter_started:
        return
    _counter_started = True

    def _loop():
        time.sleep(30)
        while True:
            try:
                count_camera_stations()
            except Exception as e:
                logger.warning("camera station count failed: %s", e)
            time.sleep(CAMERA_COUNT_INTERVAL_SEC)

    threading.Thread(target=_loop, name="camera-station-counter", daemon=True).start()
    logger.info("camera-station counter started (every %ss)", CAMERA_COUNT_INTERVAL_SEC)
