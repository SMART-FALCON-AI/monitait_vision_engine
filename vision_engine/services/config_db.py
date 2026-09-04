"""DB-backed `service_config` storage (3.22.0).

Until 3.21.x, the full data file (.env.prepared_query_data) was the only
source of truth for MVE configuration. That meant:
  - whole-file rewrite on every change (50 KB+ json round-trip)
  - no atomic per-key updates (concurrent writers can clobber each other)
  - no audit trail (who changed what when)
  - no way to push a config from a central console to multiple sites

3.22.0 moves the data into a Postgres table (`mve_config_kv`), keyed by
top-level field. Existing sites keep their JSON files — those become the
backward-compat fallback AND a snapshot dumped after every successful save.

Backward-compat rules:
  - READ:  try DB; if DB unreachable or table empty, fall back to file.
  - WRITE: write DB; *always* also dump file snapshot so legacy tooling
           that reads `.env.prepared_query_data` directly keeps working.
  - MIGRATION: on startup, if DB is empty AND the file has content, the
           current `load_data_file()` returns file content and the first
           subsequent `save_data_file()` bootstraps the DB. An explicit
           startup call (see main.py) handles the read-only case where
           nothing ever calls save during a fresh boot.

Table:
  mve_config_kv (
    key         TEXT PRIMARY KEY,         -- top-level field name
    value       JSONB NOT NULL,           -- the whole sub-tree
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_by  TEXT                      -- "mve" / "migration" / "operator:smahd" / etc
  )
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def _conn():
    """Get a pooled DB connection. None if unreachable."""
    try:
        from services.db import get_db_connection
        return get_db_connection()
    except Exception as e:
        logger.debug(f"config_db: cannot reach Postgres: {e}")
        return None


def _release(conn) -> None:
    if conn is None:
        return
    try:
        from services.db import release_db_connection
        release_db_connection(conn)
    except Exception:
        pass


_schema_ready = False

# 4.0.391 ONE-KEY — the config lives in exactly ONE row. Previously every top-level
# key was its own row, so the store could hold data the boot path never reads: boot
# only ever looked at `service_config` + `timeline_config`, and anything else (a
# `states` row, a `cameras` row…) was invisible forever. That is precisely how a
# site could show default settings while the real ones sat in the database. One row
# makes that class of bug unrepresentable.
CONFIG_KEY = "mve_config"

# The two keys the old layout was ALLOWED to have. Anything else found at the top
# level of the legacy table was a stray created by an unwrapped save_data_file().
_LEGACY_CANONICAL = ("service_config", "timeline_config")


def _ensure_schema(conn) -> bool:
    """Create the table if missing. Returns True on success.

    4.0.321 CONFIG-CACHE — after the first success this SHORT-CIRCUITS. Previously the
    `CREATE TABLE IF NOT EXISTS … ; COMMIT` ran on EVERY load_all/save_all, i.e. on every
    config read — and the config hot paths (the SSE stream, /api/status, the encoder tick)
    read config many times per second, so this was a DDL + COMMIT round-trip storm for no
    reason. The table is created once; every later call is now a no-op.
    """
    global _schema_ready
    if _schema_ready:
        return True
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mve_config_kv (
                key         TEXT PRIMARY KEY,
                value       JSONB NOT NULL,
                updated_at  TIMESTAMPTZ DEFAULT NOW(),
                updated_by  TEXT
            );
            """
        )
        conn.commit()
        cur.close()
        _schema_ready = True
        return True
    except Exception as e:
        logger.warning(f"config_db: ensure_schema failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def is_db_empty() -> Optional[bool]:
    """True if mve_config_kv has zero rows. None if DB unreachable
    (so the caller can defer the decision rather than guess)."""
    conn = _conn()
    if conn is None:
        return None
    try:
        if not _ensure_schema(conn):
            return None
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mve_config_kv")
        (n,) = cur.fetchone()
        cur.close()
        return int(n or 0) == 0
    except Exception as e:
        logger.warning(f"config_db: is_db_empty failed: {e}")
        return None
    finally:
        _release(conn)


def load_all() -> Optional[Dict[str, Any]]:
    """Return the whole config bundle from its single row.

    Returns the bundle dict on success, {} when the store is genuinely empty, and
    None when the DB is UNREACHABLE — callers rely on that distinction to fall back
    to disk without treating "down" as "empty".

    On first run against a pre-4.0.391 database this migrates the old multi-row
    layout into the single row, in one transaction. Stray rows (anything beyond
    service_config / timeline_config) are NOT merged blindly — guessing which copy
    of `states` is authoritative could destroy the good one — they are left in place
    and logged loudly so they can be recovered deliberately.
    """
    conn = _conn()
    if conn is None:
        return None
    try:
        if not _ensure_schema(conn):
            return None
        cur = conn.cursor()
        cur.execute("SELECT value FROM mve_config_kv WHERE key = %s", (CONFIG_KEY,))
        row = cur.fetchone()
        if row is not None:
            cur.close()
            v = row[0]
            if isinstance(v, (str, bytes)):
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            return v if isinstance(v, dict) else {}

        # --- no single row yet: legacy layout (or a fresh, empty store) ---
        cur.execute("SELECT key, value FROM mve_config_kv")
        rows = cur.fetchall()
        if not rows:
            cur.close()
            return {}

        bundle: Dict[str, Any] = {}
        strays = []
        for k, v in rows:
            if isinstance(v, (str, bytes)):
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            k = str(k)
            if k in _LEGACY_CANONICAL:
                bundle[k] = v
            else:
                strays.append(k)

        cur.execute(
            """
            INSERT INTO mve_config_kv (key, value, updated_by)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW(), updated_by = EXCLUDED.updated_by
            """,
            (CONFIG_KEY, json.dumps(bundle), "migration:4.0.391"),
        )
        conn.commit()
        cur.close()
        logger.warning("[ONE-KEY] migrated legacy config rows %s into the single '%s' row",
                       sorted(bundle.keys()), CONFIG_KEY)
        if strays:
            logger.error(
                "[ONE-KEY] %d STRAY config row(s) left untouched: %s — these were written by an "
                "unwrapped save_data_file() and the boot path never read them. Inspect before "
                "deleting; they may hold settings that went missing.", len(strays), sorted(strays))
        return bundle
    except Exception as e:
        logger.warning(f"config_db: load_all failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        _release(conn)


def save_all(data: Dict[str, Any], updated_by: str = "mve") -> bool:
    """Replace the ENTIRE config bundle in its single row. Returns True on success.

    4.0.391 ONE-KEY — this now REPLACES rather than upserting key-by-key, so `data`
    must always be the complete bundle. config.py::_merge_and_persist guarantees
    that by merging the caller's partial onto the current authoritative bundle
    before calling here, and writing that same object to disk — which is what keeps
    the two stores byte-identical instead of drifting apart.
    """
    if not isinstance(data, dict):
        return False
    conn = _conn()
    if conn is None:
        return False
    try:
        if not _ensure_schema(conn):
            return False
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO mve_config_kv (key, value, updated_by)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (key) DO UPDATE
                SET value      = EXCLUDED.value,
                    updated_at = NOW(),
                    updated_by = EXCLUDED.updated_by
            """,
            (CONFIG_KEY, json.dumps(data), updated_by),
        )
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        logger.warning(f"config_db: save_all failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        _release(conn)


def history_count() -> Optional[int]:
    """How many keys are currently stored. For ops/dashboards."""
    conn = _conn()
    if conn is None:
        return None
    try:
        if not _ensure_schema(conn):
            return None
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mve_config_kv")
        (n,) = cur.fetchone()
        cur.close()
        return int(n or 0)
    except Exception:
        return None
    finally:
        _release(conn)
