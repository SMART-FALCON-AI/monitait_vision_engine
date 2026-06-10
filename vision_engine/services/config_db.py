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


def _ensure_schema(conn) -> bool:
    """Create the table if missing. Returns True on success."""
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
    """Read every (key, value) row. Returns dict on success; None if DB
    unreachable so the caller can fall back to file."""
    conn = _conn()
    if conn is None:
        return None
    try:
        if not _ensure_schema(conn):
            return None
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM mve_config_kv")
        rows = cur.fetchall()
        cur.close()
        if not rows:
            return {}
        out: Dict[str, Any] = {}
        for k, v in rows:
            # psycopg2 hands JSONB back as already-parsed python; tolerate
            # the rare case where it didn't.
            if isinstance(v, (str, bytes)):
                try:
                    v = json.loads(v)
                except Exception:
                    pass
            out[str(k)] = v
        return out
    except Exception as e:
        logger.warning(f"config_db: load_all failed: {e}")
        return None
    finally:
        _release(conn)


def save_all(data: Dict[str, Any], updated_by: str = "mve") -> bool:
    """Upsert every top-level key from `data`. Returns True on success.

    Atomicity: all upserts run in a single transaction — either every key
    lands or none do. Operator-visible writes (`POST /api/audio_settings`,
    Save All Configuration, etc.) all funnel through `save_data_file` →
    here, so the DB always sees a consistent snapshot.
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
        for k, v in data.items():
            cur.execute(
                """
                INSERT INTO mve_config_kv (key, value, updated_by)
                VALUES (%s, %s::jsonb, %s)
                ON CONFLICT (key) DO UPDATE
                    SET value      = EXCLUDED.value,
                        updated_at = NOW(),
                        updated_by = EXCLUDED.updated_by
                """,
                (str(k), json.dumps(v), updated_by),
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
