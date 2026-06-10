"""Idempotent startup migrations. Each function returns True if it changed
something so the caller can decide whether to persist.

Add new migrations in chronological order. They run unconditionally on
every startup; design them to be no-ops when the migration has already
been applied.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def migrate_data_file_into_db() -> bool:
    """3.22.0 — bootstrap the `mve_config_kv` table from the legacy
    `.env.prepared_query_data` file on first run after upgrade.

    Idempotent: if the table already has rows, this is a no-op. If
    Postgres is unreachable, we skip silently and the file remains the
    source of truth until the DB is back.

    Returns True if we actually copied data in (so the caller can log).
    """
    try:
        from services.config_db import is_db_empty, save_all
    except Exception as e:
        logger.debug(f"config_db not importable; skipping DB migration: {e}")
        return False

    empty = is_db_empty()
    if empty is None:
        # DB unreachable — try again next startup.
        logger.debug("3.22 migration: DB unreachable, leaving file as source of truth")
        return False
    if not empty:
        # Table already populated — nothing to do.
        return False

    # Read the legacy file directly. NOTE: we don't call load_data_file()
    # because it would route back through the DB (which we know is empty).
    try:
        from config import DATA_FILE
    except Exception:
        DATA_FILE = ".env.prepared_query_data"

    if not os.path.exists(DATA_FILE):
        logger.info("3.22 migration: DB empty AND no file — clean install, nothing to migrate")
        return False

    try:
        with open(DATA_FILE, "r") as f:
            file_data = json.load(f)
    except Exception as e:
        logger.warning(f"3.22 migration: could not read {DATA_FILE}: {e}")
        return False

    if not isinstance(file_data, dict) or not file_data:
        logger.info("3.22 migration: file empty or non-dict; nothing to migrate")
        return False

    if save_all(file_data, updated_by="migration:3.22.0"):
        logger.info(
            f"3.22 migration: copied {len(file_data)} top-level key(s) from "
            f"{DATA_FILE} into mve_config_kv: {sorted(file_data.keys())}"
        )
        return True
    logger.warning("3.22 migration: save_all to DB failed; will retry next startup")
    return False


def migrate_object_filters_into_audio_settings(svc: dict, data: dict) -> bool:
    """3.21.22 — fold legacy `timeline_config.object_filters` into
    `service_config.audio_settings` so the annotator can read a single
    source of truth.

    For each class found in object_filters but missing from audio_settings,
    create an audio_settings entry that preserves the legacy show /
    min_confidence values. Classes that already exist in audio_settings
    are left alone (audio_settings is canonical and may carry richer
    fields like severity that object_filters never had).

    `svc` is the service_config dict (carries audio_settings). `data` is
    the full data file (carries timeline_config). Both are mutated in
    place. The caller persists them.

    Returns True if anything changed.
    """
    tc = (data.get("timeline_config", {}) or {})
    of = tc.get("object_filters", {}) or {}
    if not isinstance(of, dict) or not of:
        return False

    aud = svc.get("audio_settings", {}) or {}
    if not isinstance(aud, dict):
        aud = {}

    added = 0
    for cls, ent in of.items():
        if not isinstance(ent, dict):
            continue
        if cls in aud and isinstance(aud[cls], dict):
            continue  # audio_settings wins; no overwrite

        try:
            mc = float(ent.get("min_confidence", 0.01))
        except (TypeError, ValueError):
            mc = 0.01
        aud[cls] = {
            "show": bool(ent.get("show", True)),
            "narrate": False,
            "beep": False,
            "min_confidence": mc,
            "severity": 0,
        }
        added += 1

    if added:
        svc["audio_settings"] = aud
        logger.info(
            "migration: copied %d class(es) from timeline_config.object_filters "
            "into audio_settings", added
        )
        return True
    return False
