"""Idempotent startup migrations. Each function returns True if it changed
something so the caller can decide whether to persist.

Add new migrations in chronological order. They run unconditionally on
every startup; design them to be no-ops when the migration has already
been applied.
"""

import logging

logger = logging.getLogger(__name__)


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
