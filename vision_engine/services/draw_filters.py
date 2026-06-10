"""Single source of truth for "should we draw this detection?".

3.21.26 architecture: this module owns the audio_settings read. Callers
just ask `should_draw_class(name)` — they never pass audio_settings,
object_filters, or any other config. ONE place reads service_config;
ONE place caches it; ONE place gets cache-busted when config changes.

Before this refactor the rule lived in `should_draw_class(name, audio_settings, obj_filters)`
and four different callers had to remember to pass `audio_settings`. The
recurring bug: one of them forgot and the rule defaulted to "skip" silently.

Rule (3.21.25+, unchanged): `audio_settings.<class>.show == True` → draw.
Anything else (False, None, missing entry) → skip.

Companion: `services/detection.py::_auto_register_classes()` writes a
`show=False` stub the first time a class is detected, so the Process tab
shows every class as an unticked card. Operators opt in with one click.

Cache: invalidated on every POST to `/api/audio_settings`,
`/api/store_objects`, the bulk save endpoint, and any explicit
`/api/draw_filters/invalidate` call. Also self-invalidates after
`_TTL_SEC` as a backstop.
"""

import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# In-memory cache of audio_settings. Refreshed on demand.
_cache: Dict[str, Dict[str, Any]] = {}
_cache_loaded_at: float = 0.0
_TTL_SEC: float = 5.0   # backstop refresh — cache-bust calls handle the hot path


def invalidate_cache() -> None:
    """Drop the audio_settings cache. Called by any endpoint that mutates
    service_config so the next draw decision picks up the new value
    immediately (no waiting for the TTL backstop)."""
    global _cache_loaded_at
    _cache_loaded_at = 0.0


def _load_audio_settings() -> Dict[str, Dict[str, Any]]:
    """Return a fresh-or-cached copy of service_config.audio_settings."""
    global _cache, _cache_loaded_at
    now = time.time()
    if _cache and now - _cache_loaded_at < _TTL_SEC:
        return _cache
    try:
        import config as cfg
        svc = cfg.load_service_config() or {}
        _cache = svc.get("audio_settings", {}) or {}
    except Exception as e:
        logger.warning(f"draw_filters could not load audio_settings: {e}")
        _cache = _cache or {}
    _cache_loaded_at = now
    return _cache


def should_draw_class(name: str) -> bool:
    """Return True if the named class should be drawn on annotated frames.

    Rule: `audio_settings[name].show == True` → True. Everything else
    (no entry, show=False, show=None, malformed entry) → False.
    """
    if not name:
        return False
    au = _load_audio_settings().get(name)
    return isinstance(au, dict) and au.get("show") is True


def min_confidence_for(name: str, default: float = 0.01) -> float:
    """Per-class min_confidence threshold from audio_settings; defaults
    to 0.01 if unset (matches the historical render.py default)."""
    if not name:
        return default
    au = _load_audio_settings().get(name)
    if isinstance(au, dict) and "min_confidence" in au:
        try:
            return float(au["min_confidence"])
        except (TypeError, ValueError):
            pass
    return default
