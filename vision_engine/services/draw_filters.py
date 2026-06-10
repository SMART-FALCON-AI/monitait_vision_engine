"""Single source of truth for the "should we draw this detection?" decision.

3.21.21 introduced this helper to consolidate the rule that used to be
duplicated in three files (detection.py annotator, render.py
draw_detection_on, watcher.py timeline composite). At that point the
helper read both `audio_settings` and a legacy `object_filters` dict,
with audio_settings preferred and object_filters as a fallback for
classes that had no audio_settings entry.

3.21.22 simplifies the rule further: audio_settings is the ONLY source
of truth. The UI no longer writes to object_filters (see audio.js +
routers/timeline.py for the matching changes), and we run a one-time
startup migration (services/migrations.py::migrate_object_filters_into_audio_settings)
that copies any remaining object_filters entries into audio_settings
before the annotator runs.

Rule:
- audio_settings[name].show == True   → draw
- audio_settings[name].show == False  → skip
- audio_settings has no entry for the class → skip
  (conservative default — protects against model classes that aren't
   yet configured from spamming the annotated image)

The `obj_filters` keyword argument is kept for signature compatibility
with the three call sites and any external callers, but its value is
ignored. Remove the argument entirely once 3.21.22 has burned in.
"""

from typing import Optional, Dict, Any


def should_draw_class(
    name: str,
    audio_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    obj_filters: Optional[Dict[str, Dict[str, Any]]] = None,  # ignored; see module docstring
) -> bool:
    """Return True if `name` should be drawn on the annotated image."""
    if not name:
        return False
    audio_settings = audio_settings or {}

    au = audio_settings.get(name)
    if isinstance(au, dict) and "show" in au:
        return bool(au["show"])

    # No entry → don't draw (conservative default).
    return False


def min_confidence_for(
    name: str,
    audio_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    obj_filters: Optional[Dict[str, Dict[str, Any]]] = None,  # ignored; see module docstring
    default: float = 0.01,
) -> float:
    """Return the per-class min_confidence threshold from audio_settings."""
    if not name:
        return default
    audio_settings = audio_settings or {}

    au = audio_settings.get(name)
    if isinstance(au, dict) and "min_confidence" in au:
        try:
            return float(au["min_confidence"])
        except (TypeError, ValueError):
            pass

    return default
