"""Single source of truth for the "should we draw this detection?" decision.

Prior to 3.21.21 the show / min_confidence rule was duplicated in three
places (`services/detection.py` annotator, `services/render.py`
draw_detection_on helper, `services/watcher.py` timeline composite), each
reading from a different mix of `audio_settings.show` and
`object_filters.show`. The two dicts could drift apart, and a class present
in only one dict was handled inconsistently — that's what produced the
fabriqc-kc agh/tooli_up bug, then the PVB math-flood regression in 3.21.19.

The rule (3.21.21+):

1. `audio_settings` is the canonical source. The Process tab "Show" toggle
   writes here. If the class has an entry, that entry's `show` wins
   (True = draw, False = skip).

2. `object_filters` is the legacy fallback. The Advanced tab's "Apply
   Timeline Configuration" button is the only thing that writes it.
   Consulted only when `audio_settings` has no entry for this class.

3. If neither dict has an entry for the class, it is NOT drawn. Conservative
   default — protects against math channels (which emit hundreds of detection
   names with no operator configuration) flooding the annotated image.

Both lookups are exact-match by class name. There's no fuzzy / prefix
matching.
"""

from typing import Optional, Dict, Any


def should_draw_class(
    name: str,
    audio_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    obj_filters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Return True if `name` should be drawn on the annotated image."""
    if not name:
        return False
    audio_settings = audio_settings or {}
    obj_filters = obj_filters or {}

    au = audio_settings.get(name)
    if isinstance(au, dict) and "show" in au:
        # Canonical entry exists — its `show` wins.
        return bool(au["show"])

    of = obj_filters.get(name)
    if isinstance(of, dict) and "show" in of:
        # Legacy fallback.
        return bool(of["show"])

    # No entry in either dict — default to "don't draw" (protects against
    # math-channel name explosions).
    return False


def min_confidence_for(
    name: str,
    audio_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    obj_filters: Optional[Dict[str, Dict[str, Any]]] = None,
    default: float = 0.01,
) -> float:
    """Return the per-class min_confidence threshold, with the same dict
    precedence as `should_draw_class`. Defaults to 0.01 if neither dict
    has a value (matches the historical render.py / watcher.py default).
    """
    if not name:
        return default
    audio_settings = audio_settings or {}
    obj_filters = obj_filters or {}

    au = audio_settings.get(name)
    if isinstance(au, dict) and "min_confidence" in au:
        try:
            return float(au["min_confidence"])
        except (TypeError, ValueError):
            pass

    of = obj_filters.get(name)
    if isinstance(of, dict) and "min_confidence" in of:
        try:
            return float(of["min_confidence"])
        except (TypeError, ValueError):
            pass

    return default
