"""Single source of truth for drawing detections onto an image.

All five render paths in MVE (timeline grid, timeline single-frame popup,
multi-camera live stream, single-camera live stream, websocket push, and the
on-disk *_DETECTED.jpg) call into this module so the bbox/label/kv-overlay
rules are defined exactly ONCE.

Rule
----
  If bbox covers >= 90% of the image area, the detection is rendered as a
  "name : conf" text line in the top-left of the image — no rectangle, no
  per-bbox label. Otherwise, draw the original green rectangle + label.

This keeps the image clean for global-scalar channels (math module's
mean_L, std_L, fft_*, etc. that all share full-frame bboxes) while still
giving spatial bboxes to localized detections (YOLO classes, residuals,
blobs, etc.).
"""
from __future__ import annotations

import cv2
from typing import Any, Dict, Optional

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_RECT_COLOR = (0, 255, 0)        # green
_LABEL_FG = (0, 0, 0)            # black on green bg
_KV_BG = (0, 0, 0)               # black slab behind kv text
_KV_FG = (0, 255, 0)             # green text on black slab


def auto_font_scale(img_h: int) -> float:
    """Pick a font scale that stays readable across thumbnail and full-res."""
    return max(0.5, min(1.2, img_h / 480.0))


def draw_detection_on(
    img,
    det: Dict[str, Any],
    sx: float = 1.0,
    sy: float = 1.0,
    kv_y: int = 4,
    bbox_thickness: int = 2,
    label_thickness: int = 2,
    font_scale: Optional[float] = None,
    obj_filters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    """Draw one detection on `img` (mutates in place). Returns the next
    available y for the kv-overlay column so the caller can chain.

    Args
    ----
    img : numpy ndarray (BGR)
    det : detection dict {name, confidence, xmin, ymin, xmax, ymax, ...}
    sx, sy : scale factors when det coordinates are in original-image space
             and `img` is a thumbnail. Use 1.0 when the image is at the same
             resolution the analyzer saw.
    kv_y : current top-y for the kv overlay column. Pass the returned value
           into the next call to chain rows.
    bbox_thickness : rectangle stroke for non-kv detections (3 looks better
                     on full-res, 2 on thumbnails).
    label_thickness : putText stroke. 2 is readable on JPEG.
    font_scale : override; if None, auto-size by image height.
    obj_filters : timeline_config.object_filters mapping. Hides any det whose
                  filter has show:false or whose confidence is below the
                  per-filter min_confidence (default 0.01).

    Returns
    -------
    int : updated kv_y to feed into the next call.
    """
    try:
        name = det.get('name', '')
        confidence = float(det.get('confidence', 0) or 0)
        of = (obj_filters or {}).get(name, {})
        if of.get('show') is False:
            return kv_y
        if confidence < of.get('min_confidence', 0.01):
            return kv_y
        x1 = int(det.get('xmin', det.get('x1', 0)) * sx)
        y1 = int(det.get('ymin', det.get('y1', 0)) * sy)
        x2 = int(det.get('xmax', det.get('x2', 0)) * sx)
        y2 = int(det.get('ymax', det.get('y2', 0)) * sy)

        h, w = img.shape[:2]
        fs = auto_font_scale(h) if font_scale is None else font_scale

        # The ONE rule: bbox covers >= 90% of image area -> kv text only.
        if (x2 - x1) * (y2 - y1) >= 0.9 * h * w:
            text = f"{name} : {confidence:.2f}"
            (kw, kh), _ = cv2.getTextSize(text, _FONT, fs, label_thickness)
            cv2.rectangle(img, (2, kv_y),
                          (2 + kw + 6, kv_y + kh + 6), _KV_BG, -1)
            cv2.putText(img, text, (5, kv_y + kh + 2),
                        _FONT, fs, _KV_FG, label_thickness, cv2.LINE_AA)
            return kv_y + kh + 8

        # Else: original rectangle + label.
        cv2.rectangle(img, (x1, y1), (x2, y2), _RECT_COLOR, bbox_thickness)
        label = f"{name} {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, _FONT, fs, label_thickness)
        cv2.rectangle(img, (x1, y1 - lh - 6),
                      (x1 + lw + 6, y1), _RECT_COLOR, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 3),
                    _FONT, fs, _LABEL_FG, label_thickness, cv2.LINE_AA)
        return kv_y
    except Exception:
        return kv_y
