"""Faster JPEG encode/decode via libjpeg-turbo when available.

4.0.52 — Three MVE hot paths were spending measurable CPU on JPEG codec:
  * services/watcher.py:1405 (stream_results — MJPEG dashboard preview)
  * services/watcher.py:319  (disk writer thread — cv2.imwrite)
  * services/detection.py:~590 (timeline thumbnail encode)

Each of them was calling ``cv2.imencode('.jpg', ..., quality)`` which uses
OpenCV's bundled *stock libjpeg* — the reference codec, correct but slow.
libjpeg-turbo replaces its inner loops with SIMD (SSE2/AVX2/NEON) and
gives a ~3-4× speedup on the same input, same quality. Same JPEG bytes
downstream — the format is unchanged.

Design goals of this module:
  * Zero import-time crash if libturbojpeg0 isn't installed on the host.
    We fall back to cv2 automatically so no site is broken by a missing
    apt package.
  * One TurboJPEG instance shared across the process (constructor cost is
    small but non-zero; recreating per encode would kill the win).
  * Same call signature style as cv2.imencode so migration is a single
    line per call site.

Consumers:

    from services.jpeg_codec import encode_jpeg, decode_jpeg, HAS_TURBO
    data = encode_jpeg(frame_bgr, quality=85)
    img  = decode_jpeg(jpeg_bytes)

If you want to know which codec fired for a given call site (metrics /
health probe) inspect the module-level ``HAS_TURBO`` flag.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---- Backend discovery -----------------------------------------------------

_backend = None            # "simplejpeg" | "pyturbojpeg" | None
_tj = None                # PyTurboJPEG instance if that's the backend
HAS_TURBO = False

# 4.0.52 — Try simplejpeg FIRST because it ships pre-built manylinux wheels
# on PyPI (bundles its own copy of libjpeg-turbo, zero apt work needed).
# PyTurboJPEG is source-only and its setup.py fails to build in some
# container environments (metadata-generation-failed on egg_info's
# find_sources), so it's only a fallback.
try:
    import simplejpeg as _sj  # type: ignore
    _backend = "simplejpeg"
    HAS_TURBO = True
    logger.info("jpeg_codec: using libjpeg-turbo via simplejpeg")
except Exception:
    pass

if _backend is None:
    try:
        from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_420  # type: ignore
        _tj = TurboJPEG()
        _backend = "pyturbojpeg"
        HAS_TURBO = True
        logger.info("jpeg_codec: using libjpeg-turbo via PyTurboJPEG")
    except Exception:
        pass

if _backend is None:
    HAS_TURBO = False
    logger.info(
        "jpeg_codec: no libjpeg-turbo backend available — falling back to "
        "cv2.imencode. Install simplejpeg (has manylinux wheels) or "
        "libturbojpeg0 + PyTurboJPEG to enable the fast path."
    )

# Lazy cv2 import so the fallback path doesn't pay a startup cost twice
# (services that already imported cv2 don't re-import here).
def _cv2():
    import cv2 as _c
    return _c


# ---- Encode ----------------------------------------------------------------

def encode_jpeg(frame_bgr, quality: int = 85) -> Optional[bytes]:
    """Encode an ndarray (BGR8) to a JPEG bytes object.

    Returns None on failure. Matches the semantics callers already used
    with ``cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])``
    followed by ``buffer.tobytes()`` — same output, ~3× faster on hosts
    with libjpeg-turbo.

    Chroma subsampling is fixed at 4:2:0 to match OpenCV's default and
    keep JPEG payload sizes comparable — otherwise Gallery / Timeline
    might notice a ~15% size delta.
    """
    if frame_bgr is None:
        return None
    if _backend == "simplejpeg":
        try:
            # simplejpeg expects colorspace name; 'BGR' matches our arrays
            # exactly (no channel-swap needed).
            return _sj.encode_jpeg(frame_bgr, quality=int(quality), colorspace='BGR')
        except Exception as e:
            logger.debug(f"jpeg_codec: simplejpeg encode failed, one-frame fallback: {e}")
    elif _backend == "pyturbojpeg":
        try:
            return _tj.encode(
                frame_bgr,
                quality=int(quality),
                pixel_format=TJPF_BGR,
                jpeg_subsample=TJSAMP_420,
            )
        except Exception as e:
            logger.debug(f"jpeg_codec: turbo encode failed, one-frame fallback: {e}")
    cv2 = _cv2()
    ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return None
    return buf.tobytes()


# ---- Decode ----------------------------------------------------------------

def decode_jpeg(jpeg_bytes):
    """Decode JPEG bytes into a BGR ndarray.

    Returns None on decode failure. Matches ``cv2.imdecode`` semantics
    for the callers we care about.
    """
    if not jpeg_bytes:
        return None
    if _backend == "simplejpeg":
        try:
            return _sj.decode_jpeg(jpeg_bytes, colorspace='BGR')
        except Exception as e:
            logger.debug(f"jpeg_codec: simplejpeg decode failed, one-frame fallback: {e}")
    elif _backend == "pyturbojpeg":
        try:
            return _tj.decode(jpeg_bytes, pixel_format=TJPF_BGR)
        except Exception as e:
            logger.debug(f"jpeg_codec: turbo decode failed, one-frame fallback: {e}")
    cv2 = _cv2()
    import numpy as np
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---- imwrite convenience wrapper ------------------------------------------

def imwrite_jpeg(path: str, frame_bgr, quality: int = 85) -> bool:
    """Drop-in replacement for ``cv2.imwrite(path, frame)`` on the disk-writer
    hot path in services/watcher.py:319.

    Splits the write into (encode via libjpeg-turbo) + (blocking write to
    disk). The encode uses the fast path when available; the write goes
    straight to the filesystem.
    """
    data = encode_jpeg(frame_bgr, quality=quality)
    if data is None:
        return False
    try:
        with open(path, 'wb') as fh:
            fh.write(data)
        return True
    except OSError as e:
        logger.warning(f"jpeg_codec.imwrite_jpeg({path}) failed: {e}")
        return False
