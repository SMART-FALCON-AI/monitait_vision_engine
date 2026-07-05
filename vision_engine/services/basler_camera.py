"""Basler USB3 Vision camera adapter — the "pro" camera type.

Mirrors services.camera.CameraBuffer's public interface so the rest of the
codebase (watcher, routers/cameras.py, ROI code, drawing code) doesn't have
to know Basler is underneath. Everywhere existing code reads:

    cam.frame        cam.success        cam.stop
    cam.source       cam._saved_props   cam.roi_*
    cam.read()       cam.update_prop()  cam.buffer()

BaslerBuffer exposes the same names with the same semantics.

Sources are strings of the form ``basler://<serial>``, mirroring the existing
``rtsp://…`` convention for IP cameras. That way the camera_metadata table can
persist Basler cameras alongside USB and IP entries with just a `type: "pro"`
tag, and no other code has to be taught about serial numbers.

pypylon is imported lazily so the module can be imported on hosts that don't
have Pylon installed — get an ImportError only when someone actually opens a
Basler camera.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

BASLER_SOURCE_PREFIX = "basler://"

# Property keys — reuse cv2's constants where the meaning maps 1:1, so
# _saved_props remains a homogenous dict across UVC and Basler and existing
# consumers keep working. Basler-specific knobs use string keys.
try:
    import cv2 as _cv2
    PROP_EXPOSURE   = _cv2.CAP_PROP_EXPOSURE
    PROP_GAIN       = _cv2.CAP_PROP_GAIN
    PROP_BRIGHTNESS = _cv2.CAP_PROP_BRIGHTNESS
    PROP_CONTRAST   = _cv2.CAP_PROP_CONTRAST
    PROP_SATURATION = _cv2.CAP_PROP_SATURATION
    PROP_FPS        = _cv2.CAP_PROP_FPS
except Exception:
    # cv2 should always be present in the container; this fallback keeps
    # the file importable in a bare environment for tests.
    PROP_EXPOSURE, PROP_GAIN, PROP_BRIGHTNESS = 15, 14, 10
    PROP_CONTRAST, PROP_SATURATION, PROP_FPS   = 11, 12,  5


def is_basler_source(source) -> bool:
    """Return True when this source should be handled by BaslerBuffer.

    Cheap classifier that only looks at the source URI. Used by CameraBuffer
    to route between UVC/IP/Basler backends without importing pypylon on the
    UVC path.
    """
    return isinstance(source, str) and source.startswith(BASLER_SOURCE_PREFIX)


def _import_pylon():
    """Lazy pypylon import so this module stays importable without Pylon.

    Raises ImportError with a helpful message when Pylon is genuinely missing
    (only fires when someone tries to open a Basler camera, not on module load).
    """
    try:
        from pypylon import pylon  # type: ignore
        from pypylon import genicam  # type: ignore
        return pylon, genicam
    except ImportError as e:
        raise ImportError(
            "pypylon is not installed. Add 'pypylon>=3.0' to "
            "vision_engine/requirements.txt and rebuild the MVE container. "
            f"Original error: {e}"
        )


def detect_basler_cameras() -> List[Dict[str, Any]]:
    """Enumerate Basler USB cameras attached to the host.

    Returns a list of ``{"source": "basler://<serial>", "serial", "model",
    "vendor", "type": "pro"}`` dicts. Empty list when Pylon isn't installed
    or no Basler devices are present — never raises, so the main enumeration
    at services/camera.py:get_all_cameras() can safely mix Basler results in
    alongside the V4L2 sweep.

    Filters to USB devices (device class starting with "BaslerUsb") so this
    function doesn't accidentally surface GigE cameras on the same network —
    those need a different driver and different NIC setup.
    """
    try:
        pylon, _ = _import_pylon()
    except ImportError:
        return []
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
    except Exception as e:
        logger.warning(f"Basler enumeration failed: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for d in devices:
        try:
            device_class = d.GetDeviceClass() or ""
            # Skip GigE etc. — only surface USB.
            if not device_class.startswith("BaslerUsb"):
                continue
            serial = d.GetSerialNumber()
            out.append({
                "source": f"{BASLER_SOURCE_PREFIX}{serial}",
                "serial": serial,
                "model":  d.GetModelName(),
                "vendor": d.GetVendorName(),
                "device_class": device_class,
                "type":   "pro",
            })
        except Exception as e:
            logger.debug(f"Skipping malformed Basler device entry: {e}")
    return out


class BaslerBuffer:
    """A CameraBuffer-shaped adapter around a Basler InstantCamera.

    Contract with the rest of MVE:
      * ``self.source``       — the ``basler://<serial>`` URI
      * ``self.success``      — bool, latest grab succeeded
      * ``self.frame``        — BGR8 numpy array (matches OpenCV format so
                                downstream code sees the same shape/dtype as
                                a UVC frame)
      * ``self.stop``         — bool, thread exit flag
      * ``self._saved_props`` — dict of persisted config values so a
                                reconnect re-applies them (matches CameraBuffer)
      * ``self.roi_*``        — same ROI fields; ROI is a software crop, not
                                Basler-hardware ROI, for parity with UVC path
      * ``self.is_ip_camera`` — always False (kept so downstream branches that
                                already check this keep working)
      * ``self.is_pro_camera``— True (new flag — check this in code that needs
                                to know it's Basler)
      * ``self.camera``       — the pylon.InstantCamera handle; DO NOT call
                                cv2 methods on it. Guard with is_pro_camera.
    """

    def __init__(self, source, exposure=5000, gain=0, brightness=0,
                 contrast=0, saturation=0, fps=30, roi_config=None,
                 auto_exposure=False, pixel_format=None) -> None:
        # 4.0.50 — default pixel_format=None means "keep the camera's
        # native format" (e.g. Mono8 on daA1280-54um monochrome, BayerRG8
        # on colour models). The ImageFormatConverter below always outputs
        # BGR8packed to MVE, so downstream code sees BGR8 regardless of
        # what the camera produces. Trying to *set* BGR8 on a mono sensor
        # would fail — leaving pixel_format=None avoids that landmine.
        pylon, genicam = _import_pylon()
        self.source = source
        self.is_ip_camera = False
        self.is_pro_camera = True
        self.auto_exposure = bool(auto_exposure)
        self.pixel_format = pixel_format

        serial = source.replace(BASLER_SOURCE_PREFIX, "").strip()
        if not serial:
            raise ValueError(f"Basler source has no serial: {source!r}")

        tl_factory = pylon.TlFactory.GetInstance()
        target = None
        for d in tl_factory.EnumerateDevices():
            try:
                if d.GetSerialNumber() == serial:
                    target = d
                    break
            except Exception:
                continue
        if target is None:
            raise RuntimeError(f"Basler camera with serial {serial} not found on this host")

        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(target))
        self.model  = target.GetModelName()
        self.serial = serial
        self.camera.Open()

        # ---------- Configure the camera -----------------------------------
        # We wrap every SetValue in try/except because Basler NodeMaps vary by
        # model. For instance daA1280-54uc lacks GainAuto; acA640-90uc has it.
        # A missing knob shouldn't kill camera bring-up.
        if pixel_format:
            # Only try to change PixelFormat when the operator explicitly
            # picked one. Otherwise leave the camera in its native format
            # (e.g. Mono8 for monochrome sensors) and let the converter do
            # the BGR8 output later.
            self._safe_set("PixelFormat", pixel_format)
        # Turn off auto-exposure by default; the operator opts in via UI.
        if self.auto_exposure:
            self._safe_set("ExposureAuto", "Continuous")
        else:
            self._safe_set("ExposureAuto", "Off")
            self._safe_set("ExposureTime", float(exposure))
        # Gain: Basler uses dB (0.0 .. ~24 typical). Accept whatever the UI
        # sends and clamp inside self._safe_set via try/except.
        self._safe_set("GainAuto", "Off")
        self._safe_set("Gain", float(gain))
        # Framerate: enable AcquisitionFrameRateEnable then set the value.
        self._safe_set("AcquisitionFrameRateEnable", True)
        self._safe_set("AcquisitionFrameRate", float(fps))

        # Start grabbing — LatestImageOnly matches MVE's semantics of "just
        # want the newest frame" (mirrors cv2 non-blocking grab+retrieve).
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # Converter — turn Basler's native format into BGR8 numpy for cv2.
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Frame state
        self.frame: Optional[np.ndarray] = None
        self.success = False
        self.stop = False

        # Saved props for reconnect + get_camera_config_for_save parity.
        # Values map onto the same numeric cv2 prop keys UVC uses, plus a
        # 'basler.*' bag for Basler-only knobs so the JSON round-trips.
        self._saved_props: Dict[Any, Any] = {
            PROP_EXPOSURE:   exposure,
            PROP_GAIN:       gain,
            PROP_BRIGHTNESS: brightness,
            PROP_CONTRAST:   contrast,
            PROP_SATURATION: saturation,
            PROP_FPS:        fps,
            "basler.pixel_format":  pixel_format,
            "basler.auto_exposure": auto_exposure,
        }

        # ROI (software crop — same convention as CameraBuffer)
        if roi_config:
            self.roi_enabled = bool(roi_config.get("roi_enabled", False))
            self.roi_xmin    = int(roi_config.get("roi_xmin", 0))
            self.roi_ymin    = int(roi_config.get("roi_ymin", 0))
            self.roi_xmax    = int(roi_config.get("roi_xmax", 1280))
            self.roi_ymax    = int(roi_config.get("roi_ymax", 720))
        else:
            self.roi_enabled = False
            self.roi_xmin = 0
            self.roi_ymin = 0
            self.roi_xmax = 1280
            self.roi_ymax = 720

        logger.info(
            f"Basler camera opened: {self.model} serial={serial} "
            f"pixel_format={pixel_format} exposure={exposure}us gain={gain}dB fps={fps}"
        )
        threading.Thread(target=self.buffer, daemon=True, name=f"basler-{serial}").start()

    # ---------------------------------------------------------------
    # Grab thread
    # ---------------------------------------------------------------

    def buffer(self):
        """Continuous grab loop — mirrors CameraBuffer.buffer() semantics.

        Populates self.frame with the latest BGR8 numpy array and self.success
        with the grab status. Auto-reconnect kicks in after N consecutive
        failures. Sleeps briefly on every iteration so this thread doesn't
        peg a core.
        """
        pylon, genicam = _import_pylon()
        failure_count = 0
        max_failures = 100

        while not self.stop:
            try:
                if not self.camera.IsGrabbing():
                    logger.warning("Basler: camera not grabbing, restarting")
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                # Short timeout so we can honour self.stop promptly.
                res = self.camera.RetrieveResult(200, pylon.TimeoutHandling_Return)
                # 4.0.50 — RetrieveResult(TimeoutHandling_Return) can return a
                # GrabResult that is not None but IsValid()==False when the
                # timeout elapsed with no frame. Calling .GrabSucceeded() on
                # that object throws NULL-pointer from the underlying C++
                # SDK — hence the sporadic
                # "No grab result data is referenced. Cannot access NULL
                # pointer." errors we saw in production. Guard with IsValid.
                if res is None or not res.IsValid():
                    failure_count += 1
                    if res is not None:
                        try: res.Release()
                        except Exception: pass
                else:
                    ok = res.GrabSucceeded()
                    if ok:
                        # Convert to BGR8 (native cv2 format) so downstream
                        # code sees identical arrays to the UVC path.
                        converted = self._converter.Convert(res)
                        arr = converted.GetArray()   # H x W x 3, uint8, BGR
                        # Sanity: reject frames that are mostly a single grey
                        # value (matches UVC's grey-frame rejection).
                        h = arr.shape[0]
                        bottom = arr[h * 3 // 4:, :]
                        if np.mean(bottom == 128) > 0.9:
                            self.success = False
                            logger.debug("Basler: rejected grey frame")
                        else:
                            self.frame = arr
                            self.success = True
                            failure_count = 0
                    else:
                        self.success = False
                        failure_count += 1
                        logger.debug(
                            f"Basler grab failed code={res.GetErrorCode()} "
                            f"desc={res.GetErrorDescription()}"
                        )
                    res.Release()

                if failure_count >= max_failures:
                    logger.warning(
                        f"Basler: {failure_count} consecutive failures — reconnecting"
                    )
                    try:
                        self._reconnect()
                        failure_count = 0
                    except Exception as e:
                        logger.error(f"Basler reconnect failed: {e}")
                        time.sleep(5)

                time.sleep(0.0001)
            except Exception as e:
                logger.error(f"Basler buffer error: {e}")
                time.sleep(0.1)

    def _reconnect(self):
        """Close and re-open the InstantCamera, re-applying saved props."""
        pylon, _ = _import_pylon()
        try:
            self.camera.StopGrabbing()
        except Exception:
            pass
        try:
            self.camera.Close()
        except Exception:
            pass
        time.sleep(1.0)

        tl_factory = pylon.TlFactory.GetInstance()
        target = None
        for d in tl_factory.EnumerateDevices():
            if d.GetSerialNumber() == self.serial:
                target = d
                break
        if target is None:
            raise RuntimeError(f"Basler {self.serial} not visible on reconnect")
        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(target))
        self.camera.Open()
        self._apply_saved_props()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        logger.info(f"Basler {self.serial}: reconnect ok")

    # ---------------------------------------------------------------
    # Prop helpers
    # ---------------------------------------------------------------

    def _safe_set(self, node_name: str, value):
        """Set a Basler NodeMap value if it exists on this model. No-op if not.

        Basler models expose different NodeMaps (e.g. some cameras have
        BslExposureTimeMode, others just ExposureTime). Guarding each write
        keeps a single BaslerBuffer working across the whole product line
        instead of hard-coding a model.
        """
        try:
            node = self.camera.GetNodeMap().GetNode(node_name)
            if node is None:
                logger.debug(f"Basler {self.serial}: node {node_name!r} not present, skipping")
                return
            # pypylon exposes different setters per node kind. Simplest:
            # try attribute access on the camera itself, then fall back to
            # SetValue on the node.
            if hasattr(self.camera, node_name):
                getattr(self.camera, node_name).SetValue(value)
            else:
                node.SetValue(value)
        except Exception as e:
            logger.debug(f"Basler {self.serial}: could not set {node_name}={value}: {e}")

    def _apply_saved_props(self):
        """Re-apply persisted config after a reconnect. Mirrors CameraBuffer."""
        if not self._saved_props:
            return
        # Pixel format first so subsequent writes go on the right node map.
        # If the saved pixel_format is None/empty, leave it — camera keeps
        # its native format (see __init__ comment).
        _pf = self._saved_props.get("basler.pixel_format")
        if _pf:
            self._safe_set("PixelFormat", _pf)
        auto_exp = self._saved_props.get("basler.auto_exposure", False)
        if auto_exp:
            self._safe_set("ExposureAuto", "Continuous")
        else:
            self._safe_set("ExposureAuto", "Off")
            self._safe_set("ExposureTime", float(self._saved_props.get(PROP_EXPOSURE, 5000)))
        self._safe_set("GainAuto", "Off")
        self._safe_set("Gain", float(self._saved_props.get(PROP_GAIN, 0)))
        self._safe_set("AcquisitionFrameRateEnable", True)
        self._safe_set("AcquisitionFrameRate", float(self._saved_props.get(PROP_FPS, 30)))

    def update_prop(self, prop, value):
        """Update a property and persist it so reconnect keeps it.

        Accepts the same integer cv2 prop keys UVC uses, plus 'basler.*'
        strings. Translates the common cv2 keys onto Basler NodeMap names.
        """
        self._saved_props[prop] = value
        if prop == PROP_EXPOSURE:
            self._safe_set("ExposureAuto", "Off")
            self._safe_set("ExposureTime", float(value))
        elif prop == PROP_GAIN:
            self._safe_set("Gain", float(value))
        elif prop == PROP_FPS:
            self._safe_set("AcquisitionFrameRateEnable", True)
            self._safe_set("AcquisitionFrameRate", float(value))
        elif prop == "basler.pixel_format":
            self._safe_set("PixelFormat", str(value))
        elif prop == "basler.auto_exposure":
            self._safe_set("ExposureAuto", "Continuous" if value else "Off")
        else:
            # Silently store for the JSON round-trip, but don't attempt to
            # apply a UVC-only prop (BRIGHTNESS/CONTRAST/SATURATION don't
            # map cleanly onto Basler cameras — those are LUT-driven).
            logger.debug(f"Basler {self.serial}: no NodeMap for prop {prop}={value}")

    # ---------------------------------------------------------------
    # Read API — matches cv2.VideoCapture-shaped call sites
    # ---------------------------------------------------------------

    def read(self):
        """Return the latest frame — matches CameraBuffer.read() signature.

        4.0.50 — CameraBuffer.read() returns just `frame` (with ROI applied),
        NOT the (success, frame) tuple that cv2.VideoCapture.read() gives.
        The MJPEG stream endpoint and every other caller expect this shape.
        Returning a tuple broke the stream: cv2.imencode('.jpg', (True, arr))
        can't encode a tuple → 'Camera feed unavailable' toast.
        """
        frame = self.frame
        if frame is not None and getattr(self, 'roi_enabled', False):
            frame = frame[self.roi_ymin:self.roi_ymax, self.roi_xmin:self.roi_xmax]
        return frame

    def release(self):
        """Stop grabbing and close the device. Mirrors cv2.VideoCapture.release()."""
        self.stop = True
        try:
            self.camera.StopGrabbing()
        except Exception:
            pass
        try:
            self.camera.Close()
        except Exception:
            pass

    # Explicit context-manager support so `with BaslerBuffer(...) as cam:`
    # cleans up on exceptions — nice-to-have for tests.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
