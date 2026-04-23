"""Math worker — pure mathematical measurements of a single image.

Every channel emits detection dicts in the YOLO-compatible shape:
    {"name": "<flat_name>", "confidence": 0..1, "class": <int>,
     "xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}

Confidence is a FIXED absolute math map from the raw value — no learning, no
per-SKU baseline, no state across frames. Operator rules decide what is
acceptable via the existing MVE procedures engine.

Backend is auto-selected: CuPy on GPU when CUDA is usable at init, else numpy.
The public API does not care which one is active.
"""
from __future__ import annotations

import math
import logging
from typing import Any, Dict, List

import numpy as np
import scipy.ndimage as _np_ndimage
import scipy.signal as _np_signal

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Device auto-selection (mirrors torch's is_cuda_available pattern)
# --------------------------------------------------------------------------
USING_GPU = False
_cp = None

try:
    import cupy as _cp  # type: ignore
    import cupyx.scipy.ndimage as _cp_ndimage  # type: ignore
    import cupyx.scipy.signal as _cp_signal  # type: ignore

    # Probe: will raise if no GPU or CUDA runtime is broken.
    _cp.cuda.runtime.getDeviceCount()
    _ = _cp.asarray([0.0]).sum().item()  # force a kernel launch

    xp = _cp
    ximg = _cp_ndimage
    xsig = _cp_signal
    USING_GPU = True
    logger.info("math worker: GPU backend (CuPy)")
except Exception as e:  # pragma: no cover - depends on runtime hardware
    xp = np
    ximg = _np_ndimage
    xsig = _np_signal
    logger.info(
        f"math worker: CPU backend (numpy) — reason: {type(e).__name__}: {e}"
    )


def _to_host(a) -> np.ndarray:
    """Bring an array back to host (numpy) regardless of backend."""
    if USING_GPU and _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def _as_float(x) -> float:
    try:
        return float(_to_host(x).reshape(()))
    except Exception:
        return float(x)


# --------------------------------------------------------------------------
# Absolute confidence maps — fixed mathematical bounds, no calibration.
# --------------------------------------------------------------------------
def _clip01(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def _map_L(L100: float) -> float:
    """L* in CIE 0..100 → confidence 0..1."""
    return _clip01(L100 / 100.0)


def _map_cap(value: float, cap: float) -> float:
    """min(value / cap, 1) — generic absolute cap."""
    return _clip01(value / cap) if cap > 0 else 0.0


def _map_tilt(angle_deg: float, target_deg: float) -> float:
    """|angle - target| / 45 clipped to 0..1 — 45° deviation saturates.

    Both angles treated modulo 180° (lines are direction-agnostic).
    """
    a = angle_deg % 180.0
    t = target_deg % 180.0
    d = abs(a - t)
    d = min(d, 180.0 - d)
    return _clip01(d / 45.0)


# --------------------------------------------------------------------------
# Detection dict builder
# --------------------------------------------------------------------------
def _det(name: str, conf: float, x1: int, y1: int, x2: int, y2: int,
         class_id: int = 900) -> Dict[str, Any]:
    return {
        "name":       name,
        "confidence": round(_clip01(float(conf)), 4),
        "class":      class_id,
        "xmin":       int(x1),
        "ymin":       int(y1),
        "xmax":       int(x2),
        "ymax":       int(y2),
    }


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
class MathWorker:
    # Channel "class" ids — purely informational, grouped by family.
    C_GLOBAL   = 900
    C_QUALITY  = 910
    C_FFT_ROW  = 920
    C_FFT_COL  = 930
    C_FFT_2D   = 940
    C_BAND     = 950
    C_RESID    = 960
    C_SPACING  = 970
    C_BLOB     = 980
    C_TILE     = 990
    C_GRAD     = 1000
    C_MORPH    = 1010

    def __init__(
        self,
        tiles_x: int = 1,
        tiles_y: int = 1,
        bands: int = 8,
        fft_top_k: int = 3,
        flat_field: bool = False,
    ):
        self.tiles_x = max(1, int(tiles_x))
        self.tiles_y = max(1, int(tiles_y))
        self.bands = max(1, int(bands))
        self.fft_top_k = max(1, int(fft_top_k))
        self.flat_field = bool(flat_field)

    # -- public ---------------------------------------------------------

    def analyze(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Return a flat list of detection dicts for one frame."""
        if bgr is None or bgr.size == 0:
            return []

        # Color conversion on CPU (cv2 is fast here, ~5ms on a 2Mpx frame).
        import cv2
        lab_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        h, w = lab_u8.shape[:2]

        # Move L*/a*/b* planes onto the active backend.
        # OpenCV's L is 0..255 (scaled), a/b are 0..255 with 128 = neutral.
        L100 = xp.asarray(lab_u8[:, :, 0], dtype=xp.float32) * (100.0 / 255.0)
        a_ch = xp.asarray(lab_u8[:, :, 1], dtype=xp.float32) - 128.0
        b_ch = xp.asarray(lab_u8[:, :, 2], dtype=xp.float32) - 128.0

        if self.flat_field:
            L100 = self._flat_field(L100)

        dets: List[Dict[str, Any]] = []
        dets += self._global_stats(L100, w, h)
        dets += self._quality(L100, w, h)
        dets += self._fft_1d(L100, w, h)
        dets += self._fft_2d(L100, w, h)
        dets += self._band_stats(L100, a_ch, b_ch, w, h)
        dets += self._residuals(L100, w, h)
        dets += self._spacing_anomalies(L100, w, h)
        dets += self._gradient(L100, w, h)
        dets += self._morphology(L100, w, h)
        dets += self._blobs(L100, w, h)

        if self.tiles_x > 1 or self.tiles_y > 1:
            dets += self._tile_channels(L100, w, h)

        return dets

    def channel_names(self) -> List[str]:
        names = [
            # global
            "mean_L", "std_L", "range_L", "skew_L", "L_p01", "L_p05", "L_p95", "L_p99",
            # quality / meta
            "saturation_fraction", "sharpness_laplacian_var", "sharpness_tenengrad",
            "exposure_balance",
        ]
        # FFT peaks — flattened scalars per rank
        for k in range(1, self.fft_top_k + 1):
            names += [
                f"fft_row_peak_{k}_energy", f"fft_row_peak_{k}_period_px",
                f"fft_col_peak_{k}_energy", f"fft_col_peak_{k}_period_px",
                f"fft2d_peak_{k}_energy", f"fft2d_peak_{k}_period_px",
                f"fft2d_peak_{k}_angle_deg",
                f"fft2d_peak_{k}_tilt_from_horizontal",
                f"fft2d_peak_{k}_tilt_from_vertical",
            ]
        names += [
            # band (one detection per band)
            "band_mean_L", "band_std_L", "band_delta_from_median", "band_delta_e",
            # residuals
            "row_residual_max", "row_residual_count",
            "col_residual_max", "col_residual_count",
            # spacing anomalies (localized)
            "row_spacing_anomaly", "col_spacing_anomaly",
            "row_spacing_std", "col_spacing_std",
            # gradient / orientation
            "grad_mean", "grad_max", "grad_ori_coherence",
            "grad_ori_dominant_deg",
            "grad_ori_tilt_from_horizontal", "grad_ori_tilt_from_vertical",
            # morphology
            "tophat_max", "tophat_mean", "bothat_max", "bothat_mean",
            # blobs
            "blob_darkness", "blob_brightness",
        ]
        if self.tiles_x > 1 or self.tiles_y > 1:
            names += [
                "tile_row_period_shift", "tile_col_period_shift",
                "tile_row_energy_excess", "tile_col_energy_excess",
                "tile_mean_L_shift", "tile_std_L_shift",
            ]
        return names

    # -- families -------------------------------------------------------

    def _flat_field(self, L: xp.ndarray) -> xp.ndarray:
        """Remove large-scale illumination variation by dividing by a heavy blur."""
        try:
            sigma = max(L.shape) / 16.0
            bg = ximg.gaussian_filter(L, sigma=sigma)
            # Avoid divide-by-zero: floor at 1.
            bg = xp.where(bg < 1.0, 1.0, bg)
            mean_bg = float(_to_host(bg.mean()))
            return xp.clip(L * (mean_bg / bg), 0.0, 100.0)
        except Exception:
            return L

    def _global_stats(self, L, w, h) -> List[Dict[str, Any]]:
        L_host = _to_host(L).ravel()
        mean = float(L_host.mean())
        std = float(L_host.std())
        # Percentiles on host to use numpy's stable implementation.
        p01, p05, p95, p99 = np.percentile(L_host, [1, 5, 95, 99])
        rng = float(p99 - p01)
        # Skew (biased; fine for monitoring)
        m = mean
        s = std if std > 1e-6 else 1.0
        skew = float(((L_host - m) ** 3).mean() / (s ** 3))
        dets: List[Dict[str, Any]] = []
        dets.append(_det("mean_L",   _map_L(mean),          0, 0, w, h, self.C_GLOBAL))
        dets.append(_det("std_L",    _map_cap(std, 50.0),   0, 0, w, h, self.C_GLOBAL))
        dets.append(_det("range_L",  _map_cap(rng, 100.0),  0, 0, w, h, self.C_GLOBAL))
        dets.append(_det("skew_L",   _clip01(0.5 + math.tanh(skew) * 0.5), 0, 0, w, h, self.C_GLOBAL))
        dets.append(_det("L_p01",    _map_L(float(p01)),    0, 0, w, h, self.C_GLOBAL))
        dets.append(_det("L_p05",    _map_L(float(p05)),    0, 0, w, h, self.C_GLOBAL))
        dets.append(_det("L_p95",    _map_L(float(p95)),    0, 0, w, h, self.C_GLOBAL))
        dets.append(_det("L_p99",    _map_L(float(p99)),    0, 0, w, h, self.C_GLOBAL))
        return dets

    def _quality(self, L, w, h) -> List[Dict[str, Any]]:
        # Saturation fraction: near 0 or near 100.
        L_host = _to_host(L)
        total = L_host.size
        sat = float(((L_host <= 2.0) | (L_host >= 98.0)).sum()) / max(total, 1)

        # Laplacian variance (focus metric).
        lap = ximg.laplace(L)
        lap_var = float(_to_host(lap.var()))

        # Tenengrad (squared Sobel magnitude mean).
        gx = ximg.sobel(L, axis=1)
        gy = ximg.sobel(L, axis=0)
        gmag2 = gx * gx + gy * gy
        tenengrad = float(_to_host(gmag2.mean()))

        # Exposure balance — how far median is from 50.
        median_L = float(np.median(L_host))
        exp_bal = abs(median_L - 50.0) / 50.0

        return [
            _det("saturation_fraction",      _clip01(sat),                 0, 0, w, h, self.C_QUALITY),
            _det("sharpness_laplacian_var",  _map_cap(lap_var, 500.0),     0, 0, w, h, self.C_QUALITY),
            _det("sharpness_tenengrad",      _map_cap(tenengrad, 5000.0),  0, 0, w, h, self.C_QUALITY),
            _det("exposure_balance",         _clip01(exp_bal),             0, 0, w, h, self.C_QUALITY),
        ]

    # ---- FFT helpers --------------------------------------------------

    def _top_peaks_1d(self, spec, k):
        """Given a 1D magnitude spectrum, return top-k peaks as (idx, amp, snr)."""
        s = _to_host(spec).astype(np.float64)
        # Skip DC and immediate neighbours.
        s[:2] = 0
        peaks = []
        med = float(np.median(s[s > 0])) if np.any(s > 0) else 1.0
        med = med if med > 1e-9 else 1e-9
        # Simple: argmax, mask a neighborhood, repeat.
        mask = np.ones_like(s, dtype=bool)
        total = float(s.sum())
        for _ in range(k):
            m = s * mask
            if m.max() <= 0:
                break
            idx = int(np.argmax(m))
            amp = float(m[idx])
            frac = amp / total if total > 0 else 0.0
            snr = amp / med
            peaks.append((idx, amp, snr, frac))
            lo = max(0, idx - 3)
            hi = min(len(s), idx + 4)
            mask[lo:hi] = False
        while len(peaks) < k:
            peaks.append((0, 0.0, 0.0, 0.0))
        return peaks

    def _fft_1d(self, L, w, h) -> List[Dict[str, Any]]:
        row_mean = L.mean(axis=1)  # length h
        col_mean = L.mean(axis=0)  # length w
        row_spec = xp.abs(xp.fft.rfft(row_mean - row_mean.mean()))
        col_spec = xp.abs(xp.fft.rfft(col_mean - col_mean.mean()))

        dets: List[Dict[str, Any]] = []
        for k, (idx, amp, snr, frac) in enumerate(self._top_peaks_1d(row_spec, self.fft_top_k), start=1):
            period_px = (h / idx) if idx > 0 else 0.0
            dets.append(_det(f"fft_row_peak_{k}_energy",
                             _clip01(frac), 0, 0, w, h, self.C_FFT_ROW))
            dets.append(_det(f"fft_row_peak_{k}_period_px",
                             _map_cap(period_px, 1000.0), 0, 0, w, h, self.C_FFT_ROW))
        for k, (idx, amp, snr, frac) in enumerate(self._top_peaks_1d(col_spec, self.fft_top_k), start=1):
            period_px = (w / idx) if idx > 0 else 0.0
            dets.append(_det(f"fft_col_peak_{k}_energy",
                             _clip01(frac), 0, 0, w, h, self.C_FFT_COL))
            dets.append(_det(f"fft_col_peak_{k}_period_px",
                             _map_cap(period_px, 1000.0), 0, 0, w, h, self.C_FFT_COL))
        return dets

    def _fft_2d(self, L, w, h) -> List[Dict[str, Any]]:
        # Subtract mean + apply Hann window to reduce leakage at edges.
        L_zm = L - L.mean()
        # Windowing: separable Hann.
        wy = xp.hanning(h).astype(xp.float32)[:, None]
        wx = xp.hanning(w).astype(xp.float32)[None, :]
        Lw = L_zm * wy * wx

        F = xp.fft.fftshift(xp.fft.fft2(Lw))
        mag = xp.abs(F)
        mag = _to_host(mag).astype(np.float64)
        cy, cx = h // 2, w // 2
        # Zero out DC ball to avoid it dominating.
        dc_r = max(3, int(min(w, h) * 0.01))
        yy, xx = np.ogrid[:h, :w]
        dc_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < dc_r * dc_r
        mag[dc_mask] = 0.0

        total = mag.sum()
        # Exploit symmetry: look at upper half.
        upper = mag.copy()
        upper[cy + 1:, :] = 0.0
        dets: List[Dict[str, Any]] = []
        work = upper.copy()
        nbr = max(3, int(min(w, h) * 0.01))
        for k in range(1, self.fft_top_k + 1):
            if work.max() <= 0:
                dets.extend(self._empty_fft2d_peak(k, w, h))
                continue
            idx = int(np.argmax(work))
            py, px = divmod(idx, w)
            amp = float(work[py, px])
            frac = amp / total if total > 0 else 0.0
            # Frequency vector from center.
            dy = py - cy
            dx = px - cx
            # Spatial period in px:  min(h,w) / sqrt(dx^2 + dy^2) * min(h,w) is wrong;
            # period along dominant axis = 1/freq; freq in cycles-per-image = sqrt(dy^2+dx^2)/min(h,w).
            # Use cycles-per-image via independent axes:
            cycles_y = abs(dy)
            cycles_x = abs(dx)
            cycles = math.hypot(cycles_y, cycles_x)
            if cycles > 0:
                # Effective period along dominant direction (in pixels).
                period_px = min(h, w) / cycles
            else:
                period_px = 0.0
            # Angle of frequency vector (0°..180° mod π).
            angle_deg = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0

            dets.append(_det(f"fft2d_peak_{k}_energy",
                             _clip01(frac), 0, 0, w, h, self.C_FFT_2D))
            dets.append(_det(f"fft2d_peak_{k}_period_px",
                             _map_cap(period_px, 1000.0), 0, 0, w, h, self.C_FFT_2D))
            dets.append(_det(f"fft2d_peak_{k}_angle_deg",
                             _clip01(angle_deg / 180.0), 0, 0, w, h, self.C_FFT_2D))
            dets.append(_det(f"fft2d_peak_{k}_tilt_from_horizontal",
                             _map_tilt(angle_deg, 90.0), 0, 0, w, h, self.C_FFT_2D))
            dets.append(_det(f"fft2d_peak_{k}_tilt_from_vertical",
                             _map_tilt(angle_deg, 0.0), 0, 0, w, h, self.C_FFT_2D))
            # Mask neighborhood and its mirror (symmetry) before next peak.
            y0, y1 = max(0, py - nbr), min(h, py + nbr + 1)
            x0, x1 = max(0, px - nbr), min(w, px + nbr + 1)
            work[y0:y1, x0:x1] = 0.0
        return dets

    def _empty_fft2d_peak(self, k, w, h):
        return [
            _det(f"fft2d_peak_{k}_energy",               0.0, 0, 0, w, h, self.C_FFT_2D),
            _det(f"fft2d_peak_{k}_period_px",            0.0, 0, 0, w, h, self.C_FFT_2D),
            _det(f"fft2d_peak_{k}_angle_deg",            0.0, 0, 0, w, h, self.C_FFT_2D),
            _det(f"fft2d_peak_{k}_tilt_from_horizontal", 0.0, 0, 0, w, h, self.C_FFT_2D),
            _det(f"fft2d_peak_{k}_tilt_from_vertical",   0.0, 0, 0, w, h, self.C_FFT_2D),
        ]

    def _band_stats(self, L, a, b, w, h) -> List[Dict[str, Any]]:
        n = self.bands
        edges = np.linspace(0, w, n + 1, dtype=int)
        L_host = _to_host(L)
        a_host = _to_host(a)
        b_host = _to_host(b)

        per_band = []
        for i in range(n):
            x0, x1 = edges[i], edges[i + 1]
            if x1 <= x0:
                per_band.append((x0, x1, 0.0, 0.0, 0.0, 0.0))
                continue
            sl = L_host[:, x0:x1]
            per_band.append((
                x0, x1,
                float(sl.mean()),
                float(sl.std()),
                float(a_host[:, x0:x1].mean()),
                float(b_host[:, x0:x1].mean()),
            ))

        median_L = float(np.median([p[2] for p in per_band]))
        # ΔE vs frame-median LAB
        med_a = float(np.median([p[4] for p in per_band]))
        med_b = float(np.median([p[5] for p in per_band]))

        dets: List[Dict[str, Any]] = []
        for i, (x0, x1, mL, sL, mA, mB) in enumerate(per_band):
            dets.append(_det("band_mean_L",
                             _map_L(mL), x0, 0, x1, h, self.C_BAND))
            dets.append(_det("band_std_L",
                             _map_cap(sL, 50.0), x0, 0, x1, h, self.C_BAND))
            dets.append(_det("band_delta_from_median",
                             _map_cap(abs(mL - median_L), 30.0), x0, 0, x1, h, self.C_BAND))
            de = math.sqrt((mL - median_L) ** 2 + (mA - med_a) ** 2 + (mB - med_b) ** 2)
            dets.append(_det("band_delta_e",
                             _map_cap(de, 20.0), x0, 0, x1, h, self.C_BAND))
        return dets

    def _residuals(self, L, w, h) -> List[Dict[str, Any]]:
        row_mean = _to_host(L.mean(axis=1))   # length h
        col_mean = _to_host(L.mean(axis=0))   # length w

        row_median = float(np.median(row_mean))
        col_median = float(np.median(col_mean))

        row_resid = np.abs(row_mean - row_median)
        col_resid = np.abs(col_mean - col_median)

        row_max = float(row_resid.max()) if row_resid.size else 0.0
        col_max = float(col_resid.max()) if col_resid.size else 0.0
        row_idx = int(np.argmax(row_resid)) if row_resid.size else 0
        col_idx = int(np.argmax(col_resid)) if col_resid.size else 0

        # Count rows/cols beyond a generous threshold of 10 L* units.
        row_count = int((row_resid > 10.0).sum())
        col_count = int((col_resid > 10.0).sum())

        return [
            _det("row_residual_max",   _map_cap(row_max, 50.0),
                 0, max(0, row_idx - 2), w, min(h, row_idx + 3), self.C_RESID),
            _det("row_residual_count", _map_cap(row_count, 100.0),
                 0, 0, w, h, self.C_RESID),
            _det("col_residual_max",   _map_cap(col_max, 50.0),
                 max(0, col_idx - 2), 0, min(w, col_idx + 3), h, self.C_RESID),
            _det("col_residual_count", _map_cap(col_count, 100.0),
                 0, 0, w, h, self.C_RESID),
        ]

    def _spacing_anomalies(self, L, w, h) -> List[Dict[str, Any]]:
        from scipy.signal import find_peaks  # host-side; signal is 1D & small

        dets: List[Dict[str, Any]] = []

        def _on_signal(signal_1d, axis_len, name_prefix, std_name, bbox_fn):
            s = signal_1d - signal_1d.mean()
            if s.size < 10:
                return 0.0
            # Positive-going peaks (weft/warp are brighter than gaps, or vice-versa).
            # Take both and pick whichever has more peaks.
            peaks_pos, _ = find_peaks(s, distance=3)
            peaks_neg, _ = find_peaks(-s, distance=3)
            peaks = peaks_pos if peaks_pos.size >= peaks_neg.size else peaks_neg
            if peaks.size < 4:
                return 0.0
            gaps = np.diff(peaks).astype(np.float64)
            med = float(np.median(gaps))
            if med <= 0:
                return 0.0
            z = np.abs(gaps / med - 1.0)
            # Localized anomalies: one detection per gap with z > noise floor.
            for i, g in enumerate(gaps):
                if z[i] < 0.15:
                    continue
                start = int(peaks[i])
                end = int(peaks[i + 1])
                x1, y1, x2, y2 = bbox_fn(start, end)
                dets.append(_det(name_prefix,
                                 _clip01(float(z[i])),
                                 x1, y1, x2, y2, self.C_SPACING))
            # Global stddev of gaps relative to median.
            return float(gaps.std() / med)

        row_mean = _to_host(L.mean(axis=1))
        col_mean = _to_host(L.mean(axis=0))

        row_std_n = _on_signal(
            row_mean, h, "row_spacing_anomaly", "row_spacing_std",
            bbox_fn=lambda s, e: (0, s, w, e),
        )
        col_std_n = _on_signal(
            col_mean, w, "col_spacing_anomaly", "col_spacing_std",
            bbox_fn=lambda s, e: (s, 0, e, h),
        )

        dets.append(_det("row_spacing_std", _map_cap(row_std_n, 1.0), 0, 0, w, h, self.C_SPACING))
        dets.append(_det("col_spacing_std", _map_cap(col_std_n, 1.0), 0, 0, w, h, self.C_SPACING))
        return dets

    def _gradient(self, L, w, h) -> List[Dict[str, Any]]:
        gx = ximg.sobel(L, axis=1)
        gy = ximg.sobel(L, axis=0)
        mag = xp.sqrt(gx * gx + gy * gy)
        mag_mean = float(_to_host(mag.mean()))
        mag_max = float(_to_host(mag.max()))

        # Structure-tensor-lite: coherence and dominant angle via gradient covariance.
        gxx = float(_to_host((gx * gx).mean()))
        gyy = float(_to_host((gy * gy).mean()))
        gxy = float(_to_host((gx * gy).mean()))
        tr = gxx + gyy
        det_val = gxx * gyy - gxy * gxy
        # Eigenvalues of 2x2 symmetric matrix
        disc = max(0.0, tr * tr / 4.0 - det_val)
        sqrt_disc = math.sqrt(disc)
        lam1 = tr / 2.0 + sqrt_disc
        lam2 = max(0.0, tr / 2.0 - sqrt_disc)
        if lam1 + lam2 > 1e-9:
            coherence = (lam1 - lam2) / (lam1 + lam2)
        else:
            coherence = 0.0
        # Dominant gradient direction → normal to structure.
        # Structure orientation (line direction) is perpendicular.
        theta = 0.5 * math.degrees(math.atan2(2 * gxy, gxx - gyy + 1e-9))
        structure_deg = (theta + 90.0) % 180.0

        return [
            _det("grad_mean", _map_cap(mag_mean, 50.0), 0, 0, w, h, self.C_GRAD),
            _det("grad_max",  _map_cap(mag_max, 255.0), 0, 0, w, h, self.C_GRAD),
            _det("grad_ori_coherence",           _clip01(coherence), 0, 0, w, h, self.C_GRAD),
            _det("grad_ori_dominant_deg",        _clip01(structure_deg / 180.0), 0, 0, w, h, self.C_GRAD),
            _det("grad_ori_tilt_from_horizontal", _map_tilt(structure_deg, 0.0), 0, 0, w, h, self.C_GRAD),
            _det("grad_ori_tilt_from_vertical",   _map_tilt(structure_deg, 90.0), 0, 0, w, h, self.C_GRAD),
        ]

    def _morphology(self, L, w, h) -> List[Dict[str, Any]]:
        # Grayscale top-hat / bottom-hat with a moderate structuring element.
        size = max(5, min(w, h) // 64)
        tophat = L - ximg.grey_opening(L, size=size)
        bothat = ximg.grey_closing(L, size=size) - L
        return [
            _det("tophat_max",  _map_cap(float(_to_host(tophat.max())), 50.0),  0, 0, w, h, self.C_MORPH),
            _det("tophat_mean", _map_cap(float(_to_host(tophat.mean())), 20.0), 0, 0, w, h, self.C_MORPH),
            _det("bothat_max",  _map_cap(float(_to_host(bothat.max())), 50.0),  0, 0, w, h, self.C_MORPH),
            _det("bothat_mean", _map_cap(float(_to_host(bothat.mean())), 20.0), 0, 0, w, h, self.C_MORPH),
        ]

    def _blobs(self, L, w, h) -> List[Dict[str, Any]]:
        L_host = _to_host(L)
        median = float(np.median(L_host))
        # Simple threshold-based blob extraction, two polarities.
        dark_mask = L_host < (median - 15.0)
        bright_mask = L_host > (median + 15.0)
        dets: List[Dict[str, Any]] = []
        for mask, name in ((dark_mask, "blob_darkness"), (bright_mask, "blob_brightness")):
            lbl, n = _np_ndimage.label(mask)
            if n == 0:
                continue
            objs = _np_ndimage.find_objects(lbl)
            # Filter and emit.
            min_area = max(16, (w * h) // 10000)
            for i, sl in enumerate(objs):
                if sl is None:
                    continue
                y0, y1 = sl[0].start, sl[0].stop
                x0, x1 = sl[1].start, sl[1].stop
                area = (x1 - x0) * (y1 - y0)
                if area < min_area:
                    continue
                region = L_host[sl]
                mean_L_region = float(region.mean())
                if name == "blob_darkness":
                    conf = _clip01(1.0 - mean_L_region / 100.0)
                else:
                    conf = _clip01(mean_L_region / 100.0)
                dets.append(_det(name, conf, x0, y0, x1, y1, self.C_BLOB))
        return dets

    def _tile_channels(self, L, w, h) -> List[Dict[str, Any]]:
        """Localized frequency / intensity anomaly: divide into tiles, compare each to frame median."""
        tx, ty = self.tiles_x, self.tiles_y
        xs = np.linspace(0, w, tx + 1, dtype=int)
        ys = np.linspace(0, h, ty + 1, dtype=int)

        # Collect per-tile metrics first.
        per_tile: List[Dict[str, Any]] = []
        L_host = _to_host(L)
        for iy in range(ty):
            for ix in range(tx):
                x0, x1 = xs[ix], xs[ix + 1]
                y0, y1 = ys[iy], ys[iy + 1]
                if x1 <= x0 or y1 <= y0:
                    continue
                t = L_host[y0:y1, x0:x1]
                row_m = t.mean(axis=1)
                col_m = t.mean(axis=0)
                if row_m.size < 4 or col_m.size < 4:
                    continue
                row_spec = np.abs(np.fft.rfft(row_m - row_m.mean()))
                col_spec = np.abs(np.fft.rfft(col_m - col_m.mean()))
                row_peak = int(np.argmax(row_spec[2:]) + 2) if row_spec.size > 2 else 0
                col_peak = int(np.argmax(col_spec[2:]) + 2) if col_spec.size > 2 else 0
                row_energy = float(row_spec.sum())
                col_energy = float(col_spec.sum())
                per_tile.append({
                    "bbox": (x0, y0, x1, y1),
                    "row_period": (y1 - y0) / max(row_peak, 1),
                    "col_period": (x1 - x0) / max(col_peak, 1),
                    "row_energy": row_energy,
                    "col_energy": col_energy,
                    "mean_L":     float(t.mean()),
                    "std_L":      float(t.std()),
                })

        if not per_tile:
            return []

        # Frame-level medians of these per-tile metrics.
        med_row_period = float(np.median([p["row_period"] for p in per_tile]))
        med_col_period = float(np.median([p["col_period"] for p in per_tile]))
        med_row_energy = float(np.median([p["row_energy"] for p in per_tile]))
        med_col_energy = float(np.median([p["col_energy"] for p in per_tile]))
        med_mean_L     = float(np.median([p["mean_L"]     for p in per_tile]))
        med_std_L      = float(np.median([p["std_L"]      for p in per_tile]))

        dets: List[Dict[str, Any]] = []
        for p in per_tile:
            x0, y0, x1, y1 = p["bbox"]
            def _shift(v, m):
                return abs(v - m) / m if m > 1e-9 else 0.0
            dets.append(_det("tile_row_period_shift",
                             _map_cap(_shift(p["row_period"], med_row_period), 1.0),
                             x0, y0, x1, y1, self.C_TILE))
            dets.append(_det("tile_col_period_shift",
                             _map_cap(_shift(p["col_period"], med_col_period), 1.0),
                             x0, y0, x1, y1, self.C_TILE))
            dets.append(_det("tile_row_energy_excess",
                             _map_cap(max(0.0, p["row_energy"] - med_row_energy) / max(med_row_energy, 1e-9), 2.0),
                             x0, y0, x1, y1, self.C_TILE))
            dets.append(_det("tile_col_energy_excess",
                             _map_cap(max(0.0, p["col_energy"] - med_col_energy) / max(med_col_energy, 1e-9), 2.0),
                             x0, y0, x1, y1, self.C_TILE))
            dets.append(_det("tile_mean_L_shift",
                             _map_cap(abs(p["mean_L"] - med_mean_L), 30.0),
                             x0, y0, x1, y1, self.C_TILE))
            dets.append(_det("tile_std_L_shift",
                             _map_cap(abs(p["std_L"] - med_std_L), 30.0),
                             x0, y0, x1, y1, self.C_TILE))
        return dets
