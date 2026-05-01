"""Smoke test for the math worker.

Runs MathWorker.analyze() directly on a set of HuggingFace defect-detection
sample images and validates:

    - Output shape matches YOLO's detection contract (dict keys, types).
    - All confidences are finite and within [0, 1].
    - All bboxes are within the image.
    - Expected channels are emitted.

Then prints a small summary per image so we can eyeball whether the numbers
make sense for the defect type (broken cord → col_residual_max high;
warp streaks → vertical stripe signals high; etc.).
"""
from __future__ import annotations

import os
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np

from math_worker import MathWorker, USING_GPU

EXAMPLES_DIR = Path(r"c:\projects\HuggingFace\Industrial-Defect-Detection\examples")

SAMPLES = [
    "tire-cord-1.jpg",   # good reference
    "tire-cord-2.jpg",   # broken cord running vertically
    "tire-cord-3.jpg",   # small dark blob
    "jean-back-2.jpg",   # denim twill + vertical crease
    "knit-back-1.jpg",   # half white half ribbed
    "pvb-film-1.jpg",    # lighting hotspot
    "36846.jpg",         # periodic grid + small defect
]

REQUIRED_KEYS = {"name", "confidence", "class", "xmin", "ymin", "xmax", "ymax"}

# Channels that must exist for every frame (single-emitter channels).
EXPECTED_SINGLE_CHANNELS = {
    "mean_L", "std_L", "range_L", "skew_L",
    "L_p01", "L_p05", "L_p95", "L_p99",
    "saturation_fraction", "sharpness_laplacian_var",
    "sharpness_tenengrad", "exposure_balance",
    "row_residual_max", "row_residual_count",
    "col_residual_max", "col_residual_count",
    "row_spacing_std",  "col_spacing_std",
    "grad_mean", "grad_max",
    "grad_ori_coherence", "grad_ori_dominant_deg",
    "grad_ori_tilt_from_horizontal", "grad_ori_tilt_from_vertical",
    "tophat_max", "tophat_mean", "bothat_max", "bothat_mean",
    # First FFT peak scalars
    "fft_row_peak_1_energy", "fft_row_peak_1_period_px",
    "fft_col_peak_1_energy", "fft_col_peak_1_period_px",
    "fft2d_peak_1_energy",   "fft2d_peak_1_period_px",
    "fft2d_peak_1_angle_deg",
    "fft2d_peak_1_tilt_from_horizontal",
    "fft2d_peak_1_tilt_from_vertical",
}


def validate(dets, w, h):
    errors = []
    seen_names = set()
    for i, d in enumerate(dets):
        if not isinstance(d, dict):
            errors.append(f"det[{i}] is not a dict ({type(d).__name__})")
            continue
        missing = REQUIRED_KEYS - set(d.keys())
        if missing:
            errors.append(f"det[{i}] missing keys: {missing}")
            continue
        if not (0.0 <= d["confidence"] <= 1.0):
            errors.append(f"det[{i}] name={d['name']} confidence out of range: {d['confidence']}")
        if not math.isfinite(d["confidence"]):
            errors.append(f"det[{i}] name={d['name']} confidence NaN/inf")
        if not (0 <= d["xmin"] < d["xmax"] <= w):
            errors.append(f"det[{i}] name={d['name']} x bbox invalid: {d['xmin']}..{d['xmax']} w={w}")
        if not (0 <= d["ymin"] < d["ymax"] <= h):
            errors.append(f"det[{i}] name={d['name']} y bbox invalid: {d['ymin']}..{d['ymax']} h={h}")
        seen_names.add(d["name"])
    missing_channels = EXPECTED_SINGLE_CHANNELS - seen_names
    if missing_channels:
        errors.append(f"missing expected channels: {sorted(missing_channels)}")
    return errors, seen_names


def summarize(dets):
    """Return a readable summary of the interesting scalars."""
    by_name = {}
    for d in dets:
        by_name.setdefault(d["name"], []).append(d["confidence"])
    def first(name):
        vals = by_name.get(name, [])
        return vals[0] if vals else None
    def maxof(name):
        vals = by_name.get(name, [])
        return max(vals) if vals else None
    def count(name):
        return len(by_name.get(name, []))
    return {
        "mean_L":                               first("mean_L"),
        "std_L":                                first("std_L"),
        "saturation_fraction":                  first("saturation_fraction"),
        "sharpness_laplacian_var":              first("sharpness_laplacian_var"),
        "fft_col_peak_1_energy":                first("fft_col_peak_1_energy"),
        "fft_col_peak_1_period_px":             first("fft_col_peak_1_period_px"),
        "fft_row_peak_1_energy":                first("fft_row_peak_1_energy"),
        "fft_row_peak_1_period_px":             first("fft_row_peak_1_period_px"),
        "fft2d_peak_1_angle_deg":               first("fft2d_peak_1_angle_deg"),
        "fft2d_peak_1_tilt_from_horizontal":    first("fft2d_peak_1_tilt_from_horizontal"),
        "col_residual_max":                     first("col_residual_max"),
        "row_residual_max":                     first("row_residual_max"),
        "grad_ori_coherence":                   first("grad_ori_coherence"),
        "grad_ori_dominant_deg":                first("grad_ori_dominant_deg"),
        "band_delta_e_max":                     maxof("band_delta_e"),
        "band_delta_from_median_max":           maxof("band_delta_from_median"),
        "blob_darkness_count":                  count("blob_darkness"),
        "blob_brightness_count":                count("blob_brightness"),
        "row_spacing_anomaly_count":            count("row_spacing_anomaly"),
        "col_spacing_anomaly_count":            count("col_spacing_anomaly"),
        "total_detections":                     len(dets),
    }


def main():
    worker = MathWorker(tiles_x=1, tiles_y=1, bands=8, fft_top_k=3)
    print(f"[smoke] device={'GPU' if USING_GPU else 'CPU'}")
    all_ok = True
    for name in SAMPLES:
        p = EXAMPLES_DIR / name
        if not p.exists():
            print(f"[{name}] MISSING FILE at {p}")
            all_ok = False
            continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{name}] imread failed")
            all_ok = False
            continue
        h, w = bgr.shape[:2]
        t0 = time.time()
        dets = worker.analyze(bgr)
        elapsed = time.time() - t0
        errors, _ = validate(dets, w, h)
        summary = summarize(dets)
        print(f"\n== {name}  ({w}x{h}, {elapsed*1000:.1f} ms, {len(dets)} dets) ==")
        for k, v in summary.items():
            if v is None:
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        if errors:
            all_ok = False
            print(f"  !! VALIDATION ERRORS")
            for e in errors[:10]:
                print(f"     - {e}")
            if len(errors) > 10:
                print(f"     - ... and {len(errors)-10} more")
    print()
    if all_ok:
        print("[smoke] ALL SAMPLES PASSED VALIDATION")
        sys.exit(0)
    else:
        print("[smoke] FAILURES detected")
        sys.exit(1)


if __name__ == "__main__":
    main()
