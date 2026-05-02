# Math Module — Channel Reference

The `math_inference` worker emits **63 channels** per frame. Every channel is a
pure mathematical measurement of the captured image with a fixed absolute
mapping from the raw value to a 0..1 `confidence`. Operator rules in the
**Procedures** tab decide what counts as a defect — the worker only measures.

This doc is the catalog: what each channel measures, what defect it catches,
which rule pattern fits, and how to read the numbers.

---

## Index by defect type — pick the right family fast

| If you want to detect… | Use family |
|---|---|
| Large-area shade / brightness drift across the fabric | **A. Global stats** + **F. Band stats** |
| Color shift left↔right (cone-shading) | **F. Band stats** (`band_delta_e`, `band_delta_from_median`) |
| Periodic stripes / bars / weave irregularity | **C. 1D FFT row peaks** + **D. 1D FFT col peaks** + **E. 2D FFT peaks** |
| Stripe orientation (warp vs weft vs diagonal) | **E. 2D FFT** `tilt_from_horizontal` / `tilt_from_vertical`, **I. Gradient orientation** |
| One missing weft / one too-close pair (spacing irregularity) | **H. Spacing anomalies** |
| Single broken cord / vertical line defect running full height | **G. Residuals** (`col_residual_max/count`) |
| Single horizontal bar / weft tension fault running full width | **G. Residuals** (`row_residual_max/count`) |
| Sharp edges (holes, cuts, tears) | **I. Gradient** + **J. Morphology** (top-/bot-hat) |
| Small dark blob (oil spot, foreign matter, stain) | **K. Blobs** (`blob_darkness`) |
| Small bright blob (missed dye, hole-with-backlight) | **K. Blobs** (`blob_brightness`) |
| Texture loss / fog / wash-out | **A. Global stats** (`std_L`, `range_L`) + **I. Gradient coherence** |
| "Was the camera even good?" — frame-quality gate | **B. Quality meta** |

---

## Family A — Global statistics (8 channels)

One detection per channel, bounding box = whole frame. Cheap (<5 ms total).

| Channel | What it is mathematically | Confidence map (0..1) | Physical meaning / catches |
|---|---|---|---|
| `mean_L` | Mean of L\* over the frame | `L / 100` | Overall brightness. Drift from your normal mean = whole-frame shade. |
| `std_L` | Stddev of L\* | `min(σ / 50, 1)` | Texture energy. Drops on washed-out frames; rises on heavy texture or defects. |
| `range_L` | p99 − p01 of L\* | `min((p99-p01) / 100, 1)` | Robust dynamic range. Insensitive to a few outlier pixels. |
| `skew_L` | Skewness of the L\* distribution | `0.5 + 0.5·tanh(skew)` | Asymmetry of the histogram. >0.5 = bright tail; <0.5 = dark tail. Catches mostly-bright fabric with a few dark blemishes (or vice-versa). |
| `L_p01`, `L_p05` | 1st / 5th percentile | `L_pX / 100` | "How dark are the darkest pixels." Drops on dark defects. |
| `L_p95`, `L_p99` | 95th / 99th percentile | `L_pX / 100` | "How bright are the brightest pixels." Rises on bright defects, hot pixels, specular reflections. |

**Use:** broad whole-frame health checks. *Rule example:* `mean_L
min_confidence 0.80` → eject if frame mean L\* > 80 (too bright).

---

## Family B — Quality / image meta (4 channels)

Tells you whether the frame's measurements are trustworthy. Use these as
**gates** in front of other rules — if the frame is junk, don't act on it.

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `saturation_fraction` | Fraction of pixels with L\* ≤ 2 or L\* ≥ 98 | direct (0..1) | Clipped highlights / black crush. Above ~0.10 = unreliable measurements. |
| `sharpness_laplacian_var` | Variance of Laplacian | `min(var / 500, 1)` | Focus / motion blur. Drops on out-of-focus or fast-moving fabric. |
| `sharpness_tenengrad` | Mean squared Sobel magnitude | `min(value / 5000, 1)` | Same idea, less noise-sensitive. Use for blur detection. |
| `exposure_balance` | `|median(L) - 50| / 50` | `min(distance, 1)` | 0 = perfectly mid-grey exposure; 1 = either fully blown or fully dark. |

**Use:** `sharpness_laplacian_var min_confidence 0.30` is a cheap "is the image
in focus" check; reject frames below threshold *or* skip eject decisions on
them.

---

## Family C — 1D FFT row peaks (6 channels: top-3 ranks × 2 scalars)

Take row means → 1D FFT → find top-3 spectral peaks. Peak rank by amplitude.

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `fft_row_peak_K_energy` (K = 1..3) | Fractional energy at this peak | `peak / total` | Strength of the K-th most prominent **horizontal** periodic pattern (= weft bars / weft tension). |
| `fft_row_peak_K_period_px` | Period in pixels at the peak | `min(period / 1000, 1)` | The actual spacing of the bars. Useful as diagnostic ("bars every 6 mm" = which loom part?). |

**Use:** `fft_row_peak_1_energy min_confidence 0.40` → strong horizontal banding
detected. Stack with rank 2/3 to detect multi-frequency defects (reed +
yarn-count combo).

---

## Family D — 1D FFT col peaks (6 channels)

Same idea on column means → catches **vertical** periodic patterns (warp
stripes / cord spacing).

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `fft_col_peak_K_energy` | Fractional energy of K-th col peak | `peak / total` | Strength of K-th vertical periodic pattern. |
| `fft_col_peak_K_period_px` | Period in pixels | `min(period / 1000, 1)` | Stripe spacing. |

**Use:** primary stripe / warp-defect detector. Tire-cord, denim warp, knit ribs.

---

## Family E — 2D FFT peaks (15 channels: top-3 ranks × 5 scalars)

Full 2D FFT → top-3 peaks. Each peak gets period **and** angle, so this is
where orientation lives.

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `fft2d_peak_K_energy` | Fractional energy of K-th 2D peak | `peak / total` | Strength of K-th periodic pattern, agnostic of direction. |
| `fft2d_peak_K_period_px` | Spatial period in pixels (along peak's frequency vector) | `min(period / 1000, 1)` | Spacing of the pattern. |
| `fft2d_peak_K_angle_deg` | Angle of peak from x-axis (0..180°) | `angle / 180` | Direct angle of the periodic pattern. **Diagnostic** (rules need a deviation channel — see next two rows). |
| `fft2d_peak_K_tilt_from_horizontal` | `|angle - 90| / 45`, clipped to 0..1 | direct | Deviation from horizontal. 0 = perfectly horizontal weft, 1 = ≥ 45° tilt. **Use this for "tilt > X°" rules.** |
| `fft2d_peak_K_tilt_from_vertical` | `min(|angle|, |angle-180|) / 45` | direct | Deviation from vertical. 0 = perfectly vertical warp, 1 = ≥ 45° tilt. |

**Reading the angles:**
- For tire cord: rank-1 peak ≈ vertical (warp cords) → `tilt_from_vertical` near 0.
- Rank-2 peak ≈ horizontal (weft bars) → `tilt_from_horizontal` near 0.
- For denim twill: rank-1 peak ≈ 63° → `tilt_from_horizontal` ≈ 0.6.

**Use:** `fft2d_peak_1_tilt_from_horizontal min_confidence 0.15` → fire when the
dominant periodic pattern tilts more than ~7° off horizontal (e.g. fabric
running askew through inspection).

---

## Family F — Band statistics (4 channels × N bands per channel)

Splits the frame into `MATH_BANDS` (default 8) **vertical bands across the
width** and emits a detection per band. Each detection's bounding box covers
the band's columns. This is the workhorse for shade analysis.

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `band_mean_L` | Mean L\* of the band | `L / 100` | Per-band brightness. With `count_greater` you can require N+ bands above threshold. |
| `band_std_L` | Stddev L\* of the band | `min(σ / 50, 1)` | Per-band texture; drops on local wash-out. |
| `band_delta_from_median` | `|band_mean_L − median(all bands)| / 30` | clipped 0..1 | **One-sided shade anomaly.** Spikes when one band is brighter or darker than the rest. Catches cone-shading without separate rules. |
| `band_delta_e` | CIE ΔE of the band's L\*a\*b\* vs frame median | `min(ΔE / 20, 1)` | **Color-aware** version — catches a band that is not just brighter/darker but a *different shade* (different hue/chroma too). |

**Reading the values:** with default 8 bands, a 2 m fabric has 25 cm bands.
`min_confidence 0.25` on `band_delta_e` ≈ "any band ΔE > 5 from frame median".
With `count_greater 2` on the same rule = "≥ 3 bands are anomalous" (more
robust trigger).

**Use:**
- Cross-direction (cone) shade: `band_delta_e count_greater 0 min_confidence 0.25`
- Demanding multi-band shade: `band_delta_e count_greater 2 min_confidence 0.15`
- Per-band brightness too high: `band_mean_L min_confidence 0.80`

---

## Family G — Row / column residuals (4 channels)

Single-line defects: take the column-mean signal, subtract its running median,
report departures. Same on row-mean.

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `col_residual_max` | Largest abs deviation in column-mean from its median | `min(|ΔL| / 50, 1)` | One column (or thin column band) is much brighter or darker than the rest → **broken warp / vertical line defect**. Bbox covers the offending column range. |
| `col_residual_count` | Number of columns beyond a fixed L\* threshold | `min(count / 100, 1)` | "How wide is the line defect." Use with `area_greater` to filter narrow noise. |
| `row_residual_max` | Same on row mean | `min(|ΔL| / 50, 1)` | One row much brighter/darker → **broken weft / horizontal bar**. |
| `row_residual_count` | Rows beyond threshold | `min(count / 100, 1)` | Bar width. |

**Why not just FFT?** FFT misses non-periodic defects. A single broken cord is
not periodic — its column-mean shows up as a spike, not as a frequency peak.

---

## Family H — Spacing irregularity (4 channels)

For periodic patterns: detect peaks in the row/col-mean signal, measure the
**gaps** between consecutive peaks, flag any gap that's far from the median.
This catches "two wefts collapsed into one spacing" or "one weft missing" —
defects that don't shift the average frequency much but break local rhythm.

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `row_spacing_anomaly` | One detection per anomalous row-gap. `confidence = |gap/median - 1|` | direct (clipped) | A pair of wefts too close, or a missing weft. Bbox covers the y-range of the offending gap. |
| `col_spacing_anomaly` | Same on column peaks | direct | Two warp cords collapsed, or a warp end dropped. |
| `row_spacing_std` | `stddev(gaps) / median(gaps)` | `min(value, 1)` | Global irregularity scalar. Single number that says "how regular is the periodic pattern." |
| `col_spacing_std` | Same for columns | `min(value, 1)` | Global irregularity in column direction. |

**Use:** `row_spacing_anomaly count_greater 0 min_confidence 0.30` → reject if
any row-gap deviates ≥ 30 % from the median spacing.

---

## Family I — Gradient / orientation (6 channels)

Sobel-based gradient analysis. Catches edge-rich defects and tells you whether
the local orientation is consistent (uniform fabric) or scrambled (defect).

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `grad_mean` | Mean Sobel gradient magnitude | `min(value / 50, 1)` | Overall edge density. Drops on washed-out fabric, rises on hard edges. |
| `grad_max` | Max Sobel magnitude | `min(value / 255, 1)` | Sharpest edge in frame. Spikes on sharp defects (cuts, foreign matter edges). |
| `grad_ori_coherence` | Anisotropy from gradient covariance eigenvalues. 0 = isotropic noise, 1 = one perfect direction | direct | **Texture regularity.** High coherence = clean weave; low coherence = scrambled / damaged area. |
| `grad_ori_dominant_deg` | Dominant structure orientation, mod 180° | `angle / 180` | Diagnostic. The bulk direction the fabric texture is running. |
| `grad_ori_tilt_from_horizontal` | `|angle − 0| / 45` (mod 180) | direct | One-sided rule channel. Fires when texture is significantly off-horizontal. |
| `grad_ori_tilt_from_vertical` | `|angle − 90| / 45` (mod 180) | direct | Same vs vertical. |

**Use:** `grad_ori_coherence` is interesting *inverted* — `min_confidence 0.30`
means "fire when coherence < 0.3 ≡ texture is scrambled." Since rules only have
`min_confidence ≥`, you'd need a paired `grad_ori_incoherence = 1 - coherence`
channel; for now use `band_delta_e` or blobs to catch texture loss.

---

## Family J — Morphology — tophat / bothat (4 channels)

**Top-hat** = original − opening: highlights bright spots smaller than the
structuring element. **Bot-hat** = closing − original: highlights dark spots.
Cheap (~3 ms) and very effective for small spot defects.

| Channel | Math | Confidence map | What it catches |
|---|---|---|---|
| `tophat_max` | Max of top-hat image | `min(value / 50, 1)` | Brightest small bright spot. Hot pixel, missing dye, hole with backlight. |
| `tophat_mean` | Mean of top-hat image | `min(value / 20, 1)` | Density of bright spots across the frame. |
| `bothat_max` | Max of bot-hat image | `min(value / 50, 1)` | Darkest small dark spot. Oil spot, knot, foreign matter. |
| `bothat_mean` | Mean of bot-hat image | `min(value / 20, 1)` | Density of dark spots. |

**Use:** `bothat_max min_confidence 0.30` → reject on any dark spot causing
≥ 15 L\* localized darkening. Pair with **K. Blobs** for area filtering.

---

## Family K — Blobs (variable count)

Threshold-based connected-component extraction with two polarities. Each
connected blob becomes its own detection with a real bounding box around it,
so `area_greater` rules work naturally.

| Channel | Per detection | Confidence map | What it catches |
|---|---|---|---|
| `blob_darkness` | One detection per dark blob (region with mean L\* below frame_median − 15) | `1 − mean_L_of_blob / 100` | Localized dark defects. Confidence rises with blob darkness. |
| `blob_brightness` | One detection per bright blob (mean L\* above frame_median + 15) | `mean_L_of_blob / 100` | Localized bright defects. |

**Use:**
- Reject any moderately-sized dark blob: `blob_darkness area_greater 400 min_confidence 0.50`
- Reject only very dark big blobs: `blob_darkness area_greater 1000 min_confidence 0.70`

Tune `area` in pixels² based on your camera resolution. At 1 mm/px, 400 px² ≈
2 cm² minimum defect.

---

## Quick reading guide for one frame

When debugging a specific frame, read the channels in this order:

1. **Quality first** (`saturation_fraction`, `sharpness_laplacian_var`) — is the frame trustworthy?
2. **Global** (`mean_L`, `std_L`) — overall brightness sane?
3. **Bands** (`band_delta_e` per band) — any cross-direction shade anomaly?
4. **2D FFT rank-1** (`fft2d_peak_1_*`) — what's the dominant texture and orientation?
5. **Spacing** (`row_spacing_anomaly`, `col_spacing_anomaly`) — any irregularity in the periodic pattern?
6. **Residuals** (`col_residual_max`, `row_residual_max`) — any single line defect?
7. **Blobs** + **Morph** (`blob_*`, `tophat_*`, `bothat_*`) — any localized spot defects?

---

## Tunable parameters (env vars on the math container)

| Var | Default | What it controls |
|---|---|---|
| `MATH_TILES_X` | `1` | If > 1, also emit `tile_*_shift` channels split into tiles across the width |
| `MATH_TILES_Y` | `1` | Same for height |
| `MATH_BANDS` | `8` | Number of vertical bands in family F |
| `MATH_FFT_TOP_K` | `3` | Number of FFT peaks per family C/D/E |
| `MATH_FLAT_FIELD_ENABLE` | `false` | Subtract slow illumination variation before measuring |
| `MATH_DEVICE` | `auto` | `auto` / `cpu` / `cuda`. `auto` falls back to numpy if CuPy can talk to the GPU but JIT is broken (e.g. runtime-only base image) |

Tile channels (when `TILES_X · TILES_Y > 1`) add 6 channels per tile:

- `tile_row_period_shift`, `tile_col_period_shift` — local FFT period differs from frame median
- `tile_row_energy_excess`, `tile_col_energy_excess` — local FFT energy is anomalous
- `tile_mean_L_shift`, `tile_std_L_shift` — local brightness/texture differs

Each emits one detection per anomalous tile, bbox = tile coordinates.

---

## Common rule recipes

| Operator intent | Rule |
|---|---|
| Reject if any band is anomalous in shade | `band_delta_e count_greater 0 min_confidence 0.25` |
| Reject only on multi-band shade defect | `band_delta_e count_greater 2 min_confidence 0.20` |
| Reject if frame is too bright | `mean_L count_greater 0 min_confidence 0.80` |
| Reject vertical line defect | `col_residual_max count_greater 0 min_confidence 0.50` |
| Reject horizontal bar defect | `row_residual_max count_greater 0 min_confidence 0.50` |
| Reject dark spot of meaningful size | `blob_darkness area_greater 400 min_confidence 0.50` |
| Reject if periodic pattern lost regularity | `row_spacing_std count_greater 0 min_confidence 0.30` |
| Reject if fabric is tilted ≥ ~7° off horizontal | `fft2d_peak_1_tilt_from_horizontal count_greater 0 min_confidence 0.15` |
| Pre-gate: skip eject when frame is blurry | (combine: only enable other rules when `sharpness_laplacian_var min_confidence 0.30` first — needs ANY logic) |

---

## Tuning workflow

1. **Run with no rules** — just observe channel values for 100+ known-good
   frames. Identify the operating range of each channel for *your* fabric.
2. **Set thresholds at p99 of normal** — pick `min_confidence` slightly above
   the highest value seen on good fabric. This minimizes false positives.
3. **Validate against known defects** — confirm rules fire on known-bad frames.
   If a real defect doesn't fire, lower the threshold or add a complementary rule.
4. **Layer rules with ALL logic** for compound defects (e.g. `band_delta_e`
   *AND* `tophat_max` together = "shade band that also has a spot in it").

The thresholds in the recipes above are reasonable starting points — *not*
final tuned values. Every fabric SKU (especially color, weave density, surface
finish) shifts the operating range. Plan one shift of "learn baseline" before
enabling automatic ejection.
