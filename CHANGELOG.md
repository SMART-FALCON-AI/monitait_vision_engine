# Changelog

All notable changes to MonitaQC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.21.19] - 2026-06-08

### Fixed — Process tab "Show" toggle now actually shows bboxes (two-dict sync bug)
- **Symptom (observed on fabriqc-kc):** a class is detected, a row lands in `inference_results`, the chart's click-through drawer can find the frame — but the `_DETECTED.jpg` on disk has no green rectangle, even though the operator ticked **Show** on that class in the Process tab. Affected `agh` and `tooli_up` specifically because those classes existed in `audio_settings` (Show=True) but had no entry in the older `timeline_config.object_filters` dict.
- **Root cause:** `services/detection.py:1107` (3.21.10) read `timeline_config.object_filters` exclusively and treated a *missing entry* as "hide". But the Process tab Show toggle writes to `service_config.audio_settings.<class>.show`, not to `object_filters`. The Advanced tab's "Apply Timeline Configuration" button is the only thing that ever writes `object_filters`, and on fabriqc-kc that dict was last edited before `agh` / `tooli_up` were added to the model. So the annotator silently skipped them while the DB write further down (which uses `audio_settings.store`) happily persisted them — visible everywhere except the annotated jpg.
- **Fix:** the annotator now reads both dicts. A class is skipped only when *either* dict has an explicit `show: false`. A missing entry now means "draw" (matching the Process tab's natural UX). The Advanced tab's per-class filter list still works for operators who want to mute a class without un-Storing it.
- Files: `services/detection.py:1087-1124`. No new dep, no schema change.

## [3.21.18] - 2026-06-05

### Fixed — PDF score number alignment
- The "100.0/100" sat below the RELEASE pill's center because `leading=50` made the score paragraph's bbox 8pt taller than the visible glyphs, and the table's VALIGN centered the bbox not the text. Set `leading == fontSize` (42), `alignment=center`, and matched 12pt top/bottom padding on both cells. Score baseline and pill center now sit on the same horizontal line.

## [3.21.17] - 2026-06-05

### Fixed — PDF report layout (3.21.15 follow-up)
- **Score number overlapped the Thresholds line** because the 36pt font ran on default `leading=14`. Lifted leading to 50pt and switched the score paragraph to a dedicated `ScoreXL` style. No more overlap.
- **Verdict pill stretched the full column width** (looked like a green bar, not a pill). Pill is now an auto-sized nested table — sized to its content, 7pt vertical padding, darker border in the same hue.
- **Defects table headers "Count/encoder_unit" and "Impact/encoder_unit" overlapped neighboring columns.** Headers are now wrapped in `Paragraph` cells that break onto two lines (`Count` / `/m`). Column widths bumped from 22/28 to 22/28 with the wider Class column at 44mm.
- **"encoder_unit" placeholder showed literally** in length / throughput / per-unit rows when the operator hadn't configured a unit. Now falls back to plain `unit` / `units/sec` / `/unit`.

### Added — Encoder value + camera index in the click-through drawer
- Clicking a dot on the Camera × Encoder scatter now opens the drawer with `encoder N · cam K` in the meta line, so you can read the roll position of the dot you clicked without having to eyeball the X-axis.
- Math-channel values are now labelled `math metric X` (was just `value X`), so it's obvious that the number is a raw channel reading, not a confidence percentage.

## [3.21.16] - 2026-06-05

### Added — Shipment quality drift / trend chart (Phase 2)
- New endpoint `GET /api/shipment_quality_score/trend?window=24h&shipment=X&buckets=12`. Splits the window into ~12 time-buckets and computes per-bucket `impact_per_unit` using the same severity map + encoder normalization as the main score endpoint. Returns per-bucket `score`, `impact`, `impact_per_unit`, `encoder_span`, `detections`, `frames`, plus a `slope_label` of `improving` / `stable` / `degrading` based on first-third vs last-third average impact/unit.
- Frontend: new compact line chart under the Quality Score card showing impact/unit over time, plus a colored slope chip (green=improving, amber=stable, red=degrading) with the percentage delta. Auto-refreshes alongside the rest of the Charts tab.
- Bucket widths: 5 min for 1h, 30 min for 6h, 2 h for 24h, 14 h for 7d — about 12 buckets per window so the trend is readable but not noisy.
- This catches *quality drift within a passing shipment* — e.g. average score is still 90 but the last hour has degraded to 72. The chart and chip make it visible before the next shift starts.

## [3.21.15] - 2026-06-05

### Added — Shipment Quality Score PDF report (Phase 2)
- New endpoint `GET /api/shipment_quality_score/report.pdf?shipment=X&window=Y` streams an `application/pdf` document built with ReportLab. Same payload as the JSON endpoint, just rendered for print / email / regulatory filing.
- Layout: score-and-verdict block (color-coded green/amber/red), production KPI grid (length / duration / throughput / total impact / impact-per-unit / normalized-by / frame count), top-defects table with `count`, `count/unit`, `severity`, `impact`, `impact/unit`. Footer shows window first→last and the MVE version that generated it.
- Frontend: "📄 Download PDF" button on the Quality Score card → uses the currently-selected window and shipment → streams the file via Blob so the browser triggers the native download dialog.
- Refactor: extracted `_compute_quality_payload(shipment, window)` helper in `timeline.py` so both the JSON and PDF endpoints share one source of truth.
- New dep: `reportlab>=4.0.0,<5.0.0` in `vision_engine/requirements.txt`. On running deployments, install once with `docker exec monitait_vision_engine pip install reportlab` (rides into the next image bake automatically). Endpoint returns a 503 with the install hint if reportlab is missing — operators get a clear message instead of a server crash.

## [3.21.14] - 2026-06-05

### Changed — Shipment Quality Score is now length-aware (encoder-normalized)
- Previous formula divided by `total_detections`, which was inflated by jean / math-channel rows that have no severity weight. Result: every shipment scored ~99 regardless of defect density.
- New formula: `impact_per_unit = impact_total / encoder_span`, where `encoder_span = MAX(encoder) − MIN(encoder)` over the window. A 2-hour shipment and a 4-hour shipment with the same defect density now get the same score.
- Falls back to `frame_count` when encoder data is missing.
- The card displays the basis: "Normalized by encoder span (4,872 m)" or "Normalized by frame count (18,231 frames) — no encoder data".

### Added — Shipment Quality Score card now shows length / duration / throughput / impact-per-unit
- New fields surfaced in the card: encoder length (in the operator's chosen unit), duration in `Xh Ym`, throughput in `units/sec`, and `impact / unit`.
- Top defects panel now shows `impact_per_unit` alongside raw impact, so the actionable metric ("0.6 weft_up / m") is visible at a glance.

### Added — Encoder Calibration section in the Advanced tab
- New section with two fields: `encoder_unit` (free text label — `m`, `mm`, `pieces`, `ticks`, …) and `encoder_units_per_meter` (optional float for physical-length conversion).
- Wired through new endpoints `GET / POST /api/encoder_calibration`, stored in `service_config`.
- The card's "/unit" labels and the encoder length read pick this up immediately.



### Added — Shipment Quality Score card (Phase 2 preview, free)
- New `GET /api/shipment_quality_score?shipment=X&window=Y` returns a 0–100 quality score, a `RELEASE` / `RE-INSPECT` / `HOLD` verdict, total impact across the window, and the top 5 defect classes by impact. Score formula: `100 × (1 − impact_total/total_detections)`, clamped 0–100. Verdict thresholds: ≥85 RELEASE, 60–85 RE-INSPECT, <60 HOLD (in-code defaults; will become UI-tunable in a later release).
- Charts tab now shows a prominent summary card above the existing charts: the score, the verdict (color-coded), impact total, detection count, and the top defects with their per-class impact / count / severity. Updates automatically when the operator changes Window / Shipment.
- Reuses the severity field added in 3.21.12. Classes with severity 0 contribute nothing to the score, so the card stays meaningful even when only a few classes are weighted.

## [3.21.12] - 2026-06-05

### Fixed — yolo weights binding: uploads survive container restart
- Added single-file bind-mount `./volumes/weights/best.pt:/code/best.pt:ro` under the `yolo_inference` service. The Detector still loads from the same `/code/best.pt` path, but that file is now overlaid by the host's `./volumes/weights/best.pt`, which means replacing the file in the clean weights-only directory updates the model on next yolo restart without touching the directory that holds the yolov5 source code.
- `routers/inference.py`: both `upload_weights` and `activate_weights` endpoints now mirror the chosen file to `/weights/best.pt` as part of their flow. Result: uploads via the Process-tab UI activate immediately AND survive container restarts. Previously the activation was in-memory only and was lost when the container restarted.
- A misplaced single-file override that earlier ended up under `monitait_vision_engine` instead of `yolo_inference` was removed.

### Added — Phase 1 foundations for paid Analyze tab (free for everyone)

#### Severity per class — Process tab
- New per-class severity weight (0–100) input in the per-object card, next to Min-conf. Stored in `service_config.audio_settings[class].severity`. Default 0 = cosmetic, no contribution to impact score.
- Sent in `/api/audio_settings` payload alongside show/store/beep/narrate/min_confidence.

#### Per-class confidence baseline — read-only badge
- New endpoint `GET /api/conf_baselines` returns auto-computed `{class: {p50, p95, n}}` from the last 7 days of stored detections in `inference_results`. Cached for 1 hour.
- Process tab renders a badge under each card: "📊 normal conf: 78–94% (p50–p95) · n=27885".
- Helps operators set `min_confidence` correctly and judge whether a detection is anomalous.

#### Impact score — chart endpoints + CSV
- `defect_impact = (severity / 100) × confidence` aggregated per class (area_factor postponed until a typical-area baseline is available).
- `/api/detection_stats` response now includes `impact_by_class` and `impact_total` alongside `by_class` and `total`.
- CSV export adds two columns: `severity` and `impact`. Operators can sort/filter exported data by per-detection impact.

## [3.21.11] - 2026-06-02

### Fixed — Charts: per-class scatter so rare classes survive vs weft_up flood
- `camera × time` and `camera × encoder` scatter queries used `LIMIT 1500` on a `time DESC` order. With weft_up dominating (~27k/24h) the newest 1500 rows were all weft_up and rare classes (spot/warp/stitch) never appeared. Switched to a stratified `ROW_NUMBER() OVER (PARTITION BY cls)` slice — up to 750 newest dots **per class**, capped 6000 total. All classes are now visible regardless of class imbalance.

### Fixed — Window dropdown collapsed to "last few hours" at high detection rate
- The `recent` CTE in `/api/detection_stats`, `/api/detection_charts`, `/api/quality_charts` had a `LIMIT 20000` cap. At ~25 fps of weft_up the 20k rows represented ~13 minutes of data, so `window=7d` returned the same totals as `window=6h`. Removed the row caps; `WHERE time > NOW() - INTERVAL` already bounds the scan size. 7d now actually returns 7 days.

### Fixed — Spot/Warp ejection_events under-reported (sampled by dashboard refresh)
- The eject DB-write hook lived inside `evaluate_eject_from_detections` which is also called from the dashboard's WebSocket render loop. That loop sampled the current 24-frame dashboard window, so constant classes (weft_up) fired reliably but rare classes (spot_up, warp_up — ~71 / 579 in 24h) almost never landed in the sampled window. Added a second hook in the **live inference path** (`detection.py` after `write_inference_to_db`). Every captured frame's eject is now evaluated regardless of dashboard state. Same Redis dedupe gate.

## [3.21.10] - 2026-06-01

### Fixed — ejection_events table never populated (3+ months of missed eject history)
- `evaluate_eject_from_detections` is called from 4 places (worker thread `_process_frame_batch`, websocket dashboard push, two timeline endpoints), but only the worker-thread path had a DB-write hook — and on this workload the worker thread wasn't being reached. Result: even with procedures correctly configured (`enabled: true, store: true`), the table stayed at 0 rows.
- Moved the DB-write hook **inside `evaluate_eject_from_detections` itself**. Every caller now persists. Per-(encoder, procedure_name, shipment) dedupe via a Redis key with 60s TTL prevents the dashboard's repeated re-evaluation from inserting duplicates.

### Fixed — Process-tab Show toggle had no effect on dashboard annotations
- The Process-tab UI saves show/min_conf toggles to `timeline_config.object_filters`, but the drawing code in `detection.py:1064` was reading from `service_config.audio_settings` — two unrelated dicts that were never synced. Result: every class always drew regardless of the user's checkbox state. Re-pointed the drawing filter at `timeline_config.object_filters`. A class now draws only if it has an explicit entry with `show != false`; missing entry hides.

### Fixed — confidence-slider didn't actually filter anything
- 3.21.9 added the UI slider and threaded `min_conf` as a query param through `/api/detection_charts`, `/api/detection_stats`, `/api/quality_charts`, but most SQL queries inside those endpoints **never applied** the param. Sliding to 98% still showed low-confidence defects because the JSONB-expand step ran without a `WHERE conf >= %s` filter on the size-over-time, confidence-over-time, camera scatter, and camera×encoder scatter queries. Patched all four to include the filter at the expand step.

### Fixed — unchecking Show in Process tab didn't hide annotations
- The UI's "uncheck + Save All Configs" deletes the per-class entry from `audio_settings`, but the drawing code at `detection.py:1064-1073` defaulted to "draw" when a class had no entry. So a missing entry was treated as visible, and the user's uncheck silently did nothing. Flipped the default: **missing entry = do not draw**. New semantic: a class is drawn only if `audio_settings` has an explicit entry for it with `show != false`.

### Fixed — defect-modal zoom capped at 1x
- 3.21.9's modal-layout patch added `contain: 'inside'` to the Panzoom config to keep the image within the modal frame. That option silently caps zoom at 1x because at scale > 1 the image becomes larger than its parent, which violates the "inside" rule. Removed the `contain:` option entirely — CSS (`max-width: 100%; object-fit: contain`) handles fit-on-load; flex `1 1 0` keeps the 50/50 panel split; Panzoom now zooms freely up to `maxScale: 10`.

## [3.21.9] - 2026-06-01

### Fixed — shipment column always tagged `no_shipment` (cross-store Redis bug)
- Detections were being written to `inference_results.shipment = 'no_shipment'` regardless of the active shipment, because the **UI write path** stored the shipment in Redis **db=3** while the **DB-write read path** in `detection.py:1155` read from Redis **db=0**. Two stores, never the same value. The filesystem path (which uses the in-memory `watcher.shipment`) was unaffected — that's why `raw_images/3/…/` had the right folder but the corresponding DB rows said `no_shipment`.
- Sweep across **9 files / 22 sites**: replaced every hardcoded `db=0` and `db=3` with `config.REDIS_DB` (or the file's local config alias). Added `REDIS_DB = int(os.environ.get("REDIS_DB", 0))` to `config.py` — single env, defaults to 0, future-proof.
- Files touched: `config.py`, `services/redis_service.py`, `services/detection.py`, `services/watcher.py`, `routers/config_routes.py`, `routers/timeline.py`, `routers/health.py`, `routers/websocket.py`, `routers/inference.py`, `routers/ai.py`. No new code, all mechanical.

### Added — Confidence threshold slider on the Charts tab
- New range slider next to Window / Shipment / CSV. Slides 0–100%. On change, refreshes the charts via the existing `min_conf` query param that the three chart endpoints (`/api/detection_stats`, `/api/detection_charts`, `/api/quality_charts`) already accept. SQL filters at the JSONB-expand step, so the count tiles, distribution charts, and scatter plots all respect the threshold.

### Changed — Defect modal defaults to **Both** (raw + annotated) when both URLs are available
- Previously the modal opened in Annotated-only view. Most charts pass both URLs, so the default now shows them side-by-side at the same time. Single-image classes still open in whichever view is available.

## [3.21.8] - 2026-06-01

### Fixed — Defect modal: image stays inside the modal frame
- Panzoom was configured with `contain: 'outside'` (cover-style) which let the image pan past the modal edges, so on first open the annotated image would visibly overflow to the right/bottom. Switched to `contain: 'inside'` and call `pz.reset({animate:false})` immediately after attach to guarantee a centered initial state at scale 1.

## [3.21.4] - 2026-05-31

### Changed — Dashboard image click now opens the unified defect modal
- Clicking a frame on the **Dashboard timeline** used to open a small custom popup. Now it routes through the same **centered modal** as the chart drill-downs: Annotated/Raw/Both toggle, Download buttons, Panzoom zoom/pan. The dashboard click passes its `frameUrl` (`/api/timeline_frame`) and `rawUrl` explicitly via new `annotated_url`/`raw_url` overrides on `openDefectDrawerForFrame`, so it works even when **Store=off** and the annotated jpg isn't persisted.
- The frame's encoder + eject status + camera show up as `tags` in the modal title.

### Note
The dashboard wheel-zoom fix from 3.21.2 (`{passive:false}` on the wheel listener) is already deployed. If wheel-zoom still feels broken, hard-refresh (Ctrl+Shift+R) to force-load the new audio.js.

## [3.21.3] - 2026-05-31

### Added — Export CSV from the Charts tab
- New green **⬇ CSV** button next to the Shipment selector. Exports **all stored detections** for the current shipment + window as a flat CSV (one row per detection): `time, shipment, encoder, camera, class, confidence, xmin, ymin, xmax, ymax, image_path, inference_time_ms, model_used`.
- Backend: new `GET /api/export_csv?window=&shipment=` streams via a **server-side cursor** (low RAM even on millions of rows) with proper Content-Disposition (filename `detections_<shipment>_<window>.csv`) and csv.writer quoting (handles commas/quotes in class names safely).

## [3.21.2] - 2026-05-30

### Changed — defect drill-down is now a centered modal of THE ONE clicked frame
- Clicking a chart point used to open a side panel with up to 24 thumbnails of the class — way too much. Now click → **centered modal** showing the **exact clicked frame** at large size, with a header **Annotated / Raw / Both** toggle and a **Download** button on each image. Scroll to zoom, drag to pan, double-click to reset.
- Scatter dots (Camera × time, Camera × encoder) pass their own `image_path` + `shipment` directly — no extra fetch, no class-aggregate dump.
- Bar / pie / pareto clicks now fetch only the **latest 1** frame of that class (instead of 24) and route through the same single-frame modal.

### Changed — annotated images only draw classes with `Show=true`
- The annotated `_DETECTED.jpg` saved per frame used to draw every detection — math channels (`blob_brightness`, `blob_darkness`, `fft_*`, `band_*`, …) plastered the image and buried the real yolo defects. Now `detection.py` consults `audio_settings` and **skips drawing any class whose `Show` flag is off** (default = on, so new/unseen classes still render). Toggle `Show` in Process tab → Per-Object Configuration to control which classes appear in stored annotations.

### Changed — lightweight dashboard (no off-page records)
- Removed **"Total Frames Stored"** slider from the Advanced tab. Per-camera Redis buffer is now bound to **Rows per Page** (`max(5, num_rows)`), so the dashboard stores **only the current page** in memory. Pagination collapses to a single page (no `< >` arrows), `/status` load is lighter, no RAM bloat.

### Fixed
- Dashboard timeline-image **wheel zoom** stopped working in modern browsers because the listener was registered without `{passive: false}` → `preventDefault()` was ignored → the page scrolled instead of zooming. Added `{passive: false}` + explicit `preventDefault()`.

### Migration
Browser hard-refresh (Ctrl+Shift+R) once after deploy so the new modal markup and scripts load instead of the cached side-panel layout.



## [3.21.1] - 2026-05-30

### Changed
- **Tab order**: Dashboard → **Charts** → AI Assistant → Gallery → Hardware → Cameras → Inference → Process → Advanced. Charts moved next to the Dashboard since it's the operator's next stop for detail.
- **Dashboard pagination removed**: the `<<`/`<`/`>`/`>>` buttons on the timeline slideshow are gone. The dashboard now shows the **latest frame only** (label "live"), with zoom (+/–/Reset) and Stop/Resume retained. Eliminates the RAM leakage from paging through old stitched frames; for historical detail use the Charts tab + drawer.

## [3.21.0] - 2026-05-30

### Added — Operator chart-to-image drill-down + Camera × Encoder scatter
Charts are now an entry point to the actual defect images, not just summaries.

- **Hover a scatter dot** (camera×time or camera×encoder) → a floating thumbnail of *that exact annotated frame* (`_DETECTED.jpg` with bbox drawn) appears next to the cursor. Click the dot to open the full image. Implemented as a Chart.js v4 external tooltip + a single reusable `#chart-image-preview` overlay div.
- **Click a class** in the *Detections-by-class* bar, *Detection-distribution* pie, or *Pareto* bar → a side **defect drawer** (`#defect-drawer`) slides in with the **last 24 thumbnails** of that class (with timestamp, classes-in-frame and confidence). Click a thumbnail to open the full image. Close with `✕` or `Esc`.
- New endpoint **`GET /api/recent_detections?cls=&window=&shipment=&limit=24`** powers the drawer (`inference_results` rows containing the class, filtered by window/shipment, with `image_path` + best confidence + classes-in-frame).
- `image_path` is now included on each point of `/api/detection_charts.camera_scatter`.

### Added — Camera × Encoder scatter (roll-position defect map)
- New chart **"Camera × encoder (roll position)"** alongside the existing camera × time scatter — same per-class colors, hover-image and click-to-open behavior, but x-axis = **encoder position** so an operator can locate a defect along the roll, not just along time.
- Powered by a new `camera_scatter_encoder` field on `/api/detection_charts`.

### Added — Encoder persisted per detection (write-path change)
- `inference_results` gets a new column **`encoder_value BIGINT`** (added to `init.sql` for fresh installs and applied at runtime via `ALTER TABLE … ADD COLUMN IF NOT EXISTS` in `db.py`, so existing DBs pick it up without re-init).
- `write_inference_to_db(...)` takes a new `encoder_value=` kwarg; the call site in `services/detection.py:process_frame` passes the capture-time `encoder`.
- Older rows (before the column existed) will plot only on the time scatter; new rows populate the encoder scatter.

## [3.20.1] - 2026-05-29

### Fixed — DVR cleanup silently never triggered (data-loss bug)
- **`_ensure_disk_space` used `shutil.disk_usage().total` as the denominator**, which counts ext4's 5%-reserved-for-root blocks. Result: when `df` showed 80% disk usage, the code calculated only ~71% and concluded "under threshold, no cleanup needed" — so it never deleted oldest chunks. The disk-write queue then filled and frames were dropped (we observed ~2289 dropped frames in 5 minutes on vteam19).
- Fix: compute `pct = used * 100 // (used + free)` to match `df`'s view (excludes the reserved blocks). `_DISK_MAX_PCT = 75` now genuinely means "df shows 75%", and the DVR ring-buffer triggers as intended.

## [3.20.0] - 2026-05-25

### Added — per-phase pipeline `stride` (run a model every Nth frame)
- `PipelinePhase` gains a `stride` field (default 1 = every frame). `PipelineManager.run_inference` keeps a frame counter and skips a phase when `frame_count % stride != 0`. Lets a fast primary model run every frame (e.g. yolo `stride: 1` for per-piece ejection) while a heavy secondary samples (e.g. math `stride: 5` → every 5th frame). This keeps the inference loop from being gated by a slow secondary model — on a shared GPU it's the difference between the ejector queue backing up (CRITICAL) and keeping pace. Set per phase via `POST /api/pipelines` (the phase dict accepts `stride`). Backward compatible: existing pipelines default to `stride: 1`.

## [3.19.2] - 2026-05-21

### Fixed
- **Autoscaler crashed every cycle** (`_autoscaler` in main.py): the function declared `global INFERENCE_WORKERS` but not `global _ok_streak`, so `_ok_streak += 1` raised `local variable '_ok_streak' referenced before assignment` on every check (~every 30s). Result: inference-worker autoscaling never ran and the log spammed errors. Added `_ok_streak` to the `global` declaration.

## [3.19.1] - 2026-05-21

### Fixed — "Save All Configuration" was wiping settings
- **`build_current_service_config()` rebuilt service_config from scratch and dropped/reset persisted-only fields.** It hardcoded `inference.current_module = "gradio_hf"` and omitted `store_objects` and `audio_settings` entirely. So every "Save All Configuration" click (and the camera-save paths that call it) silently **reset the active inference module to Gradio and wiped every per-class Store / Show / Narrate / Beep flag** — making it look like changes "weren't saved" when in fact they were being overwritten.
- Now the builder **loads the persisted config and carries over `inference` (current_module + module URLs), `store_objects`, and `audio_settings`** instead of clobbering them. Runtime-derived fields (cameras, states, pipeline_config, infrastructure, ejector, etc.) are still rebuilt as before.

### Note
- There is still no UI control that *persists* a change to `inference.current_module`; switching the active model reliably currently requires editing `.env.prepared_query_data` while the container is stopped (the running container periodically reserializes service_config). Tracked for a follow-up.

## [3.19.0] - 2026-05-21

### Added — OEE + line KPIs in Production panel
`GET /api/production_stats` now also derives, from the cumulative PLC/encoder columns:
- **OEE** = Availability × Performance × Quality (per-bucket + overall)
  - **Availability** = % of samples with `is_moving`
  - **Quality** = OK / (OK+NG)
  - **Performance** = avg speed / max speed (speed = encoder delta ÷ bucket seconds; max speed = peak observed in window, used as the ideal-rate proxy)
- **Downtime** total — sum of `downtime_seconds` deltas (reset-clamped)
- **Eject / Total** — NG / total units (machine reject rate)
- **Speed avg / max** — encoder units/sec

Charts tab → Production KPIs panel gains a **KPI cards strip** (OEE colored by 85/60 thresholds, Availability, Performance, Quality, Eject/Total, Downtime, Speed avg/max) plus two charts: **OEE over time** and **Line speed vs max**. The panel now also shows when the line is moving even if OK/NG counters aren't wired (keys on units OR speed OR downtime).

## [3.18.0] - 2026-05-20

### Added — Production KPIs + Defect Diagnostics (8 new charts)
Two new panels in the Charts tab, both honoring the window + shipment selectors.

**🏭 Production KPIs** (new `GET /api/production_stats`) — finally surfaces `production_metrics`, which was collected (PLC serial OK/NG/downtime/encoder) but never charted. OKC/NGC are cumulative hardware counters, so the endpoint diffs consecutive samples per bucket and clamps negatives (counter resets on restart):
- **Yield: OK / NG + reject rate** — stacked OK/NG bars with a reject-% line on a second axis
- **Throughput** — units processed per bucket
- **Line uptime** — % of samples with `is_moving` (stoppages visible as dips)
- **SPC p-chart** — reject rate with center line + per-point 3σ control limits (UCL/LCL computed from `p̄` and the per-bucket sample size `n`); out-of-control points turn red

**🔬 Defect Diagnostics** (new `GET /api/quality_charts`, single expansion pass over `inference_results`):
- **Pareto of defects** — counts desc + cumulative-% line (the "vital few"). Honors the per-class **Show** toggle
- **Defects by camera / station** — localize which camera position sees the most defects
- **Defect location heatmap** — bbox centers binned into a 32×20 grid (normalized by max observed x/y), rendered as a density bubble field laid out like the camera frame (y reversed)
- **Inference latency over time** — avg/max `inference_time_ms` for model/pipeline health

## [3.17.0] - 2026-05-20

### Added — Ejection logging + Ejection Insights charts
- **Per-procedure `Store` checkbox** (Process tab → Ejection Procedures, next to Enabled). When ON, every time that procedure *fires an eject* one row is logged to the database. Persists immediately (like the per-class Store), saved in `timeline_config.procedures[].store`. Off = the eject still fires, it just isn't logged
- **New `ejection_events` hypertable** (`time, shipment, procedure_name, reason, encoder_value`). Created by `init.sql` on fresh installs **and** by a runtime `CREATE TABLE IF NOT EXISTS` migration in `services/db.py` (retried at startup) so existing databases get it without a manual step
- **Eject path now persists events** (`main.py`): when `should_eject`, each *triggered* procedure with `store=ON` is written via the async DB queue (`write_ejection_event_to_db`) — reusing the eject reason string, current shipment and encoder. Non-blocking; failures are swallowed so they can never stall the ejector hot path
- **New endpoint `GET /api/ejection_stats?window=1h|6h|24h|7d&shipment=`** → `{by_procedure, timeline, total, shipments}`. Bucketed via `time_bucket`; returns a well-formed empty payload if the table doesn't exist yet
- **New "Ejection Insights" panel** in the Charts tab with three charts: **Ejections by procedure** (bar), **Ejection distribution by procedure** (doughnut, with %), and **Ejections over time** (line). Honors the same window + shipment selectors as Detection Insights. Only procedures with Store=ON appear (the DB-gating is the "don't show untracked" rule applied to procedures)

## [3.16.2] - 2026-05-20

### Added
- **Detection distribution doughnut** (`insight-class-pie`): a pie/doughnut chart of detections-per-class share, paired with the existing bar. Tooltip shows count + percentage. Honors the per-class **Show** toggle (uses the same Show-filtered class list as the bar)

### Changed
- **Removed the Grafana placeholder text** from the Charts tab. The `#grafana-future-embed` hook div is kept in the DOM (now `display:none`, empty) so a future custom Grafana iframe can still be dropped in, but nothing renders for now

### Note
- "Confidence of each class vs time" was already added in 3.16.1 (`insight-confidence-class-chart`, one line per class). On machines where only one class has Store=ON it shows a single line; it splits per class as more classes opt in

## [3.16.1] - 2026-05-20

### Added
- **Confidence by class over time** (`insight-confidence-class-chart`): a multi-line chart in the Charts tab, one line per class (avg confidence per time-bucket, 0–100%), colored by the same stable per-class palette as the scatter. Complements the existing aggregate min/avg/max band — lets you see *which* class's confidence is drifting. Backed by a new `confidence_by_class` field on `GET /api/detection_charts` (`{buckets, series:{cls:[...]}}`)

### Changed — charts now honor the per-class "Show" toggle
- The per-class **Show** flag (Process tab → Per-Object Configuration, persisted in `audio_settings`) now also controls chart visibility. Classes with **Show=off** are excluded from every per-class chart: **Detections by class** (bar), **Confidence by class** (lines), and the **Camera × time scatter**. `charts.js` fetches `/api/audio_settings` on each refresh and filters client-side; default (no flag / `show!==false`) keeps the class visible. Aggregate charts (size band, confidence band, detections-over-time) are unaffected since they aren't per-class

## [3.16.0] - 2026-05-20

### Added — Charts tab: three new detection-quality charts + shipment filter
- **Object size distribution over time** (`insight-size-chart`): per time-bucket width p10/p50/p90 band + width & height medians, so size drift / outliers are visible across the window
- **Detection confidence over time** (`insight-confidence-chart`): per time-bucket min/avg/max confidence band (y axis fixed 0–100%), to spot model-confidence degradation
- **Camera × time scatter** (`insight-camera-scatter`, bubble): x = time, y = camera number, dot color = class (stable hash → palette), dot size = confidence. One dataset per class so the legend doubles as a class filter
- **Shipment filter** (`#insight-shipment`): dropdown scopes the three advanced charts to a single `shipment` (or All). Populated from distinct shipments seen in the window
- **New endpoint `GET /api/detection_charts?window=1h|6h|24h|7d&shipment=`** — returns `{shipments, size_over_time, confidence_over_time, camera_scatter}`. Uses TimescaleDB `time_bucket` + `percentile_cont` over `jsonb_array_elements`-expanded detections, capped at the most recent 20000 rows, returns a well-formed empty payload on any DB/query error

### Changed
- **Grafana service removed from `docker-compose.yml`** — the Charts tab is now fully self-served by MVE (embedded Chart.js), so the extra container is no longer needed. A `#grafana-future-embed` placeholder remains in the UI for a future custom Grafana embed; re-add the service if/when that's wired

## [3.15.7] - 2026-05-20

### Fixed — Export/Import now round-trips EVERYTHING
- **Export was incomplete**: `GET /api/export_service_config` dumped only `service_config` (via `load_service_config`), silently dropping the root-level `timeline_config` — which holds the **ejection procedures** and timeline object filters. Export → wipe → Import lost all procedures. Now exports the **entire data file** (`load_data_file()`): `service_config` (cameras, states, pipeline_config, store_objects, audio_settings, datamatrix, …) **and** `timeline_config` (procedures + filters) **and** any other top-level section
- **Import now restores the full bundle**: `POST /api/cameras/config/upload` detects the full-bundle shape (presence of a `service_config` key) and `save_data_file()`s the whole thing, then calls `apply_config_settings(svc, watcher, full_data=...)` so `timeline_config`/procedures are actually applied to the live system — previously `full_data` was never passed, so even if procedures had been in the file they wouldn't load without a restart
- **Backward compatible**: a legacy export (flat service_config, no wrapper) still imports correctly via the else-branch
- **Cache invalidation on import**: resets `services.detection._store_objects_loaded_at` / `_audio_settings_loaded_at` so imported Store/audio settings take effect immediately instead of after the 5s cache TTL

### Result
Export → Import is now a true full backup/restore: procedures, per-class Store flags, audio settings, camera config, states (incl. per-state exposure/gain), pipelines, datamatrix — all of it.

## [3.15.6] - 2026-05-20

### Fixed
- **"Save All Configuration → Last: Never" after refresh** (`static/js/audio.js`): `saveAllServiceConfig()` set the "Last:" timestamp only in the DOM via `new Date()`, so it was lost on reload and reset to the i18n "Never" default — even though the save itself persisted `service_config.saved_at` correctly. Added `loadLastSavedTime()` on DOMContentLoaded that reads `config.saved_at` from `/api/cameras/config` and populates `#last-saved-time`, so the real last-saved time survives refresh. The save handler now also prefers the server-persisted `saved_at` over the client clock

## [3.15.5] - 2026-05-20

### Fixed
- **Charts tab blank on offline/LAN machines**: 3.15.3 loaded Chart.js from the jsdelivr CDN. Air-gapped deployments (e.g. vteam12) can't reach the internet, so `Chart` was undefined and the embedded insight charts never rendered. Chart.js v4.4.1 is now **bundled locally** at `static/js/chart.umd.min.js` and referenced with a relative path — works with zero internet access

### Notes
- This release also flagged a deployment-hygiene issue: machines upgraded by piecemeal file-sync (rather than a clean image pull) can end up missing endpoints from intermediate versions — e.g. vteam12 was missing `config_routes.py` from 3.14.0, so `/api/store_objects` and `/api/audio_settings` 404'd and the Store checkbox silently did nothing. Fix is to deploy the complete image/source, not individual files. See deploy notes

## [3.15.4] - 2026-05-20

### Fixed
- **Stale cached JS after upgrades** (`routers/health.py` `/status`): the HTML page was served `no-cache` but the `/static/js/*.js` files (audio.js, app-core.js, etc.) are served by the StaticFiles mount, which browsers cache aggressively. After an MVE upgrade the fresh HTML kept loading the OLD cached JS — most visibly, the **Store checkbox (added 3.14.0) didn't appear** in the Process tab Per-Object Configuration even though the served file contained it, because the browser was running a pre-3.14.0 cached audio.js. Root cause was confirmed on vteam12: served audio.js md5 matched the Store-bearing source, yet the UI rendered only Show/Narrate/Beep
- The `/status` route now reads the HTML and rewrites every local `/static/js|css/*.js|css` include to carry `?v=<app-version>`. Since the version string changes each release, the browser is forced to refetch JS/CSS on every upgrade — no more manual hard-refresh needed after deploying

## [3.15.3] - 2026-05-20

### Added
- **Embedded Detection-Insight charts in the Charts tab** — works on every machine, no Grafana deployment required. Two Chart.js charts above the (still-present) Grafana iframe:
  - **Detections by class** (bar) — which classes/defects are most frequent in the window
  - **Detections over time** (line) — detection-rate trend
  - Window selector: 1h / 6h / 24h / 7d. Refreshes on tab open + manual button
- **New endpoint `GET /api/detection_stats?window=1h|6h|24h|7d`** — aggregates the `inference_results` hypertable into `{by_class, timeline, total, persisted}`. Uses TimescaleDB `time_bucket` + `jsonb_array_elements` to expand the per-row detections JSONB. Returns a well-formed empty payload (never errors) when the DB is unreachable or no class has Store enabled
- Graceful empty-state: if no detections are stored in the window, the panel shows "No stored detections… enable Store on a class" instead of blank charts

### Notes
- The insight charts reflect only classes with **Store=ON** (the 3.14.0 per-class DB opt-in), since they read from `inference_results`. Classes with Store off are detected/narrated but not charted
- The Grafana iframe is retained below the embedded panel for power-user dashboarding on machines where Grafana is deployed (e.g. PVB). On machines without Grafana (e.g. fabriqc-kc) the iframe is empty but the embedded panel still works

## [3.15.2] - 2026-05-19

### Fixed
- **Ejection-procedure rules: target-class dropdown rendered empty on fresh page load** (`static/js/audio.js:628`). The `<select>` for `rule.object` was sourced only from `detectedObjectClasses` — a runtime set populated as the page receives live detection events. If you reloaded the procedures tab before any detection had arrived for a saved rule's class (e.g. `arzi`, `tooli_up`), the dropdown had zero `<option>` elements and the saved class wasn't visible — making the rules look unbound even though the value was still persisted on the server. The render now seeds the option list with `rule.object` itself, so saved rules always show their target class regardless of detection history

## [3.15.1] - 2026-05-19

### Fixed
- **`auto_exposure=true` was bypassed on camera reconnect** (`services/camera.py`): the new 3.15.0 flag was respected only in the initial `CameraBuffer.__init__` open path. The reconnect/re-apply code paths — `_apply_saved_props()` (auto-fired after a USB drop) and `apply_camera_config_from_saved()` (fired by `POST /api/camera/{id}/config` and config restore) — kept unconditionally setting `CAP_PROP_AUTO_EXPOSURE = 1` (manual mode) and writing `CAP_PROP_EXPOSURE` from `_saved_props`. Result: the camera firmware AE would flip back to manual after any restart, USB drop, or runtime config Apply. Caught while end-to-end testing on vteam12
- Both fixed paths now branch on `self.auto_exposure`:
  - If ON: stay in mode 3 (camera-firmware AE), skip the `CAP_PROP_EXPOSURE` write, keep all other props (gain/brightness/contrast/saturation/fps)
  - If OFF (default): pre-3.15.1 behaviour — toggle 3→1, apply manual exposure
- Verified log lines: `auto_exposure=True — manual exposure override skipped` (init) / `auto_exposure=ON (manual exposure skipped)` (re-apply)
- State activation path (`StateManager._apply_state_camera_overrides`) already respected `auto_exposure` correctly in 3.15.0 — proven by the test log: `State 'X' applied to cam 1: exposure=auto-skipped gain=80`

## [3.15.0] - 2026-05-19

### Added
- **Per-state camera exposure/gain override** (state-machine feature). Each `State` now has two optional fields — `exposure: int | null` and `gain: int | null`. When a state is activated, the values are pushed via `cv2.CAP_PROP_EXPOSURE` / `cv2.CAP_PROP_GAIN` to **every camera listed in the state's phases** (not all cameras on the machine). Switching to a state where the field is `null` reverts each affected camera to its own configured value from `service_config["cameras"][cid]`. Use case: a single inspection line that needs different illumination per state (e.g. uplight vs backlight) without touching per-camera config
- **Per-camera Auto-Exposure opt-in** (camera-config field). New boolean `auto_exposure` on each camera. When `true`, MVE no longer forces the camera into manual mode (`CAP_PROP_AUTO_EXPOSURE=1`) at connect time — the camera firmware's own AE algorithm runs and the user-set `exposure` value is ignored. Any per-state exposure override (above) is also skipped for that camera. `gain` still applies. Useful for venues with variable ambient light. Requires a camera Restart after toggling
- **UI**: Create/Edit State form has two new inputs (Gain, Exposure) under Light Status Check; the per-camera configuration grid (Process tab → Cameras) has a new Auto-Exposure checkbox
- API: `POST /api/states` now accepts `exposure` and `gain` (null/empty for no-op); `POST /api/camera/{id}/config` now accepts `auto_exposure: bool`

### Behaviour notes
- Camera-prop writes happen **outside** the `StateManager.state_lock` critical section. V4L2 prop sets can take 50-200ms per camera; serialising state transitions on that latency would block detection. Result: state transitions stay snappy; the override applies in the background
- The override is **soft on errors**: if a camera doesn't accept the value (firmware reject), MVE logs a warning and continues with the next camera. State activation still succeeds
- Auto-Exposure flag is **persisted** in `service_config["cameras"][cid].auto_exposure` so it survives container recreates (in tandem with the [3.14.0 bind-mount fix](#3140---2026-05-18) for the data file)

## [3.14.1] - 2026-05-19

### Fixed
- **Per-class `min_confidence` was cosmetic**: the value stored under `audio_settings[class].min_confidence` was rendered in the Process tab UI and saved via `POST /api/audio_settings`, but no code path used it to suppress detections. As a result, `Show`/`Narrate`/`Beep`/`Store` all fired regardless of how high the per-class threshold was set
- **Server-side fix** (`services/detection.py`): two new helpers — `_get_audio_settings_map()` cached for 5 s (matches the existing `store_objects` pattern), and `_min_conf_for(class, audio_map)` looking up the per-class floor. Both gates now apply to:
  - DB write (Store): only detections with `Store=true` AND `confidence >= min_confidence` are persisted in `inference_results`
  - Audio detection event: only detections above the per-class floor are included in the `add_detection_event("object", ...)` payload — so the browser never sees suppressed detections and can't accidentally narrate/beep them
- **Client-side defense-in-depth** (`static/js/audio.js`): the per-detection forEach in the detection-event handler now skips narrate + beep when `det.confidence * 100 < objectConfidence[class]`. Mirrors the server gate so the UI stays correct even if a stale event slips through

### Why
- Unconfigured classes (no `audio_settings` entry, e.g. fresh installs) get `min_confidence = 0` and behave exactly as before — no surprise regressions
- Server gate is the primary; client gate is a safety net so users who downgrade or hit cache can't bypass the floor

## [3.14.0] - 2026-05-18

### Added
- **`GET /api/audio_settings`** — returns per-class `{show, narrate, beep, min_confidence}` from `service_config["audio_settings"]`
- **`POST /api/audio_settings`** — accepts `{class_name, show?, narrate?, beep?, min_confidence?}` (single-class partial update) or `{audio_settings: {...}}` (bulk replace). Persists server-side
- **UI auto-sync**: the Process tab → Per-Object Configuration checkboxes (Show/Narrate/Beep) and Min-conf input now POST to `/api/audio_settings` on every change, and pull from `/api/audio_settings` on page load. The browser's localStorage cache is preserved as a UI fallback

### Why
- Closes the AI-control gap: previously per-class alarm/display behaviour was browser-side only (localStorage), so an AI agent calling `/api/ai_query` could read system state but couldn't tune which classes narrate, beep, or render. Now everything per-object is reachable through REST
- Image tagging policy: this release is the first to be published as `monitait/mve:3.14.0` explicitly; `:latest` becomes an alias for whichever versioned tag is newest. Compose files should reference the explicit semver

## [3.13.0] - 2026-05-18

### Added
- **Per-object `Store` checkbox** in the Process tab → Per-Object Detection & Alerts. Decides which detection classes are persisted to TimescaleDB (`inference_results`). Default is **OFF** for every class — explicit opt-in only. Useful when the math pipeline emits 300+ channel "detections" per frame and you want to keep only the meaningful ones (defects, key indicators) in the DB
- **`GET /api/store_objects`** — returns `{store_objects: {class_name: bool, ...}}` from `service_config["store_objects"]`
- **`POST /api/store_objects`** — accepts either `{class_name, store}` (single-class toggle) or `{store_objects: {...}}` (bulk replace). Persists to `.env.prepared_query_data`

### Changed
- **`services/detection.py`** now filters detections by per-class `store_objects` flag before calling `write_inference_to_db`. Caches the store map for 5s to avoid disk thrash. If no detection in a frame is marked store=true, the DB write is skipped entirely (zero rows written)

## [3.10.4] - 2026-02-22

### Added
- **Ejector delay parameter**: New `EJECTOR_DELAY` setting (seconds) — time-based delay after encoder target is reached before sending the ejector command. Configurable via UI, API, and data file. Available in all 7 languages

## [3.10.3] - 2026-02-22

### Fixed
- **Camera config lost on reconnect**: Camera properties (exposure, gain, brightness, contrast, saturation) are now stored on the CameraBuffer and automatically re-applied when a camera disconnects and reconnects
- **Camera config not persisted to disk**: Changing camera settings via API now auto-saves to the data file, so settings survive container restarts
- **Timeline prev/next buttons not working**: Pagination buttons now use HTTP fallback when auto-update is paused, fixing the issue where WebSocket handler dropped incoming images during manual navigation
- **Ejector queue monitoring**: Added detailed `[EJ_QUEUE]` and `[EJ_FIRE]` logging to trace ejection queue lifecycle and serial command dispatch

## [3.10.2] - 2026-02-22

### Fixed
- **Ejector dashboard icon always red**: SSE stream was missing `ejector_enabled` field, causing dashboard to always show ejector as disabled. Now reads runtime value via config module reference
- **SSE ejector_offset stale value**: Both SSE and REST status endpoints now read ejector config from live module reference instead of import-time snapshot
- **Click-to-view wrong image**: Timeline popup now uses `d_path` for stable frame lookup instead of column index, which could shift as new frames arrived
- **Ejection evaluation filtered enabled procs**: Only enabled procedures are now passed to the evaluation function
- **Serial send logging**: `_send_message` now logs warnings when serial is unavailable, aiding hardware debugging

## [3.10.1] - 2026-02-22

### Fixed
- **Bounding box scaling**: Timeline bboxes now correctly scale from original image resolution to thumbnail dimensions — previously coordinates were drawn unscaled, causing boxes to appear out of bounds or mispositioned
- **Full-res popup scaling**: Click-to-zoom popup correctly uses original coordinates for disk images and scaled coordinates for thumbnail fallback
- **Frame metadata**: Original image dimensions (`orig_h`, `orig_w`) now stored per-frame for accurate bbox scaling

## [3.10.0] - 2026-02-20

### Fixed
- **Negative bbox area guard**: Area conditions now return False for malformed bounding boxes (negative width/height)
- **LAB color validation**: Color ΔE condition validates L\*a\*b\* array integrity before comparison
- **Empty class name filtering**: `update_color_references()` skips detections with empty class names
- **Procedure UI re-render**: Toggling "Enabled" or changing cameras now immediately updates the UI
- **New rule defaults**: Adding a rule now initializes all fields (area, max_delta_e, reference_mode) to prevent undefined values
- **States API docstring**: Fixed outdated `"enabled": true` → `"light_status_check": false`

### Changed
- **Tooltips added**: All ejection procedure UI elements now have descriptive hover tooltips
- **Documentation overhaul**: Updated README (v3.10.0, ejection procedures, new API endpoints, 7 languages), CHANGELOG (fixed version ordering, pre-release labels), USER_MANUAL (v3.10.0, correct tabs, ejection procedures section, multi-camera FAQ, pipelines FAQ)

## [3.9.0] - 2026-02-19

### Added
- **Area/Size Conditions**: New ejection conditions `Area >`, `Area <`, `Area =` for bbox size-based ejection (threshold in pixels)
- **Per-Camera Procedure Filtering**: Each procedure can now specify which cameras it applies to (e.g., "1,2") or leave empty for all cameras
- **Camera ID on Detections**: Each detection now carries `_cam` field identifying which camera captured it

### Changed
- **Conditional LAB Extraction**: L*a*b* color extraction only runs when an active `color_delta` procedure exists, saving CPU when not needed

## [3.8.0] - 2026-02-19

### Added
- **Color Delta E Condition**: New ejection procedure condition `Color ΔE >` compares detected object color (CIE L*a*b*) against a reference. Three reference modes: vs Previous, vs Running Average (last 20), vs Fixed (user-captured)
- **Color Reference API**: `POST/GET /api/color-reference/{class_name}` for setting and querying fixed color references
- **Per-State Light Status Check**: Light status check (closed-loop serial verification) is now configured per camera state instead of a global toggle

### Changed
- **Ejection Conditions Simplified**: Removed `Present`/`Not Present` conditions — replaced by `Count > 0` and `Count = 0`. Count-based conditions: Count =, Count >, Count <
- **Class Count Check Merged**: Standalone class count check (enable/disable, classes, confidence) removed from Process tab — functionality merged into procedure count conditions
- **State `enabled` Field Removed**: Replaced with `light_status_check` boolean on each State

## [3.7.0] - 2026-02-19

### Added
- **Process Tab**: New dedicated tab consolidating all detection, ejection, and image processing configuration from Hardware, Cameras, and Advanced tabs into one place
- **Dynamic GPU Auto-Tuning**: `start.py` now detects GPU VRAM via `nvidia-smi` and computes optimal YOLO replicas (1 per GPU) and workers (80% VRAM / 500MB per worker)
- **Autoscaler Scale-Down**: Workers now scale down after 3 consecutive idle checks (~90s), freeing CPU/RAM/disk resources when demand drops
- **Cold Queue Stale Frame Flushing**: Stale frames (>10 min old) are discarded at startup and during processing instead of wasting GPU on outdated images
- **HTTP Connection Pooling**: YOLO inference requests now use `requests.Session` with pooled connections instead of creating new TCP connections per call
- **Process Tab Translations**: Added i18n translations for Process tab in all 7 languages (EN, FA, AR, DE, TR, JA, ES)

### Changed
- **ColdDiskQueue Rewrite**: Replaced O(n) `glob + sort` per get() with O(1) in-memory deque index. With 121k queued frames, this fixed inference throughput from 0.7 FPS to 93 FPS
- **Autoscaler Queue Sensitivity**: Now considers cold queue depth (>1000=CRITICAL, >100=WARNING) in addition to hot queue depth
- **Max Inference Workers**: Capped at `min(cpu*2, 32)` instead of unbounded `cpu*4`
- **FPS Measurement Buffer**: Increased from 200 to 2000 entries, removing the false 20.0 FPS ceiling
- **Docker Defaults**: Default YOLO replicas lowered to 1, workers to 2 (start.py auto-overrides based on hardware)

### Fixed
- Inference bottleneck: 96 threads all globbing 121k files simultaneously caused filesystem thrashing
- Race condition in ColdDiskQueue where multiple threads could claim the same file
- Cold queue growing unbounded because autoscaler only checked hot queue depth
- Model form layout in Inference tab: all settings on one row, YOLO weight dropdown properly sized

## [1.1.0] - 2026-01-06

### Added
- **Comprehensive Documentation**:
  - 📖 **USER_MANUAL.md**: 72-page complete user guide covering all features
  - 📊 **AUDIT_REPORT.md**: System audit with optimization recommendations
  - Installation & quick start guides
  - Camera setup (USB, IP, multi-camera)
  - YOLO AI detection configuration
  - Audio alerts system documentation
  - Image processing & capture modes
  - Timeline & review features
  - Database integration (TimescaleDB + Grafana)
  - Troubleshooting guide with solutions
  - Best practices & FAQ
  - API reference & Docker commands

- **Real-time Audio Alerts System**:
  - SSE-based detection events via Redis for cross-process communication
  - Voice narration of detected object names (Text-to-Speech)
  - Per-object audio control (enable/disable individual objects)
  - 4 customizable beep sounds per object (sine, square, sawtooth, triangle)
  - Non-blocking audio processing
  - Volume control and test functionality
  - Clean UI in Advanced tab

- **Advanced UI Organization**:
  - Consolidated Audio Alerts configuration in Advanced tab
  - Database Storage configuration moved to Advanced tab
  - Data File Editor relocated to bottom of Advanced tab
  - Improved responsive layout with proper sizing
  - Fixed dropdown/button width issues

### Changed
- **Configuration Management**:
  - All settings now configurable via web interface at http://localhost:5050/status
  - Settings persisted to .env.prepared_query_data
  - Export/Import service configuration functionality

- **Detection Event Architecture**:
  - Migrated from in-memory deque to Redis for cross-process event sharing
  - ProcessPoolExecutor workers can now communicate events to main FastAPI process
  - SSE stream delivers real-time detection events
  - Polling endpoint available as fallback

### Fixed
- Audio playing for disabled objects (now respects per-object enable/disable)
- Detection events not reaching frontend (Redis cross-process solution)
- SSE stream not showing detection events (proper Redis integration)
- Dropdown too narrow and button too wide in Database Storage section

### System Audit Findings
- ✅ All 6 Docker services properly configured and necessary
- ✅ Configuration files properly synchronized
- ✅ Clean codebase with no duplicate code
- ⚠️ Identified 11 unused Python dependencies (~50MB bloat) - removal recommended
- ⚠️ Timeline buffer capped at 100 frames when quality=100% - documented for future fix

### Performance
- Low CPU usage during idle operation
- Efficient Redis LRU eviction for cache management
- Optimized disk cleanup service with automated space management
- SSE provides real-time event delivery with minimal overhead

## [1.0.0] - 2026-01-05

### Added
- **Unlimited Camera Support**: Updated from fixed 4-camera limit to dynamic camera detection
  - Merged improvements from `zarrin-error-fix` branch
  - Auto-detects all available cameras on startup
  - Backward compatible with legacy 4-camera configuration
- **IP Camera Support**: Native support for RTSP and HTTP/MJPEG cameras
  - Configure via `IP_CAMERAS` environment variable (comma-separated URLs)
  - Mix USB and IP cameras seamlessly
  - Automatic backend selection (V4L2 for USB, FFMPEG for IP)
  - Support for authentication (username/password in URL)
  - Comprehensive setup guide in [docs/IP_CAMERA_SETUP.md](docs/IP_CAMERA_SETUP.md)
- **Lightweight MJPEG Streaming**: Integrated live camera feed into status page
  - `/video_feed` endpoint with 10 FPS, JPEG quality 40 for minimal bandwidth
  - Embedded directly in status monitoring interface at port 5050
  - No additional services or dependencies required
  - Uses existing FastAPI StreamingResponse capability
- **Full Deployment Mode**: Created `docker-compose.full.yml` with all optional services
  - Shipment fulfillment web interface (Django)
  - PostgreSQL database
  - Celery workers (high/low priority queues)
  - Image gallery (Pigallery2)
  - Legacy streaming service

### Changed
- **Container Naming**: Renamed all containers from `monitaqc_*` to `monitait_*` prefix
  - `monitait_vision_engine` - Core CV processing
  - `monitait_redis` - Message broker
  - `monitait_yolo` - AI inference
  - `monitait_cleanup` - Disk management
- **Branding Update**: Changed from VirasAd/Monitait to Smart Falcon AI
  - Support: admin@smartfalcon-ai.com
  - Removed client-specific references
- **Base Image**: Updated Dockerfile from Debian Buster to Bookworm (Python 3.10-slim)

### Planned
- Merge with fabric inspection capabilities (from FabriQC)
- Merge with signal counting capabilities (from PartQC Signal Counter)
- Unified admin interface for all QC modes
- Multi-application mode support
- Enhanced API with OpenAPI documentation
- Advanced analytics and reporting
- Cloud synchronization improvements

## Pre-release (2025-12-29)

### 0.3.0 — Project Reorganization
- Moved core service files to dedicated `vision_engine/` directory
- Service renamed: `monitaqc_counter` → `monitaqc_vision`
- Updated docker-compose.yml build context to `./vision_engine`

### 0.2.0 — Lightweight Architecture
- Reduced from 11+ containers to 4 core services
- Optimized Redis with 256MB memory limit and LRU eviction
- Changed Redis from `redis:latest` to `redis:7-alpine`
- Created `docker-compose.full.yml` with all optional services

### 0.1.0 — Initial Fork
- Fork from PartQC Box Counter
- MonitaQC branding and naming
- Standardized docker-compose container names
