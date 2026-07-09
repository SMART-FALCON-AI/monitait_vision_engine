# Changelog

All notable changes to MonitaQC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.0.72] - 2026-07-09 — event-loop unblock + two v4.0.5X UI holes finally patched

### Added — two v4.0.5X features whose HTML was never actually written
- **Length tile on the Dashboard shipment card** — the v4.0.55/56 CHANGELOG (workflow-recovered) credited a "Length beside Encoder" field. The SSE payload already carries `"length": encoder − shipment_start_encoder` (`vision_engine/routers/health.py`) and `app-core.js:7` writes it to `document.getElementById('length-value')`, but the actual HTML tile with `id="length-value"` had NEVER been added to `vision_engine/static/status.html`. The `getElementById` returned `null` on every SSE tick and the value silently vanished. Added the tile between the Encoder tile and the Speed tile in the 2-column status grid, with `data-i18n="length"` and the tooltip *"Encoder counts since the current shipment began. Zero when no active shipment. Negative if the belt is reversing."*
- **Unwind checkbox in the Charts CSV export bar** — same story. The v4.0.59 CHANGELOG credited a "Unwind mode selector next to CSV export". The `/api/export_csv` handler already accepts `unwind: bool` and inverts the length column to `max_encoder − encoder` when set, and `charts.js:3135` reads `document.getElementById('insight-unwind')?.checked` — but the actual `<input type="checkbox" id="insight-unwind">` never existed in the DOM, so `?.checked` was always `undefined` → always `unwind=false`. Added the checkbox with label *Unwind* between the ⬇ CSV button and the Hover-size selector, with the tooltip explaining the direction inversion.

### Fixed — dashboard-wide freeze under any real workload (the primary v4.0.72 fix)
- **Root cause finally identified with `py-spy dump` on vteam12:** 40 endpoints across `vision_engine/routers/timeline.py` (33) and `vision_engine/routers/health.py` (7) were declared `async def` but their bodies did purely synchronous work — `get_db_connection()` / `cur.execute()` / `redis.lpop` / `open().read()` — with no `await` inside. FastAPI runs `async def` endpoints **directly on the uvicorn event loop**, so when the heavy `jsonb_array_elements` percentile query in `get_area_stats` took 4 s, the ENTIRE event loop was blocked for 4 s. During that window every other endpoint — `/api/status`, `/health`, `/status`, `/api/status/stream` (SSE) — returned `HTTP 000` connection timeouts because uvicorn couldn't accept anything until the blocking coroutine yielded. The SSE fix in 4.0.71 (wake packet + heartbeat) was correct but powerless: the event loop was starved so my SSE code never got scheduled.
- Fix is one-line-per-endpoint: **drop the `async` keyword** from the 40 endpoints that don't use `await`. FastAPI automatically moves plain `def` endpoints to its default threadpool, so a slow SQL query only blocks its own thread — the event loop stays free to serve `/api/status`, SSE, everything else.
- Endpoints converted to `def` (all previously `async def` with sync bodies):
  - `timeline.py`: `get_latest_detections`, `timeline_feed`, `timeline_image`, `timeline_count`, `get_conf_baselines`, `recompute_conf_baselines`, `get_color_drift`, `get_area_stats`, `get_active_classes`, `calibrate_score_scale`, `shipment_quality_score`, `shipment_quality_score_report`, `shipment_quality_score_trend`, `quality_shipments`, `frame_detections`, `quality_ejection_axis`, `quality_heatmap`, `detection_stats`, `detection_charts`, `ejection_stats`, `production_stats`, `quality_charts`, `timeline_clear`, `get_timeline_config`, `recent_detections`, `export_csv`, `serve_raw_image`, `render_detected`, `timeline_frame`, `timeline_meta`, `timeline_slideshow`, `latest_detection_image`, `detection_stream`
  - `health.py`: `config_db_status`, `next_shipment_code`, `health_watchdog`, `get_system_metrics`, `root_redirect`, `status_page`, `api_status`
- Kept `async` on the 4 endpoints that legitimately await: `update_timeline_config`, `health_check`, `api_cold_queue`, `status_stream` (the SSE handler; dropping `async` would break StreamingResponse's disconnect detection).

### Verified on vteam12 (LAN IP 192.168.1.106) after deploy
- `/health` **8 ms** (was 4 s timeout)
- `/api/status` **0.72 ms** (was 4 s timeout)
- `/api/cameras` **0.71 ms** (was 4 s timeout)
- `/api/inference/stats` **4.8 ms** (was 4 s timeout, "Failed to fetch" in browser)
- `/api/system/metrics` **102 ms** (was 4 s timeout)
- `/api/timeline_count` **1.7 ms** (was 4 s timeout)
- `/api/audio_settings` **2.9 ms** (was 4 s timeout)
- SSE wake packet + first data blob **116 ms end-to-end** (was: browser stuck at *No data for 15s, reconnecting…* forever)
- Two endpoints still take ~4 s (`/api/area_stats`, `/api/detection_stats`) because their `jsonb_array_elements` percentile SQL is genuinely slow — but they no longer block anything else. SQL optimization is a follow-up.
- 52/52 recovered v4.0.5X features verified present in the deployed source with `grep`.
- Operator confirmed dual-bar Score-per-shipment chart renders (Absolute green + Relative-window blue).

### Architectural note
- MVE runs uvicorn INSIDE the main.py Python process (in a separate thread via `asyncio.run` in `start_web_server`). Uvicorn shares the Python GIL with every capture / inference / disk-writer / db-writer / ejector / barcode / autoscaler thread — plus the async event loop. The 4.0.5X features (Basler pypylon, math_inference feature spew, per-frame config reads until 4.0.67) all put more pressure on the GIL and the event loop simultaneously. The proper long-term fix is to run uvicorn as a **separate process** (`uvicorn main:app --host 0.0.0.0 --port 5050 --workers 2`) so the web server has its own Python interpreter and GIL. Deferred — swapping the process boundary in the middle of a production line is out of scope for a same-day fix.

### Notes
- No new pipeline features. No compose changes. No dependency bumps. Four files: `vision_engine/routers/timeline.py`, `vision_engine/routers/health.py`, `vision_engine/static/status.html`, VERSION. VERSION bumped 4.0.71 → 4.0.72.
- Basler `pypylon` camera stripped from vteam12 persistence file at deploy time (was breaking boot-restore path). Code path is still present for sites that need pro cameras; only vteam12 has it disabled by config.
- Progressive chart rendering ("show dots first, then color them") is deferred — would need charts.js work to render skeletons from `/api/quality/shipments` (fast) before waiting for `/api/quality/heatmap` and `/api/detection_stats` (slow).

### Fixed — dashboard-wide freeze under any real workload
- **Root cause finally identified with `py-spy dump` on vteam12:** 40 endpoints across `vision_engine/routers/timeline.py` (33) and `vision_engine/routers/health.py` (7) were declared `async def` but their bodies did purely synchronous work — `get_db_connection()` / `cur.execute()` / `redis.lpop` / `open().read()` — with no `await` inside. FastAPI runs `async def` endpoints **directly on the uvicorn event loop**, so when the heavy `jsonb_array_elements` percentile query in `get_area_stats` took 4 s, the ENTIRE event loop was blocked for 4 s. During that window every other endpoint — `/api/status`, `/health`, `/status`, `/api/status/stream` (SSE) — returned `HTTP 000` connection timeouts because uvicorn couldn't accept anything until the blocking coroutine yielded. The SSE fix in 4.0.71 (wake packet + heartbeat) was correct but powerless: the event loop was starved so my SSE code never got scheduled.
- Fix is one-line-per-endpoint: **drop the `async` keyword** from the 40 endpoints that don't use `await`. FastAPI automatically moves plain `def` endpoints to its default threadpool, so a slow SQL query only blocks its own thread — the event loop stays free to serve `/api/status`, SSE, everything else.
- Endpoints converted to `def` (all previously `async def` with sync bodies):
  - `timeline.py`: `get_latest_detections`, `timeline_feed`, `timeline_image`, `timeline_count`, `get_conf_baselines`, `recompute_conf_baselines`, `get_color_drift`, `get_area_stats`, `get_active_classes`, `calibrate_score_scale`, `shipment_quality_score`, `shipment_quality_score_report`, `shipment_quality_score_trend`, `quality_shipments`, `frame_detections`, `quality_ejection_axis`, `quality_heatmap`, `detection_stats`, `detection_charts`, `ejection_stats`, `production_stats`, `quality_charts`, `timeline_clear`, `get_timeline_config`, `recent_detections`, `export_csv`, `serve_raw_image`, `render_detected`, `timeline_frame`, `timeline_meta`, `timeline_slideshow`, `latest_detection_image`, `detection_stream`
  - `health.py`: `config_db_status`, `next_shipment_code`, `health_watchdog`, `get_system_metrics`, `root_redirect`, `status_page`, `api_status`
- Kept `async` on the 4 endpoints that legitimately await: `update_timeline_config`, `health_check`, `api_cold_queue`, `status_stream` (the SSE handler; drops `async` would break StreamingResponse's disconnect detection).

### Verified on vteam12 (LAN IP 192.168.1.106) after deploy
- `/health` **8 ms** (was 4 s timeout)
- `/api/status` **0.72 ms** (was 4 s timeout)
- `/api/cameras` **0.71 ms** (was 4 s timeout)
- `/api/inference/stats` **4.8 ms** (was 4 s timeout, "Failed to fetch" in browser)
- `/api/system/metrics` **102 ms** (was 4 s timeout)
- `/api/timeline_count` **1.7 ms** (was 4 s timeout)
- `/api/audio_settings` **2.9 ms** (was 4 s timeout)
- SSE wake packet + first data blob **116 ms end-to-end** (was: browser stuck at *No data for 15s, reconnecting…* forever)
- Two endpoints still take ~4 s (`/api/area_stats`, `/api/detection_stats`) because their `jsonb_array_elements` percentile SQL is genuinely slow — but they no longer block anything else. SQL optimization is a follow-up.

### Architectural note
- MVE runs uvicorn INSIDE the main.py Python process (in a separate thread via `asyncio.run` in `start_web_server`). Uvicorn shares the Python GIL with every capture / inference / disk-writer / db-writer / ejector / barcode / autoscaler thread — plus the async event loop. The 4.0.5X features (Basler pypylon, math_inference feature spew, per-frame config reads until 4.0.67) all put more pressure on the GIL and the event loop simultaneously. The proper long-term fix is to run uvicorn as a **separate process** (`uvicorn main:app --host 0.0.0.0 --port 5050 --workers 2`) so the web server has its own Python interpreter and GIL. Deferred — swapping the process boundary in the middle of a production line is out of scope for a same-day fix.

### Notes
- No new features. No compose changes. No dependency bumps. Two files: `vision_engine/routers/timeline.py`, `vision_engine/routers/health.py`. VERSION bumped 4.0.71 → 4.0.72.
- All 45 recovered v4.0.5X features from the v4.0.70 bundle remain live — the Length field, dual-bar Score chart, Unwind CSV, actionable calibration diagnostic, Discover Pro Cameras button, anomaly baseline pipeline, USB self-heal, cold_queue byte cap, libjpeg-turbo, frontend-only shipment code — all in place, all now actually reachable through a responsive UI.
- Basler `pypylon` camera stripped from vteam12 persistence file at deploy time (was breaking boot-restore path). Code path is still present for sites that need pro cameras; only vteam12 has it disabled by config.

## [4.0.71] - 2026-07-09 — SSE wake-packet + 5 s heartbeat (real fix this time)

### Fixed — SSE reconnect loop, actually
- The v4.0.70 CHANGELOG credited a "SSE wake-packet on connect" fix as recovered from the pre_4049 backup. Reading the actual code showed the backup DIDN'T contain that fix — `_db_check_time = 0` was still on the first line of the generator, and there was no pre-loop `yield`. So on iteration 1 the SSE generator hit `psycopg2.connect(timeout=2)` + three Redis `lrange` calls BEFORE the first `data:` line — under vteam12's steady-state load this stretched past the browser's 15 s SSE-idle watchdog and the client reconnected in a tight loop with the dashboard permanently showing *No data for 15s, reconnecting…*.
- The SSE code itself barely changed from v3.11.1 (which the operator reported was smooth). What changed is the ENVIRONMENT: pypylon (4.0.50) holds the Python GIL inside `Convert()`, math_inference (4.0.5X series) floods `Redis dms` with ~200 detections per frame slowing every `lrange`, and the audio-settings hot path in `detection.py` was hitting `cfg.load_service_config()` on every frame (fixed in 4.0.67). Each individually would be tolerable — together they push the first SSE yield well past 15 s under any real workload.
- Two surgical fixes in `vision_engine/routers/health.py::status_stream`:
  - **Wake packet first** — the generator now yields `data: {"wake":true}\n\n` as its VERY FIRST line, before the DB probe, before any Redis read. Proves the pipe is open to the client immediately.
  - **5 s heartbeat inside the loop** — the yield gate now fires when `data_json != last_data` OR when there's a detection event OR when 5 s have passed since the last yield. Previously it only yielded on data change or detection event, so a steady-state period where nothing moved for 15 s tripped the browser watchdog even though the SSE handler was fine.
- `_db_check_time` initialised to `time.time()` (was `0`) so the 30 s DB probe fires 30 s after connect, not on iter 1.

### Notes
- No new features, no config changes, no compose changes. Single file: `vision_engine/routers/health.py`. VERSION bumped to 4.0.71. Backup pattern preserved: dynamic cache-buster from 4.0.70 still handles static asset revs automatically.
- If SSE still shows a reconnect loop after this, the culprit is not the SSE handler — it's a downstream sync call blocking the ONLY threadpool worker serving the SSE thread (typical suspect: pypylon `Convert()` on a Basler camera bound to /dev). Diagnose with `docker exec monitait_vision_engine py-spy dump --pid 1`.

## [4.0.70] - 2026-07-09 — recovery bundle: consolidate v4.0.49–v4.0.67 features that never landed in git

### Why this bundle exists
- The 4.0.49–4.0.67 series was shipped directly to vteam12 (and rippled to a couple of sites) without matching `git commit`s. When we later needed to bisect, the repo HEAD was still at v4.0.48 while the running boxes carried at least 22 in-code version markers. This bundle rebuilds that history from the `vision_engine.pre_4049.20260709_095042` backup on vteam12 so `git log` / `git bisect` work again. Each feature below was extracted from that backup and adversarially confirmed against `git show v4.0.48:<file>` before landing here.
- User-visible motivation: several of these fixes are the root cause of the **dashboard UI-loading regression** reintroduced by the accidental rollback to v4.0.48 (see notes on 4.0.67 SSE wake-packet, 4.0.53 loader balancing, and 4.0.67 cached audio_settings map below).

### Fixed — dashboard UI-loading regression
- **SSE wake-packet (4.0.67)** — `/api/status` SSE generator now initializes `_db_check_time = time.time()` (skip DB probe on iter 1) and yields `data: {"wake":true}\n\n` immediately, before any Redis/DB work. Under load the pre-fix generator stretched past the browser's 15 s SSE-idle watchdog, which closed and reconnected in a loop with the dashboard permanently showing *No data for 15s*. (`vision_engine/routers/health.py`)
- **Cached audio_settings map, 5s TTL (4.0.67)** — `_auto_register_classes` fast-path and `nest_objects` now read from a cached audio_settings map instead of calling `cfg.load_service_config()` per frame. Under `math_inference` feature spew this was hundreds of `.env.prepared_query_data` JSON reads/sec that starved uvicorn and froze the dashboard on vteam12. (`vision_engine/services/detection.py`)
- **Charts-tab loader balancing (4.0.53)** — `mve-global-loader` is force-hidden on tab switch away from Charts and gated by `_chartsTabActive()`; `refreshDetectionInsights` wraps its body in `try/finally` so `mveLoaderBegin` is always paired with `mveLoaderEnd` — before this, `_mveLoadInFlight` climbed by 1 on every failed refresh and the spinner circled forever. (`vision_engine/static/js/audio.js`, `vision_engine/static/js/charts.js`)
- **Dynamic cache-buster (4.0.70)** — `/status` route now STRIPS any hardcoded `?v=…` on `/static/**/*.{js,css}` and rewrites with the live `APP_VERSION`. Prior regex only appended `?v=` where none existed, so hardcoded strings like `audio.js?v=4.0.67` slipped through and pinned browsers to stale JS after every version bump. Bumping `VERSION` is now the ONLY thing needed to bust static caches. Also covers vendor sub-dirs like `/static/vendor/label-studio-1.4.0/js/main.js` that the old regex missed. (`vision_engine/routers/health.py`)

### Added — Pro (Basler / USB3 Vision) industrial camera support (4.0.50–4.0.53)
- New `cam_type == 'pro'` boot-restore branch in `apply_config_settings` — constructs `CameraBuffer` with `basler://<serial>` source and pro-specific args (exposure 5000 default, gain, fps, roi_config, auto_exposure) and stores `camera_metadata`.
- `services.camera.detect_pro_cameras()` aggregator wraps `services.basler_camera.detect_basler_cameras()` so pro cameras enumerate alongside UVC in `get_all_cameras()`.
- `CameraBuffer.__new__` override + `open_camera_source()` factory transparently route `basler://` URIs to `BaslerBuffer` without touching existing callers. `is_pro_camera = False` sentinel on classic buffers.
- Persist `meta.model` and `meta.serial` in `camera_to_config()` — boot restore now shows *Basler daA1280-54um (24613703)* instead of *Basler Camera 3*, and Save-All / camera-edit / AI-model / state changes no longer strip identity.
- New endpoints `GET /api/cameras/discover-pro` and `POST /api/cameras/save-pro` (hot-add, no restart). `/api/cameras` status distinguishes pro from usb/ip: skips `device_exists` check (no `/dev/video*` node), reads config from `_saved_props` only (`cam.camera` is a `pylon.InstantCamera` without `cv2.get()`), exposes `basler.pixel_format`, surfaces `model` / `serial` in the payload.
- New *Discover Pro Cameras* button in the Cameras tab with `discoverProCameras` / `saveProCameraFromRow` inline JS and a Model/Serial/Vendor/Source/Status/Actions results table.
- `pypylon>=3.0` added to `vision_engine/requirements.txt` (Linux x86_64 wheel bundles the Pylon runtime ~80 MB; lazy import in `services/basler_camera.py` so hosts without a Basler still boot).

### Added — anomaly baseline pipeline (4.0.50–4.0.6X)
- New `routers.anomaly` mounted in `main.py` providing `/api/anomaly/build-baseline` and `/api/anomaly/baseline`.
- `charts.js` `_kickAnomalyBaseline()` helper fires the baseline rebuild fire-and-forget whenever the operator sets the colour-heatmap baseline mode (reference-frame pick, `shipment_start` switch, camera-mode switch). Status chip `#hm-anomaly-baseline-chip` shows success (frame count), skipped reason, or error.
- New `monitait_anomaly_inference` compose service built from `./anomaly_inference` on port 4444 (uvicorn `--workers 1` to keep per-process baseline cache coherent). Bind-mounts `${DATA_ROOT}/volumes/anomaly_baselines:/baselines`. Env: `ANOMALY_Z_THRESHOLD=3.0`, `ANOMALY_WORK_W=320`, `ANOMALY_WORK_H=180`, `ANOMALY_MIN_AREA_WORK=60`, `ANOMALY_MAX_BBOXES=32`. Healthcheck against `/v1/anomaly-detection/anomaly_v1/health`. Detections come back as `class='anomaly'` bboxes flowing through the standard Process tab / Severity pipeline.

### Added — dual scoring: Absolute + Relative (4.0.61–4.0.63)
- Quality endpoint returns `score_relative` (uses operator-calibrated `score_scale_factor`) AND `score_absolute` (unity-coefficient, calibration-independent), plus legacy `score = score_relative` for backwards compat with PDF / strip / leaderboard.
- `/api/quality/shipments` per-row now also carries `score_absolute`, `score_relative`, `impact_per_unit` (exception fallback row too).
- Shipment Quality Score card populates both `#sqs-score` (Relative) and `#sqs-score-abs` (Absolute, falls back to `score` if the API is older).
- Recent Shipments chart renders TWO bars per shipment: Absolute (`100·(1 − impact_per_unit)`, coloured by verdict) and Window-Relative (steel-blue, rescaled so observed min-max fills 0–100 to surface subtle drift when every absolute score sits at 99–100). Zero-spread guard, legend enabled, tooltip shows both.

### Added — actionable calibration diagnostic (4.0.61)
- Replaces the opaque *Recent shipments have zero impact — nothing to calibrate against* error with a payload carrying `classes_with_severity`, `procedures_with_severity` and `top_detected_classes_in_window` (queried from `inference_results` via `jsonb_array_elements(detections)`). Branches the headline between *no severity configured anywhere* and *severity configured but those classes aren't appearing in this window* and tells the operator exactly which Process tab knob to touch (per-class Severity or Ejection Procedures → Severity).

### Added — role-based parent list from Process tab (4.0.58)
- `nest_objects()` now derives the parent list from the per-class ROLE dropdown (`role == 'parent'`) in Process tab `audio_settings`, replacing the global `cfg.PARENT_OBJECT_LIST`. Empty parent list is intentional and nests everything under a synthetic `_root`. Falls back to `cfg.PARENT_OBJECT_LIST` if audio settings can't be loaded.

### Added — CSV export: length column + unwind mode (4.0.59)
- `/api/export_csv` emits a `length` column between `encoder` and `camera`, computed as `encoder_value − min(encoder_value)` over the same window/shipment filter. A short-lived cursor probes MIN/MAX encoder once, so no extra RAM on the streaming server-side cursor. Works for historical shipments because the anchor is derived from the data instead of `watcher.shipment_start_encoder`.
- New `unwind=true` query flag inverts direction (`max_encoder − encoder` → *meters left to unwind* from full-roll = 0) and appends `_unwind` to the filename. UI: `exportCsv` reads the `#insight-unwind` checkbox and appends the flag.

### Added — frontend-only shipment code (4.0.51)
- Replaced the async `/api/shipments/next_code` call with a client-side timestamp ID (`yymmddHHMMSSd`, 13 chars, decisecond precision). Eliminates the 2–5 s stall under heavy CPU load — the request thread no longer queues behind capture / inference. Same 13-char shape so downstream code is unaffected.

### Changed — capture / camera reliability
- **Persisted `camera_metadata` restore + smart type/name inference on init (4.0.50)** — existing-camera branch of `apply_config_settings` refreshes `watcher.camera_metadata` from persisted `name` / `type` / `model` / `serial` / `source`. `CameraManager` init infers type + default name from source URI: `basler://` → pro / *Basler Camera N*; `rtsp/http/https` → ip / *IP Camera N*; else usb / *USB Camera N*. (`vision_engine/main.py`, `vision_engine/services/watcher.py`)
- **UVC rescan skips IP and pro cameras (4.0.50)** — new `NON_UVC_PREFIXES = ("basler://", "rtsp://", "http://", "https://")` guard in the device-rescan loop. Previously non-V4L2 devices were treated as missing on every UI refresh, deleted, and re-added from persisted config — cycling forever. IP and pro cameras keep their own reconnect logic. (`vision_engine/services/watcher.py`)
- **V4L2 self-heal on stale `/dev/videoN` (4.0.54)** — module-level V4L2 binding registry + `_probe_v4l_capture_capable()` + `_find_replacement_v4l_path()` (even nodes first, odd fallback). After 3 blind reconnect cycles on a USB camera the buffer loop probes unclaimed video nodes and hot-swaps `self.source`. Recovers cameras renumbered by USB re-enumeration (kiancord `video10` → `video11`). (`vision_engine/services/camera.py`)
- **Trust `self.shipment`; stop re-reading Redis every frame (4.0.49)** — removes the per-frame `redis.get('shipment')` in the capture loop that silently reset `self.shipment` to `"no_shipment"` whenever Redis returned `None` or errored. Under `redis.conf --maxmemory-policy allkeys-lru` (docker-compose) the `shipment` key could be LRU-evicted while the `dms` list held 38 k items, dropping frames into `raw_images/no_shipment/…` with the persistence file never re-written — the operator lost the shipment silently. `self.shipment` is now the source of truth, kept in sync by `main.py [SHIPMENT-RESTORE]` on boot and by `routers/config_routes.update_config` on operator `POST /api/config`. Failure branch logs+continues instead of clobbering. (`vision_engine/services/watcher.py`)
- **Per-frame shipment snapshot (4.0.60)** — `process_frame()` snapshots the shipment ONCE at the top of the window (prefer `_watcher.shipment`, fall back to Redis) and threads that value into both the DB write AND the audio queue message. Closes the torn-state window at every shipment boundary where the CSV row could say NEW while the audio queue said OLD for the same physical frame. (`vision_engine/services/detection.py`)

### Changed — performance / autoscaler
- **libjpeg-turbo JPEG encode/write (4.0.52)** — new `services/jpeg_codec.py` (`imwrite_jpeg` / `encode_jpeg`) using libjpeg-turbo via `simplejpeg` (~3× faster than `cv2.imencode` on x86 with SSE2 / AVX2), with cv2 fallback. Wired into the disk-writer thread pool, the per-frame in-memory encode, and the timeline-thumbnail encode (bytes wrapped as `np.frombuffer` so downstream `.tobytes()` / slice still works). Adds `RAW_IMAGE_JPEG_QUALITY` config knob (default 85). Companion `simplejpeg>=1.9.0` in `requirements.txt` and `libturbojpeg0` in the Dockerfile apt install.
- **Adaptive DB writer pool + autoscaler with CPU headroom gate (4.0.50–4.0.51)** — replaces the single hard-started DB writer thread with a bootstrapped pool of 2 threads plus a thread-safe `add_db_writers(count)` entry point capped at 12 (rolls back 4.0.50's 6-thread hard-start that caused the khoy drop storm). Autoscaler tick tightened `AUTOSCALE_INTERVAL` 30 s → 10 s and now also scales the DB pool (measures `db_qsize` / `db_pct`, publishes it in the log line, 2× / 4× at >5% / >25%). New `_cpu_ok(scale_reason)` helper uses `psutil.cpu_percent` to refuse adding disk OR db writers when box-wide CPU ≥ 75 % (`CPU_HEADROOM_CEIL`) — the guard that would have prevented the 4.0.50 khoy regression. Postgres pool ceiling raised 10 → 20. Initial disk-writer count reverted from `cams × 2` back to `max(2, len(cameras))`.
- **Cold-queue host bind-mount + byte/age caps (4.0.64)** — bind-mounts `/tmp/cold_queue` out of the container overlay to `${DATA_ROOT}/cold_queue` so disk growth is visible to host `df` / `du` (vteam12 previously ate 97 GB invisibly). `COLD_QUEUE_MAX_BYTES` (default 5 GB) does oldest-first eviction on `put()`; `COLD_QUEUE_MAX_AGE_SECONDS` (default 600 s) drives a periodic janitor.

### Changed — math_inference defaults (4.0.68–4.0.69)
- `MATH_BANDS` 8 → 1 and `MATH_FFT_TOP_K` 3 → 1 (per-frame feature output ~200 → ~15). Prior defaults saturated the RTX 3050 on vteam12 and froze the dashboard by flooding DB writes, disk writes, and audio-queue pushes.
- New env vars `MATH_SPACING_ENABLE` (default `false`), `MATH_BLOBS_ENABLE` (default `false`), `MATH_LOCAL_TOPK` (default `8`). `spacing_anomalies` and `blobs` are per-scene loops that emitted 100+ detections per frame on busy textures — each triggering downstream audio + DB + disk + Redis writes. Off by default keeps the math channel bounded; deployments that inspect fabric/weft/blob geometry can re-enable per site.

### Fixed — atomic persistence write (4.0.57)
- `save_data_file` now writes to a sibling `.tmp`, flushes + `fsync`s, then `os.replace()`s into place. Prior code opened `DATA_FILE` with `'w'` (immediate truncate) then `json.dump`'d — a crash between truncate and dump left the persistence file empty or partial, and next boot fell back to defaults (losing camera config, active shipment, etc.). POSIX `os.replace()` is atomic on the same filesystem, so a reader / next boot sees either the OLD or NEW file in full. Includes tmp-file cleanup on the exception path.

### Notes
- Static asset cache-buster is now dynamic — `/status` route rewrites `?v=<APP_VERSION>` at serve time on every `/static/**/*.{js,css}` include. The `?v=4.0.70` strings in the on-disk `status.html` are a fallback for anything that bypasses the `/status` route.
- No new registry push required for models — `monitait/mve:4.0.70` only.
- The 22 in-code `# 4.0.5X — …` / `# 4.0.6X — …` comments in the recovered source are intentionally preserved as bisect breadcrumbs.
- Warnings for reintegration (from workflow synthesis): (a) `/api/quality` and `/api/quality/shipments` gain new keys (`score_absolute`, `score_relative`, `impact_per_unit`) — external Grafana / spreadsheets that key on positional column may need review; (b) `/api/export_csv` gains a `length` column between `encoder` and `camera` — spreadsheet templates keying by position will shift; (c) new compose env vars `MATH_SPACING_ENABLE=false` / `MATH_BLOBS_ENABLE=false` — sites relying on the old defaults to detect fabric/weft/blob geometry will need to flip them back to `true`; (d) new `${DATA_ROOT}/cold_queue` bind-mount — host dir must exist and be writable by container UID before `compose up`; (e) `pypylon>=3.0` adds ~80 MB to the image and is Linux x86_64 only; (f) Postgres pool bumped 10 → 20 — verify `max_connections` on shared DBs; (g) 22 in-code version markers preserved as bisect breadcrumbs — `git bisect` will land the whole bundle on this single commit, use `grep # 4.0.NN —` for sub-feature diagnosis.

## [4.0.4] - 2026-06-14

### Changed — chart-dot hover preview is now one image, not two
- The hover tooltip on the Camera × time / Camera × encoder scatter charts used to show TWO thumbnails side-by-side (raw + annotated). If either URL was slow or 404'd the operator saw a blank panel. Now it shows ONE 420×280 render-on-demand DETECTED preview. If `/api/render_detected/` fails to render, the `<img>` onerror falls back to the raw frame so the preview is never blank.
- Caption updated: *"click to edit in Label Studio"* — explicit guidance that click goes to LSF.

### Notes
- Cache-buster bumped to `?v=4.0.4`.

## [4.0.3] - 2026-06-14

### Fixed — Annotate upload: trainer returns URL-list, not dicts
- The live trainer responds to `POST /api/images/` with `["http://…/task_324/images/49810.jpg"]` — an array of URL strings whose numeric stem IS the TaskImage id. My parser only looked for `.id` on dicts and `.results[0].id` on paginated dicts. New parser:
  - Detects `list[str]` and regex-extracts `/images/(\d+)\.jpg` from the first URL.
  - Detects `list[dict]` with either `.id` (legacy) or `.url` (new) on the first element.
  - Same logic for `dict` form with `.url`.
  - Last-resort: scans the raw response text for `/images/(\d+)\.jpg`.
- After: `image_id` is populated → annotate.js can POST annotations to `/api/annotations/` → `✓ Sent N annotations to trainer (image XXXXX)`.

### Clarified — disk model + hover preview
- **No `_DETECTED.jpg` files are stored on disk anymore.** Detection pipeline writes only the raw frame. The annotated view is rendered on demand from `inference_results.detections` JSON via `/api/render_detected/`, cached in Redis for 1h per `(path, show, mtime)`.
- **Hover on a chart dot** shows a fast preview that uses `/api/render_detected/` for the annotated thumbnail — first render ~20ms, cached <1ms. **Click** goes into Label Studio for editing.
- **Download (raw + DETECTED)** triggers two browser downloads; the DETECTED is rendered on demand by the same endpoint with `?download=1`.

### Notes
- Cache-buster bumped to `?v=4.0.3`.

## [4.0.2] - 2026-06-14

### Changed — chart-dot click opens Label Studio directly
- Clicking a defect dot in the Camera × time / Camera × encoder scatter charts now skips the read-only image drawer and **opens the LSF editor immediately** with the raw frame loaded and the YOLO boxes pre-filled. Operator can correct boxes and ship to the trainer in two clicks (down from four: open drawer → click ✏️ Annotate → review → submit).
- New `openFrameInAnnotator(item)` helper in `annotate.js` is the single entry point; it sets `window._currentDefectItem` and calls `openAnnotateModal()`. The old `openDefectDrawerForFrame` is unchanged and still used by Dashboard timeline tile clicks (where the operator wants the read-only viewer + render-on-demand annotated thumbnail).
- **Render-on-demand scope clarified**: `/api/render_detected/` is for the Dashboard timeline tiles + the Download button only. Chart-dot click uses raw + LSF prefill (no render-on-demand needed).

### Added — Download bundle
- New 📥 **Download (raw + DETECTED)** button on the LSF modal toolbar. Triggers two browser downloads — `<stem>.jpg` (raw) and `<stem>_DETECTED.jpg` (render-on-demand). Each gets the `download` attribute so Chrome accepts both without prompting.

### Fixed — Latency / Inf / Cap still black
- The latency/FPS row's `<strong>`s were inheriting through a `font-size: 9px;` grid whose parent didn't pin a colour. Pinned `color: var(--text-primary)` on both the grid and every `<strong>` inside.
- Added a global CSS fallback `label, strong, b, em, i { color: inherit }` so future `<strong>`s in the dark theme won't fall back to UA-default black.

### Notes
- Cache-buster bumped to `?v=4.0.2`.

## [4.0.1] - 2026-06-14

### Fixed — Dashboard counter text + Timeline-config radio labels reverted to black
- Several counters under the live-status box (Speed, Pulses/s, Movement, Ej Queue, Ej Active, Ej Enable, Ej Offset, Downtime, Analog, Power) had no explicit `color:` set and were inheriting from somewhere that drifted to black against the dark background. Pinned every counter `<div>` + `<span>` to `color: var(--text-primary)`.
- Timeline Configuration's *Ascending / Descending / Custom* radio labels had the same issue (bare `<label>` inheriting black). Pinned the inline style, and added a global CSS rule `label { color: var(--text-primary) }` so future labels can't regress.

### Notes
- Cache-buster bumped to `?v=4.0.1`.

## [4.0.0] - 2026-06-14

### Changed — render-on-demand annotated frames (one file per frame, not two)
- **Pipeline no longer writes `<frame>_DETECTED.jpg`.** Every detected frame used to land on disk twice — raw + pre-rendered annotated. From 4.0 on, only the raw frame is persisted. The annotated view is rendered on demand from `inference_results.detections`.
- **New endpoint `GET /api/render_detected/<path>?show=&download=`** — reads the raw JPG + the per-frame detections, draws bboxes via the existing `services.render.draw_detection_on`, returns JPEG bytes. Redis cache keyed on `(path, show, mtime)` for 1 hour — first view costs ~20 ms of OpenCV; every subsequent hit is a sub-millisecond Redis lookup.
- **Download button** (existing PDF report uses this) and dashboard thumbnails switch transparently: `_imgUrlsFor(pt).ann` now points at `/api/render_detected/<raw_path>` instead of the disk file. Operators see no change.
- **`?download=1`** sets `Content-Disposition: attachment; filename="<stem>_DETECTED.jpg"` so the on-demand render still saves to disk as `_DETECTED.jpg` for downstream tooling — but only when the operator actually clicks Download.
- **Legacy URL fallback**: `/api/render_detected/<stem>_DETECTED.jpg` strips the suffix and serves the same render, so any cached link from the pre-4.0 era still works.

### Disk impact
- Across the fleet `raw_images/` was carrying ~2× the necessary bytes. Cleanup of existing `*_DETECTED.jpg` files is a follow-up — the system runs fine with them in place but reclaiming the disk requires a one-shot walk per site. Plan: run `find raw_images -name '*_DETECTED.jpg' -delete` on each site after vteam12 is verified.

### Active-learning groundwork (lands in 4.1)
- The inline LSF editor as the default drawer view (no more separate ✏️ Annotate modal click) is deferred to 4.1.0 — the existing modal stays for now so the canary is testing only the render path. Once render is verified the editor moves inline.

### Notes
- Cache-buster bumped to `?v=4.0.0`.

## [3.26.4] - 2026-06-14

### Changed — AI models config + active shipment now survive restart
- **AI models** (DeepSeek/Kimi/Claude name + key + base_url + active) are now persisted to `service_config.ai_models` → TimescaleDB `mve_config_kv` + `.env.prepared_query_data` snapshot. Redis kept as a hot read cache for non-migrated readers (`config_routes.py:32`), but the source of truth is the DB. `docker compose down`, `--force-recreate`, host reboot — all preserve your AI models now.
- **Auto-migration**: first call after upgrade copies any Redis-only data into the DB. No manual step required.
- **Active shipment**: now persisted to `service_config.current_shipment`. On every boot, `apply_config_settings()` reads it, primes the watcher's `.shipment` attribute, warms the Redis cache, and pre-creates the `raw_images/<id>/` directory — operator never finds the line back on `no_shipment` after restart. Sentinel `"no_shipment"` is *not* restored (avoids clobbering a fresh default).

## [3.26.3] - 2026-06-14

### Fixed — Annotate save: "upload succeeded but no TaskImage id returned"
- `POST /api/ai_trainer/upload` was returning the trainer's response as a truncated string (`r.text[:500]`) which couldn't be parsed for the TaskImage id. Backend now also parses the JSON and surfaces `image_id` directly. `annotate.js` reads `upData.image_id` first, then falls back to scanning `trainer_response_json`.

## [3.26.2] - 2026-06-14

### Fixed — Annotate modal "issue loading URL from $image value"
- `GET /api/frame_detections` returned `image_url` as `/raw_images/<path>` which is not a route MVE serves. The correct prefix is `/api/raw_image/<path>` (the same one the defect drawer's image view uses). LSF tried to fetch the non-existent path and surfaced its generic "couldn't load $image" error.
- Also strip the trailing `_DETECTED.jpg` from the stored `image_path` so the editor opens the RAW frame — drawing corrected boxes over YOLO's already-drawn boxes is wrong.

### Notes
- Cache-buster bumped to `?v=3.26.2`.

## [3.26.1] - 2026-06-14

### Changed — AI Trainer config now uses base_url
- The Advanced → AI Trainer form field renamed **"Trainer URL"** → **"Trainer base URL"**. Operator enters just the origin (`https://ai-trainer.monitait.com`); MVE appends the right path (`/api/images/`, `/api/categories/`, `/api/annotations/`, …) per endpoint.
- Backend `service_config.ai_trainer.base_url` is the new canonical field; legacy `url` is kept as a derived alias for any older client code. If only the legacy `url` is set (pointed at `/api/images/`), `base_url` is derived from it automatically on next read — no data migration required.
- `_trainer_origin()` now reads `base_url` directly instead of stripping the upload URL each call.

### Changed — Default scatter axis = Encoder
- Detection Insights now defaults to 📏 **Encoder** instead of 📍 Time (operators on roll-based lines care about position more than wall-clock). Strip titles + button styling start in encoder mode; the toggle still flips both at once.

### Fixed — Annotate diagnostic shows real cause of "bundle did not load"
- The old alert only said "check /static/vendor/label-studio-1.4.0/js/main.js" which gave no actionable info. New alert lists every `<script>` tag matching `label-studio` actually present on the page, reports `typeof LabelStudio`, and tells the operator to hard-reload (the #1 cause is a status.html cached from before the 3.26.0 deploy that introduced the LSF `<script>` tag).
- The "No frame selected" bug from 3.26.0 (cross-script `let` binding not visible to annotate.js) was already fixed in the canary build; this release just bumps the cache-buster so browsers actually fetch the fix.

### Notes
- **Cache-buster bumped to `?v=3.26.1` on every custom JS script tag** — this is the version you'll see when the change reaches your browser. From now on every meaningful UX change ships with a fresh `?v=` so the operator can confirm the new code is live.

## [3.26.0] - 2026-06-13

### Added — In-line annotation with the SAME editor as ai-trainer.monitait.com
- New ✏️ **Annotate** button on the defect drawer opens a full-screen modal containing **Label Studio Frontend 1.4.0** — byte-identical to the editor on ai-trainer.monitait.com (same npm package and version, confirmed against the trainer source in `monitait_all_services_deployment/singularity/rutilea_singularity_trainer_frontend/package.json:12`).
- **Self-hosted** at `/static/vendor/label-studio-1.4.0/{js,css}/main.{js,css}` (~2 MB JS + ~565 KB CSS). Air-gapped factories run identically — no CDN dependency.
- **Categories fetched live** from the trainer on every modal-open via new `GET /api/ai_trainer/labels` (proxies `GET /api/categories/?task=<id>` using the existing JWT flow). The LSF `<RectangleLabels>` config + Label `value` strings match the trainer's own `Training.vue:621` (`"${category_name} - ${id}"`).
- **Pre-fill from YOLO** — new `GET /api/frame_detections?image_path=<rel>` returns the stored detection list for one frame; the JS converts pixel `xmin/ymin/xmax/ymax` → LSF percentages and feeds them in as the initial annotation result. Operator sees every box on first paint.
- **MVE class ↔ trainer category mapping**, persisted in `service_config.ai_trainer.class_map`. First time a frame contains an unmapped MVE class, an inline yellow panel asks the operator to pick the trainer category once; after Save, future frames pre-fill silently. Endpoints: `GET/POST /api/ai_trainer/class_map`.
- **Save flow** = `submitAnnotations()` → uploads the raw frame via existing `/api/ai_trainer/upload` → receives `TaskImage.id` → POSTs the corrected box list to `POST /api/ai_trainer/annotate`, which proxies to the trainer's `POST /api/annotations/` (shape matches `AnnotationCreateUpdateSerializer` in the trainer's `serializers.py:144` — `bbox`/`normalized_percent_points`/`area`/`image_id`/`category_id`/`task_id`/`ignore=true`/`is_crowd=false`/`segmentation=[]`).
- Companion proxies for completeness: `PUT /api/ai_trainer/annotate/{id}`, `DELETE /api/ai_trainer/annotate/{id}`.

### Notes
- Cache-buster bumped to `?v=3.26.0` on every custom JS script tag.
- Bundle gets fetched once, cached by the browser per `?v=` string; subsequent version bumps don't re-download LSF unless the file path changes (it doesn't — only `?v=` on `status.html`'s reference does).

## [3.25.13] - 2026-06-13

### Changed — merged Time + Encoder scatters into one with a toggle
- The two redundant "Camera × time" and "Camera × encoder" scatter charts collapsed into a single scatter with a **📍 Time / 📏 Encoder** toggle above it. Less real estate, less duplication.
- The detection scatter, quality strip, and ejection strip all share the SAME X-axis. Toggling the axis re-renders the scatter (data + tick formatter + click handler all switch) and reloads both strips so a vertical column on any of the three points to the same time or encoder position.
- New `setInsightAxis('time'|'encoder')` in `charts.js`. Default = time. State persists for the page session.
- Strip element IDs unified (`quality-strip`, `quality-strip-axis`, `quality-strip-wrap`, `ejection-strip`, `ejection-strip-legend`, `ejection-strip-title`, `quality-strip-title`). Old IDs (`quality-time-strip`, `quality-encoder-strip`, etc.) are gone.

### Fixed — strip labels timezone + edge alignment
- Quality strip and ejection strip labels were formatted in UTC by the backend (`strftime("%H:%M")` on a `timestamptz`), but the scatter was rendered in the browser's local TZ via `toLocaleTimeString`. Result: a 3.5h offset on Iran sites where the strip read "18:38" while the scatter read "22:08". Backend now also returns `ts` (ISO); frontend formats every time-strip label via `toLocaleTimeString` so the operator sees their own clock.
- Both time-axis backends (`/api/quality/heatmap` and `/api/quality/ejection_axis`) now bucket the FULL window `(NOW() - interval, NOW())` instead of `MIN/MAX` of the data rows. The strip now spans the same range as the scatter even when detections only fill part of the window — empty edge buckets render transparent.
- Cache-buster bumped to `?v=3.25.13` on every custom JS script tag.

## [3.25.12] - 2026-06-13

### Fixed — quality strip stuck on all-green
- Root cause: `score = 100 × (1 − impact_per_unit × SCALE)` with SCALE hard-coded at 1.0. On encoder-normalised lines, `impact_per_unit` is typically ~1e-4 (impact_total ≈ 25, encoder_span ≈ 250 000), so the score asymptotes to 99.99 regardless of severity values.
- **Fix:** new `score_scale_factor` in `service_config`, defaults to 1.0 (backwards-compat). Applied in `_compute_quality_payload`, the trend computation, and both axes of `/api/quality/heatmap`.
- **Operator UX:** new 🎯 **Calibrate score** button on the Shipment Quality Score card. First click previews the recommended factor (no DB write); second click applies. Target: p50=85, p5=60 over last 7 days. After applying, the strip starts showing red/yellow where the data warrants it.
- **New endpoint** `POST /api/score/calibrate {target_p50, target_p5, window, apply}` does the math + optionally writes `score_scale_factor` to `service_config`. Returns the projected p5/p50/p95 of the score distribution.

### Added — ejection axis strip
- Two new strips under the existing Quality strips, one per axis (time + encoder). Cell color = the procedure that fired most in that bucket (golden-angle hue, deterministic by name → same color across both strips so an ejection cluster matches visually). Hover shows the per-procedure breakdown. Legend below each strip lists every procedure that fired in the window with totals.
- **New endpoint** `GET /api/quality/ejection_axis?axis=time|encoder&window=24h&buckets=48` — returns per-bucket `{n, top_procedure, by_procedure: {name: count}}`.

### Fixed — shipment code 404 + new format
- `/api/shipments/next_code` was 404 on sites where `routers/health.py` was stale (drift-check earlier flagged this). New `health.py` deployed to all four sites.
- **New code format:** `yyyymmddXXYYZZZ` (15 chars) where `XX` = 2-digit ID of the active capture state and `YY` = 2-digit ID of the active inference pipeline (alphabetical index, padded). `ZZZ` is the per-day chronological counter (Redis + DB hybrid, same as 3.25.0). Legacy 11-char codes from earlier today are still respected by the DB scan so the counter never collides.

### Fixed — grid overflow on Detection Insights
- `min-width: 0` added to `#insight-charts` + `#ejection-charts` grid containers and every direct child. Chart.js canvases default to their intrinsic device-pixel width and were pushing the donut chart off-screen on smaller viewports (reproduced on khoy-vteam19 screenshot).

### Notes
- Cache-buster bumped to `?v=3.25.12` on all custom JS so a normal F5 picks up the new code.

## [3.25.10] - 2026-06-13

### Added — AI Severity Suggester now scores ejection procedures too
- The 🤖 **Suggest severities** button on the Process tab now returns suggestions for BOTH detection classes AND ejection procedures in a single AI call. Two tables appear in the green review panel — *Detection classes (N)* on top, *Ejection procedures (N)* below — each with the same per-row Apply / Apply-all flow.
- `POST /api/suggest_severities` payload now includes `procedure_suggestions: [{procedure, suggested_severity, current_severity, tier, reason, kind:"procedure"}]` alongside the existing `suggestions` array. The AI receives each procedure's rule summary, enabled/store flags, 7-day event count, and current severity, and is told to use the same 0–100 tier scheme. Procedures with 0 events_7d and 0 saved severity are auto-suggested 0 client-side (no AI roundtrip).
- New `POST /api/apply_procedure_severities` (`{updates: [{procedure, severity}]}`) writes the suggested severity back into `service_config.procedures[*].severity`. The Process tab updates without a page reload (calls `renderProcedures()`).
- **🚀 Auto-tune all** now covers procedures too — one click adjusts every class severity AND every procedure severity together, and the AI sees the full picture so the rebalance is coherent (e.g. lowering a class severity but raising the matching ejection-procedure severity stays in the same shipment-quality-score budget).
- Backwards-compatible parser: if an older AI prompt returns a bare JSON array, it's treated as classes-only and the procedure_suggestions array comes back empty.
- `?v=3.25.10` cache-buster on every custom JS script tag (see [[3.25.9]] for the why).

## [3.25.9] - 2026-06-13

### Fixed — Browser cache no longer pins old JS after a deploy
- Added `?v=X.Y.Z` cache-busters to every custom script tag in `status.html` (i18n.js, iframes.js, charts.js, audio.js, app-core.js). Browsers treat `app-core.js?v=3.25.9` as a different resource from `app-core.js?v=3.25.8`, so the new JS is fetched on the first page reload — no more empty Dashboard dropdowns + no more "Ctrl+Shift+R or it won't work" instruction after each deploy.
- Operator-visible symptom this fixes: after upgrading from 3.25.7 → 3.25.8 the Dashboard Capture + Inference dropdowns stayed empty because the cached old `app-core.js` didn't have `loadDashboardStatePicker`/`loadDashboardPipelinePicker`.
- Bump the literal `?v=` strings every release. CHANGELOG entry serves as the reminder.

## [3.25.8] - 2026-06-12

### Added — Ejection procedures get their own Severity score
- **New `Sev` field on every ejection procedure** (Process tab → Ejection Procedures, header row next to Enabled / Store). Range 0–100, persists immediately on change (same auto-save path as the Store checkbox). Default 0 = legacy behavior, doesn't affect score.
- **Quality score now includes ejection impact.** `_compute_quality_payload` joins `ejection_events` (rows with Store=ON) against the procedures config:
  - `ejection_impact = Σ (proc.severity / 100 × event_count)` for the window.
  - Folded into the existing `impact_total` and surfaced in `top_defects` as `"⏏ <proc_name>"` rows with `kind: "ejection"`, alongside the per-class detection rows.
  - Separately exposed in the payload as `ejection_impact` and `ejection_counts` for transparency / future visualizations.
- Same scale as per-class severity: a single ejection at severity 50 ≈ 0.5 impact-units (≈ one detection of severity 50 at confidence 1.0). So ejections and detections are commensurable in the score formula — no calibration knob needed.

## [3.25.7] - 2026-06-12

### Changed — Detection Insights grid reordered by importance
- **Top of the grid: Pareto bar + Pareto pie + Camera × time + Camera × encoder** (the at-a-glance answers — "what's the biggest defect" and "where on the line / when").
- **"Secondary diagnostics" separator + the four less-glanceable charts pushed below**: Detection over time, Object size, Detection confidence over time, Detection confidence by class. Operator scrolls to them only when investigating a specific dimension; the top-of-fold view stays focused on the high-value questions.

## [3.25.6] - 2026-06-12

### Fixed — Strip ↔ scatter alignment
- **The quality-by-time strip now lines up 1:1 with the Camera × time scatter above it.** Chart.js insets the plot area by the Y-axis label width — the strip was painting at 100% of the container so the timestamps drifted. Fix wraps the strip in a positioned div and uses Chart.js' `animation.onComplete` + `onResize` hooks to set the wrap's `marginLeft` / `marginRight` from `scatter.chartArea` after every render. Same fix applied to the encoder strip.
- **Scatter X-axis now spans the full selected window** (24h / 6h / 1h / 7d) instead of auto-fitting to the data range. So if there are only 5 hours of detections in a 24h window, the scatter shows 24 hours of empty space — and the strip below it represents the same 24 hours. Direct visual cross-reference between scatter and strip.
- Strip wrap has `transition: margin 120ms ease` so the inset glides into place on window resize instead of snapping.

### Changed — Charts tab restructured
- **One chart per concern, no duplicates.** The standalone *Quality by time* and *Quality by encoder* 1-D heatmap strips were merged into the existing **Camera × time** and **Camera × encoder** scatter plots — each strip now sits flush against the bottom of its matching scatter, sharing the same X-axis context. Same IDs (`quality-time-strip`, `quality-encoder-strip`) so the JS handlers don't change.
- **Detection Insights pinned at the TOP** of the Charts tab (it carries the Shipment Quality Score card + scatters + strips — the most-used panel).
- **New panel order**: Detection Insights → Score per shipment → Ejection Insights → Production KPIs (reordered by frequency of operator use).
- *Score per shipment* card slimmed down — it now only carries the per-shipment bar chart; the window picker is hidden (both heatmaps live inside Detection Insights and use Detection Insights' window selector).

## [3.25.4] - 2026-06-12

### Added — Quality charts on the Charts tab
- **📊 Score per shipment** — horizontal bar chart, one bar per recent shipment (last 30 with non-`no_shipment` IDs). Bar height = quality score (0–100), color = verdict (green RELEASE / amber RE-INSPECT / red HOLD). Tooltip shows top 3 defects. Backed by new `GET /api/quality/shipments?n=30&window=30d`.
- **📍 Quality by time (1D heatmap)** — a 48-cell colored strip across the chosen window. Each cell = a time slice (~30 min for 24h). Color encodes the quality score in that slice. Hover shows the slice label + total detections + top class. Spots an operator can act on: "around 14:00 we had a red cluster — what shipment was running?"
- **📏 Quality by encoder position (1D heatmap)** — same idea but X-axis is encoder position (roll/conveyor distance). Operators see "trouble around encoder 80k–90k" without ploughing through frames. Falls back to "no encoder data" when the line wasn't moving.
- Both heatmaps share `GET /api/quality/heatmap?axis=time|encoder&window=24h&buckets=48`. Buckets are derived per call; score uses `Σ (severity × confidence × n_per_bucket)` against the per-class severity already in `audio_settings`.
- **Window picker** + **Refresh** on the card; auto-refresh every 60s while the Charts tab is visible.

## [3.25.3] - 2026-06-12

### Added — 🤖 AI auto-pick capture state on the Dashboard
- New **🤖 ?** button right next to the Dashboard capture-state dropdown. When the operator doesn't know which state to use, click it — the AI inspects the defined states, the active inference pipeline, the camera count, last-hour detection activity, and (optionally) a product description, then recommends the best-fit state.
- New backend endpoint `POST /api/states/ai_recommend` returns a structured answer:
  ```json
  {"recommended_state": "infinite", "current_state": "default",
   "reason": "Continuous high-volume detections across cam 1+2 fit a steps=-1 state better.",
   "confidence": "high",
   "alternatives": [{"name": "default", "reason": "..."}]}
  ```
- The Dashboard expands a small green panel under the picker with the recommendation, reason, alternatives, and an **✓ Apply** button that hits the standard activate endpoint. If the AI picks an unknown state name, the endpoint returns 502 with the list of valid options.
- Uses the same `call_ai_model(tools_enabled=False)` path as the severity suggester — small token budget (2048), one-shot call, logs to `ai_usage_log`.

## [3.25.2] - 2026-06-12

### Added — 🤖 AI-suggested Severity values for every class
- New **🤖 Suggest severities** button at the top of the Process tab (in the filter bar). Clicking opens a green panel with an optional "describe your product" input, a **🤖 Run (review)** button, a **🚀 Auto-tune all** one-click button, and a **✓ Apply all** button.
- New backend endpoint `POST /api/suggest_severities` gathers every `audio_settings` class with its detection count, p5/p50/p95 confidence from `/api/conf_baselines`, current Severity, plus whether ColorE/Area are on, builds a structured prompt for the active AI model, and asks for a 0–100 Severity per class with a one-sentence reason and a tier (`COSMETIC` / `MODERATE` / `SERIOUS` / `CRITICAL` / `NONE`).
- **Shipment-score-aware**: the prompt also includes a snapshot of the last 30 shipments' quality scores (p5 / p50 / p95 of the distribution + per-shipment verdicts and top defects). The system prompt tells the AI to aim for severities that keep p50 between 80 and 92 with bad shipments dropping below 70 — so the resulting score distribution is calibrated, not all-clumped at 99 or all-clumped at 30.
- **Active vs silent class split**: classes with ≥ 5 detections in 7 days go to the AI; the rest are auto-suggested 0 without an AI roundtrip. Keeps prompt size manageable even on sites with 200+ historical classes.
- **`max_tokens` lifted to 16384** for this endpoint so even very long JSON responses (50+ active classes × ~80 chars each) don't get truncated mid-string.
- **🚀 Auto-tune all** button = one-click "suggest + apply" sequence. Operator gets a confirmation modal explaining "this overwrites your existing severity values" then watches the panel update.
- Severity tier scale baked into the system prompt:
  - `0` — math channels / metrics (mean_L, fft_*, blob_*, std_L, …) — auto-suggested at 0
  - `1–20` cosmetic
  - `21–50` moderate
  - `51–80` serious
  - `81–100` critical (single instance can reject the shipment)
- Results land in a sortable table with columns Class · Current · Suggested · Tier · Reason · Apply (✓ per row, ✗ disabled when current == suggested). Tier color-coded for quick scanning.
- **Apply all** button bulk-applies every non-equal suggestion via `POST /api/apply_severities` (which writes to `audio_settings.<class>.severity` + busts the draw_filters cache so the new values take effect on the next captured frame).
- Per-row and bulk applies are immediately reflected in the per-class card severity input — no page refresh needed.
- AI call uses `tools_enabled=False` (no agentic loop), respects the operator's UI language, and logs to `ai_usage_log` like other AI calls so the per-site bill captures it.

## [3.25.1] - 2026-06-12

### Changed — internal cleanup
- Docs, comments, UI tooltips, and changelog wording cleaned up. The notifications subsystem documents its bot host as "any Telegram-compatible service (self-hosted gateway, custom relay)" without naming specific external products.
- Legacy-channel cleanup code refactored to drop any non-`telegram` channel keys generically rather than enumerating one by name.

## [3.25.0] - 2026-06-12

### Added — Dashboard operator quick-controls
- **Shipment block is bigger**: 200px column width (was 160px), 18px shipment text (was 10px), highlighted gradient border. The Shipment label moves up to a small uppercase header, the ID itself is now prominent and readable at a glance from the line.
- **🎲 Auto-generate button** next to the shipment input. Produces a `yyyymmddXXX` ID where `XXX` is today's chronological sequence (001, 002, …). Backed by a new `GET /api/shipments/next_code` endpoint that scans the day's existing shipment rows in `inference_results` and picks the next free number — same operator can keep clicking 🎲 and get 001, 002, 003 across the shift; another operator clicking later in the day continues the sequence cleanly.
- **Inline pickers for capture state + inference pipeline** on the Dashboard. The two read-only status labels are replaced by `<select>` dropdowns populated from `/api/states` and `/api/pipelines`. Picking a different option calls the existing activate endpoints (`POST /api/states/{name}/activate`, `POST /api/pipelines/activate/{name}`). Operators can switch between `infinite` / `infinite-max` / `KC-Back` (or whichever states the site has defined) or between `yolo_plus_math` / `yolo` / `default` pipelines without leaving the Dashboard.
- Pickers self-refresh every 30 s so changes made by another operator on the same site stay visible.



### Fixed — AI Trainer upload: send `status=" "` + list-of-tuples multipart + auth-path retries
- The trainer's `TaskImage.status` DB column is NOT NULL (model default = `" "` literal space — `TaskImageStatus.NULL` in their `choices_fields.py`). Their view does `status=request.POST.get("status")` which returns Python `None` when we don't send it, overriding the model default and triggering an `IntegrityError: null value in column "status" of relation "api_taskimage" violates not-null constraint`. That surfaced to MVE as `400 "No files were processed successfully"` because the view's per-file `except` catches the DB error, logs it, and the empty `upload_locations` list at the end returns the generic 400.
- We now send `status=" "` (matching the model default) in the multipart form data so the row inserts cleanly. Other valid values per their enum: `OK`, `NG`, `NA`, `MU`.
- The multipart payload for `files` is now built as a list-of-tuples (`[("files", (name, bytes, mime))]`) instead of a dict (`{"files": (...)}`) — the list form is the canonical multi-file shape Django's `request.FILES.getlist("files")` expects.
- `_trainer_login` and `_trainer_refresh` (the JWT auth helpers) now use the same 3-attempt retry-with-backoff as the upload, so the transient docker-bridge stalls that bit the upload also don't bite the login path.

## [3.24.8] - 2026-06-11

### Added — AI Trainer JWT auth + transient-DNS retry
- The trainer (Django + DRF + `rest_framework_simplejwt`) uses JWT auth, not a static API key. Tokens have a 6h access / 1d refresh lifetime — too short for a long-running production site to use a manually-pasted token. MVE now handles the full login flow:
  - **`POST /api/ai_trainer/config`** accepts a new `email` and `password` (in addition to the legacy `api_key`).
  - On every upload, `_ensure_access_token(cfg)` checks the cached access token; if expired or missing, refreshes (via stored refresh token) or re-logs-in with `email`+`password`. The fresh tokens are persisted so subsequent uploads reuse them. Operator never sees a token.
  - **`POST /api/ai_trainer/login_test`** runs a credentials check without uploading — used by the new **Test login** button.
- Auth precedence: explicit `api_key` (legacy override) wins over JWT. Both clear and documented in the Advanced → AI Trainer panel.
- New UI fields in Advanced → AI Trainer: `Trainer email` and `Trainer password` (write-only). Saving a new password invalidates the cached tokens so the next upload picks the new ones up.
- Default URL placeholder updated to the corrected `https://ai-trainer.monitait.com/api/images/` (was the now-incorrect `/api/tasks/{task_id}/upload`).

### Fixed — transient DNS failures on upload
- The MVE container's embedded Docker DNS resolver occasionally fails to look up external hosts ("Errno -5: No address associated with hostname") even when the host's resolver is fine. The upload code now retries 3 times with brief backoff (0.5s, 1.0s) — empirically eliminates the intermittent 502s that the operator was hitting.
- The file is read into bytes once and reused across attempts so retries don't re-open or re-stream the source jpg.

## [3.24.7] - 2026-06-11

### Fixed — AI Trainer upload was hitting a 404 URL
- The default upload URL was `https://ai-trainer.monitait.com/api/tasks/{task_id}/upload` — that path never existed on the actual trainer (confirmed by reading the trainer's Django source on prod: routes are registered via DRF's `router.register("images", TaskImageViewSet)`, so the real endpoint is `POST /api/images/`).
- Default URL is now `https://ai-trainer.monitait.com/api/images/`.
- Field names corrected to match the trainer's `swagger_auto_schema` declaration:
  - **`task_id`** (form field) → **`task`** (integer)
  - **`file`** (single file) → **`files`** (multi-file capable)
- After this fix the trainer responds with a clear `401 "Authentication credentials were not provided."` for unauthenticated requests — operators paste an API key into Advanced → AI Trainer and uploads go through as `201 Created`.
- The original `502` the operator saw was MVE's `requests` library mapping the trainer's `404 Not Found` (wrong URL) into a generic gateway error.

## [3.24.6] - 2026-06-11

### Added — Schedule trigger types + multi-shipment support
- Schedules now have a **`trigger_type`** picker — two flavours:
  - **⏰ `cron`** (default) — fire on a 5-field crontab expression, e.g. `0 8 * * *` for 08:00 daily. Unchanged behaviour.
  - **📦 `shipment_change`** — fire automatically when the Redis `shipment` key changes (a shipment closing). The report is generated for the PREVIOUS shipment, the one that just ended. Pair with a `shipment_filter` glob to scope which closures trigger a notification.
- Both trigger types can run in parallel on the same site. Typical setup: one `cron` schedule at end of each fixed shift PLUS one `shipment_change` schedule that catches any out-of-shift shipment closes.
- **`shipment_filter`** is now a glob pattern, not just a literal name:
  - `""` or `"*"` — matches every shipment (any closing fires the schedule)
  - `"shoga-*"` — matches all SHOGA shipments
  - `"shoga-2026-q2"` — exact name only
  - Multi-line factories can run several `shipment_change` schedules each filtered to one line (`line-a-*`, `line-b-*`, …) routing to different chat IDs.
- Same multi-shipment logic on the cron path: when a cron-triggered schedule has a `shipment_filter` glob, the matching shipment is picked at fire time (currently: filter is treated as a literal at fire time; multi-shipment expansion is a follow-up).

### Notes
- The scheduler tracks `_last_seen_shipment` in memory only (initialised on first tick). A reboot mid-shipment doesn't re-fire on first wake-up because we initialise `last_seen` from the current value before checking for change.

## [3.24.5] - 2026-06-11

### Fixed — shift-report scheduler couldn't generate the PDF
- `services/scheduler.py::_fetch_shipment_report_pdf` was calling a function name + signature that didn't exist on this branch. Replaced with the real handler (`routers/timeline.py::shipment_quality_score_report`), passing a minimal Request shim (the handler only reads `request.app`).
- The PDF endpoint returns a `StreamingResponse`, not a buffered one — switched to iterating `body_iterator` to collect the bytes properly.
- After this fix, **Send now** in Advanced → Notifications generates and delivers the quality PDF to the configured chat in real time. Verified end-to-end: `message_id: 7, status: ok`.

## [3.24.4] - 2026-06-11

### Changed — Notifications: single Telegram channel with configurable base URL
- There is now a **single Telegram channel** with a configurable **Base URL** field — leave it blank to use the standard Telegram API (`https://api.telegram.org/`), or point it at any Telegram-compatible bot service (self-hosted gateway, custom relay, …) without code changes. The wire format is the same; only the host differs.
- `services/messaging.py::send_document(...)` / `send_text(...)` now take an optional `base_url` instead of a `channel` arg. Default = Telegram. The `channel` kwarg is kept as ignored for back-compat so older callers (custom integrations) don't break.
- `routers/notifications.py` only accepts `channel="telegram"`. Any unsupported legacy channel key in storage is silently dropped on the next read/save — operators don't see it again.
- `services/scheduler.py` only sends through the telegram channel; it reads `base_url` from the channel config and passes it to `send_document`.
- UI: Advanced → Notifications collapses to a 4-field grid (Bot token / Default chat id / Base URL / Enable). The schedule rows lose the "channels" column since there's only one channel — that frees space for a slightly wider chat-id column.

## [3.24.3] - 2026-06-11

### Fixed — Why? no longer freezes other endpoints
- `call_ai_model(...)` now takes a `tools_enabled` flag (defaults `True` for backward-compat with the chat endpoint `/api/ai_query`). `/api/why` passes `False`, which: (a) skips the function-calling setup entirely, (b) collapses the 5-iteration agentic loop to a single round-trip, (c) prevents the sync `execute_tool(...)` chain that was running on the event loop and blocking concurrent requests (dashboard SSE, timeline composite, charts queries) while a Why? call was in flight.
- We pre-load all the context the AI needs in `/api/why` already (per-class baselines, color drift, area stats, quality-score payload, per-minute count timeline, top co-occurring classes, Redis sys-state). Letting the model also chase its own tools just added latency without improving the answer.
- Net effect: clicking 🤔 Why? is now ~3× faster and the rest of the app keeps moving.

## [3.24.2] - 2026-06-11

### Fixed — Why? on per-class card had ZERO context (lying about "no recent samples")
- `mode=class` on the Process tab Why? chip never passes a timestamp, but `/api/why`'s SQL block only ran when `ts_in` was truthy. So `ctx_lines` came back empty and the AI hallucinated "no activity" / "no recent samples" even though the card on screen showed 200k+ samples and rich percentile data. Now defaults `ts_in` to `now` when missing — every call gets the surrounding-context queries.

### Added — Per-class Why? now sees the same analytics the operator sees on the card
- For `mode=class`, the prompt now includes the per-class baseline data the operator stares at:
  - **📊 confidence percentiles** (overall + per-camera p5/p50/p95 + n) — same as the 📊 normal-conf badge on the card
  - **🎨 CIELAB color** (L/a/b percentiles + per-camera breakdown + E magnitude percentiles) — same as the 🎨 color line
  - **📐 bbox area percentiles** (overall + per-camera) — same as the 📐 area line
  - Per-minute count timeline over the last hour + top-5 co-occurring classes
- Backend now calls `_compute_conf_baselines()`, `get_color_drift()`, and `get_area_stats()` in-process (no HTTP round-trip) and serializes the results into the prompt.

## [3.24.1] - 2026-06-11

### Fixed — Why? on Quality Score now has real defect data
- `/api/why mode=score` fetches the shipment quality-score payload (`_compute_quality_payload`) and stuffs it into the AI prompt: top defects with impact / count / severity, impact total, impact-per-unit, normalization mode, throughput, encoder span. Previously the AI said "no quality subcomponent data is shown" because the prompt only had the bare score + Redis sys-state.

### Fixed — AI calls no longer block the FastAPI event loop
- The Anthropic and OpenAI SDK clients are synchronous; calling `client.messages.create(...)` / `client.chat.completions.create(...)` directly from an async handler blocked other requests until the AI provider responded (often 5-30 s). When two operators clicked 🤔 Why? at the same time, the second queued behind the first; worse, ANY other endpoint was blocked too.
- Both calls are now wrapped in `asyncio.to_thread(...)`, so the AI provider's response runs on a worker thread and the FastAPI loop stays free. Concurrent Why? clicks are now genuinely parallel and unrelated endpoints (timeline composite, /api/cameras, …) keep their normal latency under AI load.

### Changed — Notifications UI: compact channel table
- The two-stack-of-fields layout (one section per channel) is replaced by a single 5-column table — channel | bot token | default chat id | enabled | actions. Same fields, half the vertical space, easier to compare at a glance. Backend unchanged; same `/api/notifications/config` shape.

## [3.24.0] - 2026-06-11

### Added — Telegram shift report delivery (no SMTP, no email)
- Shift-end PDF reports now arrive in **Telegram**, not email. Backend wrapper: `services/messaging.py::send_document(channel, token, chat_id, file_bytes, caption)`.
- **Cron-style scheduler** in `services/scheduler.py` runs as a background asyncio task spawned at MVE startup. Wakes once per minute, checks each enabled schedule against the current local time, fires the matching ones. Supports a useful cron subset: `*`, lists (`0,15,30,45`), simple ranges (`8-17`), step (`*/15`), 5-field `MIN HOUR DOM MON DOW`.
- **Notifications panel** in Advanced tab — tokens are write-only from the UI (the API only ever returns the masked form `1234…wxyz`), with per-channel enable + default chat ID, a schedule editor (one row per shift with cron / channels / chat IDs / Why?-include / shipment-filter toggles), an explicit **Send now** button to skip the cron, and a live audit log of recent sends.
- **🤔 Why? caption pre-amble** — every shift report can optionally include the active AI model's 2-sentence diagnosis of the shipment quality score, prepended to the bot message caption. Operator gets a glance answer without opening the PDF.
- New endpoints:
  - `GET  /api/notifications/config` — masked config
  - `POST /api/notifications/config` — partial channel update OR full schedule replace
  - `POST /api/notifications/test_send` — fire-and-forget test message to validate token+chat_id
  - `POST /api/notifications/send_now` — manual trigger that skips the cron
  - `GET  /api/notifications/log` — recent send audit (status, latency, errors)
- Notification audit lands in a new `notification_log` table, schema bootstrapped on first use.

### Added — Per-call AI usage tracking + billing data
- Every `/api/why` call now logs to a new `ai_usage_log` table: endpoint, mode, model name, provider, token-in / token-out approximations, estimated cost USD, latency ms, status, operator (X-Operator header), and error if any.
- Per-model rate cards: defaults to Kimi K2's public rate (≈ $0.28 / 1M input · $1.40 / 1M output). Per-model overrides supported via `audio_settings.ai.<model>.rate_input_per_mtoken` / `rate_output_per_mtoken`, so resellers can charge their own price.
- New endpoint: `GET /api/ai_usage/summary?window=30d` returns total calls, total cost USD, by-endpoint and by-day breakdowns — the data feed for the per-site monthly bill.
- Logged regardless of outcome (errors land too) so quota-exceeded / timeout attempts are visible in billing dashboards.

### Expanded — AI Assistant API access (continuation of 3.23.1)
- Internal API tool list extends to cover the new endpoints: `/api/notifications/log`, `/api/ai_usage/summary` so the chat AI can summarize "how much did Why? cost us this month" and "which schedule failed last night".

## [3.23.1] - 2026-06-11

### Added — Why? chip spread to more entry points + multilingual replies
- 🤔 chip next to every per-class card title on the **Process tab**. Mode = `class` — asks "why is THIS class behaving this way right now?" Result lands in a small inline panel under the card header.
- 🤔 chip next to the **Shipment Quality Score** card on the Dashboard. Mode = `score` — explains the current score and the dominant contributor in 2 sentences.
- `/api/why` now accepts a `mode` parameter (`dot` | `class` | `score` | `verdict`) so the same backend handles every entry point with appropriate prompt scaffolding.
- `/api/why` now respects a `language` field. The AI's natural-language prose is generated in the operator's UI language (English / Persian / Arabic / German / Turkish / Japanese / Spanish — sourced from the existing `currentLang` localStorage key). Technical identifiers (class names, endpoint paths, numeric values, timestamps) stay in their original form.

### Expanded — AI Assistant API access
- The AI Assistant chat's internal-API tool now covers the analytics endpoints shipped this week: `/api/conf_baselines`, `/api/color_drift`, `/api/area_stats`, `/api/active_classes`, `/api/config/db_status`, `/api/shipment_quality_score`, `/api/audio_settings`, `/api/timeline_config`. The AI can now reach into all the per-class baseline data, color/area drift stats, config-storage health, and per-shipment quality figures without us hand-pasting context.
- The tool also exposes `POST /api/why` so the AI can chain a Why? lookup as part of a longer answer.

## [3.23.0] - 2026-06-11

### Added — 🤔 Why? chip on chart dots (operator-delight tier)
- Every chart-dot drawer now has a **🤔 Why?** button next to the AI Trainer upload. Click → the active AI model gets called with a focused, context-rich prompt about that specific dot and returns a **2-sentence diagnosis** rendered inline in a new green panel under the drawer body. No tab-switching.
- Pre-loaded context in the prompt (so the operator never types a precise question): metric/class name, value, timestamp, camera, encoder, shipment, current capture state + active pipeline name, real-time system snapshot from Redis (encoder, moving, OK/NG counters, downtime), surrounding chart context from `inference_results` (per-minute counts of THIS class over ±5 min, top-5 other classes firing in the same window).
- New endpoint `POST /api/why` with payload `{metric, value, timestamp, camera, encoder, shipment, image_path, window_seconds}`. Returns `{answer, model, usage}`. Reuses the existing active-AI-model routing (provider + base_url + model_id) introduced in 3.21.23, so any OpenAI-compatible endpoint plugs in unchanged.
- UI: the Why panel auto-clears when a new dot is opened, shows a "🤔 thinking…" placeholder while in flight, and gracefully handles error responses with red-tinted inline text.
- Why this matters: turns the AI Assistant tab from "novelty" to "indispensable." The biggest single operator-delight feature in the roadmap, shipped on its own as a minor version.

## [3.22.5] - 2026-06-11

### Added — Absolute E (color magnitude), per-detection, usable for ejection
- New scalar **E** stored on every detection that has `lab_color`. Computed at detection time as `E = √(L² + a² + b²)` — the Euclidean magnitude of the CIELAB vector. Single number, no moving baseline, comparable across shifts.
- New ejection-procedure conditions in `services/detection.py`:
  - **`E_greater`** with field `E` — eject when measured E exceeds threshold
  - **`E_less`** with field `E` — eject when measured E is below threshold
  - **`E_between`** with fields `E_min` and `E_max` — keep detections inside the band, eject outside
- Rule-detail formatter handles the new conditions so the ejection log line reads `'TB' E 112.3 > 100.0` instead of just "rule triggered".
- New row on the Process tab per-class card: **`✴ E (abs): 51.4 · 54.7 · 58.9 (p5–p50–p95)`** with an (i) tooltip explaining the formula and how to wire it into a procedure. Pair the displayed p5/p95 with the ejection threshold and you get a one-click "catch both ends of color drift" rule.
- `/api/color_drift` response now includes an `E: {p5, p50, p95}` block per class (and per camera), in addition to the existing L/a/b breakdown. SQL uses `COALESCE(det.E, √(L²+a²+b²))` so old detections without the stored `E` field still contribute by computing from `lab_color`.

## [3.22.4] - 2026-06-11

### Changed — Absolute CIELAB instead of ΔE; (i) tooltips with formulas
- The 🎨 color card row now shows **absolute L\*a\*b\* percentiles** instead of a single ΔE drift number. A single ΔE collapsed all three color dimensions into one scalar — you couldn't tell whether the color shift was L (exposure / dye fade), a (green↔red), or b (blue↔yellow). Three rows of p5/p50/p95 — one per channel — keep the diagnosis legible.
- Display: `🎨 color L: 50 · 54 · 58   a: -2 · 1 · 3   b: -8 · -5 · -2   (p5–p50–p95)   n=1.7k` plus a per-camera version below.
- `/api/color_drift` response shape changed accordingly: each class now has `L`, `a`, `b` keys each with `{p5, p50, p95}` (and the same shape inside each `by_camera` entry). Removed `reference_lab`, `p5_de`, `p50_de`, `p95_de` (ΔE can still be computed client-side as √((ΔL)² + (Δa)² + (Δb)²) — documented in the tooltip).
- Added an **(i) tooltip on each percentile badge** (normal conf / color / area) explaining what the metric is, the formula, and how to read it. Operators can hover instead of guessing.
- Card section order is now: header → toggles row → Min conf + Severity inputs → 📊 normal conf → 🎨 color → 📐 area → beep sound select. Actions on top, numbers below.

## [3.22.3] - 2026-06-11

### Added — Area per-class toggle + bbox-area p5/p50/p95 analytics
- New **Area** checkbox on every per-class card in the Process tab, next to ColorE. When ticked, the card shows a **📐 area** line with p5 · p50 · p95 of the bounding-box area (pixel²) over the last 7 days, plus a per-camera breakdown.
- New endpoint `GET /api/area_stats?window=7d` returns per-class bbox-area percentiles + per-camera breakdown.
- **YOLO inference output doesn't include `area` natively** (just xyxy bbox) — same with the math channels. So when a class is opted-in via the Area checkbox, `services/detection.py` now writes `det.area = (xmax-xmin)*(ymax-ymin)` once at detection time, right next to where `_cam` and `lab_color` are already attached. Stored once, read by everything downstream — `/api/area_stats`, future ejection procedures (`when area > N`), defect-size trending, etc. Backward-compat: the area_stats SQL uses `COALESCE(det.area, bbox computation)` so old detections without `area` still contribute their bbox-computed value.
- Display formatting: areas are abbreviated for readability (`12.4k px²` instead of `12,450 px²`, `1.23M px²` instead of `1,234,567`). Operators read patterns faster.
- **Use case:** spot false positives ("TB normally covers 12k–35k px²; today's coming in at 2k px² are probably FP — raise min_conf") and yarn/merge issues ("the p95 just jumped from 35k to 120k, two objects are getting merged"). Pairs cleanly with the existing confidence percentiles + ΔE drift to give a full statistical picture per class.
- Persistence: `audio_settings.<class>.area` is the new field. Backwards compat: absent = off, just like ColorE.

## [3.22.2] - 2026-06-11

### Added — ColorE per-class toggle + ΔE drift analytics
- New **ColorE** checkbox on every per-class card in the Process tab, alongside Show / Narrate / Beep / Store. When ticked, the annotator extracts CIELAB color (`L*a*b*`) from the detection's bbox on every frame for that class and stores it on the detection (`det.lab_color`).
- New endpoint `GET /api/color_drift?window=7d` computes per-class color drift: the mean LAB over the window becomes the **reference color**, and we return the **p5 / p50 / p95** of CIE76 ΔE distances against that reference, plus a per-camera breakdown. Low p50 means the class's color is staying consistent; rising p95 over time means the line is drifting (lighting change, dye batch swap, etc).
- New display line on the per-class card when ColorE is on:
  ```
  🎨 ΔE drift: 0.8 · 2.4 · 6.1 (p5–p50–p95) · n=4567   ref L=72 a=-3 b=12
  cam 1: 0.5 · 1.9 · 4.8 n=2.2k · cam 2: 1.2 · 3.1 · 7.4 n=2.4k
  ```
  Operators ticking ColorE on a class get immediate "is the color holding steady?" feedback without setting up a full color_delta ejection procedure.
- Backend extraction is CPU-aware: when no `color_delta` ejection procedure exists (legacy path), the annotator only extracts LAB for the specific classes opted in via ColorE. With a `color_delta` procedure, the legacy path stays — extracts for all detections — for backward compatibility.
- Cache and persistence: ColorE flips persist in `audio_settings.<class>.color_e`. The bulk-save endpoint accepts the field. The single-class POST accepts it (`{class_name, color_e}`).

## [3.22.1] - 2026-06-10

### Removed — 🪄 auto-suggest buttons on the per-class baseline lines
- Both the overall "🪄 auto p5" button and the small per-camera 🪄 buttons next to each percentile have been removed from the Process tab cards. They didn't fit visually in the existing layout. The percentile text (p5 · p50 · p95 + n) stays — the operator can read the noise floor and type Min-Conf manually. The `suggestMinConfFromBaseline()` JS helper is kept around in case we re-introduce it in a different location later.

## [3.22.0] - 2026-06-10

### Added — service_config moves to Postgres (with backward compat)
- New table `mve_config_kv (key TEXT PK, value JSONB, updated_at, updated_by)` becomes the source of truth for the MVE configuration that used to live entirely in `.env.prepared_query_data`. Each top-level field (`service_config`, `timeline_config`, `models`, `pipelines`, `current_pipeline`, …) becomes one row keyed by name.
- **Backward compatibility is preserved everywhere:**
  - **Read** (`config.load_data_file`) tries DB first, falls back to the JSON file if DB is unreachable or table is empty. So even if Postgres is down at boot, MVE still comes up using the file.
  - **Write** (`config.save_data_file`) dual-writes: persists to DB AND dumps the same JSON to `.env.prepared_query_data` on every save. Legacy tooling that reads the file directly (Advanced tab Data File Editor, `cat | jq`, pre-3.22 builds at remote sites) keeps working without code changes.
  - **Migration** is idempotent and one-shot: `services/migrations.py::migrate_data_file_into_db()` runs at MVE startup. If the DB table is empty AND the legacy file has content, it loads the file and bootstraps the DB with one transaction. After that, the file is a downstream snapshot only.
- **Why this matters:**
  - Atomic per-key writes (no more whole 50 KB file rewrites on a single toggle).
  - Audit trail (`updated_at` + `updated_by` columns answer "who turned off Store for arzi yesterday?").
  - Cross-site config management becomes possible (write once to a central DB, sites pull on next read).
  - The "MVE owns the file in memory and clobbers your edits" race goes away — DB is the truth, file is downstream.
- **What didn't change:**
  - The Advanced tab Data File Editor UI still shows the same JSON view and import/export endpoints — the data just comes from DB now.
  - No new dependencies (existing `psycopg2` connection pool in `services/db.py` is reused).
  - No operator action required after upgrade — sites self-migrate on first boot.
- **New endpoint:** `GET /api/config/db_status` returns `{backend, db_reachable, key_count, file_exists}` so the Fleet Dashboard (Tier 1 roadmap) can show config-source health per site.

### Changed — draw_filters is now self-sufficient (real single source of truth)
- `services/draw_filters.py` now owns the audio_settings read. It loads from `service_config` itself, caches with a 5 s TTL backstop, and exposes a `invalidate_cache()` hook. Callers just call `should_draw_class(name)` and `min_confidence_for(name)` — no parameters except the class name.
- Before this refactor four different files had to remember to pass `audio_settings` into `draw_filters` helpers; each time a caller forgot, the rule silently defaulted to "skip". This is what made the dashboard timeline composite blank in 3.21.25 — `routers/websocket.py` had its own annotation path that called `draw_detection_on(... obj_filters=...)` without audio_settings, so every class hit the default-skip branch.
- Updated callers (now parameter-free): `services/detection.py` annotator, `services/render.py::draw_detection_on`, `services/watcher.py` timeline composite, `routers/websocket.py::_build_timeline_composite`.
- Cache-bust hooks added to `POST /api/audio_settings` and the auto-register write path, so a Show-toggle change takes effect on the very next captured frame instead of waiting up to 5 s.
- From now on this is the pattern for any future per-class draw decision: it lives ONLY in `services/draw_filters.py`. Everything else just asks.

## [3.21.26] - 2026-06-10

### Changed — draw_filters is now self-sufficient (real single source of truth)
- `services/draw_filters.py` now owns the audio_settings read. It loads from `service_config` itself, caches with a 5 s TTL backstop, and exposes a cache-bust hook. Callers just call `should_draw_class(name)` and `min_confidence_for(name)` — no parameters except the class name.
- Before this refactor four different files had to remember to pass `audio_settings` into `draw_filters` helpers, and each time a caller forgot, the rule silently defaulted to "skip". This is what made the dashboard timeline composite blank in 3.21.25 — `routers/websocket.py` had its own annotation path that called `draw_detection_on(... obj_filters=...)` without audio_settings, so every class hit the strict rule's default-skip branch.
- Updated callers (now parameter-free): `services/detection.py` annotator, `services/render.py::draw_detection_on`, `services/watcher.py` timeline composite, `routers/websocket.py::_build_timeline_composite`.
- Cache-bust hooks added to `POST /api/audio_settings` and the auto-register write path, so a Show-toggle change takes effect on the very next captured frame instead of waiting up to 5 s.
- This is the architectural pattern from now on: any future per-class draw decision lives ONLY in `services/draw_filters.py`. Everything else just asks.

## [3.21.25] - 2026-06-10

### Changed — Show is now explicit opt-in (final answer)
- The "should we draw this class?" rule is now strict: `audio_settings.<class>.show == True` draws, **anything else (False, None, missing entry) skips**. This matches the Process tab's single Show checkbox: tick = draw, untick or missing = skip.
- Previous attempts: 3.21.22 was strict (broke yolo classes the operator hadn't explicitly Show-ticked → silent on restart). 3.21.23 flipped to default-on (math channels with no entry started flooding the annotated jpgs). Neither matched operator expectations. This is the right rule.
- Companion change: **auto-register newly-seen classes**. `services/detection.py::_auto_register_classes()` runs after every inference call — any class that appears in `yolo_res` but isn't in `audio_settings` gets a stub entry with `show=False`. So nothing silently disappears (every detected class shows up as a card in the Process tab, just unticked), and nothing draws unless the operator explicitly ticks it.
- UI: the Show checkbox now renders as **unchecked** when there's no saved value (was checked-by-default before). This matches the new backend rule so what you see in the UI is what the annotator does.
- Also adds the `p50` value to the per-class confidence baseline display: `📊 normal conf: 6 · 9 · 14% (p5–p50–p95) · n=15111` plus same format on each per-camera line.

## [3.21.24] - 2026-06-10

### Added — Process tab filter & busy-first sort
- New filter bar above the per-object grid: 🔍 search box, "Show only active" toggle, time-window picker (15m / 1h / 6h / 24h / 7d), and a counter chip ("Showing 12 of 134 · 8 active in last 1h").
- Classes with detections in the chosen window now float to the top of the grid automatically — even with the toggle off, the busy classes are first. Tail of idle classes stays alphabetical.
- New backend endpoint `GET /api/active_classes?window=1h` (cheap SQL group-by; returns class names, counts, and per-camera lists).

### Added — Per-camera confidence baselines + auto-suggest Min-Conf
- `/api/conf_baselines` now also returns `p5` and `by_camera` percentiles (p5/p50/p95/n per camera). Operators can finally tune `Min-Conf` per camera against the noise floor of THAT camera.
- Per-class card now shows a second line below the existing badge: `cam 1: 14–31% n=6.2k · cam 2: 6–12% n=8.3k`.
- 🪄 buttons next to each row set Min-Conf to that camera's p5 in one click — "set the threshold to capture everything that's historically real on this camera, drop nothing".

### Fixed (config, no code change to mainline) — math stride bias on 2-camera lines
- On the test box (vteam12) flipped the `math_v1` phase stride from `10` to `9`. With the global frame counter and 2 cameras alternating, an even stride means math only ever lands on cam 2; an odd stride alternates evenly. Future-proof per-camera-counter refactor queued for next release.

## [3.21.23] - 2026-06-10

### Fixed — bbox drawing for yolo classes that aren't explicitly in audio_settings
- 3.21.22's "no entry → don't draw" default was too strict. After every restart, yolo classes that the operator hadn't explicitly Show=true'd in the Process tab went silent. Reverted to "no entry → DRAW" (default-on for unknown classes).
- Behavior now: explicit `audio_settings.<class>.show == False` always wins; explicit `True` always wins; missing entry → draws (the original pre-3.21.10 behavior).
- The math-flood concern from earlier — where PVB had ~300 math detections per frame all drawing as kv-text labels — stays addressable by ticking Show=off on those specific classes once. Their show=False is then persistent.
- Also fixed the `timeline.py:2071` crash that surfaced when the operator hit "Apply Timeline Configuration" — leftover `len(object_filters)` reference from the 3.21.22 cleanup. Replaced with the preserved value.

## [3.21.22] - 2026-06-10

### Removed — stream:5000 log spam (dead service)
- `services/watcher.py:1537` was writing `output.jpg` and POSTing it to `http://stream:5000/send_frame_from_file/1` every loop iteration. The `stream` service was removed from `docker-compose.yml` long ago, so every site has been logging "Temporary failure in name resolution" once per frame ever since. Deleted the two lines.

### Changed — audio_settings is now the only canonical Show / min_confidence source
- The legacy `timeline_config.object_filters` dict is no longer written by the UI (`audio.js:1418`) and no longer accepted by the backend (`POST /api/timeline_config` now ignores the field with a log warning). The Process tab "Show" toggle is the canonical source.
- `services/draw_filters.py` simplified to read only `audio_settings`. The `obj_filters` kwarg is kept for signature compatibility but ignored.
- One-time startup migration `services/migrations.py::migrate_object_filters_into_audio_settings` folds any pre-3.21.22 `object_filters` entries into `audio_settings` so no class is silently dropped. Idempotent — safe to re-run.

### Added — AI Trainer upload from the defect drawer
- New "📤 Upload to AI Trainer" button in the defect-drawer header. One click POSTs the *raw* (un-annotated) frame for the currently-shown dot to a configured ai-trainer URL together with `task_id` + class + shipment + camera. Used to feed mislabelled / surprising detections into the model retraining pipeline at `ai-trainer.monitait.com`.
- New `routers/ai_trainer.py` with three endpoints:
  - `GET  /api/ai_trainer/config` — current url / task_id / has_api_key flag (key itself never returned)
  - `POST /api/ai_trainer/config` — set url / task_id / api_key
  - `POST /api/ai_trainer/upload` — body `{image_path, task_id?, class_name?, shipment?, camera?}` — strips `_DETECTED.jpg` to get the raw, validates the path is under `raw_images/`, forwards to the trainer URL with optional `Authorization: Bearer …` header.
- New Advanced tab → 🧠 AI Trainer section (URL + Task ID + API key + Save).
- URL template supports `{task_id}` substitution; otherwise task_id goes as a form field. Default URL: `https://ai-trainer.monitait.com/api/tasks/{task_id}/upload`.

## [3.21.21] - 2026-06-09

### Changed — unified show / min_confidence rule (one helper across three draw paths)
- Prior to 3.21.21 the "should we draw this detection?" decision was duplicated in three files (`services/detection.py` annotator save, `services/render.py` `draw_detection_on` helper, `services/watcher.py` timeline composite). Each read a different mix of `audio_settings.show` and `object_filters.show`. The dicts drifted apart and produced surprising behavior — that's what caused the fabriqc-kc agh/tooli_up missing-bbox bug (3.21.18 and earlier), then the PVB math-flood regression (3.21.19, my fix made it worse).
- New: `services/draw_filters.py` holds the single canonical rule:
  1. `audio_settings` is the source of truth (Process tab writes here). If the class has an entry, its `show` wins.
  2. `object_filters` is the legacy fallback (Advanced tab → Apply Timeline Configuration writes here). Consulted only when `audio_settings` has no entry.
  3. No entry in either dict → not drawn. Protects against math channels (which emit hundreds of names with no operator config) flooding the annotated image.
- `draw_detection_on()` now accepts both `obj_filters` and `audio_settings` parameters. All three call sites pass both and route the decision through `should_draw_class()` / `min_confidence_for()`.
- Tested first on vteam12 (test env). Math channels skipped, yolo classes drawn — no regressions in either direction.

## [3.21.20] - 2026-06-08

### Fixed — silence `decode_objects` log spam (DataMatrix decoder defensive guard)
- On PVB the DataMatrix decoder was logging `Error in decode_objects: string indices must be integers` thousands of times per minute. Inference and ejection were unaffected (the error sat inside a try/except, annotated images still saved with correct detection counts), but it filled `docker logs` and made real errors hard to spot.
- Root cause: `query_data` occasionally contains a non-dict entry (a stray string from upstream serialisation), and `obj["chars"]` then raises `TypeError`. Added a defensive `isinstance(obj, dict) and "chars" in obj` guard at `detection.py:923` so the loop just skips bad entries.
- One-line change. No behavioural impact on sites where the error was already absent.

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
