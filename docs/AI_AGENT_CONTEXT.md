# MVE Agent Context

Configuration context for AI agents. Optimized for token density — not human readability.
Companion: `docs/MATH_CHANNELS.md` (per-channel formulas).

## INVARIANTS

- All HTTP services on docker network `monitaqc_main_network`.
- Detection dict is the universal contract:
  `{name:str, confidence:float[0..1], class:int, xmin:int, ymin:int, xmax:int, ymax:int, lab_color?:[L,a,b], _<meta>?:any}`.
- Procedure rules target `name`. `_meta` fields are invisible to rules.
- Pipeline = ordered list of phases. Each phase = one model. All enabled phases run; outputs concatenated; metadata (`_cam`, encoder, shipment, capture_t) attached **after** by MVE.
- Inference workers are stateless. They receive `multipart image` only. They never see encoder/shipment/cam_id.
- Rules: `min_confidence` is `>=` only. No `<=`. For two-sided anomalies use one-sided channels (`band_delta_e`, `*_residual_max`, `tilt_from_*`).
- GPU reservation = soft. CPU host still starts containers. Workers auto-detect device.
- `math_inference` on `nvidia/cuda:*-runtime` base = no CUDA headers → CuPy NVRTC JIT fails on first ufunc → worker auto-falls-back to numpy. Force CPU via `MATH_DEVICE=cpu`. Unlock GPU = rebuild on `*-devel` base.

## CONTAINERS

| service | image | host port | role |
|---|---|---|---|
| `monitait_vision_engine` | built | 80 | orchestrator + web UI |
| `yolo_inference` (replicated) | built | 4442 (expose) | YOLOv5 worker |
| `math_inference` (replicated) | built | 4443 (expose) | math worker |
| `monitait_redis` | redis:7-alpine | 6379 (expose) | queues, color refs, timeline |
| `monitait_timescaledb` | timescale/timescaledb:latest-pg15 | 5432 (expose) | metrics |
| `monitait_grafana` | grafana/grafana:latest | 3000 | dashboards |
| `monitait_pigallery2` | bpatrik/pigallery2 | 5000 | image browser |

## SCHEMAS

### Detection (analyzer output)
```json
{"name":"<str>","confidence":0.0,"class":0,"xmin":0,"ymin":0,"xmax":0,"ymax":0}
```
Optional: `lab_color:[L,a,b]` (auto-attached by MVE if any procedure uses `color_delta`); `_<anything>` metadata.

### InferenceModel
```json
{"model_id":"<key>","name":"<label>","model_type":"yolo|gradio",
 "inference_url":"<http>","model_name":"<str>","confidence_threshold":0.0}
```
`model_type:"yolo"` = generic HTTP-multipart dispatcher (works for ANY analyzer that takes `files={image}` and returns detection list). `model_type:"gradio"` = HuggingFace Spaces.

### Pipeline
```json
{"name":"<str>","description":"<str>","enabled":true,
 "phases":[{"model_id":"<key>","enabled":true,"order":0}]}
```

### Procedure
```json
{"name":"<str>","enabled":true,"logic":"any|all","cameras":[],
 "rules":[{"object":"<name>","condition":"<cond>",
   "count":1,"area":10000,"max_delta_e":5.0,
   "reference_mode":"previous|running_avg|fixed",
   "min_confidence":0}]}
```
Conditions: `present | not_present | count_equals | count_greater | count_less | area_greater | area_less | area_equals | color_delta`.
`min_confidence` is `int 0..100`, applied as `det.confidence >= min_confidence/100`.
`area_*` uses bbox area of the **highest-confidence** matching detection only.
`color_delta` only valid for YOLO classes (LAB extracted from bbox crop).
`cameras:[]` = all; `[1,3]` = filter to those `_cam` ids.

### State (capture state machine)
```json
{"name":"<str>","steps":-1,"analog":-1,
 "phases":[{"light_mode":"U_ON_B_OFF|U_OFF_B_ON|U_ON_B_ON|U_OFF_B_OFF",
   "delay":50,"cameras":[1,2],"steps":-1,"analog":-1}]}
```
`steps:-1` = always capture (free-running). `analog:-1` = analog gate disabled.

## API (vision_engine, host port 80)

### Discovery — call these BEFORE making changes
| GET | purpose |
|---|---|
| `/api/pipelines/current` | active pipeline + current model |
| `/api/models` | all registered models |
| `/api/model_classes` | YOLO class names from currently-loaded weights |
| `/api/procedures` | all procedures |
| `/api/states` | all states |
| `/api/cameras` | configured cameras + IDs |
| `/api/inference/stats` | live FPS, latency, queue depths, autoscaler |
| `/api/timeline_config` | dashboard display config (object_filters, bbox toggle, …) |
| `/api/latest_detections` | most recent detection list (verify channel names before writing rules) |
| `http://math_inference:4443/v1/math-analysis/math_v1/channels` | math worker's advertised channel names |
| `http://math_inference:4443/v1/math-analysis/math_v1/health` | math worker device (`GPU`/`CPU`) |

### Models / pipelines
| method | path | body / notes |
|---|---|---|
| POST | `/api/models` | InferenceModel JSON; `model_id` required |
| DELETE | `/api/models/{id}` | id starting `default_` rejected |
| GET/POST | `/api/models/weights` GET / `/api/models/upload-weights` POST | list / multipart upload `.pt` |
| POST | `/api/models/activate-weights` | `{"weights_path":"/weights/best.pt"}` — hot-swaps every YOLO replica via `/v1/object-detection/yolov5s/set-model` |
| GET | `/api/gradio/models?url=<space>` | probe HuggingFace Space submodels |
| POST | `/api/pipelines` | Pipeline JSON |
| POST | `/api/pipelines/activate/{name}` | activate by name |
| DELETE | `/api/pipelines/{name}` | name `default` rejected |

### Procedures + color refs
| method | path | body / notes |
|---|---|---|
| POST | `/api/procedures` | `{"procedures":[...]}` — REPLACES the list |
| GET | `/api/color-reference/{class}?mode=fixed\|previous\|running_avg` | read |
| POST | `/api/color-reference/{class}` | `{"capture":true}` (use last detection) or `{"lab":[L,a,b]}` |

### States / cameras
| method | path | body / notes |
|---|---|---|
| POST | `/api/states` | State JSON |
| POST | `/api/states/{name}/activate` | activate |
| POST | `/api/states/trigger-capture` | manual one-shot |
| POST | `/api/cameras/rescan` | re-scan USB devices |
| POST | `/api/camera/{id}/config` | per-cam exposure/gain/brightness/contrast/saturation/fps/roi |
| POST | `/api/camera/{id}/restart` | reinit |
| GET | `/api/camera/{id}/snapshot` | one JPEG |
| GET | `/api/camera/{id}/stream` | MJPEG |
| POST | `/api/cameras/discover` | network discovery for IP cams |

### Timeline / display
| method | path | body / notes |
|---|---|---|
| GET | `/api/timeline_count` | `{total_frames,total_pages,...}` |
| GET | `/timeline_image?page=N` | rendered JPEG of dashboard grid |
| GET | `/api/timeline_frame?cam=N&col=K&page=P` | single full-res frame |
| POST | `/api/timeline_config` | merges with current; fields: `show_bounding_boxes` (bool), `object_filters` (`{<name>:{show:bool, min_confidence:float}}`), `camera_order` ("normal"/"reverse"/"custom"), `custom_camera_order` (csv), `image_quality` (int), `num_rows`, `buffer_size`, `image_rotation` (0/90/180/270), `procedures` |
| POST | `/api/timeline_clear` | wipe Redis timeline buffer |
| GET | `/detection_stream` | SSE of detection events |

### Persistence
| method | path | body / notes |
|---|---|---|
| GET/POST | `/api/data-file` | master DATA_FILE |
| POST | `/api/save_service_config` / `/api/save_data_file` | force-flush |
| GET | `/api/export_service_config` | download as JSON |
| POST | `/api/config` | update infra/serial/ejector/capture/image |

### AI assistant (LLM-backed)
| method | path | body / notes |
|---|---|---|
| POST | `/api/ai_config` | OpenAI-compatible LLM config |
| POST | `/api/ai_query` | natural language → answer + suggested config |

### Health / status
| method | path | body / notes |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/api/status` | full snapshot JSON |
| GET | `/api/status/stream` | SSE |
| GET | `/api/system/metrics` | CPU/RAM/disk/GPU |

### Worker endpoints (call from inside docker network)
- `POST http://yolo_inference:4442/v1/object-detection/yolov5s/detect/` — multipart `image` → detection list.
- `POST http://yolo_inference:4442/v1/object-detection/yolov5s/set-model` — `model_path=<path>` form.
- `GET  http://yolo_inference:4442/v1/object-detection/yolov5s/{health,classes}`.
- `POST http://math_inference:4443/v1/math-analysis/math_v1/detect` — multipart `image` → detection list.
- `GET  http://math_inference:4443/v1/math-analysis/math_v1/{health,channels}`.
- `POST http://math_inference:4443/v1/math-analysis/math_v1/set-config` — query params `tiles_x, tiles_y, bands, fft_top_k, flat_field`.

## MATH MODULE — channel families

63 channels. Detection per channel; bbox = region the measurement applies to. Full formulas in `docs/MATH_CHANNELS.md`.

| family | channels | catches | typical confidence map |
|---|---|---|---|
| A global | `mean_L, std_L, range_L, skew_L, L_p01, L_p05, L_p95, L_p99` | overall brightness/contrast | `L/100`, `min(σ/50,1)`, etc. |
| B quality | `saturation_fraction, sharpness_laplacian_var, sharpness_tenengrad, exposure_balance` | frame trustworthiness — gate other rules with these | direct or `min(x/cap,1)` |
| C 1D-fft-row | `fft_row_peak_{1,2,3}_{energy,period_px}` | weft bars / horizontal periodic | `peak/total` or `min(period/1000,1)` |
| D 1D-fft-col | `fft_col_peak_{1,2,3}_{energy,period_px}` | warp stripes / vertical periodic | same |
| E 2D-fft | `fft2d_peak_{1,2,3}_{energy,period_px,angle_deg,tilt_from_horizontal,tilt_from_vertical}` | periodic + orientation. `tilt_from_*` are one-sided rule-able | `tilt = min(|angle-target|,180-|angle-target|)/45` |
| F bands | `band_mean_L, band_std_L, band_delta_from_median, band_delta_e` × N bands | shade across width / cone-shading. `band_delta_*` are one-sided | `min(ΔE/20,1)` for delta_e |
| G residuals | `row_residual_{max,count}, col_residual_{max,count}` | single-line defects (broken cord/weft) | `min(ΔL/50,1)` and `min(count/100,1)` |
| H spacing | `row_spacing_anomaly, col_spacing_anomaly, row_spacing_std, col_spacing_std` | spacing irregularity (missing/collapsed) | `|gap/median-1|` |
| I gradient | `grad_mean, grad_max, grad_ori_coherence, grad_ori_dominant_deg, grad_ori_tilt_from_{horizontal,vertical}` | edge density, texture coherence, orientation | `min(g/cap,1)` |
| J morph | `tophat_max, tophat_mean, bothat_max, bothat_mean` | small bright/dark spots | `min(ΔL/50,1)` |
| K blobs | `blob_darkness, blob_brightness` (variable count, real bboxes) | localized defects with `area_greater` rules | `1-mean_L/100` or `mean_L/100` |

Optional tile family (when `MATH_TILES_X|Y > 1`): `tile_row_period_shift, tile_col_period_shift, tile_row_energy_excess, tile_col_energy_excess, tile_mean_L_shift, tile_std_L_shift` — one detection per anomalous tile.

### Defect → family mapping
```
shade across width            F (band_delta_e, band_delta_from_median)
periodic stripes              C/D/E
stripe orientation/tilt       E (fft2d_peak_K_tilt_from_*)
missing weft / collapsed pair H (row_spacing_anomaly)
broken cord (full-height)     G (col_residual_max)
broken weft (full-width)      G (row_residual_max)
small dark spot               J + K (bothat_max, blob_darkness)
small bright spot             J + K (tophat_max, blob_brightness)
sharp-edged (cuts, tears)     I + J (grad_max, tophat_max)
texture loss / wash-out       A (std_L) + I (grad_ori_coherence)
frame quality gate            B (sharpness_laplacian_var)
```

### math env vars
| var | default | effect |
|---|---|---|
| `MATH_DEVICE` | `auto` | `auto`/`cpu`/`cuda`. auto = try CuPy, fall back to numpy on probe failure |
| `MATH_TILES_X` | `1` | tile grid width — enables `tile_*` family if >1 |
| `MATH_TILES_Y` | `1` | tile grid height |
| `MATH_BANDS` | `8` | vertical bands for family F |
| `MATH_FFT_TOP_K` | `3` | FFT peaks per family C/D/E |
| `MATH_FLAT_FIELD_ENABLE` | `false` | divide by gaussian-blurred self before measuring |

## RECIPES (copy-paste-ready, replace `<host>`)

### Plug math worker into pipeline alongside YOLO
```bash
curl -sX POST http://<host>/api/models -H 'Content-Type: application/json' -d '{
"model_id":"math_v1","name":"Math Analyzer","model_type":"yolo",
"inference_url":"http://math_inference:4443/v1/math-analysis/math_v1/detect",
"model_name":"math_v1","confidence_threshold":0.0}'
curl -sX POST http://<host>/api/pipelines -H 'Content-Type: application/json' -d '{
"name":"yolo_plus_math","enabled":true,"phases":[
{"model_id":"default_yolo","enabled":true,"order":0},
{"model_id":"math_v1","enabled":true,"order":1}]}'
curl -sX POST http://<host>/api/pipelines/activate/yolo_plus_math
```

### Reject on shade anomaly
```bash
curl -sX POST http://<host>/api/procedures -H 'Content-Type: application/json' -d '{"procedures":[
{"name":"shade","enabled":true,"logic":"any","cameras":[],
 "rules":[{"object":"band_delta_e","condition":"count_greater","count":0,"min_confidence":25}]}]}'
```
`min_confidence:25` ⇔ `ΔE>5` (because confidence map = `min(ΔE/20,1)`).

### Reject hole > area, with confidence floor
```json
{"object":"hole","condition":"area_greater","area":800,"min_confidence":60}
```

### Combined defect (same frame)
```json
{"name":"combo","logic":"all","rules":[
 {"object":"fft_col_peak_1_energy","condition":"count_greater","count":0,"min_confidence":50},
 {"object":"band_delta_e","condition":"count_greater","count":0,"min_confidence":25}]}
```

### Hide noisy global-scalar channels from dashboard
```bash
# GET current first to merge:
curl -s http://<host>/api/timeline_config | jq '.object_filters += {
 "mean_L":{"show":false},"std_L":{"show":false},"range_L":{"show":false},"skew_L":{"show":false},
 "L_p01":{"show":false},"L_p05":{"show":false},"L_p95":{"show":false},"L_p99":{"show":false},
 "saturation_fraction":{"show":false},"sharpness_laplacian_var":{"show":false},
 "sharpness_tenengrad":{"show":false},"exposure_balance":{"show":false}}' | \
curl -sX POST http://<host>/api/timeline_config -H 'Content-Type: application/json' -d @-
```

### Upload + activate YOLO weights
```bash
curl -sX POST http://<host>/api/models/upload-weights -F "file=@/local/best.pt"
curl -sX POST http://<host>/api/models/activate-weights -H 'Content-Type: application/json' \
  -d '{"weights_path":"/weights/best.pt"}'
```

### Set fixed color reference
```bash
# from latest detection of class:
curl -sX POST http://<host>/api/color-reference/jean -H 'Content-Type: application/json' -d '{"capture":true}'
# explicit:
curl -sX POST http://<host>/api/color-reference/jean -H 'Content-Type: application/json' -d '{"lab":[40,1.5,-22]}'
```

### Manual capture
```bash
curl -sX POST http://<host>/api/states/trigger-capture
```

### New analyzer container template
```python
# main.py — must serve POST /<path>/detect, GET /<path>/health
from fastapi import FastAPI, File, UploadFile
import cv2, numpy as np
app = FastAPI()
@app.post("/v1/myanalyzer/detect")
async def detect(image: UploadFile = File(...)):
    bgr = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    return [{"name":"my_channel","confidence":0.42,"class":1100,
             "xmin":0,"ymin":0,"xmax":bgr.shape[1],"ymax":bgr.shape[0]}]
@app.get("/v1/myanalyzer/health")
async def h(): return {"status":"healthy"}
```
Compose service + register via `POST /api/models` + add to pipeline + activate.

## OPERATIONAL FACTS

- Frame queue: hot RAM LIFO (ejector decisions) + cold disk FIFO (history). No drop. Hot-spill on full → cold gets `allow_eject=False`.
- Inference workers autoscale: count = `max(8, min(cpu_logical*2, 32))`. See `/api/inference/stats.autoscaler`.
- Ejector flow: rules fire → push `{"encoder":E,"dm":D}` to Redis list `ejector_queue` → serial thread fires actuator at `E + EJECTOR_OFFSET` after `EJECTOR_DELAY` ms for `EJECTOR_DURATION` ms.
- Fail-safe: YOLO failure on hot path after retry → fail-safe eject pushed.
- DVR ring buffer: `raw_images/` capped at 75% disk usage. Oldest hourly chunks deleted. Current hour never deleted.
- Color refs in Redis: `color_ref:<class>:{previous,running_avg_list,fixed}`. Running avg window = 20.
- Timeline buffer in Redis: `timeline:<cam_id>` — pickled `(ts,jpeg,detections,meta)` tuples, capped by `timeline_config.buffer_size` (default 100).
- `VERSION` file mounted at `/code/VERSION:ro`. Dashboard reads on every page load.

## FAILURE MODES → DIAGNOSTIC

| symptom | likely cause | check |
|---|---|---|
| `service_type` shows old pipeline name after activate | `set_current_pipeline` returned 404 (typo) | `GET /api/pipelines` — verify name |
| `device:CPU` on math after expected GPU | NVRTC JIT failed — runtime base lacks headers | math container logs: `math worker: CPU backend (numpy) — reason:` |
| Procedures don't fire on visible defect | wrong `name` (typo), or `min_confidence` too high | `GET /api/latest_detections` — confirm `name`, `confidence` of real detections |
| `inference_fps:0`, `capture_fps>0` | worker URL unreachable | `docker logs monitaqc-yolo_inference-1`, check network alias resolves |
| Eject queued but actuator silent | encoder offset wrong, or actuator wiring | `[EJ_FIRE]` log lines in vision_engine; hardware test in UI |
| Bbox labels off-canvas | pre-v3.12.1 | upgrade |
| Dashboard "No frames yet" but `/api/timeline_count > 0` | rendering route mismatch | hit `/timeline_image` directly; check vision_engine logs |
| Math request times out > 30s | tiles too high or flat-field on slow CPU | reduce `MATH_TILES_X*Y`, set `MATH_FLAT_FIELD_ENABLE=false` |

## PRE-FLIGHT BEFORE WRITING RULES

1. `GET /api/latest_detections` — confirm real detections include the `name` you'll target.
2. `GET http://math_inference:4443/v1/math-analysis/math_v1/channels` — confirm channel exists.
3. `GET /api/model_classes` — confirm YOLO class exists.
4. `GET /api/cameras` — confirm camera IDs you'll filter on.
5. `GET /api/procedures` — read existing rules; you typically REPLACE the full list when posting.

## CROSS-REFS
- Per-channel formulas: `docs/MATH_CHANNELS.md`
- Operator guide: `akamod/EJECTION_PROCEDURES_GUIDE.md`
- Brochure / use cases: `docs/BROCHURE.md`
- User manual: `docs/USER_MANUAL.md`
- Changelog: `CHANGELOG.md`
