# MVE — AI Agent Configuration Context

**Purpose.** Attach this file to an AI assistant prompt when you want it to
configure or reason about a Monitait Vision Engine deployment. It contains the
component map, conceptual model, the full HTTP API surface, the analyzer
catalog, and worked examples of every common configuration task.

**Audience.** Both AI agents and the humans onboarding them. Self-contained —
no other docs required, though [`MATH_CHANNELS.md`](./MATH_CHANNELS.md) is the
deeper reference for the math module's 63 channels.

---

## 1. System architecture in one diagram

```
              ┌──────────────────────────────────────────────────────┐
              │                  monitait_vision_engine              │
              │                  (FastAPI, port 80)                  │
              │                                                      │
hardware  ───►│ • serial parser (encoder, status bits)               │
(serial,      │ • capture orchestrator (cameras × phases × lights)   │
 cameras,     │ • inference pipeline manager (sequence of models)    │
 barcode,     │ • procedure evaluator (rule engine → eject decision) │
 ejector)     │ • timeline buffer (Redis-backed)                     │
              │ • TimescaleDB writer (production metrics)            │
              │ • web UI (Dashboard / Inference / Procedures / etc.) │
              └──────────────────────────────────────────────────────┘
                       │                │                │
            HTTP POST  │      HTTP POST │     SQL writes │
            multipart  │      multipart │                │
                       ▼                ▼                ▼
              ┌─────────────┐  ┌────────────────┐  ┌────────────┐
              │yolo_inference│  │math_inference  │  │timescaledb │
              │(port 4442)   │  │(port 4443)     │  │(port 5432) │
              │PyTorch + GPU │  │CuPy/numpy      │  │PostgreSQL  │
              └─────────────┘  └────────────────┘  └────────────┘

                       redis (port 6379)  ◄──── all services
                       grafana (port 3000) ──── reads timescaledb
                       pigallery2 (port 5000) ── browses raw frames
```

### Containers (docker-compose.yml + docker-compose.math.yml)

| Container | Image | Role |
|---|---|---|
| `monitait_vision_engine` | `monitait/mve` (built from `vision_engine/`) | Orchestrator + web UI on host port 80 |
| `monitaqc-yolo_inference-N` | built from `yolo_inference/` | YOLOv5 worker, port 4442 (GPU if reservation honored) |
| `monitaqc-math_inference-N` | built from `math_inference/` | Math worker, port 4443 (GPU-soft) |
| `monitait_redis` | `redis:7-alpine` | Cache, pub/sub, timeline buffer, color refs |
| `monitait_timescaledb` | `timescale/timescaledb:latest-pg15` | Production metrics + inference history |
| `monitait_grafana` | `grafana/grafana:latest` | Dashboards on port 3000 |
| `monitait_pigallery2` | `bpatrik/pigallery2:latest` | Captured-image browser on port 5000 |

All on docker network `monitaqc_main_network`. GPU reservations are *soft* — a
host with no GPU still starts everything (workers fall back to CPU).

---

## 2. The conceptual model — five things that are configurable

In order of evaluation, every captured frame goes through:

1. **State** — *when* to capture and *which* cameras / lighting / phases to use.
2. **Pipeline** — an ordered list of inference *phases*; each phase invokes one
   *model*.
3. **Models** — addressable inference workers (YOLO, math, Gradio cloud, …).
   Each is just an HTTP endpoint that takes an image and returns a list of
   detection dicts.
4. **Procedures** — the rule engine that turns the merged detection list into
   an *eject* / *don't eject* decision.
5. **Object filters / timeline config** — *display-only* overrides for how the
   dashboard renders frames (which class names show their bbox, what counts
   as "low confidence" for display).

Detection dict schema (every analyzer must produce this):

```json
{
  "name":       "string",   // class or channel name — what rules target
  "confidence": 0.0..1.0,
  "class":      <int>,      // numeric class id (informational only)
  "xmin": <px>, "ymin": <px>, "xmax": <px>, "ymax": <px>,
  "lab_color":  [L*, a*, b*],   // optional, only when a color_delta procedure exists
  "_<anything>": ...        // optional metadata (not visible to rules)
}
```

---

## 3. YOLO inference — how the model side works

### What it is
- Python service in `yolo_inference/` running `ultralytics/yolov5:v7.0` base.
- Endpoint: `POST /v1/object-detection/yolov5s/detect/` with multipart `image`.
- Returns: JSON list of detections in the schema above.
- `name` field comes from the loaded weights' `model.names`. Operator rules
  target these names directly (`hole`, `stain`, `knot`, etc.).

### Per-frame contract
- MVE's pipeline manager calls the endpoint with raw JPEG bytes.
- MVE attaches metadata (camera id, encoder, shipment, capture timestamp)
  *after* the call returns — the worker is stateless and never sees them.
- If a `color_delta` procedure exists, MVE additionally extracts a CIE L\*a\*b\*
  color from each YOLO bbox crop and stores it in `det['lab_color']`.

### Replacing the model
Two ways:

1. **Upload a `.pt` weights file** via the web UI ("Inference → Upload Weights")
   or API:
   ```
   POST /api/models/upload-weights        (multipart, file=<my_model.pt>)
   POST /api/models/activate-weights      (json, {"weights_path": "/weights/my_model.pt"})
   ```
   This routes through every YOLO replica's `/v1/object-detection/yolov5s/set-model`
   so the model is hot-swapped with no container restart.

2. **Point at a different inference URL** (e.g., a HuggingFace Gradio space) by
   adding a new model with `model_type: "gradio"`:
   ```json
   POST /api/models
   {
     "model_id": "hf_defects",
     "name": "HuggingFace Defects",
     "model_type": "gradio",
     "inference_url": "https://<space>.hf.space",
     "model_name": "Data Matrix",
     "confidence_threshold": 0.25
   }
   ```

### Class management
- `GET /api/model_classes` returns the currently loaded class list (asks every
  YOLO replica and intersects).
- Rules reference class names (`name` field). Renaming classes in the weights
  requires updating every rule that references them.

---

## 4. Math module — how the analyzer side works

Math is *not* trained — every channel is a fixed mathematical measurement
mapped to a 0..1 confidence via a universal absolute formula. Run the worker
once with no rules to learn what good fabric looks like, then write rules.

### Endpoint contract — same shape as YOLO
- `POST /v1/math-analysis/math_v1/detect` (multipart `image`)
- `GET  /v1/math-analysis/math_v1/health`
- `GET  /v1/math-analysis/math_v1/channels` — list of all channel names this
  worker emits, useful for AI agents auto-discovering rule targets.
- `POST /v1/math-analysis/math_v1/set-config` — runtime tweaks (tile grid,
  bands, FFT top-k) without restart.

### 63 channels in 11 families

| # | Family | Channels | One-line purpose | Detail |
|---|---|---|---|---|
| A | Global stats | 8 | Whole-frame brightness/contrast/percentiles | mean/std/range/skew + 4 percentile points |
| B | Quality meta | 4 | Image trustworthiness — focus, saturation, exposure | Use as gates in front of other rules |
| C | 1D FFT row peaks | 6 | Horizontal periodic patterns (weft bars) | top-3 × {energy, period_px} |
| D | 1D FFT col peaks | 6 | Vertical periodic patterns (warp stripes) | same |
| E | 2D FFT peaks | 15 | Periodic patterns + orientation | top-3 × {energy, period, angle, tilt-h, tilt-v} |
| F | Band stats | 4 (× N bands) | Shade analysis across the fabric width | one detection per band |
| G | Row/col residuals | 4 | Single-line defects (broken cords, weft bars) | non-periodic, FFT misses these |
| H | Spacing anomalies | 4 | Irregular spacing (missing weft, collapsed pair) | FFT misses these too |
| I | Gradient / orientation | 6 | Edge density + texture coherence | catches cuts, tears, scrambled texture |
| J | Morphology (top/bot-hat) | 4 | Small bright/dark spot enhancers | cheap (~3 ms) |
| K | Blobs | 2 (variable count) | Localized dark/bright defects with real bboxes | feeds `area_greater` rules naturally |

Full channel-by-channel reference (formulas, ranges, defect coverage,
recipes): [`MATH_CHANNELS.md`](./MATH_CHANNELS.md).

### Defect → family quick map

```
shade drift across width      →  F (band_delta_e, band_delta_from_median)
periodic stripes              →  C/D/E (FFT)
stripe orientation/tilt       →  E (fft2d_peak_K_tilt_from_*)
missing weft / collapsed pair →  H (row_spacing_anomaly)
broken cord (full-height)     →  G (col_residual_max)
broken weft (full-width bar)  →  G (row_residual_max)
small spots (oil, foreign)    →  J + K (bothat_max, blob_darkness)
sharp-edged (cuts, tears)     →  I + J (grad_max, tophat_max)
texture loss / wash-out       →  A (std_L) + I (grad_ori_coherence)
"is the frame even good?"     →  B (sharpness_laplacian_var)
```

### Tunable env vars on `math_inference`

| Var | Default | Effect |
|---|---|---|
| `MATH_TILES_X` | `1` | If > 1, also emit `tile_*_shift` channels (localized frequency anomaly) |
| `MATH_TILES_Y` | `1` | Same for height |
| `MATH_BANDS` | `8` | Vertical bands across width for family F |
| `MATH_FFT_TOP_K` | `3` | Number of FFT peaks per family C/D/E |
| `MATH_FLAT_FIELD_ENABLE` | `false` | Pre-divide by low-pass to remove vignette/hotspot |
| `MATH_DEVICE` | `auto` | `auto`/`cpu`/`cuda`. `auto` falls back to numpy if CuPy can talk to the GPU but JIT is broken (e.g. `nvidia/cuda:*-runtime` base lacks headers). |

---

## 5. Procedures — the rule engine

A **procedure** is a named, enabled bag of **rules** that MVE evaluates against
every frame's merged detections. If the rules fire, MVE pushes
`{"encoder": ..., "dm": ...}` to the `ejector_queue` Redis list, and the
ejector thread fires the actuator after the configured offset.

### Procedure JSON schema

```json
{
  "name":     "human-readable",
  "enabled":  true,
  "logic":    "any" | "all",
  "cameras":  [1, 2],            // empty = all cameras
  "rules": [
    {
      "object":         "<channel or class name>",
      "condition":      "present | not_present | count_equals | count_greater
                       | count_less | area_greater | area_less | area_equals
                       | color_delta",
      "count":          1,            // for count_*
      "area":           10000,        // for area_* (px²)
      "max_delta_e":    5.0,          // for color_delta
      "reference_mode": "previous | running_avg | fixed",
      "min_confidence": 70            // 0..100; detections below this are ignored
    }
  ]
}
```

### Conditions

| Condition | Fires when | Notes |
|---|---|---|
| `present` | ≥ 1 matching detection | shorthand for count_greater 0 |
| `not_present` | 0 matching detections | use for "must be there" gates inverted |
| `count_equals` | exactly N | strict |
| `count_greater` | > N | use `0` to mean "≥ 1" |
| `count_less` | < N | useful for "must be at least N" |
| `area_greater/less/equals` | bbox area of highest-conf det vs. threshold | px², highest-confidence detection only |
| `color_delta` | ΔE of bbox-LAB vs. reference > threshold | only on YOLO classes; MVE auto-extracts LAB |

### `min_confidence` is one-sided
Always `≥`. To express "value too LOW" (frame too dark, coherence too low),
either pair an inverted channel or use a one-sided anomaly channel like
`band_delta_e` that's already high-when-anomalous in either direction.

### Reference modes (color_delta)
- `previous` — last product of same class
- `running_avg` — rolling mean of last 20
- `fixed` — golden sample captured via UI ("Set Fixed Reference") or
  `POST /api/color-reference/<class>` body `{"capture": true}` or `{"lab": [L,a,b]}`

### Worked recipes

| Intent | Procedure |
|---|---|
| Reject any hole | `{name:"holes", logic:"any", rules:[{object:"hole", condition:"present", min_confidence:60}]}` |
| Reject hole > 2 cm² (1 mm/px) | `area_greater 200 min_confidence 60` |
| Reject any band shade anomaly | `band_delta_e count_greater 0 min_confidence 25` |
| Reject 3+ bands anomalous | `band_delta_e count_greater 2 min_confidence 20` |
| Reject vertical line defect | `col_residual_max count_greater 0 min_confidence 50` |
| Reject dark spot ≥ 4×4 mm | `blob_darkness area_greater 16 min_confidence 50` |
| Color drift vs golden | `{object:"jean", condition:"color_delta", max_delta_e:3.5, reference_mode:"fixed", min_confidence:50}` |
| Combined defect (logic:all) | two rules: vertical stripe + dark band, both must fire same frame |
| Camera 2 only | add `"cameras":[2]` |

---

## 6. States & cameras — the capture side

A **state** describes a sequence of capture *phases*. Each phase says "set
these lights, wait this many ms, then capture from these cameras". States are
named; the system runs the **active** state in a loop, gated by encoder pulses
or a free-running infinite loop.

### State JSON

```json
{
  "name":      "infinite",
  "steps":    -1,            // encoder steps between captures; -1 = always capture
  "analog":   -1,            // analog threshold; -1 = disabled
  "phases": [
    {"light_mode": "U_ON_B_OFF", "delay": 50, "cameras": [1, 2], "steps": -1, "analog": -1},
    {"light_mode": "U_OFF_B_ON", "delay": 50, "cameras": [1, 2], "steps": -1, "analog": -1}
  ]
}
```

Light modes: `U_ON_B_OFF` (uplight only), `U_OFF_B_ON` (backlight only),
`U_ON_B_ON`, `U_OFF_B_OFF`. PWM brightness is set separately per
`set_PWM_uplight` / `set_PWM_backlight` via the Hardware tab.

Cameras are 1-indexed and can be USB (`/dev/video*`) or RTSP/IP. They're
configured via `/api/cameras/*` (per-cam exposure/gain/brightness/ROI/FPS).

---

## 7. Full HTTP API reference

All endpoints are on the `monitait_vision_engine` container (host port 80).
JSON in/out unless noted.

### Health & status
| Method | Path | Use |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/status` | Web UI page (HTML) |
| GET | `/api/status` | JSON status snapshot |
| GET | `/api/status/stream` | SSE stream of live status |
| GET | `/api/system/metrics` | CPU/RAM/disk/GPU stats |
| GET | `/api/inference/stats` | Live FPS, latency, queue depths |

### Inference models & pipelines
| Method | Path | Use |
|---|---|---|
| GET | `/api/models` | List all registered models |
| POST | `/api/models` | Create or update a model (body: `{model_id, name, model_type, inference_url, model_name, confidence_threshold}`) |
| DELETE | `/api/models/{model_id}` | Remove |
| GET | `/api/model_classes` | Class names from the loaded YOLO weights |
| GET | `/api/models/weights` | List `.pt` files in `volumes/weights/` |
| POST | `/api/models/upload-weights` | multipart upload |
| POST | `/api/models/activate-weights` | swap loaded weights on every YOLO replica |
| GET | `/api/gradio/models` | Probe Gradio space for available submodels |
| GET | `/api/pipelines` | All pipelines + models (status dump) |
| GET | `/api/pipelines/current` | Active pipeline |
| POST | `/api/pipelines` | Create or update |
| POST | `/api/pipelines/activate/{name}` | Activate by name |
| DELETE | `/api/pipelines/{name}` | Remove |

### Procedures (ejection rules)
| Method | Path | Use |
|---|---|---|
| GET | `/api/procedures` | All procedures |
| POST | `/api/procedures` | Replace the list (body: `{procedures: [...]}`) |
| GET | `/api/color-reference/{class}` | Get LAB ref (`?mode=fixed|previous|running_avg`) |
| POST | `/api/color-reference/{class}` | Set fixed LAB (`{lab:[L,a,b]}` or `{capture:true}`) |

### States (capture state machine)
| Method | Path | Use |
|---|---|---|
| GET | `/api/states` | List all states |
| GET | `/api/states/{name}` | Single state |
| POST | `/api/states` | Create or update |
| DELETE | `/api/states/{name}` | Remove |
| POST | `/api/states/{name}/activate` | Activate |
| POST | `/api/states/trigger-capture` | Manually trigger one capture cycle |
| POST | `/api/states/save` / `/load` | Persist or reload from disk |

### Cameras
| Method | Path | Use |
|---|---|---|
| GET | `/video_feed` | MJPEG stream |
| GET | `/video_feed_detections` | MJPEG stream with bboxes |
| GET | `/api/cameras` | List configured cameras |
| POST | `/api/cameras/rescan` | Re-scan USB devices, hot-plug aware |
| POST | `/api/camera/{id}/restart` | Reinit one camera |
| POST | `/api/camera/{id}/config` | Set per-cam exposure/gain/etc |
| GET | `/api/camera/{id}/snapshot` | One-shot JPEG |
| GET | `/api/camera/{id}/stream` | Single-camera MJPEG |
| POST | `/api/cameras/discover` | Network discovery for IP cameras |
| POST | `/api/cameras/test` | Test a camera URL/path |
| POST | `/api/cameras/save` / `/load` / `/upload` | Persist camera configs |

### Timeline & frames
| Method | Path | Use |
|---|---|---|
| GET | `/timeline_image` | Dashboard's main timeline grid (paginated JPEG) |
| GET | `/timeline_feed` | MJPEG of the live timeline |
| GET | `/api/timeline_count` | Total frames + page count |
| GET | `/api/timeline_config` | Display config (bbox toggle, object filters, …) |
| POST | `/api/timeline_config` | Update |
| POST | `/api/timeline_clear` | Wipe Redis timeline buffer |
| GET | `/api/timeline_frame?cam=N&col=K&page=P` | Full-res rendered frame |
| GET | `/api/timeline_meta` | Per-column metadata across cameras |
| GET | `/api/raw_image/{path}` | Serve a raw_images file |
| GET | `/api/latest_detections` | Latest detection list |
| GET | `/latest_detection_image` | Latest annotated frame |
| GET | `/detection_stream` | SSE stream of detection events |
| GET | `/timeline_slideshow` | Slideshow viewer page |

### Persistence & config
| Method | Path | Use |
|---|---|---|
| GET | `/config` | Web UI config page (HTML) |
| POST | `/api/config` | Update infra/serial/ejector/capture/image config |
| GET | `/api/data-file` / POST | Read/write the master DATA_FILE |
| POST | `/api/save_service_config` / `/api/load_service_config` | Service config persistence |
| GET | `/api/export_service_config` | Download as JSON |

### AI assistant (LLM-powered config helper)
| Method | Path | Use |
|---|---|---|
| GET | `/api/ai_config` | List configured LLM endpoints |
| POST | `/api/ai_config` | Add/update an LLM config (OpenAI-compatible) |
| POST | `/api/ai_config/activate` | Activate |
| DELETE | `/api/ai_config/{name}` | Remove |
| POST | `/api/ai_query` | Natural-language query → answer + suggested config |
| GET | `/api/db_config` | LLM-context DB profiles |
| POST | `/api/db_config` / `/activate` / DELETE | CRUD |

---

## 8. Common configuration tasks — copy-paste recipes for AI agents

### A. Add the math worker as a pipeline phase next to YOLO

```bash
# 1) Register the math model
curl -s -X POST http://<host>/api/models -H 'Content-Type: application/json' -d '{
  "model_id": "math_v1",
  "name": "Math Analyzer",
  "model_type": "yolo",
  "inference_url": "http://math_inference:4443/v1/math-analysis/math_v1/detect",
  "model_name": "math_v1",
  "confidence_threshold": 0.0
}'

# 2) Create a pipeline with both phases
curl -s -X POST http://<host>/api/pipelines -H 'Content-Type: application/json' -d '{
  "name": "yolo_plus_math",
  "description": "YOLO defects + 63 math channels",
  "enabled": true,
  "phases": [
    {"model_id": "default_yolo", "enabled": true, "order": 0},
    {"model_id": "math_v1",      "enabled": true, "order": 1}
  ]
}'

# 3) Activate it
curl -s -X POST http://<host>/api/pipelines/activate/yolo_plus_math
```

`model_type: "yolo"` is the dispatcher used for any plain HTTP-multipart
analyzer — it's not specific to YOLO weights.

### B. Add a procedure that rejects on shade anomaly

```bash
curl -s -X POST http://<host>/api/procedures -H 'Content-Type: application/json' -d '{
  "procedures": [
    {
      "name": "Cross-direction shade anomaly",
      "enabled": true,
      "logic": "any",
      "cameras": [],
      "rules": [
        {"object": "band_delta_e", "condition": "count_greater",
         "count": 0, "min_confidence": 25}
      ]
    }
  ]
}'
```

`min_confidence: 25` corresponds to ΔE > 5 (because the channel maps
`min(ΔE/20, 1)` → confidence). Match the threshold to the SKU.

### C. Reject by YOLO class with size and confidence floor

```json
{
  "name": "Reject big holes only",
  "enabled": true,
  "logic": "any",
  "cameras": [],
  "rules": [
    {"object": "hole", "condition": "area_greater",
     "area": 800, "min_confidence": 60}
  ]
}
```

### D. Combined defect (must fire in same frame)

```json
{
  "name": "Vertical stripe + dark band",
  "logic": "all",
  "rules": [
    {"object": "fft_col_peak_1_energy", "condition": "count_greater",
     "count": 0, "min_confidence": 50},
    {"object": "band_delta_e", "condition": "count_greater",
     "count": 0, "min_confidence": 25}
  ]
}
```

### E. Hide noisy global-scalar channels from the dashboard view

Many math channels (mean_L, std_L, etc.) emit one full-frame bbox each, which
clutters the timeline thumbnails. They still drive procedures — just don't
draw them.

```bash
curl -s -X POST http://<host>/api/timeline_config -H 'Content-Type: application/json' -d '{
  "show_bounding_boxes": true,
  "object_filters": {
    "mean_L":               {"show": false},
    "std_L":                {"show": false},
    "range_L":              {"show": false},
    "skew_L":               {"show": false},
    "L_p01":                {"show": false},
    "L_p05":                {"show": false},
    "L_p95":                {"show": false},
    "L_p99":                {"show": false},
    "saturation_fraction":  {"show": false},
    "sharpness_laplacian_var": {"show": false},
    "sharpness_tenengrad":  {"show": false},
    "exposure_balance":     {"show": false}
  }
}'
```

The other config fields the POST handler honors (camera_order, image_quality,
num_rows, buffer_size, image_rotation, procedures) inherit current values when
omitted — but it's safer to GET the current config first and merge.

### F. Upload custom YOLO weights and activate

```bash
curl -s -X POST http://<host>/api/models/upload-weights \
     -F "file=@/local/path/best.pt"
# Response includes the saved path under /weights/

curl -s -X POST http://<host>/api/models/activate-weights -H 'Content-Type: application/json' \
     -d '{"weights_path": "/weights/best.pt"}'
```

The activation routes through every YOLO replica's `/set-model` endpoint and
reloads `model.names` — `GET /api/model_classes` returns the new list.

### G. Set a fixed color reference (golden sample)

```bash
# Capture from the most recent detection of the class:
curl -s -X POST http://<host>/api/color-reference/jean -H 'Content-Type: application/json' \
     -d '{"capture": true}'

# Or set explicitly:
curl -s -X POST http://<host>/api/color-reference/jean -H 'Content-Type: application/json' \
     -d '{"lab": [40.0, 1.5, -22.0]}'
```

Then any `color_delta` rule with `reference_mode: "fixed"` on `jean` uses it.

### H. Trigger a manual capture (no encoder needed)

```bash
curl -s -X POST http://<host>/api/states/trigger-capture
```

Useful for testing pipelines on demand without waiting for hardware pulses.

### I. Add a new analyzer container (any HTTP service)

Any service that accepts `multipart/form-data` with field `image` and returns
a JSON list of detection dicts can be plugged in. Minimum image:

```python
# main.py
from fastapi import FastAPI, File, UploadFile
import cv2, numpy as np

app = FastAPI()

@app.post("/v1/myanalyzer/detect")
async def detect(image: UploadFile = File(...)):
    bgr = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    # ... your math ...
    return [
        {"name": "my_channel", "confidence": 0.42, "class": 1100,
         "xmin": 0, "ymin": 0, "xmax": bgr.shape[1], "ymax": bgr.shape[0]}
    ]

@app.get("/v1/myanalyzer/health")
async def health():
    return {"status": "healthy"}
```

Add to compose, register via `POST /api/models`, add to a pipeline, activate.

---

## 9. Operational behaviors AI agents should know about

- **Frame queue is two-tier.** Hot RAM queue (LIFO, ejector-critical) + cold
  disk queue (FIFO, history). No frame is ever dropped. If hot is full, frames
  spill to cold — inference still runs but ejection decisions are skipped for
  cold-replayed frames.
- **Inference workers auto-scale.** Default count is `max(8, min(cpu_logical*2, 32))`.
  Dynamic up/down based on hot-queue depth. See `/api/inference/stats →
  autoscaler`.
- **Ejector is encoder-based.** Eject decision pushes
  `{"encoder": E, "dm": D}` to Redis list `ejector_queue`. The serial thread
  watches encoder advance and fires the actuator at `E + EJECTOR_OFFSET` with
  `EJECTOR_DELAY` lead time and `EJECTOR_DURATION` pulse. Fail-safe: if YOLO
  fails after retry on the hot path, MVE issues a fail-safe eject.
- **DVR-style ring buffer.** When disk usage > 75 %, oldest hourly chunks of
  `raw_images/` are deleted (never current hour). Set by `_DISK_MAX_PCT` in
  `services/watcher.py`. Capture never stops.
- **Color references live in Redis** under `color_ref:<class>:{previous, running_avg_list, fixed}`.
  Running average window is 20 (`_COLOR_RUNNING_AVG_SIZE`).
- **Timeline buffer in Redis** under `timeline:<cam_id>` — pickled
  `(ts, jpeg_bytes, detections, meta)` tuples, length capped by
  `timeline_config.buffer_size` (default 100).
- **Soft GPU reservation** — `deploy.resources.reservations.devices`. Hosts
  with no GPU still start the container; framework auto-detects (PyTorch /
  CuPy) and falls back to CPU.
- **Math GPU caveat** — `math_inference` built `FROM nvidia/cuda:*-runtime` is
  missing the CUDA headers needed for CuPy NVRTC JIT (e.g. `vector_types.h`).
  The worker probes JIT at startup and falls back to numpy if it errors.
  To force CPU: `MATH_DEVICE=cpu`. To unlock GPU: rebuild on `*-devel` base.

---

## 10. Failure modes and how to read them

| Symptom | Likely cause | Where to look |
|---|---|---|
| Dashboard says "No frames yet" but `/api/timeline_count > 0` | Browser cached an old endpoint URL or rendering route was renamed | `vision_engine` logs, `/timeline_image` directly |
| Inference FPS = 0, capture FPS > 0 | Inference worker not reaching out to model URLs | `docker logs monitaqc-yolo_inference-1` and `…math_inference-1` |
| `device: GPU` flips to `device: CPU` after restart | NVRTC JIT failed; auto-fallback engaged | math container logs for "math worker: CPU backend (numpy) — reason:" |
| Procedures don't fire on visible defect | `min_confidence` too high, or rule targeting wrong `name` | `GET /api/latest_detections` to see actual names + confidences flowing |
| Eject command issued but actuator never fires | Encoder offset wrong, or actuator wiring | `[EJ_FIRE]` log lines in vision_engine, hardware test buttons in UI |
| Timeline labels off-canvas | Pre-v3.12.1 — full-frame bbox labels drawn at y < 0 | upgrade to ≥ v3.12.1 |
| Heavy CPU during shade analysis | `math_inference` running on CPU because of NVRTC | switch to `*-devel` base or accept ~4 FPS per replica |

---

## 11. What an AI agent should verify before applying changes

1. **Live state** — `GET /api/pipelines/current`, `/api/procedures`, `/api/states`.
2. **Available channels** — for the math worker, `GET http://math_inference:4443/v1/math-analysis/math_v1/channels`. Don't write rules against names not in the list.
3. **Available YOLO classes** — `GET /api/model_classes`.
4. **Camera count and IDs** — `GET /api/cameras`. Use these in `procedure.cameras` filters.
5. **Recent detections** — `GET /api/latest_detections` to confirm rules will see what you expect *before* enabling eject.
6. **Save to disk** — most CRUD endpoints auto-persist to `DATA_FILE`. If
   you're scripting bulk changes, finish with `POST /api/save_service_config`
   or `POST /api/save_data_file` to be safe.

---

## 12. Versioning & change history

- `VERSION` file at repo root is the source of truth, mounted into the
  vision_engine container at `/code/VERSION:ro`. The dashboard reads it on
  every page load.
- See [`CHANGELOG.md`](../CHANGELOG.md) for release notes.

---

## 13. Cross-references

- Math channel formulas, ranges, defect coverage, recipes →
  [`MATH_CHANNELS.md`](./MATH_CHANNELS.md)
- Operator-facing ejection rules guide (one-shot pastable) → 
  [`../../akamod/EJECTION_PROCEDURES_GUIDE.md`](../../akamod/EJECTION_PROCEDURES_GUIDE.md)
  (customer-specific path, may be moved into this docs/ folder later)
- Sales / system overview → [`BROCHURE.md`](./BROCHURE.md)
- User manual (humans, web UI walkthrough) → [`USER_MANUAL.md`](./USER_MANUAL.md)
- How-to guides (specific tasks) → [`HOW-TO.md`](./HOW-TO.md)
