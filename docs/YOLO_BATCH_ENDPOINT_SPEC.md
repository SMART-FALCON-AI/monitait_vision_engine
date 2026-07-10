# YOLO batch inference endpoint — contract for MVE v4.0.82+

Written for the team maintaining the `yolo_inference` container image. MVE will auto-detect and use batching if the container implements this contract; otherwise it falls back to the current single-image `/detect` path with zero regression.

## Why we're adding this

- MVE's single-image `POST /detect` path tops out around **85 fps** on the current vteam12 GPU because most of a YOLO forward pass is kernel-launch overhead, not per-pixel work.
- Batching 8–16 images into a single forward pass typically reaches **300–500 fps** on the same GPU.
- Our production target is **240 fps sustained** end-to-end. Batching is the biggest single lever.
- MVE dispatches from up to 24 concurrent worker threads today; concurrency alone doesn't help unless the inference service can vectorize the batch.

## What MVE will do (already shipped as v4.0.82, dark until this endpoint exists)

- Probes `GET <host>/capabilities` at startup and every 60 s.
- If it sees `{"batch": true, "max_batch": N}`, MVE starts calling `POST <host>/<inference_path>/batch-detect/` with multipart batches whenever the hot queue is stacking up (queue depth ≥ 8 with hysteresis; drops back to single when depth ≤ 2).
- If probe fails or returns `batch: false`, MVE keeps using `POST /detect` per image. No behavior change.

Fully backward compatible — nothing you ship breaks anything until MVE probes success.

---

## Contract

### 1) `GET /capabilities`

Purpose: let MVE discover batching support and the safe upper bound.

**Request:** no body.

**Response 200 (batch-capable server):**

```json
{
  "batch": true,
  "max_batch": 16
}
```

- `batch` (boolean, required) — `true` to opt in. `false` or field absent → MVE stays on single-image path.
- `max_batch` (integer, required when `batch=true`) — largest N you'll accept in one `POST /batch-detect`. Pick based on GPU memory; typical range 8–32. MVE will chunk larger requests to this cap; **do not** overrun.

**Response 200 (server that doesn't support batching):**

```json
{ "batch": false }
```

or return **404** to `GET /capabilities` — MVE treats both as "not batch-capable, use single-image."

**Timeout:** MVE gives the probe 2 s. Any timeout / 5xx → treated as `batch: false` until the next probe cycle.

### 2) `POST <inference_path>/batch-detect/`

Where `<inference_path>` is the same prefix as your existing `POST /detect/`. For example, if MVE is configured with `inference_url = http://yolo:4442/v1/object-detection/yolov5s/detect/`, MVE derives `batch-detect` URL by substituting the trailing `/detect/` → `/batch-detect/`. Same host, same path base, only the last segment changes.

**Request:** multipart/form-data with N `image` parts, one per input image. `N` will be between 2 and the `max_batch` you advertised.

```
POST /v1/object-detection/yolov5s/batch-detect/ HTTP/1.1
Content-Type: multipart/form-data; boundary=…

--…
Content-Disposition: form-data; name="image"; filename="img0.jpg"
Content-Type: image/jpeg

<binary JPEG 0>
--…
Content-Disposition: form-data; name="image"; filename="img1.jpg"
Content-Type: image/jpeg

<binary JPEG 1>
--…
```

- Content type is JPEG for all images (MVE always encodes as JPEG before sending).
- Multiple parts share the field name `image`. Read them in the order they arrive; MVE relies on that order for result matching.

**Response 200:** either shape is accepted by MVE. Pick whichever fits your stack:

**Shape A — wrapped:**

```json
{
  "results": [
    [ {det}, {det}, ... ],   // detections for image 0
    [ {det}, ... ],          // detections for image 1
    ...
  ]
}
```

**Shape B — bare list:**

```json
[
  [ {det}, {det}, ... ],
  [ {det}, ... ],
  ...
]
```

MVE will auto-detect the shape. **Length MUST equal the number of input images** — MVE rejects any response where `len(results) != N` and falls back to per-image calls for that chunk (so it's safe but slow — please match the count).

The `{det}` object should be identical to what `POST /detect` returns today per image — same fields, same coordinate system, same confidence semantics. MVE's downstream code (color, area, ejector rules, DB write) doesn't care whether a detection came from single or batch.

**Errors:** 4xx / 5xx → MVE logs the error, falls back per-image for that chunk, no data loss. Please still return errors when appropriate (bad JPEG, GPU OOM, etc.) rather than an empty result — silent empties look identical to "no detections found."

**Timeout:** MVE waits up to 30 s for a batched response (vs 10 s for single).

---

## Sanity-check recipes

**Probe:**

```bash
curl -sS http://yolo:4442/capabilities
```

Expect `{"batch": true, "max_batch": <n>}`.

**Batched call with 3 images:**

```bash
curl -sS -X POST http://yolo:4442/v1/object-detection/yolov5s/batch-detect/ \
    -F "image=@img0.jpg" -F "image=@img1.jpg" -F "image=@img2.jpg" | jq .
```

Expect a `results` array (or bare list) of length 3.

**Quick correctness check** — verify batched and single are equivalent:

```bash
# Single
curl -sS -F "image=@img0.jpg" http://yolo:4442/v1/object-detection/yolov5s/detect/ | jq . > /tmp/single.json

# Batched
curl -sS -F "image=@img0.jpg" http://yolo:4442/v1/object-detection/yolov5s/batch-detect/ | jq '.results[0]' > /tmp/batch.json

diff /tmp/single.json /tmp/batch.json    # should be empty (or ordering differences only)
```

## Operational notes

- Please advertise a `max_batch` you can genuinely serve without OOM — MVE trusts this and chunks accordingly.
- No need to keep the batch endpoint hot when idle — MVE probes on a 60 s cadence, so a container restart is picked up automatically.
- If you later increase `max_batch` on the server, MVE will notice within 60 s and start sending larger batches automatically.
- If you ship this and want confirmation MVE is using it, grep MVE logs for `probe_batch_capability: url=… batch=True max_batch=<n>` and `batch mode ON`.

## Contact / questions

MVE side is [vision_engine/services/pipeline.py](../vision_engine/services/pipeline.py) — search for `probe_batch_capability`, `_run_yolo_inference_batch`, and `run_inference_multi`. Adaptive threshold logic and the worker gathering loop are in [vision_engine/main.py](../vision_engine/main.py) — search for `MVE_BATCH_QUEUE_HIGH` and `_gather_more_groups`.
