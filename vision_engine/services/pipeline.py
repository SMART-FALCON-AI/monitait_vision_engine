"""Pipeline management for inference models.

Extracted from main.py - contains InferenceModel, PipelinePhase, Pipeline,
and PipelineManager classes for managing inference pipelines.
"""

import os
import time
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlsplit, urlunsplit

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import requests
except ImportError:
    requests = None

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None
    handle_file = None

# These will be overridden when config.py is available
YOLO_INFERENCE_URL = os.environ.get("YOLO_INFERENCE_URL", "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/")
GRADIO_MODEL = os.environ.get("GRADIO_MODEL", "Data Matrix")
GRADIO_CONFIDENCE_THRESHOLD = float(os.environ.get("GRADIO_CONFIDENCE_THRESHOLD", "0.25"))

logger = logging.getLogger(__name__)


@dataclass
class InferenceModel:
    """Represents a single inference model configuration.

    A model defines how to run inference on an image:
    - name: Human-readable name
    - model_type: "gradio" or "yolo"
    - inference_url: API endpoint URL
    - model_name: Specific model to use (for Gradio)
    - confidence_threshold: Minimum confidence for detections
    """
    name: str
    model_type: str = "gradio"  # "gradio" or "yolo"
    inference_url: str = "https://smartfalcon-ai-industrial-defect-detection.hf.space"
    model_name: str = "Data Matrix"
    confidence_threshold: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "inference_url": self.inference_url,
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceModel':
        """Create an InferenceModel from a dictionary."""
        return cls(
            name=data.get("name", "Unknown Model"),
            model_type=data.get("model_type", "gradio"),
            inference_url=data.get("inference_url", ""),
            model_name=data.get("model_name", "Data Matrix"),
            confidence_threshold=float(data.get("confidence_threshold", 0.25))
        )


@dataclass
class PipelinePhase:
    """A single phase in an inference pipeline.

    Each phase runs a specific model on the captured images.
    Multiple phases allow sequential processing (e.g., defect detection then classification).
    """
    model_id: str  # Reference to a model by ID
    enabled: bool = True
    order: int = 0  # Order in pipeline (lower = first)
    stride: int = 1  # Run this phase every Nth frame (1 = every frame). Lets a
                     # heavy secondary model (e.g. math) sample frames while a
                     # fast primary (yolo) runs every frame for ejection.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "enabled": self.enabled,
            "order": self.order,
            "stride": self.stride
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelinePhase':
        return cls(
            model_id=data.get("model_id", "default_gradio"),
            enabled=data.get("enabled", True),
            order=int(data.get("order", 0)),
            stride=max(1, int(data.get("stride", 1) or 1))
        )


@dataclass
class Pipeline:
    """Represents an inference pipeline configuration.

    A pipeline consists of one or more phases, each running a model.
    This is similar to State for capture - Pipeline is for inference.
    """
    name: str
    phases: List[PipelinePhase] = field(default_factory=list)
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phases": [p.to_dict() for p in self.phases],
            "enabled": self.enabled,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        phases_data = data.get("phases", [])
        phases = [PipelinePhase.from_dict(p) for p in phases_data] if phases_data else []
        # Sort phases by order
        phases.sort(key=lambda p: p.order)

        return cls(
            name=data.get("name", "default"),
            phases=phases,
            enabled=data.get("enabled", True),
            description=data.get("description", "")
        )


class PipelineManager:
    """Manages inference pipelines and models.

    Similar to StateManager for capture, this handles:
    - Available inference models (Gradio, YOLO, etc.)
    - Pipeline definitions (sequences of models)
    - Active pipeline selection
    - Running inference through the active pipeline
    """

    # Default Gradio model URL
    DEFAULT_GRADIO_URL = "https://smartfalcon-ai-industrial-defect-detection.hf.space"
    # Default YOLO model URL
    DEFAULT_YOLO_URL = "http://yolo_inference:4442/v1/object-detection/yolov5s/detect/"

    def __init__(self):
        self.models: Dict[str, InferenceModel] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.current_pipeline: Optional[Pipeline] = None
        self.pipeline_lock = threading.Lock()
        self._frame_counter = 0  # advanced once per run_inference, for phase stride

        # Gradio client cache (for reuse)
        self._gradio_clients: Dict[str, Any] = {}

        # HTTP session with connection pooling for YOLO requests
        self._http_session = requests.Session() if requests else None
        if self._http_session:
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=8, pool_maxsize=32, max_retries=1
            )
            self._http_session.mount("http://", adapter)
            self._http_session.mount("https://", adapter)

        # v4.0.82 — Per-yolo-URL batch capability cache: url -> (has_batch, max_batch, probed_at_epoch).
        # Populated lazily via GET <base>/capabilities. Missing / 404 / timeout -> (False, 1, ts).
        # Re-probed every _BATCH_PROBE_TTL_S so a yolo container upgrade lights the batch path up
        # automatically without an MVE restart.
        self._batch_cache: Dict[str, Tuple[bool, int, float]] = {}
        self._batch_cache_lock = threading.Lock()

        # Initialize default models and pipeline
        self._init_defaults()

    # v4.0.82 batch-endpoint knobs
    _BATCH_PROBE_TTL_S = 60.0
    _BATCH_PROBE_TIMEOUT_S = 2.0
    _BATCH_POST_TIMEOUT_S = 30.0  # batched yolo may take longer than single (30 vs single 10s)

    def _derive_capabilities_url(self, inference_url: str) -> str:
        """From an inference URL like http://host:port/v1/.../detect/, return http://host:port/capabilities."""
        parts = urlsplit(inference_url)
        return urlunsplit((parts.scheme, parts.netloc, "/capabilities", "", ""))

    def _derive_batch_url(self, inference_url: str) -> str:
        """From an inference URL ending in /detect(/), return the same URL with /detect replaced by /batch-detect."""
        # Preserve trailing slash presence.
        if inference_url.endswith("/detect/"):
            return inference_url[:-len("/detect/")] + "/batch-detect/"
        if inference_url.endswith("/detect"):
            return inference_url[:-len("/detect")] + "/batch-detect"
        # Fallback: last path segment substitution.
        parts = urlsplit(inference_url)
        path = parts.path.rstrip("/")
        if path.endswith("/detect"):
            new_path = path[:-len("/detect")] + "/batch-detect" + ("/" if parts.path.endswith("/") else "")
        else:
            new_path = parts.path + ("batch-detect" if parts.path.endswith("/") else "/batch-detect")
        return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))

    def probe_batch_capability(self, model: 'InferenceModel') -> Tuple[bool, int]:
        """Return (has_batch, max_batch) for the given YOLO model, cached for _BATCH_PROBE_TTL_S.

        Non-YOLO models always report (False, 1) — Gradio has no batch endpoint contract.
        On any failure (timeout / non-2xx / bad JSON / missing "batch" field) → (False, 1).
        """
        if model.model_type != "yolo":
            return (False, 1)
        if not self._http_session:
            return (False, 1)

        url = model.inference_url
        now = time.time()
        with self._batch_cache_lock:
            cached = self._batch_cache.get(url)
            if cached and (now - cached[2]) < self._BATCH_PROBE_TTL_S:
                return (cached[0], cached[1])

        has_batch = False
        max_batch = 1
        try:
            probe_url = self._derive_capabilities_url(url)
            resp = self._http_session.get(probe_url, timeout=self._BATCH_PROBE_TIMEOUT_S)
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict):
                    has_batch = bool(body.get("batch", False))
                    max_batch = int(body.get("max_batch", 8)) if has_batch else 1
                    if max_batch < 1:
                        max_batch = 1
        except Exception as e:
            logger.debug(f"probe_batch_capability({url}) failed: {e} — fallback to single-frame")

        with self._batch_cache_lock:
            self._batch_cache[url] = (has_batch, max_batch, now)
        logger.info(f"probe_batch_capability: url={url} batch={has_batch} max_batch={max_batch}")
        return (has_batch, max_batch)

    def batch_available(self) -> bool:
        """True iff at least one YOLO phase in the current pipeline has a probed-batch-capable model.

        Non-YOLO phases (e.g., Gradio) still run per-image inside run_inference_multi, but their
        presence alone doesn't disable batching for the other phases. This method is meant as a
        cheap gate for callers deciding whether to accumulate a batch upstream.
        """
        if not self.current_pipeline:
            return False
        for phase in self.current_pipeline.phases:
            if not phase.enabled:
                continue
            model = self.models.get(phase.model_id)
            if model is None or model.model_type != "yolo":
                continue
            has_batch, _ = self.probe_batch_capability(model)
            if has_batch:
                return True
        return False

    def effective_max_batch(self) -> int:
        """Smallest max_batch across YOLO phases in current pipeline (or 1 if no batch-capable phase).

        Non-YOLO phases don't restrict this — they'll be looped internally regardless.
        """
        m = None
        if not self.current_pipeline:
            return 1
        for phase in self.current_pipeline.phases:
            if not phase.enabled:
                continue
            model = self.models.get(phase.model_id)
            if model is None or model.model_type != "yolo":
                continue
            has_batch, max_batch = self.probe_batch_capability(model)
            if not has_batch:
                continue
            m = max_batch if m is None else min(m, max_batch)
        return m if m else 1

    def _init_defaults(self):
        """Initialize default models and pipeline."""
        # Default Gradio model (HuggingFace)
        self.models["default_gradio"] = InferenceModel(
            name="Gradio HuggingFace",
            model_type="gradio",
            inference_url=self.DEFAULT_GRADIO_URL,
            model_name="Data Matrix",
            confidence_threshold=0.25
        )

        # Default YOLO model (local)
        self.models["default_yolo"] = InferenceModel(
            name="Local YOLO",
            model_type="yolo",
            inference_url=self.DEFAULT_YOLO_URL,
            model_name="yolov5s",
            confidence_threshold=0.3
        )

        # Default pipeline using Gradio
        default_pipeline = Pipeline(
            name="default",
            phases=[
                PipelinePhase(model_id="default_gradio", enabled=True, order=0)
            ],
            enabled=True,
            description="Default pipeline using Gradio HuggingFace model"
        )
        self.pipelines["default"] = default_pipeline
        self.current_pipeline = default_pipeline

        logger.info(f"PipelineManager initialized with {len(self.models)} models, {len(self.pipelines)} pipelines")

    def add_model(self, model_id: str, model: InferenceModel) -> bool:
        """Add or update an inference model."""
        try:
            with self.pipeline_lock:
                self.models[model_id] = model
                logger.info(f"Added/updated model: {model_id} ({model.name})")
                return True
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return False

    def remove_model(self, model_id: str) -> bool:
        """Remove an inference model."""
        try:
            with self.pipeline_lock:
                if model_id.startswith("default_"):
                    logger.warning(f"Cannot remove default model: {model_id}")
                    return False
                if model_id in self.models:
                    del self.models[model_id]
                    logger.info(f"Removed model: {model_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error removing model: {e}")
            return False

    def add_pipeline(self, pipeline: Pipeline) -> bool:
        """Add or update a pipeline."""
        try:
            with self.pipeline_lock:
                self.pipelines[pipeline.name] = pipeline
                logger.info(f"Added/updated pipeline: {pipeline.name}")
                return True
        except Exception as e:
            logger.error(f"Error adding pipeline: {e}")
            return False

    def remove_pipeline(self, name: str) -> bool:
        """Remove a pipeline."""
        try:
            with self.pipeline_lock:
                if name == "default":
                    logger.warning("Cannot remove default pipeline")
                    return False
                if name in self.pipelines:
                    del self.pipelines[name]
                    logger.info(f"Removed pipeline: {name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error removing pipeline: {e}")
            return False

    def set_current_pipeline(self, name: str) -> bool:
        """Set the currently active pipeline."""
        try:
            with self.pipeline_lock:
                if name not in self.pipelines:
                    logger.error(f"Pipeline not found: {name}")
                    return False
                self.current_pipeline = self.pipelines[name]
                logger.info(f"Set current pipeline to: {name}")
                return True
        except Exception as e:
            logger.error(f"Error setting pipeline: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[InferenceModel]:
        """Get a model by ID."""
        return self.models.get(model_id)

    def get_current_model(self) -> Optional[InferenceModel]:
        """Get the first enabled model in the current pipeline."""
        if not self.current_pipeline or not self.current_pipeline.phases:
            return self.models.get("default_gradio")

        for phase in sorted(self.current_pipeline.phases, key=lambda p: p.order):
            if phase.enabled and phase.model_id in self.models:
                return self.models[phase.model_id]

        return self.models.get("default_gradio")

    def run_inference(self, image_bytes: bytes) -> tuple:
        """Run inference through the current pipeline.

        Returns:
            Tuple of (detections_list, model_name_used)
        """
        if not self.current_pipeline:
            logger.warning("No pipeline set, using default model")
            model = self.models.get("default_gradio")
            if model:
                return self._run_model_inference(image_bytes, model), model.name
            return [], "unknown"

        all_detections = []
        models_used = []

        # Frame counter drives per-phase stride (run phase every Nth frame).
        self._frame_counter += 1
        fc = self._frame_counter

        # Run through each enabled phase in order
        for phase in sorted(self.current_pipeline.phases, key=lambda p: p.order):
            if not phase.enabled:
                continue

            # Per-phase stride: skip this phase on frames that aren't its turn.
            stride = getattr(phase, "stride", 1) or 1
            if stride > 1 and (fc % stride) != 0:
                continue

            model = self.models.get(phase.model_id)
            if not model:
                logger.warning(f"Model not found for phase: {phase.model_id}")
                continue

            try:
                detections = self._run_model_inference(image_bytes, model)
                if detections:
                    all_detections.extend(detections)
                    models_used.append(model.name)
            except Exception as e:
                logger.error(f"Error running inference with model {model.name}: {e}")

        model_names = ", ".join(models_used) if models_used else "none"
        return all_detections, model_names

    def _run_model_inference(self, image_bytes: bytes, model: InferenceModel) -> List[Dict]:
        """Run inference using a specific model."""
        if model.model_type == "gradio":
            return self._run_gradio_inference(image_bytes, model)
        elif model.model_type == "yolo":
            return self._run_yolo_inference(image_bytes, model)
        else:
            logger.error(f"Unknown model type: {model.model_type}")
            return []

    def _run_gradio_inference(self, image_bytes: bytes, model: InferenceModel) -> List[Dict]:
        """Run inference through Gradio API."""
        try:
            from gradio_client import Client, handle_file
            import tempfile

            # Get or create cached client
            if model.inference_url not in self._gradio_clients:
                logger.info(f"Initializing Gradio client for {model.inference_url}")
                self._gradio_clients[model.inference_url] = Client(model.inference_url)

            client = self._gradio_clients[model.inference_url]

            # Write image to temp file (Gradio needs file path)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            try:
                # Call Gradio API
                result = client.predict(
                    handle_file(tmp_path),
                    model.model_name,
                    model.confidence_threshold,
                    api_name="/detect"
                )

                # Convert Gradio response to standard format
                detections = []
                if isinstance(result, list):
                    for det in result:
                        if isinstance(det, dict):
                            detections.append({
                                "xmin": det.get("x1", det.get("xmin", 0)),
                                "ymin": det.get("y1", det.get("ymin", 0)),
                                "xmax": det.get("x2", det.get("xmax", 0)),
                                "ymax": det.get("y2", det.get("ymax", 0)),
                                "confidence": det.get("confidence", 0),
                                "class": det.get("class_id", det.get("class", 0)),
                                "name": det.get("name", f"Class {det.get('class_id', 0)}")
                            })

                return detections

            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.debug(f"Could not delete temp file {tmp_path}: {e}")

        except Exception as e:
            logger.error(f"Gradio inference error: {e}")
            return []

    def _run_yolo_inference(self, image_bytes: bytes, model: InferenceModel) -> List[Dict]:
        """Run inference through YOLO API."""
        try:
            http = self._http_session or requests
            response = http.post(
                model.inference_url,
                files={"image": image_bytes},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            # Some YOLO endpoints double-encode: return a JSON string instead of a list.
            # Unwrap if needed.
            if isinstance(result, str):
                result = json.loads(result)
            return result
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    def _run_yolo_inference_batch(self, image_bytes_list: List[bytes], model: 'InferenceModel') -> Optional[List[List[Dict]]]:
        """v4.0.82 — POST a batch of images to the model's /batch-detect endpoint.

        Returns per-image detection lists in the same order as image_bytes_list, or None if the
        endpoint returned a shape we can't destructure (caller then falls back per-image so no
        detections are lost).
        """
        if not self._http_session:
            return None
        try:
            batch_url = self._derive_batch_url(model.inference_url)
            # Multipart with N 'image' parts. requests accepts repeated field name via list-of-tuples.
            files = [("image", (f"img{i}.jpg", img, "image/jpeg")) for i, img in enumerate(image_bytes_list)]
            response = self._http_session.post(batch_url, files=files, timeout=self._BATCH_POST_TIMEOUT_S)
            response.raise_for_status()
            body = response.json()
            if isinstance(body, str):
                body = json.loads(body)
            # Accept either { "results": [ [det,...], [det,...], ... ] } or a bare list of the same shape.
            if isinstance(body, dict) and "results" in body:
                results = body["results"]
            elif isinstance(body, list):
                results = body
            else:
                logger.warning(f"batch YOLO returned unexpected shape (type={type(body).__name__}) — falling back per-image")
                return None
            if not isinstance(results, list) or len(results) != len(image_bytes_list):
                logger.warning(f"batch YOLO length mismatch: got {len(results) if isinstance(results, list) else '?'} expected {len(image_bytes_list)} — falling back per-image")
                return None
            out: List[List[Dict]] = []
            for r in results:
                if isinstance(r, list):
                    out.append([d for d in r if isinstance(d, dict)])
                elif isinstance(r, dict) and "detections" in r:
                    out.append([d for d in r.get("detections", []) if isinstance(d, dict)])
                else:
                    out.append([])
            return out
        except Exception as e:
            logger.error(f"_run_yolo_inference_batch error: {e}")
            return None

    def run_inference_multi(self, image_bytes_list: List[bytes]) -> List[Tuple[List[Dict], str]]:
        """v4.0.82 — Run the current pipeline over a batch of images.

        For each phase in order:
          - YOLO + probe_batch_capability returns has_batch=True + all in-stride indices > 1
              -> ONE POST for the batch (chunked by max_batch), distribute results
          - else -> per-image loop (identical to run_inference())

        Returns:
            list of (detections_list, models_used_str) tuples, in the same order as inputs.

        Fallback behavior on any batch failure: the specific chunk falls back to per-image calls,
        so no image is ever silently dropped.
        """
        N = len(image_bytes_list)
        if N == 0:
            return []
        if N == 1:
            det, name = self.run_inference(image_bytes_list[0])
            return [(det, name)]
        if not self.current_pipeline:
            return [self.run_inference(b) for b in image_bytes_list]

        all_dets_per_image: List[List[Dict]] = [[] for _ in range(N)]
        models_used_per_image: List[List[str]] = [[] for _ in range(N)]

        # Bump the frame counter by N so per-phase stride semantics stay consistent with the
        # single-image path (each image counts as one frame).
        with self.pipeline_lock:
            fc_start = self._frame_counter
            self._frame_counter += N

        for phase in sorted(self.current_pipeline.phases, key=lambda p: p.order):
            if not phase.enabled:
                continue

            stride = getattr(phase, "stride", 1) or 1
            if stride > 1:
                # Position i in the batch acts as if it were the (fc_start + i + 1)-th call.
                active_indices = [i for i in range(N) if ((fc_start + i + 1) % stride) == 0]
                if not active_indices:
                    continue
            else:
                active_indices = list(range(N))

            model = self.models.get(phase.model_id)
            if not model:
                logger.warning(f"Model not found for phase: {phase.model_id}")
                continue

            do_batch = False
            max_batch = 1
            if model.model_type == "yolo":
                has_batch, max_batch = self.probe_batch_capability(model)
                do_batch = has_batch and len(active_indices) > 1

            if do_batch:
                # Chunk if the active set exceeds server max_batch.
                for chunk_start in range(0, len(active_indices), max_batch):
                    chunk = active_indices[chunk_start:chunk_start + max_batch]
                    images_for_chunk = [image_bytes_list[i] for i in chunk]
                    per_image_dets = None
                    try:
                        per_image_dets = self._run_yolo_inference_batch(images_for_chunk, model)
                    except Exception as e:
                        logger.error(f"Batched YOLO exception on model {model.name}: {e}")
                    if per_image_dets is None or len(per_image_dets) != len(chunk):
                        # Chunk-level fallback: run each image single. Never loses detections.
                        per_image_dets = [self._run_yolo_inference(image_bytes_list[i], model) for i in chunk]
                    for offset, i in enumerate(chunk):
                        dets = per_image_dets[offset] or []
                        if dets:
                            all_dets_per_image[i].extend(dets)
                            if model.name not in models_used_per_image[i]:
                                models_used_per_image[i].append(model.name)
            else:
                # Per-image loop — same as single-frame path.
                for i in active_indices:
                    try:
                        dets = self._run_model_inference(image_bytes_list[i], model)
                    except Exception as e:
                        logger.error(f"Per-image inference error for model {model.name} at index {i}: {e}")
                        dets = []
                    if dets:
                        all_dets_per_image[i].extend(dets)
                        if model.name not in models_used_per_image[i]:
                            models_used_per_image[i].append(model.name)

        return [
            (all_dets_per_image[i], ", ".join(models_used_per_image[i]) or "none")
            for i in range(N)
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline manager status."""
        return {
            "models": {mid: m.to_dict() for mid, m in self.models.items()},
            "pipelines": {name: p.to_dict() for name, p in self.pipelines.items()},
            "current_pipeline": self.current_pipeline.name if self.current_pipeline else None,
            "current_model": self.get_current_model().to_dict() if self.get_current_model() else None
        }

    def to_config(self) -> Dict[str, Any]:
        """Export configuration for saving to DATA_FILE."""
        return {
            "models": {mid: m.to_dict() for mid, m in self.models.items()},
            "pipelines": {name: p.to_dict() for name, p in self.pipelines.items()},
            "current_pipeline": self.current_pipeline.name if self.current_pipeline else "default"
        }

    def from_config(self, config: Dict[str, Any]) -> bool:
        """Load configuration from DATA_FILE."""
        try:
            # Load models
            if "models" in config:
                for model_id, model_data in config["models"].items():
                    self.models[model_id] = InferenceModel.from_dict(model_data)

            # Load pipelines
            if "pipelines" in config:
                for name, pipeline_data in config["pipelines"].items():
                    self.pipelines[name] = Pipeline.from_dict(pipeline_data)

            # Set current pipeline
            current_name = config.get("current_pipeline", "default")
            if current_name in self.pipelines:
                self.current_pipeline = self.pipelines[current_name]

            logger.info(f"Loaded pipeline config: {len(self.models)} models, {len(self.pipelines)} pipelines")
            return True
        except Exception as e:
            logger.error(f"Error loading pipeline config: {e}")
            return False
