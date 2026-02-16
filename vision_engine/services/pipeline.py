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
from typing import Optional, List, Dict, Any

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "enabled": self.enabled,
            "order": self.order
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelinePhase':
        return cls(
            model_id=data.get("model_id", "default_gradio"),
            enabled=data.get("enabled", True),
            order=int(data.get("order", 0))
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

        # Gradio client cache (for reuse)
        self._gradio_clients: Dict[str, Any] = {}

        # Initialize default models and pipeline
        self._init_defaults()

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

        # Run through each enabled phase in order
        for phase in sorted(self.current_pipeline.phases, key=lambda p: p.order):
            if not phase.enabled:
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
            response = requests.post(
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
