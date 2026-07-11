import torch
import os

# Configuration from environment variables
YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.3"))
YOLO_IOU_THRESHOLD = float(os.environ.get("YOLO_IOU_THRESHOLD", "0.4"))


class Detector():

    def __init__(self):
        self.model = torch.hub.load(
            "/code/yolov5/", 'custom', source='local', path="best.pt")
        self.model.iou = YOLO_IOU_THRESHOLD
        self.model.agnostic = True
        self.model.conf = YOLO_CONF_THRESHOLD
        print(f"YOLO Detector initialized: conf={YOLO_CONF_THRESHOLD}, iou={YOLO_IOU_THRESHOLD}")

    def set_model(self, model_path):
        self.model = torch.hub.load(
            "/code/yolov5/", 'custom', source='local', path=model_path)
        self.model.iou = YOLO_IOU_THRESHOLD
        self.model.agnostic = True
        self.model.conf = YOLO_CONF_THRESHOLD
        print(f"YOLO model changed to {model_path}: conf={YOLO_CONF_THRESHOLD}, iou={YOLO_IOU_THRESHOLD}")

    def detect(self, img):
        """
        img: PIL image
        """
        results = self.model(img)
        return results.pandas().xyxy[0]

    def detect_batch(self, imgs):
        """MVE-batch v1 — batched inference.

        imgs: list of PIL Images (length N, N >= 1).
        Returns: list of pandas DataFrames (length N), one per input image, each row
        is one detection (same shape as .detect()'s return).

        YOLOv5's hub model supports list input natively — model([img1, img2, ...])
        runs ONE forward pass through the batched tensor, which is exactly what makes
        batching a throughput win vs N sequential /detect calls.
        """
        results = self.model(imgs)
        # results.pandas().xyxy is a list of per-image DataFrames when input is a list.
        return results.pandas().xyxy

    def detect_render(self, img):
        """
        img: PIL image
        """
        results = self.model(img)
        return results.render()[0]
