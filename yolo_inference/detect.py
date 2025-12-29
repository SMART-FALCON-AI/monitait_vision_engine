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

    def detect_render(self, img):
        """
        img: PIL image
        """
        results = self.model(img)
        return results.render()[0]
