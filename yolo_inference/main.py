
"""
Run a rest API exposing the yolov5s object detection model
"""
import io
import json
from fastapi import FastAPI, File, UploadFile, Request, Form
from PIL import Image
from detect import Detector
import os

app = FastAPI(title="YOLOv5 Service")

DETECTION_URL = "/v1/object-detection/yolov5s"

detector = Detector()


@app.post(os.path.join(DETECTION_URL, 'detect') + '/')
@app.post(os.path.join(DETECTION_URL, 'detect'))
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    results = detector.detect(img)
    # Parse JSON string to prevent double-encoding by FastAPI
    return json.loads(results.to_json(orient="records"))


@app.post(os.path.join(DETECTION_URL, 'set-model'))
async def set_model(request: Request, model_path: str = Form(...)):
    form_data = await request.form()
    model_path = form_data.get('model_path')
    detector.set_model(model_path)
    return {'message': 'ok'}


@app.get(os.path.join(DETECTION_URL, 'health'))
async def health_check():
    """Health check endpoint for monitoring."""
    return {'status': 'healthy', 'service': 'YOLO Inference'}


@app.get(os.path.join(DETECTION_URL, 'classes'))
async def get_classes():
    """Return the list of class names from the loaded model."""
    names = getattr(detector.model, 'names', {})
    if isinstance(names, dict):
        class_list = list(names.values())
    elif isinstance(names, list):
        class_list = names
    else:
        class_list = []
    return {'classes': class_list}
