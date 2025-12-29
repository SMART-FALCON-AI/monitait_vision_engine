from unittest import result
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi_health import health
from model.model import EasyOCR
from utils.fastapi_utils import is_server_active, ping
from os.path import join as pjoin
import os
from PIL import Image
import io
from utils.utils import pil_to_np, normalizer
import urllib.request

BASE_URL = os.environ.get('BASE_URL', '/api/v1')
app = FastAPI(
    title="EasyOCRService",
    description="An Easy Service for OCR",
    version="0.1.0",
)

easy_ocr = EasyOCR()
app.add_api_route(BASE_URL + "/health", health([ping, is_server_active]))


@app.post(BASE_URL + "/set-languages")
async def set_languages(languages: list[str] = Form(...)):
    try:
        easy_ocr.set_language(languages)
        return {"languages": languages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(BASE_URL + "/get-languages")
async def get_languages():
    try:
        return {"languages": easy_ocr.get_languages()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(BASE_URL + "/predict_image_url")
async def predict(image_url: str):
    try:
        if not os.path.exists("images/"):
            os.mkdir("images")
        # convert image to np array
        image_path = f"images/{image_url.split('/')[-1]}"
        urllib.request.urlretrieve(image_url, image_path)
        image = open(image_path, mode="rb")
        contents = image.read()
        img = Image.open(io.BytesIO(contents))
        # convert to np array
        img = pil_to_np(img)
        os.remove(image_path)
        return {'result': list(map(normalizer, list(easy_ocr.predict(img))))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(BASE_URL + "/predict_image_file")
async def predict(image: UploadFile = File(...)):
    try:
        # convert image to np array
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        # convert to np array
        img = pil_to_np(img)
        # return {'result': list(map(normalizer, list(easy_ocr.predict(img))))}
        return {'result': easy_ocr.predict(img)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))