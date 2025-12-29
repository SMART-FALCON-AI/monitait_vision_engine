from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
from fastapi_frame_stream import FrameStreamer
from fastapi.responses import HTMLResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import base64
import uuid
import socket
import os
import asyncio

app = FastAPI()
fs = FrameStreamer()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class InputImg(BaseModel):
    img_base64str : str


IP = "0.0.0.0"
templates = Jinja2Templates(directory="templates")

@app.post("/send_frame_from_string/{stream_id}")
async def send_frame_from_string(stream_id: str, d:InputImg):
    await fs.send_frame(stream_id, d.img_base64str)


@app.post("/send_frame_from_file/{stream_id}")
async def send_frame_from_file(stream_id: str, file: UploadFile = File(...)):
    await fs.send_frame(stream_id, file)


@app.get("/video_feed/{stream_id}")
async def video_feed(stream_id: str):
    return fs.get_stream(stream_id)


@app.get("/image_feed/{stream_id}")
async def video_feed(stream_id: str):
    image = fs._get_image(f"org_{stream_id}")
    image_path = f"/images/{str(uuid.uuid4())}.jpg"
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image))
    mode = stream_id.split("_")[0]
    requests.get(f"http://web:8000/api/send-image-to-singularity", data={"image_path":image_path, "mode":mode})
    return True


@app.get("/stream/", response_class=HTMLResponse)
async def read_item(request: Request):
    cams = requests.get("http://web:8000/api/get-active-cameras")
    _cams = []
    for cam in json.loads(cams.text):
        _cams.append(f"back_{cam}")
        _cams.append(f"up_{cam}")
    return templates.TemplateResponse("stream.html", {"request": request, "cams": _cams})


@app.get("/stream/defects", response_class=HTMLResponse)
async def read_item(request: Request):
    _cams = []
    _cams.append(f"back")
    _cams.append(f"up")
    return templates.TemplateResponse("stream.html", {"request": request, "cams": _cams})


@app.get("/stream/config", response_class=HTMLResponse)
async def config(request: Request):
    cams = requests.get("http://web:8000/api/get-cameras")
    _cams = json.loads(cams.text)
    stream_ip = f"http://{IP}:5000/video_feed/"
    return templates.TemplateResponse("stream_config.html", {"request": request, "cams": _cams, "ip": stream_ip})
