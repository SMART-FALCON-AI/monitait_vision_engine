# CUDA runtime base — has the libs CuPy needs.
# Runs fine without GPU passthrough: CuPy fails to find a device at import time,
# we catch it, and the worker falls back to numpy transparently.
#
# Mirror of yolo_inference's behaviour:
#   - docker-compose reserves nvidia GPU as a *reservation* (soft) → no error on CPU-only hosts
#   - framework autodetects device at startup
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /code
COPY requirements.txt /code/
RUN pip install --break-system-packages -r requirements.txt

COPY . /code/
