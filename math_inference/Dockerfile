# CUDA runtime base + targeted dev headers — the smallest combo that lets CuPy
# JIT-compile element-wise kernels at runtime.
#
# Why: plain cuda:runtime lacks vector_types.h / cuda_fp16.h so CuPy's NVRTC
# compilation fails, and math_worker silently falls back to numpy (see comment
# in math_worker.py). cuda:devel fixes this but bloats the image from ~3GB to
# ~7.5GB which exceeds the registry's nginx body limit. So we stick with the
# runtime base and pull in only the dev packages that supply the missing
# headers (cudart-dev for vector_types.h family, nvrtc-dev for the JIT path).
#
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
      cuda-cudart-dev-12-2 cuda-nvrtc-dev-12-2 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /code
COPY requirements.txt /code/
# Ubuntu 22.04's pip (≈22.x) predates PEP 668 enforcement and doesn't accept
# --break-system-packages; newer images (Ubuntu 24.04+) need it. Try with the
# flag first, fall back to plain install so the Dockerfile works on both.
RUN pip install --break-system-packages -r requirements.txt \
 || pip install -r requirements.txt

COPY . /code/
