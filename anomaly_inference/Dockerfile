# Anomaly Inference Worker — pixel-stat baseline anomaly detection.
# Mirrors the yolo_inference / math_inference container layout: FastAPI on
# 4444, code bind-mounted from ./anomaly_inference, baselines persisted to a
# named volume so they survive container recreation.

FROM python:3.10-slim

# OpenCV runtime libs (-headless wheel still needs libgl + libglib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Baselines are stored at /baselines (bind-mounted from
# ./volumes/anomaly_baselines on the host).
RUN mkdir -p /baselines

COPY . /code

EXPOSE 4444

# overridden by docker-compose `command:` but useful for local docker run.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4444", "--workers", "2"]
