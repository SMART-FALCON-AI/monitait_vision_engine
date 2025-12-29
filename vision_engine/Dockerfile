FROM python:3.10-slim-bookworm
ENV PYTHONUNBUFFERED=1
RUN apt-get update -y && apt-get install -y ffmpeg libsm6 libxext6 libpq-dev gcc libdmtx0b && apt-get clean && rm -rf /var/lib/apt/lists/* 
COPY requirements.txt /code/requirements.txt
WORKDIR /code/
RUN pip3 install -r requirements.txt