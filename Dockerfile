FROM python:3.10.4-slim-buster
ENV PYTHONUNBUFFERED=1
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 libpq-dev gcc libdmtx0b -y 
COPY requirements.txt /code/requirements.txt
WORKDIR /code/
RUN pip3 install -r requirements.txt