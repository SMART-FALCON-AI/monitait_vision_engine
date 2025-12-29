FROM python:3.10.4-slim-buster
ENV PYTHONUNBUFFERED=1
RUN apt-get update -y
RUN apt-get install python3-dev python-pip gcc espeak-ng alsa-utils libespeak1 -y
COPY requirements.txt /code/requirements.txt
WORKDIR /code/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt