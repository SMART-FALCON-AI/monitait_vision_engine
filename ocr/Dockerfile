FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.10-cuda11.3.1
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git curl -y

WORKDIR /code/
COPY requirements.txt /code/
RUN pip install -r requirements.txt

