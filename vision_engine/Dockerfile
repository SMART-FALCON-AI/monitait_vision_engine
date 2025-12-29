FROM python:3.10-slim-bookworm
ENV PYTHONUNBUFFERED=1
# Use Tsinghua mirror for faster package downloads in China/Asia
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free non-free-firmware\n\
deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free non-free-firmware\n\
deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware\n" \
> /etc/apt/sources.list && \
    apt-get update -y && \
    apt-get install -y ffmpeg libsm6 libxext6 libpq-dev gcc libdmtx0b && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 
COPY requirements.txt /code/requirements.txt
WORKDIR /code/
RUN pip3 install -r requirements.txt