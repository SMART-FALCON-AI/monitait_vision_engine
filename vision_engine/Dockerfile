FROM python:3.10-slim-bookworm
ENV PYTHONUNBUFFERED=1
# Use HTTPS mirrors to bypass potential firewall blocks
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update -y && \
    apt-get install -y ffmpeg libsm6 libxext6 libpq-dev gcc libdmtx0b && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 
COPY requirements.txt /code/requirements.txt
WORKDIR /code/
RUN pip3 install -r requirements.txt