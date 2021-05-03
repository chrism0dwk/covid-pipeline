FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app
VOLUME /results

COPY . /app

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System utils
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get install -y python3.7
RUN apt-get install -y python3-distutils

# Upgrade to latest pip to ensure PEP517 compatibility
RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN /usr/bin/python3.7 get-pip.py

# Install covid_pipeline
RUN python3.7 -m pip install .

ENDPOINT /usr/bin/python3.7 -m covid_pipeline.pipeline -c config.yaml -r /results
