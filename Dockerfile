FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean

WORKDIR /app

# optuna
RUN apt-get update && \
    apt-get install -y gcc mysql-client python3-dev libmysqlclient-dev && \
    apt-get clean && \
    pip install optuna mysqlclient

# install requirements
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/
COPY pyproject.toml /app/
RUN uv sync
