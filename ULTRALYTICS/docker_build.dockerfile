
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Linux Packages

RUN apt-get update && \
    apt-get install -y \
        git \
        libglib2.0-0 \
        ffmpeg \
        libsm6 \
        libxext6 \
        tree \
        nano \
        htop


# Project Directory

COPY ./ULTRALYTICS /workspace/ULTRALYTICS
RUN rm -rf /workspace/ULTRALYTICS/runs_ultralytics

# Python Packages

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r /workspace/ULTRALYTICS/requirements.txt
