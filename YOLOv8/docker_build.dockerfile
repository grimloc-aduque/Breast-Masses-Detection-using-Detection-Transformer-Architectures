
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Linux Packages

RUN apt-get update && \
    apt-get install -y \
        git \
        python3.8 \
        python3-pip \
        python3-dev \
        libgl1 \
        tree \
        nano \
        htop


# Project Directory

COPY ./YOLOv8 /home/YOLOv8
RUN rm -rf ./home/YOLOv8/runs_usfq_server

# Python Packages

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r /home/YOLOv8/requirements.txt
