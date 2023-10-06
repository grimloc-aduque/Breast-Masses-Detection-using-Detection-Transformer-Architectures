
# FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
FROM nvcr.io/nvidia/pytorch:2.1-py3

ENV DEBIAN_FRONTEND=noninteractive

# Linux Packages

RUN apt-get update && \
    apt-get install -y \
        git \
        python3.10 \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        tree \
        nano \
        htop

# Project Directory

COPY ./DETR /home/DETR
RUN rm -rf ./home/DETR/lightning_logs

# Python Packages

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r /home/DETR/requirements.txt
