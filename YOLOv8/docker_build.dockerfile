
FROM python

# Linux Packages

RUN apt-get update
RUN apt-get install libgl1 tree nano htop -y

# Project Directory

COPY ./YOLOv8 /home/YOLOv8
RUN rm -rf ./home/YOLOv8/runs_usfq_server

# Python Packages

RUN pip3 install -r /home/YOLOv8/requirements.txt
RUN pip3 install ultralytics

