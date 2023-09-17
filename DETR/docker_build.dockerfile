
FROM python

# Linux Packages

RUN apt-get update
RUN apt-get install libgl1 tree nano htop -y

# Project Directory

COPY ./DETR /home/DETR
RUN rm -rf ./home/DETR/lightning_logs

# Python Packages

RUN pip3 install -r /home/DETR/requirements.txt
