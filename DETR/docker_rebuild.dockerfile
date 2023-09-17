
FROM grimloc13/detr-detection:v03

# Replace Project Directory

RUN rm -rf ./home/DETR
COPY ./DETR /home/DETR
RUN rm -rf ./home/DETR/lightning_logs
