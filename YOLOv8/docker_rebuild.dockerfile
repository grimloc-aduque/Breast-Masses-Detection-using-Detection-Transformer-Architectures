
FROM grimloc13/yolo-detection:v05

# Replace Project Directory

RUN rm -rf ./home/YOLOv8
COPY ./YOLOv8 /home/YOLOv8
RUN rm -rf ./home/YOLOv8/runs_usfq_server

