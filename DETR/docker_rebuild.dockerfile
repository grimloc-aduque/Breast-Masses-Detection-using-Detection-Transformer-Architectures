
FROM grimloc13/detr:v2

# Replace Project Directory

RUN rm -rf /workspace/DETR
COPY ./DETR /workspace/DETR
RUN rm -rf /workspace/DETR/lightning_logs
