
FROM grimloc13/ultralytics-cuda:v1

# Replace Project Directory

RUN rm -rf /workspace/ULTRALYTICS
COPY ./ULTRALYTICS /workspace/ULTRALYTICS
RUN rm -rf /workspace/ULTRALYTICS/runs_ultralytics

