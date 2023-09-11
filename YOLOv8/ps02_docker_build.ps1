
param(
    [string]$version
)

docker build -t grimloc13/yolo-detection:$version -f ./YOLOv8/dockerfile .
