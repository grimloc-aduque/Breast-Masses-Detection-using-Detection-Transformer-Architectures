
param(
    [string]$version
)

docker run -d --name yolo-detection-$version -it grimloc13/yolo-detection:$version /bin/bash
