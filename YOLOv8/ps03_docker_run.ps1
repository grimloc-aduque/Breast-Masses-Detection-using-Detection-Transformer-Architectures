
param(
    [string]$version
)

docker run -d --name yolo-detection-$version -it grimloc13/yolo-detection:$version /bin/bash
docker start yolo-detection-$version
docker exec -it yolo-detection-$version /bin/bash