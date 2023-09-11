
param(
    [string]$version
)

docker start yolo-detection-$version
docker exec -it yolo-detection-$version /bin/bash
