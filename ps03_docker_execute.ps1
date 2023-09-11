
param(
    [string]$model,
    [string]$version
)

docker start $model-detection-$version
docker exec -it $model-detection-$version /bin/bash
