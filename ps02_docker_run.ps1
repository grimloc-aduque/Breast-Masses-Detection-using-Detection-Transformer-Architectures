
param(
    [string]$model,
    [string]$version
)

docker run -d --name $model-detection-$version -it grimloc13/$model-detection:$version /bin/bash
docker start $model-detection-$version
docker exec -it $model-detection-$version /bin/bash
