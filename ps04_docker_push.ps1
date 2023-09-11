
param(
    [string]$model,
    [string]$version
)

docker push grimloc13/$model-detection:$version
