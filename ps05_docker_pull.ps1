
param(
    [string]$model,
    [string]$version
)

docker pull grimloc13/$model-detection:$version
