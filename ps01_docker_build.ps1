
param(
    [string]$model,
    [string]$version
)

if ($model -eq "yolo"){
    $project_root = 'YOLOv8'
}else{
    $project_root = 'DETR'
}

docker build -t grimloc13/$model-detection:$version -f ./$project_root/dockerfile .
