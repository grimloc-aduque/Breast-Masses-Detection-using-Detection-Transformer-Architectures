
$command=$args[0]
$model=$args[1]
$version=$args[2]

if ($model -eq "yolo"){
    $project_root = "YOLOv8"
}else{
    $project_root = "DETR"
}


if($command -eq "build"){
    docker build -t grimloc13/$model-detection:$version -f ./$project_root/docker_build.dockerfile .
}
if($command -eq "rebuild"){
    docker build -t grimloc13/$model-detection:$version -f ./$project_root/docker_rebuild.dockerfile .
}
if($command -eq "run"){
    docker run -d --name $model-$version -it grimloc13/$model-detection:$version /bin/bash
    docker start $model-$version
    docker exec -it $model-$version /bin/bash
}
if($command -eq "run-gpu"){
    docker run -d --name $model-$version --gpus all -it grimloc13/$model-detection:$version /bin/bash
    docker start $model-$version
    docker exec -it $model-$version /bin/bash
}
if($command -eq "execute"){
    docker start $model-$version
    docker exec -it $model-$version /bin/bash
}
if($command -eq "push"){
    docker push grimloc13/$model-detection:$version
}
if($command -eq "pull"){
    docker pull grimloc13/$model-detection:$version
}
if($command -eq "remove"){
    try{
        docker stop $model-$version
    }catch{}
    try{
        docker rm $model-$version
    }catch{}
}

