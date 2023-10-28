
$command=$args[0]
${model}=$args[1]
${version}=$args[2]

if (${model} -eq "detr-cuda"){
    $project_root = "DETR"
}
if (${model} -eq "ultralytics-cuda"){
    $project_root = "ULTRALYTICS"
}


if($command -eq "build"){
    docker build -t grimloc13/${model}:${version} -f ./$project_root/docker_build.dockerfile .
}
if($command -eq "rebuild"){
    docker build -t grimloc13/${model}:${version} -f ./$project_root/docker_rebuild.dockerfile .
}
if($command -eq "run"){
    docker run -d --name ${model}-${version} -it grimloc13/${model}:${version} /bin/bash
    docker start ${model}-${version}
    docker exec -it ${model}-${version} /bin/bash
}
if($command -eq "run-gpu"){
    docker run -d --name ${model}-${version} --gpus all -it grimloc13/${model}:${version} /bin/bash
    docker start ${model}-${version}
    docker exec -it ${model}-${version} /bin/bash
}
if($command -eq "execute"){
    docker start ${model}-${version}
    docker exec -it ${model}-${version} /bin/bash
}
if($command -eq "push"){
    docker push grimloc13/${model}:${version}
}
if($command -eq "pull"){
    docker pull grimloc13/${model}:${version}
}
