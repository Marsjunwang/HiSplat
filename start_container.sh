#!/bin/bash

WORKDIR="/home/narwal/workspace/HiSplat"
DATA_DIR="/home/narwal/workspace/hisplat_data"
CONTAINER_NAME="hisplat-container_0728"
IMAGE_NAME="hisplat:v1.0"
HOST_PORT=8081
CONTAINER_PORT=8080
SHM_SIZE="32G"

docker run -it --gpus all \
    -v $(pwd):/ws \
    -v /home/narwal/workspace/hisplat_data:/ws/datasets \
    --shm-size 32G \
    -p 8081:8080 \
    --name hisplat-container_0728 \
    hisplat:v1.0