#!/bin/bash

VIDEO="${1?"usage: $0 <path_to_video> [max_volume_percent] [images_per_percent] [real_duration]"}"
MAX_VOLUME_PERCENT=${2:-100}
IMAGES_PER_PERCENT=${3:-1}
VIDEO_DURATION=${4:-$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${VIDEO})}

IMAGES_DIR="tmp"
DATASET_DIR="dataset"


FPS=$(echo "scale=4; ${MAX_VOLUME_PERCENT} * ${IMAGES_PER_PERCENT} / ${VIDEO_DURATION}"|bc)

mkdir -p "${IMAGES_DIR}"
rm -f "${IMAGES_DIR}/*.jpg"

ffmpeg -i "${VIDEO}" -t ${VIDEO_DURATION} -vf fps=${FPS},scale=640:360 "${IMAGES_DIR}/$(basename ${VIDEO})%05d.jpg"
