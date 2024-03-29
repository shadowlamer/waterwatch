#!/bin/bash

ACTUAL_DATASET_LINK=${1?"Usage: $0 <link_to_dataset_archive_on_yandex_disk>"}
DOWNLOADS_DIR="tmp"
DATASET_DIR="dataset"

TMP_DATASET="${DOWNLOADS_DIR}/dataset.zip"

mkdir -p ${DOWNLOADS_DIR}

[ -f "${TMP_DATASET}" ] || wget -O "${TMP_DATASET}" $(yadisk-direct "${ACTUAL_DATASET_LINK}")
[ -d "${DATASET_DIR}" ] || unzip "${TMP_DATASET}" -d "${DATASET_DIR}"
