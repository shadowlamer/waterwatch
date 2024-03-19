#!/bin/bash

ROOT="dataset"
VOLUMES="1.5 5 12 19"
PERCENTS="10 20 30 40 50 60 70 80 90 100"

for VOLUME in ${VOLUMES}; do
  for PERCENT in ${PERCENTS}; do
    mkdir -p "${ROOT}/${VOLUME}/${PERCENT}"
  done
done

mkdir -p "${ROOT}/empty"
mkdir -p "${ROOT}/hands"
mkdir -p "${ROOT}/incident"
