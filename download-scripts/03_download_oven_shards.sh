#!/bin/bash
# Step 3: Download OVEN image shards (shard01~05)
# shard01~04: train images  shard05: val/test images
# Total: ~141GB extracted
# With fast internet (2932 Mbps) = ~10분

set -e

WORKSPACE=${WORKSPACE:-/workspace}
IMG_DIR=${WORKSPACE}/KnowCoL/dataset/oven_images
HF_TOKEN=${HF_TOKEN}  # Set via: export HF_TOKEN="your_token"

mkdir -p ${IMG_DIR}
cd ${IMG_DIR}

echo "HF login..."
huggingface-cli login --token ${HF_TOKEN}

# Download all shards in parallel (2 at a time)
download_and_extract() {
    local shard=$1
    echo "[shard${shard}] Downloading..."
    huggingface-cli download ychenNLP/oven shard${shard}.tar \
        --repo-type dataset --local-dir ${IMG_DIR}
    echo "[shard${shard}] Extracting..."
    tar xf ${IMG_DIR}/shard${shard}.tar -C ${IMG_DIR}
    rm ${IMG_DIR}/shard${shard}.tar
    echo "[shard${shard}] DONE"
}

echo "Downloading shard01 + shard02 in parallel..."
download_and_extract 01 &
download_and_extract 02 &
wait

echo "Downloading shard03 + shard04 in parallel..."
download_and_extract 03 &
download_and_extract 04 &
wait

echo "Downloading shard05 (val/test images)..."
download_and_extract 05

echo "All shards done!"
ls -lh ${IMG_DIR}/
