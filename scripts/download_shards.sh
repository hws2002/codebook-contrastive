#!/bin/bash
# shard04: 3시간 후 다운로드
# shard05: 6시간 후 다운로드

HF=/data/guozhiqiang/hanyoushuo/conda_envs/mmrec/bin/huggingface-cli
IMG=/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL/dataset/oven_images

download_and_extract() {
    local shard=$1
    echo "[shard${shard}] Downloading..."
    ${HF} download ychenNLP/oven shard${shard}.tar \
        --repo-type dataset --local-dir ${IMG}
    echo "[shard${shard}] Extracting..."
    tar xf ${IMG}/shard${shard}.tar -C ${IMG} --no-same-owner
    rm ${IMG}/shard${shard}.tar
    echo "[shard${shard}] DONE"
}

# shard04: 3시간 후
(sleep 10800 && download_and_extract 04) &

# shard05: 6시간 후
(sleep 21600 && download_and_extract 05) &

echo "Scheduled: shard04 in 3h, shard05 in 6h"
wait
echo "All downloads done."
