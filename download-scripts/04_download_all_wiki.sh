#!/bin/bash
# Step 4: Download all_wikipedia_images.tar (entity images, 32.4GB)
# 추출 후 KnowCoL의 entity image로 사용

set -e

WORKSPACE=${WORKSPACE:-/workspace}
DATASET=${WORKSPACE}/KnowCoL/dataset
HF_TOKEN=${HF_TOKEN}  # Set via: export HF_TOKEN="your_token"

mkdir -p ${DATASET}
cd ${DATASET}

echo "Downloading all_wikipedia_images.tar (~32.4GB)..."
huggingface-cli login --token ${HF_TOKEN}
huggingface-cli download ychenNLP/oven all_wikipedia_images.tar \
    --repo-type dataset --local-dir ${DATASET}

echo "Extracting..."
mkdir -p ${DATASET}/all_wikipedia_images
tar xf ${DATASET}/all_wikipedia_images.tar -C ${DATASET}/all_wikipedia_images
rm ${DATASET}/all_wikipedia_images.tar

echo "Done!"
du -sh ${DATASET}/all_wikipedia_images/
