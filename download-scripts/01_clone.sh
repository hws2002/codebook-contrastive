#!/bin/bash
# Step 1: Clone all repos

set -e

export WORKSPACE=${WORKSPACE:-/workspace}

echo "[1/3] Cloning codebook-contrastive..."
git clone https://github.com/hws2002/codebook-contrastive.git ${WORKSPACE}/codebook-contrastive

echo "[2/3] Cloning KnowCoL (hws2002 fork)..."
git clone https://github.com/hws2002/KnowCoL.git ${WORKSPACE}/KnowCoL

# absolute path 업데이트
find ${WORKSPACE}/KnowCoL/conf -name "*.yaml" \
    -exec sed -i "s|/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL|${WORKSPACE}/KnowCoL|g" {} \;

echo "[3/3] Cloning mKG-RAG..."
git clone https://github.com/xandery-geek/mKG-RAG.git ${WORKSPACE}/mKG-RAG

echo "Done! Repos at ${WORKSPACE}"
