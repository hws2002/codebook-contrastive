#!/bin/bash
# Step 2: Download OVEN JSONL files + KnowCoL KG
# These are text files, small and fast

set -e

WORKSPACE=${WORKSPACE:-/workspace}
DATASET=${WORKSPACE}/KnowCoL/dataset
HF_TOKEN=${HF_TOKEN:-""}   # optional, set if needed

mkdir -p ${DATASET}/oven_data ${DATASET}/test_data

echo "[1/2] Downloading OVEN JSONL from Google Storage..."
cd ${DATASET}/oven_data
wget -q --show-progress https://storage.googleapis.com/gresearch/open-vision-language/oven/oven_entity_train.jsonl
wget -q --show-progress https://storage.googleapis.com/gresearch/open-vision-language/oven/oven_entity_val.jsonl

cd ${DATASET}/test_data
wget -q --show-progress https://storage.googleapis.com/gresearch/open-vision-language/oven/oven_entity_test.jsonl

echo "[2/2] Downloading KnowCoL KG (wikidata_subgraph_v1)..."
# KG is bundled with KnowCoL dataset on HuggingFace
if [ -n "${HF_TOKEN}" ]; then
  huggingface-cli login --token ${HF_TOKEN}
fi
# Download from KnowCoL HF dataset if available
# Or copy from existing location if you have it

echo "JSONL download complete."
echo "  train: $(wc -l < ${DATASET}/oven_data/oven_entity_train.jsonl) samples"
echo "  val:   $(wc -l < ${DATASET}/oven_data/oven_entity_val.jsonl) samples"
echo "  test:  $(wc -l < ${DATASET}/test_data/oven_entity_test.jsonl) samples"
