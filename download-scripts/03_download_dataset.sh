#!/bin/bash
# Step 3: Download OVEN dataset
# - JSONL (entity + query, train/val/test)
# - KG, knowledge_base, all_wikipedia_images
# - shard01~05 (train: 01~04, val/test: 05)

set -e

export WORKSPACE=${WORKSPACE:-/workspace}
export DATASET=${WORKSPACE}/KnowCoL/dataset
export HF_TOKEN=${HF_TOKEN}
export HF_HUB_TOKEN=${HF_TOKEN}

mkdir -p ${DATASET}/oven_data ${DATASET}/oven_images ${DATASET}/test_data

# ── ovenid2impath.csv ──────────────────────────────────────────────────────
echo "[0/5] ovenid2impath.csv (~505MB)..."
hf download ychenNLP/oven ovenid2impath.csv \
    --repo-type dataset --local-dir ${DATASET} --token ${HF_TOKEN}

# ── JSONL (동시 다운로드) ──────────────────────────────────────────────────
echo "[1/5] OVEN JSONL (entity + query, train/val/test)..."
BASE_URL="https://storage.googleapis.com/gresearch/open-vision-language/oven"

wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_entity_train.jsonl
wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_entity_val.jsonl
wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_query_train.jsonl
wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_query_val.jsonl
wget -q --show-progress -P ${DATASET}/test_data ${BASE_URL}/oven_entity_test.jsonl
wget -q --show-progress -P ${DATASET}/test_data ${BASE_URL}/oven_query_test.jsonl
wget -q --show-progress -P ${DATASET}/test_data ${BASE_URL}/oven_human.jsonl
echo "JSONL done."

# ── KG ────────────────────────────────────────────────────────────────────
echo "[2/5] wikidata_subgraph_v1 (~74MB)..."
hf download zhKingg/wikidata_oven_subgraph \
    --repo-type dataset --local-dir ${DATASET}/wikidata_subgraph_v1 --token ${HF_TOKEN}

# ── Knowledge Base ─────────────────────────────────────────────────────────
echo "[3/5] knowledge_base (~3.4GB)..."
hf download zhKingg/wikipedia_knowledge_base \
    --repo-type dataset --local-dir ${DATASET}/knowledge_base --token ${HF_TOKEN}

# ── Entity images ──────────────────────────────────────────────────────────
echo "[4/5] all_wikipedia_images (~32GB)..."
mkdir -p ${DATASET}/knowledge_base
hf download ychenNLP/oven all_wikipedia_images.tar \
    --repo-type dataset --local-dir ${DATASET} --token ${HF_TOKEN}
tar xf ${DATASET}/all_wikipedia_images.tar -C ${DATASET}/knowledge_base --no-same-owner
rm ${DATASET}/all_wikipedia_images.tar
echo "all_wikipedia done."

# ── Query images shard01~05 ────────────────────────────────────────────────
echo "[5/5] OVEN image shards 01~05..."

download_shard() {
    local s=$1
    echo "[shard${s}] Downloading..."
    hf download ychenNLP/oven shard${s}.tar \
        --repo-type dataset --local-dir ${DATASET}/oven_images --token ${HF_TOKEN}
    echo "[shard${s}] Extracting..."
    tar xf ${DATASET}/oven_images/shard${s}.tar -C ${DATASET}/oven_images --no-same-owner
    rm ${DATASET}/oven_images/shard${s}.tar
    echo "[shard${s}] Done"
}

download_shard 01
download_shard 02
download_shard 03
download_shard 04
download_shard 05

# shard05 tar은 내부 폴더명이 09/ → 05/로 rename
if [ -d "${DATASET}/oven_images/09" ] && [ ! -d "${DATASET}/oven_images/05" ]; then
    echo "Renaming oven_images/09 → 05 (shard05 internal folder name fix)..."
    mv ${DATASET}/oven_images/09 ${DATASET}/oven_images/05
fi

echo ""
echo "All done!"
du -sh ${DATASET}/oven_images/*/
