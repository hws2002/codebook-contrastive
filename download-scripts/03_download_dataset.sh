#!/bin/bash
# Step 3: Download OVEN dataset (train + val + test)
# test는 GT 없어서 평가 불가, 단 inference용으로 다운받아둠
# KnowCoL과 codebook-contrastive가 같은 dataset 경로 공유

set -e

WORKSPACE=${WORKSPACE:-/workspace}
DATASET=${WORKSPACE}/KnowCoL/dataset
HF_TOKEN=${HF_TOKEN}  # export HF_TOKEN="your_token" 먼저 실행

mkdir -p ${DATASET}/oven_data ${DATASET}/oven_images

huggingface-cli login --token ${HF_TOKEN}

# ── ovenid2impath.csv (505MB, image_id → archive path 매핑) ────────────────
echo "[0/5] ovenid2impath.csv..."
huggingface-cli download ychenNLP/oven ovenid2impath.csv \
    --repo-type dataset --local-dir ${DATASET}

# ── JSONL ──────────────────────────────────────────────────────────────────
echo "[1/5] OVEN JSONL (entity + query, train/val/test 전부)..."
mkdir -p ${DATASET}/test_data
BASE_URL="https://storage.googleapis.com/gresearch/open-vision-language/oven"

# train + val → oven_data/
wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_entity_train.jsonl &
wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_entity_val.jsonl &
wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_query_train.jsonl &
wget -q --show-progress -P ${DATASET}/oven_data ${BASE_URL}/oven_query_val.jsonl &

# test → test_data/
wget -q --show-progress -P ${DATASET}/test_data ${BASE_URL}/oven_entity_test.jsonl &
wget -q --show-progress -P ${DATASET}/test_data ${BASE_URL}/oven_query_test.jsonl &
wget -q --show-progress -P ${DATASET}/test_data ${BASE_URL}/oven_human.jsonl &
wait

# ── KG ─────────────────────────────────────────────────────────────────────
echo "[2/5] wikidata_subgraph_v1 (~74MB)..."
huggingface-cli download zhKingg/wikidata_oven_subgraph \
    --repo-type dataset --local-dir ${DATASET}/wikidata_subgraph_v1

# ── Knowledge Base ──────────────────────────────────────────────────────────
echo "[3/5] knowledge_base (~3.4GB)..."
huggingface-cli download zhKingg/wikipedia_knowledge_base \
    --repo-type dataset --local-dir ${DATASET}/knowledge_base

# ── Entity images (all_wikipedia) ──────────────────────────────────────────
echo "[4/5] all_wikipedia_images (~32GB)..."
mkdir -p ${DATASET}/knowledge_base
huggingface-cli download ychenNLP/oven all_wikipedia_images.tar \
    --repo-type dataset --local-dir ${DATASET}
tar xf ${DATASET}/all_wikipedia_images.tar -C ${DATASET}/knowledge_base --no-same-owner
rm ${DATASET}/all_wikipedia_images.tar

# ── Query images shard01~08 ─────────────────────────────────────────────────
echo "[5/5] OVEN image shards 01~08..."
download_shard() {
    local s=$1
    huggingface-cli download ychenNLP/oven shard${s}.tar \
        --repo-type dataset --local-dir ${DATASET}/oven_images
    tar xf ${DATASET}/oven_images/shard${s}.tar -C ${DATASET}/oven_images --no-same-owner
    rm ${DATASET}/oven_images/shard${s}.tar
    echo "shard${s} done"
}

download_shard 01 & download_shard 02 & wait
download_shard 03 & download_shard 04 & wait
download_shard 05 & download_shard 06 & wait
download_shard 07 & download_shard 08 & wait

echo ""
echo "All done! Dataset at ${DATASET}"
du -sh ${DATASET}/
