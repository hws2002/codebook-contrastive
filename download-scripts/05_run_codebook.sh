#!/bin/bash
# Step 5: codebook-contrastive pipeline (4x GPU DDP)
# Phase 0: entity embedding 추출 (single GPU, 1회만)
# Phase 1: text codebook 빌드 (1회만)
# Phase 2: image codebook 빌드 (1회만)
# Phase 3: filtered JSONL 생성
# Phase 4: 학습 (dual codebook HN, 4x GPU DDP)

set -e

WORKSPACE=${WORKSPACE:-/workspace}
PYTHON_VQA=${WORKSPACE}/miniconda3/envs/vqa/bin/python
PYTHON_FAISS=${WORKSPACE}/miniconda3/envs/faiss-gpu/bin/python
CB=${WORKSPACE}/codebook-contrastive
DATASET=${WORKSPACE}/KnowCoL/dataset
EMB_DIR=${CB}/outputs/entity_embs_img
RQ_TEXT=${CB}/outputs/codebook_rq_text
RQ_IMG=${CB}/outputs/codebook_rq_img
FILTERED=${DATASET}/filtered

GPUS=${GPUS:-"0,1,2,3"}
N_GPUS=$(echo ${GPUS} | tr ',' '\n' | wc -l)
BATCH=${BATCH:-256}      # per-GPU batch
EPOCHS=${EPOCHS:-10}
N_HN=${N_HN:-256}
RUN_NAME="cb-rq-dual-hn-b${BATCH}x${N_GPUS}-nhn${N_HN}-${EPOCHS}ep"
PHASE0_GPU=$(echo ${GPUS} | cut -d',' -f1)   # Phase 0~2는 첫 번째 GPU만 사용

cd ${WORKSPACE}/KnowCoL

# Phase 0: entity embedding 추출
if [ -f "${EMB_DIR}/entity_text_embs.npy" ]; then
    echo "[Phase 0] Entity embeddings exist, skipping."
else
    echo "[Phase 0] Extracting entity embeddings (text + image)..."
    mkdir -p ${EMB_DIR}
    CUDA_VISIBLE_DEVICES=${PHASE0_GPU} ${PYTHON_VQA} ${CB}/scripts/extract_entity_embeddings.py \
        --output_dir ${EMB_DIR} \
        --kb_path ${DATASET}/knowledge_base \
        --kg_dir ${DATASET}/wikidata_subgraph_v1 \
        2>&1 | tee ${EMB_DIR}/extract.log
fi

# Phase 1: text codebook
if [ -f "${RQ_TEXT}/entity_codes_rq.npy" ]; then
    echo "[Phase 1] Text codebook exists, skipping."
else
    echo "[Phase 1] Building text RQ codebook..."
    mkdir -p ${RQ_TEXT} /tmp/text_emb_dir
    cp ${EMB_DIR}/entity_text_embs.npy /tmp/text_emb_dir/entity_text_embs.npy
    cp ${EMB_DIR}/entity_id_list.json  /tmp/text_emb_dir/
    ${PYTHON_FAISS} ${CB}/scripts/build_codebook_rq.py \
        --emb_dir /tmp/text_emb_dir \
        --output_dir ${RQ_TEXT} \
        2>&1 | tee ${RQ_TEXT}/build.log
fi

# Phase 2: image codebook
if [ -f "${RQ_IMG}/entity_codes_rq.npy" ]; then
    echo "[Phase 2] Image codebook exists, skipping."
else
    echo "[Phase 2] Building image RQ codebook..."
    mkdir -p ${RQ_IMG} /tmp/img_emb_dir
    cp ${EMB_DIR}/entity_img_embs.npy /tmp/img_emb_dir/entity_text_embs.npy
    cp ${EMB_DIR}/entity_id_list.json  /tmp/img_emb_dir/
    ${PYTHON_FAISS} ${CB}/scripts/build_codebook_rq.py \
        --emb_dir /tmp/img_emb_dir \
        --output_dir ${RQ_IMG} \
        2>&1 | tee ${RQ_IMG}/build.log
fi

# Phase 3: filtered JSONL
echo "[Phase 3] Preparing filtered JSONL..."
${PYTHON_VQA} ${CB}/scripts/prepare_image_filtered_dataset.py \
    --dataset_dir ${DATASET} \
    --output_dir ${FILTERED}

# Phase 4: 학습 (4x GPU DDP)
echo "[Phase 4] Training codebook-contrastive (${N_GPUS} GPUs, batch=${BATCH}/GPU)..."
CUDA_VISIBLE_DEVICES=${GPUS} WANDB_DISABLED=true \
${PYTHON_VQA} ${CB}/scripts/train.py \
    --data_dir ${DATASET} \
    --train_jsonl oven_entity_train_img.jsonl \
    --val_jsonl oven_entity_val_img.jsonl \
    --test_jsonl oven_entity_test.jsonl \
    --emb_dir ${EMB_DIR} \
    --codebook_type rq \
    --codebook_dir ${RQ_TEXT} \
    --codebook_img_dir ${RQ_IMG} \
    --hn_mode per_sample \
    --n_hard_neg ${N_HN} \
    --batch_size ${BATCH} \
    --num_workers 16 \
    --gpus ${N_GPUS} \
    --max_epochs ${EPOCHS} \
    --run_name ${RUN_NAME} \
    --output_dir ${CB}/outputs/checkpoints/${RUN_NAME} \
    2>&1 | tee ${CB}/outputs/checkpoints/${RUN_NAME}.log

echo "Done."
