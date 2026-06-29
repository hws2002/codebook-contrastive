#!/bin/bash
# Step 5: codebook-contrastive pipeline (4x GPU DDP)
# Phase 0: entity embedding 추출 (1회만)
# Phase 1: KMeans text codebook (RQ-VAE warm-start용, 1회만)
# Phase 2: KMeans image codebook (RQ-VAE warm-start용, 1회만)
# Phase 3: RQ-VAE 학습 (text + image, 1회만)
# Phase 4: filtered JSONL 생성
# Phase 5: 학습 (dual RQ-VAE codebook HN, 4x GPU DDP)

set -e

WORKSPACE=${WORKSPACE:-/workspace}
_find_python() {
    local env=$1
    for p in "/venv/${env}/bin/python" \
              "${WORKSPACE}/miniconda3/envs/${env}/bin/python" \
              "/opt/conda/envs/${env}/bin/python" \
              "/root/miniconda3/envs/${env}/bin/python"; do
        [ -f "$p" ] && echo "$p" && return
    done
    conda run -n ${env} which python
}
PYTHON_VQA=${PYTHON_VQA:-$(_find_python vqa)}
PYTHON_FAISS=${PYTHON_FAISS:-$(_find_python faiss-gpu)}
CB=${WORKSPACE}/codebook-contrastive
DATASET=${WORKSPACE}/KnowCoL/dataset
EMB_DIR=${CB}/outputs/entity_embs_img
RQ_TEXT=${CB}/outputs/codebook_rq_text       # KMeans (warm-start용)
RQ_IMG=${CB}/outputs/codebook_rq_img         # KMeans (warm-start용)
RQVAE_TEXT=${CB}/outputs/rqvae_text          # RQ-VAE 학습 결과
RQVAE_IMG=${CB}/outputs/rqvae_img            # RQ-VAE 학습 결과
FILTERED=${DATASET}/filtered

GPUS=${GPUS:-"0,1,2,3"}
N_GPUS=$(echo ${GPUS} | tr ',' '\n' | wc -l)
BATCH=${BATCH:-256}
EPOCHS=${EPOCHS:-10}
RQVAE_EPOCHS=${RQVAE_EPOCHS:-100}
N_HN=${N_HN:-256}
RUN_NAME="cb-rqvae-dual-hn-b${BATCH}x${N_GPUS}-nhn${N_HN}-${EPOCHS}ep"
PHASE0_GPU=$(echo ${GPUS} | cut -d',' -f1)

cd ${WORKSPACE}/KnowCoL

# Phase 0: entity embedding 추출
if [ -f "${EMB_DIR}/entity_text_embs.npy" ]; then
    echo "[Phase 0] Entity embeddings exist, skipping."
else
    echo "[Phase 0] Extracting entity embeddings (text + image)..."
    mkdir -p ${EMB_DIR}
    KNOWCOL_DIR=${WORKSPACE}/KnowCoL \
    CUDA_VISIBLE_DEVICES=${PHASE0_GPU} ${PYTHON_VQA} ${CB}/scripts/extract_entity_embeddings.py \
        --output_dir ${EMB_DIR} \
        --kb_path ${DATASET}/knowledge_base \
        --kg_dir ${DATASET}/wikidata_subgraph_v1 \
        2>&1 | tee ${EMB_DIR}/extract.log
fi

# Phase 1: KMeans text codebook (RQ-VAE warm-start용)
if [ -f "${RQ_TEXT}/entity_codes_rq.npy" ]; then
    echo "[Phase 1] KMeans text codebook exists, skipping."
else
    echo "[Phase 1] Building KMeans text codebook (warm-start)..."
    mkdir -p ${RQ_TEXT} /tmp/text_emb_dir
    cp ${EMB_DIR}/entity_text_embs.npy /tmp/text_emb_dir/entity_text_embs.npy
    cp ${EMB_DIR}/entity_id_list.json  /tmp/text_emb_dir/
    ${PYTHON_FAISS} ${CB}/scripts/build_codebook_rq.py \
        --emb_dir /tmp/text_emb_dir \
        --output_dir ${RQ_TEXT} \
        2>&1 | tee ${RQ_TEXT}/build.log
fi

# Phase 2: KMeans image codebook (RQ-VAE warm-start용)
if [ -f "${RQ_IMG}/entity_codes_rq.npy" ]; then
    echo "[Phase 2] KMeans image codebook exists, skipping."
else
    echo "[Phase 2] Building KMeans image codebook (warm-start)..."
    mkdir -p ${RQ_IMG} /tmp/img_emb_dir
    cp ${EMB_DIR}/entity_img_embs.npy /tmp/img_emb_dir/entity_text_embs.npy
    cp ${EMB_DIR}/entity_id_list.json  /tmp/img_emb_dir/
    ${PYTHON_FAISS} ${CB}/scripts/build_codebook_rq.py \
        --emb_dir /tmp/img_emb_dir \
        --output_dir ${RQ_IMG} \
        2>&1 | tee ${RQ_IMG}/build.log
fi

# Phase 3: RQ-VAE 학습
cd ${CB}

if [ -f "${RQVAE_TEXT}/entity_codes_rq.npy" ]; then
    echo "[Phase 3a] Text RQ-VAE exists, skipping."
else
    echo "[Phase 3a] Training text RQ-VAE..."
    mkdir -p ${RQVAE_TEXT}
    CUDA_VISIBLE_DEVICES=${PHASE0_GPU} ${PYTHON_VQA} scripts/train_rqvae.py \
        --emb_path ${EMB_DIR}/entity_text_embs.npy \
        --init_codebook_dir ${RQ_TEXT} \
        --output_dir ${RQVAE_TEXT} \
        --epochs ${RQVAE_EPOCHS} \
        --batch_size 512 \
        --gpu ${PHASE0_GPU} \
        2>&1 | tee ${RQVAE_TEXT}/train.log
fi

if [ -f "${RQVAE_IMG}/entity_codes_rq.npy" ]; then
    echo "[Phase 3b] Image RQ-VAE exists, skipping."
else
    echo "[Phase 3b] Training image RQ-VAE..."
    mkdir -p ${RQVAE_IMG}
    CUDA_VISIBLE_DEVICES=${PHASE0_GPU} ${PYTHON_VQA} scripts/train_rqvae.py \
        --emb_path ${EMB_DIR}/entity_img_embs.npy \
        --init_codebook_dir ${RQ_IMG} \
        --output_dir ${RQVAE_IMG} \
        --epochs ${RQVAE_EPOCHS} \
        --batch_size 512 \
        --gpu ${PHASE0_GPU} \
        2>&1 | tee ${RQVAE_IMG}/train.log
fi

# Phase 4: filtered JSONL
cd ${WORKSPACE}/KnowCoL
echo "[Phase 4] Preparing filtered JSONL..."
${PYTHON_VQA} ${CB}/scripts/prepare_image_filtered_dataset.py \
    --dataset_dir ${DATASET} \
    --output_dir ${FILTERED}

mkdir -p ${DATASET}/oven_data
ln -sf ${FILTERED}/oven_entity_train_img.jsonl ${DATASET}/oven_data/oven_entity_train_img.jsonl 2>/dev/null || true
ln -sf ${FILTERED}/oven_entity_val_img.jsonl   ${DATASET}/oven_data/oven_entity_val_img.jsonl   2>/dev/null || true
ln -sf ${FILTERED}/oven_query_train_img.jsonl  ${DATASET}/oven_data/oven_query_train_img.jsonl  2>/dev/null || true
ln -sf ${FILTERED}/oven_query_val_img.jsonl    ${DATASET}/oven_data/oven_query_val_img.jsonl    2>/dev/null || true

# Phase 5: 학습 (RQ-VAE codebook 사용)
echo "[Phase 5] Training codebook-contrastive (${N_GPUS} GPUs, batch=${BATCH}/GPU)..."
cd ${CB}
CUDA_VISIBLE_DEVICES=${GPUS} WANDB_DISABLED=true \
PYTHONPATH=${CB} ${PYTHON_VQA} scripts/train.py \
    --data_dir ${DATASET} \
    --train_jsonl oven_entity_train_img.jsonl \
    --val_jsonl oven_entity_val_img.jsonl \
    --test_jsonl oven_entity_test.jsonl \
    --emb_dir ${EMB_DIR} \
    --codebook_type rq \
    --codebook_dir ${RQVAE_TEXT} \
    --codebook_img_dir ${RQVAE_IMG} \
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
