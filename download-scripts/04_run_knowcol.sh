#!/bin/bash
# Step 4: KnowCoL training
# filtered JSONL 기반으로 학습 (image 있는 샘플만)

set -e

WORKSPACE=${WORKSPACE:-/workspace}
PYTHON=${WORKSPACE}/miniconda3/envs/vqa/bin/python
DATASET=${WORKSPACE}/KnowCoL/dataset
FILTERED=${DATASET}/filtered

GPU=${GPU:-0}
BATCH=${BATCH:-256}
EPOCHS=${EPOCHS:-10}
RUN_NAME="knowcol-img-filtered-b${BATCH}-${EPOCHS}ep"

# filtered JSONL 생성
echo "[0] Preparing filtered JSONL..."
${PYTHON} ${WORKSPACE}/codebook-contrastive/scripts/prepare_image_filtered_dataset.py \
    --dataset_dir ${DATASET} \
    --output_dir ${FILTERED}

# symlink to oven_data/ (KnowCoL reads from data_dir/oven_data/)
mkdir -p ${DATASET}/oven_data
ln -sf ${FILTERED}/oven_entity_train_img.jsonl ${DATASET}/oven_data/oven_entity_train_img.jsonl 2>/dev/null || true
ln -sf ${FILTERED}/oven_entity_val_img.jsonl   ${DATASET}/oven_data/oven_entity_val_img.jsonl 2>/dev/null || true

echo "[1] Starting KnowCoL training..."
cd ${WORKSPACE}/KnowCoL

CUDA_VISIBLE_DEVICES=${GPU} WANDB_DISABLED=true \
${PYTHON} -m knowcol.training \
    datamodule.train_dataset_cfg.data_dir=${DATASET} \
    "datamodule.train_dataset_cfg.jsonl_files=[oven_entity_train_img.jsonl]" \
    datamodule.val_dataset_cfg.data_dir=${DATASET} \
    "datamodule.val_dataset_cfg.jsonl_files=[oven_entity_val_img.jsonl]" \
    datamodule.batch_size=${BATCH} \
    datamodule.num_workers=16 \
    trainer.max_epochs=${EPOCHS} \
    hydra.run.dir=checkpoints/${RUN_NAME} \
    2>&1 | tee checkpoints/${RUN_NAME}.log

echo "Done."
