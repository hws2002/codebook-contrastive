#!/bin/bash
# Step 4: KnowCoL training (4x GPU DDP)

set -e

WORKSPACE=${WORKSPACE:-/workspace}
# conda 위치 유연하게 처리
PYTHON=${PYTHON_VQA:-""}
if [ -z "${PYTHON}" ]; then
    for p in "/venv/vqa/bin/python" \
              "${WORKSPACE}/miniconda3/envs/vqa/bin/python" \
              "/opt/conda/envs/vqa/bin/python" \
              "/root/miniconda3/envs/vqa/bin/python"; do
        [ -f "$p" ] && PYTHON="$p" && break
    done
fi
[ -z "${PYTHON}" ] && PYTHON=$(conda run -n vqa which python)
DATASET=${WORKSPACE}/KnowCoL/dataset
FILTERED=${DATASET}/filtered

GPUS=${GPUS:-"0,1,2,3"}
N_GPUS=$(echo ${GPUS} | tr ',' '\n' | wc -l)
BATCH=${BATCH:-512}      # per-GPU batch (total effective = BATCH * N_GPUS)
EPOCHS=${EPOCHS:-10}
RUN_NAME="knowcol-img-filtered-b${BATCH}x${N_GPUS}-${EPOCHS}ep"

# filtered JSONL 생성
echo "[0] Preparing filtered JSONL..."
${PYTHON} ${WORKSPACE}/codebook-contrastive/scripts/prepare_image_filtered_dataset.py \
    --dataset_dir ${DATASET} \
    --output_dir ${FILTERED}

# symlink to oven_data/
mkdir -p ${DATASET}/oven_data
ln -sf ${FILTERED}/oven_entity_train_img.jsonl ${DATASET}/oven_data/oven_entity_train_img.jsonl 2>/dev/null || true
ln -sf ${FILTERED}/oven_entity_val_img.jsonl   ${DATASET}/oven_data/oven_entity_val_img.jsonl 2>/dev/null || true

echo "[1] Starting KnowCoL training (${N_GPUS} GPUs, batch=${BATCH}/GPU)..."
cd ${WORKSPACE}/KnowCoL

CUDA_VISIBLE_DEVICES=${GPUS} WANDB_DISABLED=true \
${PYTHON} -m knowcol.training \
    datamodule.train_dataset_cfg.data_dir=${DATASET} \
    "datamodule.train_dataset_cfg.jsonl_files=[oven_entity_train_img.jsonl]" \
    datamodule.val_dataset_cfg.data_dir=${DATASET} \
    "datamodule.val_dataset_cfg.jsonl_files=[oven_entity_val_img.jsonl]" \
    datamodule.batch_size=${BATCH} \
    datamodule.num_workers=16 \
    trainer.devices=${N_GPUS} \
    trainer.strategy=ddp \
    trainer.max_epochs=${EPOCHS} \
    hydra.run.dir=checkpoints/${RUN_NAME} \
    2>&1 | tee checkpoints/${RUN_NAME}.log

echo "Done."
