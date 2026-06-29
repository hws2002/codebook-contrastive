#!/bin/bash
# Step 4: KnowCoL training (4x GPU DDP)

set -e

export WORKSPACE=${WORKSPACE:-/workspace}
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
export PYTHON=${PYTHON_VQA:-$(_find_python vqa)}
export DATASET=${WORKSPACE}/KnowCoL/dataset
export FILTERED=${DATASET}/filtered

export GPUS=${GPUS:-"0,1,2,3"}
export N_GPUS=$(echo ${GPUS} | tr ',' '\n' | wc -l)
export BATCH=${BATCH:-512}
export EPOCHS=${EPOCHS:-10}
export RUN_NAME="knowcol-img-filtered-b${BATCH}x${N_GPUS}-${EPOCHS}ep"

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
