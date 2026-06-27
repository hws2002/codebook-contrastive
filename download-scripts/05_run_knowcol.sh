#!/bin/bash
# Step 5: Run KnowCoL training (full dataset, 4x GPU DDP)

set -e

WORKSPACE=${WORKSPACE:-/workspace}
PYTHON=${WORKSPACE}/miniconda3/envs/vqa/bin/python

cd ${WORKSPACE}/KnowCoL

CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_DISABLED=true \
PYTHONPATH=${WORKSPACE}/KnowCoL/knowcol:$PYTHONPATH \
  ${PYTHON} -m knowcol.training \
  datamodule.batch_size=512 \
  trainer.max_epochs=10 \
  trainer.devices=4 \
  trainer.strategy=ddp \
  hydra.run.dir=checkpoints/knowcol-full-b2048-10ep \
  2>&1 | tee checkpoints/knowcol-full.log
