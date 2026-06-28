#!/bin/bash
# Step 2: Setup conda environments
# - vqa: from KnowCoL/environment.yaml (for KnowCoL + codebook-contrastive)
# - faiss-gpu: for codebook building (PQ/RQ)

set -e

WORKSPACE=${WORKSPACE:-/workspace}

echo "[1/2] Creating vqa env from KnowCoL/environment.yaml..."
conda env create -f ${WORKSPACE}/KnowCoL/environment.yaml -n vqa
conda run -n vqa pip install -e ${WORKSPACE}/KnowCoL -q
echo "vqa done."

echo "[2/2] Creating faiss-gpu env..."
conda create -n faiss-gpu python=3.10 -y
conda run -n faiss-gpu pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 -q
conda install -n faiss-gpu -c conda-forge faiss-gpu -y
conda run -n faiss-gpu pip install numpy tqdm scikit-learn huggingface_hub -q
echo "faiss-gpu done."

echo "[3/3] wandb login..."
# export WANDB_API_KEY="your_key" 먼저 실행 (https://wandb.ai/settings)
conda run -n vqa wandb login ${WANDB_API_KEY}

echo "Done!"
