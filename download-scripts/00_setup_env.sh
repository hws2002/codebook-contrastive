#!/bin/bash
# Step 0: Install miniconda + python packages
# Run once on new instance

set -e

WORKSPACE=${WORKSPACE:-/workspace}

# 1. Miniconda
echo "[1/3] Installing Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ${WORKSPACE}/miniconda3
rm /tmp/miniconda.sh
export PATH="${WORKSPACE}/miniconda3/bin:$PATH"
echo "export PATH=\"${WORKSPACE}/miniconda3/bin:\$PATH\"" >> ~/.bashrc

# 2. conda env
echo "[2/3] Creating conda env 'vqa' (python 3.10)..."
conda create -n vqa python=3.10 -y
source ${WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate vqa

# 3. pip packages
echo "[3/3] Installing packages..."
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install pytorch-lightning==2.2.5 open_clip_torch hydra-core omegaconf -q
pip install transformers wandb tqdm pillow scikit-learn faiss-cpu -q
pip install huggingface_hub -q

echo "Done! Run: conda activate vqa"
