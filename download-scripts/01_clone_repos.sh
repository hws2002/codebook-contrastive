#!/bin/bash
# Step 1: Clone KnowCoL (원본) + codebook-contrastive (hwooseok123 fork)
# KnowCoL 원본 clone 후 우리 수정사항 patch 적용

set -e

WORKSPACE=${WORKSPACE:-/workspace}

echo "[1/3] Cloning KnowCoL (original)..."
git clone https://github.com/boschresearch/KnowCoL.git ${WORKSPACE}/KnowCoL
cd ${WORKSPACE}/KnowCoL
pip install -e . -q
echo "KnowCoL installed."

echo "[2/3] Applying our modifications (patch)..."
# patch 파일은 codebook-contrastive repo 안에 있음
PATCH_FILE=${WORKSPACE}/codebook-contrastive/download-scripts/knowcol_changes.patch
if [ -f "${PATCH_FILE}" ]; then
    git apply ${PATCH_FILE}
    echo "Patch applied successfully."
else
    echo "WARNING: patch file not found at ${PATCH_FILE}"
    echo "Manually copy modified files from codebook-contrastive/download-scripts/"
fi

echo "[3/3] Cloning codebook-contrastive (hwooseok123)..."
git clone https://github.com/hwooseok123/codebook-contrastive.git ${WORKSPACE}/codebook-contrastive

# Update absolute paths in KnowCoL conf to current workspace
echo "Updating conf data_dir paths to ${WORKSPACE}..."
find ${WORKSPACE}/KnowCoL/conf -name "*.yaml" \
  -exec sed -i "s|/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL|${WORKSPACE}/KnowCoL|g" {} \;

echo ""
echo "Done!"
echo "  KnowCoL:              ${WORKSPACE}/KnowCoL"
echo "  codebook-contrastive: ${WORKSPACE}/codebook-contrastive"
