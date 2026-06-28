export HF_TOKEN="xx"
export WANDB_API_KEY="xx"

# 새 서버 완전 셋업 가이드

---

## Step 0: 환경변수 설정
```bash
export WORKSPACE=/workspace
export HF_TOKEN="your_hf_token_here"
export WANDB_API_KEY="your_wandb_key_here"

# conda PATH 확인 (서버마다 다름)
# vast.ai: /opt/conda/bin/conda 또는 ~/miniconda3/bin/conda
which conda || source /opt/conda/etc/profile.d/conda.sh
```

---

## Step 1: codebook-contrastive 먼저 수동 clone
```bash
git clone https://github.com/hws2002/codebook-contrastive.git ${WORKSPACE}/codebook-contrastive
```

---

## Step 2: 나머지 repos clone (KnowCoL, mKG-RAG)
```bash
bash ${WORKSPACE}/codebook-contrastive/download-scripts/01_clone.sh
```

---

## Step 3: conda 환경 설치
```bash
bash ${WORKSPACE}/codebook-contrastive/download-scripts/02_setup_env.sh
```
> vqa env, faiss-gpu env 생성 (~20분)
> 설치 후 04/05 스크립트의 PYTHON 경로가 맞는지 확인:
> `${WORKSPACE}/miniconda3/envs/vqa/bin/python` → 없으면 `which python` 으로 확인

---

## Step 4: 데이터 다운로드 (shard01~05 + all_wikipedia + KG)
```bash
export HF_TOKEN="your_token"
bash ${WORKSPACE}/codebook-contrastive/download-scripts/03_download_dataset.sh
```
> 약 3~5시간 소요

---

## Step 5: KnowCoL 학습 (터미널 1)
```bash
# 4x GPU
GPUS=0,1,2,3 BATCH=512 bash ${WORKSPACE}/codebook-contrastive/download-scripts/04_run_knowcol.sh

# 1x GPU
GPUS=0 BATCH=256 bash ${WORKSPACE}/codebook-contrastive/download-scripts/04_run_knowcol.sh
```

---

## Step 6: codebook-contrastive 학습 (터미널 2)
```bash
# 4x GPU (Step 5와 동시 실행 불가 - GPU 충돌)
GPUS=0,1,2,3 BATCH=256 N_HN=256 bash ${WORKSPACE}/codebook-contrastive/download-scripts/05_run_codebook.sh

# 1x GPU (Step 5와 다른 GPU 쓰면 동시 가능)
GPUS=1 BATCH=256 N_HN=256 bash ${WORKSPACE}/codebook-contrastive/download-scripts/05_run_codebook.sh
```

---

## 주의사항
- Step 5, 6은 Step 4 완료 후 실행
- 4x GPU로 둘 다 돌리면 GPU 충돌 → 순차 실행하거나 8GPU 서버에서 각각 0-3 / 4-7 사용
- shard 추가 시: 해당 shard만 받고 Step 5/6 재실행 → filtered JSONL 자동 갱신
- HF_TOKEN은 절대 코드에 하드코딩 금지
- conda 경로가 다를 경우 04/05 스크립트 상단 PYTHON_VQA, PYTHON_FAISS 경로 수정 필요
