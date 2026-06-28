# New Server Setup (vast.ai)

## 실행 순서

```bash
export WORKSPACE=/workspace
export HF_TOKEN="your_huggingface_token_here"

# 1. 코드 clone
bash ${WORKSPACE}/codebook-contrastive/download-scripts/01_clone.sh

# 2. conda 환경 설치
bash ${WORKSPACE}/codebook-contrastive/download-scripts/02_setup_env.sh

# 3. 데이터 다운로드 (OVEN train+val)
bash ${WORKSPACE}/codebook-contrastive/download-scripts/03_download_dataset.sh

# 4. KnowCoL 학습
bash ${WORKSPACE}/codebook-contrastive/download-scripts/04_run_knowcol.sh

# 5. codebook-contrastive 학습
bash ${WORKSPACE}/codebook-contrastive/download-scripts/05_run_codebook.sh
```

## 첫 명령어 (새 서버 접속 직후)

```bash
git clone https://github.com/hws2002/codebook-contrastive.git /workspace/codebook-contrastive
export WORKSPACE=/workspace
export HF_TOKEN="your_token"
bash /workspace/codebook-contrastive/download-scripts/01_clone.sh
```

## 필요 shards

| shard | 용도 | 크기 |
|---|---|---|
| shard01~04 | train 이미지 | ~95GB |
| shard05 | val 이미지 | ~42GB |
| shard00 | HF에 없음 → black image | - |
| shard06~08 | test 전용 → 불필요 | - |

## 필요 용량

| 항목 | 크기 |
|---|---|
| OVEN 이미지 shard01~05 | ~141GB |
| all_wikipedia_images | ~40GB |
| JSONL + KG + knowledge_base | ~5GB |
| 체크포인트 | ~10GB |
| 합계 | ~200GB |

## 주의사항
- mKG-RAG repo URL은 01_clone.sh에서 직접 수정
- vast.ai instance는 Stop만, Destroy 절대 금지
- HF_TOKEN은 환경변수로만, 절대 코드에 하드코딩 금지
