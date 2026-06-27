# New Server Setup Scripts

## 실행 순서

### 사전 준비 (현재 서버에서)
1. KnowCoL fork → GitHub push
2. codebook-contrastive fork → GitHub push
3. `01_clone_repos.sh`의 `KNOWCOL_REPO`, `CB_REPO` 수정

### 새 서버 (vast.ai)에서
```bash
# 전부 /workspace 기준으로 동작
export WORKSPACE=/workspace
export HF_TOKEN="your_huggingface_token_here"  # Get from https://huggingface.co/settings/tokens

bash 00_setup_env.sh      # conda + pip 설치 (~10분)
source ~/.bashrc
conda activate vqa

bash 01_clone_repos.sh    # GitHub에서 코드 clone

bash 02_download_oven_jsonl.sh   # JSONL 다운 (~1분)
bash 03_download_oven_shards.sh  # 이미지 shard01~05 (~10분, 141GB)
bash 04_download_all_wiki.sh     # entity 이미지 (~5분, 32GB)

bash 05_run_knowcol.sh    # KnowCoL 학습 시작
```

## 파일 구조 (완성 후)
```
/workspace/
├── miniconda3/
├── KnowCoL/
│   ├── dataset/
│   │   ├── oven_data/
│   │   │   ├── oven_entity_train.jsonl  (4.9M samples)
│   │   │   └── oven_entity_val.jsonl   (126K samples)
│   │   ├── test_data/
│   │   │   └── oven_entity_test.jsonl  (709K samples)
│   │   ├── oven_images/
│   │   │   ├── 01/  (1M images, shard01)
│   │   │   ├── 02/  (1M images, shard02)
│   │   │   ├── 03/  (1M images, shard03)
│   │   │   ├── 04/  (925K images, shard04)
│   │   │   └── 05/  (val/test images, shard05)
│   │   ├── all_wikipedia_images/       (entity images)
│   │   └── wikidata_subgraph_v1/       (KG)
│   └── checkpoints/
└── codebook-contrastive/
    └── outputs/

## 필요 용량
| 항목 | 크기 |
|---|---|
| shard01~05 이미지 | ~141GB |
| all_wikipedia | ~40GB |
| JSONL + KG | ~3GB |
| 모델 체크포인트 | ~10GB |
| 합계 | ~200GB |

## 주의사항
- Stop 시 데이터 보존됨 (Destroy는 절대 금지)
- HF_TOKEN은 본인 토큰으로 교체
- 01_clone_repos.sh에서 GitHub fork URL 반드시 수정
```
