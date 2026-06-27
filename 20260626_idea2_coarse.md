# 20260626 idea2 coarse experiment

> **주의**: 모든 수치는 shard01-only partial reproduction 기준. 공식 OVEN full split 결과 아님.
> entity 이미지 없음 (all_wikipedia_images 미다운로드 → text-only fallback)

---

## 공통 설정

| 항목 | 값 |
|------|---|
| dataset | dataset_01 (shard01, 114K train / 6K val / 6K test) |
| entity store | 32,122 entities (wikidata_subgraph_v1/entity.txt) |
| backbone | CLIP ViT-L-14 (commonpool_xl_s13b_b90k), frozen |
| entity image | ❌ 없음 (text-only fallback) |
| eval metric | Recall@1 / Recall@5 / Recall@10 |
| test set | 6,006 samples (entity_seen split) |
| GPU | A800 80GB × 1 |

---

## 실험 목록

### [1] KnowCoL baseline (batch=32)

| 하이퍼파라미터 | 값 |
|---|---|
| trainable params | 26.2M (KGE 24.7M + LP1/LP2 1.2M) |
| loss | alignment + KE loss + proxy loss |
| batch_size | 32 |
| epochs | 10 |
| entity 표현 | QID별 KGE lookup embedding |
| optimizer | AdamW + CosineAnnealingLR |

| Metric | Score |
|--------|-------|
| Recall@1 | 42.64% |
| Recall@5 | 72.44% |
| Recall@10 | 80.72% |

checkpoint: `KnowCoL/checkpoints/2026-06-26/18-43-49/`

---

### [2] codebook-contrastive — in-batch only (⚠️ bad Phase 0)

| 하이퍼파라미터 | 값 |
|---|---|
| trainable params | 2.4M (QueryEntityHead 1.2M + EntityItemHead 1.2M) |
| MLP | Linear→GELU→LayerNorm→Linear, in=1536, hidden=768, out=512 |
| loss | symmetric InfoNCE (in-batch only) |
| batch_size | 256 |
| epochs | 10 |
| lr | 1e-4 (CosineAnnealingLR) |
| hard negatives | 없음 |
| ⚠️ Phase 0 문제 | entity 6,722개(20.9%) empty text → 동일 CLIP embedding |

| Metric | Score |
|--------|-------|
| Recall@1 | 62.64% |
| Recall@5 | 86.46% |
| Recall@10 | 91.33% |

checkpoint: `outputs/checkpoints/codebook-inbatch-shard01/`

---

### [3] codebook-contrastive — KMeans HN (⚠️ bad Phase 0 + bad codebook)

| 하이퍼파라미터 | 값 |
|---|---|
| [2]와 동일 + | |
| codebook | KMeans, depth=1, n_clusters=256 |
| HN mode | per_sample (pool = B+K per query) |
| n_hard_neg | 16 |
| ⚠️ codebook 문제 | empty text 6,722개 → max bucket=6,722 (ideal=125) |

| Metric | Score |
|--------|-------|
| Recall@1 | 68.68% |
| Recall@5 | 92.54% |
| Recall@10 | 96.09% |

checkpoint: `outputs/checkpoints/codebook-hn-shard01/`

---

### [4] codebook-contrastive — in-batch only (✅ fixed Phase 0, batch=256)

| 하이퍼파라미터 | 값 |
|---|---|
| [2]와 동일 | |
| ✅ Phase 0 수정 | title+summary fallback → 모든 entity 유효 text |
| batch_size | 256 |
| hard negatives | 없음 |

| Metric | Score |
|--------|-------|
| Recall@1 | 🔄 running |
| Recall@5 | 🔄 running |
| Recall@10 | 🔄 running |

checkpoint: `outputs/checkpoints/cb-inbatch-b256/`

---

### [5] codebook-contrastive — KMeans HN (✅ fixed, batch=256)

| 하이퍼파라미터 | 값 |
|---|---|
| [4]와 동일 + | |
| codebook | KMeans (flat), n_clusters=256 |
| ✅ codebook | 256/256 cluster 사용, imbalance=6.8x, max=847, mean=125.5 |
| HN mode | per_sample (pool = B+K = 272 per query) |
| n_hard_neg | 16 |
| batch_size | 256 |

| Metric | Score |
|--------|-------|
| Recall@1 | **68.70%** |
| Recall@5 | 92.31% |
| Recall@10 | 95.62% |

checkpoint: `outputs/checkpoints/cb-kmeans-per_sample-b256/`

---

### [6] codebook-contrastive — RQ HN (✅ fixed, batch=256)

| 하이퍼파라미터 | 값 |
|---|---|
| [4]와 동일 + | |
| codebook | Residual Quantization, depth=4, n_centroids=256 per level |
| 학습에 사용한 level | prefix_1 (coarsest) |
| HN mode | per_sample (pool = B+K = 272 per query) |
| n_hard_neg | 16 |
| batch_size | 256 |

| Metric | Score |
|--------|-------|
| Recall@1 | 68.53% |
| Recall@5 | 92.37% |
| Recall@10 | 95.49% |

checkpoint: `outputs/checkpoints/cb-rq-per_sample-b256/`

---

### [7] codebook-contrastive — OPQ HN (✅ fixed, batch=256)

| 하이퍼파라미터 | 값 |
|---|---|
| [4]와 동일 + | |
| codebook | Product Quantization (sklearn, no rotation), M=4 subspaces, K=256 |
| HN mode | per_sample (pool = B+K per query) |
| n_hard_neg | 16 |
| batch_size | 256 |

| Metric | Score |
|--------|-------|
| Recall@1 | ⏳ pending |
| Recall@5 | ⏳ pending |
| Recall@10 | ⏳ pending |

checkpoint: `outputs/checkpoints/cb-opq-per_sample-b256/`

---

### [8] KnowCoL baseline (batch=256) — 공정 비교용

| 하이퍼파라미터 | 값 |
|---|---|
| [1]과 동일 | |
| batch_size | 256 |

| Metric | Score |
|--------|-------|
| Recall@1 | 🔄 running |
| Recall@5 | 🔄 running |
| Recall@10 | 🔄 running |

---

## Batch Size별 비교 요약

| # | 모델 | codebook | HN mode | epoch | B32 R@1 | B32 R@5 | B32 R@10 | B128 R@1 | B128 R@5 | B128 R@10 | B256 R@1 | B256 R@5 | B256 R@10 | 비고 |
|---|------|----------|---------|-------|---------|---------|----------|----------|----------|-----------|----------|----------|-----------|------|
| 1 | KnowCoL | - | - | 10 | 42.64% | 72.44% | 80.72% | 40.19% | 70.38% | 78.49% | 🔄 running | 🔄 | 🔄 | baseline |
| 2 | CB in-batch | - | - | 10 | 58.33% | 84.13% | 89.66% | ⏳ | ⏳ | ⏳ | 62.79% | 86.13% | 91.24% | ✅ fixed |
| 3 | CB KMeans HN | KMeans flat | per_sample B+K | 10 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | **68.70%** | 92.31% | 95.62% | ✅ fixed |
| 4 | CB RQ HN | RQ depth=4 prefix_1 | per_sample B+K | 10 | **66.20%** | 91.08% | 94.59% | ⏳ | ⏳ | ⏳ | 68.53% | 92.37% | 95.49% | ✅ fixed |
| 5 | CB OPQ HN | faiss OPQ M=4 K=256 | per_sample B+K | 10 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | pending |
| 6 | CB KMeans HN | KMeans flat | per_sample B+K | **5** | ⏳ | ⏳ | ⏳ | **65.75%** | **90.08%** | **94.19%** | ⏳ | ⏳ | ⏳ | ✅ n_hn=110 |

> Batch size 축 비교 목적: 32 / 128 / 256에서 같은 모델 설정을 반복해 in-batch negative 수 차이와 hard-negative 효과를 분리한다.
> ⚠️ #1 KnowCoL(batch=32): in-batch negative 31개 vs CB(batch=256) 255개 → 직접 비교 시 주의
> ⚠️ shard01-only, 공식 OVEN benchmark 아님

---

## Codebook 품질 분포

### KMeans (flat, n_clusters=256) — ✅ fixed Phase 0

| 지표 | 값 |
|---|---|
| 사용 cluster | 256/256 (100%) |
| imbalance ratio | 6.8x |
| max bucket | 847 |
| mean bucket | 125.5 (ideal=125) |
| 50% entity 커버 | 상위 63개 cluster |
| 80% entity 커버 | 상위 148개 cluster |
| intra-cluster sim | ⏳ analyze 실행 후 |
| hard/easy neg gap | ⏳ analyze 실행 후 |

### RQ (depth=4, n_centroids=256) — prefix_1 level

| 지표 | 값 |
|---|---|
| 사용 cluster | ⏳ |
| imbalance ratio | ⏳ |
| intra-cluster sim | ⏳ |
| hard/easy neg gap | ⏳ |

### OPQ / sklearn PQ (M=4, K=256)

| 지표 | 값 |
|---|---|
| 사용 cluster | ⏳ pending |

> 분포 시각화: `outputs/codebook_analysis/{kmeans,rq,opq}/codebook_distribution.png`

---

## ⚠️ Phase 0 이슈 및 수정

**원인**: Wiki6M에 없는 entity 6,722개(20.9%)가 empty text → 동일 CLIP embedding
→ KMeans에서 하나의 거대 bucket으로 몰림 (max=6,722, imbalance=49x)

**수정**: `extract_entity_embeddings.py`
- `summary (+ title prefix) > title-only > f"entity {qid}"` 순서로 fallback
- 결과: summary+title=25,400 / title-only=16 / QID-only=6,706 (Wiki6M 미등록)
- fixed codebook: 256/256 cluster 사용, imbalance 49x → 6.8x

---

## 다음 단계

- [x] KnowCoL 10 epoch (batch=32)
- [x] CB in-batch ⚠️ bad Phase 0
- [x] CB KMeans HN ⚠️ bad codebook
- [x] Phase 0 수정 (title+summary fallback)
- [x] Phase 1 KMeans fixed
- [x] Phase 1 RQ (depth=4, n_centroids=256)
- [x] CB KMeans per_sample b256 ✅
- [x] CB RQ per_sample b256 ✅
- [ ] CB in-batch b256 fixed 🔄 running
- [ ] KnowCoL batch=256 🔄 running
- [ ] Phase 1 OPQ (sklearn backend)
- [ ] CB OPQ per_sample b256
- [ ] codebook 분포 분석 (KMeans, RQ, OPQ) — analyze_codebook.py
- [ ] all_wikipedia_images.tar 다운로드 → entity image 포함 재실험
