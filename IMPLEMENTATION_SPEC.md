# codebook-contrastive: 구현 명세서 (v2)

## 목표

OVEN entity linking에서 KnowCoL 대비 좋은 성능을 내고,
이를 기반으로 InfoSeek / Encyclopedic-VQA로 확장.
비교 기준: KnowCoL (동일한 dataset_01 기준).
목표 학회: AAAI.

---

## 핵심 설계 원칙 (idea2_codebook.md 기반)

```
1. base encoder (CLIP) 는 완전 frozen
2. entity-side: EntityHead MLP만 학습 (g_ent)
3. query-side:  QueryHead MLP만 학습  (h_ent)
4. codebook은 offline index — loss 없음, routing + hard negative 전용
5. head별 별도 학습 (L_ent, L_fact, L_text 각각)
6. MVP = entity retrieval (OVEN) 만 먼저
```

---

## 전체 아키텍처

### Query side
```
I_q, q
  -> CLIP ImageEncoder (frozen) -> z_q_img  (D=768)
  -> CLIP TextEncoder  (frozen) -> z_q_text (D=768)
  -> h_ent = QueryHead MLP (TRAINABLE)
     input:  concat([z_q_img, z_q_text])  (2D=1536)
     output: z_q_ent  (D=768), normalized
```

### Entity side
```
entity item (title/aliases/summary/leading image)
  -> CLIP TextEncoder  (frozen) -> z_ent_text (D=768)  [Phase 0에서 사전 추출]
  -> CLIP ImageEncoder (frozen) -> z_ent_img  (D=768)  [Phase 0에서 사전 추출]
  -> g_ent = EntityHead MLP (TRAINABLE)
     input:  concat([z_ent_text, z_ent_img])  (2D=1536)
     output: z_ent  (D=768), normalized
```

KnowCoL 차이점:
- KnowCoL: KGE lookup table (32K×768 = 24.7M 파라미터, entity마다 별도 embedding)
- 우리:    EntityHead MLP (1536→768 = 1.2M 파라미터, entity-agnostic → unseen entity 일반화)

### Trainable params
| 모듈 | 구조 | Params |
|------|------|--------|
| QueryHead (h_ent) | Linear(1536→768) + bias | ~1.2M |
| EntityHead (g_ent) | Linear(1536→768) + bias | ~1.2M |
| **합계** | | **~2.4M** |

(KnowCoL: 26.2M)

---

## Phase 0: Base Embedding 추출 (`scripts/extract_entity_embeddings.py`)

**entity별로 text embedding과 image embedding을 분리 저장.**

```python
for each entity in KG (32,122 entities):
    entity_text = f"{title}. {', '.join(aliases)}. {summary}"
    z_ent_text = CLIP_TextEncoder(entity_text)   # (768,)

    if leading_image exists:
        z_ent_img = CLIP_ImageEncoder(leading_image)  # (768,)
    else:
        z_ent_img = z_ent_text.clone()  # fallback

    # 별도 저장 (EntityHead 학습 시 concat해서 씀)
    save(qid, z_ent_text, z_ent_img)
```

출력:
- `entity_text_embs.npy`   shape: (N_ent, 768)
- `entity_img_embs.npy`    shape: (N_ent, 768)
- `entity_id_list.json`    QID 순서 보존

---

## Phase 1: Offline Codebook 생성 (`scripts/build_codebook.py`)

### Codebook input
```python
# Phase 0의 두 embedding을 concat해서 z_ent_base 구성
z_ent_base = concat([z_ent_text, z_ent_img])  # (N_ent, 1536)
# 또는 간단히 평균:
z_ent_base = (z_ent_text + z_ent_img) / 2    # (N_ent, 768)
```

codebook은 z_ent_base 위에서 만든다 (EntityHead 학습 전, frozen base embedding 기준).

### RQ-style codebook (depth=4, codebook_size=256)
```
level 1: KMeans(z_ent_base) → c1 (256 centroids)
level 2: KMeans(z_ent_base - centroid[c1]) → c2
level 3: KMeans(residual2) → c3
level 4: KMeans(residual3) → c4

entity i의 code: (c1_i, c2_i, c3_i, c4_i)
```

prefix bucket 인덱스:
```python
prefix1_bucket[(c1,)]         = [entity_ids...]
prefix2_bucket[(c1, c2)]      = [entity_ids...]
prefix3_bucket[(c1, c2, c3)]  = [entity_ids...]
full_bucket[(c1, c2, c3, c4)] = [entity_ids...]
```

출력:
- `entity_codes.npy`     shape: (N_ent, 4)
- `prefix_buckets.pkl`   dict: tuple → list of entity indices
- `codebook_weights.pkl` RQ centroids (for query coding at inference)

---

## Phase 2: Head Contrastive 학습 (`scripts/train_entity_head.py`)

### 학습 대상
- QueryHead (h_ent): trainable
- EntityHead (g_ent): trainable
- CLIP: frozen

### Forward pass
```python
# Query encoding
z_q_img  = clip.encode_image(query_image)          # frozen
z_q_text = clip.encode_text(query_text)            # frozen
z_q_ent  = normalize(query_head(cat([z_q_img, z_q_text])))  # trainable

# Entity encoding (학습 중 EntityHead 통과)
z_ent_text = entity_text_embs[entity_idx]   # pre-extracted, frozen on disk
z_ent_img  = entity_img_embs[entity_idx]    # pre-extracted, frozen on disk
z_ent      = normalize(entity_head(cat([z_ent_text, z_ent_img])))  # trainable
```

### Hard Negative 구성 (학습 전 offline 생성)
```python
for each training sample (query_i, positive_entity_i):
    code = entity_codes[positive_entity_i]   # (c1, c2, c3, c4)

    # prefix-1 bucket (가장 broad, visually/semantically similar category)
    hard_neg_p1 = sample(prefix1_bucket[(code[0],)], k=N1, exclude=positive)

    # prefix-2 bucket (더 가까운 hard negative)
    hard_neg_p2 = sample(prefix2_bucket[(code[0], code[1])], k=N2, exclude=positive)

    # prefix-3 bucket (가장 가까운 hard negative)
    hard_neg_p3 = sample(prefix3_bucket[(code[0], code[1], code[2])], k=N3, exclude=positive)

권장: N1=2, N2=4, N3=4 (총 10 hard negatives per sample)
```

### Loss (InfoNCE)
```python
# positives:        z_q_ent_i & z_ent_positive_i
# in-batch neg:     다른 샘플의 z_ent (B-1개)
# hard neg:         codebook prefix bucket entities (10개)

all_neg_ents = cat([in_batch_ents, hard_neg_ents])   # (B-1+10, D)
logits = z_q_ent @ all_ents.T / temperature           # (B, B+10)
L_ent = CrossEntropy(logits, labels=diagonal)
```

### 데이터
- train: `dataset_01/oven_data/oven_entity_train.jsonl` (114,124 entries)
- val:   `dataset_01/oven_data/oven_entity_val.jsonl` (6,006 entries)
- 이미지: `dataset_01/oven_images/01/`

### 학습 설정
```
backbone: CLIP ViT-L-14 (frozen)
optimizer: AdamW, lr=1e-4
batch_size: 32
temperature: 0.07
hard_neg per sample: 10 (N1=2, N2=4, N3=4)
epochs: 10
GPU: A800 80GB x1
```

---

## Inference Pipeline

### Step 1. Entity retrieval
```python
z_q_ent = normalize(query_head(cat([CLIP(img), CLIP(text)])))

# query에 codebook code 할당 (routing용)
query_code = RQ_assign(z_q_ent, codebook_weights)   # (c1, c2, c3, c4)

# dense retrieval + codebook routing 합집합
candidates_faiss = FAISS_search(z_q_ent, k=100)
candidates_code  = prefix_bucket[(query_code[0], query_code[1])]  # prefix-2 bucket
candidates = union(candidates_faiss, candidates_code)

# rerank by dense score
z_cands = entity_head(cat([entity_text_embs[candidates], entity_img_embs[candidates]]))
scores = z_q_ent @ z_cands.T
top_K = argsort(scores, descending=True)[:K]
```

---

## 평가 (`evaluate/eval_entity_linking.py`)

KnowCoL과 동일한 포맷으로 출력 → 같은 eval 코드 재사용.

```json
{"data_id": "...", "entity_id": "Q123", "pred_entity_ids": ["Q456", "Q789", ...top-10]}
```

```python
from knowcol.evaluations.evaluation import recall_at_k
result = recall_at_k(predictions, ks=[1, 5, 10])
# → {"recall@1": ..., "recall@5": ..., "recall@10": ...}
```

entity store: 32,122 entities (KnowCoL과 동일 KG 기준으로 비교)

---

## KnowCoL vs codebook-contrastive 비교 요약

| 항목 | KnowCoL | codebook-contrastive |
|------|---------|---------------------|
| Backbone | CLIP ViT-L-14 (frozen) | CLIP ViT-L-14 (frozen) |
| Query head | LP1+LP2 Linear (trainable) | QueryHead MLP (trainable) |
| Entity repr. | KGE lookup (32K×768, 24.7M) | EntityHead MLP (1536→768, 1.2M) |
| Hard negative | KG triplet corruption | Codebook same-prefix |
| Loss | alignment + KE + proxy | InfoNCE only |
| Unseen entity | KGE 없으면 embedding 없음 | EntityHead는 metadata 기반 → 일반화 가능 |
| Trainable params | 26.2M | **2.4M** |

---

## 디렉토리 구조

```
codebook-contrastive/
├── IMPLEMENTATION_SPEC.md
├── scripts/
│   ├── extract_entity_embeddings.py   # Phase 0
│   ├── build_codebook.py              # Phase 1
│   ├── build_hard_negatives.py        # Phase 2 사전 준비
│   └── train_entity_head.py           # Phase 2 학습
├── src/
│   ├── models/
│   │   ├── heads.py          # QueryHead, EntityHead MLP
│   │   └── entity_model.py   # PL LightningModule
│   ├── codebook/
│   │   ├── rq_codebook.py    # RQ 생성 + code 할당
│   │   └── hard_negative.py  # prefix bucket 샘플링
│   ├── datasets/
│   │   └── oven_dataset.py   # dataset_01 호환 Dataset
│   └── utils/
│       └── embeddings.py     # entity embedding 로드/캐시
└── evaluate/
    └── eval_entity_linking.py   # recall_at_k, KnowCoL 동일 포맷
```

---

## 데이터 경로

```
/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL/dataset_01/
├── oven_data/oven_entity_train.jsonl   (114,124)
├── oven_data/oven_entity_val.jsonl     (6,006)
└── oven_images/01/                     (120,130 JPEG)

/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL/dataset/
├── knowledge_base/Wiki6M_ver_1_1.jsonl   (entity metadata: title/aliases/summary/image)
└── wikidata_subgraph_v1/entity.txt        (32,122 QIDs)
```

---

## MVP 구현 순서

```
[D1] Phase 0: extract_entity_embeddings.py
     → entity_text_embs.npy, entity_img_embs.npy, entity_id_list.json

[D2] Phase 1: build_codebook.py (RQ depth=2 먼저, depth=4 나중)
     → entity_codes.npy, prefix_buckets.pkl, codebook_weights.pkl

[D3] build_hard_negatives.py
     → 각 train sample별 hard negative entity idx 사전 생성 → hdf5 or jsonl

[D4] src/models/heads.py (QueryHead, EntityHead)
     src/models/entity_model.py (PL LightningModule + InfoNCE)
     src/datasets/oven_dataset.py

[D5] train_entity_head.py 실행 → 학습

[D6] evaluate/eval_entity_linking.py → Recall@1/5/10 (KnowCoL과 동일 포맷)
```

## 하지 않는 것 (MVP)

```
No codebook loss (L_code, L_usage, L_div)
No codebook refresh after training
No FactHead / TextEvidenceHead (OVEN entity linking만 먼저)
No LoRA / CLIP fine-tuning
No joint L_total
No InfoSeek pseudo text evidence
```
