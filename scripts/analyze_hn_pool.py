"""
Analyze HN pool size distribution for training samples (shard01).
Compares KMeans prefix_1 vs RQ prefix_2 per training sample.

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive
    python scripts/analyze_hn_pool.py
"""
import json, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path('/data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive')
KNOWCOL_BASE = Path('/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL')
TRAIN_JSONL = KNOWCOL_BASE / 'dataset_01/oven_data/oven_entity_train.jsonl'
EMB_DIR     = BASE / 'outputs/entity_embs'
KM_DIR      = BASE / 'outputs/codebook'
RQ_DIR      = BASE / 'outputs/codebook_rq'
OUT_DIR     = BASE / 'outputs/codebook_analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
km_labels = np.load(KM_DIR / 'entity_cluster_labels.npy')
with open(KM_DIR / 'cluster_buckets.pkl','rb') as f:
    km_buckets = pickle.load(f)

rq_codes = np.load(RQ_DIR / 'entity_codes_rq.npy')
with open(RQ_DIR / 'prefix_buckets_rq.pkl','rb') as f:
    rq_prefix = pickle.load(f)
rq_p2 = rq_prefix['prefix_2']

with open(EMB_DIR / 'entity_id_list.json') as f:
    entity_ids = json.load(f)
id2idx = {qid: i for i, qid in enumerate(entity_ids)}
N = len(entity_ids)

# ── Collect positive entities per training sample ─────────────────────────────
pos_per_sample = []
with open(TRAIN_JSONL) as f:
    for line in f:
        d = json.loads(line)
        qid = d.get('entity_id','')
        if qid in id2idx:
            pos_per_sample.append(id2idx[qid])

unique_pos = set(pos_per_sample)

# ── KMeans HN pool (prefix_1) ─────────────────────────────────────────────────
km_pool = np.array([len(km_buckets[int(km_labels[e])]) - 1 for e in pos_per_sample])

# ── RQ prefix_2 HN pool (c1 AND c2 일치) ─────────────────────────────────────
rq_pool = np.array([
    max(len(rq_p2.get((int(rq_codes[e,0]), int(rq_codes[e,1])), [])) - 1, 0)
    for e in pos_per_sample
])

# ── Print stats ───────────────────────────────────────────────────────────────
print(f"총 training samples    : {len(pos_per_sample):,}")
print(f"unique positive entity : {len(unique_pos):,} / {N:,} ({len(unique_pos)/N*100:.1f}%)")

def print_pool_stats(name, pool):
    print(f"\n[{name}]")
    print(f"  mean={pool.mean():.1f}  median={np.median(pool):.1f}  "
          f"min={pool.min()}  max={pool.max()}")
    for k in [3, 8, 16, 32, 64, 128]:
        lt = (pool < k).mean() * 100
        ge = (pool >= k).mean() * 100
        print(f"  pool < {k:3d}: {lt:5.1f}%  |  pool >= {k:3d}: {ge:5.1f}%")

print_pool_stats('KMeans prefix_1', km_pool)
print_pool_stats('RQ prefix_2 (c1 AND c2 일치)', rq_pool)

# ── Save JSON ─────────────────────────────────────────────────────────────────
summary = {
    'n_train_samples': len(pos_per_sample),
    'n_unique_pos_entities': len(unique_pos),
    'n_total_entities': N,
    'kmeans_prefix1': {
        'mean': float(km_pool.mean()), 'median': float(np.median(km_pool)),
        'min': int(km_pool.min()), 'max': int(km_pool.max()),
        'pct_lt_16': float((km_pool<16).mean()*100),
        'pct_lt_128': float((km_pool<128).mean()*100),
    },
    'rq_prefix2': {
        'mean': float(rq_pool.mean()), 'median': float(np.median(rq_pool)),
        'min': int(rq_pool.min()), 'max': int(rq_pool.max()),
        'pct_zero': float((rq_pool==0).mean()*100),
        'pct_lt_3': float((rq_pool<3).mean()*100),
        'pct_gte_3': float((rq_pool>=3).mean()*100),
        'pct_lt_16': float((rq_pool<16).mean()*100),
    }
}
with open(OUT_DIR / 'hn_pool_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nStats → {OUT_DIR}/hn_pool_stats.json")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HN Pool Size per Training Sample (shard01)', fontsize=13)

axes[0].hist(km_pool, bins=np.arange(0, km_pool.max()+10, 5),
             color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(16,  color='red',    linestyle='--', lw=1.5, label='K=16')
axes[0].axvline(128, color='green',  linestyle='--', lw=1.5, label='K=128')
axes[0].axvline(km_pool.mean(), color='orange', linestyle='--', lw=1.5,
                label=f'mean={km_pool.mean():.0f}')
axes[0].set_title(f'KMeans prefix_1\nmean={km_pool.mean():.1f} | <16: {(km_pool<16).mean()*100:.1f}% | <128: {(km_pool<128).mean()*100:.1f}%')
axes[0].set_xlabel('HN pool size per sample')
axes[0].set_ylabel('# training samples')
axes[0].legend()

max_bin = min(int(rq_pool.max())+2, 80)
axes[1].hist(rq_pool, bins=np.arange(0, max_bin, 1),
             color='darkorange', edgecolor='white', alpha=0.85)
axes[1].axvline(3,  color='purple',   linestyle='--', lw=1.5, label='K=3')
axes[1].axvline(16, color='red',      linestyle='--', lw=1.5, label='K=16')
axes[1].axvline(rq_pool.mean(), color='steelblue', linestyle='--', lw=1.5,
                label=f'mean={rq_pool.mean():.1f}')
axes[1].set_title(f'RQ prefix_2 (c1+c2 일치)\nmean={rq_pool.mean():.1f} | pool=0: {(rq_pool==0).mean()*100:.1f}% | >=3: {(rq_pool>=3).mean()*100:.1f}%')
axes[1].set_xlabel('HN pool size per sample')
axes[1].set_ylabel('# training samples')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / 'hn_pool_comparison.png', dpi=130)
plt.close()
print(f"Plot  → {OUT_DIR}/hn_pool_comparison.png")
