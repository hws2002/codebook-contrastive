"""
Phase 1: Build offline codebook from entity base embeddings.
Uses KMeans (depth=1, n_clusters=256) for coarse experiment.
RQ-style depth can be increased later.

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive
    python scripts/build_codebook.py \
        --emb_dir outputs/entity_embs \
        --output_dir outputs/codebook \
        --n_clusters 256
"""
import argparse
import json
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir',    default='outputs/entity_embs')
    parser.add_argument('--output_dir', default='outputs/codebook')
    parser.add_argument('--n_clusters', type=int, default=256)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load entity base embeddings (text only for coarse)
    print("Loading entity embeddings...")
    text_embs = np.load(emb_dir / 'entity_text_embs.npy').astype(np.float32)
    with open(emb_dir / 'entity_id_list.json') as f:
        entity_ids = json.load(f)
    n_ent = len(entity_ids)
    print(f"  {n_ent} entities, dim={text_embs.shape[1]}")

    # Normalize before clustering (cosine similarity space)
    norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    text_embs_norm = text_embs / (norms + 1e-8)

    # KMeans clustering
    print(f"Running MiniBatchKMeans (n_clusters={args.n_clusters})...")
    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=4096,
        n_init=3,
        verbose=1,
    )
    cluster_labels = km.fit_predict(text_embs_norm)   # (N_ent,)
    print(f"Clustering done. Unique clusters: {len(set(cluster_labels))}")

    # Build bucket: cluster_id → list of entity indices
    cluster_buckets = defaultdict(list)
    for idx, cid in enumerate(cluster_labels):
        cluster_buckets[int(cid)].append(idx)

    bucket_sizes = [len(v) for v in cluster_buckets.values()]
    all_counts = np.zeros(args.n_clusters, dtype=int)
    for cid, idxs in cluster_buckets.items():
        all_counts[cid] = len(idxs)

    used   = int(np.sum(all_counts > 0))
    unused = args.n_clusters - used

    print(f"\n=== Codebook Index 사용 통계 ===")
    print(f"전체 index 수     : {args.n_clusters}")
    print(f"실제 사용 index   : {used}  ({used/args.n_clusters*100:.1f}%)")
    print(f"미사용 index      : {unused} ({unused/args.n_clusters*100:.1f}%)")
    print(f"\nBucket size stats:")
    print(f"  min    = {min(bucket_sizes)}")
    print(f"  max    = {max(bucket_sizes)}")
    print(f"  mean   = {np.mean(bucket_sizes):.1f}  (이상적: {n_ent//used:.0f})")
    print(f"  median = {np.median(bucket_sizes):.1f}")
    print(f"  std    = {np.std(bucket_sizes):.1f}")
    imbalance = max(bucket_sizes) / np.mean(bucket_sizes)
    print(f"  imbalance ratio (max/mean) = {imbalance:.1f}x  (이상적: 1.0x)")

    # Top 5 largest
    top5 = sorted(cluster_buckets.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"\nTop 5 largest clusters:")
    for cid, idxs in top5:
        print(f"  cluster {cid:3d}: {len(idxs):5d} entities ({len(idxs)/n_ent*100:.1f}%)")

    # Distribution: how concentrated
    sizes_sorted = sorted(bucket_sizes, reverse=True)
    cumsum = np.cumsum(sizes_sorted)
    for pct in [50, 80]:
        n_needed = int(np.searchsorted(cumsum, n_ent * pct / 100)) + 1
        print(f"  상위 {n_needed}개 cluster가 전체 {pct}% entity 커버")

    # Save
    np.save(out_dir / 'entity_cluster_labels.npy', cluster_labels)
    with open(out_dir / 'cluster_buckets.pkl', 'wb') as f:
        pickle.dump(dict(cluster_buckets), f)
    with open(out_dir / 'codebook_config.json', 'w') as f:
        json.dump({
            'n_clusters': args.n_clusters,
            'n_entities': n_ent,
            'seed': args.seed,
            'input': 'entity_text_embs (normalized)',
        }, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  entity_cluster_labels.npy  shape: {cluster_labels.shape}")
    print(f"  cluster_buckets.pkl        {args.n_clusters} clusters")
    print(f"  codebook_config.json")


if __name__ == '__main__':
    main()
