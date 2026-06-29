"""
RQ (Residual Quantization) codebook — depth=4, each level n_centroids centroids.
idea2_codebook.md 원래 설계와 일치.

code = (c1, c2, c3, c4)
prefix-1: (c1,)
prefix-2: (c1, c2)
prefix-3: (c1, c2, c3)
full:      (c1, c2, c3, c4)

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive
    python scripts/build_codebook_rq.py \
        --emb_dir outputs/entity_embs \
        --output_dir outputs/codebook_rq \
        --depth 4 \
        --n_centroids 256
"""
import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans


def rq_encode(embs: np.ndarray, centroids_list: list) -> np.ndarray:
    """
    RQ encoding: sequentially quantize residuals.
    Returns codes shape (N, depth).
    """
    N = embs.shape[0]
    depth = len(centroids_list)
    codes = np.zeros((N, depth), dtype=np.int32)
    residual = embs.copy()

    for d, centroids in enumerate(centroids_list):
        # assign to nearest centroid
        dists = np.sum((residual[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # (N, K)
        assign = np.argmin(dists, axis=1)   # (N,)
        codes[:, d] = assign
        # subtract centroid → residual for next level
        residual = residual - centroids[assign]

    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir',      default='outputs/entity_embs')
    parser.add_argument('--output_dir',   default='outputs/codebook_rq')
    parser.add_argument('--depth',        type=int, default=4)
    parser.add_argument('--n_centroids',  type=int, default=256)
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading entity embeddings...")
    text_embs = np.load(emb_dir / 'entity_text_embs.npy').astype(np.float32)
    with open(emb_dir / 'entity_id_list.json') as f:
        entity_ids = json.load(f)
    n_ent = len(entity_ids)

    # normalize
    norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    embs_norm = text_embs / (norms + 1e-8)

    # ── Train RQ codebooks level by level ──────────────────────────────────
    centroids_list = []
    residual = embs_norm.copy()

    for d in range(args.depth):
        print(f"Training level {d+1}/{args.depth} KMeans (n_centroids={args.n_centroids})...")
        km = MiniBatchKMeans(
            n_clusters=args.n_centroids,
            random_state=args.seed + d,
            batch_size=4096,
            n_init=3,
        )
        km.fit(residual)
        centroids = km.cluster_centers_.astype(np.float32)  # (K, D)
        centroids_list.append(centroids)

        assign = km.labels_  # (N,)
        residual = residual - centroids[assign]

    # ── Encode all entities ─────────────────────────────────────────────────
    print("Encoding all entities with RQ...")
    codes = rq_encode(embs_norm, centroids_list)   # (N, depth)

    # ── Build prefix buckets ─────────────────────────────────────────────────
    prefix_buckets = {}   # {(c1,...,ck): [entity_idx,...]} for k=1,2,3,4
    for depth_k in range(1, args.depth + 1):
        buckets = defaultdict(list)
        for i, code in enumerate(codes):
            key = tuple(code[:depth_k].tolist())
            buckets[key].append(i)
        prefix_buckets[f'prefix_{depth_k}'] = dict(buckets)

    # ── Stats ───────────────────────────────────────────────────────────────
    print(f"\n=== RQ Codebook Stats (depth={args.depth}, n_centroids={args.n_centroids}) ===")
    for depth_k in range(1, args.depth + 1):
        bkts = prefix_buckets[f'prefix_{depth_k}']
        sizes = [len(v) for v in bkts.values()]
        print(f"prefix-{depth_k}: {len(bkts)} buckets | "
              f"min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f} "
              f"imbalance={max(sizes)/np.mean(sizes):.1f}x")

    # ── Save ────────────────────────────────────────────────────────────────
    np.save(out_dir / 'entity_codes_rq.npy', codes)
    with open(out_dir / 'centroids_rq.pkl', 'wb') as f:
        pickle.dump(centroids_list, f)
    with open(out_dir / 'prefix_buckets_rq.pkl', 'wb') as f:
        pickle.dump(prefix_buckets, f)
    with open(out_dir / 'codebook_config.json', 'w') as f:
        json.dump({'type': 'RQ', 'depth': args.depth,
                   'n_centroids': args.n_centroids, 'n_entities': n_ent}, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  entity_codes_rq.npy     shape: {codes.shape}")
    print(f"  prefix_buckets_rq.pkl   depth 1~{args.depth}")


if __name__ == '__main__':
    main()
