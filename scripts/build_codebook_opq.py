"""
OPQ (Optimized Product Quantization) codebook — faiss 기반.
M개 subspace, 각 subspace K centroids.
faiss가 없으면 일반 PQ (rotation 없음)로 fallback.

code = (c1, c2, ..., cM)  각 ci ∈ {0,...,K-1}
→ 유사한 entity일수록 subcode 많이 공유
→ 공유 subcode 수로 granularity 조절 가능

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive
    python scripts/build_codebook_opq.py \
        --emb_dir outputs/entity_embs \
        --output_dir outputs/codebook_opq \
        --M 4 \
        --K 256
"""
import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict


def build_opq(embs: np.ndarray, M: int, K: int, n_iter: int = 20, backend: str = 'auto'):
    """
    OPQ via faiss (with rotation) or sklearn PQ (no rotation, faster on CPU).
    backend: 'auto' tries faiss-gpu first then sklearn, 'sklearn' forces sklearn, 'faiss' forces faiss
    Returns:
        codes: (N, M) int32
        quantizer: faiss index object (or centroids list for sklearn)
        R: rotation matrix or None
    """
    use_faiss = False
    if backend in ('auto', 'faiss'):
        try:
            import faiss
            # Only use faiss if GPU is available — CPU faiss OPQ is very slow
            if hasattr(faiss, 'StandardGpuResources'):
                use_faiss = True
            elif backend == 'faiss':
                use_faiss = True   # forced
            else:
                print("faiss-cpu detected (no GPU) — falling back to sklearn PQ (much faster)")
        except ImportError:
            pass

    if use_faiss:
        import faiss
        d = embs.shape[1]
        assert d % M == 0, f"dim {d} must be divisible by M={M}"
        print(f"Using faiss OPQ (M={M}, K={K}, d={d}, niter={n_iter})")
        opq = faiss.OPQMatrix(d, M)
        opq.niter = n_iter
        opq.verbose = True
        faiss.normalize_L2(embs)
        opq.train(embs)
        embs_rot = opq.apply_py(embs)
        pq = faiss.ProductQuantizer(d, M, int(np.log2(K)))
        pq.train(embs_rot)
        codes = pq.compute_codes(embs_rot)
        return codes, pq, opq
    else:
        return build_pq_sklearn(embs, M, K)


def build_pq_sklearn(embs: np.ndarray, M: int, K: int):
    """Simple PQ without rotation using sklearn KMeans per subspace."""
    from sklearn.cluster import MiniBatchKMeans

    N, d = embs.shape
    assert d % M == 0
    sub_d = d // M

    print(f"sklearn PQ fallback (M={M}, K={K}, sub_d={sub_d})")
    codes = np.zeros((N, M), dtype=np.int32)
    centroids_list = []

    for m in range(M):
        sub_embs = embs[:, m*sub_d:(m+1)*sub_d]
        km = MiniBatchKMeans(n_clusters=K, random_state=42+m, batch_size=4096, n_init=3)
        km.fit(sub_embs)
        codes[:, m] = km.labels_
        centroids_list.append(km.cluster_centers_.astype(np.float32))
        print(f"  subspace {m+1}/{M} done")

    return codes, centroids_list, None


def build_prefix_buckets(codes: np.ndarray, min_shared: int = 1):
    """
    Build buckets by number of shared subcodes.
    bucket[k] = {code_tuple_prefix_k: [entity_idxs]}
    """
    M = codes.shape[1]
    prefix_buckets = {}
    for k in range(1, M + 1):
        buckets = defaultdict(list)
        for i, code in enumerate(codes):
            key = tuple(code[:k].tolist())
            buckets[key].append(i)
        prefix_buckets[f'prefix_{k}'] = dict(buckets)
    return prefix_buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir',    default='outputs/entity_embs')
    parser.add_argument('--output_dir', default='outputs/codebook_opq')
    parser.add_argument('--M',       type=int, default=4,      help='number of subspaces')
    parser.add_argument('--K',       type=int, default=256,    help='centroids per subspace (must be power of 2 for faiss)')
    parser.add_argument('--iter',    type=int, default=20,     help='OPQ rotation training iterations (faiss only)')
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'faiss', 'sklearn'],
                        help='auto: faiss-gpu if available else sklearn | sklearn: always fast CPU PQ | faiss: force faiss (slow on CPU)')
    args = parser.parse_args()

    assert args.K & (args.K - 1) == 0, f"K={args.K} must be power of 2 for faiss OPQ (e.g. 64, 128, 256)"

    emb_dir = Path(args.emb_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading entity embeddings...")
    text_embs = np.load(emb_dir / 'entity_text_embs.npy').astype(np.float32)
    with open(emb_dir / 'entity_id_list.json') as f:
        entity_ids = json.load(f)
    n_ent = len(entity_ids)
    print(f"  {n_ent} entities, dim={text_embs.shape[1]}")

    # Normalize
    norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    embs_norm = text_embs / (norms + 1e-8)

    # Build OPQ
    codes, quantizer, rotation = build_opq(embs_norm, args.M, args.K, args.iter, args.backend)
    print(f"Codes shape: {codes.shape}")

    # Build prefix buckets (same structure as RQ)
    print("Building prefix buckets...")
    prefix_buckets = build_prefix_buckets(codes)

    # Stats
    print(f"\n=== OPQ Codebook Stats (M={args.M}, K={args.K}) ===")
    for k in range(1, args.M + 1):
        bkts = prefix_buckets[f'prefix_{k}']
        sizes = [len(v) for v in bkts.values()]
        print(f"prefix-{k}: {len(bkts)} buckets | "
              f"min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f} "
              f"imbalance={max(sizes)/np.mean(sizes):.1f}x")

    # Save
    np.save(out_dir / 'entity_codes_opq.npy', codes)
    with open(out_dir / 'prefix_buckets_opq.pkl', 'wb') as f:
        pickle.dump(prefix_buckets, f)
    # faiss objects (SwigPyObject) are not picklable — skip quantizer, only codes needed for training
    is_faiss = rotation is not None and not isinstance(quantizer, list)
    with open(out_dir / 'codebook_config.json', 'w') as f:
        json.dump({'type': 'OPQ', 'M': args.M, 'K': args.K,
                   'n_entities': n_ent,
                   'backend': 'faiss' if is_faiss else 'sklearn_fallback'}, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  entity_codes_opq.npy      shape: {codes.shape}")
    print(f"  prefix_buckets_opq.pkl    M={args.M} levels")


if __name__ == '__main__':
    main()
