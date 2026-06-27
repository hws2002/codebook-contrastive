"""
Analyze codebook quality and distribution.
Supports kmeans (flat) and rq (all prefix levels 1~4).

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive

    # KMeans
    python scripts/analyze_codebook.py \
        --emb_dir outputs/entity_embs \
        --codebook_dir outputs/codebook \
        --codebook_type kmeans \
        --output_dir outputs/codebook_analysis/kmeans

    # RQ — all 4 prefix levels
    python scripts/analyze_codebook.py \
        --emb_dir outputs/entity_embs \
        --codebook_dir outputs/codebook_rq \
        --codebook_type rq \
        --output_dir outputs/codebook_analysis/rq
"""
import argparse
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def bucket_stats(buckets: dict, n_ent: int):
    sizes = [len(v) for v in buckets.values()]
    imbalance = max(sizes) / np.mean(sizes)
    sizes_sorted = sorted(sizes, reverse=True)
    cumsum = np.cumsum(sizes_sorted)
    cov50 = int(np.searchsorted(cumsum, n_ent * 0.50)) + 1
    cov80 = int(np.searchsorted(cumsum, n_ent * 0.80)) + 1
    return {
        'n_used': len(buckets),
        'min': int(min(sizes)),
        'max': int(max(sizes)),
        'mean': float(np.mean(sizes)),
        'std': float(np.std(sizes)),
        'imbalance': float(imbalance),
        'cov50': cov50,
        'cov80': cov80,
        'sizes_sorted': sizes_sorted,
    }


def hn_quality(buckets: dict, labels: np.ndarray, text_embs: np.ndarray, n_samples: int = 100):
    """Measure hard neg vs easy neg similarity gap."""
    rng = np.random.default_rng(42)
    n_ent = len(labels)
    sample_ents = rng.choice(n_ent, size=n_samples, replace=False)
    within_sim, cross_sim = [], []
    for eidx in sample_ents:
        cid  = int(labels[eidx])
        pool = [i for i in buckets.get(cid, []) if i != eidx]
        if len(pool) == 0:
            continue
        neg_idxs = rng.choice(pool, size=min(5, len(pool)), replace=False)
        e_norm = text_embs[eidx] / (np.linalg.norm(text_embs[eidx]) + 1e-8)
        for nidx in neg_idxs:
            n_norm = text_embs[nidx] / (np.linalg.norm(text_embs[nidx]) + 1e-8)
            within_sim.append(float(e_norm @ n_norm))
        rand_idxs = rng.choice(n_ent, size=5, replace=False)
        for ridx in rand_idxs:
            r_norm = text_embs[ridx] / (np.linalg.norm(text_embs[ridx]) + 1e-8)
            cross_sim.append(float(e_norm @ r_norm))
    return float(np.mean(within_sim)), float(np.mean(cross_sim))


def plot_level(ax_row, stats: dict, title: str):
    sizes_sorted = stats['sizes_sorted']
    n_ent = sum(sizes_sorted)
    ax_row[0].hist(sizes_sorted, bins=30, color='steelblue', edgecolor='white')
    ax_row[0].axvline(stats['mean'], color='red', linestyle='--', label=f"mean={stats['mean']:.0f}")
    ax_row[0].set_title(f'{title}\nBucket Size Dist')
    ax_row[0].set_xlabel('Entities per bucket')
    ax_row[0].legend(fontsize=8)

    ax_row[1].bar(range(len(sizes_sorted)), sizes_sorted, color='steelblue', width=1.0)
    ax_row[1].set_title(f'{title}\nSorted Sizes (imbalance={stats["imbalance"]:.1f}x)')
    ax_row[1].set_xlabel('Bucket rank')

    cum_pct = np.cumsum(sizes_sorted) / n_ent * 100
    ax_row[2].plot(range(1, len(cum_pct)+1), cum_pct, color='steelblue')
    ax_row[2].axhline(80, color='red', linestyle='--', label='80%')
    ax_row[2].axhline(50, color='orange', linestyle='--', label='50%')
    ax_row[2].set_title(f'{title}\nCumulative Coverage')
    ax_row[2].set_xlabel('# buckets (ranked)')
    ax_row[2].set_ylabel('% entities')
    ax_row[2].legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir',       default='outputs/entity_embs')
    parser.add_argument('--codebook_dir',  default='outputs/codebook')
    parser.add_argument('--codebook_type', default='kmeans', choices=['kmeans', 'rq'])
    parser.add_argument('--output_dir',    default='outputs/codebook_analysis/kmeans')
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    cb_dir  = Path(args.codebook_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(emb_dir / 'entity_id_list.json') as f:
        entity_ids = json.load(f)
    text_embs = np.load(emb_dir / 'entity_text_embs.npy')
    n_ent = len(entity_ids)

    print("=" * 60)
    print(f"CODEBOOK ANALYSIS  [{args.codebook_type.upper()}]")
    print("=" * 60)

    # ── KMeans: single level ────────────────────────────────────────────────
    if args.codebook_type == 'kmeans':
        labels = np.load(cb_dir / 'entity_cluster_labels.npy')
        with open(cb_dir / 'cluster_buckets.pkl', 'rb') as f:
            buckets = pickle.load(f)
        with open(cb_dir / 'codebook_config.json') as f:
            cfg = json.load(f)

        s = bucket_stats(buckets, n_ent)
        hn_sim, easy_sim = hn_quality(buckets, labels, text_embs)

        print(f"n_clusters: {cfg['n_clusters']}  used: {s['n_used']}")
        print(f"min={s['min']}  max={s['max']}  mean={s['mean']:.1f}  std={s['std']:.1f}")
        print(f"imbalance={s['imbalance']:.2f}x")
        print(f"50% coverage: top {s['cov50']} clusters")
        print(f"80% coverage: top {s['cov80']} clusters")
        print(f"HN sim={hn_sim:.4f}  easy sim={easy_sim:.4f}  gap={hn_sim-easy_sim:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('KMeans Codebook', fontsize=13)
        plot_level([axes[0], axes[1], axes[2]], s, 'KMeans (flat, 256 clusters)')
        plt.tight_layout()
        plt.savefig(out_dir / 'codebook_distribution.png', dpi=120)
        plt.close()

        summary = {
            'codebook_type': 'kmeans', 'n_entities': n_ent,
            **{k: s[k] for k in ['n_used','min','max','mean','std','imbalance','cov50','cov80']},
            'hard_neg_sim': hn_sim, 'easy_neg_sim': easy_sim, 'gap': hn_sim - easy_sim,
        }
        with open(out_dir / 'codebook_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nPlot  → {out_dir}/codebook_distribution.png")
        print(f"Stats → {out_dir}/codebook_stats.json")

    # ── RQ: all 4 prefix levels ─────────────────────────────────────────────
    elif args.codebook_type == 'rq':
        codes = np.load(cb_dir / 'entity_codes_rq.npy')   # (N, depth)
        with open(cb_dir / 'prefix_buckets_rq.pkl', 'rb') as f:
            prefix_bkts = pickle.load(f)
        with open(cb_dir / 'codebook_config.json') as f:
            cfg = json.load(f)
        depth = cfg['depth']

        all_stats = {}
        fig, axes = plt.subplots(depth, 3, figsize=(15, 4 * depth))
        fig.suptitle(f'RQ Codebook (depth={depth}, n_centroids={cfg["n_centroids"]})', fontsize=13)

        print(f"depth={depth}, n_centroids={cfg['n_centroids']}\n")
        print(f"{'Level':<10} {'Buckets':>8} {'max':>7} {'mean':>7} {'imb':>7} {'cov50':>7} {'cov80':>7} {'HN gap':>8}")
        print("-" * 70)

        for k in range(1, depth + 1):
            key = f'prefix_{k}'
            raw_bkts = prefix_bkts[key]
            # keys are tuples (c1,...,ck) → use tuple directly for lookup
            # rebuild int-keyed for labels (only valid for k=1, for display use tuple keys)
            buckets_tuple = raw_bkts   # {(c1,...,ck): [idxs]}

            # For hn_quality we need labels aligned with buckets
            # use codes[:, :k] row as key
            labels_k = np.array([tuple(codes[i, :k].tolist()) for i in range(n_ent)])
            # rebuild int-indexed buckets for hn_quality by index mapping
            idx_to_bucket = {}
            for tkey, idxs in raw_bkts.items():
                for idx in idxs:
                    idx_to_bucket[idx] = tkey
            # create int labels and int-keyed buckets for hn_quality helper
            unique_keys = {k: i for i, k in enumerate(raw_bkts.keys())}
            int_labels = np.array([unique_keys[idx_to_bucket[i]] for i in range(n_ent)], dtype=np.int64)
            int_buckets = {unique_keys[k]: v for k, v in raw_bkts.items()}

            s = bucket_stats(int_buckets, n_ent)
            hn_sim, easy_sim = hn_quality(int_buckets, int_labels, text_embs)
            all_stats[key] = {**{x: s[x] for x in ['n_used','min','max','mean','std','imbalance','cov50','cov80']},
                              'hard_neg_sim': hn_sim, 'easy_neg_sim': easy_sim, 'gap': hn_sim - easy_sim}

            print(f"prefix-{k:<5} {s['n_used']:>8} {s['max']:>7} {s['mean']:>7.1f} "
                  f"{s['imbalance']:>6.1f}x {s['cov50']:>7} {s['cov80']:>7} {hn_sim-easy_sim:>8.4f}")

            plot_level(axes[k-1], s, f'prefix-{k} ({s["n_used"]} buckets, imb={s["imbalance"]:.1f}x)')

        print()
        plt.tight_layout()
        plt.savefig(out_dir / 'codebook_distribution.png', dpi=100)
        plt.close()

        summary = {'codebook_type': 'rq', 'n_entities': n_ent,
                   'depth': depth, 'n_centroids': cfg['n_centroids'], 'levels': all_stats}
        with open(out_dir / 'codebook_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Plot  → {out_dir}/codebook_distribution.png")
        print(f"Stats → {out_dir}/codebook_stats.json")


if __name__ == '__main__':
    main()
