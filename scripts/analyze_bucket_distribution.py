"""
각 codebook의 level별 코드 분포 분석.
각 entity는 (c1, c2, c3, c4) 4개의 코드를 가짐, 각 ci ∈ {0,...,255}.
level별로 32K entity가 256개 코드에 어떻게 분포되는지 시각화.

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive
    python scripts/analyze_bucket_distribution.py
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path('outputs/codebook_analysis/bucket_distribution')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODEBOOKS = {
    'text':  'outputs/codebook_rq_text/entity_codes_rq.npy',
    'image': 'outputs/codebook_rq_img/entity_codes_rq.npy',
}

for name, path in CODEBOOKS.items():
    if not Path(path).exists():
        print(f"{name}: not found")
        continue

    codes = np.load(path)   # (N, 4), each value ∈ {0,...,255}
    N, depth = codes.shape
    print(f"\n=== {name} codebook: {N} entities, depth={depth} ===")

    fig, axes = plt.subplots(depth, 1, figsize=(16, 4 * depth))
    stats = {}

    for level in range(depth):
        col = codes[:, level]                    # (N,) 각 entity의 level별 코드
        counts = np.bincount(col, minlength=256) # 각 코드(0~255)에 몇 개 entity

        stats[f'level_{level+1}'] = {
            'min': int(counts.min()),
            'max': int(counts.max()),
            'mean': float(counts.mean()),
            'std': float(counts.std()),
            'imbalance': float(counts.max() / counts.mean()),
            'empty_codes': int((counts == 0).sum()),
        }
        print(f"  level {level+1}: min={counts.min()} max={counts.max()} "
              f"mean={counts.mean():.1f} imbalance={counts.max()/counts.mean():.1f}x "
              f"empty={int((counts==0).sum())}")

        ax = axes[level]
        ax.bar(np.arange(256), counts, width=0.8)
        ax.axhline(counts.mean(), color='r', linestyle='--', label=f'mean={counts.mean():.1f}')
        ax.set_title(f'{name} codebook - level {level+1} (code index 0~255)')
        ax.set_xlabel('Code index')
        ax.set_ylabel('Entity count')
        ax.set_xlim(-1, 256)
        ax.legend()

    plt.tight_layout()
    out_png = OUT_DIR / f'{name}_code_distribution.png'
    plt.savefig(out_png, dpi=120)
    plt.close()

    with open(OUT_DIR / f'{name}_code_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Saved → {out_png}")
