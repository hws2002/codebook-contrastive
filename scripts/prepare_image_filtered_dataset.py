"""
사용 가능한 shard 이미지 기반으로 JSONL 필터링.
실행할 때마다 oven_images/ 디렉토리 확인해서 자동으로 coverage 결정.

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/KnowCoL
    python ../codebook-contrastive/scripts/prepare_image_filtered_dataset.py \
        --dataset_dir dataset \
        --output_dir dataset/filtered
"""
import argparse
import json
import os
from pathlib import Path
from collections import Counter


def get_available_prefixes(img_dir: Path) -> set:
    """oven_images/ 아래 존재하는 숫자 폴더만 수집 (01~08)."""
    available = set()
    for d in img_dir.iterdir():
        if d.is_dir() and d.name.isdigit() and len(d.name) == 2:
            # 실제 이미지가 있는지 확인
            if any(d.iterdir()):
                available.add(d.name)
    return available


def filter_jsonl(input_path: Path, output_path: Path, available_prefixes: set):
    """image_id prefix가 available_prefixes에 있는 샘플만 저장."""
    total = kept = 0
    prefix_cnt = Counter()

    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            d = json.loads(line)
            prefix = d['image_id'][5:7]  # oven_XXYYYYYY → XX
            total += 1
            if prefix in available_prefixes:
                fout.write(line)
                kept += 1
                prefix_cnt[prefix] += 1

    return total, kept, prefix_cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='dataset')
    parser.add_argument('--output_dir',  default='dataset/filtered')
    parser.add_argument('--shards', default=None,
                        help='comma-separated shard list to include, e.g. "02,03,04". '
                             'If None, auto-detect from oven_images/')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = dataset_dir / 'oven_images'
    if args.shards:
        available = set(args.shards.split(','))
        print(f"Using specified shards: {sorted(available)}")
    else:
        available = get_available_prefixes(img_dir)
        print(f"Auto-detected shards: {sorted(available)}")

    splits = [
        (dataset_dir / 'oven_data' / 'oven_entity_train.jsonl', output_dir / 'oven_entity_train_img.jsonl'),
        (dataset_dir / 'oven_data' / 'oven_entity_val.jsonl',   output_dir / 'oven_entity_val_img.jsonl'),
        (dataset_dir / 'oven_data' / 'oven_query_train.jsonl',  output_dir / 'oven_query_train_img.jsonl'),
        (dataset_dir / 'oven_data' / 'oven_query_val.jsonl',    output_dir / 'oven_query_val_img.jsonl'),
    ]

    for inp, out in splits:
        if not inp.exists():
            print(f"SKIP (not found): {inp.name}")
            continue
        total, kept, cnt = filter_jsonl(inp, out, available)
        print(f"\n{inp.name}")
        print(f"  {kept:,} / {total:,} kept ({100*kept/total:.1f}%)")
        for p in sorted(cnt):
            print(f"  shard{p}: {cnt[p]:,}")

    print(f"\nFiltered JSONLs saved to: {output_dir}")


if __name__ == '__main__':
    main()
