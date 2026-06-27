"""
Train codebook-contrastive entity retrieval model.

Usage (from KnowCoL root dir):
    python ../codebook-contrastive/scripts/train.py \
        --emb_dir ../codebook-contrastive/outputs/entity_embs \
        --data_dir dataset_01 \
        --max_epochs 10 \
        --batch_size 32 \
        --gpus 1 \
        --run_name codebook-shard01
"""
import sys, os, time, json
sys.path.insert(0, '/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL')
sys.path.insert(0, '/data/guozhiqiang/hanyoushuo/multimodal/codebook-contrastive')

import argparse
import json
import numpy as np
import torch
import open_clip
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.entity_model import CodebookEntityModel
from src.datasets.oven_dataset import OvenEntityDataset
from knowcol.datasets.data_module import _transform


class TrainingStatsCallback(Callback):
    """Tracks training time and GPU memory per epoch, saves to JSON."""

    def __init__(self, out_path: str):
        self.out_path = Path(out_path)
        self.train_start = None
        self.epoch_start = None
        self.epoch_stats = []

    def on_train_start(self, trainer, pl_module):
        self.train_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"\n[Stats] Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch_start is None:
            return
        epoch_secs = time.time() - self.epoch_start
        peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2
                       if torch.cuda.is_available() else 0)
        stat = {
            'epoch': trainer.current_epoch,
            'epoch_time_sec': round(epoch_secs, 1),
            'peak_gpu_mem_mb': round(peak_mem_mb, 1),
            'train_loss': float(trainer.callback_metrics.get('train/loss', 0)),
            'val_loss':   float(trainer.callback_metrics.get('val/loss', 0)),
        }
        self.epoch_stats.append(stat)
        print(f"[Stats] Epoch {stat['epoch']} | "
              f"time={stat['epoch_time_sec']}s | "
              f"peak_GPU={stat['peak_gpu_mem_mb']}MB")
        self._save()

    def on_train_end(self, trainer, pl_module):
        total_secs = time.time() - self.train_start
        summary = {
            'total_train_time_sec': round(total_secs, 1),
            'total_train_time_min': round(total_secs / 60, 1),
            'epochs': self.epoch_stats,
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Stats] Total training time: {total_secs/60:.1f} min")
        print(f"[Stats] Stats saved → {self.out_path}")

    def _save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, 'w') as f:
            json.dump({'epochs': self.epoch_stats}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir',    default='../codebook-contrastive/outputs/entity_embs')
    parser.add_argument('--data_dir',   default='dataset_01')
    parser.add_argument('--max_epochs', type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--temperature',type=float, default=0.07)
    parser.add_argument('--num_workers',type=int,   default=8)
    parser.add_argument('--gpus',       type=int,   default=1)
    parser.add_argument('--run_name',   default='codebook-contrastive-shard01')
    parser.add_argument('--output_dir', default='../codebook-contrastive/outputs/checkpoints')
    parser.add_argument('--wandb_project', default='codebook-contrastive-oven')
    parser.add_argument('--offline',       action='store_true')
    parser.add_argument('--resume_from',   default=None)
    # codebook hard negative
    parser.add_argument('--codebook_dir',  default=None,
                        help='dir with codebook files (kmeans or rq)')
    parser.add_argument('--codebook_type', default='kmeans',
                        choices=['kmeans', 'rq'],
                        help='kmeans: entity_cluster_labels.npy + cluster_buckets.pkl  '
                             'rq: entity_codes_rq.npy + prefix_buckets_rq.pkl (prefix_1 used)')
    parser.add_argument('--n_hard_neg',    type=int, default=16)
    parser.add_argument('--hn_mode',       default='per_sample',
                        choices=['per_sample', 'shared'],
                        help='per_sample: pool=B+K per query  shared: pool=B+B*K (all HN concatenated)')
    args = parser.parse_args()

    # ── Load pre-computed entity embeddings ──────────────────────────────
    emb_dir = Path(args.emb_dir)
    print(f"Loading entity embeddings from {emb_dir}...")
    entity_text_embs = np.load(emb_dir / 'entity_text_embs.npy')
    entity_img_embs  = np.load(emb_dir / 'entity_img_embs.npy')
    with open(emb_dir / 'entity_id_list.json') as f:
        entity_id_list = json.load(f)
    entity_id2idx = {qid: i for i, qid in enumerate(entity_id_list)}

    print(f"  {len(entity_id_list)} entities, text_embs {entity_text_embs.shape}, img_embs {entity_img_embs.shape}")

    # ── CLIP tokenizer & transform ───────────────────────────────────────
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    transform = _transform(224)

    # ── Datasets ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path('/data/guozhiqiang/hanyoushuo/multimodal/KnowCoL') / args.data_dir

    train_ds = OvenEntityDataset(
        data_dir=str(data_dir),
        jsonl_files=['oven_entity_train.jsonl'],
        entity_text_embs=entity_text_embs,
        entity_img_embs=entity_img_embs,
        entity_id2idx=entity_id2idx,
        transform=transform,
        tokenizer=tokenizer,
        split='train',
    )
    val_ds = OvenEntityDataset(
        data_dir=str(data_dir),
        jsonl_files=['oven_entity_val.jsonl'],
        entity_text_embs=entity_text_embs,
        entity_img_embs=entity_img_embs,
        entity_id2idx=entity_id2idx,
        transform=transform,
        tokenizer=tokenizer,
        split='val',
    )
    test_ds = OvenEntityDataset(
        data_dir=str(data_dir),
        jsonl_files=['oven_entity_test.jsonl'],
        entity_text_embs=entity_text_embs,
        entity_img_embs=entity_img_embs,
        entity_id2idx=entity_id2idx,
        transform=transform,
        tokenizer=tokenizer,
        split='test',
        jsonl_folder='test_data',
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ── Model ────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_dir = Path(args.codebook_dir) if args.codebook_dir else None

    # Resolve codebook file paths by type
    codebook_labels_path  = None
    codebook_buckets_path = None
    if codebook_dir:
        if args.codebook_type == 'rq':
            import pickle, numpy as _np
            # RQ: use prefix_1 (coarsest level) as cluster labels/buckets
            codes = _np.load(codebook_dir / 'entity_codes_rq.npy')   # (N, depth)
            rq_labels = codes[:, 0].astype(_np.int64)                 # (N,) first level
            with open(codebook_dir / 'prefix_buckets_rq.pkl', 'rb') as _f:
                prefix_bkts = pickle.load(_f)
            # prefix_1 keys are (c1,) tuples → convert to int-keyed dict
            rq_buckets = {k[0]: v for k, v in prefix_bkts['prefix_1'].items()}

            # Save as temp files that entity_model.py can load normally
            _tmp = codebook_dir / '_tmp_rq_compat'
            _tmp.mkdir(exist_ok=True)
            _np.save(_tmp / 'entity_cluster_labels.npy', rq_labels)
            with open(_tmp / 'cluster_buckets.pkl', 'wb') as _f:
                pickle.dump(rq_buckets, _f)
            codebook_labels_path  = str(_tmp / 'entity_cluster_labels.npy')
            codebook_buckets_path = str(_tmp / 'cluster_buckets.pkl')
            print(f"[RQ] Using prefix_1: {len(rq_buckets)} buckets from {codebook_dir}")
        else:  # kmeans
            codebook_labels_path  = str(codebook_dir / 'entity_cluster_labels.npy')
            codebook_buckets_path = str(codebook_dir / 'cluster_buckets.pkl')

    model = CodebookEntityModel(
        clip_model_name='ViT-L-14',
        clip_pretrained='commonpool_xl_s13b_b90k',
        clip_dim=768,
        hidden_dim=768,
        out_dim=512,
        temperature=args.temperature,
        lr=args.lr,
        entity_text_embs_path=str(emb_dir / 'entity_text_embs.npy'),
        entity_img_embs_path=str(emb_dir / 'entity_img_embs.npy'),
        entity_id_list_path=str(emb_dir / 'entity_id_list.json'),
        codebook_labels_path=codebook_labels_path,
        codebook_buckets_path=codebook_buckets_path,
        n_hard_negatives=args.n_hard_neg,
        hn_mode=args.hn_mode,
    )

    # ── Callbacks & logger ───────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir),
        filename='{epoch}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        verbose=True,
    )
    lr_cb    = LearningRateMonitor(logging_interval='step')
    stats_cb = TrainingStatsCallback(out_path=str(out_dir / 'training_stats.json'))

    wandb_mode = 'offline' if args.offline else 'online'
    os.environ['WANDB_MODE'] = wandb_mode
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.run_name,
        group='entity-linking',
        save_dir=str(out_dir),
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpus,
        strategy='auto',
        precision='16-mixed',
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_cb, stats_cb],
        check_val_every_n_epoch=1,
    )

    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)

    best_ckpt = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    trainer.test(model, test_loader, ckpt_path=best_ckpt)


if __name__ == '__main__':
    main()
