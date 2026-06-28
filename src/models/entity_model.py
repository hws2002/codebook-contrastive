import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import open_clip
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from .heads import QueryEntityHead, EntityItemHead
from knowcol.datasets.data_module import _transform
from knowcol.datasets.utils import _load_image_from_path
from knowcol.evaluations.evaluation import recall_at_k


class CodebookEntityModel(pl.LightningModule):
    """
    frozen CLIP + QueryEntityHead + EntityItemHead.
    Loss: symmetric InfoNCE with in-batch + optional codebook hard negatives.
    """

    def __init__(
        self,
        clip_model_name: str = 'ViT-L-14',
        clip_pretrained: str = 'commonpool_xl_s13b_b90k',
        clip_dim: int = 768,
        hidden_dim: int = 768,
        out_dim: int = 512,
        temperature: float = 0.07,
        lr: float = 1e-4,
        entity_text_embs_path: str = None,
        entity_img_embs_path: str = None,
        entity_id_list_path: str = None,
        # codebook hard negative (text)
        codebook_labels_path: str = None,
        codebook_buckets_path: str = None,
        # codebook hard negative (image) — optional dual codebook
        codebook_img_labels_path: str = None,
        codebook_img_buckets_path: str = None,
        n_hard_negatives: int = 16,
        hn_mode: str = 'per_sample',   # 'per_sample' (B+K pool) or 'shared' (B+B*K pool)
    ):
        super().__init__()
        self.save_hyperparameters()

        clip, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
        self.clip = clip.eval()
        self.clip.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        self.query_entity_head = QueryEntityHead(in_dim=2*clip_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.entity_item_head  = EntityItemHead(in_dim=2*clip_dim,  hidden_dim=hidden_dim, out_dim=out_dim)

        self.temperature = temperature
        self.lr = lr
        self.n_hard_negatives = n_hard_negatives
        self.hn_mode = hn_mode

        import pickle

        # Text codebook
        self.use_codebook_hn = (codebook_labels_path is not None and
                                codebook_buckets_path is not None)
        if self.use_codebook_hn:
            self._cluster_labels = np.load(codebook_labels_path)
            with open(codebook_buckets_path, 'rb') as f:
                self._cluster_buckets = pickle.load(f)
            print(f"[Text Codebook HN] {len(self._cluster_buckets)} clusters, K={n_hard_negatives}")
        else:
            self._cluster_labels  = None
            self._cluster_buckets = None

        # Image codebook (optional dual)
        self.use_img_codebook_hn = (codebook_img_labels_path is not None and
                                    codebook_img_buckets_path is not None)
        if self.use_img_codebook_hn:
            self._img_cluster_labels = np.load(codebook_img_labels_path)
            with open(codebook_img_buckets_path, 'rb') as f:
                self._img_cluster_buckets = pickle.load(f)
            print(f"[Image Codebook HN] {len(self._img_cluster_buckets)} clusters, K={n_hard_negatives}")
        else:
            self._img_cluster_labels  = None
            self._img_cluster_buckets = None

        # Pre-compute HN index table (N_ent, K) — fixed, built once
        self._hn_table = None
        if self.use_codebook_hn or self.use_img_codebook_hn:
            self._hn_table = self._build_hn_table(n_hard_negatives)
            print(f"[HN Table] Pre-computed: shape={self._hn_table.shape}")

        # Load entity embeddings for hard negative encoding
        if entity_text_embs_path:
            self._all_text_embs = torch.from_numpy(
                np.load(entity_text_embs_path)).float()
            self._all_img_embs  = torch.from_numpy(
                np.load(entity_img_embs_path)).float()
        else:
            self._all_text_embs = None
            self._all_img_embs  = None

        self._entity_store = None
        self._entity_ids   = None

    # ── encoding ────────────────────────────────────────────────────────────

    def encode_query(self, images, question_toks):
        with torch.no_grad():
            z_img  = self.clip.encode_image(images)
            z_text = self.clip.encode_text(question_toks)
        return self.query_entity_head(torch.cat([z_img, z_text], dim=-1))   # (B, D)

    def encode_entity(self, ent_text_embs, ent_img_embs):
        return self.entity_item_head(torch.cat([ent_text_embs, ent_img_embs], dim=-1))  # (B, D)

    # ── training ────────────────────────────────────────────────────────────

    def _infonce_inbatch(self, z_q, z_e):
        """Standard symmetric InfoNCE with in-batch negatives only. (B, B) logits."""
        logits = z_q @ z_e.T / self.temperature
        labels = torch.arange(z_q.size(0), device=self.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def _infonce_per_sample_hn(self, z_q, z_e, z_hn_per):
        """
        Per-sample InfoNCE with codebook hard negatives.
        Each sample i sees: B in-batch entities + its own K HN.
        Pool per sample = B + K (not B + B*K).
        False negatives (batch positives in HN) already filtered at sampling.

        z_q:       (B, D)
        z_e:       (B, D)   in-batch entity embs
        z_hn_per:  (B, K, D) per-sample hard negatives
        """
        B = z_q.size(0)
        losses = []
        for i in range(B):
            # pool_i: [all B in-batch entities] + [K HN for sample i]
            pool_i  = torch.cat([z_e, z_hn_per[i]], dim=0)    # (B+K, D)
            logit_i = z_q[i] @ pool_i.T / self.temperature    # (B+K,)
            label_i = torch.tensor(i, device=self.device)
            losses.append(F.cross_entropy(logit_i.unsqueeze(0), label_i.unsqueeze(0)))
        return torch.stack(losses).mean()

    def _infonce_shared_hn(self, z_q, z_e, z_hn_per):
        """
        Shared InfoNCE: each sample i sees B in-batch + ALL B*K hard negatives.
        Pool per sample = B + B*K.
        z_hn_per: (B, K, D)  — already false-negative-filtered per sample,
                               so no sample j's HN contains sample i's positive.
        """
        B = z_q.size(0)
        z_hn_flat = z_hn_per.view(-1, z_hn_per.size(-1))   # (B*K, D)
        pool = torch.cat([z_e, z_hn_flat], dim=0)           # (B + B*K, D)
        losses = []
        for i in range(B):
            logit_i = z_q[i] @ pool.T / self.temperature    # (B+B*K,)
            label_i = torch.tensor(i, device=self.device)
            losses.append(F.cross_entropy(logit_i.unsqueeze(0), label_i.unsqueeze(0)))
        return torch.stack(losses).mean()

    def _build_hn_table(self, K: int) -> np.ndarray:
        """전체 entity에 대해 HN 인덱스 테이블 (N, K) 1회 생성."""
        rng = np.random.default_rng(42)
        N = len(self._cluster_labels) if self.use_codebook_hn else len(self._img_cluster_labels)
        K_text = K // 2 if (self.use_codebook_hn and self.use_img_codebook_hn) else K
        K_img  = K - K_text if self.use_img_codebook_hn else 0
        table  = np.empty((N, K), dtype=np.int64)

        for eidx in range(N):
            cols = []
            if self.use_codebook_hn:
                cid  = (int(self._cluster_labels[eidx]),)
                pool = [x for x in self._cluster_buckets.get(cid, []) if x != eidx]
                if len(pool) < K_text:
                    pool = [x for x in range(N) if x != eidx]
                chosen = rng.choice(pool, size=K_text, replace=len(pool) < K_text)
                cols.append(chosen)
            if self.use_img_codebook_hn:
                cid  = (int(self._img_cluster_labels[eidx]),)
                pool = [x for x in self._img_cluster_buckets.get(cid, []) if x != eidx]
                if len(pool) < K_img:
                    pool = [x for x in range(N) if x != eidx]
                chosen = rng.choice(pool, size=K_img, replace=len(pool) < K_img)
                cols.append(chosen)
            table[eidx] = np.concatenate(cols)

        return table  # (N, K)

    def _sample_from_codebook(self, idxs_np, batch_positives, K,
                              cluster_labels, cluster_buckets):
        """한 codebook에서 per-sample HN index 샘플링. (B, K) ndarray 반환."""
        rng = np.random.default_rng()
        B = len(idxs_np)
        all_hn_idxs = np.empty((B, K), dtype=np.int64)
        for i, eidx in enumerate(idxs_np):
            cid  = int(cluster_labels[eidx])
            pool = [x for x in cluster_buckets[cid]
                    if x != eidx and x not in batch_positives]
            if len(pool) < K:
                pool = [x for x in range(len(cluster_labels))
                        if x not in batch_positives and x != eidx]
            k = min(K, len(pool))
            chosen = rng.choice(pool, size=k, replace=False)
            if k < K:
                chosen = np.concatenate([chosen, rng.choice(chosen, size=K - k, replace=True)])
            all_hn_idxs[i] = chosen
        return all_hn_idxs

    def _sample_hard_negatives(self, entity_idxs: torch.Tensor):
        """
        Pre-computed HN table에서 인덱싱 → Python loop 없음.
        Returns (B, K, D).
        """
        idxs_np = entity_idxs.cpu().numpy()
        # 테이블 인덱싱 (B, K)
        all_hn_idxs = self._hn_table[idxs_np]  # numpy fancy indexing, 매우 빠름

        flat = all_hn_idxs.flatten()
        te = self._all_text_embs[flat].to(self.device)
        ie = self._all_img_embs[flat].to(self.device)
        z_flat = self.encode_entity(te, ie)
        B, K = all_hn_idxs.shape
        return z_flat.view(B, K, -1)  # (B, K, D)

    def training_step(self, batch, batch_idx):
        z_q = self.encode_query(batch['image'], batch['question_tok'])
        z_e = self.encode_entity(batch['ent_text_emb'], batch['ent_img_emb'])  # (B, D)

        if self.use_codebook_hn:
            z_hn_per = self._sample_hard_negatives(batch['entity_idx'])  # (B, K, D)
            if self.hn_mode == 'shared':
                loss = self._infonce_shared_hn(z_q, z_e, z_hn_per)
            else:
                loss = self._infonce_per_sample_hn(z_q, z_e, z_hn_per)
        else:
            loss = self._infonce_inbatch(z_q, z_e)

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        # Build entity store once per val epoch for Recall@K
        hp = self.hparams
        if not hasattr(self, '_val_entity_store') or self._val_entity_store is None:
            text_embs = self._all_text_embs if self._all_text_embs is not None else \
                        torch.from_numpy(np.load(hp.entity_text_embs_path)).float()
            img_embs  = self._all_img_embs  if self._all_img_embs  is not None else \
                        torch.from_numpy(np.load(hp.entity_img_embs_path)).float()
            with open(hp.entity_id_list_path) as f:
                self._val_entity_ids = json.load(f)
            self._val_entity_id2idx = {qid: i for i, qid in enumerate(self._val_entity_ids)}

        # Build encoded store (updated each epoch as heads change)
        n, B = len(self._val_entity_ids), 512
        text_embs = self._all_text_embs if self._all_text_embs is not None else \
                    torch.from_numpy(np.load(hp.entity_text_embs_path)).float()
        img_embs  = self._all_img_embs  if self._all_img_embs  is not None else \
                    torch.from_numpy(np.load(hp.entity_img_embs_path)).float()
        store = []
        with torch.no_grad():
            for s in range(0, n, B):
                te = text_embs[s:s+B].to(self.device)
                ie = img_embs[s:s+B].to(self.device)
                store.append(self.encode_entity(te, ie).cpu())
        self._val_entity_store = torch.cat(store, dim=0)  # (N_ent, D)
        self._val_preds = []

    def validation_step(self, batch, batch_idx):
        z_q = self.encode_query(batch['image'], batch['question_tok'])
        z_e = self.encode_entity(batch['ent_text_emb'], batch['ent_img_emb'])
        loss = self._infonce_inbatch(z_q, z_e)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        # Recall@K against full entity store
        store = self._val_entity_store.to(self.device)
        scores   = z_q @ store.T
        topk_idx = torch.topk(scores, k=10, dim=-1).indices
        for i in range(z_q.shape[0]):
            self._val_preds.append({
                'entity_id':       batch['entity_id'][i],
                'pred_entity_ids': [self._val_entity_ids[j] for j in topk_idx[i].tolist()],
            })

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        from knowcol.evaluations.evaluation import recall_at_k
        result = recall_at_k(self._val_preds, ks=[1, 5, 10])
        self.log('val/recall@1',  result['recall@1'],  prog_bar=True)
        self.log('val/recall@5',  result['recall@5'])
        self.log('val/recall@10', result['recall@10'])
        self._val_preds = []

    # ── test ────────────────────────────────────────────────────────────────

    def on_test_start(self):
        hp = self.hparams
        text_embs = torch.from_numpy(np.load(hp.entity_text_embs_path)).float()
        img_embs  = torch.from_numpy(np.load(hp.entity_img_embs_path)).float()
        with open(hp.entity_id_list_path) as f:
            self._entity_ids = json.load(f)

        n, B = text_embs.shape[0], 512
        store = []
        self.entity_item_head.eval()
        with torch.no_grad():
            for s in tqdm(range(0, n, B), desc='Building entity store'):
                te = text_embs[s:s+B].to(self.device)
                ie = img_embs[s:s+B].to(self.device)
                store.append(self.encode_entity(te, ie).cpu())
        self._entity_store = torch.cat(store, dim=0)   # (N_ent, D)
        print(f"Entity store: {self._entity_store.shape}")
        self._test_preds = []

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        z_q   = self.encode_query(batch['image'], batch['question_tok'])
        store = self._entity_store.to(self.device)
        scores   = z_q @ store.T                                      # (B, N_ent)
        topk_idx = torch.topk(scores, k=10, dim=-1).indices           # (B, 10)
        for i in range(z_q.shape[0]):
            self._test_preds.append({
                'data_id':         batch['data_id'][i],
                'entity_id':       batch['entity_id'][i],
                'pred_entity_ids': [self._entity_ids[j] for j in topk_idx[i].tolist()],
            })

    def on_test_epoch_end(self):
        result = recall_at_k(self._test_preds, ks=[1, 5, 10])
        self.log('test/recall@1',  result['recall@1'],  add_dataloader_idx=False)
        self.log('test/recall@5',  result['recall@5'],  add_dataloader_idx=False)
        self.log('test/recall@10', result['recall@10'], add_dataloader_idx=False)
        print(f"\n[TEST] Recall@1={result['recall@1']:.2f}  "
              f"Recall@5={result['recall@5']:.2f}  "
              f"Recall@10={result['recall@10']:.2f}  "
              f"N={len(self._test_preds)}")
        import json as _json

        # run별 고유 폴더에 저장 (checkpoint 폴더 기준)
        try:
            ckpt_dir = Path(self.trainer.checkpoint_callback.dirpath)
        except Exception:
            ckpt_dir = Path('outputs/results/unknown')
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        with open(ckpt_dir / 'test_predictions.jsonl', 'w') as f:
            for p in self._test_preds:
                f.write(_json.dumps(p) + '\n')
        result['n_samples'] = len(self._test_preds)
        with open(ckpt_dir / 'test_results.json', 'w') as f:
            _json.dump(result, f, indent=2)
        print(f"Saved → {ckpt_dir}/test_predictions.jsonl")
        print(f"Saved → {ckpt_dir}/test_results.json")

    # ── optimizer ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        params = (list(self.query_entity_head.parameters()) +
                  list(self.entity_item_head.parameters()))
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {'optimizer': opt, 'lr_scheduler': sched}
