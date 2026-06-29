"""
Phase 0: Extract CLIP embeddings for all entities in KG.
Saves z_ent_text and z_ent_img separately.

Usage:
    cd /data/guozhiqiang/hanyoushuo/multimodal/KnowCoL
    python ../codebook-contrastive/scripts/extract_entity_embeddings.py \
        --output_dir ../codebook-contrastive/outputs/entity_embs
"""
import sys, os
sys.path.insert(0, '/workspace/KnowCoL')

import argparse
import json
import numpy as np
import torch
import open_clip
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from knowcol.datasets.kg import KG
from knowcol.datasets.data_module import _transform
from knowcol.datasets.utils import _load_image_from_path

CLIP_MODEL   = 'ViT-L-14'
CLIP_PRETRAINED = 'commonpool_xl_s13b_b90k'
N_PX = 224
BATCH_SIZE = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='../codebook-contrastive/outputs/entity_embs')
    parser.add_argument('--kb_path',    default='dataset/knowledge_base')
    parser.add_argument('--kg_dir',     default='dataset/wikidata_subgraph_v1')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load CLIP
    print(f"Loading CLIP {CLIP_MODEL}...")
    clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    transform = _transform(N_PX)

    # Load KG
    print("Loading KG...")
    kg = KG(
        knowledge_base_path=args.kb_path,
        entity_set_path=f'{args.kg_dir}/entity.txt',
        relation_path=f'{args.kg_dir}/relation.txt',
        triplet_h_path=f'{args.kg_dir}/triplet_h.jsonl',
        triplet_t_path=f'{args.kg_dir}/triplet_t.jsonl',
    )
    n_ent = kg.n_ent
    print(f"Total entities: {n_ent}")

    entity_ids = kg.entities  # list of QIDs in order

    # Load Wiki6M to get title as fallback for empty summaries
    print("Loading Wiki6M for title fallback...")
    qid2title = {}
    wiki6m_path = Path('dataset/knowledge_base/Wiki6M_ver_1_1.jsonl')
    if wiki6m_path.exists():
        import sys as _sys
        from knowcol.__init__ import __file__ as _kc_file
        wiki6m_abs = Path(_kc_file).parent.parent / wiki6m_path
        ent_set = set(entity_ids)
        with open(wiki6m_abs) as f:
            for line in f:
                d = json.loads(line)
                if d['wikidata_id'] in ent_set:
                    qid2title[d['wikidata_id']] = d.get('wikipedia_title', '')

    def get_entity_text(qid):
        """title + summary. Fallback 순서: summary > title > QID"""
        summary = kg.ent_info[qid]['text'] if qid in kg.ent_info else ''
        title   = qid2title.get(qid, '')
        if summary:
            return f"{title}. {summary}" if title else summary
        elif title:
            return title
        else:
            return f"entity {qid}"  # last resort

    text_embs = np.zeros((n_ent, 768), dtype=np.float32)
    img_embs  = np.zeros((n_ent, 768), dtype=np.float32)

    # --- Text embeddings (batch) ---
    print("Extracting text embeddings...")
    texts = [get_entity_text(qid) for qid in entity_ids]

    # Coverage stats
    n_has_summary = sum(1 for qid in entity_ids if qid in kg.ent_info and kg.ent_info[qid]['text'])
    n_title_only  = sum(1 for qid in entity_ids if (not (qid in kg.ent_info and kg.ent_info[qid]['text'])) and qid2title.get(qid))
    n_qid_only    = n_ent - n_has_summary - n_title_only
    print(f"  summary+title: {n_has_summary} | title-only: {n_title_only} | QID-only fallback: {n_qid_only}")

    with torch.no_grad():
        for start in tqdm(range(0, n_ent, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, n_ent)
            batch_texts = texts[start:end]
            tok = tokenizer(batch_texts).to(device)
            emb = clip_model.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            text_embs[start:end] = emb.float().cpu().numpy()

    # --- Image embeddings (one by one, some missing) ---
    print("Extracting image embeddings...")
    # kg.get_ent_image() → knowledge_base/wikipedia_images_full/Q488/Q488.jpg
    img_paths = [kg._qid2img(qid) for qid in entity_ids]

    n_has_img = sum(1 for p in img_paths if Path(p).exists())
    print(f"  images found: {n_has_img} / {n_ent}")

    with torch.no_grad():
        for i in tqdm(range(n_ent)):
            img_path = img_paths[i]
            if img_path and Path(img_path).exists():
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(device)
                    emb = clip_model.encode_image(img_t)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    img_embs[i] = emb.float().cpu().numpy()
                except Exception:
                    img_embs[i] = text_embs[i]
            else:
                # fallback: use text embedding
                img_embs[i] = text_embs[i]

    # Save
    np.save(out_dir / 'entity_text_embs.npy', text_embs)
    np.save(out_dir / 'entity_img_embs.npy',  img_embs)
    with open(out_dir / 'entity_id_list.json', 'w') as f:
        json.dump(entity_ids, f)

    print(f"\nSaved to {out_dir}")
    print(f"  entity_text_embs.npy : {text_embs.shape}")
    print(f"  entity_img_embs.npy  : {img_embs.shape}")
    print(f"  entity_id_list.json  : {len(entity_ids)} QIDs")


if __name__ == '__main__':
    main()
