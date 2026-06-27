import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


extensions = ['.JPEG', '.jpg', '.jpeg', '.png']


def _load_image(img_id: str, data_dir: Path) -> Image.Image:
    shard = img_id[5:7]  # oven_01XXXXXX → '01'
    for ext in extensions:
        p = data_dir / 'oven_images' / shard / (img_id + ext)
        if p.exists():
            try:
                return Image.open(p).convert('RGB')
            except Exception:
                pass
    return Image.new('RGB', (224, 224))


class OvenEntityDataset(Dataset):
    """
    Dataset for entity linking.
    Returns query image, tokenized question, and pre-computed entity embeddings.

    entity_text_embs: np.ndarray (N_ent, D)
    entity_img_embs:  np.ndarray (N_ent, D)
    entity_id_list:   list of QIDs, index matches embedding rows
    """

    def __init__(
        self,
        data_dir: str,
        jsonl_files: list,
        entity_text_embs: np.ndarray,
        entity_img_embs: np.ndarray,
        entity_id2idx: dict,
        transform,
        tokenizer,
        split: str = 'train',
        jsonl_folder: str = 'oven_data',
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.entity_text_embs = torch.from_numpy(entity_text_embs).float()
        self.entity_img_embs  = torch.from_numpy(entity_img_embs).float()
        self.entity_id2idx = entity_id2idx
        self.split = split

        self.samples = []
        for jf in jsonl_files:
            with open(self.data_dir / jsonl_folder / jf) as f:
                for line in f:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = _load_image(item['image_id'], self.data_dir)
        if self.transform:
            image = self.transform(image)

        question_tok = self.tokenizer(item['question']).squeeze()

        entity_id  = item['entity_id']
        entity_idx = self.entity_id2idx.get(entity_id, 0)

        ent_text_emb = self.entity_text_embs[entity_idx]
        ent_img_emb  = self.entity_img_embs[entity_idx]

        return {
            'image':         image,
            'question_tok':  question_tok,
            'entity_id':     entity_id,
            'entity_idx':    entity_idx,
            'ent_text_emb':  ent_text_emb,
            'ent_img_emb':   ent_img_emb,
            'data_id':       item['data_id'],
        }
