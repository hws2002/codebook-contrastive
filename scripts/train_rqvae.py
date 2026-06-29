"""
EntityRQVAE 학습 스크립트.
입력: entity_text_embs.npy or entity_img_embs.npy (32122, 768)
출력: entity_codes_rqvae.npy (32122, 4) + model checkpoint

Usage:
    cd /workspace/codebook-contrastive
    python scripts/train_rqvae.py \
        --emb_path outputs/entity_embs_img/entity_text_embs.npy \
        --init_codebook_dir outputs/codebook_rq_text \
        --output_dir outputs/rqvae_text \
        --epochs 100 \
        --batch_size 512
"""
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

from src.models.rqvae import EntityRQVAE


def load_kmeans_centroids(codebook_dir: Path, latent_dim: int) -> list:
    """
    기존 build_codebook_rq.py 결과에서 centroid 로드.
    centroids_rq.pkl: list of (256, original_dim) arrays
    → encoder output dim(latent_dim)과 다를 수 있으므로 PCA로 맞춤.
    """
    with open(codebook_dir / 'centroids_rq.pkl', 'rb') as f:
        centroids_list = pickle.load(f)  # list of (256, D)

    result = []
    for c in centroids_list:
        c = np.array(c)
        if c.shape[1] != latent_dim:
            # PCA로 차원 맞추기
            from sklearn.decomposition import PCA
            pca = PCA(n_components=latent_dim)
            c = pca.fit_transform(c)
        # L2 normalize
        c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
        result.append(c.astype(np.float32))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path',          required=True)
    parser.add_argument('--init_codebook_dir', default=None,
                        help='기존 KMeans codebook dir (Option B 초기화)')
    parser.add_argument('--output_dir',        default='outputs/rqvae')
    parser.add_argument('--input_dim',  type=int, default=768)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--depth',      type=int, default=4)
    parser.add_argument('--n_e',        type=int, default=256)
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--mu',         type=float, default=0.25)
    parser.add_argument('--beta',       type=float, default=0.1)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--gpu',        type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load embeddings
    embs = np.load(args.emb_path).astype(np.float32)   # (32122, 768)
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    print(f"Loaded embeddings: {embs_norm.shape}")

    dataset = TensorDataset(torch.from_numpy(embs_norm))
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = EntityRQVAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        depth=args.depth,
        n_e=args.n_e,
        mu=args.mu,
        beta=args.beta,
        recon_weight=args.recon_weight,
    ).to(device)

    # Option B: KMeans centroid로 초기화
    if args.init_codebook_dir:
        print("Initializing codebook from KMeans centroids...")
        centroids = load_kmeans_centroids(Path(args.init_codebook_dir), args.latent_dim)
        model.init_from_kmeans(centroids)
        print("Done.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = recon_loss_sum = rq_loss_sum = 0.0
        for (x,) in loader:
            x = x.to(device)
            loss, recon_loss, rq_loss, _ = model(x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss    += loss.item()
            recon_loss_sum += recon_loss.item()
            rq_loss_sum    += rq_loss.item()
        scheduler.step()

        n = len(loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:3d}/{args.epochs}] "
                  f"loss={total_loss/n:.4f} "
                  f"recon={recon_loss_sum/n:.4f} "
                  f"rq={rq_loss_sum/n:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), out_dir / 'best.pt')

    # Extract codes with best model
    print("\nExtracting codes...")
    model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device))
    model.eval()
    all_embs = torch.from_numpy(embs_norm).to(device)
    with torch.no_grad():
        codes = model.encode(all_embs).cpu().numpy()   # (32122, 4)

    np.save(out_dir / 'entity_codes_rqvae.npy', codes)
    print(f"Saved codes: {codes.shape} → {out_dir}/entity_codes_rqvae.npy")

    # Build prefix_buckets_rq.pkl (same format as build_codebook_rq.py)
    import pickle
    from collections import defaultdict
    prefix_buckets = {}
    for depth in range(1, args.depth + 1):
        key = f'prefix_{depth}'
        buckets = defaultdict(list)
        for eidx, code in enumerate(codes):
            prefix = tuple(code[:depth].tolist())
            buckets[prefix].append(eidx)
        prefix_buckets[key] = dict(buckets)

    with open(out_dir / 'prefix_buckets_rq.pkl', 'wb') as f:
        pickle.dump(prefix_buckets, f)

    # entity_codes_rq.npy로도 저장 (train.py 호환)
    np.save(out_dir / 'entity_codes_rq.npy', codes)

    print(f"Saved prefix_buckets_rq.pkl and entity_codes_rq.npy")

    # Codebook usage stats
    for level in range(args.depth):
        unique = len(np.unique(codes[:, level]))
        print(f"  level {level+1}: {unique}/256 codes used")


if __name__ == '__main__':
    main()
