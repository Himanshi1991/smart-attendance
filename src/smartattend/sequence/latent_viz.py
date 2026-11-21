"""Latent-space visualisations + a small convolutional VAE (generative component) on face crops."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from smartattend.data.drowsiness import ENGAGEMENT_CLASSES


class ConvVAE(nn.Module):
    def __init__(self, z: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Flatten())       # 64x8x8
        self.mu, self.logvar = nn.Linear(64 * 64, z), nn.Linear(64 * 64, z)
        self.dec_in = nn.Linear(z, 64 * 64)
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 16, 4, 2, 1),
                                 nn.ReLU(), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(self.dec_in(z).view(-1, 64, 8, 8))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv


def load_faces64(paths) -> np.ndarray:
    return np.stack([cv2.resize(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE), (64, 64),
                                interpolation=cv2.INTER_AREA) for p in paths]).astype(np.float32)[:, None] / 255


def train_vae(Xtr, Xva, epochs=25, z=8, beta=1.0, seed=0, logger=None):
    torch.manual_seed(seed)
    vae = ConvVAE(z)
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    Xtr, Xva = torch.tensor(Xtr), torch.tensor(Xva)
    hist = []
    for ep in range(epochs):
        vae.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            x = Xtr[perm[i:i + 64]]
            rec, mu, lv = vae(x)
            rl = nn.functional.binary_cross_entropy(rec, x, reduction="sum") / len(x)
            kl = -0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp()) / len(x)
            opt.zero_grad()
            (rl + beta * kl).backward()
            opt.step()
        vae.eval()
        with torch.no_grad():
            rec, mu, lv = vae(Xva)
            hist.append({"epoch": ep + 1, "val_recon_bce": float(nn.functional.binary_cross_entropy(rec, Xva, reduction="sum") / len(Xva)),
                         "val_kl": float(-0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp()) / len(Xva))})
        if logger and (ep + 1) % 5 == 0:
            logger.info(f"[vae] epoch {ep + 1} {hist[-1]}")
    return vae, hist


def run(df, engagement_model, out_dir: Path, logger: logging.Logger) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = (df[df["split"] == s] for s in ["train", "val", "test"])
    Xtr, Xva, Xte = load_faces64(tr["path"]), load_faces64(va["path"]), load_faces64(te["path"])
    vae, hist = train_vae(Xtr, Xva, logger=logger)
    torch.save(vae.state_dict(), out_dir / "vae_faces.pt")

    # 1. engagement CNN embeddings (t-SNE) vs VAE latent (PCA), coloured by true class
    from smartattend.vision.engagement import _load_gray

    crops = [_load_gray(p) for p in te["path"]]
    E = engagement_model.embed(crops)
    with torch.no_grad():
        mu, _ = vae.encode(torch.tensor(Xte))
    y = te["engagement"].to_numpy()
    Zt = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(E)
    Zv = PCA(2).fit_transform(mu.numpy())
    from sklearn.metrics import silhouette_score

    res = {"vae_history": hist,
           "silhouette_cnn_embedding": float(silhouette_score(E, y)),
           "silhouette_vae_latent": float(silhouette_score(mu.numpy(), y))}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, name in enumerate(ENGAGEMENT_CLASSES):
        axes[0].scatter(*Zt[y == c].T, s=10, label=name)
        axes[1].scatter(*Zv[y == c].T, s=10, label=name)
    axes[0].set_title(f"supervised ResNet-18 embedding (t-SNE), silhouette={res['silhouette_cnn_embedding']:.2f}")
    axes[1].set_title(f"unsupervised VAE latent (PCA), silhouette={res['silhouette_vae_latent']:.2f}")
    for a in axes:
        a.legend(), a.grid(alpha=0.3)
    fig.savefig(out_dir / "latent_space.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. VAE reconstructions, random samples and a latent traversal along the engaged->disengaged direction
    with torch.no_grad():
        rec, _, _ = vae(torch.tensor(Xte[:8]))
        samples = vae.decode(torch.randn(8, mu.shape[1]))
        m_eng, m_dis = mu[y == 0].mean(0), mu[y == 2].mean(0)
        trav = vae.decode(torch.stack([m_eng + a * (m_dis - m_eng) for a in np.linspace(0, 1, 8)]))
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
    for j in range(8):
        for i, (img, title) in enumerate([(Xte[j, 0], "input"), (rec[j, 0], "reconstruction"),
                                          (samples[j, 0], "sample z~N(0,I)"), (trav[j, 0], "engaged -> disengaged")]):
            axes[i, j].imshow(np.asarray(img), cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(title, fontsize=8, loc="left")
    fig.suptitle("Convolutional VAE on face crops (generative model)")
    fig.savefig(out_dir / "vae_faces.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    (out_dir / "latent_space.json").write_text(json.dumps(res, indent=2))
    logger.info(f"[latent] silhouettes cnn={res['silhouette_cnn_embedding']:.3f} vae={res['silhouette_vae_latent']:.3f}")
    return res
