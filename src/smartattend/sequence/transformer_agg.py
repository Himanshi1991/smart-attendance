"""Transformer feature aggregator: per-student tokens -> class-level engagement estimate.

Each frame yields a variable-size SET of student tokens:
    [p_engaged, p_neutral, p_disengaged, face_size, detection_conf, head_turn]
The aggregator (a Transformer encoder with a learnable [CLS] token, no positional encoding, so it is
permutation-invariant) predicts the TRUE fraction of engaged students. Small / low-confidence faces
give unreliable classifier outputs, so a good aggregator must learn to discount them - attention
weights make that visible.

Training frames are simulated, but the per-student probability vectors are BOOTSTRAPPED from the real
engagement classifier's held-out predictions (grouped by true class), so classifier errors are real.
Baselines: mean of hard labels, mean of probabilities, DeepSets (per-token MLP + mean pool).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TOKEN_FEATURES = ["p_engaged", "p_neutral", "p_disengaged", "face_size", "det_conf", "head_turn"]


def simulate_frames(n: int, probs_by_class: dict[int, np.ndarray], seed: int = 0, max_students: int = 30):
    """Return tokens (n, max_students, 6), pad mask (n, max_students) [True = padding], targets (n,)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, max_students, len(TOKEN_FEATURES)), np.float32)
    pad = np.ones((n, max_students), bool)
    y = np.zeros(n, np.float32)
    for f in range(n):
        k = rng.integers(5, max_students + 1)
        frac_eng = rng.beta(2, 2)
        true = np.where(rng.random(k) < frac_eng, 0, rng.choice([1, 2], size=k, p=[0.6, 0.4]))
        size = rng.uniform(0.0, 1.0, k)                  # 0 = tiny back-row face, 1 = large front-row face
        conf = np.clip(0.5 + 0.5 * size + rng.normal(0, 0.1, k), 0, 1)
        turn = np.abs(rng.normal(0, 0.3, k)) + (true != 0) * rng.uniform(0, 0.4, k)
        for i, c in enumerate(true):
            p = probs_by_class[c][rng.integers(len(probs_by_class[c]))]
            # small faces: classifier output degrades toward a random other student's output
            reliability = size[i] ** 1.5
            if rng.random() > reliability:
                other = rng.integers(3)
                p = probs_by_class[other][rng.integers(len(probs_by_class[other]))]
            X[f, i] = [*p, size[i], conf[i], turn[i]]
        pad[f, :k] = False
        y[f] = np.mean(true == 0)
    return X, pad, y


class TransformerAggregator(nn.Module):
    def __init__(self, d_in: int = 6, d_model: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, heads, 64, dropout=0.1, batch_first=True)
                                     for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x, pad, return_attn: bool = False):
        h = torch.cat([self.cls.expand(len(x), -1, -1), self.embed(x)], 1)
        mask = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool), pad], 1)
        attn = None
        for i, layer in enumerate(self.layers):
            if return_attn and i == len(self.layers) - 1:
                _, attn = layer.self_attn(h, h, h, key_padding_mask=mask, need_weights=True, average_attn_weights=True)
            h = layer(h, src_key_padding_mask=mask)
        out = self.head(h[:, 0]).squeeze(-1)
        return (out, attn[:, 0, 1:]) if return_attn else out  # CLS -> student attention


class DeepSets(nn.Module):
    def __init__(self, d_in: int = 6, d: int = 32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1), nn.Sigmoid())

    def forward(self, x, pad):
        h = self.phi(x) * (~pad).unsqueeze(-1)
        return self.rho(h.sum(1) / (~pad).sum(1, keepdim=True)).squeeze(-1)


def _fit(model, data, epochs=30, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    (Xtr, Ptr, ytr), (Xva, Pva, yva) = data
    Xtr, Ptr, ytr = map(torch.tensor, (Xtr, Ptr, ytr))
    Xva, Pva, yva = map(torch.tensor, (Xva, Pva, yva))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, state = np.inf, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.mse_loss(model(Xtr[idx], Ptr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float((model(Xva, Pva) - yva).abs().mean())
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model


def run(probs_by_class: dict[int, np.ndarray], out_dir: Path, logger: logging.Logger) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = (simulate_frames(n, probs_by_class, seed=s) for n, s in [(6000, 0), (1000, 1), (2000, 2)])
    Xte, Pte, yte = te
    valid = ~Pte
    res = {"simulated_frames": True,
           "mean_hard_label_MAE": float(np.mean(np.abs(((Xte[..., :3].argmax(-1) == 0) * valid).sum(1) / valid.sum(1) - yte))),
           "mean_prob_MAE": float(np.mean(np.abs((Xte[..., 0] * valid).sum(1) / valid.sum(1) - yte)))}
    ds = _fit(DeepSets(), (tr, va))
    tf = _fit(TransformerAggregator(), (tr, va))
    with torch.no_grad():
        res["deepsets_MAE"] = float((ds(torch.tensor(Xte), torch.tensor(Pte)) - torch.tensor(yte)).abs().mean())
        pred, attn = tf(torch.tensor(Xte), torch.tensor(Pte), return_attn=True)
        res["transformer_MAE"] = float((pred - torch.tensor(yte)).abs().mean())
    attn = attn.numpy()
    sizes, weights = Xte[..., 3][valid], attn[valid]
    counts = valid.sum(1, keepdims=True).repeat(valid.shape[1], 1)[valid]
    rel = weights * counts  # >1 means more attention than uniform
    res["attention_vs_face_size_corr"] = float(np.corrcoef(sizes, rel)[0, 1])
    logger.info(f"[aggregator] {res}")
    torch.save(tf.state_dict(), out_dir / "transformer_aggregator.pt")
    (out_dir / "transformer_aggregator.json").write_text(json.dumps(res, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    names = ["mean hard label", "mean prob", "DeepSets", "Transformer"]
    vals = [res["mean_hard_label_MAE"], res["mean_prob_MAE"], res["deepsets_MAE"], res["transformer_MAE"]]
    axes[0].bar(names, vals, color=["C7", "C7", "C0", "C1"])
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.002, f"{v:.3f}", ha="center")
    axes[0].set_title("class engaged-fraction MAE (simulated frames)")
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(sizes, bins) - 1
    axes[1].plot((bins[:-1] + bins[1:]) / 2, [rel[idx == b].mean() for b in range(10)], "o-")
    axes[1].axhline(1, color="k", ls="--", lw=1)
    axes[1].set_xlabel("face size (0 = back row, 1 = front)"), axes[1].set_ylabel("attention / uniform")
    axes[1].set_title(f"CLS attention vs face size (r={res['attention_vs_face_size_corr']:.2f})")
    f = int(np.argmax(valid.sum(1)))
    k = int(valid[f].sum())
    axes[2].bar(range(k), attn[f, :k], color=plt.cm.viridis(Xte[f, :k, 3]))
    axes[2].set_xlabel("student token (colour = face size)"), axes[2].set_ylabel("attention weight")
    axes[2].set_title(f"one frame: true {yte[f]:.2f}, predicted {float(pred[f]):.2f}")
    for a in axes:
        a.grid(alpha=0.3)
    fig.savefig(out_dir / "transformer_aggregator.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return res
