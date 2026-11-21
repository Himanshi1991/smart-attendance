"""Compact energy-based experiment: a binary Restricted Boltzmann Machine trained with CD-1."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from smartattend.nnfund.neurons import sigmoid


def bars_and_stripes(n: int = 4) -> np.ndarray:
    """All n x n images where every row (stripes) or every column (bars) is uniformly on/off."""
    patterns = set()
    for bits in itertools.product([0, 1], repeat=n):
        col = np.array(bits)[:, None]
        patterns.add(tuple(np.repeat(col, n, axis=1).ravel()))  # stripes
        patterns.add(tuple(np.repeat(col.T, n, axis=0).ravel()))  # bars
    return np.array(sorted(patterns), dtype=np.float64)


class RBM:
    """E(v, h) = -a.v - b.h - v^T W h for binary visible v and hidden h."""

    def __init__(self, n_visible: int, n_hidden: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(scale=0.01, size=(n_visible, n_hidden))
        self.a = np.zeros(n_visible)
        self.b = np.zeros(n_hidden)

    def init_visible_bias(self, V: np.ndarray) -> "RBM":
        """Start each visible bias at log(p / (1 - p)) of that pixel's average (Hinton, 'A practical guide to training
        RBMs', 2010). Without it, CD-1 on face images spends its updates on the biases and barely learns (Requirement 6)."""
        p = np.clip(V.mean(axis=0), 0.02, 0.98)
        self.a = np.log(p / (1 - p))
        return self

    def p_h(self, v):
        return sigmoid(v @ self.W + self.b)

    def p_v(self, h):
        return sigmoid(h @ self.W.T + self.a)

    def energy(self, v, h):
        return -(v @ self.a) - (h @ self.b) - np.sum((v @ self.W) * h, axis=-1)

    def free_energy(self, v):
        """F(v) = -log sum_h exp(-E(v,h)) = -a.v - sum_j softplus(b_j + v.W_j)."""
        return -(v @ self.a) - np.sum(np.logaddexp(0.0, v @ self.W + self.b), axis=-1)

    def reconstruct(self, v):
        return self.p_v(self.p_h(v))

    def cd1_step(self, v0: np.ndarray, lr: float) -> None:
        ph0 = self.p_h(v0)
        h0 = (self.rng.random(ph0.shape) < ph0).astype(np.float64)
        pv1 = self.p_v(h0)
        v1 = (self.rng.random(pv1.shape) < pv1).astype(np.float64)
        ph1 = self.p_h(v1)
        n = len(v0)
        # positive phase (data) minus negative phase (model sample)
        self.W += lr * (v0.T @ ph0 - v1.T @ ph1) / n
        self.a += lr * np.mean(v0 - v1, axis=0)
        self.b += lr * np.mean(ph0 - ph1, axis=0)

    def fit(self, V: np.ndarray, epochs: int = 2000, lr: float = 0.1, batch_size: int = 10) -> dict:
        hist = {"recon_error": [], "free_energy": []}
        for _ in range(epochs):
            order = self.rng.permutation(len(V))
            for s in range(0, len(V), batch_size):
                self.cd1_step(V[order[s:s + batch_size]], lr)
            hist["recon_error"].append(float(np.mean((V - self.reconstruct(V)) ** 2)))
            hist["free_energy"].append(float(np.mean(self.free_energy(V))))
        return hist


def make_figures(out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = bars_and_stripes(4)
    rbm = RBM(16, 12, seed=0)
    hist = rbm.fit(V, epochs=1500, lr=0.1)

    rng = np.random.default_rng(1)
    noise_v = (rng.random((len(V), 16)) < 0.5).astype(float)

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(hist["recon_error"]), ax1.set_title("reconstruction MSE"), ax1.set_xlabel("epoch")
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(hist["free_energy"]), ax2.set_title("mean free energy of data"), ax2.set_xlabel("epoch")
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(rbm.free_energy(V), bins=15, alpha=0.7, label="bars & stripes")
    ax3.hist(rbm.free_energy(noise_v), bins=15, alpha=0.7, label="random images")
    ax3.legend(), ax3.set_title("free energy: data sits in low-energy valleys")
    for i in range(6):
        v = V[i * 5].copy()
        corrupt = v.copy()
        flip = rng.choice(16, 2, replace=False)
        corrupt[flip] = 1 - corrupt[flip]
        rec = rbm.reconstruct(corrupt)
        ax = fig.add_subplot(2, 6, 7 + i)
        ax.imshow(np.hstack([v.reshape(4, 4), np.full((4, 1), 0.5), corrupt.reshape(4, 4),
                             np.full((4, 1), 0.5), rec.reshape(4, 4)]), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title("orig | noisy | recon", fontsize=8)
    fig.suptitle("RBM (16 visible, 12 hidden) trained with CD-1 on 4x4 bars and stripes")
    path = out_dir / "rbm_energy_model.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [path]
