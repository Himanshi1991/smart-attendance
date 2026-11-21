"""Classic learning rules: Hebbian, modified Hebbian (Oja), competitive, error-correction (delta)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def hebbian(X: np.ndarray, lr: float = 0.01, epochs: int = 20, seed: int = 0):
    """Plain Hebb: dw = lr * y * x with y = w.x. The weight norm grows without bound."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.1, size=X.shape[1])
    norms = []
    for _ in range(epochs):
        for x in X[rng.permutation(len(X))]:
            y = w @ x
            w = w + lr * y * x
        norms.append(float(np.linalg.norm(w)))
    return w, np.array(norms)


def oja(X: np.ndarray, lr: float = 0.01, epochs: int = 20, seed: int = 0):
    """Oja's rule (modified Hebbian): dw = lr * y * (x - y w). Converges to the unit-norm
    first principal component of zero-mean X."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.1, size=X.shape[1])
    norms = []
    for _ in range(epochs):
        for x in X[rng.permutation(len(X))]:
            y = w @ x
            w = w + lr * y * (x - y * w)
        norms.append(float(np.linalg.norm(w)))
    return w, np.array(norms)


def competitive(X: np.ndarray, k: int, lr: float = 0.05, epochs: int = 20, seed: int = 0):
    """Winner-take-all: only the closest prototype moves towards each sample."""
    rng = np.random.default_rng(seed)
    W = X[rng.choice(len(X), size=k, replace=False)].astype(np.float64).copy()
    for _ in range(epochs):
        for x in X[rng.permutation(len(X))]:
            winner = np.argmin(np.sum((W - x) ** 2, axis=1))
            W[winner] += lr * (x - W[winner])
    return W


def delta_rule(X: np.ndarray, t: np.ndarray, lr: float = 0.01, epochs: int = 50, seed: int = 0):
    """Error-correction (Widrow-Hoff / LMS): dw = lr * (t - y) * x for a linear unit."""
    rng = np.random.default_rng(seed)
    Xb = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(Xb.shape[1])
    mse = []
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            w += lr * (t[i] - Xb[i] @ w) * Xb[i]
        mse.append(float(np.mean((Xb @ w - t) ** 2)))
    return w, np.array(mse)


def make_figures(out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Correlated 2-D cloud for Hebb vs Oja
    C = np.array([[3.0, 1.2], [1.2, 1.0]])
    X = rng.multivariate_normal([0, 0], C, size=400)
    X -= X.mean(axis=0)
    pc1 = np.linalg.eigh(np.cov(X.T))[1][:, -1]
    w_hebb, n_hebb = hebbian(X, lr=0.002, epochs=10)
    w_oja, n_oja = oja(X, lr=0.002, epochs=10)

    # Clusters for competitive learning
    centers = np.array([[-3, -2], [3, -1], [0, 3]])
    Xc = np.vstack([c + 0.6 * rng.normal(size=(150, 2)) for c in centers])
    W = competitive(Xc, k=3)

    # Linear target for delta rule
    Xd = rng.normal(size=(200, 2))
    td = 1.5 * Xd[:, 0] - 2.0 * Xd[:, 1] + 0.5 + 0.1 * rng.normal(size=200)
    _, mse = delta_rule(Xd, td)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    ax = axes[0]
    ax.scatter(*X.T, s=5, alpha=0.4)
    for vec, c, lbl in [(pc1 * 3, "k", "PCA PC1"), (w_oja / np.linalg.norm(w_oja) * 3, "C1", "Oja w")]:
        ax.plot([0, vec[0]], [0, vec[1]], color=c, lw=2, label=lbl)
    ax.set_aspect("equal"), ax.legend(), ax.set_title("Oja's rule finds the 1st principal component")
    axes[1].semilogy(n_hebb, label="Hebbian |w|")
    axes[1].semilogy(n_oja, label="Oja |w|")
    axes[1].set_xlabel("epoch"), axes[1].legend(), axes[1].set_title("Hebb diverges, Oja stays at |w|=1")
    axes[2].scatter(*Xc.T, s=5, alpha=0.4)
    axes[2].scatter(*W.T, c="red", marker="X", s=200, label="prototypes")
    axes[2].legend(), axes[2].set_title("Competitive (winner-take-all) learning")
    axes[3].plot(mse)
    axes[3].set_xlabel("epoch"), axes[3].set_ylabel("MSE"), axes[3].set_yscale("log")
    axes[3].set_title("Delta rule (error-correction)")
    for a in axes:
        a.grid(alpha=0.3)
    path = out_dir / "classic_learning_rules.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [path]
