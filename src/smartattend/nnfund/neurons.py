"""McCulloch-Pitts and sigmoid neurons, plus activation functions with derivatives."""
from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- activations
def step(z: np.ndarray) -> np.ndarray:
    return (np.asarray(z) >= 0).astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # 0.5 * (1 + tanh(z/2)) is algebraically sigmoid but never overflows.
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


# name -> (f(z), f'(z) expressed via z and a = f(z))
ACTIVATIONS = {
    "step": (step, lambda z, a: np.zeros_like(z)),
    "sigmoid": (sigmoid, lambda z, a: a * (1.0 - a)),
    "tanh": (tanh, lambda z, a: 1.0 - a**2),
    "relu": (relu, lambda z, a: (z > 0).astype(z.dtype)),
    "identity": (lambda z: z, lambda z, a: np.ones_like(z)),
}


# --------------------------------------------------------------------------- McCulloch-Pitts
class McCullochPitts:
    """Binary threshold unit: fires (1) iff sum_i w_i x_i >= threshold. Weights are fixed, not learned."""

    def __init__(self, weights: list[float], threshold: float):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.threshold = float(threshold)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(x) @ self.weights >= self.threshold).astype(int)


MP_GATES = {
    "AND": McCullochPitts([1, 1], 2),
    "OR": McCullochPitts([1, 1], 1),
    "NOT": McCullochPitts([-1], 0),
    "NOR": McCullochPitts([-1, -1], 0),
}


# --------------------------------------------------------------------------- sigmoid neuron
class SigmoidNeuron:
    """Single logistic unit trained by gradient descent on binary cross-entropy."""

    def __init__(self, n_inputs: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(scale=0.1, size=n_inputs)
        self.b = 0.0
        self.loss_history: list[float] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.5, epochs: int = 2000) -> "SigmoidNeuron":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        for _ in range(epochs):
            p = self.predict_proba(X)
            self.loss_history.append(float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))))
            # dL/dz = p - y for sigmoid + BCE
            err = p - y
            self.w -= lr * X.T @ err / len(y)
            self.b -= lr * float(np.mean(err))
        return self


BOOL_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
TRUTH_TABLES = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


def make_figures(out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smartattend.nnfund.mlp import MLP
    from smartattend.nnfund.optimizers import Momentum
    from smartattend.nnfund.trainer import TrainConfig, fit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Step vs sigmoid vs tanh and their derivatives
    z = np.linspace(-6, 6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name in ["step", "sigmoid", "tanh"]:
        f, df = ACTIVATIONS[name]
        a = f(z)
        axes[0].plot(z, a, label=name)
        axes[1].plot(z, df(z, a), label=name)
    axes[0].set_title("activation f(z)"), axes[1].set_title("derivative f'(z)")
    axes[1].text(-5.5, 0.85, "step: f'=0 almost everywhere\n-> no gradient signal", fontsize=8)
    for ax in axes:
        ax.legend(), ax.grid(alpha=0.3)
    paths.append(out_dir / "activations_step_sigmoid_tanh.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Decision boundaries: sigmoid neuron on AND/OR/XOR, MLP on XOR
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, gate in zip(axes[:3], ["AND", "OR", "XOR"]):
        neuron = SigmoidNeuron(2).fit(BOOL_INPUTS, TRUTH_TABLES[gate])
        acc = np.mean(neuron.predict(BOOL_INPUTS) == TRUTH_TABLES[gate])
        ax.contourf(gx, gy, neuron.predict_proba(grid).reshape(gx.shape), levels=20, cmap="RdBu_r", alpha=0.7)
        ax.scatter(*BOOL_INPUTS.T, c=TRUTH_TABLES[gate], cmap="RdBu_r", edgecolors="k", s=80)
        ax.set_title(f"sigmoid neuron: {gate} (acc={acc:.2f})")
    mlp = MLP.build([2, 8, 2], activation="tanh", init="xavier", seed=1)
    fit(mlp, Momentum(lr=0.2, mu=0.9), BOOL_INPUTS, TRUTH_TABLES["XOR"], TrainConfig(epochs=500, batch_size=None))
    probs = mlp.predict_proba(grid)[:, 1]
    acc = np.mean(mlp.predict(BOOL_INPUTS) == TRUTH_TABLES["XOR"])
    axes[3].contourf(gx, gy, probs.reshape(gx.shape), levels=20, cmap="RdBu_r", alpha=0.7)
    axes[3].scatter(*BOOL_INPUTS.T, c=TRUTH_TABLES["XOR"], cmap="RdBu_r", edgecolors="k", s=80)
    axes[3].set_title(f"MLP 2-8-2: XOR (acc={acc:.2f})")
    paths.append(out_dir / "neuron_decision_boundaries.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths
