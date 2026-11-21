"""Training loop shared by every NumPy experiment."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from smartattend.nnfund.mlp import MLP, softmax_cross_entropy
from smartattend.nnfund.optimizers import Optimizer


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int | None = 32      # None -> full-batch gradient descent
    l2: float = 0.0                  # weight decay on Dense weights only
    augment_noise: float = 0.0       # std of Gaussian feature jitter (augmentation)
    seed: int = 0
    log_every: int = 0               # 0 -> silent


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(X) == y))


def evaluate(model: MLP, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits = model.forward(X, train=False)
    loss, _ = softmax_cross_entropy(logits, y)
    return loss, float(np.mean(np.argmax(logits, axis=1) == y))


def fit(model: MLP, opt: Optimizer, X: np.ndarray, y: np.ndarray, cfg: TrainConfig,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
        logger: logging.Logger | None = None) -> dict:
    rng = np.random.default_rng(cfg.seed)
    n = len(X)
    bs = n if cfg.batch_size is None else cfg.batch_size
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "step_loss": [],
            "hyperparameters": {**asdict(cfg), **opt.hyperparameters(), **model.hyperparameters()},
            "n_parameters": model.parameter_count()}

    for epoch in range(cfg.epochs):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            xb = X[idx]
            if cfg.augment_noise > 0:
                xb = xb + rng.normal(scale=cfg.augment_noise, size=xb.shape)
            loss, grad = softmax_cross_entropy(model.forward(xb, train=True), y[idx])
            model.backward(grad)
            if cfg.l2 > 0:
                for layer, name in model.trainable():
                    if name == "W":
                        loss += 0.5 * cfg.l2 * float(np.sum(layer.params[name] ** 2))
                        layer.grads[name] += cfg.l2 * layer.params[name]
            opt.step(model)
            hist["step_loss"].append(loss)

        tl, ta = evaluate(model, X, y)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        if X_val is not None:
            vl, va = evaluate(model, X_val, y_val)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
        if logger and cfg.log_every and (epoch + 1) % cfg.log_every == 0:
            msg = f"epoch {epoch + 1:4d} train_loss={tl:.4f} train_acc={ta:.3f}"
            if X_val is not None:
                msg += f" val_loss={hist['val_loss'][-1]:.4f} val_acc={hist['val_acc'][-1]:.3f}"
            logger.info(msg)
    return hist
