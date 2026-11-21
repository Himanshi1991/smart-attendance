"""Gradient-descent optimizers. Batch GD vs SGD is decided by the trainer's batch size;
the update rules below decide how a gradient becomes a parameter change."""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, lr: float):
        self.lr = lr
        self.state: dict[tuple[int, str], np.ndarray] = {}

    def step(self, model) -> None:
        for layer, name in model.trainable():
            self._update(layer.params[name], layer.grads[name], (id(layer), name))

    def _update(self, p: np.ndarray, g: np.ndarray, key) -> None:
        raise NotImplementedError

    def hyperparameters(self) -> dict:
        return {"optimizer": type(self).__name__, "lr": self.lr}


class SGD(Optimizer):
    """p <- p - lr * g"""

    def _update(self, p, g, key):
        p -= self.lr * g


class Momentum(Optimizer):
    """Heavy ball: v <- mu v - lr g ;  p <- p + v"""

    def __init__(self, lr: float, mu: float = 0.9):
        super().__init__(lr)
        self.mu = mu

    def _update(self, p, g, key):
        v = self.state.setdefault(key, np.zeros_like(p))
        v *= self.mu
        v -= self.lr * g
        p += v

    def hyperparameters(self):
        return {**super().hyperparameters(), "mu": self.mu}


class Nesterov(Momentum):
    """Nesterov accelerated gradient in the 'look-ahead' form that only needs g at p:
    v_new = mu v - lr g ;  p <- p - mu v_old + (1 + mu) v_new"""

    def _update(self, p, g, key):
        v = self.state.setdefault(key, np.zeros_like(p))
        v_old = v.copy()
        v *= self.mu
        v -= self.lr * g
        p += -self.mu * v_old + (1 + self.mu) * v


OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "nesterov": Nesterov}
