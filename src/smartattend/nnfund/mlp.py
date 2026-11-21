"""Multilayer feed-forward network with explicit forward/backward passes (NumPy only).

Each layer keeps ``params`` (learned by the optimizer) and ``grads`` (filled by ``backward``).
Anything chosen by hand - layer sizes, activation, init scheme, dropout rate - is a
hyperparameter and is reported separately by ``MLP.hyperparameters()``.
"""
from __future__ import annotations

import numpy as np

from smartattend.nnfund.neurons import ACTIVATIONS


class Layer:
    params: dict[str, np.ndarray]
    grads: dict[str, np.ndarray]

    def __init__(self) -> None:
        self.params, self.grads = {}, {}

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"type": type(self).__name__}


class Dense(Layer):
    INITS = ("he", "xavier", "normal", "zeros")

    def __init__(self, n_in: int, n_out: int, init: str = "he", rng: np.random.Generator | None = None):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        std = {"he": np.sqrt(2.0 / n_in), "xavier": np.sqrt(2.0 / (n_in + n_out)),
               "normal": 0.01, "zeros": 0.0}[init]
        self.init = init
        self.params = {"W": rng.standard_normal((n_in, n_out)) * std, "b": np.zeros(n_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x, train=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):
        self.grads["W"][...] = self.x.T @ grad_out
        self.grads["b"][...] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T

    def describe(self):
        n_in, n_out = self.params["W"].shape
        return {"type": "Dense", "in": n_in, "out": n_out, "init": self.init}


class Activation(Layer):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.f, self.df = ACTIVATIONS[name]

    def forward(self, x, train=True):
        self.z = x
        self.a = self.f(x)
        return self.a

    def backward(self, grad_out):
        return grad_out * self.df(self.z, self.a)

    def describe(self):
        return {"type": "Activation", "fn": self.name}


class Dropout(Layer):
    """Inverted dropout: scale kept units by 1/(1-p) at train time so eval needs no rescaling."""

    def __init__(self, p: float, rng: np.random.Generator | None = None):
        super().__init__()
        self.p = p
        self.rng = rng or np.random.default_rng(0)

    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out if self.mask is None else grad_out * self.mask

    def describe(self):
        return {"type": "Dropout", "p": self.p}


class BatchNorm(Layer):
    """Batch normalisation over the batch axis, with running statistics for evaluation."""

    def __init__(self, n: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.params = {"gamma": np.ones(n), "beta": np.zeros(n)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.running_mean, self.running_var = np.zeros(n), np.ones(n)

    def forward(self, x, train=True):
        if train:
            mean, var = x.mean(axis=0), x.var(axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean, var = self.running_mean, self.running_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mean) * self.inv_std
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, grad_out):
        n = grad_out.shape[0]
        self.grads["gamma"][...] = np.sum(grad_out * self.xhat, axis=0)
        self.grads["beta"][...] = grad_out.sum(axis=0)
        dxhat = grad_out * self.params["gamma"]
        return (self.inv_std / n) * (n * dxhat - dxhat.sum(axis=0) - self.xhat * np.sum(dxhat * self.xhat, axis=0))

    def describe(self):
        return {"type": "BatchNorm", "n": self.params["gamma"].size}


# --------------------------------------------------------------------------- losses
def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and its gradient w.r.t. the logits ((p - onehot) / N)."""
    n = logits.shape[0]
    p = softmax(logits)
    loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
    grad = p.copy()
    grad[np.arange(n), y] -= 1.0
    return float(loss), grad / n


def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    n = pred.shape[0]
    diff = pred - target
    return float(0.5 * np.sum(diff**2) / n), diff / n


# --------------------------------------------------------------------------- network
class MLP:
    def __init__(self, layers: list[Layer], config: dict | None = None):
        self.layers = layers
        self.config = config or {}

    @classmethod
    def build(cls, sizes: list[int], activation: str = "relu", init: str = "he",
              batchnorm: bool = False, dropout: float = 0.0, seed: int = 0) -> "MLP":
        """sizes = [n_in, hidden..., n_out]; the last layer outputs logits (no activation)."""
        rng = np.random.default_rng(seed)
        layers: list[Layer] = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(Dense(n_in, n_out, init=init, rng=rng))
            if i < len(sizes) - 2:
                if batchnorm:
                    layers.append(BatchNorm(n_out))
                layers.append(Activation(activation))
                if dropout > 0:
                    layers.append(Dropout(dropout, rng=rng))
        config = dict(sizes=list(sizes), activation=activation, init=init, batchnorm=batchnorm,
                      dropout=dropout, seed=seed)
        return cls(layers, config)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x, train=False))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x, train=False), axis=1)

    def trainable(self):
        """Yield (layer, name) for every learned parameter."""
        for layer in self.layers:
            for name in layer.params:
                yield layer, name

    def parameter_count(self) -> int:
        return int(sum(layer.params[name].size for layer, name in self.trainable()))

    def hyperparameters(self) -> dict:
        return dict(self.config)

    def summary(self) -> dict:
        """Parameters (learned) versus hyperparameters (chosen), for experiment logs."""
        return {
            "parameters": {
                "total": self.parameter_count(),
                "per_layer": [{**layer.describe(), "params": {k: list(v.shape) for k, v in layer.params.items()}}
                              for layer in self.layers],
            },
            "hyperparameters": self.hyperparameters(),
        }
