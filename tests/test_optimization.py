"""Requirement 5: optimizers, normalisation, augmentation and ablation plumbing (small synthetic images, fast)."""
import numpy as np
import pytest

from smartattend.nnfund import optimization as opt


def _toy_faces(n=240, seed=0):
    """32x32 'images': class 0 bright top half, class 1 bright bottom half, class 2 bright left half, plus noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.uniform(0, 0.3, size=(n, 32, 32))
    for i, c in enumerate(y):
        if c == 0:
            X[i, :16] += 0.6
        elif c == 1:
            X[i, 16:] += 0.6
        else:
            X[i, :, :16] += 0.6
    X[:, 0, 0] = 0.0                                     # a constant pixel, like the black corners of real frames
    return np.clip(X, 0, 1), y


@pytest.fixture(scope="module")
def data():
    Xtr, ytr = _toy_faces(240, 0)
    Xva, yva = _toy_faces(60, 1)
    Xte, yte = _toy_faces(60, 2)
    return {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}


def test_normalizer_uses_train_stats_and_floors_std(data):
    n = opt.Normalizer().fit(data["train"][0])
    assert n.std.min() >= 0.05                           # constant pixel no longer divides by ~0
    shifted = np.roll(data["train"][0][:5], 1, axis=2)
    assert np.abs(n(shifted)).max() < 100                # the bug this floor fixed produced values around 1e5


@pytest.mark.parametrize("kind", ["none", "flip", "shift", "brightness", "flip+shift+brightness"])
def test_augment_keeps_shape_and_range(kind):
    X = np.random.default_rng(0).uniform(size=(8, 32, 32))
    A = opt.augment(X, kind, np.random.default_rng(1))
    assert A.shape == X.shape and A.min() >= 0 and A.max() <= 1
    if kind == "none":
        assert A is X


def test_flip_mirrors_left_right():
    X = np.zeros((64, 32, 32))
    X[:, :, :8] = 1.0
    A = opt.augment(X, "flip", np.random.default_rng(0))
    flipped = A[:, :, -8:].mean(axis=(1, 2)) == 1.0
    assert 0 < flipped.sum() < 64                         # about half the batch is mirrored


@pytest.mark.parametrize("rule", ["sgd", "momentum", "nesterov"])
def test_every_optimizer_learns(data, rule):
    h = opt.train(data, optimizer=rule, lr=0.01, batch=32, epochs=5, seed=0)
    assert h["train_loss"][-1] < h["train_loss"][0] and h["test_f1_last"] > 0.9


def test_batch_gd_makes_one_update_per_epoch(data):
    h = opt.train(data, optimizer="sgd", lr=0.1, batch=None, epochs=3)
    assert h["updates"] == [1, 2, 3]


def test_zero_init_cannot_break_symmetry(data):
    h = opt.train(data, init="zeros", epochs=5)
    assert h["train_f1"][-1] < 0.5                        # every hidden unit stays identical (and ReLU'(0)=0)


def test_momentum_beats_plain_gd_on_narrow_valley():
    final = {k.split(" (")[0]: 0.5 * (p[-1, 0] ** 2 + 25 * p[-1, 1] ** 2) for k, p in opt.bowl_paths().items()}
    assert final["Momentum"] < final["Gradient descent"] and final["Nesterov"] < final["Gradient descent"]
