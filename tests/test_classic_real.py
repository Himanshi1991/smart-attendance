"""Requirement 6 on real-data code paths: Sanger, competitive restarts, ARI/purity, perceptron, delta rule, RBM bias init."""
import numpy as np
import pytest

from smartattend.nnfund import classic_real as cr
from smartattend.nnfund.boltzmann import RBM


def _cloud(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(500, 5)) * np.array([3.0, 2.0, 1.0, 0.5, 0.2])     # clear, distinct variances
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    X = X @ Q.T
    return (X - X.mean(0)) / np.sqrt(np.mean(np.sum((X - X.mean(0)) ** 2, 1)))


def test_sanger_matches_svd_components():
    X = _cloud()
    W = cr.sanger(X, k=3, lr=0.1, epochs=40)
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    for i in range(3):
        assert abs(W[i] @ Vt[i]) / np.linalg.norm(W[i]) > 0.98
        assert abs(np.linalg.norm(W[i]) - 1) < 0.05


def test_plain_hebb_explodes_but_oja_stays_unit():
    X = _cloud()
    _, hn = cr.hebbian_run(X, lr=0.05, epochs=30)
    _, on = cr.oja_run(X, lr=0.01, epochs=10)
    assert hn[-1] > 1e6 and abs(on[-1] - 1) < 0.05


def test_ari_and_purity_against_sklearn():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 4, 200), rng.integers(0, 5, 200)
    assert cr.adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)
    assert cr.adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert cr.purity(a, a) == 1.0


def test_competitive_learning_separates_clear_identities():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(5, 16))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    X = np.repeat(centres, 30, axis=0) + 0.05 * rng.normal(size=(150, 16))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat(np.arange(5), 30)
    W, fits = cr.competitive_restarts(X, k=5, restarts=5)
    assert len(fits) == 5 and cr.adjusted_rand_index(y, np.argmax(X @ W.T, axis=1)) > 0.9


def test_perceptron_and_delta_rule_learn_separable_classes():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    X = rng.normal(size=(300, 6)) + 4 * np.eye(3, 6)[y]
    Wp, mistakes = cr.perceptron(X, y, 3, epochs=10)
    Wd, mse = cr.delta_rule_multi(X, y, 3, epochs=10, lr=1e-3)
    assert mistakes[-1] < mistakes[0] and mse[-1] < mse[0]
    assert np.mean(cr.predict_linear(Wp, X) == y) > 0.95 and np.mean(cr.predict_linear(Wd, X) == y) > 0.95


def test_rbm_visible_bias_init_is_log_odds():
    V = np.tile([0.1, 0.5, 0.9], (20, 1))
    rbm = RBM(3, 2).init_visible_bias(V)
    np.testing.assert_allclose(1 / (1 + np.exp(-rbm.a)), [0.1, 0.5, 0.9], atol=1e-9)
