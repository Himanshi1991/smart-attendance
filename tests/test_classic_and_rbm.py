import itertools

import numpy as np

from smartattend.nnfund.boltzmann import RBM, bars_and_stripes
from smartattend.nnfund.classic_learning import competitive, delta_rule, hebbian, oja


def _cloud(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal([0, 0], [[3.0, 1.2], [1.2, 1.0]], size=400)
    return X - X.mean(axis=0)


def test_oja_converges_to_first_principal_component():
    X = _cloud()
    w, norms = oja(X, lr=0.002, epochs=10)
    pc1 = np.linalg.eigh(np.cov(X.T))[1][:, -1]
    assert abs(w @ pc1) / np.linalg.norm(w) > 0.99
    assert abs(norms[-1] - 1.0) < 0.05


def test_hebbian_weight_norm_grows():
    _, norms = hebbian(_cloud(), lr=0.002, epochs=10)
    assert norms[-1] > 100 * norms[0]


def test_competitive_learning_finds_cluster_centres():
    rng = np.random.default_rng(0)
    centers = np.array([[-3, -2], [3, -1], [0, 3]])
    X = np.vstack([c + 0.5 * rng.normal(size=(100, 2)) for c in centers])
    W = competitive(X, k=3)
    for c in centers:
        assert np.min(np.linalg.norm(W - c, axis=1)) < 0.5


def test_delta_rule_recovers_linear_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    t = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5
    w, mse = delta_rule(X, t, epochs=30)
    np.testing.assert_allclose(w, [1.5, -2.0, 0.5], atol=1e-3)
    assert mse[-1] < mse[0]


def test_bars_and_stripes_count():
    assert len(bars_and_stripes(4)) == 2 * 2**4 - 2  # all-off and all-on are both bars and stripes


def test_rbm_free_energy_matches_brute_force():
    rbm = RBM(3, 2, seed=0)
    rbm.W = np.random.default_rng(1).normal(size=(3, 2))
    rbm.a, rbm.b = np.array([0.1, -0.2, 0.3]), np.array([0.5, -0.5])
    v = np.array([1.0, 0.0, 1.0])
    hs = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
    brute = -np.log(np.sum(np.exp(-rbm.energy(np.tile(v, (4, 1)), hs))))
    assert abs(rbm.free_energy(v) - brute) < 1e-10


def test_rbm_training_reduces_reconstruction_error():
    V = bars_and_stripes(4)
    rbm = RBM(16, 12, seed=0)
    hist = rbm.fit(V, epochs=300, lr=0.3)
    assert hist["recon_error"][-1] < 0.1 * hist["recon_error"][0]
    # training data should sit in lower-energy regions than random images
    noise = (np.random.default_rng(1).random((len(V), 16)) < 0.5).astype(float)
    assert rbm.free_energy(V).mean() < rbm.free_energy(noise).mean()
