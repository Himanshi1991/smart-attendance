import numpy as np

from smartattend.data.synthetic import Standardizer, leakage_safe_split, make_engagement_features
from smartattend.nnfund.mlp import MLP
from smartattend.nnfund.neurons import BOOL_INPUTS, MP_GATES, TRUTH_TABLES, SigmoidNeuron
from smartattend.nnfund.optimizers import SGD, Momentum, Nesterov
from smartattend.nnfund.trainer import TrainConfig, evaluate, fit


def test_mcculloch_pitts_gates():
    assert MP_GATES["AND"](BOOL_INPUTS).tolist() == [0, 0, 0, 1]
    assert MP_GATES["OR"](BOOL_INPUTS).tolist() == [0, 1, 1, 1]
    assert MP_GATES["NOR"](BOOL_INPUTS).tolist() == [1, 0, 0, 0]
    assert MP_GATES["NOT"](np.array([[0], [1]])).tolist() == [1, 0]


def test_sigmoid_neuron_linear_vs_xor():
    for gate in ["AND", "OR"]:
        n = SigmoidNeuron(2).fit(BOOL_INPUTS, TRUTH_TABLES[gate])
        assert (n.predict(BOOL_INPUTS) == TRUTH_TABLES[gate]).all()
        assert n.loss_history[-1] < n.loss_history[0]
    xor = SigmoidNeuron(2).fit(BOOL_INPUTS, TRUTH_TABLES["XOR"])
    assert (xor.predict(BOOL_INPUTS) == TRUTH_TABLES["XOR"]).mean() < 1.0  # not linearly separable


def test_mlp_solves_xor():
    m = MLP.build([2, 8, 2], activation="tanh", init="xavier", seed=1)
    fit(m, Momentum(lr=0.2, mu=0.9), BOOL_INPUTS, TRUTH_TABLES["XOR"], TrainConfig(epochs=500, batch_size=None))
    assert (m.predict(BOOL_INPUTS) == TRUTH_TABLES["XOR"]).all()


class _Quadratic:
    """Fake 'model' with one parameter; loss 0.5 x^T A x with condition number 50."""

    def __init__(self):
        self.A = np.diag([1.0, 50.0])
        self.params = {"x": np.array([5.0, 1.0])}
        self.grads = {"x": np.zeros(2)}

    def trainable(self):
        yield self, "x"

    def loss(self):
        x = self.params["x"]
        return 0.5 * x @ self.A @ x

    def compute_grad(self):
        self.grads["x"][...] = self.A @ self.params["x"]


def _run(opt, steps=100):
    q = _Quadratic()
    for _ in range(steps):
        q.compute_grad()
        opt.step(q)
    return q.loss()


def test_momentum_and_nesterov_beat_sgd_on_ill_conditioned_quadratic():
    sgd, mom, nag = _run(SGD(lr=0.019)), _run(Momentum(lr=0.019, mu=0.9)), _run(Nesterov(lr=0.019, mu=0.9))
    assert mom < sgd and nag < sgd


def test_engagement_mlp_learns():
    X, y, _ = make_engagement_features(n=900, seed=0)
    tr, va, _ = leakage_safe_split(len(X), seed=0)
    sc = Standardizer().fit(X[tr])
    m = MLP.build([8, 32, 16, 3], seed=0)
    fit(m, Momentum(lr=0.01, mu=0.9), sc.transform(X[tr]), y[tr], TrainConfig(epochs=40, batch_size=32))
    _, acc = evaluate(m, sc.transform(X[va]), y[va])
    assert acc > 0.6  # chance = 0.33


def test_split_is_disjoint_and_standardizer_uses_train_only():
    tr, va, te = leakage_safe_split(1000, seed=0)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == 1000
    X = np.arange(20.0).reshape(10, 2)
    sc = Standardizer().fit(X[:5])
    np.testing.assert_allclose(sc.mean, X[:5].mean(axis=0))
