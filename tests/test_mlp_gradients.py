"""Automated gradient tests: analytic backprop must match central differences for every parameter."""
import numpy as np
import pytest

from smartattend.nnfund.mlp import MLP, mse_loss, softmax_cross_entropy
from smartattend.nnfund.tensor_sandbox import numerical_gradient, relative_error


def _check(model, X, y, loss_fn=softmax_cross_entropy, tol=1e-6):
    loss, grad = loss_fn(model.forward(X, train=True), y)
    dX = model.backward(grad)
    f = lambda: loss_fn(model.forward(X, train=True), y)[0]
    for layer, name in model.trainable():
        analytic = layer.grads[name].copy()
        numeric = numerical_gradient(f, layer.params[name])
        assert relative_error(analytic, numeric) < tol, f"{type(layer).__name__}.{name}"
    assert relative_error(dX, numerical_gradient(f, X)) < tol, "input gradient"


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
@pytest.mark.parametrize("batchnorm", [False, True])
def test_mlp_gradients(activation, batchnorm):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 4))
    y = rng.integers(0, 3, size=6)
    model = MLP.build([4, 5, 3], activation=activation, init="xavier", batchnorm=batchnorm, seed=7)
    _check(model, X, y)


def test_mse_gradients():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(5, 3)), rng.normal(size=(5, 2))
    _check(MLP.build([3, 4, 2], activation="tanh", seed=1), X, Y, loss_fn=mse_loss)


def test_softmax_cross_entropy_is_stable():
    loss, grad = softmax_cross_entropy(np.array([[1000.0, 0.0, -1000.0]]), np.array([0]))
    assert np.isfinite(loss) and np.all(np.isfinite(grad))


def test_parameter_vs_hyperparameter_tracking():
    m = MLP.build([8, 16, 3], dropout=0.2)
    s = m.summary()
    assert s["parameters"]["total"] == 8 * 16 + 16 + 16 * 3 + 3
    assert s["hyperparameters"]["dropout"] == 0.2 and s["hyperparameters"]["sizes"] == [8, 16, 3]


def test_dropout_is_identity_in_eval():
    m = MLP.build([4, 32, 2], dropout=0.5, seed=0)
    X = np.random.default_rng(0).normal(size=(3, 4))
    np.testing.assert_allclose(m.forward(X, train=False), m.forward(X, train=False))
