"""Requirement 4: McCulloch-Pitts, sigmoid neuron, worked backprop example, framework agreement."""
import numpy as np
import pytest

from smartattend.nnfund import neural_networks as nn
from smartattend.nnfund.neurons import BOOL_INPUTS, TRUTH_TABLES


def test_two_layer_mcculloch_pitts_computes_xor():
    assert nn.mp_xor(BOOL_INPUTS).tolist() == TRUTH_TABLES["XOR"].tolist()
    assert nn.NAND(BOOL_INPUTS).tolist() == [1, 1, 1, 0]


def test_sigmoid_neuron_demo_reports_xor_failure():
    lines = nn.demo_sigmoid_neuron()
    acc = {l.split(":")[0].strip(): float(l.split("accuracy=")[1].split()[0]) for l in lines if "accuracy=" in l}
    assert acc["AND"] == 1.0 and acc["OR"] == 1.0 and acc["XOR"] < 1.0


def test_worked_example_matches_hand_calculation():
    m = nn.tiny_network()
    x = np.array([[1.0, 2.0]])
    z1 = m.layers[0].forward(x)
    np.testing.assert_allclose(z1, [[1.0, 1.2]])          # 1*0.5 + 2*0.2 + 0.1 ,  1*-0.3 + 2*0.8 - 0.1
    assert any("correct" in l for l in nn.demo_worked_example())


def test_balanced_indices_equalise_classes():
    y = np.array([0] * 50 + [1] * 10 + [2] * 3)
    idx = nn.balanced_indices(y, np.random.default_rng(0))
    assert np.bincount(y[idx]).tolist() == [50, 50, 50]


def test_standardize_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    data = {"train": (rng.normal(5, 2, size=(200, 3)), np.zeros(200, int)),
            "val": (rng.normal(50, 9, size=(20, 3)), np.zeros(20, int)),
            "test": (rng.normal(-9, 1, size=(20, 3)), np.zeros(20, int))}
    s = nn.standardize(data)
    np.testing.assert_allclose(s["train"][0].mean(axis=0), 0, atol=1e-9)
    assert s["val"][0].mean() > 10                          # val was NOT re-centred on its own mean


def test_numpy_pytorch_tensorflow_gradients_agree():
    pytest.importorskip("torch")
    r = nn.framework_gradients()
    assert r["pytorch"]["loss"] == pytest.approx(r["numpy"]["loss"], abs=1e-9)   # our loss adds 1e-12 inside log() to avoid log(0)
    assert r["pytorch"]["max_grad_diff"] < 1e-12
    if "skipped" not in r["tensorflow"]:
        assert r["tensorflow"]["loss"] == pytest.approx(r["numpy"]["loss"], abs=1e-9)   # our loss adds 1e-12 inside log() to avoid log(0)
        assert r["tensorflow"]["max_grad_diff"] < 1e-12


def test_numpy_mlp_learns_separable_features():
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=4, size=(3, 10))
    make = lambda n: (lambda y: (centres[y] + rng.normal(size=(n, 10)), y))(rng.integers(0, 3, n))
    data = nn.standardize({"train": make(300), "val": make(90), "test": make(90)})
    model, hist = nn.train_numpy_mlp(data, hidden=(16,), epochs=10, dropout=0.0)
    assert np.mean(model.predict(data["test"][0]) == data["test"][1]) > 0.9
    assert 1 <= hist["best_epoch"] <= 10
