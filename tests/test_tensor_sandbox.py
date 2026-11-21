import numpy as np
import pytest

from smartattend.nnfund import tensor_sandbox as ts


@pytest.mark.parametrize("a,b", [((3, 1), (1, 4)), ((5,), (2, 5)), ((2, 1, 3), (4, 1)), ((1,), (7, 7))])
def test_broadcast_shape_matches_numpy(a, b):
    assert ts.broadcast_shape(a, b) == np.broadcast_shapes(a, b)


def test_broadcast_shape_rejects_incompatible():
    with pytest.raises(ValueError):
        ts.broadcast_shape((3, 2), (4,))


def test_explicit_broadcast_add():
    a, b = np.arange(3.0).reshape(3, 1), np.arange(4.0)
    np.testing.assert_allclose(ts.explicit_broadcast_add(a, b), a + b)


def test_inner_product_and_matmul():
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=7), rng.normal(size=7)
    assert ts.inner_product(u, v) == pytest.approx(u @ v)
    A, B = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    np.testing.assert_allclose(ts.matmul_as_inner_products(A, B), A @ B)
    assert ts.cosine_similarity(u, 2 * u) == pytest.approx(1.0)


def test_reshape_orders():
    d = ts.reshape_demo()
    assert d["C_order"][0].tolist() == [0, 1, 2, 3]
    assert d["F_order"][:, 0].tolist() == [0, 1, 2]


def test_low_rank_error_decreases_and_full_rank_exact():
    A = np.random.default_rng(0).normal(size=(10, 8))
    errs = [np.linalg.norm(A - ts.low_rank_approx(A, k)) for k in range(1, 9)]
    assert all(e1 >= e2 for e1, e2 in zip(errs, errs[1:]))
    assert errs[-1] < 1e-10
    assert ts.svd_energy(A)[-1] == pytest.approx(1.0)


def test_gradient_check_detects_right_and_wrong_gradients():
    x = np.random.default_rng(0).normal(size=5)
    f = lambda x: float(np.sum(np.sin(x) * x**2))
    good = lambda x: np.cos(x) * x**2 + 2 * x * np.sin(x)
    bad = lambda x: np.cos(x) * x**2
    assert ts.gradient_check(f, good, x) < 1e-7
    assert ts.gradient_check(f, bad, x) > 1e-2


def test_demo_numerical_gradient_matches_analytic():
    lines = ts.demo()
    analytic = next(l for l in lines if l.strip().startswith("analytic"))
    numerical = next(l for l in lines if l.strip().startswith("numerical"))
    assert analytic.split("=")[1].strip() == numerical.split("=")[1].strip()
    assert any("not broadcastable" in l for l in lines)
