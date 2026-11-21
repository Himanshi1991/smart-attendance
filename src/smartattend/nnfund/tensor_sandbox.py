"""Tensor maths sandbox: broadcasting, inner products, reshaping, SVD and numerical gradient checking.

Every function is written so the computation is visible, and ``make_figures`` renders plots
that explain what each operation does.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- broadcasting
def broadcast_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    """Apply NumPy's broadcasting rule by hand: align from the right, sizes must match or be 1."""
    ndim = max(len(shape_a), len(shape_b))
    a = (1,) * (ndim - len(shape_a)) + tuple(shape_a)
    b = (1,) * (ndim - len(shape_b)) + tuple(shape_b)
    out = []
    for da, db in zip(a, b):
        if da != db and 1 not in (da, db):
            raise ValueError(f"shapes {shape_a} and {shape_b} are not broadcastable")
        out.append(max(da, db))
    return tuple(out)


def explicit_broadcast_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a + b computed by explicitly tiling both operands to the broadcast shape."""
    shape = broadcast_shape(a.shape, b.shape)
    return np.broadcast_to(a, shape).copy() + np.broadcast_to(b, shape).copy()


# --------------------------------------------------------------------------- inner products
def inner_product(u: np.ndarray, v: np.ndarray) -> float:
    """<u, v> = sum_i u_i v_i, written as an explicit loop."""
    u, v = np.ravel(u), np.ravel(v)
    if u.shape != v.shape:
        raise ValueError("inner product needs equal-length vectors")
    total = 0.0
    for ui, vi in zip(u, v):
        total += float(ui) * float(vi)
    return total


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return inner_product(u, v) / (np.sqrt(inner_product(u, u) * inner_product(v, v)) + 1e-12)


def matmul_as_inner_products(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C[i, j] = <row i of A, column j of B>."""
    C = np.empty((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = inner_product(A[i], B[:, j])
    return C


# --------------------------------------------------------------------------- reshaping
def reshape_demo(n: int = 12, shape: tuple[int, ...] = (3, 4)) -> dict[str, np.ndarray]:
    """Show that reshape re-reads the same flat buffer in row-major (C) or column-major (F) order."""
    x = np.arange(n)
    return {
        "flat": x,
        "C_order": x.reshape(shape, order="C"),
        "F_order": x.reshape(shape, order="F"),
        "transpose_then_flatten": x.reshape(shape).T.ravel(),
    }


# --------------------------------------------------------------------------- SVD
def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation (Eckart-Young): keep the k largest singular values."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def svd_energy(A: np.ndarray) -> np.ndarray:
    """Cumulative fraction of squared Frobenius norm captured by the top-k singular values."""
    S = np.linalg.svd(A, compute_uv=False)
    return np.cumsum(S**2) / np.sum(S**2)


# --------------------------------------------------------------------------- gradient checking
def numerical_gradient(f: Callable[[], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of scalar ``f()`` w.r.t. array ``x``.

    ``x`` is perturbed in place (and restored), so ``f`` should read ``x`` from its closure.
    This lets the same routine check gradients of any network parameter.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max(|a| + |b|, floor): ~1e-7 or less means the analytic gradient is right.

    The 1e-8 floor stops round-off noise from looking like a 100% error when the true gradient
    is exactly zero (e.g. a bias feeding straight into batch norm).
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)))


def gradient_check(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray],
                   x: np.ndarray, eps: float = 1e-5) -> float:
    """Compare an analytic gradient against central differences at point ``x``."""
    x = np.array(x, dtype=np.float64)
    numeric = numerical_gradient(lambda: f(x), x, eps)
    return relative_error(grad_f(x), numeric)


# --------------------------------------------------------------------------- printed demo
def demo() -> list[str]:
    """Small worked examples whose printed results are the 'expected output' in the guide."""
    np.set_printoptions(precision=3, suppress=True)
    out = []
    p = out.append

    p("1. TENSORS: shape = size of each axis")
    t = np.arange(24).reshape(2, 3, 4)
    p(f"   t = arange(24).reshape(2,3,4) -> shape {t.shape}, ndim {t.ndim}, size {t.size}")
    p(f"   t[1, 2, 3] = {t[1, 2, 3]}  (last element)")

    p("2. BROADCASTING: align shapes from the right; a size-1 axis is stretched")
    a = np.array([[0.0], [10.0], [20.0]])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    p(f"   (3,1) + (4,) -> {broadcast_shape(a.shape, b.shape)}   numpy agrees: {np.broadcast_shapes(a.shape, b.shape)}")
    p("   a + b =\n" + "\n".join("      " + str(r) for r in (a + b)))
    try:
        broadcast_shape((3, 2), (4,))
    except ValueError as e:
        p(f"   (3,2) + (4,) -> error: {e}")

    p("3. INNER PRODUCT and cosine similarity")
    u, v = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    p(f"   <u,v> = 1*4 + 2*5 + 3*6 = {inner_product(u, v):g}")
    p(f"   cos(u,v) = {cosine_similarity(u, v):.4f}   cos(u,2u) = {cosine_similarity(u, 2 * u):.4f}   "
      f"cos(u,-u) = {cosine_similarity(u, -u):.4f}")
    A, B = np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(3, 2)
    p(f"   matmul as inner products equals A @ B: {np.allclose(matmul_as_inner_products(A, B), A @ B)}")

    p("4. RESHAPING: same 12 numbers, read in a different order")
    d = reshape_demo()
    p("   C order (row by row):\n" + "\n".join("      " + str(r) for r in d["C_order"]))
    p("   F order (column by column):\n" + "\n".join("      " + str(r) for r in d["F_order"]))

    p("5. SVD: A = U S V^T; keeping the k largest singular values gives the best rank-k approximation")
    img = _synthetic_image()
    S = np.linalg.svd(img, compute_uv=False)
    p(f"   96x96 image, top 5 singular values: {np.round(S[:5], 2)}")
    for k in [1, 5, 15, 40]:
        err = np.linalg.norm(img - low_rank_approx(img, k)) / np.linalg.norm(img)
        p(f"   rank {k:2d}: relative error {err:.3f}, energy kept {svd_energy(img)[k - 1]:.1%}, "
          f"storage {k * (96 + 96 + 1)} vs {96 * 96} numbers")

    p("6. NUMERICAL GRADIENT CHECK: (f(x+eps) - f(x-eps)) / 2eps vs the formula")
    x = np.array([0.5, -1.2, 2.0])
    f = lambda x: float(np.sum(np.sin(x) * x ** 2))
    good = lambda x: np.cos(x) * x ** 2 + 2 * x * np.sin(x)
    bad = lambda x: np.cos(x) * x ** 2          # forgot the product rule
    p(f"   analytic  = {good(x)}")
    xc = x.copy()   # numerical_gradient nudges xc in place, so f must read xc (not x)
    p(f"   numerical = {numerical_gradient(lambda: f(xc), xc)}")
    p(f"   relative error, correct formula: {gradient_check(f, good, x):.2e}   (< 1e-7: correct)")
    p(f"   relative error, buggy formula:   {gradient_check(f, bad, x):.2e}   (large: bug found)")
    return out


# --------------------------------------------------------------------------- figures
def _synthetic_image(size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 0.5 * np.sin(6 * xx) * np.cos(4 * yy) + 0.5
    img[(xx - 0.65) ** 2 + (yy - 0.35) ** 2 < 0.03] = 1.0
    img[int(size * 0.6):int(size * 0.85), int(size * 0.15):int(size * 0.45)] = 0.0
    return img


def make_figures(out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 1. Broadcasting: (3,1) + (1,4) -> (3,4)
    a = np.array([[0.0], [10.0], [20.0]])
    b = np.array([[1.0, 2.0, 3.0, 4.0]])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, m, title in zip(axes, [np.broadcast_to(a, (3, 4)), np.broadcast_to(b, (3, 4)), a + b],
                            ["a (3,1) stretched", "b (1,4) stretched", "a + b  (3,4)"]):
        ax.imshow(m, cmap="viridis")
        for (i, j), val in np.ndenumerate(m):
            ax.text(j, i, f"{val:g}", ha="center", va="center", color="white")
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Broadcasting: size-1 axes are virtually repeated")
    paths.append(out_dir / "broadcasting.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. Inner product geometry: projection of u on v
    u, v = np.array([2.0, 1.0]), np.array([1.0, 2.5])
    proj = inner_product(u, v) / inner_product(v, v) * v
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for vec, c, lbl in [(u, "C0", "u"), (v, "C1", "v"), (proj, "C2", "proj_v(u)")]:
        ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
        ax.text(*(vec * 1.05), lbl, color=c)
    ax.plot([u[0], proj[0]], [u[1], proj[1]], "k--", lw=1)
    ax.set_xlim(-0.2, 3), ax.set_ylim(-0.2, 3), ax.set_aspect("equal"), ax.grid(alpha=0.3)
    ax.set_title(f"<u,v>={inner_product(u, v):.2f}, cos={cosine_similarity(u, v):.2f}")
    paths.append(out_dir / "inner_product.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. Reshape order
    demo = reshape_demo()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, key in zip(axes, ["C_order", "F_order"]):
        m = demo[key]
        ax.imshow(m, cmap="tab20")
        for (i, j), val in np.ndenumerate(m):
            ax.text(j, i, str(val), ha="center", va="center")
        ax.set_title(f"arange(12).reshape(3,4, order='{key[0]}')")
        ax.set_xticks([]), ax.set_yticks([])
    paths.append(out_dir / "reshape_order.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 4. SVD low-rank reconstruction
    img = _synthetic_image()
    ranks = [1, 5, 15, 40]
    fig, axes = plt.subplots(1, len(ranks) + 2, figsize=(16, 3))
    axes[0].imshow(img, cmap="gray"), axes[0].set_title("original")
    for ax, k in zip(axes[1:], ranks):
        rec = low_rank_approx(img, k)
        err = np.linalg.norm(img - rec) / np.linalg.norm(img)
        ax.imshow(rec, cmap="gray"), ax.set_title(f"rank {k}  err={err:.3f}")
    energy = svd_energy(img)
    axes[-1].plot(np.arange(1, len(energy) + 1), energy)
    axes[-1].set_title("cumulative energy"), axes[-1].set_xlabel("k"), axes[-1].grid(alpha=0.3)
    for ax in axes[:-1]:
        ax.set_xticks([]), ax.set_yticks([])
    paths.append(out_dir / "svd_low_rank.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 5. Gradient check: error vs epsilon for f(x) = sum(sin(x) * x^2)
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=6)
    f = lambda x: float(np.sum(np.sin(x) * x**2))
    g = lambda x: np.cos(x) * x**2 + 2 * x * np.sin(x)
    eps_values = np.logspace(-12, -1, 23)
    errs = [gradient_check(f, g, x0, eps) for eps in eps_values]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.loglog(eps_values, errs, "o-")
    ax.set_xlabel("epsilon"), ax.set_ylabel("relative error")
    ax.set_title("Gradient check: truncation (right) vs round-off (left)")
    ax.grid(alpha=0.3, which="both")
    paths.append(out_dir / "gradient_check.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths
