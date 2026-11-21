# Requirement 3 — Mathematical sandbox (tensors, broadcasting, inner products, reshaping, SVD, gradient checking)

## Why this module exists

Every deep-learning model in this project is built from a handful of operations: arrays of numbers
(**tensors**), adding arrays of different shapes (**broadcasting**), multiplying and summing
(**inner products**), rearranging data (**reshaping**), compressing data (**SVD**), and computing
gradients for training. If one of these is wrong, the model trains badly and nothing tells you why.
The sandbox writes each operation out by hand, checks it against NumPy, and draws what it does.

Code: `src/smartattend/nnfund/tensor_sandbox.py` · tests: `tests/test_tensor_sandbox.py` ·
Colab: `notebooks/03_math_sandbox_colab.ipynb` (CPU is enough).

## Run it

```bash
python -m smartattend math                        # prints worked examples + draws 5 figures
python -m pytest tests/test_tensor_sandbox.py     # 11 tests
```

## 1. Tensors

A tensor is an n-dimensional array. Its **shape** lists the size of each axis. A colour image is
`(height, width, 3)`; a batch of 32 images for PyTorch is `(32, 3, 224, 224)`.

```
t = arange(24).reshape(2,3,4) -> shape (2, 3, 4), ndim 3, size 24
t[1, 2, 3] = 23  (last element)
```

## 2. Broadcasting

Adding a `(3,1)` array to a `(4,)` array works without copying data: NumPy compares shapes **from the
right**. Two sizes are compatible if they are equal or one of them is 1, and a size-1 axis is stretched.

```python
def broadcast_shape(shape_a, shape_b):
    ndim = max(len(shape_a), len(shape_b))
    a = (1,) * (ndim - len(shape_a)) + tuple(shape_a)      # pad the shorter shape with 1s on the left
    b = (1,) * (ndim - len(shape_b)) + tuple(shape_b)
    for da, db in zip(a, b):
        if da != db and 1 not in (da, db):                 # neither equal nor 1 -> impossible
            raise ValueError(...)
    return tuple(max(da, db) for da, db in zip(a, b))
```

```
(3,1) + (4,) -> (3, 4)   numpy agrees: (3, 4)
a + b =
   [ 1.  2.  3.  4.]
   [11. 12. 13. 14.]
   [21. 22. 23. 24.]
(3,2) + (4,) -> error: shapes (3, 2) and (4,) are not broadcastable
```

**Where we use it:** adding a bias to every row of a layer's output (`x @ W + b`), normalising images
(`(img - mean) / std` with a per-channel mean).
**Figure:** `reports/figures/nnfund/broadcasting.png`

## 3. Inner products

`<u, v> = u1·v1 + u2·v2 + …`. Dividing by both lengths gives the **cosine similarity**: 1 means the same
direction, 0 means unrelated, −1 means opposite. A matrix product is a table of inner products
(row of A · column of B). `matmul_as_inner_products` computes it with explicit loops and is tested against `A @ B`.

```
<u,v> = 1*4 + 2*5 + 3*6 = 32
cos(u,v) = 0.9746   cos(u,2u) = 1.0000   cos(u,-u) = -1.0000
matmul as inner products equals A @ B: True
```

**Where we use it:** every neuron computes `w · x + b`, and **face recognition is a cosine similarity**
between two 128-number face embeddings (≥ 0.363 = same person).
**Figure:** `inner_product.png` draws the projection of u onto v.

## 4. Reshaping

Reshape re-reads the same numbers in a new layout. **C order** fills row by row; **F order** fills column by column.

```
C order (row by row):        F order (column by column):
   [ 0  1  2  3]                [ 0  3  6  9]
   [ 4  5  6  7]                [ 1  4  7 10]
   [ 8  9 10 11]                [ 2  5  8 11]
```

**Where we use it:** OpenCV gives images as `(H, W, 3)`, but PyTorch wants `(3, H, W)`, so we transpose
them (`img.transpose(2, 0, 1)`). Using reshape instead of transpose here is a classic silent bug.
**Figure:** `reshape_order.png`

## 5. SVD (singular value decomposition)

Any matrix can be written `A = U · S · Vᵀ`, where S holds **singular values** sorted from largest to smallest.
Keeping only the k largest gives the **best possible rank-k approximation** (Eckart–Young theorem).

```python
def low_rank_approx(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]
```

```
96x96 image, top 5 singular values: [54.46 24.39  7.26  3.67  2.17]
rank  1: relative error 0.431, energy kept 81.4%, storage 193 vs 9216 numbers
rank  5: relative error 0.059, energy kept 99.7%, storage 965 vs 9216 numbers
rank 15: relative error 0.019, energy kept 100.0%, storage 2895 vs 9216 numbers
```

Rank 5 keeps 99.7 % of the image's "energy" using about a tenth of the numbers.
**Where it connects:** the same idea is behind PCA, which we use to plot the LSTM's hidden states
(Requirement 9), and behind Oja's learning rule (Requirement 6).
**Figure:** `svd_low_rank.png`

## 6. Numerical gradient checking

Training needs the gradient (slope) of the loss with respect to every weight. We write formulas for it
(backpropagation), and a wrong formula is easy to write. We can check any formula against the
definition of a derivative:

```
slope ≈ ( f(x + ε) − f(x − ε) ) / 2ε          (central difference)
```

```python
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for each index i of x:
        old = x[i]
        x[i] = old + eps; f_plus = f()
        x[i] = old - eps; f_minus = f()
        x[i] = old                                 # always put the value back
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad
```

The relative error `|a − b| / (|a| + |b|)` should be below about 1e-7:

```
analytic  = [0.699 2.759 1.973]
numerical = [0.699 2.759 1.973]
relative error, correct formula: 3.74e-11   (< 1e-7: correct)
relative error, buggy formula:   1.00e+00   (large: bug found)
```

The "buggy formula" forgot the product rule, and the check catches it immediately. In Requirement 4 the
same routine checks **every weight of every layer** of our neural network.

**Choosing ε** (`gradient_check.png`): too large an ε and the straight-line approximation is poor
(truncation error, right side of the plot); too small and floating-point round-off dominates (left side).
The best values are around 1e-5 to 1e-6, which is why 1e-5 is the default.

> **A real bug this check caught while writing this guide:** the first version of the demo printed
> `numerical = [0. 0. 0.]`. The function read the original `x` while `numerical_gradient` was nudging a
> *copy*, so the nudges never reached the function. Lesson: `f` must read the very array being perturbed.
> A test now guards against it (`test_demo_numerical_gradient_matches_analytic`).

## Checklist
- [x] tensors, broadcasting (own rule, checked against NumPy), inner products, reshaping, SVD, gradient check
- [x] visual output for each (`reports/figures/nnfund/`)
- [x] printed worked examples (`python -m smartattend math`)
- [x] 11 automated tests
