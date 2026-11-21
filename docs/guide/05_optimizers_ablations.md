# Requirement 5 — Optimizers (Batch GD, SGD, Momentum, Nesterov) and ablations

## Why this module exists

Requirement 4 built a network that learns. This module asks **how to train it well**:
- which update rule moves the weights fastest and most reliably (**optimizers**)
- which design choices actually matter (**ablations**: change exactly one thing, keep everything else fixed,
  repeat with several random seeds)

Everything runs on **real face frames** (drowsiness dataset: engaged / neutral / disengaged), with the leakage-safe
time-block split from Requirement 2. Each frame is a 32×32 grayscale face, so augmentation is real image
augmentation. The network is the NumPy MLP from Requirement 4: 1024 → 128 → 64 → 3.

Code: `src/smartattend/nnfund/optimization.py`, `optimizers.py` · tests: `tests/test_optimization.py` (13) ·
Colab: `notebooks/05_optimizers_ablations_colab.ipynb` (CPU is enough).

## Run it

```bash
python -m smartattend optimize                 # about 25 minutes on a laptop CPU
python -m pytest tests/test_optimization.py
```

Results go to `reports/eval/optimization.json`, figures to `reports/figures/nnfund/`.

| split | engaged | neutral | disengaged |
|---|---|---|---|
| train | 658 | 491 | 144 |
| validation | 49 | 115 | 49 |
| test | 92 | 140 | 40 |

**Limitation:** every frame in this dataset shows **one person** with one camera (`docs/DATASET_CARD.md`). The
comparisons between optimizers and settings are valid, but the scores say nothing about other students.

## 1. The update rules

All optimizers use the gradient `g` from backpropagation. They differ in **how much data** each gradient is
computed on, and **how** `g` becomes a weight change:

| optimizer | gradient computed on | update |
|---|---|---|
| Batch GD | all 1,293 training images | `w ← w − lr·g` (1 update per epoch) |
| SGD | 1 image | `w ← w − lr·g` (1,293 updates per epoch) |
| Mini-batch SGD | 32 images | `w ← w − lr·g` (41 updates per epoch) |
| Momentum | 32 images | `v ← μ·v − lr·g ;  w ← w + v` (a heavy ball that keeps rolling) |
| Nesterov | 32 images | like Momentum, but the gradient is taken where the ball is *about to be* |

```python
class Momentum(Optimizer):
    def _update(self, p, g, key):
        v = self.state.setdefault(key, np.zeros_like(p))   # velocity, one per weight array
        v *= self.mu                                        # keep 90 % of the previous direction
        v -= self.lr * g                                    # add the new downhill push
        p += v

class Nesterov(Momentum):                                   # look-ahead form, needs g only at p
    def _update(self, p, g, key):
        v = self.state.setdefault(key, np.zeros_like(p))
        v_old = v.copy()
        v *= self.mu
        v -= self.lr * g
        p += -self.mu * v_old + (1 + self.mu) * v
```

Batch GD vs SGD vs mini-batch is set by the batch size in the training loop (`batch=None, 1, 32`).

### First, a picture: a narrow valley

`optimizer_paths.png`: the loss `f = 0.5(x² + 25y²)` is 25 times steeper across the valley than along it.
Gradient descent zig-zags across the valley. Momentum averages the zig-zags out and builds up speed along the
valley. Nesterov looks ahead, so it overshoots less.

```
f after 40 steps (lower is better):  Gradient descent 1.2e-01   Momentum 1.3e-02   Nesterov 4.2e-03
```

### On real data

**A fair comparison:** every optimizer gets its **own learning rate**, chosen from
`[0.3, 0.1, 0.03, 0.01, 0.003, 0.001]` by the lowest training loss after 10 epochs. Then it is trained for 30 epochs
with 3 seeds. Comparing optimizers at one shared learning rate would just measure which one that rate happens to suit.

```
optimizer                  lr  updates/epoch  s/epoch  train loss  val F1        test F1
Batch GD                  0.3              1     0.06      0.2353   0.644   0.718 ± 0.025
SGD (batch 1)           0.003           1293     5.37      0.0406   0.724   0.741 ± 0.067
Mini-batch SGD (32)       0.1             41     0.17      0.0409   0.740   0.787 ± 0.037
Momentum (32)            0.01             41     0.19      0.0393   0.725   0.747 ± 0.046
Nesterov (32)            0.01             41     0.27      0.0452   0.719   0.730 ± 0.058
```

Training loss after 5 epochs: Batch GD 0.559, mini-batch SGD 0.212, Momentum 0.109, Nesterov 0.096, SGD (batch 1) 0.110.

**What this shows** (`optimizer_comparison_real.png`):
- **Batch GD is the most stable but by far the slowest per epoch.** It makes only 30 updates in 30 epochs
  and is still far behind (loss 0.24 vs 0.04). Its best rate, 0.3, is the top of the grid, so it could go
  somewhat higher, but 30 updates is the real limit.
- **SGD with batch 1 reaches a low loss in few epochs but is 30× slower in wall-clock time** (5.4 s vs 0.17 s per
  epoch): one image at a time cannot use fast matrix maths. Its path is the noisiest, and it is only stable at a small
  learning rate (0.3 diverged).
- **Mini-batch (32) is the practical middle**, which is why every deep-learning library defaults to it.
- **Momentum and Nesterov get down fastest at the start** (half of plain mini-batch SGD's loss after 5 epochs),
  with a 10× smaller learning rate. With μ = 0.9 the steps add up to about `lr / (1 − μ) = 0.1`, the same effective
  step as the best plain SGD rate, so by epoch 30 all three reach the same loss.
- **Test macro-F1 differences between the mini-batch methods (0.73–0.79) are within one standard deviation.** On this
  problem the optimizer changes the speed, not the final quality. Be careful not to over-read a 0.04 difference
  measured with 3 seeds.

## 2. Ablations: one change at a time

Baseline: Momentum (lr 0.01), He initialisation, per-pixel standardisation, no regularisation, no augmentation,
30 epochs. Each row changes **one** thing and is run with **5 seeds**. `gap` = train F1 − validation F1
(large = overfitting). *Early stop* = the test score of the epoch with the best validation F1.

```
                                     train F1  val F1   test F1 (last)  test F1 (early stop)    gap
Initialisation
  He (baseline)                         0.982   0.732    0.773 ± 0.049         0.778 ± 0.054  0.250
  Xavier                                0.980   0.750    0.764 ± 0.059         0.775 ± 0.047  0.230
  small normal (0.01)                   0.977   0.774    0.798 ± 0.012         0.803 ± 0.016  0.203
  all zeros                             0.225   0.125    0.168 ± 0.000         0.168 ± 0.000  0.100
Normalisation
  standardise per pixel (baseline)      0.982   0.732    0.773 ± 0.049         0.778 ± 0.054  0.250
  scale to 0-1 only                     0.745   0.579    0.631 ± 0.137         0.717 ± 0.045  0.166
  raw 0-255                             0.227   0.126    0.183 ± 0.029         0.192 ± 0.028  0.101
  standardise + batch norm              0.979   0.733    0.810 ± 0.029         0.778 ± 0.031  0.246
Regularisation
  none (baseline)                       0.982   0.732    0.773 ± 0.049         0.778 ± 0.054  0.250
  L2 1e-3                               0.982   0.747    0.777 ± 0.047         0.785 ± 0.044  0.235
  L2 1e-2                               0.972   0.742    0.771 ± 0.034         0.807 ± 0.011  0.230
  dropout 0.3                           0.971   0.779    0.785 ± 0.025         0.780 ± 0.051  0.191
  dropout 0.3 + L2 1e-3                 0.974   0.763    0.801 ± 0.019         0.811 ± 0.034  0.211
Augmentation
  none (baseline)                       0.982   0.732    0.773 ± 0.049         0.778 ± 0.054  0.250
  horizontal flip                       0.970   0.768    0.744 ± 0.018         0.756 ± 0.022  0.202
  shift +/-2 px                         0.904   0.784    0.738 ± 0.037         0.746 ± 0.017  0.119
  brightness                            0.971   0.783    0.763 ± 0.029         0.735 ± 0.044  0.188
  flip + shift + brightness             0.783   0.699    0.542 ± 0.059         0.608 ± 0.057  0.084
```

### Initialisation
- **All zeros fails completely** (test 0.168, the same for every seed). Every hidden unit starts identical, gets the
  identical gradient and stays identical. With ReLU it is worse still: `ReLU'(0) = 0`, so no gradient flows at all.
  Only the output bias learns, and the network always predicts one class. **Random initialisation exists to break
  this symmetry.**
- He, Xavier and small normal all work. For a 2-hidden-layer network the differences are small; He/Xavier matter
  more for deep networks (Requirement 7), where a bad scale makes signals shrink or explode layer after layer.

### Normalisation
- **Raw 0–255 pixels: training fails** (0.18). The first layer's outputs are 255× too large, so the very first
  updates are huge and the network gets stuck predicting one class.
- **Scale to 0–1** trains, but slowly and unstably (± 0.137 across seeds).
- **Per-pixel standardisation** (baseline) is clearly better. **Batch norm** adds a further +0.037 test F1, though
  validation F1 does not change, so treat it as "probably helps".
- **A real bug found here:** four corner pixels are black in every frame (standard deviation 0). Dividing by
  `std + 1e-6` turned any shifted image into inputs of about 100,000, and training diverged. The fix is
  `std = max(std, 0.05)`. Lesson: **constant features plus standardisation is a trap**; a test now guards it.

### Regularisation (against overfitting)
- The baseline **overfits**: train F1 0.98 vs validation 0.73 (`overfitting_real.png`: training loss keeps falling
  while validation loss turns up after about 10 epochs).
- **Dropout 0.3 + L2 1e-3** is the best and most consistent setting: test 0.801 ± 0.019 (vs 0.773 ± 0.049), and
  0.811 with early stopping.
- **Early stopping** (keep the best-validation epoch) helps a little almost everywhere and costs nothing.
- The gaps stay large (about 0.2): 1,293 images from few people is a small dataset for 139,651 weights.

### Augmentation
- Augmentation **reduces overfitting** (gap 0.25 → 0.08) and **raises validation F1** (shift 0.784, brightness 0.783
  vs 0.732), **but test F1 goes down**. Validation and test do not agree, so we **do not** adopt it.
- Combining all three in 30 epochs **underfits** (train F1 only 0.78): the task got harder and the network needs
  more epochs.
- An MLP sees pixels as an unordered list, so a 2-pixel shift is a completely different input to it. Augmentation
  pays off much more for convolutional networks (Requirement 7), which are built to handle shifts.

### What we keep
He (or small normal) init + per-pixel standardisation + dropout 0.3 + L2 1e-3 + early stopping, trained with
mini-batch Momentum. Choices are made on **validation**; the test numbers are reported, not used to choose.

## Checklist
- [x] Batch GD, SGD, mini-batch SGD, Momentum, Nesterov, each with its own tuned learning rate, 3 seeds
- [x] loss per epoch, per update and wall-clock time compared; narrow-valley visualisation
- [x] ablations: initialisation (4), normalisation (4), regularisation (5), augmentation (5), each over 5 seeds
- [x] overfitting shown and reduced; early stopping evaluated
- [x] a real normalisation bug found and fixed, with a regression test
- [x] 13 tests in `tests/test_optimization.py`
