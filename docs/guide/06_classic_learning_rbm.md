# Requirement 6 — Hebbian, modified Hebbian, competitive, error-correction learning and the Boltzmann machine

## Why this module exists

Before backpropagation there were simpler learning rules, and each one still lives inside modern systems:

| rule | idea in one line | where it lives today |
|---|---|---|
| Hebbian | "neurons that fire together wire together": strengthen `w` when input and output agree | the basis of PCA-style feature learning |
| modified Hebbian (Oja, Sanger) | Hebb plus a decay term that keeps `w` bounded; finds principal components | eigenfaces, data compression |
| competitive | the closest prototype "wins" and moves toward the input | clustering, self-organising maps, vector quantisation |
| error-correction (perceptron, delta / LMS) | change the weights in proportion to the error | the ancestor of backpropagation |
| Boltzmann machine (RBM) | learn an *energy*: low for data, high for everything else | pre-training, generative models |

Each rule is run on the project's **own data** and checked against a standard method.

Code: `src/smartattend/nnfund/classic_real.py`, `classic_learning.py`, `boltzmann.py` ·
tests: `tests/test_classic_real.py` (6), `tests/test_classic_and_rbm.py` (7) ·
Colab: `notebooks/06_classic_learning_rbm_colab.ipynb` (CPU, about 2 minutes).

## Run it

```bash
python -m smartattend classic
python -m pytest tests/test_classic_real.py tests/test_classic_and_rbm.py
```

Results: `reports/eval/classic_learning.json`. Figures: `reports/figures/nnfund/hebbian_eigenfaces.png`,
`competitive_errorcorr_rbm.png`, `rbm_face_reconstructions.png`.

**Data:** the 1,293 training face frames from Requirement 5 (32×32 = 1,024 pixels), and LFW face embeddings
(128 numbers per face, from the attendance recogniser).
**Limitation:** the drowsiness frames show **one person** (see `docs/DATASET_CARD.md`), so the eigenfaces and the
engagement scores describe that person's face, not students in general.

## 6a. Hebbian learning → eigenfaces

```python
y = w @ x
w += lr * y * x                    # plain Hebb: nothing stops |w| from growing
w += lr * y * (x - y * w)          # Oja: the "- y² w" decay keeps |w| = 1
W += lr * (np.outer(y, x) - np.tril(np.outer(y, y)) @ W)   # Sanger: k neurons, each removes what earlier ones found
```

The face frames are centred (mean face removed). SVD (Requirement 3) gives the exact principal components, which
we use as the answer key.

```
plain Hebb   |w| per epoch: 4.0e-01, 7.0e+00, 2.0e+02, 5.9e+03, 1.7e+05 ...  -> grows without limit (stopped after 10 epochs)
Oja          |w| after 15 epochs = 1.0016;  |cos| with SVD's 1st component = 0.9993
Sanger (8 components) |cos| with SVD components 1-8: 1.00 1.00 1.00 1.00 1.00 0.92 0.80 0.85
variance explained by components 1-8: 26.2% 19.5% 10.9% 6.6% 4.0% 2.9% 2.6% 2.4%
```

- **Plain Hebb explodes** (×30 per epoch): it only ever adds, so `|w|` grows without limit.
- **Oja's single neuron finds the first eigenface** (cos 0.9993) and keeps `|w| = 1`.
- **Sanger's 8 neurons find eigenfaces 1–5 exactly.** Components 6–8 are close (0.80–0.92) because they explain
  almost the same variance (2.4–2.9 %), so any rotation among them is nearly as good. With a learning rate of 0.02
  instead of 0.1, components 4–8 had not converged (cos 0.27, 0.03, …); the learning rate matters here too.
- Figure `hebbian_eigenfaces.png`: SVD eigenfaces (top) and Sanger's (middle) look the same. Eigenface 1 is mostly
  lighting from one side; eigenface 2 is head position.

## 6b. Competitive learning → grouping unknown faces

**Project use:** group faces of people who are *not yet enrolled* (for example, visitors), so a teacher can name each
group once instead of every photo.

```python
j = np.argmax(W @ x)               # winner = most similar prototype (cosine; embeddings have length 1)
W[j] += lr * (x - W[j])            # only the winner learns
W[j] /= np.linalg.norm(W[j])
```

Test: a simulated class of **30 people × 20 photos** from LFW, names hidden, 30 prototypes, 5 different random classes.
**ARI** (adjusted Rand index) is 1 for perfect grouping and 0 for random. **Purity** is the share of faces in a group
dominated by their own identity.

```
method                         ARI          purity        clusters used
competitive, 1 run         0.815 ± 0.034   0.833 ± 0.036       30.0
competitive, best of 10    0.846 ± 0.014   0.853 ± 0.016       30.0
k-means, best of 10        0.989 ± 0.018   0.992 ± 0.014       30.0
random assignment         -0.000 ± 0.001   0.135 ± 0.004       30.0
our ARI matches scikit-learn's adjusted_rand_score to 0e+00
```

- Competitive learning groups the faces well (ARI 0.85, far above random), using only "the winner moves closer".
- **k-means is clearly better (0.99).** For a fair comparison both get 10 restarts, and the restart is chosen
  *without* labels. Spreading out the starting points (k-means++ seeding) did not help (0.80). The difference is the
  update itself: online winner-take-all leaves each prototype where its last few samples pushed it, while k-means
  moves it to the exact average of its group. **For the product we would use k-means;** competitive learning is the
  online, neuron-style version of the same idea.

## 6c. Error-correction learning → engagement from face pixels

```python
# perceptron: learns only from mistakes
p = argmax(W @ x)
if p != y:  W[y] += x;  W[p] -= x
# delta / LMS (Widrow-Hoff): every sample, in proportion to the error
W += lr * np.outer(onehot(y) - W @ x, x)
```

Engaged / neutral / disengaged from 1,024 standardised pixels, same split as Requirement 5:

```
perceptron rule (update only on mistakes): training mistakes 26.2% -> 4.3%;  TEST accuracy 0.647  macro-F1 0.616
delta / LMS rule (update by the size of the error): training MSE 0.206 -> 0.041;  TEST accuracy 0.746  macro-F1 0.740
for comparison, the MLP with hidden layers (Requirement 5 baseline): TEST macro-F1 0.773
```

- **The delta rule generalises much better than the perceptron** (0.740 vs 0.616). The perceptron stops caring as soon
  as a sample is classified correctly, even by a hair. The delta rule keeps pushing toward a margin, because it
  reduces the error *size*, not just the error count.
- **One linear layer gets within 0.03 of the MLP.** For this single person, engagement is almost linearly separable.
- The delta rule is backpropagation with **no hidden layer**: add hidden layers and the chain rule, and you have
  Requirement 4.

## 6d. Restricted Boltzmann Machine → face features without labels

An RBM has visible units `v` (the 1,024 pixels) and hidden units `h` (128 features), and an energy
`E(v, h) = −a·v − b·h − vᵀWh`. Training with **contrastive divergence (CD-1)** lowers the energy of real faces and
raises the energy of the model's own "dreams":

```python
ph0 = sigmoid(v0 @ W + b);  h0 = sample(ph0)          # positive phase: what the data activates
v1  = sample(sigmoid(h0 @ W.T + a))                    # one step of the model "dreaming"
ph1 = sigmoid(v1 @ W + b)                              # negative phase
W  += lr * (v0.T @ ph0 - v1.T @ ph1) / n               # data correlations minus dream correlations
```

```
reconstruction MSE 0.0276 -> 0.0056
free energy: real test faces -1772.2  vs  the same pixels shuffled -1290.1  (real is lower in 100.0% of pairs)
on the 128 RBM features (8x fewer inputs, learned WITHOUT labels): TEST macro-F1 perceptron 0.575, delta rule 0.554
```

- **The RBM learned what this face looks like.** Every real test face has lower free energy than the *same pixels
  shuffled*, so it measures *structure*, not brightness. Reconstructions (`rbm_face_reconstructions.png`) keep the
  face, glasses and expression.
- Its 128 features were learned **without labels**. They carry some engagement signal (0.575, chance is 0.33), but
  less than the raw pixels with labels (0.74). Features learned to *reconstruct* a face are not the same as features
  that *separate* engaged from disengaged.
- **A real problem found and fixed:** with all biases starting at 0, the RBM did not learn at all (reconstruction
  error 0.0287 → 0.0287, features at chance). Starting each visible bias at `log(p / (1 − p))` of that pixel's average
  (Hinton's *Practical guide to training RBMs*, 2010) fixed it: error 0.028 → 0.0056 and useful features. It is now
  `RBM.init_visible_bias`, with a test.

## Checklist
- [x] Hebbian (diverges), Oja (first component, |w| = 1), Sanger (8 eigenfaces), checked against SVD
- [x] competitive learning on real face embeddings, vs k-means and random, with our ARI checked against scikit-learn
- [x] perceptron and delta (LMS) error-correction rules on real face pixels, vs the MLP
- [x] RBM with CD-1 on faces: reconstruction, free energy, unsupervised features
- [x] figures for every part; results in `reports/eval/classic_learning.json`
- [x] 13 tests (6 new + 7 existing)
