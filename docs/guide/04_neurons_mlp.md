# Requirement 4 — McCulloch-Pitts neuron, sigmoid neuron, multilayer network with backpropagation

## Why this module exists

Every model in this project (ResNet, YOLO, LSTM, Transformer) is many simple neurons stacked in layers and trained
with **backpropagation**. This module builds that idea up from scratch in NumPy, one step at a time, and then
checks it against **PyTorch** and **TensorFlow/Keras**:

1. a **McCulloch-Pitts** neuron (weights set by hand, cannot learn)
2. a **sigmoid neuron** (learns its weights, but only for straight-line problems)
3. a **multilayer network (MLP)**: forward pass, loss, backward pass, worked by hand with real numbers
4. the MLP trained on **real classroom crops** (attentive / distracted / inactive), plus the same network in Keras

Code: `src/smartattend/nnfund/neurons.py`, `mlp.py`, `neural_networks.py` ·
tests: `tests/test_neural_networks.py`, `tests/test_mlp_gradients.py`, `tests/test_neurons_optimizers.py` ·
Colab: `notebooks/04_neurons_mlp_colab.ipynb` (CPU is enough).

## Run it

```bash
python -m smartattend neurons              # everything below, about 2 minutes (+10 min the first time for ResNet features)
python -m smartattend neurons --no-keras   # without TensorFlow
python -m pytest tests/test_neural_networks.py tests/test_mlp_gradients.py tests/test_neurons_optimizers.py
```

Results are saved to `reports/eval/neural_networks.json`, figures to `reports/figures/nnfund/`.

## 1. McCulloch-Pitts neuron (1943)

Output 1 if `w · x ≥ threshold`, else 0. Weights are chosen by hand: there is no learning.

```python
class McCullochPitts:
    def __init__(self, weights, threshold):
        self.weights, self.threshold = np.asarray(weights, float), threshold
    def __call__(self, x):
        return (np.atleast_2d(x) @ self.weights >= self.threshold).astype(int)

AND  = McCullochPitts([1, 1], 2)      # both inputs needed
OR   = McCullochPitts([1, 1], 1)      # one input is enough
NAND = McCullochPitts([-1, -1], -1)   # fires unless both are 1
```

One neuron cannot compute XOR ("exactly one input is on"), but **two layers** can: XOR = AND(OR, NAND).

```
   x1 x2 | AND OR NAND | XOR (two layers: AND(OR, NAND))
    0  0 |  0   0   1  |  0
    0  1 |  0   1   1  |  1
    1  0 |  0   1   1  |  1
    1  1 |  1   1   0  |  0
```

## 2. Sigmoid neuron: the first neuron that learns

`p = sigmoid(w · x + b)` gives a probability between 0 and 1. The weights are learned by gradient descent on
**cross-entropy loss**; for sigmoid + cross-entropy the gradient is simply `(p − y) · x`.

```
AND: w=[7.41 7.41] b=-11.29  p=0.00 0.02 0.02 0.97  accuracy=1.00  loss 0.693 -> 0.017
OR : w=[8.66 8.66] b=-3.87   p=0.02 0.99 0.99 1.00  accuracy=1.00  loss 0.693 -> 0.009
XOR: w=[0. 0.]     b=+0.00   p=0.50 0.50 0.50 0.50  accuracy=0.50  loss 0.693 -> 0.693
```

It learns AND and OR, but on XOR it gives up at 50 % because **no single straight line separates XOR**.
That is the reason for hidden layers. Figure `neuron_decision_boundaries.png` shows the straight boundaries of the
single neuron and the curved boundary of a 2-8-2 MLP that solves XOR.

**Why not the step function?** Figure `activations_step_sigmoid_tanh.png`: the step's slope is 0 everywhere,
so gradient descent gets no signal. Sigmoid and tanh are smooth, so the slope tells each weight which way to move.

## 3. Multilayer network: one example worked by hand

Network: 2 inputs → 2 sigmoid hidden units → 2 output scores (softmax). Input `x = [1, 2]`, true class 1.
The weights are round numbers so you can check every line with a calculator.

**Forward pass** (input → prediction):
```
z1 = x W1 + b1             = [1. , 1.2]          (1·0.5 + 2·0.2 + 0.1 = 1.0)
a1 = sigmoid(z1)           = [0.7311, 0.7685]
z2 = a1 W2 + b2            = [0.3468, 0.0069]
p  = softmax(z2)           = [0.5842, 0.4158]    (the network is 42 % sure of the true class)
loss = -log p[1]           = 0.8775
```

**Backward pass** (the chain rule, from the loss back to every weight):
```
dL/dz2 = p - onehot(y)     = [ 0.5842, -0.5842]
dL/dW2 = a1^T dL/dz2       = [ 0.4271, -0.4271,  0.4489, -0.4489]
dL/da1 = dL/dz2 W2^T       = [ 1.1683, -0.701 ]
dL/dz1 = dL/da1 * a1(1-a1) = [ 0.2297, -0.1247]     (sigmoid's slope is a(1-a))
dL/dW1 = x^T dL/dz1        = [ 0.2297, -0.1247,  0.4594, -0.2494]
gradient check (all 12 weights and biases): worst relative error = 2.5e-11  (correct)
```

The last line uses the gradient check from Requirement 3: every hand-derived gradient matches the numerical slope.

In `mlp.py` each layer has a `forward` and a `backward`, and the network just calls them in order:

```python
def forward(self, x, train=True):
    for layer in self.layers:
        x = layer.forward(x, train)
    return x

def backward(self, grad):                  # grad = dLoss/dOutput
    for layer in reversed(self.layers):
        grad = layer.backward(grad)       # each layer stores dW, db and passes dL/dInput back
    return grad

class Dense(Layer):
    def forward(self, x, train=True):
        self.x = x                         # remember the input for the backward pass
        return x @ self.params["W"] + self.params["b"]
    def backward(self, grad_out):
        self.grads["W"][...] = self.x.T @ grad_out
        self.grads["b"][...] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T
```

Training loop (one step): forward → loss → backward → `W ← W − lr · dW`. `tests/test_mlp_gradients.py` checks every
gradient for sigmoid, tanh and ReLU networks, with and without batch normalisation.

## 4a. The MLP on real data: Kaggle classroom crops

**Task:** given one student's crop, predict attentive / distracted / inactive. Data and split come from Requirement 2,
and the test set is **a classroom never seen in training**:

| split | attentive | distracted | inactive |
|---|---|---|---|
| train (3 recording groups) | 1756 | 448 | 70 |
| validation (taimoor-class) | 954 | 143 | 70 |
| test (taimoor-zaeem) | 421 | 646 | 92 |

**How it is trained** (`train_numpy_mlp`):
- inputs standardised with the **train** mean and standard deviation only (using val/test statistics would leak)
- **class-balanced batches**, so the 70 inactive crops are not ignored
- minibatch SGD with momentum 0.9, L2 weight decay, dropout
- the epoch with the best **validation** macro-F1 is kept; the test set is used once, at the end

Two input representations, same NumPy MLP code:

| input | network | parameters | train F1 | val F1 | **test accuracy** | **test macro-F1** |
|---|---|---|---|---|---|---|
| baseline: always "attentive" | – | 0 | – | – | 0.363 | 0.178 |
| raw pixels, 32×32 gray | 1024→128→64→3 | 139,651 | 0.790 | 0.374 | 0.498 | 0.385 |
| raw pixels, **Keras** | same | 139,651 | – | 0.375 | 0.501 | 0.380 |
| frozen ResNet18 features | 512→64→3 | 33,027 | 0.641 | 0.377 | 0.572 | **0.442** |
| ResNet18 features, **Keras** | same | 33,027 | – | 0.373 | 0.596 | 0.437 |

Per-class test F1 (ResNet features): attentive 0.54, distracted 0.65, inactive 0.14.

**How to read this (honestly):**
- **Overfitting is visible.** In `mlp_crops_training.png` training loss falls to near 0 while validation loss rises
  from the first epochs. With raw pixels the network memorises 3 classrooms (train F1 near 1.0 by the end) and is
  near chance (0.33) on a new one.
- **Better inputs help more than a bigger network.** The same MLP on ResNet18 features, with a quarter of the
  parameters, gains +0.06 macro-F1 and +0.07 accuracy on the test classroom. This is the motivation for the
  convolutional networks of Requirement 7.
- **Inactive (sleeping) is almost never found:** only 70 training examples, from few students.
- **Keras and NumPy agree** within 0.005 macro-F1. Small differences come from random initialisation and dropout masks.
- An MLP on single crops is **not** good enough for the product. The app uses face-based head-turn and eye
  cues (Requirement 7) instead. This module is about understanding how networks learn, and it shows why
  architecture and data matter.

Head sizes for the ResNet features were chosen by **validation** macro-F1 from three settings; the test set was not
used to choose anything.

## 4b. Same network in NumPy, PyTorch and TensorFlow

Identical weights and batch, float64, a 4→5→3 tanh network. PyTorch (`autograd`) and TensorFlow
(`GradientTape`) compute gradients automatically; ours are written by hand:

```
NumPy (our backprop)   loss = 1.0590666939
PyTorch autograd       loss = 1.0590666939   largest gradient difference = 3.5e-17
TF GradientTape        loss = 1.0590666939   largest gradient difference = 5.6e-17
```

A difference of 1e-17 is floating-point round-off: **our backpropagation is exactly what the big libraries do.**
(The losses differ at the 12th decimal because our loss adds 1e-12 inside `log` to avoid `log(0)`.)

Keras version of the network, for reference:

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(512,)),
    tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3),                         # logits
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

## Checklist
- [x] McCulloch-Pitts AND / OR / NOT / NOR / NAND, and XOR from two layers
- [x] sigmoid neuron learns AND / OR and fails on XOR, with the reason
- [x] MLP forward pass, loss and backpropagation, worked by hand and gradient-checked
- [x] trained and evaluated on real data with a leakage-safe train / validation / test split, against a baseline
- [x] the same network in TensorFlow/Keras; NumPy = PyTorch = TensorFlow gradients
- [x] figures: activations, decision boundaries, training curves and confusion matrices
- [x] tests: 7 new in `test_neural_networks.py`, plus the existing gradient and neuron tests
