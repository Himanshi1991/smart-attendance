"""Requirement 4 walkthrough: McCulloch-Pitts neurons -> sigmoid neuron -> multilayer network with backpropagation.

    python -m smartattend neurons            # printed demo + MLP on real classroom crops + framework check

Four parts, each printed and each tested:
  1. McCulloch-Pitts gates (fixed weights), and XOR built from two layers of them
  2. a sigmoid neuron that *learns* AND / OR but cannot learn XOR
  3. a tiny network worked through by hand: forward pass, loss, backward pass, gradient check
  4. a NumPy MLP trained on the Kaggle classroom student crops (attentive / distracted / inactive),
     and a check that NumPy, PyTorch and TensorFlow compute the same loss and gradients
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smartattend.nnfund.mlp import MLP, softmax, softmax_cross_entropy
from smartattend.nnfund.neurons import BOOL_INPUTS, MP_GATES, TRUTH_TABLES, McCullochPitts, SigmoidNeuron, sigmoid
from smartattend.nnfund.tensor_sandbox import numerical_gradient, relative_error

STATES = ["attentive", "distracted", "inactive"]


# --------------------------------------------------------------------------- 1. McCulloch-Pitts
NAND = McCullochPitts([-1, -1], -1)          # fires unless both inputs are 1


def mp_xor(x: np.ndarray) -> np.ndarray:
    """XOR = AND(OR(x), NAND(x)): impossible for one threshold unit, easy for two layers."""
    hidden = np.column_stack([MP_GATES["OR"](x), NAND(x)])
    return MP_GATES["AND"](hidden)


def demo_mcculloch_pitts() -> list[str]:
    out = ["1) McCulloch-Pitts neuron: output 1 if  w . x >= threshold, else 0  (weights fixed by hand)",
           "   x1 x2 | AND OR NAND | XOR (two layers: AND(OR, NAND))"]
    xor = mp_xor(BOOL_INPUTS)
    for i, (a, b) in enumerate(BOOL_INPUTS.astype(int)):
        x = BOOL_INPUTS[i:i + 1]
        out.append(f"    {a}  {b} |  {MP_GATES['AND'](x)[0]}   {MP_GATES['OR'](x)[0]}   {NAND(x)[0]}  |  {xor[i]}")
    return out


# --------------------------------------------------------------------------- 2. sigmoid neuron
def demo_sigmoid_neuron() -> list[str]:
    out = ["", "2) Sigmoid neuron: p = sigmoid(w . x + b), weights LEARNED by gradient descent on cross-entropy"]
    for gate in ["AND", "OR", "XOR"]:
        n = SigmoidNeuron(2).fit(BOOL_INPUTS, TRUTH_TABLES[gate])
        acc = float(np.mean(n.predict(BOOL_INPUTS) == TRUTH_TABLES[gate]))
        probs = " ".join(f"{p:.2f}" for p in n.predict_proba(BOOL_INPUTS))
        out.append(f"   {gate:3s}: w={np.round(n.w, 2) + 0.0} b={n.b + 0.0:+.2f}  p={probs}  accuracy={acc:.2f}  "
                   f"loss {n.loss_history[0]:.3f} -> {n.loss_history[-1]:.3f}")
    out.append("   XOR is not linearly separable: one straight line cannot split it, so one neuron fails.")
    return out


# --------------------------------------------------------------------------- 3. worked example
def tiny_network() -> MLP:
    """2 inputs -> 2 sigmoid hidden units -> 2 output logits, with round weights so the numbers are easy to follow."""
    m = MLP.build([2, 2, 2], activation="sigmoid", seed=0)
    m.layers[0].params["W"][...] = [[0.5, -0.3], [0.2, 0.8]]
    m.layers[0].params["b"][...] = [0.1, -0.1]
    m.layers[2].params["W"][...] = [[1.0, -1.0], [-0.5, 0.7]]
    m.layers[2].params["b"][...] = [0.0, 0.2]
    return m


def demo_worked_example() -> list[str]:
    m = tiny_network()
    x, y = np.array([[1.0, 2.0]]), np.array([1])
    W1, b1, W2, b2 = (m.layers[0].params["W"], m.layers[0].params["b"], m.layers[2].params["W"], m.layers[2].params["b"])
    z1 = x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    p = softmax(z2)
    loss, dz2 = softmax_cross_entropy(m.forward(x), y)
    m.backward(dz2)
    f = lambda: softmax_cross_entropy(m.forward(x), y)[0]
    worst = max(relative_error(layer.grads[name].copy(), numerical_gradient(f, layer.params[name]))
                for layer, name in m.trainable())
    r = lambda a: np.array2string(np.round(np.asarray(a).ravel(), 4), separator=", ")
    return [
        "", "3) Worked example: network 2 -> 2 (sigmoid) -> 2 (softmax), one input x=[1, 2], true class 1",
        "   FORWARD",
        f"   z1 = x W1 + b1           = {r(z1)}",
        f"   a1 = sigmoid(z1)         = {r(a1)}",
        f"   z2 = a1 W2 + b2          = {r(z2)}",
        f"   p  = softmax(z2)         = {r(p)}",
        f"   loss = -log p[1]         = {loss:.4f}",
        "   BACKWARD (chain rule, from the loss back to every weight)",
        f"   dL/dz2 = p - onehot(y)   = {r(dz2)}",
        f"   dL/dW2 = a1^T dL/dz2     = {r(m.layers[2].grads['W'])}",
        f"   dL/da1 = dL/dz2 W2^T     = {r(dz2 @ W2.T)}",
        f"   dL/dz1 = dL/da1 * a1(1-a1) = {r((dz2 @ W2.T) * a1 * (1 - a1))}",
        f"   dL/dW1 = x^T dL/dz1      = {r(m.layers[0].grads['W'])}",
        f"   gradient check (all 12 weights and biases): worst relative error = {worst:.1e}  "
        f"({'correct' if worst < 1e-7 else 'BUG'})",
    ]


# --------------------------------------------------------------------------- 4a. real data
def load_crop_arrays(processed_dir: str | Path, size: int = 32) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Grayscale student crops resized to size x size and flattened. Splits come from Requirement 2 (by recording group)."""
    import cv2

    root = Path(processed_dir) / "crops"
    data = {}
    for split in ["train", "val", "test"]:
        X, y = [], []
        for label, state in enumerate(STATES):
            for f in sorted((root / split / state).glob("*.jpg")):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                X.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).ravel())
                y.append(label)
        data[split] = (np.asarray(X, np.float64) / 255.0, np.asarray(y))
    return data


def resnet18_features(processed_dir: str | Path, cache: str | Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """512-number description of every crop from an ImageNet-pretrained ResNet18 with its last layer removed.
    The ResNet is frozen (not trained here); only our NumPy MLP on top of it learns. Cached to an .npz file."""
    cache = Path(cache or Path(processed_dir) / "resnet18_features.npz")
    if cache.exists():
        z = np.load(cache)
        return {s: (z[f"{s}_X"], z[f"{s}_y"]) for s in ["train", "val", "test"]}
    import cv2
    import torch
    import torchvision

    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    data, arrays = {}, {}
    for split in ["train", "val", "test"]:
        files = [(f, label) for label, state in enumerate(STATES)
                 for f in sorted((Path(processed_dir) / "crops" / split / state).glob("*.jpg"))]
        feats = []
        for s in range(0, len(files), 64):
            batch = [(cv2.resize(cv2.imread(str(f))[:, :, ::-1], (224, 224)) / 255.0 - mean) / std for f, _ in files[s:s + 64]]
            with torch.no_grad():
                feats.append(net(torch.tensor(np.stack(batch).transpose(0, 3, 1, 2), dtype=torch.float32)).numpy())
        data[split] = (np.concatenate(feats).astype(np.float64), np.array([l for _, l in files]))
        arrays[f"{split}_X"], arrays[f"{split}_y"] = data[split]
    np.savez_compressed(cache, **arrays)
    return data


def standardize(data: dict) -> dict:
    """Per-pixel mean / std from the TRAIN split only (using val/test statistics would leak)."""
    mean = data["train"][0].mean(axis=0)
    std = data["train"][0].std(axis=0) + 1e-6
    return {k: ((X - mean) / std, y) for k, (X, y) in data.items()}


def balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Oversample every class to the size of the largest one (inactive has only 70 training crops)."""
    n = np.bincount(y).max()
    return rng.permutation(np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=True) for c in np.unique(y)]))


def train_numpy_mlp(data: dict, hidden=(128, 64), epochs: int = 40, lr: float = 0.01, batch: int = 64,
                    l2: float = 1e-3, dropout: float = 0.3, seed: int = 0) -> tuple[MLP, dict]:
    """Minibatch SGD with momentum; keeps the epoch with the best VALIDATION macro-F1 (test is never looked at)."""
    from smartattend.eval.metrics import macro_f1
    from smartattend.nnfund.optimizers import Momentum

    (Xtr, ytr), (Xva, yva) = data["train"], data["val"]
    rng = np.random.default_rng(seed)
    model = MLP.build([Xtr.shape[1], *hidden, len(STATES)], activation="relu", init="he", dropout=dropout, seed=seed)
    opt = Momentum(lr=lr, mu=0.9)
    hist = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best = (-1.0, None, 0)
    for epoch in range(epochs):
        idx = balanced_indices(ytr, rng)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            loss, grad = softmax_cross_entropy(model.forward(Xtr[b], train=True), ytr[b])
            model.backward(grad)
            for layer, name in model.trainable():
                if name == "W":
                    layer.grads[name] += l2 * layer.params[name]
            opt.step(model)
        for split, X, y in [("train", Xtr, ytr), ("val", Xva, yva)]:
            logits = model.forward(X, train=False)
            hist[f"{split}_loss"].append(softmax_cross_entropy(logits, y)[0])
            hist[f"{split}_f1"].append(macro_f1(y, logits.argmax(1), len(STATES)))
        if hist["val_f1"][-1] > best[0]:
            best = (hist["val_f1"][-1], [{k: v.copy() for k, v in L.params.items()} for L in model.layers], epoch + 1)
    for layer, params in zip(model.layers, best[1]):
        layer.params.update(params)
    hist["best_epoch"] = best[2]
    hist["hyperparameters"] = dict(hidden=list(hidden), epochs=epochs, lr=lr, momentum=0.9, batch=batch, l2=l2,
                                   dropout=dropout, input="32x32 grayscale, standardized", sampling="class-balanced")
    hist["n_parameters"] = model.parameter_count()
    return model, hist


def train_keras_mlp(data: dict, hidden=(128, 64), epochs: int = 40, lr: float = 0.01, batch: int = 64,
                    l2: float = 1e-3, dropout: float = 0.3, seed: int = 0) -> tuple[object, dict]:
    """The same network, written in TensorFlow / Keras, trained on the same balanced batches."""
    import tensorflow as tf

    from smartattend.eval.metrics import macro_f1

    tf.keras.utils.set_random_seed(seed)
    (Xtr, ytr), (Xva, yva) = data["train"], data["val"]
    reg = tf.keras.regularizers.l2(l2 / 2)          # Keras adds l2*sum(w^2); our NumPy loop adds (l2/2)*sum(w^2)
    model = tf.keras.Sequential([tf.keras.Input(shape=(Xtr.shape[1],))])
    for h in hidden:
        model.add(tf.keras.layers.Dense(h, activation="relu", kernel_initializer="he_normal", kernel_regularizer=reg))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(len(STATES), kernel_regularizer=reg))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    rng = np.random.default_rng(seed)
    hist = {"val_f1": []}
    best = (-1.0, None, 0)
    for epoch in range(epochs):
        idx = balanced_indices(ytr, rng)
        model.fit(Xtr[idx].astype("float32"), ytr[idx], batch_size=batch, epochs=1, shuffle=False, verbose=0)
        pred = model.predict(Xva.astype("float32"), verbose=0).argmax(1)
        hist["val_f1"].append(macro_f1(yva, pred, len(STATES)))
        if hist["val_f1"][-1] > best[0]:
            best = (hist["val_f1"][-1], model.get_weights(), epoch + 1)
    model.set_weights(best[1])
    hist["best_epoch"] = best[2]
    hist["n_parameters"] = int(model.count_params())
    return model, hist


# --------------------------------------------------------------------------- 4b. framework agreement
def framework_gradients(seed: int = 0) -> dict:
    """Same weights, same batch -> loss and every gradient from NumPy (our backprop), PyTorch autograd and TF GradientTape."""
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(8, 4)), rng.integers(0, 3, size=8)
    m = MLP.build([4, 5, 3], activation="tanh", init="xavier", seed=seed)
    W1, b1, W2, b2 = (m.layers[0].params["W"], m.layers[0].params["b"], m.layers[2].params["W"], m.layers[2].params["b"])
    loss_np, g = softmax_cross_entropy(m.forward(X), y)
    m.backward(g)
    grads_np = [m.layers[0].grads["W"], m.layers[0].grads["b"], m.layers[2].grads["W"], m.layers[2].grads["b"]]
    result = {"numpy": {"loss": loss_np}}

    import torch

    tp = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (W1, b1, W2, b2)]
    logits = torch.tanh(torch.tensor(X) @ tp[0] + tp[1]) @ tp[2] + tp[3]
    lt = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
    lt.backward()
    result["pytorch"] = {"loss": lt.item(), "max_grad_diff": max(float(np.abs(t.grad.numpy() - gn).max())
                                                                 for t, gn in zip(tp, grads_np))}
    try:
        import tensorflow as tf
    except ImportError:
        result["tensorflow"] = {"skipped": "tensorflow not installed"}
        return result
    tv = [tf.Variable(a, dtype=tf.float64) for a in (W1, b1, W2, b2)]
    with tf.GradientTape() as tape:
        lg = tf.tanh(tf.constant(X) @ tv[0] + tv[1]) @ tv[2] + tv[3]
        lf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.constant(y), logits=lg))
    gt = tape.gradient(lf, tv)
    result["tensorflow"] = {"loss": float(lf), "max_grad_diff": max(float(np.abs(t.numpy() - gn).max())
                                                                    for t, gn in zip(gt, grads_np))}
    return result


def demo_frameworks() -> list[str]:
    r = framework_gradients()
    out = ["", "4b) Same network, same weights, same batch in three libraries (float64):",
           f"   NumPy (our backprop)   loss = {r['numpy']['loss']:.10f}"]
    for lib, name in [("pytorch", "PyTorch autograd"), ("tensorflow", "TF GradientTape")]:
        if "skipped" in r[lib]:
            out.append(f"   {name:22s} skipped ({r[lib]['skipped']})")
        else:
            out.append(f"   {name:22s} loss = {r[lib]['loss']:.10f}   largest gradient difference = "
                       f"{r[lib]['max_grad_diff']:.1e}")
    return out


# --------------------------------------------------------------------------- run everything
# The two input representations compared in 4a. Head sizes for the ResNet features were chosen on VALIDATION macro-F1
# among (64, d0.3, l2 1e-3), (128, d0.5, l2 1e-2) and (64, d0.5, l2 1e-2); the test split was not used to choose.
EXPERIMENTS = {
    "raw_pixels": dict(label="raw pixels (32x32 grayscale = 1024 numbers)", hidden=(128, 64), dropout=0.3, l2=1e-3, lr=0.01),
    "resnet18_features": dict(label="frozen ImageNet ResNet18 features (512 numbers)", hidden=(64,), dropout=0.5, l2=1e-2,
                              lr=0.005),
}


def _plot_run(plt, ax_loss, ax_f1, hist, epochs, title):
    ep = np.arange(1, epochs + 1)
    ax_loss.plot(ep, hist["train_loss"], label="train"), ax_loss.plot(ep, hist["val_loss"], label="validation")
    ax_loss.set_title(f"{title}: loss"), ax_loss.set_xlabel("epoch")
    ax_f1.plot(ep, hist["train_f1"], label="train"), ax_f1.plot(ep, hist["val_f1"], label="validation")
    ax_f1.axhline(1 / 3, color="k", lw=0.8, ls=":", label="chance (1/3)")
    ax_f1.axvline(hist["best_epoch"], color="gray", ls="--", label=f"kept epoch {hist['best_epoch']}")
    ax_f1.set_title(f"{title}: macro-F1"), ax_f1.set_xlabel("epoch"), ax_f1.set_ylim(0, 1.02)
    for ax in (ax_loss, ax_f1):
        ax.grid(alpha=0.3), ax.legend(fontsize=8)


def _plot_confusion(ax, cm, title):
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), STATES), ax.set_yticks(range(3), STATES)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title(title)


def run(processed_dir: str | Path, eval_dir: str | Path, fig_dir: str | Path, epochs: int = 40,
        with_keras: bool = True) -> tuple[list[str], dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smartattend.eval.metrics import classification_report

    lines = demo_mcculloch_pitts() + demo_sigmoid_neuron() + demo_worked_example()
    inputs = {"raw_pixels": standardize(load_crop_arrays(processed_dir)),
              "resnet18_features": standardize(resnet18_features(processed_dir))}
    yte = inputs["raw_pixels"]["test"][1]
    counts = {k: np.bincount(v[1], minlength=3).tolist() for k, v in inputs["raw_pixels"].items()}
    majority = int(np.bincount(inputs["raw_pixels"]["train"][1]).argmax())
    base = classification_report(yte, np.full_like(yte, majority), 3)
    lines += ["", "4a) NumPy MLP on real student crops (Kaggle classroom; split by recording group in Requirement 2)",
              f"   crops [attentive, distracted, inactive]: {counts}",
              f"   baseline 'always attentive': TEST accuracy {base['accuracy']:.3f}  macro-F1 {base['macro_f1']:.3f}"]
    result = {"class_counts": counts, "majority_baseline": base, "experiments": {}}
    hists, reps = {}, {}
    for key, ex in EXPERIMENTS.items():
        data = inputs[key]
        t0 = time.perf_counter()
        model, hist = train_numpy_mlp(data, hidden=ex["hidden"], dropout=ex["dropout"], l2=ex["l2"], lr=ex["lr"],
                                      epochs=epochs)
        secs = time.perf_counter() - t0
        rep = classification_report(yte, model.predict(data["test"][0]), 3)
        hists[key], reps[key] = hist, rep
        sizes = " -> ".join(str(s) for s in [data["train"][0].shape[1], *ex["hidden"], 3])
        lines += [f"   [{ex['label']}]",
                  f"     NumPy MLP {sizes}, {hist['n_parameters']:,} parameters, {epochs} epochs in {secs:.0f} s",
                  f"     train macro-F1 {hist['train_f1'][hist['best_epoch'] - 1]:.3f}   val {max(hist['val_f1']):.3f} "
                  f"(epoch {hist['best_epoch']})   TEST accuracy {rep['accuracy']:.3f}  macro-F1 {rep['macro_f1']:.3f}",
                  "     per-class test F1  " + "  ".join(f"{s} {f:.2f}" for s, f in zip(STATES, rep["f1"]))]
        result["experiments"][key] = {"numpy_mlp": {"test": rep, "val_macro_f1": max(hist["val_f1"]),
                                                    "train_macro_f1_at_best": hist["train_f1"][hist["best_epoch"] - 1],
                                                    "best_epoch": hist["best_epoch"], "train_seconds": round(secs, 1),
                                                    "hyperparameters": hist["hyperparameters"],
                                                    "n_parameters": hist["n_parameters"]}}
        if with_keras:
            try:
                t0 = time.perf_counter()
                kmodel, khist = train_keras_mlp(data, hidden=ex["hidden"], dropout=ex["dropout"], l2=ex["l2"],
                                                lr=ex["lr"], epochs=epochs)
                krep = classification_report(yte, kmodel.predict(data["test"][0].astype("float32"), verbose=0).argmax(1), 3)
                result["experiments"][key]["keras_mlp"] = {
                    "test": krep, "val_macro_f1": max(khist["val_f1"]), "best_epoch": khist["best_epoch"],
                    "train_seconds": round(time.perf_counter() - t0, 1), "n_parameters": khist["n_parameters"]}
                lines.append(f"     Keras (TensorFlow), same network: {khist['n_parameters']:,} parameters, val "
                             f"{max(khist['val_f1']):.3f}   TEST accuracy {krep['accuracy']:.3f}  macro-F1 {krep['macro_f1']:.3f}")
            except ImportError:
                lines.append("     Keras comparison skipped: tensorflow not installed")

    lines += demo_frameworks()
    result["framework_gradients"] = framework_gradients()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for row, (key, ex) in enumerate(EXPERIMENTS.items()):
        title = "raw pixels" if key == "raw_pixels" else "ResNet18 features"
        _plot_run(plt, axes[row, 0], axes[row, 1], hists[key], epochs, title)
        _plot_confusion(axes[row, 2], reps[key]["confusion"], f"{title}: test confusion")
    fig.suptitle("NumPy MLP on Kaggle classroom crops (test = a classroom never seen in training)")
    fig.tight_layout()
    paths = [fig_dir / "mlp_crops_training.png"]
    fig.savefig(paths[0], dpi=110, bbox_inches="tight")
    plt.close(fig)

    from smartattend.nnfund import neurons

    paths += neurons.make_figures(fig_dir)
    result["figures"] = [str(p) for p in paths]
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "neural_networks.json").write_text(json.dumps(result, indent=2))
    lines.append(f"\nsaved {eval_dir / 'neural_networks.json'}")
    lines += [f"figure: {p}" for p in paths]
    return lines, result
