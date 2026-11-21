"""Requirement 5: optimizers (Batch GD, SGD, mini-batch SGD, Momentum, Nesterov) and ablations of initialisation,
normalisation, regularisation and augmentation, on REAL face frames.

    python -m smartattend optimize

Data: the drowsiness face frames (engaged / neutral / disengaged) with the leakage-safe time-block split of
Requirement 2. Each frame is a 32x32 grayscale face, so augmentation is real image augmentation (flip, shift,
brightness). The network is the NumPy MLP of Requirement 4: 1024 -> 128 -> 64 -> 3.
Every number is reported on the validation split for choices and on the test split once, as mean +/- std over seeds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smartattend.nnfund.mlp import MLP, softmax_cross_entropy
from smartattend.nnfund.optimizers import SGD, Momentum, Nesterov

CLASSES = ["engaged", "neutral", "disengaged"]
SIZE = 32
HIDDEN = (128, 64)


# --------------------------------------------------------------------------- data
def load_faces(raw_dir: str | Path, seed: int = 42, cache: str | Path | None = None) -> dict:
    """{'train'|'val'|'test': (images N x 32 x 32 in [0,1], labels)} using the Requirement 2 time-block split."""
    cache = Path(cache) if cache else Path(raw_dir).parent.parent / "processed" / "drowsiness_faces32.npz"
    if cache.exists():
        z = np.load(cache)
        return {s: (z[f"{s}_X"], z[f"{s}_y"]) for s in ["train", "val", "test"]}
    import cv2

    from smartattend.data.drowsiness import load_split_frames

    df = load_split_frames(raw_dir, seed=seed)
    out, arrays = {}, {}
    for s in ["train", "val", "test"]:
        d = df[df["split"] == s]
        X = np.stack([cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                      for p in d["path"]]).astype(np.float64) / 255.0
        out[s] = (X, d["engagement"].to_numpy())
        arrays[f"{s}_X"], arrays[f"{s}_y"] = out[s]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    return out


class Normalizer:
    """Turns images into network inputs. Statistics come from the TRAIN split only."""

    MODES = ("raw_0_255", "scale_0_1", "standardize")

    def __init__(self, mode: str = "standardize"):
        self.mode = mode

    def fit(self, X: np.ndarray) -> "Normalizer":
        flat = X.reshape(len(X), -1)
        # Floor the std: 4 corner pixels are always black (std 0). Dividing by ~1e-6 there turns any augmented
        # (shifted) image into inputs of size 1e5 and training diverges. 0.05 is about the 5th percentile of pixel std.
        self.mean, self.std = flat.mean(axis=0), np.maximum(flat.std(axis=0), 0.05)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        flat = X.reshape(len(X), -1)
        if self.mode == "raw_0_255":
            return flat * 255.0
        if self.mode == "scale_0_1":
            return flat
        return (flat - self.mean) / self.std


# --------------------------------------------------------------------------- augmentation (images in [0,1])
def augment(X: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "none":
        return X
    X = X.copy()
    n = len(X)
    if kind in ("flip", "flip+shift+brightness"):
        flip = rng.random(n) < 0.5
        X[flip] = X[flip, :, ::-1]
    if kind in ("shift", "flip+shift+brightness"):
        for i, (dy, dx) in enumerate(rng.integers(-2, 3, size=(n, 2))):
            X[i] = np.roll(np.roll(X[i], dy, axis=0), dx, axis=1)
    if kind in ("brightness", "flip+shift+brightness"):
        X = np.clip(X * rng.uniform(0.75, 1.25, size=(n, 1, 1)) + rng.uniform(-0.08, 0.08, size=(n, 1, 1)), 0, 1)
    return X


# --------------------------------------------------------------------------- one training run
OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "nesterov": Nesterov}


def train(data: dict, optimizer: str = "momentum", lr: float = 0.01, batch: int | None = 32, epochs: int = 30,
          init: str = "he", activation: str = "relu", norm: str = "standardize", batchnorm: bool = False,
          dropout: float = 0.0, l2: float = 0.0, aug: str = "none", seed: int = 0) -> dict:
    """Train the 1024-128-64-3 MLP and return per-epoch curves plus final and best-validation scores."""
    from smartattend.eval.metrics import macro_f1

    rng = np.random.default_rng(seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    normalizer = Normalizer(norm).fit(Xtr)
    Vva, Vte = normalizer(Xva), normalizer(Xte)
    model = MLP.build([SIZE * SIZE, *HIDDEN, len(CLASSES)], activation=activation, init=init, batchnorm=batchnorm,
                      dropout=dropout, seed=seed)
    opt = OPTIMIZERS[optimizer](lr=lr) if optimizer == "sgd" else OPTIMIZERS[optimizer](lr=lr, mu=0.9)
    bs = len(Xtr) if batch is None else batch
    h = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "updates": [], "seconds": []}
    updates, t0, best = 0, time.perf_counter(), (-1.0, None, 0)
    Vtr_eval = normalizer(Xtr)
    for epoch in range(epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            xb = normalizer(augment(Xtr[b], aug, rng))
            loss, grad = softmax_cross_entropy(model.forward(xb, train=True), ytr[b])
            model.backward(grad)
            if l2:
                for layer, name in model.trainable():
                    if name == "W":
                        layer.grads[name] += l2 * layer.params[name]
            opt.step(model)
            updates += 1
        for split, V, y in [("train", Vtr_eval, ytr), ("val", Vva, yva)]:
            logits = model.forward(V, train=False)
            loss = softmax_cross_entropy(logits, y)[0]
            h[f"{split}_loss"].append(loss if np.isfinite(loss) else float("nan"))
            h[f"{split}_f1"].append(macro_f1(y, np.nan_to_num(logits).argmax(1), len(CLASSES)))
        h["updates"].append(updates)
        h["seconds"].append(time.perf_counter() - t0)
        if h["val_f1"][-1] > best[0]:
            best = (h["val_f1"][-1], model.predict(Vte), epoch + 1)
    h["test_f1_last"] = macro_f1(yte, model.predict(Vte), len(CLASSES))
    h["test_f1_early_stop"] = macro_f1(yte, best[1], len(CLASSES))
    h["val_f1_last"], h["val_f1_best"], h["best_epoch"] = h["val_f1"][-1], best[0], best[2]
    h["diverged"] = not np.isfinite(h["train_loss"][-1]) or h["train_loss"][-1] > 50
    return h


# --------------------------------------------------------------------------- 5a. optimizers
OPTIMIZER_RUNS = {          # name: (update rule, batch size)
    "Batch GD": ("sgd", None),
    "SGD (batch 1)": ("sgd", 1),
    "Mini-batch SGD (32)": ("sgd", 32),
    "Momentum (32)": ("momentum", 32),
    "Nesterov (32)": ("nesterov", 32),
}
LR_GRID = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def compare_optimizers(data: dict, epochs: int = 30, seeds=(0, 1, 2), tune_epochs: int = 10) -> dict:
    """Each optimizer gets its own learning rate (lowest training loss after tune_epochs, seed 0), then 3 seeds."""
    res = {}
    for name, (rule, bs) in OPTIMIZER_RUNS.items():
        tried = {}
        for lr in LR_GRID:
            h = train(data, rule, lr, bs, epochs=tune_epochs, seed=0)
            tried[lr] = float("inf") if h["diverged"] or np.isnan(h["train_loss"][-1]) else h["train_loss"][-1]
        lr = min(tried, key=tried.get)
        runs = [train(data, rule, lr, bs, epochs=epochs, seed=s) for s in seeds]
        res[name] = {"lr": lr, "lr_search_train_loss": tried, "runs": runs,
                     "updates_per_epoch": runs[0]["updates"][0], "seconds_per_epoch": runs[0]["seconds"][-1] / epochs}
    return res


class _Bowl:
    """Fake one-parameter 'model' f(p) = 0.5 (p0^2 + 25 p1^2): a long narrow valley (condition number 25)."""

    def __init__(self, start):
        self.params, self.grads = {"p": np.array(start, float)}, {"p": np.zeros(2)}

    def trainable(self):
        yield self, "p"


def bowl_paths(steps: int = 40) -> dict[str, np.ndarray]:
    paths = {}
    for name, opt in [("Gradient descent (lr 0.07)", SGD(0.07)), ("Momentum (lr 0.02, mu 0.8)", Momentum(0.02, 0.8)),
                      ("Nesterov (lr 0.02, mu 0.8)", Nesterov(0.02, 0.8))]:
        m = _Bowl([-9.0, 1.6])
        pts = [m.params["p"].copy()]
        for _ in range(steps):
            m.grads["p"][...] = m.params["p"] * np.array([1.0, 25.0])
            opt.step(m)
            pts.append(m.params["p"].copy())
        paths[name] = np.array(pts)
    return paths


# --------------------------------------------------------------------------- 5b. ablations
BASE = dict(optimizer="momentum", lr=0.01, batch=32, init="he", activation="relu", norm="standardize",
            batchnorm=False, dropout=0.0, l2=0.0, aug="none")
ABLATIONS = {
    "Initialisation": {
        "He (baseline)": {},
        "Xavier": {"init": "xavier"},
        "small normal (0.01)": {"init": "normal"},
        "all zeros": {"init": "zeros"},
    },
    "Normalisation": {
        "standardise per pixel (baseline)": {},
        "scale to 0-1 only": {"norm": "scale_0_1"},
        "raw 0-255": {"norm": "raw_0_255"},
        "standardise + batch norm": {"batchnorm": True},
    },
    "Regularisation": {
        "none (baseline)": {},
        "L2 1e-3": {"l2": 1e-3},
        "L2 1e-2": {"l2": 1e-2},
        "dropout 0.3": {"dropout": 0.3},
        "dropout 0.3 + L2 1e-3": {"dropout": 0.3, "l2": 1e-3},
    },
    "Augmentation": {
        "none (baseline)": {},
        "horizontal flip": {"aug": "flip"},
        "shift +/-2 px": {"aug": "shift"},
        "brightness": {"aug": "brightness"},
        "flip + shift + brightness": {"aug": "flip+shift+brightness"},
    },
}


def run_ablations(data: dict, epochs: int = 30, seeds=(0, 1, 2, 3, 4)) -> dict:
    out = {}
    for group, variants in ABLATIONS.items():
        out[group] = {}
        for name, change in variants.items():
            runs = [train(data, epochs=epochs, seed=s, **{**BASE, **change}) for s in seeds]
            agg = lambda k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
            out[group][name] = {"change": change, "val_f1_last": agg("val_f1_last"), "test_f1_last": agg("test_f1_last"),
                                "test_f1_early_stop": agg("test_f1_early_stop"),
                                "train_f1_last": [float(np.mean([r["train_f1"][-1] for r in runs])),
                                                  float(np.std([r["train_f1"][-1] for r in runs]))],
                                "train_val_gap": float(np.mean([r["train_f1"][-1] - r["val_f1_last"] for r in runs])),
                                "curves": {"train_loss": np.mean([r["train_loss"] for r in runs], 0).tolist(),
                                           "val_loss": np.mean([r["val_loss"] for r in runs], 0).tolist()}}
    return out


# --------------------------------------------------------------------------- figures + report
def _figures(opt_res: dict, abl: dict, fig_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. optimizer paths on a narrow valley
    fig, ax = plt.subplots(figsize=(8, 4))
    gx, gy = np.meshgrid(np.linspace(-10, 2, 300), np.linspace(-2.2, 2.2, 300))
    ax.contour(gx, gy, 0.5 * (gx**2 + 25 * gy**2), levels=np.geomspace(0.05, 80, 14), cmap="Greys", linewidths=0.7)
    for name, p in bowl_paths().items():
        ax.plot(p[:, 0], p[:, 1], "o-", ms=2.5, lw=1.2, label=f"{name}: f after 40 steps = "
                f"{0.5 * (p[-1, 0]**2 + 25 * p[-1, 1]**2):.1e}")
    ax.plot(0, 0, "k*", ms=12)
    ax.set_title("Optimizer paths on a narrow valley f = 0.5(x² + 25y²), start (-9, 1.6), minimum *")
    ax.legend(fontsize=8, loc="lower right")
    paths.append(fig_dir / "optimizer_paths.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. optimizers on real data: loss vs epoch, vs number of updates, vs seconds
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for i, (name, r) in enumerate(opt_res.items()):
        loss = np.mean([run["train_loss"] for run in r["runs"]], 0)
        val = np.mean([run["val_f1"] for run in r["runs"]], 0)
        lab = f"{name}, lr {r['lr']}"
        axes[0].plot(np.arange(1, len(loss) + 1), loss, color=f"C{i}", label=lab)
        axes[1].plot(r["runs"][0]["updates"], loss, color=f"C{i}", label=lab)
        axes[2].plot(np.arange(1, len(val) + 1), val, color=f"C{i}", label=lab)
    axes[0].set(title="training loss per epoch", xlabel="epoch", yscale="log")
    axes[1].set(title="training loss per weight update", xlabel="number of updates (log)", xscale="log", yscale="log")
    axes[2].set(title="validation macro-F1", xlabel="epoch")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Optimizers on real face frames (mean of 3 seeds, each with its own tuned learning rate)")
    fig.tight_layout()
    paths.append(fig_dir / "optimizer_comparison_real.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. ablations: test macro-F1 per variant, grouped
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True)
    for ax, (group, variants) in zip(axes, abl.items()):
        names = list(variants)
        m = [variants[n]["test_f1_last"][0] for n in names]
        s = [variants[n]["test_f1_last"][1] for n in names]
        ax.barh(names, m, xerr=s, color=["C7"] + ["C0"] * (len(names) - 1), capsize=3)
        ax.axvline(m[0], color="k", ls="--", lw=0.8)
        ax.axvline(1 / 3, color="r", ls=":", lw=0.8)
        ax.set_title(group), ax.invert_yaxis(), ax.grid(alpha=0.3, axis="x"), ax.set_xlim(0, 1)
        ax.set_xlabel("test macro-F1 (mean ± std, 5 seeds)")
    fig.suptitle("Ablations (grey = baseline; dashed = baseline level; red dotted = chance)")
    fig.tight_layout()
    paths.append(fig_dir / "ablations_real.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 4. overfitting: train vs val loss with and without regularisation
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(["none (baseline)", "dropout 0.3 + L2 1e-3"]):
        c = abl["Regularisation"][name]["curves"]
        ep = np.arange(1, len(c["train_loss"]) + 1)
        ax.plot(ep, c["train_loss"], color=f"C{i}", label=f"{name}: train")
        ax.plot(ep, c["val_loss"], color=f"C{i}", ls="--", label=f"{name}: validation")
    ax.set(title="Overfitting: training loss keeps falling, validation loss turns up", xlabel="epoch",
           ylabel="cross-entropy", yscale="log")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    paths.append(fig_dir / "overfitting_real.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths


def run(raw_dir: str | Path, eval_dir: str | Path, fig_dir: str | Path, epochs: int = 30, seed: int = 42,
        logger=None) -> tuple[list[str], dict]:
    log = (lambda m: logger.info(m)) if logger else (lambda m: None)
    data = load_faces(raw_dir, seed=seed)
    counts = {s: np.bincount(y, minlength=3).tolist() for s, (_, y) in data.items()}
    lines = ["Requirement 5 on real face frames (drowsiness dataset, time-block split from Requirement 2)",
             f"  [engaged, neutral, disengaged] per split: {counts};  network 1024 -> 128 -> 64 -> 3", "",
             "5a) Optimizers (each with its own learning rate from the grid " + str(LR_GRID) + ", 3 seeds)",
             f"  {'optimizer':22s} {'lr':>6s} {'updates/epoch':>13s} {'s/epoch':>8s} {'train loss':>11s} "
             f"{'val F1':>7s} {'test F1':>14s}"]
    t0 = time.perf_counter()
    opt_res = compare_optimizers(data, epochs=epochs)
    summary = {}
    for name, r in opt_res.items():
        tl = np.mean([x["train_loss"][-1] for x in r["runs"]])
        vf = np.mean([x["val_f1_last"] for x in r["runs"]])
        tf_ = [x["test_f1_last"] for x in r["runs"]]
        loss_by_epoch = np.mean([x["train_loss"] for x in r["runs"]], 0)
        summary[name] = {"lr": r["lr"], "updates_per_epoch": r["updates_per_epoch"],
                         "seconds_per_epoch": round(r["seconds_per_epoch"], 3), "final_train_loss": float(tl),
                         "train_loss_epoch5": float(loss_by_epoch[min(4, len(loss_by_epoch) - 1)]),
                         "val_f1": float(vf), "test_f1": [float(np.mean(tf_)), float(np.std(tf_))],
                         "lr_search_train_loss": r["lr_search_train_loss"]}
        lines.append(f"  {name:22s} {r['lr']:>6g} {r['updates_per_epoch']:>13d} {r['seconds_per_epoch']:>8.2f} "
                     f"{tl:>11.4f} {vf:>7.3f} {np.mean(tf_):>7.3f} ± {np.std(tf_):.3f}")
    log(f"[optimize] optimizers done in {time.perf_counter() - t0:.0f}s")
    bowl = {k: float(0.5 * (p[-1, 0] ** 2 + 25 * p[-1, 1] ** 2)) for k, p in bowl_paths().items()}
    lines += ["  narrow-valley test (40 steps, f at the end; lower is better): "
              + ";  ".join(f"{k.split(' (')[0]} {v:.1e}" for k, v in bowl.items())]

    t0 = time.perf_counter()
    abl = run_ablations(data, epochs=epochs)
    log(f"[optimize] ablations done in {time.perf_counter() - t0:.0f}s")
    lines += ["", "5b) Ablations: one change at a time from the baseline (Momentum lr 0.01, He init, standardised input,",
              "    no regularisation, no augmentation). Test macro-F1 mean ± std over 5 seeds; train-val gap shows overfitting.",
              f"  {'':36s} {'train F1':>9s} {'val F1':>7s} {'test F1 (last)':>16s} {'test F1 (early stop)':>21s} {'gap':>6s}"]
    for group, variants in abl.items():
        lines.append(f"  {group}")
        for name, v in variants.items():
            lines.append(f"    {name:34s} {v['train_f1_last'][0]:>9.3f} {v['val_f1_last'][0]:>7.3f} "
                         f"{v['test_f1_last'][0]:>8.3f} ± {v['test_f1_last'][1]:.3f} "
                         f"{v['test_f1_early_stop'][0]:>13.3f} ± {v['test_f1_early_stop'][1]:.3f} {v['train_val_gap']:>6.3f}")

    paths = _figures(opt_res, abl, Path(fig_dir))
    result = {"class_counts": counts, "optimizers": summary, "narrow_valley_final_loss": bowl,
              "ablations": {g: {n: {k: v for k, v in d.items() if k != "curves"} for n, d in vs.items()}
                            for g, vs in abl.items()},
              "baseline": BASE, "epochs": epochs, "figures": [str(p) for p in paths]}
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "optimization.json").write_text(json.dumps(result, indent=2))
    lines += ["", f"saved {eval_dir / 'optimization.json'}"] + [f"figure: {p}" for p in paths]
    return lines, result
