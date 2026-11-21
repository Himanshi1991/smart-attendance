"""Optimizer comparison, ablations and an overfitting study on engagement features.

Results (JSON + CSV + PNG) go to reports/experiments/ so every number in the report is reproducible.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

from smartattend.data.synthetic import Standardizer, leakage_safe_split, make_engagement_features
from smartattend.nnfund.mlp import MLP
from smartattend.nnfund.optimizers import SGD, Momentum, Nesterov
from smartattend.nnfund.trainer import TrainConfig, evaluate, fit


def _load_data(n: int, seed: int):
    X, y, group = make_engagement_features(n=n, seed=seed)
    tr, va, te = leakage_safe_split(len(X), seed=seed)
    sc = Standardizer().fit(X[tr])
    return {k: (sc.transform(X[idx]), y[idx], group[idx]) for k, idx in
            [("train", tr), ("val", va), ("test", te)]}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def optimizer_comparison(cfg: dict, out_dir: Path, logger: logging.Logger) -> dict:
    d = _load_data(cfg["n_samples"], cfg.get("seed", 0))
    Xtr, ytr, _ = d["train"]
    Xva, yva, _ = d["val"]
    sizes = [Xtr.shape[1], *cfg["hidden_sizes"], 3]
    runs = {
        "BatchGD": (lambda: SGD(lr=0.5), None),
        "SGD": (lambda: SGD(lr=cfg["lr"]), cfg["batch_size"]),
        "Momentum": (lambda: Momentum(lr=cfg["lr"] / 5, mu=cfg["momentum"]), cfg["batch_size"]),
        "Nesterov": (lambda: Nesterov(lr=cfg["lr"] / 5, mu=cfg["momentum"]), cfg["batch_size"]),
    }
    results = {}
    for name, (make_opt, bs) in runs.items():
        model = MLP.build(sizes, activation=cfg["activation"], init=cfg["init"], seed=0)
        hist = fit(model, make_opt(), Xtr, ytr, TrainConfig(epochs=cfg["epochs"], batch_size=bs, seed=0), Xva, yva)
        results[name] = hist
        logger.info(f"[optimizers] {name:9s} final train_loss={hist['train_loss'][-1]:.4f} "
                    f"val_acc={hist['val_acc'][-1]:.3f}")

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in results.items():
        axes[0].plot(h["train_loss"], label=name)
        axes[1].plot(h["val_acc"], label=name)
    axes[0].set_title("training loss per epoch"), axes[0].set_yscale("log")
    axes[1].set_title("validation accuracy")
    for ax in axes:
        ax.set_xlabel("epoch"), ax.legend(), ax.grid(alpha=0.3)
    fig.savefig(out_dir / "optimizer_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    summary = {k: {"final_train_loss": v["train_loss"][-1], "final_val_acc": v["val_acc"][-1],
                   "best_val_acc": max(v["val_acc"]), "hyperparameters": v["hyperparameters"]}
               for k, v in results.items()}
    (out_dir / "optimizer_comparison.json").write_text(json.dumps(summary, indent=2))
    return summary


ABLATIONS = {
    "baseline":            dict(),
    "init=xavier":         dict(init="xavier"),
    "init=normal(0.01)":   dict(init="normal"),
    "activation=sigmoid":  dict(activation="sigmoid"),
    "activation=tanh":     dict(activation="tanh"),
    "+batchnorm":          dict(batchnorm=True),
    "+dropout(0.3)":       dict(dropout=0.3),
    "+L2(1e-3)":           dict(l2=1e-3),
    "+augment(noise 0.2)": dict(augment_noise=0.2),
}


def ablations(cfg: dict, out_dir: Path, logger: logging.Logger, seeds=(0, 1, 2)) -> list[dict]:
    rows = []
    for name, change in ABLATIONS.items():
        val_accs, test_accs = [], []
        for seed in seeds:
            d = _load_data(cfg["n_samples"], seed)
            Xtr, ytr, _ = d["train"]
            arch = dict(activation=cfg["activation"], init=cfg["init"], batchnorm=False, dropout=0.0)
            arch.update({k: v for k, v in change.items() if k in arch})
            train = dict(l2=0.0, augment_noise=0.0)
            train.update({k: v for k, v in change.items() if k in train})
            model = MLP.build([Xtr.shape[1], *cfg["hidden_sizes"], 3], seed=seed, **arch)
            fit(model, Momentum(lr=cfg["lr"] / 5, mu=cfg["momentum"]), Xtr, ytr,
                TrainConfig(epochs=cfg["epochs"], batch_size=cfg["batch_size"], seed=seed, **train))
            val_accs.append(evaluate(model, *d["val"][:2])[1])
            test_accs.append(evaluate(model, *d["test"][:2])[1])
        row = {"variant": name, "val_acc_mean": float(np.mean(val_accs)), "val_acc_std": float(np.std(val_accs)),
               "test_acc_mean": float(np.mean(test_accs)), "test_acc_std": float(np.std(test_accs))}
        rows.append(row)
        logger.info(f"[ablation] {name:20s} val={row['val_acc_mean']:.3f} +/- {row['val_acc_std']:.3f}")

    with open(out_dir / "ablations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [r["variant"] for r in rows]
    ax.barh(names, [r["val_acc_mean"] for r in rows], xerr=[r["val_acc_std"] for r in rows], color="C0")
    ax.axvline(rows[0]["val_acc_mean"], color="k", ls="--", lw=1)
    ax.set_xlabel(f"validation accuracy (mean ± std over {len(seeds)} seeds)")
    ax.set_xlim(0.3, 1.0), ax.invert_yaxis(), ax.grid(alpha=0.3, axis="x")
    fig.savefig(out_dir / "ablations.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return rows


def overfitting_study(cfg: dict, out_dir: Path, logger: logging.Logger) -> dict:
    """Large net + tiny training set: watch train/val curves diverge, then regularise."""
    d = _load_data(cfg["n_samples"], 0)
    Xtr, ytr, _ = d["train"]
    Xtr, ytr = Xtr[:60], ytr[:60]
    Xva, yva, _ = d["val"]
    curves = {}
    for name, arch, train in [("no regularisation", {}, {}),
                              ("dropout 0.5", {"dropout": 0.5}, {}),
                              ("L2 5e-2", {}, {"l2": 5e-2})]:
        model = MLP.build([Xtr.shape[1], 128, 128, 3], activation="relu", init="he", seed=0, **arch)
        curves[name] = fit(model, Momentum(lr=0.01, mu=0.9), Xtr, ytr,
                           TrainConfig(epochs=400, batch_size=16, seed=0, **train), Xva, yva)
        h = curves[name]
        h["early_stop_epoch"] = int(np.argmin(h["val_loss"])) + 1
        logger.info(f"[overfit] {name:18s} train_acc={h['train_acc'][-1]:.3f} val_acc={h['val_acc'][-1]:.3f} "
                    f"gap={h['train_acc'][-1] - h['val_acc'][-1]:.3f} "
                    f"early-stop epoch={h['early_stop_epoch']} (val_acc there={h['val_acc'][h['early_stop_epoch'] - 1]:.3f})")

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, h) in enumerate(curves.items()):
        axes[0].plot(h["train_loss"], color=f"C{i}", label=f"{name} (train)")
        axes[0].plot(h["val_loss"], color=f"C{i}", ls="--", label=f"{name} (val)")
        axes[1].plot(h["train_acc"], color=f"C{i}", label=f"{name} (train)")
        axes[1].plot(h["val_acc"], color=f"C{i}", ls="--", label=f"{name} (val)")
    axes[0].set_title("loss: val rising = overfitting"), axes[0].set_yscale("log")
    axes[1].set_title("accuracy: train vs val gap")
    for ax in axes:
        ax.set_xlabel("epoch"), ax.legend(fontsize=8), ax.grid(alpha=0.3)
    fig.savefig(out_dir / "overfitting.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    summary = {k: {"train_acc": v["train_acc"][-1], "val_acc": v["val_acc"][-1],
                   "early_stop_epoch": v["early_stop_epoch"]} for k, v in curves.items()}
    (out_dir / "overfitting.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_all(cfg: dict, out_dir: str | Path, logger: logging.Logger) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer_comparison(cfg, out_dir, logger)
    ablations(cfg, out_dir, logger)
    overfitting_study(cfg, out_dir, logger)
