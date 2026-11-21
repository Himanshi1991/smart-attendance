"""LSTM engagement-trend models.

1. Student-level (REAL data): from the last L frames of facial features (eye/mouth aspect ratio,
   head roll/pitch) forecast the mean engagement score over the next H frames. Windows never cross
   the time-block split, so train/val/test stay leakage-free.
2. Class-level (SIMULATED sessions): forecast the class engagement curve for the next minutes from the
   past minutes; observation noise uses the real engagement classifier's confusion matrix.

A hand-written LSTM cell (FlexLSTM) lets the cell non-linearity be tanh (standard), sigmoid, or a
step function trained with a straight-through estimator, to compare their behaviour.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from smartattend.data.drowsiness import FEATURES


# --------------------------------------------------------------------------- models
class StepSTE(torch.autograd.Function):
    """Heaviside forward; backward passes the gradient where |x| < 1 (hard-tanh surrogate)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() < 1).to(g.dtype)


ACTS = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "step": StepSTE.apply}


class FlexLSTM(nn.Module):
    def __init__(self, n_in: int, hidden: int, act: str = "tanh"):
        super().__init__()
        self.hidden, self.act = hidden, ACTS[act]
        self.x2g = nn.Linear(n_in, 4 * hidden)
        self.h2g = nn.Linear(hidden, 4 * hidden, bias=False)

    def forward(self, x):  # x: (B, T, F) -> all hidden states (B, T, H)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        c = x.new_zeros(B, self.hidden)
        xs = self.x2g(x)
        hs = []
        for t in range(T):
            i, f, g, o = (xs[:, t] + self.h2g(h)).chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f + 1.0), torch.sigmoid(o)  # forget bias 1
            c = f * c + i * self.act(g)
            h = o * self.act(c)
            hs.append(h)
        return torch.stack(hs, 1)


class TrendLSTM(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32, act: str = "tanh", n_out: int = 1):
        super().__init__()
        self.rnn = FlexLSTM(n_in, hidden, act) if act != "torch" else None
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True) if act == "torch" else None
        self.out = nn.Linear(hidden, n_out)

    def hidden_states(self, x):
        return self.rnn(x) if self.rnn is not None else self.lstm(x)[0]

    def forward(self, x):
        return self.out(self.hidden_states(x)[:, -1])


def fit_regressor(model, Xtr, ytr, Xva, yva, epochs=60, lr=3e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, ytr = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xva, yva = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, state, hist = np.inf, None, []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(Xtr[idx]).squeeze(-1), ytr[idx].squeeze(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.l1_loss(model(Xva).squeeze(-1), yva.squeeze(-1)))
        hist.append(v)
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model, hist


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()


# --------------------------------------------------------------------------- student-level (real data)
def make_windows(df: pd.DataFrame, L: int = 24, H: int = 12, stride: int = 2):
    """Windows inside one time block only. Features standardised with TRAIN statistics."""
    df = df.copy()
    df["score"] = 1.0 - df["drowsiness_score"]  # engagement score in [0, 1]
    tr = df[df["split"] == "train"]
    mu, sd = tr[FEATURES].mean(), tr[FEATURES].std() + 1e-6
    df[FEATURES] = (df[FEATURES] - mu) / sd
    out = {s: ([], [], []) for s in ["train", "val", "test"]}
    for (blk, split), g in df.groupby(["block", "split"]):
        X = g[FEATURES + ["score"]].to_numpy(np.float32)  # past score is an input (autoregressive)
        s = g["score"].to_numpy(np.float32)
        for t in range(L, len(g) - H + 1, stride):
            out[split][0].append(X[t - L:t])
            out[split][1].append(s[t:t + H].mean())
            out[split][2].append(blk)
    return {k: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for k, v in out.items()}


def student_level(df: pd.DataFrame, out_dir: Path, logger: logging.Logger, seeds=(0, 1, 2)) -> dict:
    W = make_windows(df)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = W["train"], W["val"], W["test"]
    logger.info(f"[trend] windows train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    res = {"n_windows": {k: int(len(v[0])) for k, v in W.items()}}
    mae = lambda p: float(np.mean(np.abs(p - yte)))
    res["last_value"] = mae(Xte[:, -1, -1])
    res["window_mean"] = mae(Xte[:, :, -1].mean(1))
    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=1.0).fit(Xtr.reshape(len(Xtr), -1), ytr)
    res["ridge_flat"] = mae(ridge.predict(Xte.reshape(len(Xte), -1)))
    curves = {}
    for act in ["tanh", "sigmoid", "step"]:
        maes = []
        for seed in seeds:
            torch.manual_seed(seed)
            m, hist = fit_regressor(TrendLSTM(Xtr.shape[2], 32, act), Xtr, ytr, Xva, yva, seed=seed)
            maes.append(mae(predict(m, Xte)))
            if seed == 0:
                curves[act] = hist
                if act == "tanh":
                    best_model = m
        res[f"lstm_{act}"] = {"mean": float(np.mean(maes)), "std": float(np.std(maes))}
        logger.info(f"[trend] LSTM({act}) test MAE {np.mean(maes):.4f} +/- {np.std(maes):.4f}")
    torch.save(best_model.state_dict(), out_dir / "lstm_trend_student.pt")
    _plot_student(best_model, Xte, yte, curves, res, out_dir)
    return res


# --------------------------------------------------------------------------- class-level (simulated)
def simulate_class_sessions(n_sessions: int, confusion: np.ndarray, minutes: int = 45, n_students: int = 30,
                            seed: int = 0) -> np.ndarray:
    """Per-minute observed class engagement fraction for simulated lectures.

    Each student's latent engagement decays with fatigue, jumps after interactive moments (random
    'activity' minutes), and drops after a long monologue; the observed label is the true label
    pushed through the real classifier's row-normalised confusion matrix.
    """
    rng = np.random.default_rng(seed)
    P = confusion / confusion.sum(1, keepdims=True)
    out = np.zeros((n_sessions, minutes), np.float32)
    for s in range(n_sessions):
        base = rng.normal(1.2, 0.4, n_students)
        fatigue = rng.uniform(0.02, 0.06, n_students)
        activities = rng.choice(np.arange(8, minutes - 3), size=rng.integers(1, 4), replace=False)
        z = base.copy()
        for t in range(minutes):
            z = z - fatigue + rng.normal(0, 0.15, n_students)
            if t in activities:
                z = z + rng.uniform(0.6, 1.2)
            true = np.where(z > 0.6, 0, np.where(z > -0.2, 1, 2))   # engaged / neutral / disengaged
            obs = np.array([rng.choice(3, p=P[c]) for c in true])
            out[s, t] = np.mean(obs == 0)
    return out


def class_level(confusion: np.ndarray, out_dir: Path, logger: logging.Logger, past: int = 10, ahead: int = 5) -> dict:
    sessions = {k: simulate_class_sessions(n, confusion, seed=i) for i, (k, n) in
                enumerate([("train", 300), ("val", 60), ("test", 100)])}

    def windows(S):
        X, Y = [], []
        for s in S:
            for t in range(past, len(s) - ahead + 1):
                X.append(s[t - past:t, None])
                Y.append(s[t:t + ahead])
        return np.array(X), np.array(Y)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = windows(sessions["train"]), windows(sessions["val"]), windows(sessions["test"])
    model, _ = fit_regressor(TrendLSTM(1, 32, "torch", n_out=ahead), Xtr, Ytr, Xva, Yva, epochs=40)
    model.eval()
    with torch.no_grad():
        P = model(torch.tensor(Xte)).numpy()
    res = {"lstm_MAE": float(np.mean(np.abs(P - Yte))),
           "persistence_MAE": float(np.mean(np.abs(Xte[:, -1, :] - Yte))),
           "moving_avg3_MAE": float(np.mean(np.abs(Xte[:, -3:, 0].mean(1, keepdims=True) - Yte))),
           "horizon_minutes": ahead, "context_minutes": past, "simulated": True}
    logger.info(f"[trend] class-level (simulated) {res}")
    torch.save(model.state_dict(), out_dir / "lstm_trend_class.pt")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = sessions["test"][0]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(s, "k.-", label="observed engaged fraction (simulated lecture)")
    for t in range(past, len(s) - ahead + 1, ahead):
        with torch.no_grad():
            p = model(torch.tensor(s[t - past:t, None][None])).numpy()[0]
        ax.plot(range(t, t + ahead), p, "C1-", lw=2, label="LSTM 5-min forecast" if t == past else None)
    ax.set_xlabel("minute"), ax.set_ylabel("fraction engaged"), ax.legend(), ax.grid(alpha=0.3)
    ax.set_title(f"Class-level trend forecast: MAE LSTM {res['lstm_MAE']:.3f} vs persistence {res['persistence_MAE']:.3f}")
    fig.savefig(out_dir / "trend_class_forecast.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return res


def _plot_student(model, Xte, yte, curves, res, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    p = predict(model, Xte)
    axes[0].plot(yte[:150], "k.-", ms=3, label="actual next-12-frame engagement")
    axes[0].plot(p[:150], "C1-", label="LSTM forecast")
    axes[0].plot(Xte[:150, -1, -1], "C2:", label="last value")
    axes[0].set_title("student-level forecast (test windows)"), axes[0].legend(fontsize=8)
    for act, h in curves.items():
        axes[1].plot(h, label=f"{act}: test MAE {res[f'lstm_{act}']['mean']:.3f}")
    axes[1].set_title("LSTM cell activation: val MAE per epoch"), axes[1].set_xlabel("epoch"), axes[1].legend()
    with torch.no_grad():
        H = model.hidden_states(torch.tensor(Xte, dtype=torch.float32))[:, -1].numpy()
    Z = PCA(2).fit_transform(H)
    sc = axes[2].scatter(Z[:, 0], Z[:, 1], c=yte, cmap="RdYlGn", s=10)
    plt.colorbar(sc, ax=axes[2], label="future engagement")
    axes[2].set_title("LSTM hidden-state space (PCA)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.savefig(out_dir / "trend_student_lstm.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(df: pd.DataFrame, confusion: np.ndarray, out_dir: Path, logger: logging.Logger) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"student_level_real": student_level(df, out_dir, logger), "class_level_simulated": class_level(confusion, out_dir, logger)}
    (out_dir / "trend_models.json").write_text(json.dumps(res, indent=2))
    return res
