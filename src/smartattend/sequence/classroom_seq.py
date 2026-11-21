"""Requirement 9 on real classroom video: LSTM vs Transformer over time, attention across students, latent spaces,
and step / sigmoid / tanh inside the LSTM cell.

    python -m smartattend sequence

Input to every model here is what the APP produces per frame (fine-tuned YOLOv8n boxes -> ResNet18 state probabilities),
target is the LABELLED share of attentive students. Training frames come from the validation-classroom videos
(taimoor-class-2/3/4, early stopping on taimoor-class-5), which the state classifier was never trained on, so the
inputs look like deployment inputs. The test classroom (taimoor-zaeem, 85 consecutive frames) is used once.
Hyperparameters are fixed in advance (no tuning on test); every learned model is run with 5 seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from smartattend.sequence.lstm_trend import ACTS, FlexLSTM, TrendLSTM, fit_regressor, predict
from smartattend.sequence.transformer_agg import DeepSets, TransformerAggregator

L = 8                       # frames of history
MAX_STUDENTS = 32
TRAIN_VIDEOS, STOP_VIDEO = ["taimoor-class-2", "taimoor-class-3", "taimoor-class-4"], "taimoor-class-5"
FRAME_FEATURES = ["mean p(attentive)", "mean p(distracted)", "mean p(inactive)", "share predicted attentive",
                  "share predicted distracted", "students / 20", "std p(attentive)"]


# --------------------------------------------------------------------------- features from the app's outputs
def frame_features(cfg: dict, split: str, log=print) -> list[dict]:
    """Per frame: every detected student's [p_att, p_dis, p_ina, cx, cy, w, h] and the labelled attentive share."""
    from smartattend.vision import classroom_eval as ce

    P = lambda k: Path(cfg["paths"][k])
    cache = P("experiments_dir") / "sequence" / f"classroom_frames_{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    processed, models_dir = P("data_processed") / "kaggle_classroom", P("models_dir")
    recs = ce.load_records(processed, split)
    pred_cache = P("experiments_dir") / "classroom" / f"preds_yolov8n-kaggle_{split}.json"
    det = None
    if not pred_cache.exists():
        from smartattend.vision.detectors import make_detector

        det = make_detector("yolov8n-kaggle", models_dir)
    preds, _ = ce.predictions(det, recs, pred_cache)
    thr = json.loads((P("eval_dir") / "classroom_eval.json").read_text())["detection"]["yolov8n-kaggle"]["count_threshold_from_val"]
    net = ce._net("resnet18")
    net.load_state_dict(torch.load(models_dir / "classroom_states_resnet18.pt", map_location="cpu"))
    out = []
    for r, (b, s) in zip(recs, preds):
        img = cv2.imread(str(r["path"]))
        H, W = img.shape[:2]
        keep = [k for k in b[s >= thr] if k[2] - k[0] >= 4 and k[3] - k[1] >= 4][:MAX_STUDENTS]
        crops = [cv2.resize(img[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1), ::-1], (160, 160)) for x0, y0, x1, y1 in keep]
        probs = ce.predict_cnn(net, np.stack(crops)) if crops else np.zeros((0, 3))
        geo = [[(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H] for x0, y0, x1, y1 in keep]
        out.append({"video": r["video"], "frame": r["frame"], "path": str(r["path"]),
                    "boxes": [list(map(float, k)) for k in keep],
                    "students": np.c_[probs, np.array(geo).reshape(-1, 4)].round(4).tolist(),
                    "true_share": float(np.mean([x == "attentive" for x in r["states"]])) if r["states"] else None,
                    "true_count": len(r["states"])})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    log(f"[sequence] {split}: {len(out)} frames")
    return out


def frame_vector(f: dict) -> np.ndarray:
    S = np.array(f["students"]).reshape(-1, 7)
    if not len(S):
        return np.zeros(7, np.float32)
    p = S[:, :3]
    hard = p.argmax(1)
    return np.array([p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean(), (hard == 0).mean(), (hard == 1).mean(),
                     len(S) / 20, p[:, 0].std()], np.float32)


def windows(frames: list[dict], videos: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Sliding windows of L frames inside each video (the start is padded by repeating the first frame)."""
    X, y, meta = [], [], []
    for v in sorted({f["video"] for f in frames}):
        if videos is not None and v not in videos:
            continue
        fs = sorted([f for f in frames if f["video"] == v and f["true_share"] is not None], key=lambda f: f["frame"])
        vec = np.stack([frame_vector(f) for f in fs])
        for t in range(len(fs)):
            idx = np.clip(np.arange(t - L + 1, t + 1), 0, None)
            X.append(vec[idx])
            y.append(fs[t]["true_share"])
            meta.append(fs[t])
    return np.stack(X), np.array(y, np.float32), meta


def set_arrays(frames: list[dict], videos: list[str] | None = None):
    fs = [f for f in frames if f["true_share"] is not None and (videos is None or f["video"] in videos)]
    X = np.zeros((len(fs), MAX_STUDENTS, 7), np.float32)
    pad = np.ones((len(fs), MAX_STUDENTS), bool)
    for i, f in enumerate(fs):
        S = np.array(f["students"], np.float32).reshape(-1, 7)[:MAX_STUDENTS]
        X[i, :len(S)], pad[i, :max(len(S), 1)] = S, False        # a frame with no detections keeps one zero "student"
    return X, pad, np.array([f["true_share"] for f in fs], np.float32), fs


# --------------------------------------------------------------------------- models
class TemporalTransformer(nn.Module):
    """One self-attention block over the L frames (learned position embedding); the last frame's token is read out.
    Written with nn.MultiheadAttention so the attention weights can be returned and plotted."""

    def __init__(self, n_in: int = 7, d: int = 16, heads: int = 2, length: int = L):
        super().__init__()
        self.embed = nn.Linear(n_in, d)
        self.pos = nn.Parameter(torch.zeros(1, length, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, d))
        self.out = nn.Linear(d, 1)

    def forward(self, x, return_attn: bool = False):
        h = self.embed(x) + self.pos
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        h = self.n1(h + a)
        h = self.n2(h + self.ff(h))
        y = self.out(h[:, -1])
        return (y, w[:, -1]) if return_attn else y            # attention of the current frame over the past


def fit_set_model(model, tr, va, epochs: int = 100, lr: float = 2e-3, seed: int = 0):
    torch.manual_seed(seed)
    Xtr, Ptr, ytr = (torch.tensor(a) for a in tr)
    Xva, Pva, yva = (torch.tensor(a) for a in va)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, state = np.inf, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            loss = nn.functional.mse_loss(model(Xtr[idx], Ptr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float((model(Xva, Pva) - yva).abs().mean())
        if v < best:
            best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval()


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 0, 1)
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 1e-9 else 0.0
    return {"mae": float(np.mean(np.abs(y - p))), "corr": r}


def gradient_through_time(act: str, T: int = 40, trials: int = 30, hidden: int = 16) -> np.ndarray:
    """|d h_T / d x_t| for each earlier step t, averaged over random cells and inputs: how far back learning can reach."""
    out = np.zeros(T)
    for s in range(trials):
        torch.manual_seed(s)
        cell = FlexLSTM(7, hidden, act)
        x = torch.randn(1, T, 7, requires_grad=True)
        cell(x)[:, -1].sum().backward()
        out += x.grad[0].norm(dim=1).numpy()
    return out / trials


def keras_lstm(Xtr, ytr, Xva, yva, Xte, seed: int = 0, epochs: int = 150):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([tf.keras.Input(shape=Xtr.shape[1:]), tf.keras.layers.LSTM(16), tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="mse")
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=64, verbose=0, callbacks=[stop])
    return m.predict(Xte, verbose=0).ravel()


def leave_one_video_out(frames: list[dict], seeds=(0, 1, 2)) -> dict:
    """Train on three videos of the validation classroom, predict the fourth, for every choice of the fourth.
    Shows whether a model generalises to a NEW video without touching the test classroom."""
    from sklearn.linear_model import Ridge

    videos = [*TRAIN_VIDEOS, STOP_VIDEO]
    per_video, out = {}, {k: [] for k in ["current frame only", "ridge regression", "LSTM (tanh)", "Transformer"]}
    for held in videos:
        Xtr, ytr, _ = windows(frames, [v for v in videos if v != held])
        Xte, yte, _ = windows(frames, [held])
        mae = lambda p: float(np.mean(np.abs(yte - np.clip(p, 0, 1))))
        row = {"true_mean_share": float(yte.mean()), "frames": int(len(yte)), "current frame only": mae(Xte[:, -1, 3]),
               "ridge regression": mae(Ridge(alpha=1.0).fit(Xtr.reshape(len(Xtr), -1), ytr).predict(Xte.reshape(len(Xte), -1)))}
        for name, make in [("LSTM (tanh)", lambda: TrendLSTM(7, 16, "tanh")), ("Transformer", TemporalTransformer)]:
            vals = []
            for s in seeds:
                torch.manual_seed(s)
                m, _ = fit_regressor(make(), Xtr, ytr, Xtr, ytr, epochs=60, seed=s)   # no held-out data for early stopping
                vals.append(mae(predict(m, Xte)))
            row[name] = float(np.mean(vals))
        per_video[held] = row
        for k in out:
            out[k].append(row[k])
    return {"per_video": per_video, "mean_mae": {k: float(np.mean(v)) for k, v in out.items()}}


# --------------------------------------------------------------------------- run everything
def run(cfg: dict, log=print, seeds=(0, 1, 2, 3, 4), with_keras: bool = True) -> tuple[list[str], dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsClassifier

    from smartattend.vision import classroom_eval as ce

    P = lambda k: Path(cfg["paths"][k])
    fig_dir = P("figures_dir") / "sequence"
    fig_dir.mkdir(parents=True, exist_ok=True)
    val, test = frame_features(cfg, "val", log), frame_features(cfg, "test", log)
    Xtr, ytr, _ = windows(val, TRAIN_VIDEOS)
    Xva, yva, _ = windows(val, [STOP_VIDEO])
    Xte, yte, meta_te = windows(test)
    result = {"windows": {"train": len(Xtr), "early_stop": len(Xva), "test": len(Xte)}, "history_frames": L}
    agg = lambda runs: {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in ["mae", "corr"]}

    # ---- 9a. over time: baselines, LSTM (3 cells), Transformer, Keras LSTM
    temporal = {"current frame only (app output)": scores(yte, Xte[:, -1, 3])}
    ks = [2, 3, 5, 8]
    k_best = min(ks, key=lambda k: np.mean(np.abs(ytr - Xtr[:, -k:, 3].mean(1))))        # chosen on training videos
    temporal[f"moving average of last {k_best} frames"] = scores(yte, Xte[:, -k_best:, 3].mean(1))
    ridge = Ridge(alpha=1.0).fit(Xtr.reshape(len(Xtr), -1), ytr)
    temporal["ridge regression on the window"] = scores(yte, ridge.predict(Xte.reshape(len(Xte), -1)))
    preds, lstm_models = {}, {}
    for act in ["tanh", "sigmoid", "step"]:
        runs = []
        for s in seeds:
            torch.manual_seed(s)
            m, _ = fit_regressor(TrendLSTM(7, 16, act), Xtr, ytr, Xva, yva, epochs=150, seed=s)
            p = predict(m, Xte)
            runs.append(scores(yte, p))
            if s == seeds[0]:
                preds[f"LSTM ({act})"], lstm_models[act] = np.clip(p, 0, 1), m
        temporal[f"LSTM ({act} cell)"] = agg(runs)
    runs, attn_time = [], None
    for s in seeds:
        torch.manual_seed(s)
        m, _ = fit_regressor(TemporalTransformer(), Xtr, ytr, Xva, yva, epochs=150, seed=s)
        p = predict(m, Xte)
        runs.append(scores(yte, p))
        if s == seeds[0]:
            preds["Transformer"] = np.clip(p, 0, 1)
            with torch.no_grad():
                attn_time = m(torch.tensor(Xte), return_attn=True)[1].numpy()
    temporal["Transformer (1 block, 2 heads)"] = agg(runs)
    if with_keras:
        try:
            temporal["Keras LSTM (TensorFlow, tanh)"] = agg([scores(yte, keras_lstm(Xtr, ytr, Xva, yva, Xte, s)) for s in seeds[:3]])
        except ImportError:
            pass
    result["temporal"] = temporal
    lovo = leave_one_video_out(val)
    result["leave_one_video_out"] = lovo
    result["training_share_by_video"] = {v: {"frames": int(sum(f["video"] == v for f in val)),
                                             "mean_true_share": float(np.mean([f["true_share"] for f in val if f["video"] == v]))}
                                         for v in [*TRAIN_VIDEOS, STOP_VIDEO]}
    log("[sequence] temporal models done")

    # ---- 9b. attention across the students of one frame
    tr_set, va_set, te_set = set_arrays(val, TRAIN_VIDEOS), set_arrays(val, [STOP_VIDEO]), set_arrays(test)
    Xs, Ps, ys, fs_te = te_set
    sets = {"mean p(attentive)": scores(ys, np.array([frame_vector(f)[0] for f in fs_te])),
            "share predicted attentive": scores(ys, np.array([frame_vector(f)[3] for f in fs_te]))}
    set_runs = {"DeepSets (no attention)": [], "Transformer over students": []}
    attn_model = None
    for s in seeds:
        for name, make in [("DeepSets (no attention)", lambda: DeepSets(7, 32)),
                           ("Transformer over students", lambda: TransformerAggregator(7, 32, 4, 2))]:
            torch.manual_seed(s)
            m = fit_set_model(make(), tr_set[:3], va_set[:3], seed=s)
            with torch.no_grad():
                set_runs[name].append(scores(ys, m(torch.tensor(Xs), torch.tensor(Ps)).numpy()))
            if name.startswith("Transformer") and s == seeds[0]:
                attn_model = m
    sets.update({k: agg(v) for k, v in set_runs.items()})
    with torch.no_grad():
        _, A = attn_model(torch.tensor(Xs), torch.tensor(Ps), return_attn=True)
    A = A.numpy()
    # what does the model look at? correlation of attention with each student feature (real students only)
    real = ~Ps
    feat_names = ["p(attentive)", "p(distracted)", "p(inactive)", "x", "y", "width", "height"]
    attn_corr = {n: float(np.corrcoef(A[real], Xs[real][:, j])[0, 1]) for j, n in enumerate(feat_names)}
    result["set_models"], result["student_attention_corr"] = sets, attn_corr
    n_real = real.sum(1)
    result["student_attention_spread"] = {
        "max_over_uniform": float(np.mean([A[k][real[k]].max() * n_real[k] for k in range(len(A))])),
        "min_over_uniform": float(np.mean([A[k][real[k]].min() * n_real[k] for k in range(len(A))]))}
    result["temporal_attention_mean_per_step"] = attn_time.mean(0).tolist()
    log("[sequence] set models done")

    # ---- 9c. latent spaces
    net = ce._net("resnet18")
    net.load_state_dict(torch.load(P("models_dir") / "classroom_states_resnet18.pt", map_location="cpu"))
    net.fc = nn.Identity()
    Xc, yc = ce._crop_arrays(P("data_processed") / "kaggle_classroom", "test")
    net.eval()
    with torch.no_grad():
        E = np.concatenate([net(ce._to_tensor(Xc[i:i + 64])).numpy() for i in range(0, len(Xc), 64)])
    Z = PCA(2, random_state=0).fit_transform(E)
    order = np.random.default_rng(0).permutation(len(E))
    half = len(E) // 2
    knn = KNeighborsClassifier(5).fit(E[order[:half]], yc[order[:half]])
    knn_acc = float(knn.score(E[order[half:]], yc[order[half:]]))
    majority = float(np.bincount(yc).max() / len(yc))
    with torch.no_grad():
        Hs = lstm_models["tanh"].hidden_states(torch.tensor(Xte))[:, -1].numpy()
    Zh = PCA(2, random_state=0).fit(Hs)
    Hz = Zh.transform(Hs)
    pc1_corr = float(abs(np.corrcoef(Hz[:, 0], yte)[0, 1]))
    result["latent"] = {"resnet_embedding_5nn_accuracy": knn_acc, "majority_class_share": majority,
                        "lstm_hidden_pc1_variance": float(Zh.explained_variance_ratio_[0]),
                        "lstm_hidden_pc1_corr_with_true_share": pc1_corr}

    # ---- 9d. step / sigmoid / tanh
    grads = {a: gradient_through_time(a) for a in ACTS}
    single = {}
    old = P("experiments_dir") / "sequence" / "trend_models.json"
    if old.exists():
        sl = json.loads(old.read_text())["student_level_real"]
        single = {a: [sl[f"lstm_{a}"]["mean"], sl[f"lstm_{a}"]["std"]] for a in ACTS} | {"last_value": sl["last_value"]}
    result["activations"] = {"classroom_mae": {a: temporal[f"LSTM ({a} cell)"]["mae"] for a in ACTS},
                             "classroom_corr": {a: temporal[f"LSTM ({a} cell)"]["corr"] for a in ACTS},
                             "single_person_forecast_mae": single,
                             "gradient_reaching_20_steps_back": {a: float(g[-21] / g[-1]) for a, g in grads.items()}}

    # ---- text report
    fmt = lambda v: (f"{v['mae'][0]:.3f} ± {v['mae'][1]:.3f}   {v['corr'][0]:+.2f} ± {v['corr'][1]:.2f}" if isinstance(v["mae"], list)
                     else f"{v['mae']:.3f}           {v['corr']:+.2f}")
    lines = ["Requirement 9 on real classroom video (test classroom: 85 consecutive frames)",
             f"training windows {len(Xtr)} (3 videos), early-stopping windows {len(Xva)}, test windows {len(Xte)}; history {L} frames", "",
             "9a) Attentive share of the class, estimated from the app's per-frame outputs",
             f"   {'model':40s} {'MAE':>15s} {'correlation':>14s}"]
    lines += [f"   {k:40s} {fmt(v)}" for k, v in temporal.items()]
    lines += ["   leave-one-video-out on the validation classroom (MAE on the held-out video):",
              "     " + f"{'held-out video (frames, true share)':38s}" + "".join(f"{k:>20s}" for k in lovo["mean_mae"])]
    for v, row in lovo["per_video"].items():
        label = f"{v} ({row['frames']}, {row['true_mean_share']:.2f})"
        lines.append("     " + f"{label:38s}" + "".join(f"{row[k]:>20.3f}" for k in lovo["mean_mae"]))
    lines.append("     " + f"{'mean of the 4 videos':38s}" + "".join(f"{v:>20.3f}" for v in lovo["mean_mae"].values()))
    lines += ["", "9b) One frame, many students: models that read the set of students",
              f"   {'model':40s} {'MAE':>15s} {'correlation':>14s}"]
    lines += [f"   {k:40s} {fmt(v)}" for k, v in sets.items()]
    sp = result["student_attention_spread"]
    lines += [f"   attention per student ranges from {sp['min_over_uniform']:.2f}x to {sp['max_over_uniform']:.2f}x the uniform share (mean over test frames)"]
    lines += ["   attention weight vs student feature (correlation): " + ", ".join(f"{k} {v:+.2f}" for k, v in attn_corr.items()),
              "   Transformer over time, attention of the current frame on frames t-7 ... now: "
              + " ".join(f"{v:.3f}" for v in attn_time.mean(0)) + "  (uniform = 0.125)",
              "", "9c) Latent spaces",
              f"   ResNet18 512-d embedding of test students: 5-nearest-neighbour accuracy {knn_acc:.3f} "
              f"(always guessing the largest class: {majority:.3f})",
              f"   LSTM hidden state: first principal component holds {Zh.explained_variance_ratio_[0]:.0%} of the variance, "
              f"|correlation| with the labelled share {pc1_corr:.2f}",
              "", "9d) step / sigmoid / tanh inside the LSTM cell",
              "   classroom task (MAE, correlation): " + "; ".join(
                  f"{a} {temporal[f'LSTM ({a} cell)']['mae'][0]:.3f}, {temporal[f'LSTM ({a} cell)']['corr'][0]:+.2f}" for a in ACTS)]
    if single:
        lines.append("   single-person forecast (earlier stage, MAE): " + "; ".join(f"{a} {single[a][0]:.3f} ± {single[a][1]:.3f}" for a in ACTS)
                     + f"; last-value baseline {single['last_value']:.3f}")
    lines.append("   gradient that reaches 20 steps back (relative to the last step): "
                 + "; ".join(f"{a} {v:.1e}" for a, v in result["activations"]["gradient_reaching_20_steps_back"].items()))

    # ---- figures
    paths = []
    x = np.arange(len(yte))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].plot(x, yte, "k-", lw=2, label="labelled")
    axes[0].plot(x, Xte[:, -1, 3], color="C7", lw=1, label="current frame only")
    axes[0].plot(x, preds["LSTM (tanh)"], color="C0", label="LSTM (tanh)")
    axes[0].plot(x, preds["Transformer"], color="C3", label="Transformer")
    axes[0].set(title="Attentive share over the test video", xlabel="frame", ylim=(0, 1)), axes[0].legend(fontsize=8), axes[0].grid(alpha=0.3)
    names = list(temporal)
    corr = [v["corr"][0] if isinstance(v["corr"], list) else v["corr"] for v in temporal.values()]
    err = [v["corr"][1] if isinstance(v["corr"], list) else 0 for v in temporal.values()]
    axes[1].barh(names, corr, xerr=err, color=["C7"] * 3 + ["C0"] * 3 + ["C3"] + ["C2"] * (len(names) - 7), capsize=3)
    axes[1].invert_yaxis(), axes[1].set(title="Correlation with the labelled trend (test)", xlim=(0, 1)), axes[1].grid(alpha=0.3, axis="x")
    axes[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    paths.append(fig_dir / "classroom_lstm_transformer.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].bar(np.arange(L), attn_time.mean(0), yerr=attn_time.std(0), color="C3", capsize=3)
    axes[0].set_xticks(np.arange(L), [f"t-{L - 1 - i}" if i < L - 1 else "now" for i in range(L)])
    axes[0].set(title="Transformer over time: where the current frame looks", ylabel="attention weight (mean ± std over test frames)")
    axes[0].axhline(1 / L, color="k", ls=":", lw=0.8)
    i = int(np.argmax([len(f["students"]) for f in fs_te]))
    img = cv2.imread(fs_te[i]["path"])[:, :, ::-1].copy()
    a_raw = A[i][:len(fs_te[i]["boxes"])]
    a = (a_raw - a_raw.min()) / (a_raw.max() - a_raw.min() + 1e-9)        # min-max so the differences are visible
    p_dis = np.array(fs_te[i]["students"])[:, 1]
    for (x0, y0, x1, y1), w, raw, pd_ in zip(np.array(fs_te[i]["boxes"]).astype(int), a, a_raw, p_dis):
        col = (int(255 * w), 40, int(255 * (1 - w)))
        cv2.rectangle(img, (x0, y0), (x1, y1), col, max(2, int(2 + 7 * w)))
        cv2.putText(img, f"{raw:.0%}", (x0 + 4, max(28, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3)
    axes[1].imshow(img), axes[1].axis("off")
    axes[1].set_title(f"Transformer over students: share of attention per student (red = most, blue = least; "
                      f"uniform would be {1 / len(a_raw):.0%})", fontsize=10)
    fig.tight_layout()
    paths.append(fig_dir / "classroom_attention.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c, (name, col) in enumerate(zip(ce.STATES, ["C2", "C1", "C3"])):
        axes[0].scatter(*Z[yc == c].T, s=8, alpha=0.5, color=col, label=f"{name} ({(yc == c).sum()})")
    axes[0].set(title=f"ResNet18 embedding of test students (PCA of 512 numbers)\n5-NN accuracy {knn_acc:.2f} vs {majority:.2f} baseline",
                xlabel="PC 1", ylabel="PC 2"), axes[0].legend(fontsize=8)
    sc = axes[1].scatter(Hz[:, 0], Hz[:, 1], c=yte, cmap="RdYlGn", vmin=0, vmax=1, s=30, edgecolor="k", lw=0.3)
    axes[1].plot(Hz[:, 0], Hz[:, 1], color="gray", lw=0.5, alpha=0.6)
    axes[1].set(title=f"LSTM memory over the test video (PCA of 16 numbers)\nPC 1 vs labelled share: |r| = {pc1_corr:.2f}",
                xlabel="PC 1", ylabel="PC 2")
    fig.colorbar(sc, ax=axes[1], label="labelled attentive share")
    fig.tight_layout()
    paths.append(fig_dir / "classroom_latent_space.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    z = torch.linspace(-4, 4, 400)
    for a, col in zip(ACTS, ["C0", "C1", "C3"]):
        axes[0].plot(z, ACTS[a](z), color=col, label=a)
        g = grads[a]
        axes[1].semilogy(np.arange(len(g))[::-1], np.maximum(g / g[-1], 1e-12), color=col, label=a)
    axes[0].set(title="The three cell non-linearities"), axes[0].legend(), axes[0].grid(alpha=0.3)
    axes[1].set(title="Gradient reaching earlier frames (random cells)", xlabel="steps back in time",
                ylabel="relative size (log)"), axes[1].legend(), axes[1].grid(alpha=0.3)
    acts = list(ACTS)
    w = 0.38
    axes[2].bar(np.arange(3) - w / 2, [temporal[f"LSTM ({a} cell)"]["mae"][0] for a in acts], w,
                yerr=[temporal[f"LSTM ({a} cell)"]["mae"][1] for a in acts], capsize=3, label="classroom (this module)")
    if single:
        axes[2].bar(np.arange(3) + w / 2, [single[a][0] for a in acts], w, yerr=[single[a][1] for a in acts], capsize=3,
                    label="single-person forecast")
    axes[2].set_xticks(np.arange(3), acts), axes[2].set(title="Test MAE by cell non-linearity (lower is better)")
    axes[2].legend(fontsize=8), axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    paths.append(fig_dir / "lstm_activations.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    result["figures"] = [str(p) for p in paths]
    out = P("eval_dir") / "sequence_models.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    lines += ["", f"saved {out}"] + [f"figure: {p}" for p in paths]
    return lines, result
