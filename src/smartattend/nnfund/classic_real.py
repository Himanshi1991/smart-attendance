"""Requirement 6 on real data: Hebbian, modified Hebbian (Oja / Sanger), competitive, error-correction learning and a
Restricted Boltzmann Machine, each applied to the project's own data and checked against a standard method.

    python -m smartattend classic

  6a Hebbian vs Oja vs Sanger  -> "eigenfaces" of the 32x32 student face frames, checked against SVD (Requirement 3)
  6b competitive learning      -> group unknown faces by identity from 128-d face embeddings (LFW), vs k-means
  6c error correction          -> perceptron rule and delta (LMS) rule: engaged / neutral / disengaged faces, vs the MLP
  6d Boltzmann machine         -> RBM learns face features with CD-1; free energy separates real faces from noise
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smartattend.nnfund.boltzmann import RBM

CLASSES = ["engaged", "neutral", "disengaged"]
RBM_HIDDEN, RBM_EPOCHS = 128, 50


# --------------------------------------------------------------------------- 6a Hebbian family
def hebbian_run(X: np.ndarray, lr: float, epochs: int, seed: int = 0, blowup: float = 1e12):
    """Plain Hebb dw = lr*y*x. Records |w| per epoch and stops once it explodes (it always does)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.01, size=X.shape[1])
    norms = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(epochs):
            for x in X[rng.permutation(len(X))]:
                w = w + lr * (w @ x) * x
            norms.append(float(np.linalg.norm(w)))
            if not np.isfinite(norms[-1]) or norms[-1] > blowup:
                break
    return w, np.array(norms)


def oja_run(X: np.ndarray, lr: float, epochs: int, seed: int = 0):
    """Oja dw = lr*y*(x - y*w): Hebb plus a decay term that keeps |w| = 1; converges to the first principal component."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.01, size=X.shape[1])
    norms = []
    for _ in range(epochs):
        for x in X[rng.permutation(len(X))]:
            y = w @ x
            w += lr * y * (x - y * w)
        norms.append(float(np.linalg.norm(w)))
    return w, np.array(norms)


def sanger(X: np.ndarray, k: int, lr: float, epochs: int, seed: int = 0) -> np.ndarray:
    """Generalised Hebbian Algorithm: dW = lr*(y x^T - LT(y y^T) W). Row i converges to principal component i."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.01, size=(k, X.shape[1]))
    for epoch in range(epochs):
        step = lr / (1 + epoch)                     # decaying rate so the rows settle
        for x in X[rng.permutation(len(X))]:
            y = W @ x
            W += step * (np.outer(y, x) - np.tril(np.outer(y, y)) @ W)
    return W


def eigenfaces(X: np.ndarray, epochs: int = 15, k: int = 8, seed: int = 0) -> dict:
    """X: N x 1024 face pixels in [0,1]. Centre with the mean face, scale so lr is size-independent."""
    Xc = X - X.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(Xc ** 2, axis=1)))
    Xs = Xc / scale                                  # average |x| = 1
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    w_hebb, n_hebb = hebbian_run(Xs, lr=0.01, epochs=epochs, seed=seed)
    w_oja, n_oja = oja_run(Xs, lr=0.01, epochs=epochs, seed=seed)
    W = sanger(Xs, k=k, lr=0.1, epochs=40, seed=seed)
    cos = lambda a, b: float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return {"mean_face": X.mean(axis=0), "svd": Vt[:k], "variance_ratio": var[:k],
            "hebb": w_hebb, "hebb_norms": n_hebb, "oja": w_oja, "oja_norms": n_oja, "sanger": W,
            "cos_oja_pc1": cos(w_oja, Vt[0]), "cos_hebb_pc1": cos(np.nan_to_num(w_hebb), Vt[0]),
            "cos_sanger": [cos(W[i], Vt[i]) for i in range(k)],
            "sanger_row_norms": np.linalg.norm(W, axis=1).tolist()}


# --------------------------------------------------------------------------- 6b competitive learning
def competitive_cosine(X: np.ndarray, k: int, lr: float = 0.1, epochs: int = 20, seed: int = 0) -> np.ndarray:
    """Winner-take-all on unit vectors: the most similar prototype moves toward x and is re-normalised."""
    rng = np.random.default_rng(seed)
    W = X[rng.choice(len(X), size=k, replace=False)].copy()
    for epoch in range(epochs):
        step = lr * (1 - epoch / epochs) + 0.005
        for x in X[rng.permutation(len(X))]:
            j = int(np.argmax(W @ x))
            W[j] += step * (x - W[j])
            W[j] /= np.linalg.norm(W[j])
    return W


def competitive_restarts(X: np.ndarray, k: int, restarts: int = 10, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Run competitive learning several times; keep the run whose prototypes fit the data best (highest mean
    similarity of each face to its winner). Labels are NOT used, so this is a fair match for k-means' n_init=10."""
    best, fits = None, []
    for r in range(restarts):
        W = competitive_cosine(X, k, seed=seed * 1000 + r)
        fits.append(float(np.mean(np.max(X @ W.T, axis=1))))
        if fits[-1] == max(fits):
            best = W
    return best, fits


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement of two groupings, 1 = identical, about 0 = random (Hubert & Arabie 1985)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(M, (ai, bi), 1)
    c2 = lambda n: n * (n - 1) / 2
    s_ij, s_a, s_b = c2(M).sum(), c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    expected = s_a * s_b / c2(len(a))
    return float((s_ij - expected) / (0.5 * (s_a + s_b) - expected))


def purity(true: np.ndarray, pred: np.ndarray) -> float:
    """Share of faces whose cluster's most common identity is their own."""
    total = 0
    for c in np.unique(pred):
        total += np.bincount(np.unique(true, return_inverse=True)[1][pred == c]).max()
    return total / len(true)


def class_of_faces(emb_npz: str | Path, n_people: int = 30, per_person: int = 20, seed: int = 0):
    """A simulated class: n_people LFW identities with >= per_person detected faces, per_person photos each."""
    z = np.load(emb_npz)
    keep = z["detected"]
    ids, E = z["identity"][keep], z["emb"][keep]
    names, counts = np.unique(ids, return_counts=True)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(names[counts >= per_person], size=n_people, replace=False)
    idx = np.concatenate([rng.choice(np.flatnonzero(ids == n), per_person, replace=False) for n in chosen])
    E = E[idx] / np.linalg.norm(E[idx], axis=1, keepdims=True)
    return E, np.repeat(np.arange(n_people), per_person), chosen


def clustering(emb_npz: str | Path, seeds=(0, 1, 2, 3, 4), n_people: int = 30) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    rows = {"competitive": [], "competitive_10_restarts": [], "kmeans": [], "random": []}
    for s in seeds:
        E, y, _ = class_of_faces(emb_npz, n_people=n_people, seed=s)
        pred = np.argmax(E @ competitive_cosine(E, k=n_people, seed=s).T, axis=1)
        pred10 = np.argmax(E @ competitive_restarts(E, k=n_people, seed=s)[0].T, axis=1)
        km = KMeans(n_clusters=n_people, n_init=10, random_state=s).fit_predict(E)
        rnd = np.random.default_rng(s).integers(0, n_people, len(y))
        for name, p in [("competitive", pred), ("competitive_10_restarts", pred10), ("kmeans", km), ("random", rnd)]:
            rows[name].append({"ari": adjusted_rand_index(y, p), "purity": purity(y, p),
                               "clusters_used": int(len(np.unique(p))), "ari_sklearn": float(adjusted_rand_score(y, p))})
    out = {}
    for name, r in rows.items():
        out[name] = {k: [float(np.mean([x[k] for x in r])), float(np.std([x[k] for x in r]))]
                     for k in ["ari", "purity", "clusters_used"]}
        out[name]["max_ari_diff_vs_sklearn"] = float(max(abs(x["ari"] - x["ari_sklearn"]) for x in r))
    return out


# --------------------------------------------------------------------------- 6c error correction
def _standardize(tr, *others):
    mu, sd = tr.mean(0), np.maximum(tr.std(0), 0.05)
    return [(a - mu) / sd for a in (tr, *others)]


def perceptron(X, y, n_classes, epochs=30, lr=1.0, seed=0):
    """Multi-class perceptron: on a mistake, w_true += x and w_predicted -= x (only errors cause updates)."""
    rng = np.random.default_rng(seed)
    Xb = np.column_stack([X, np.ones(len(X))])
    W = np.zeros((n_classes, Xb.shape[1]))
    mistakes = []
    for _ in range(epochs):
        m = 0
        for i in rng.permutation(len(X)):
            p = int(np.argmax(W @ Xb[i]))
            if p != y[i]:
                W[y[i]] += lr * Xb[i]
                W[p] -= lr * Xb[i]
                m += 1
        mistakes.append(m / len(X))
    return W, np.array(mistakes)


def delta_rule_multi(X, y, n_classes, epochs=30, lr=1e-4, seed=0):
    """Widrow-Hoff / LMS: linear outputs o = W x, target one-hot t, dW = lr * (t - o) x^T on every sample."""
    rng = np.random.default_rng(seed)
    Xb = np.column_stack([X, np.ones(len(X))])
    T = np.eye(n_classes)[y]
    W = np.zeros((n_classes, Xb.shape[1]))
    mse = []
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            W += lr * np.outer(T[i] - W @ Xb[i], Xb[i])
        mse.append(float(np.mean((Xb @ W.T - T) ** 2)))
    return W, np.array(mse)


def predict_linear(W, X):
    return np.argmax(np.column_stack([X, np.ones(len(X))]) @ W.T, axis=1)


# --------------------------------------------------------------------------- run everything
def run(faces_npz: str | Path, emb_npz: str | Path, eval_dir: str | Path, fig_dir: str | Path,
        mlp_reference: float | None = None) -> tuple[list[str], dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smartattend.eval.metrics import classification_report

    z = np.load(faces_npz)
    data = {s: (z[f"{s}_X"].reshape(len(z[f"{s}_X"]), -1), z[f"{s}_y"]) for s in ["train", "val", "test"]}
    Xtr, ytr = data["train"]
    Xte, yte = data["test"]
    lines, result = ["Requirement 6 on real data", ""], {}

    # 6a
    t0 = time.perf_counter()
    ef = eigenfaces(Xtr)
    result["hebbian"] = {"hebb_norm_per_epoch": ef["hebb_norms"].tolist(), "oja_norm_per_epoch": ef["oja_norms"].tolist(),
                         "cos_oja_vs_svd_pc1": ef["cos_oja_pc1"], "cos_sanger_vs_svd": ef["cos_sanger"],
                         "sanger_row_norms": ef["sanger_row_norms"], "variance_ratio": ef["variance_ratio"].tolist()}
    hn = ef["hebb_norms"]
    lines += ["6a) Hebbian learning on 1,293 training faces (32x32 = 1024 pixels, mean face removed)",
              f"   plain Hebb   |w| per epoch: {', '.join(f'{v:.1e}' for v in hn[:5])}{' ...' if len(hn) > 5 else ''}"
              f"  -> grows without limit (stopped after {len(hn)} epochs)",
              f"   Oja          |w| after {len(ef["oja_norms"])} epochs = {ef['oja_norms'][-1]:.4f};  |cos| with SVD's 1st component = "
              f"{ef['cos_oja_pc1']:.4f}",
              "   Sanger (8 components) |cos| with SVD components 1-8: " + " ".join(f"{c:.2f}" for c in ef["cos_sanger"]),
              "   variance explained by components 1-8: " + " ".join(f"{v:.1%}" for v in ef["variance_ratio"]),
              f"   ({time.perf_counter() - t0:.0f} s)"]

    # 6b
    t0 = time.perf_counter()
    cl = clustering(emb_npz)
    result["competitive"] = cl
    lines += ["", "6b) Competitive learning: 30 people x 20 photos (LFW face embeddings, 128 numbers), names hidden,",
              "    30 prototypes; mean ± std over 5 random classes",
              f"   {'method':22s} {'ARI':>15s} {'purity':>15s} {'clusters used':>14s}"]
    for name, lab in [("competitive", "competitive, 1 run"), ("competitive_10_restarts", "competitive, best of 10"),
                      ("kmeans", "k-means, best of 10"),
                      ("random", "random assignment")]:
        r = cl[name]
        lines.append(f"   {lab:22s} {r['ari'][0]:>7.3f} ± {r['ari'][1]:.3f} {r['purity'][0]:>7.3f} ± {r['purity'][1]:.3f} "
                     f"{r['clusters_used'][0]:>10.1f}")
    lines.append(f"   our ARI matches scikit-learn's adjusted_rand_score to {cl['kmeans']['max_ari_diff_vs_sklearn']:.0e}"
                 f"   ({time.perf_counter() - t0:.0f} s)")

    # 6c
    t0 = time.perf_counter()
    Str, Ste = _standardize(Xtr, Xte)
    Wp, mistakes = perceptron(Str, ytr, 3)
    Wd, mse = delta_rule_multi(Str, ytr, 3)
    rp = classification_report(yte, predict_linear(Wp, Ste), 3)
    rd = classification_report(yte, predict_linear(Wd, Ste), 3)
    result["error_correction"] = {"perceptron": {"test": rp, "train_mistake_rate": mistakes.tolist()},
                                  "delta_rule": {"test": rd, "train_mse": mse.tolist()}, "mlp_reference": mlp_reference}
    lines += ["", "6c) Error-correction learning: engaged / neutral / disengaged from 1024 face pixels (Requirement 5 split)",
              f"   perceptron rule (update only on mistakes): training mistakes {mistakes[0]:.1%} -> {mistakes[-1]:.1%};  "
              f"TEST accuracy {rp['accuracy']:.3f}  macro-F1 {rp['macro_f1']:.3f}",
              f"   delta / LMS rule (update by the size of the error): training MSE {mse[0]:.3f} -> {mse[-1]:.3f};  "
              f"TEST accuracy {rd['accuracy']:.3f}  macro-F1 {rd['macro_f1']:.3f}"]
    if mlp_reference is not None:
        lines.append(f"   for comparison, the MLP with hidden layers (Requirement 5 baseline): TEST macro-F1 {mlp_reference:.3f}")
    lines.append(f"   ({time.perf_counter() - t0:.0f} s)")

    # 6d RBM on faces (pixel intensities used as Bernoulli probabilities)
    t0 = time.perf_counter()
    rbm = RBM(1024, RBM_HIDDEN, seed=0).init_visible_bias(Xtr)
    hist = rbm.fit(Xtr, epochs=RBM_EPOCHS, lr=0.05, batch_size=20)
    rng = np.random.default_rng(0)
    shuffled = np.array([x[rng.permutation(1024)] for x in Xte])           # same pixels, face structure destroyed
    fe_real, fe_shuf = rbm.free_energy(Xte), rbm.free_energy(shuffled)
    auc = float(np.mean(fe_real[:, None] < fe_shuf[None, :]))
    Htr, Hte = rbm.p_h(Xtr), rbm.p_h(Xte)
    Hs_tr, Hs_te = _standardize(Htr, Hte)
    Wr, _ = perceptron(Hs_tr, ytr, 3)
    rr = classification_report(yte, predict_linear(Wr, Hs_te), 3)
    Wr2, _ = delta_rule_multi(Hs_tr, ytr, 3, lr=1e-3)
    rr2 = classification_report(yte, predict_linear(Wr2, Hs_te), 3)
    result["rbm"] = {"recon_mse_per_epoch": hist["recon_error"], "free_energy_real_mean": float(fe_real.mean()),
                     "free_energy_shuffled_mean": float(fe_shuf.mean()), "real_lower_than_shuffled_auc": auc,
                     "perceptron_on_rbm_features_test": rr, "delta_rule_on_rbm_features_test": rr2,
                     "hidden": RBM_HIDDEN, "epochs": RBM_EPOCHS, "lr": 0.05, "visible_bias_init": "log-odds of mean pixel"}
    lines += ["", f"6d) Restricted Boltzmann Machine: 1024 visible pixels, {RBM_HIDDEN} hidden units, CD-1, {RBM_EPOCHS} epochs",
              f"   reconstruction MSE {hist['recon_error'][0]:.4f} -> {hist['recon_error'][-1]:.4f}",
              f"   free energy: real test faces {fe_real.mean():.1f}  vs  the same pixels shuffled {fe_shuf.mean():.1f}  "
              f"(real is lower in {auc:.1%} of pairs)",
              f"   on the {RBM_HIDDEN} RBM features ({1024 // RBM_HIDDEN}x fewer inputs, learned WITHOUT labels): TEST macro-F1 "
              f"perceptron {rr['macro_f1']:.3f}, delta rule {rr2['macro_f1']:.3f}",
              f"   ({time.perf_counter() - t0:.0f} s)"]

    # figures
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    img = lambda v: v.reshape(32, 32)
    fig, axes = plt.subplots(3, 9, figsize=(15, 5.6))
    axes[0, 0].imshow(img(ef["mean_face"]), cmap="gray"), axes[0, 0].set_title("mean face", fontsize=9)
    axes[1, 0].semilogy(np.arange(1, len(hn) + 1), hn, "o-", label="Hebb")
    axes[1, 0].semilogy(np.arange(1, len(ef["oja_norms"]) + 1), ef["oja_norms"], label="Oja")
    axes[1, 0].set_title("|w| per epoch", fontsize=9), axes[1, 0].legend(fontsize=7)
    axes[2, 0].imshow(img(np.nan_to_num(ef["oja"])), cmap="RdBu_r")
    axes[2, 0].set_title(f"Oja w (cos {ef['cos_oja_pc1']:.2f})", fontsize=9)
    for i in range(8):
        sv, sg = ef["svd"][i], ef["sanger"][i]
        sg = sg * np.sign(sg @ sv)                                   # a component and its negative are the same
        axes[0, i + 1].imshow(img(sv), cmap="RdBu_r"), axes[0, i + 1].set_title(f"SVD {i + 1} ({ef['variance_ratio'][i]:.0%})", fontsize=9)
        axes[1, i + 1].imshow(img(sg), cmap="RdBu_r"), axes[1, i + 1].set_title(f"Sanger {i + 1} (cos {ef['cos_sanger'][i]:.2f})", fontsize=9)
    for i in range(8):
        axes[2, i + 1].imshow(img(rbm.W[:, i]), cmap="RdBu_r"), axes[2, i + 1].set_title(f"RBM unit {i + 1}", fontsize=9)
    for ax in axes.ravel():
        if ax is not axes[1, 0]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Eigenfaces: SVD (top) vs Sanger's Hebbian rule (middle); RBM hidden-unit weights (bottom)")
    fig.tight_layout()
    paths.append(fig_dir / "hebbian_eigenfaces.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    names = ["competitive\n1 run", "competitive\nbest of 10", "k-means\nbest of 10", "random"]
    vals = [cl[k]["ari"] for k in ["competitive", "competitive_10_restarts", "kmeans", "random"]]
    axes[0].bar(names, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=4, color=["C0", "C0", "C7", "C3"])
    axes[0].set_title("Grouping 30 unknown people (ARI, 1 = perfect)"), axes[0].set_ylim(0, 1)
    axes[1].plot(np.arange(1, len(mistakes) + 1), mistakes, label="perceptron: share of mistakes")
    axes[1].plot(np.arange(1, len(mse) + 1), mse, label="delta rule: MSE")
    axes[1].set_title("Error-correction learning (training)"), axes[1].set_xlabel("epoch"), axes[1].legend()
    axes[2].plot(np.arange(1, len(hist["recon_error"]) + 1), hist["recon_error"]), axes[2].set_title("RBM reconstruction error")
    axes[2].set_xlabel("epoch")
    axes[3].hist(fe_real, bins=30, alpha=0.7, label="real faces"), axes[3].hist(fe_shuf, bins=30, alpha=0.7, label="pixels shuffled")
    axes[3].set_title("RBM free energy (low = 'looks like the data')"), axes[3].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    paths.append(fig_dir / "competitive_errorcorr_rbm.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
    for i in range(8):
        x = Xte[i * 30]
        axes[0, i].imshow(img(x), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(img(rbm.reconstruct(x[None])[0]), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off"), axes[1, i].axis("off")
    axes[0, 0].set_title("test faces", fontsize=9, loc="left"), axes[1, 0].set_title("RBM reconstruction", fontsize=9, loc="left")
    paths.append(fig_dir / "rbm_face_reconstructions.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    result["figures"] = [str(p) for p in paths]
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "classic_learning.json").write_text(json.dumps(result, indent=2))
    lines += ["", f"saved {eval_dir / 'classic_learning.json'}"] + [f"figure: {p}" for p in paths]
    return lines, result
