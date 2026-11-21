"""Face-recognition evaluation on LFW: verification, simulated class sessions, fairness, robustness.

Demographic groups come from the Columbia LFW attribute classifier scores (Kumar et al. 2009):
gender = sign of the 'Male' score, ethnicity = argmax of the Asian/White/Black scores. These are
*predicted* attributes, not self-reported identity, and the groups are imbalanced; both caveats are
carried into the report.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from smartattend.eval.metrics import fairness_gap


# --------------------------------------------------------------------------- data
def load_attributes(path: str | Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        f.readline()
        header = f.readline().lstrip("#").strip().split("\t")
    a = pd.read_csv(path, sep="\t", skiprows=2, header=None, names=header)
    a["identity"] = a["person"].str.replace(" ", "_")
    a["gender"] = np.where(a["Male"] > 0, "male", "female")
    a["ethnicity"] = a[["Asian", "White", "Black"]].idxmax(axis=1).str.lower()
    return a[["identity", "imagenum", "gender", "ethnicity", "Male", "Asian", "White", "Black"]]


def _decode(b) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b["bytes"] if isinstance(b, dict) else b, np.uint8), cv2.IMREAD_COLOR)


def degrade(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Classroom-style corruptions used for robustness tests."""
    if kind == "clean":
        return img
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 2.5)
    if kind == "low_light":
        return cv2.convertScaleAbs(img, alpha=0.35, beta=0)
    if kind == "noise":
        return np.clip(img + rng.normal(0, 18, img.shape), 0, 255).astype(np.uint8)
    if kind == "small_face":  # back row: face ~ 40 px wide, upscaled back
        h, w = img.shape[:2]
        return cv2.resize(cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_AREA), (w, h))
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 15])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if kind == "mask_occlusion":  # lower face covered (mask / hand)
        out = img.copy()
        h, w = img.shape[:2]
        out[int(h * 0.58):int(h * 0.80), int(w * 0.30):int(w * 0.70)] = (200, 200, 210)
        return out
    raise ValueError(kind)


DEGRADATIONS = ["clean", "blur", "low_light", "noise", "small_face", "jpeg", "mask_occlusion"]


def embed_image(engine, img: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Embed the detected face closest to the image centre (LFW faces are centred)."""
    faces = engine.analyze(img)
    if not faces:
        return None, 0.0
    h, w = img.shape[:2]
    f = min(faces, key=lambda f: (f.box[0] + f.box[2] / 2 - w / 2) ** 2 + (f.box[1] + f.box[3] / 2 - h / 2) ** 2)
    return f.embedding, f.score


def compute_embeddings(engine, parquet: Path, cache: Path, ids: set[str] | None = None, max_per_id: int | None = None,
                       degradation: str = "clean", seed: int = 0, logger=None) -> pd.DataFrame:
    """Embeddings for LFW images (optionally selected identities, first max_per_id images each), cached."""
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return pd.DataFrame({"identity": d["identity"], "image_num": d["image_num"],
                             "detected": d["detected"], "emb": list(d["emb"])})
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(parquet, columns=["identity", "image_num", "image"])
    if ids is not None:
        df = df[df["identity"].isin(ids)]
    if max_per_id is not None:
        df = df.sort_values(["identity", "image_num"]).groupby("identity").head(max_per_id)
    embs, det = [], []
    for i, b in enumerate(df["image"]):
        e, _ = embed_image(engine, degrade(_decode(b), degradation, rng))
        det.append(e is not None)
        embs.append(e if e is not None else np.zeros(128))
        if logger and i % 2000 == 0:
            logger.info(f"[lfw:{degradation}] embedded {i}/{len(df)}")
    out = pd.DataFrame({"identity": df["identity"].to_numpy(), "image_num": df["image_num"].to_numpy(),
                        "detected": det, "emb": embs})
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, identity=out["identity"].to_numpy(dtype=str), image_num=out["image_num"].to_numpy(),
             detected=np.array(det), emb=np.stack(embs))
    return out


# --------------------------------------------------------------------------- verification
def verification_pairs(df: pd.DataFrame, n_impostor: int = 60000, seed: int = 0):
    """All genuine pairs of identities with >= 2 detected images + random impostor pairs."""
    rng = np.random.default_rng(seed)
    df = df[df["detected"]].reset_index(drop=True)
    E = np.stack(df["emb"].to_numpy())
    gen = []
    for _, idx in df.groupby("identity").indices.items():
        idx = idx[:30]  # cap very frequent identities (e.g. George_W_Bush) to avoid domination
        gen += [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]]
    gen = np.array(gen)
    imp = rng.integers(0, len(df), size=(n_impostor * 2, 2))
    imp = imp[df["identity"].to_numpy()[imp[:, 0]] != df["identity"].to_numpy()[imp[:, 1]]][:n_impostor]
    sim = lambda p: np.einsum("ij,ij->i", E[p[:, 0]], E[p[:, 1]])
    return df, gen, sim(gen), imp, sim(imp)


def rates_at(gen_sim, imp_sim, thr: float) -> dict:
    return {"FAR": float(np.mean(imp_sim >= thr)), "FRR": float(np.mean(gen_sim < thr)),
            "TAR": float(np.mean(gen_sim >= thr))}


def tar_at_far(gen_sim, imp_sim, far: float) -> tuple[float, float]:
    thr = float(np.quantile(imp_sim, 1 - far))
    return float(np.mean(gen_sim >= thr)), thr


# --------------------------------------------------------------------------- class sessions
def simulate_sessions(df: pd.DataFrame, attrs: pd.DataFrame, threshold: float, n_sessions: int = 200,
                      class_size: int = 30, n_absent: int = 5, n_strangers: int = 3, n_enroll: int = 3,
                      probes_per_student: int = 2, seed: int = 0) -> dict:
    """Each session: enrol `class_size` students from `n_enroll` photos each; present students appear in
    `probes_per_student` held-out photos (frames); absent students do not appear; strangers appear but are
    not enrolled. Uses the same greedy one-to-one matcher and >= min_frames rule as the product."""
    from smartattend.attendance.roster import Roster
    from smartattend.attendance.session import AttendanceSession

    rng = np.random.default_rng(seed)
    det = df[df["detected"]]
    groups = det.groupby("identity").indices
    eligible = [i for i, idx in groups.items() if len(idx) >= n_enroll + probes_per_student]
    others = [i for i, idx in groups.items() if len(idx) < n_enroll + probes_per_student]
    E = np.stack(det["emb"].to_numpy())
    gender = attrs.groupby("identity")["gender"].agg(lambda s: s.mode().iloc[0]).to_dict()
    ethnicity = attrs.groupby("identity")["ethnicity"].agg(lambda s: s.mode().iloc[0]).to_dict()

    rows, sess_rows = [], []
    for s in range(n_sessions):
        students = list(rng.choice(eligible, class_size, replace=False))
        strangers = list(rng.choice(others, n_strangers, replace=False))
        absent = set(rng.choice(students, n_absent, replace=False))
        cents = []
        probe_frames = [[] for _ in range(probes_per_student)]
        for sid in students:
            idx = rng.permutation(groups[sid])
            c = E[idx[:n_enroll]].mean(0)
            cents.append(c / np.linalg.norm(c))
            if sid not in absent:
                for f in range(probes_per_student):
                    probe_frames[f].append(E[idx[n_enroll + f]])
        for st in strangers:
            idx = groups[st]
            for f in range(probes_per_student):
                probe_frames[f].append(E[idx[f % len(idx)]])
        roster = Roster(students, np.stack(cents))
        sess = AttendanceSession(roster, min_frames=probes_per_student)
        for f, frame in enumerate(probe_frames):
            labels, scores = roster.match(np.stack(frame), threshold)
            sess.update(f, labels, scores)
        for sid in students:
            rows.append({"session": s, "identity": sid, "present_truth": sid not in absent,
                         "marked_present": sess.is_present(sid),
                         "gender": gender.get(sid, "unknown"), "ethnicity": ethnicity.get(sid, "unknown")})
        n_pred = sum(sess.is_present(x) for x in students)
        sess_rows.append({"true_present": class_size - n_absent, "pred_present": n_pred,
                          "faces_seen": len(probe_frames[0])})

    r = pd.DataFrame(rows)
    pres = r[r["present_truth"]]
    absn = r[~r["present_truth"]]
    out = {
        "attendance_recall": float(pres["marked_present"].mean()),
        "false_present_rate": float(absn["marked_present"].mean()),
        "count_MAE": float(np.mean([abs(x["pred_present"] - x["true_present"]) for x in sess_rows])),
        "sessions": n_sessions, "class_size": class_size,
    }
    for attr in ["gender", "ethnicity"]:
        by = pres.groupby(attr)["marked_present"].agg(["mean", "count"])
        out[f"recall_by_{attr}"] = {k: {"recall": float(v["mean"]), "n": int(v["count"])} for k, v in by.iterrows()}
        out[f"recall_gap_{attr}"] = fairness_gap({k: float(v["mean"]) for k, v in by.iterrows() if v["count"] >= 30})
    return out


def group_verification(df: pd.DataFrame, attrs: pd.DataFrame, threshold: float, seed: int = 0) -> dict:
    """FAR/FRR per demographic group, with impostor pairs drawn WITHIN the group (harder, standard practice)."""
    rng = np.random.default_rng(seed)
    gmap = {a: attrs.groupby("identity")[a].agg(lambda s: s.mode().iloc[0]).to_dict() for a in ["gender", "ethnicity"]}
    det = df[df["detected"]].reset_index(drop=True)
    E = np.stack(det["emb"].to_numpy())
    ident = det["identity"].to_numpy()
    out = {}
    for attr, mp in gmap.items():
        grp = np.array([mp.get(i, "unknown") for i in ident])
        res = {}
        for g in sorted(set(grp) - {"unknown"}):
            idx = np.where(grp == g)[0]
            gen = []
            for _, gi in det.iloc[idx].groupby("identity").indices.items():
                gi = idx[gi][:30]
                gen += [(a, b) for k, a in enumerate(gi) for b in gi[k + 1:]]
            if len(gen) < 50:
                continue
            gen = np.array(gen)
            imp = rng.choice(idx, size=(20000, 2))
            imp = imp[ident[imp[:, 0]] != ident[imp[:, 1]]]
            gs = np.einsum("ij,ij->i", E[gen[:, 0]], E[gen[:, 1]])
            is_ = np.einsum("ij,ij->i", E[imp[:, 0]], E[imp[:, 1]])
            res[g] = {**rates_at(gs, is_, threshold), "genuine_pairs": len(gen),
                      "identities": int(len(set(ident[idx])))}
        out[attr] = {"by_group": res,
                     "FRR_gap": fairness_gap({k: -v["FRR"] for k, v in res.items()})["gap"],
                     "FAR_gap": fairness_gap({k: -v["FAR"] for k, v in res.items()})["gap"]}
    return out


def run(cfg: dict, engine, out_dir: Path, logger: logging.Logger) -> dict:
    raw = Path(cfg["paths"]["data_raw"]) / "lfw"
    proc = Path(cfg["paths"]["data_processed"]) / "lfw"
    thr = cfg["face"]["match_threshold"]
    attrs = load_attributes(raw / "lfw_attributes.txt")
    df = compute_embeddings(engine, raw / "lfw.parquet", proc / "emb_clean.npz", logger=logger)
    results = {"detection_rate": float(df["detected"].mean()), "images": len(df), "threshold": thr}

    _, _, gs, _, is_ = verification_pairs(df)
    results["verification"] = {**rates_at(gs, is_, thr), "genuine_pairs": len(gs), "impostor_pairs": len(is_)}
    for far in [1e-2, 1e-3, 1e-4]:
        tar, t = tar_at_far(gs, is_, far)
        results["verification"][f"TAR@FAR={far:g}"] = {"TAR": tar, "threshold": t}
    results["group_verification"] = group_verification(df, attrs, thr)
    results["sessions"] = simulate_sessions(df, attrs, thr)
    logger.info(f"[lfw] verification {results['verification']}")
    logger.info(f"[lfw] sessions recall={results['sessions']['attendance_recall']:.4f} "
                f"false_present={results['sessions']['false_present_rate']:.4f}")

    # robustness: degrade probe images of identities with >= 5 images, keep clean enrolment
    ids = set(df.groupby("identity").size().loc[lambda s: s >= 5].index)
    rob = {}
    clean_sub = (df[df["identity"].isin(ids)].sort_values(["identity", "image_num"])
                 .groupby("identity").head(6).reset_index(drop=True))
    for kind in DEGRADATIONS[1:]:
        deg = compute_embeddings(engine, raw / "lfw.parquet", proc / f"emb_{kind}.npz", ids=ids, max_per_id=6,
                                 degradation=kind, logger=logger)
        rob[kind] = _robustness_eval(clean_sub, deg, thr)
        logger.info(f"[lfw] robustness {kind}: {rob[kind]}")
    rob["clean"] = _robustness_eval(clean_sub, clean_sub, thr)
    results["robustness"] = rob

    (out_dir / "face_recognition.json").write_text(json.dumps(results, indent=2))
    _plots(gs, is_, thr, results, out_dir)
    return results


def _robustness_eval(clean: pd.DataFrame, deg: pd.DataFrame, thr: float, n_enroll: int = 3) -> dict:
    """Enrol from first n_enroll CLEAN images; probe with the remaining (possibly degraded) images."""
    key = lambda d: d.set_index(["identity", "image_num"])
    c, d = key(clean), key(deg)
    ids = sorted(set(c.index.get_level_values(0)))
    cents, probes, probe_ids = [], [], []
    for i in ids:
        ci = c.loc[i]
        ci = ci[ci["detected"]]
        if len(ci) < n_enroll + 1:
            continue
        nums = sorted(ci.index)
        e = np.stack(ci.loc[nums[:n_enroll], "emb"].to_numpy()).mean(0)
        cents.append(e / np.linalg.norm(e))
        for n in nums[n_enroll:]:
            row = d.loc[(i, n)]
            probes.append(row["emb"] if row["detected"] else None)
            probe_ids.append(len(cents) - 1)
    C = np.stack(cents)
    det = np.array([p is not None for p in probes])
    P = np.stack([p if p is not None else np.zeros(C.shape[1]) for p in probes])
    sims = P @ C.T
    best = sims.argmax(1)
    correct = det & (best == np.array(probe_ids)) & (sims.max(1) >= thr)
    return {"face_detected": float(det.mean()), "rank1_identified": float(correct.mean()), "probes": int(len(P))}


def _plots(gs, is_, thr, results, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(is_, bins=100, alpha=0.6, density=True, label="impostor pairs")
    axes[0].hist(gs, bins=100, alpha=0.6, density=True, label="genuine pairs")
    axes[0].axvline(thr, color="k", ls="--", label=f"threshold {thr}")
    axes[0].set_xlabel("cosine similarity"), axes[0].legend(), axes[0].set_title("LFW: SFace similarity")
    thrs = np.linspace(-0.2, 1, 300)
    far = [np.mean(is_ >= t) for t in thrs]
    tar = [np.mean(gs >= t) for t in thrs]
    axes[1].semilogx(np.maximum(far, 1e-6), tar)
    axes[1].set_xlabel("FAR"), axes[1].set_ylabel("TAR"), axes[1].set_title("ROC"), axes[1].grid(alpha=0.3, which="both")
    rob = results["robustness"]
    kinds = list(rob)
    axes[2].barh(kinds, [rob[k]["rank1_identified"] for k in kinds])
    axes[2].set_xlim(0, 1), axes[2].set_title("robustness: correctly identified probes")
    for i, k in enumerate(kinds):
        axes[2].text(rob[k]["rank1_identified"] + 0.01, i, f"{rob[k]['rank1_identified']:.3f}", va="center")
    fig.savefig(out_dir / "face_recognition.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
