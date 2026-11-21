"""Reproducible experiment stages. Each stage reads the config, writes JSON + figures under reports/,
and can be re-run on its own:  python -m smartattend stage <name>   (or `stage all`)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

STAGES = ["data", "kaggle_data", "conv_viz", "engagement", "engagement_cv", "sequence", "detection", "seating", "states", "face", "pipeline_latency"]


def _p(cfg, key) -> Path:
    return Path(cfg["paths"][key])


def stage_data(cfg, logger):
    from smartattend.data import coco_classroom, download

    raw = _p(cfg, "data_raw")
    download.download_lfw(raw, logger)
    download.download_drowsiness(raw, logger)
    return coco_classroom.prepare(raw, _p(cfg, "data_processed") / "coco_classroom", logger)


def stage_kaggle_data(cfg, logger):
    from smartattend.data.kaggle_classroom import prepare

    return prepare(_p(cfg, "data_raw"), _p(cfg, "data_processed") / "kaggle_classroom", logger)


def stage_conv_viz(cfg, logger):
    from smartattend.data.coco_classroom import load_split
    from smartattend.vision.conv_viz import make_figures

    d = _p(cfg, "data_processed") / "coco_classroom"
    rec = max(load_split(d, "test"), key=lambda r: len(r["persons"]))
    return [str(p) for p in make_figures(d / "images" / "test" / rec["file_name"], _p(cfg, "figures_dir") / "vision")]


def stage_engagement(cfg, logger):
    import torch

    from smartattend.data.drowsiness import load_split_frames
    from smartattend.eval.metrics import classification_report
    from smartattend.vision import engagement as E

    out = _p(cfg, "experiments_dir") / "engagement"
    out.mkdir(parents=True, exist_ok=True)
    df = load_split_frames(_p(cfg, "data_raw") / "drowsiness", seed=cfg["seed"])
    tr, va, te = (df[df["split"] == s] for s in ["train", "val", "test"])
    split_counts = {s: df[df["split"] == s]["engagement"].value_counts().sort_index().tolist() for s in ["train", "val", "test"]}
    results = {"split_counts(engaged,neutral,disengaged)": split_counts}
    yte = te["engagement"].to_numpy()

    # 1. linear probes on frozen ImageNet backbones (classification benchmark: ResNet vs GoogLeNet)
    for name in ["resnet18", "googlenet"]:
        bb, _ = E.backbone(name)
        t0 = time.time()
        F = {s: E.extract_features(bb, list(d["path"])) for s, d in [("train", tr), ("val", va), ("test", te)]}
        ms = (time.time() - t0) / len(df) * 1000
        head, mu, sd = E.train_linear_probe(F["train"], tr["engagement"].to_numpy(), F["val"], va["engagement"].to_numpy())
        with torch.no_grad():
            p = head(torch.tensor((F["test"] - mu) / sd)).argmax(1).numpy()
        results[f"{name}_linear_probe"] = {**classification_report(yte, p, 3), "feature_ms_per_image": ms,
                                           "params_M": sum(x.numel() for x in bb.parameters()) / 1e6}
        logger.info(f"[engagement] {name} probe macroF1={results[f'{name}_linear_probe']['macro_f1']:.3f}")

    # 2. fine-tuned ResNet-18 (layer4 + head) with augmentation -> product model
    net, hist = E.finetune_resnet18(tr, va, epochs=cfg.get("engagement_epochs", 4), seed=cfg["seed"], logger=logger)
    torch.save(net.state_dict(), Path(cfg["paths"]["models_dir"]) / "engagement_resnet18.pt")
    clf = E.EngagementClassifier(Path(cfg["paths"]["models_dir"]) / "engagement_resnet18.pt")
    crops = [E._load_gray(p) for p in te["path"]]
    t0 = time.time()
    P = np.concatenate([clf.predict_proba(crops[i:i + 32]) for i in range(0, len(crops), 32)])
    ms = (time.time() - t0) / len(crops) * 1000
    results["resnet18_finetuned"] = {**classification_report(yte, P.argmax(1), 3), "history": hist, "ms_per_crop": ms}
    # bootstrap 95% CI of macro-F1 (test set is small)
    rng = np.random.default_rng(0)
    boots = [classification_report(yte[i], P.argmax(1)[i], 3)["macro_f1"]
             for i in (rng.integers(0, len(yte), len(yte)) for _ in range(500))]
    results["resnet18_finetuned"]["macro_f1_95CI"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    np.savez(out / "test_probs.npz", probs=P, y=yte)
    logger.info(f"[engagement] fine-tuned macroF1={results['resnet18_finetuned']['macro_f1']:.3f} "
                f"CI={results['resnet18_finetuned']['macro_f1_95CI']}")
    (out / "engagement.json").write_text(json.dumps(results, indent=2, default=float))
    _plot_confusions(results, out)
    return results


def stage_engagement_cv(cfg, logger, k: int = 5):
    """Grouped k-fold CV over 5-s time blocks for the linear probes. With one subject and ~150 test
    frames a single split is high-variance, so this is the estimate to trust for backbone comparison."""
    import torch

    from smartattend.data.drowsiness import load_frames, time_block_split
    from smartattend.eval.metrics import classification_report
    from smartattend.vision import engagement as E

    out = _p(cfg, "experiments_dir") / "engagement"
    out.mkdir(parents=True, exist_ok=True)
    df = time_block_split(load_frames(_p(cfg, "data_raw") / "drowsiness"), seed=cfg["seed"]).reset_index(drop=True)
    y = df["engagement"].to_numpy()
    # stratified assignment of blocks to folds (by block majority label)
    rng = np.random.default_rng(cfg["seed"])
    major = df.groupby("block")["engagement"].agg(lambda s: s.mode().iloc[0])
    fold_of = {}
    for _, grp in major.groupby(major):
        names = list(grp.index)
        rng.shuffle(names)
        for i, b in enumerate(names):
            fold_of[b] = i % k
    fold = df["block"].map(fold_of).to_numpy()
    res = {"folds": k, "frames": int(len(df))}
    for name in ["resnet18", "googlenet"]:
        bb, _ = E.backbone(name)
        F = E.extract_features(bb, list(df["path"]))
        scores = []
        for f in range(k):
            te = fold == f
            va = fold == (f + 1) % k
            tr = ~(te | va)
            head, mu, sd = E.train_linear_probe(F[tr], y[tr], F[va], y[va])
            with torch.no_grad():
                p = head(torch.tensor((F[te] - mu) / sd)).argmax(1).numpy()
            scores.append(classification_report(y[te], p, 3)["macro_f1"])
        res[name] = {"macro_f1_folds": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}
        logger.info(f"[engagement-cv] {name} macroF1 {np.mean(scores):.3f} +/- {np.std(scores):.3f} folds={np.round(scores, 3).tolist()}")
    (out / "engagement_cv.json").write_text(json.dumps(res, indent=2))
    return res


def _plot_confusions(results, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smartattend.data.drowsiness import ENGAGEMENT_CLASSES

    keys = [k for k in results if isinstance(results[k], dict) and "confusion" in results[k]]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.2))
    for ax, k in zip(np.atleast_1d(axes), keys):
        cm = np.array(results[k]["confusion"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks(range(3), ENGAGEMENT_CLASSES, rotation=20), ax.set_yticks(range(3), ENGAGEMENT_CLASSES)
        ax.set_xlabel("predicted"), ax.set_ylabel("true")
        ax.set_title(f"{k}\nmacro-F1={results[k]['macro_f1']:.3f}", fontsize=10)
    fig.savefig(out / "engagement_confusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def stage_sequence(cfg, logger):
    from smartattend.data.drowsiness import load_split_frames
    from smartattend.eval.metrics import confusion_matrix
    from smartattend.sequence import latent_viz, lstm_trend, transformer_agg
    from smartattend.vision.engagement import EngagementClassifier

    out = _p(cfg, "experiments_dir") / "sequence"
    df = load_split_frames(_p(cfg, "data_raw") / "drowsiness", seed=cfg["seed"])
    tp = np.load(_p(cfg, "experiments_dir") / "engagement" / "test_probs.npz")
    cm = confusion_matrix(tp["y"], tp["probs"].argmax(1), 3) + 1  # +1 Laplace smoothing
    res = {"trend": lstm_trend.run(df, cm, out, logger),
           "aggregator": transformer_agg.run({c: tp["probs"][tp["y"] == c] for c in range(3)}, out, logger),
           "latent": latent_viz.run(df, EngagementClassifier(Path(cfg["paths"]["models_dir"]) / "engagement_resnet18.pt"), out, logger)}
    return res


def stage_detection(cfg, logger):
    from smartattend.vision.benchmark import add_crowd_free_counts, benchmark, failure_gallery

    d = _p(cfg, "data_processed") / "coco_classroom"
    out = _p(cfg, "experiments_dir") / "detection"
    benchmark(d, cfg["paths"]["models_dir"], out, logger)
    res = add_crowd_free_counts(d, out)
    best = max(res, key=lambda n: res[n]["mAP50"])
    failure_gallery(d, out, best, res[best]["count_threshold(val)"])
    return res


def stage_seating(cfg, logger):
    import cv2

    from smartattend.data.coco_classroom import gt_boxes_xyxy, gt_person_mask, load_split
    from smartattend.eval.metrics import mask_iou
    from smartattend.vision import seating as S

    d = _p(cfg, "data_processed") / "coco_classroom"
    out = _p(cfg, "experiments_dir") / "seating"
    out.mkdir(parents=True, exist_ok=True)
    test = load_split(d, "test")
    sem = S.SemanticSegmenter()
    ins = S.InstanceSegmenter(Path(cfg["paths"]["models_dir"]) / "yolo" / "yolov8n-seg.pt")
    ious_sem, ious_ins, lat_sem, lat_ins, chairs = [], [], [], [], []
    for r in test:
        img = cv2.imread(str(d / "images" / "test" / r["file_name"]))
        gt = gt_person_mask(r)
        t0 = time.time()
        lab = sem(img)
        lat_sem.append((time.time() - t0) * 1000)
        t0 = time.time()
        inst = ins(img)
        lat_ins.append((time.time() - t0) * 1000)
        pm = np.zeros(gt.shape, bool)
        for i in inst:
            if i.cls == S.COCO_PERSON and i.mask is not None:
                pm |= cv2.resize(i.mask.astype(np.uint8), (gt.shape[1], gt.shape[0])).astype(bool)
        ious_sem.append(mask_iou(lab == S.VOC_PERSON, gt))
        ious_ins.append(mask_iou(pm, gt))
        chairs.append(sum(i.cls == S.COCO_CHAIR for i in inst))
    res = {"semantic_LRASPP_person_IoU": float(np.nanmean(ious_sem)), "instance_YOLOv8nSeg_person_IoU": float(np.nanmean(ious_ins)),
           "semantic_latency_ms": float(np.mean(lat_sem)), "instance_latency_ms": float(np.mean(lat_ins)),
           "images": len(test), "mean_chairs_detected": float(np.mean(chairs))}
    logger.info(f"[seating] {res}")
    _seating_examples(test, d, sem, ins, out)
    (out / "seating.json").write_text(json.dumps(res, indent=2))
    return res


def _seating_examples(test, d, sem, ins, out):
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from smartattend.vision import seating as S

    picks = sorted(test, key=lambda r: -len(r["persons"]))[:3]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for row, r in zip(axes, picks):
        img = cv2.imread(str(d / "images" / "test" / r["file_name"]))
        lab = sem(img)
        inst = ins(img)
        semv = S.overlay_mask(S.overlay_mask(img, lab == S.VOC_PERSON), lab == S.VOC_CHAIR, (60, 200, 60))
        insv = img.copy()
        rng = np.random.default_rng(0)
        for i in inst:
            if i.mask is not None:
                m = cv2.resize(i.mask.astype(np.uint8), (img.shape[1], img.shape[0])).astype(bool)
                insv = S.overlay_mask(insv, m, tuple(int(c) for c in rng.integers(40, 255, 3)), 0.5)
        persons = np.array([i.box for i in inst if i.cls == S.COCO_PERSON]).reshape(-1, 4)
        seats = np.array([i.box for i in inst if i.cls in (S.COCO_CHAIR, S.COCO_COUCH, S.COCO_BENCH)]).reshape(-1, 4)
        sm = S.build_seat_map(persons, seats)
        for ax, im, t in zip(row, [semv, insv, S.draw_seat_map(img, sm)],
                             ["semantic (orange person, green chair)", "instance masks",
                              f"seat map: {sm.n_rows} rows, {sm.occupied} occupied, {sm.empty} empty"]):
            ax.imshow(im[:, :, ::-1]), ax.set_title(t, fontsize=10), ax.axis("off")
    fig.savefig(out / "seating_examples.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def stage_face(cfg, logger):
    from smartattend.attendance.face import FaceEngine, ensure_models
    from smartattend.eval import face_eval

    eng = FaceEngine(*ensure_models(cfg["paths"]["models_dir"]), score_threshold=cfg["face"]["detector_score_threshold"])
    out = _p(cfg, "eval_dir")
    out.mkdir(parents=True, exist_ok=True)
    return face_eval.run(cfg, eng, out, logger)


def stage_pipeline_latency(cfg, logger):
    """End-to-end latency/memory of the integrated product on classroom test images."""
    import cv2
    import psutil

    from smartattend.data.coco_classroom import load_split
    from smartattend.pipeline import ClassroomAnalyzer

    d = _p(cfg, "data_processed") / "coco_classroom"
    an = ClassroomAnalyzer(cfg, roster=None)
    lat = {k: [] for k in ["detect", "faces", "engagement", "seating", "total"]}
    peak = 0.0
    for r in load_split(d, "test")[:40]:
        res = an.analyze_frame(cv2.imread(str(d / "images" / "test" / r["file_name"])))
        for k in lat:
            lat[k].append(res["timing_ms"][k])
        peak = max(peak, psutil.Process().memory_info().rss / 1e6)
    out = {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))} for k, v in lat.items()}
    out["peak_rss_mb"] = peak
    logger.info(f"[pipeline] {out}")
    p = _p(cfg, "eval_dir")
    p.mkdir(parents=True, exist_ok=True)
    (p / "pipeline_latency.json").write_text(json.dumps(out, indent=2))
    return out


def stage_states(cfg, logger):
    """Attentive / distracted / inactive: (1) state mix by face size on the classroom test set (shows why
    faces < min_face_px are 'unclear'); (2) can 5-point landmarks measure head pitch? Checked against the
    drowsiness dataset's per-frame head_pitch (it cannot, so 'looking down' is not used)."""
    import cv2
    from scipy.stats import spearmanr

    from smartattend.data.coco_classroom import load_split
    from smartattend.data.drowsiness import load_frames
    from smartattend.pipeline import STATES, ClassroomAnalyzer, engagement_state, head_turn

    d = _p(cfg, "data_processed") / "coco_classroom"
    an = ClassroomAnalyzer(cfg)
    es = cfg.get("engagement_states", {})
    sizes, raw_states = [], []
    for r in load_split(d, "test"):
        res = an.analyze_frame(cv2.imread(str(d / "images" / "test" / r["file_name"])))
        for s in res["students"]:
            sizes.append(s["face_size_px"])
            raw_states.append(engagement_state(np.array(s["probs"]), s["head_turn"], es.get("distracted_head_turn", 0.30)))
    sizes, raw_states = np.array(sizes), np.array(raw_states)
    by_size = {}
    for lo, hi in [(0, 30), (30, 40), (40, 60), (60, 100), (100, 10_000)]:
        m = (sizes >= lo) & (sizes < hi)
        if m.any():
            by_size[f"{lo}-{hi}px"] = {"faces": int(m.sum()), **{s: float((raw_states[m] == s).mean()) for s in STATES}}

    rows = []
    df = load_frames(_p(cfg, "data_raw") / "drowsiness").sample(600, random_state=cfg["seed"])
    for p, pitch in zip(df["path"], df["head_pitch"]):
        f = an.faces.detect(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR))
        if len(f):
            f = f[0]
            eye_y, mouth_y = (f[5] + f[7]) / 2, (f[11] + f[13]) / 2
            rows.append((pitch, (f[9] - eye_y) / (mouth_y - eye_y + 1e-6)))
    a = np.array(rows)
    res = {"state_mix_by_face_size(test split, no size cut-off)": by_size,
           "pitch_proxy_vs_ground_truth": {"frames": len(a), "spearman": float(spearmanr(a[:, 0], a[:, 1])[0])},
           "min_face_px": es.get("min_face_px", 30), "distracted_head_turn": es.get("distracted_head_turn", 0.30)}
    logger.info(f"[states] {res}")
    out = _p(cfg, "eval_dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "engagement_states.json").write_text(json.dumps(res, indent=2))
    return res


def run_stage(name: str, cfg: dict, logger: logging.Logger):
    names = STAGES if name == "all" else [name]
    results = {}
    for n in names:
        t0 = time.time()
        logger.info(f"=== stage {n} ===")
        results[n] = globals()[f"stage_{n}"](cfg, logger)
        logger.info(f"=== stage {n} done in {time.time() - t0:.0f}s ===")
    return results
