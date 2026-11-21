"""Requirement 8: one benchmark of R-CNN, Fast R-CNN, Faster R-CNN, RetinaNet and YOLO (detection) and ResNet vs
GoogLeNet (as R-CNN backbones and as attentive/distracted/inactive classifiers), on the Kaggle TEST classroom.

    python -I scripts/selective_search.py --cv2-path models/tools/opencv_contrib    # proposals, once
    python -m smartattend benchmark

Same test frames, same metric code, thresholds chosen on validation, timing = median after a warm-up image.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from smartattend.vision.classroom_eval import load_records

VAL_STEP = 4          # thresholds are chosen on every 4th validation frame (45 frames) to bound R-CNN's CPU time


def cached_preds(name: str, split: str, records: list[dict], make_fn, cache_dir: Path):
    """make_fn() -> callable(record) -> (boxes, scores). Results cached as JSON (same format as Requirement 7)."""
    cache = cache_dir / f"preds_{name}_{split}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        preds = [(np.array(b).reshape(-1, 4), np.array(s)) for b, s in d["preds"]]
        if len(preds) == len(records):
            return preds, d["ms"]
        if split == "val" and len(d["preds"]) > len(records):           # full-val cache from Requirement 7
            preds = preds[::VAL_STEP]
            return preds, d["ms"][::VAL_STEP]
    fn = make_fn()
    fn(records[0])                                                     # warm-up
    preds, ms = [], []
    for r in records:
        t0 = time.perf_counter()
        b, s = fn(r)
        ms.append((time.perf_counter() - t0) * 1000)
        preds.append((np.asarray(b).reshape(-1, 4), np.asarray(s)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"preds": [(b.tolist(), s.tolist()) for b, s in preds], "ms": ms}))
    return preds, ms


COUNT_GRID = np.r_[np.round(np.arange(0.10, 0.90, 0.05), 2), 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]


def count_threshold(preds, gts) -> float:
    """Score threshold with the lowest count error on VALIDATION. The grid goes up to 0.999 because R-CNN's
    scores bunch up near 1.0 (with a grid ending at 0.85 it counted 20 students too many per frame)."""
    mae = [np.mean([abs(int((s >= t).sum()) - len(g)) for (_, s), g in zip(preds, gts)]) for t in COUNT_GRID]
    return float(COUNT_GRID[int(np.argmin(mae))])


def retime(cfg: dict, n_frames: int = 5, log=print) -> dict:
    """Fair timing: every detector on the SAME frames, interleaved (model A, B, C ... on frame 1, then frame 2 ...),
    after one warm-up each, so CPU throttling over a long run affects all of them equally. Median ms per frame."""
    import torch
    import torchvision.models.detection as D

    from smartattend.vision import rcnn
    from smartattend.vision.detectors import make_detector

    P = lambda k: Path(cfg["paths"][k])
    processed, models_dir, cache_dir = P("data_processed") / "kaggle_classroom", P("models_dir"), P("experiments_dir") / "classroom"
    test = load_records(processed, "test")
    frames = test[::max(1, len(test) // n_frames)][:n_frames]
    props = rcnn.load_proposals(processed, "test")
    fns = {}
    for arch in ["resnet18", "googlenet"]:
        info = json.loads((cache_dir / f"rcnn_{arch}_tuning.json").read_text())["chosen"]
        z = np.load(models_dir / f"rcnn_{arch}_train_features.npz")
        m = rcnn.RCNN(arch).fit_heads(z["F"], z["Y"], z["T"], C=info["C"], alpha=info["alpha"])
        fns[f"rcnn_{arch}"] = lambda img, name, m=m, nms=info["nms"]: m.detect_from_features(props[name], m.features(img, props[name]), nms)
    fast = rcnn.FastRCNN(short_side=600)
    fast.load_state_dict(torch.load(models_dir / "fast_rcnn_resnet18.pt", map_location="cpu"))
    fns["fast_rcnn_resnet18"] = lambda img, name: fast.detect(img, props[name])
    for name, ctor in [("fasterrcnn_resnet50_fpn", D.fasterrcnn_resnet50_fpn), ("retinanet_resnet50_fpn", D.retinanet_resnet50_fpn)]:
        det = make_detector(name, models_dir)
        fns[name] = lambda img, _n, det=det: det(img, 0.05)
        model = ctor(weights=None, weights_backbone=None, min_size=540, max_size=960)
        model.load_state_dict(torch.load(models_dir / f"{name}_kaggle.pt", map_location="cpu"))
        ft = rcnn.TorchvisionFinetuned(model.eval(), name)
        fns[f"{name}_kaggle"] = lambda img, _n, ft=ft: ft(img, 0.05)
    for name in ["yolov8n", "yolov8s", "yolov8n-kaggle"]:
        det = make_detector(name, models_dir)
        fns[name] = lambda img, _n, det=det: det(img, 0.05)
    imgs = [(cv2.imread(str(r["path"])), r["path"].name) for r in frames]
    for fn in fns.values():
        fn(*imgs[0])                                                   # warm-up
    ms = {k: [] for k in fns}
    for i, (img, name) in enumerate(imgs):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            fn(img, name)
            ms[k].append((time.perf_counter() - t0) * 1000)
        log(f"[retime] frame {i + 1}/{len(imgs)}")
    out = {k: float(np.median(v)) for k, v in ms.items()}
    out["_selective_search_ms"] = 1560.0     # measured by scripts/selective_search.py (1.56 s per frame), paid by R-CNN and Fast R-CNN
    (cache_dir / "timing_interleaved.json").write_text(json.dumps(out, indent=1))
    return out


def proposal_recall(props: dict, records: list[dict], iou: float = 0.5) -> float:
    from smartattend.vision.rcnn import box_iou

    hit = n = 0
    for r in records:
        if len(r["boxes"]):
            P = np.array(props[r["path"].name]).reshape(-1, 4)
            hit += int((box_iou(r["boxes"], P).max(1) >= iou).sum()) if len(P) else 0
            n += len(r["boxes"])
    return hit / n


def tune_rcnn(arch: str, rec: dict, props: dict, models_dir: Path, cache_dir: Path, log=print) -> dict:
    """R-CNN end to end, with its hyperparameters chosen on the validation frames.

    The CNN features of every region are computed once and cached (the slow part: about 10 s per frame on CPU).
    The linear classifier (C) and box regressor (alpha) and the NMS overlap are then tried on a small grid and the
    setting with the best VALIDATION mAP50 is kept. Test frames are only scored with that one setting."""
    from smartattend.eval.metrics import coco_map
    from smartattend.vision import rcnn

    info_path = cache_dir / f"rcnn_{arch}_tuning.json"
    if info_path.exists() and all((cache_dir / f"preds_rcnn_{arch}_{s}.json").exists() for s in ["val", "test"]):
        return json.loads(info_path.read_text())
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = rcnn.RCNN(arch)
    train_path = models_dir / f"rcnn_{arch}_train_features.npz"
    t0 = time.perf_counter()
    if train_path.exists():
        z = np.load(train_path)
        F, Y, T = z["F"], z["Y"], z["T"]
    else:
        F, Y, T = model.train_features(rec["train"], props["train"], log=log)
        np.savez_compressed(train_path, F=F.astype(np.float32), Y=Y, T=T)
    train_s = time.perf_counter() - t0

    feats, ms = {}, {}
    for split in ["val", "test"]:
        # one small file per frame, so an interrupted run resumes where it stopped
        d = cache_dir / f"rcnn_features_{arch}_{split}"
        d.mkdir(exist_ok=True)
        feats[split], ms[split] = [], []
        warmed = False
        for i, r in enumerate(rec[split]):
            f_path, t_path = d / f"{i}.npy", d / f"{i}.ms"
            if not (f_path.exists() and t_path.exists()):
                if not warmed:
                    model.features(cv2.imread(str(r["path"])), props[split][r["path"].name][:8])
                    warmed = True
                t = time.perf_counter()
                f = model.features(cv2.imread(str(r["path"])), props[split][r["path"].name])
                t_ms = (time.perf_counter() - t) * 1000
                np.save(f_path, f.astype(np.float16))
                t_path.write_text(str(t_ms))
                if (i + 1) % 10 == 0:
                    log(f"[rcnn-{arch}] {split} features {i + 1}/{len(rec[split])} ({t_ms / 1000:.1f} s/frame)")
            feats[split].append(np.load(f_path).astype(np.float32))
            ms[split].append(float(t_path.read_text()))

    def predict(split, nms_thr):
        return [model.detect_from_features(props[split][r["path"].name], f, nms_thr) for r, f in zip(rec[split], feats[split])]

    gv = [r["boxes"] for r in rec["val"]]
    grid, best = [], (-1.0, None)
    for C in [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]:
        for alpha in [10.0, 100.0, 1000.0, 1e4, 1e5]:
            model.fit_heads(F, Y, T, C=C, alpha=alpha)
            for nms_thr in [0.3, 0.5]:
                m = coco_map(predict("val", nms_thr), gv)["mAP50"]
                grid.append({"C": C, "alpha": alpha, "nms": nms_thr, "val_mAP50": m})
                if m > best[0]:
                    best = (m, (C, alpha, nms_thr))
    C, alpha, nms_thr = best[1]
    log(f"[rcnn-{arch}] best on validation: C={C} alpha={alpha} nms={nms_thr} (val mAP50 {best[0]:.3f})")
    model.fit_heads(F, Y, T, C=C, alpha=alpha)
    for split in ["val", "test"]:
        preds = predict(split, nms_thr)
        (cache_dir / f"preds_rcnn_{arch}_{split}.json").write_text(json.dumps(
            {"preds": [(b.tolist(), s.tolist()) for b, s in preds], "ms": ms[split]}))
    info = {"train_regions": int(len(Y)), "positives": int(Y.sum()), "train_feature_seconds": train_s,
            "chosen": {"C": C, "alpha": alpha, "nms": nms_thr, "val_mAP50": best[0]}, "grid": grid}
    info_path.write_text(json.dumps(info, indent=1))
    return info


def run(cfg: dict, log=print, fast_epochs: int = 6, tv_epochs: int = 3) -> tuple[list[str], dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from smartattend.eval.metrics import coco_map, count_errors, recall_at
    from smartattend.vision import rcnn
    from smartattend.vision.benchmark import best_count_threshold
    from smartattend.vision.detectors import make_detector

    P = lambda k: Path(cfg["paths"][k])
    processed, models_dir = P("data_processed") / "kaggle_classroom", P("models_dir")
    cache_dir, fig_dir = P("experiments_dir") / "classroom", P("figures_dir") / "vision"
    rec = {s: load_records(processed, s) for s in ["train", "val", "test"]}
    rec["val"] = rec["val"][::VAL_STEP]
    props = {s: rcnn.load_proposals(processed, s) for s in ["train", "val", "test"]}
    trained = {}
    train_info = {}

    # ---- the models that have to be trained here
    def rcnn_fn(arch):
        def make():
            raise RuntimeError(f"R-CNN ({arch}) predictions should have been written by tune_rcnn")
        return make

    def fast_fn():
        path = models_dir / "fast_rcnn_resnet18.pt"
        m = rcnn.FastRCNN(short_side=600)
        if path.exists():
            m.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            t0 = time.perf_counter()
            train_info["fast_rcnn_resnet18"] = {**m.fit(rec["train"], props["train"], epochs=fast_epochs, log=log),
                                                "train_seconds": time.perf_counter() - t0}
            torch.save(m.state_dict(), path)
        return lambda r: m.detect(cv2.imread(str(r["path"])), props[split_of[r["path"].name]][r["path"].name])

    def tv_fn(name):
        def make():
            path = models_dir / f"{name}_kaggle.pt"
            import torchvision.models.detection as D

            if path.exists():
                ctor = D.fasterrcnn_resnet50_fpn if name.startswith("faster") else D.retinanet_resnet50_fpn
                model = ctor(weights=None, weights_backbone=None, min_size=540, max_size=960)
                model.load_state_dict(torch.load(path, map_location="cpu"))
                model.eval()
            else:
                t0 = time.perf_counter()
                model, hist = rcnn.finetune_torchvision(name, rec["train"], epochs=tv_epochs, log=log,
                                                         ckpt=models_dir / f"{name}_kaggle.ckpt")
                train_info[f"{name}_kaggle"] = {"loss": hist, "train_seconds": time.perf_counter() - t0}
                torch.save(model.state_dict(), path)
            det = rcnn.TorchvisionFinetuned(model, name)
            return lambda r: det(cv2.imread(str(r["path"])), 0.05)
        return make

    def pretrained_fn(name):
        def make():
            det = make_detector(name, models_dir)
            return lambda r: det(cv2.imread(str(r["path"])), 0.05)
        return make

    split_of = {r["path"].name: s for s in rec for r in rec[s]}
    MODELS = {   # name: (family, trained on Kaggle?, factory)
        "R-CNN (ResNet18)": ("R-CNN", True, rcnn_fn("resnet18"), "rcnn_resnet18"),
        "R-CNN (GoogLeNet)": ("R-CNN", True, rcnn_fn("googlenet"), "rcnn_googlenet"),
        "Fast R-CNN (ResNet18)": ("Fast R-CNN", True, fast_fn, "fast_rcnn_resnet18"),
        "Faster R-CNN (COCO)": ("Faster R-CNN", False, pretrained_fn("fasterrcnn_resnet50_fpn"), "fasterrcnn_resnet50_fpn"),
        "Faster R-CNN (fine-tuned)": ("Faster R-CNN", True, tv_fn("fasterrcnn_resnet50_fpn"), "fasterrcnn_resnet50_fpn_kaggle"),
        "RetinaNet (COCO)": ("RetinaNet", False, pretrained_fn("retinanet_resnet50_fpn"), "retinanet_resnet50_fpn"),
        "RetinaNet (fine-tuned)": ("RetinaNet", True, tv_fn("retinanet_resnet50_fpn"), "retinanet_resnet50_fpn_kaggle"),
        "YOLOv8n (COCO)": ("YOLO", False, pretrained_fn("yolov8n"), "yolov8n"),
        "YOLOv8s (COCO)": ("YOLO", False, pretrained_fn("yolov8s"), "yolov8s"),
        "YOLOv8n (fine-tuned)": ("YOLO", True, pretrained_fn("yolov8n-kaggle"), "yolov8n-kaggle"),
    }
    PARAMS_M = {"rcnn_resnet18": 11.2, "rcnn_googlenet": 5.6, "fast_rcnn_resnet18": 2.8 + 13.9,
                "fasterrcnn_resnet50_fpn": 41.8, "fasterrcnn_resnet50_fpn_kaggle": 41.8,
                "retinanet_resnet50_fpn": 34.0, "retinanet_resnet50_fpn_kaggle": 34.0,
                "yolov8n": 3.2, "yolov8s": 11.2, "yolov8n-kaggle": 3.2}

    for arch in ["resnet18", "googlenet"]:
        train_info[f"rcnn_{arch}"] = tune_rcnn(arch, rec, props, models_dir, cache_dir, log)

    timing_path = cache_dir / "timing_interleaved.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else retime(cfg, log=log)

    gts = [r["boxes"] for r in rec["test"]]
    det = {}
    for label, (family, on_kaggle, make, key) in MODELS.items():
        log(f"[benchmark] {label}")
        pv, _ = cached_preds(key, "val", rec["val"], make, cache_dir)
        pt, ms = cached_preds(key, "test", rec["test"], make, cache_dir)
        thr = count_threshold(pv, [r["boxes"] for r in rec["val"]])
        counts = [int((s >= thr).sum()) for _, s in pt]
        det[label] = {"family": family, "trained_on_kaggle": on_kaggle, **coco_map(pt, gts),
                      **{f"count_{k.lower()}": v for k, v in count_errors(counts, [len(g) for g in gts]).items()},
                      "count_threshold_from_val": thr, "recall50_at_threshold": recall_at(pt, gts, thr),
                      "ms_per_image_median": timing.get(key, float(np.median(ms))),
                      "ms_per_image_long_run": float(np.median(ms)),
                      "proposal_ms": timing["_selective_search_ms"] if family in ("R-CNN", "Fast R-CNN") else 0.0,
                      "params_M": PARAMS_M[key],
                      **({"training": train_info[key]} if key in train_info else {})}
        log(f"[benchmark] {label}: mAP50 {det[label]['mAP50']:.3f} count MAE {det[label]['count_mae']:.2f} "
            f"{det[label]['ms_per_image_median']:.0f} ms")
    ceiling = {"iou0.5": proposal_recall(props["test"], rec["test"], 0.5),
               "iou0.7": proposal_recall(props["test"], rec["test"], 0.7)}

    # classification: ResNet18 vs GoogLeNet from Requirement 7 (same test classroom)
    cls = json.loads((P("eval_dir") / "classroom_eval.json").read_text())["states_cnn"]
    classifiers = {a: {"macro_f1": c["test"]["macro_f1"], "macro_f1_95CI": c["macro_f1_95CI"], "accuracy": c["test"]["accuracy"],
                       "ms_per_crop": c["ms_per_crop"], "params_M": c["params_M"]} for a, c in cls.items()}

    lines = ["Requirement 8 benchmark on the Kaggle TEST classroom (85 frames, 1,159 students); thresholds from validation",
             f"selective-search proposals (1,000 per frame) cover {ceiling['iou0.5']:.0%} of students at IoU 0.5 and "
             f"{ceiling['iou0.7']:.0%} at IoU 0.7: the ceiling for R-CNN and Fast R-CNN", "",
             f"{'model':27s} {'Kaggle':>6s} {'mAP50':>6s} {'mAP50-95':>9s} {'recall':>7s} {'count MAE':>10s} "
             f"{'ms/img':>8s} {'params M':>9s}"]
    for label, d in det.items():
        lines.append(f"{label:27s} {'yes' if d['trained_on_kaggle'] else 'no':>6s} {d['mAP50']:>6.3f} {d['mAP50_95']:>9.3f} "
                     f"{d['recall50_at_threshold']:>7.3f} {d['count_mae']:>10.2f} {d['ms_per_image_median']:>8.0f} "
                     f"{d['params_M']:>9.1f}")
    lines += ["", "ResNet vs GoogLeNet as attentive/distracted/inactive classifiers (Requirement 7, same test classroom):"]
    for a, c in classifiers.items():
        lines.append(f"  {a:10s} macro-F1 {c['macro_f1']:.3f} [95% CI {c['macro_f1_95CI'][0]:.3f}-{c['macro_f1_95CI'][1]:.3f}]  "
                     f"{c['ms_per_crop']:.0f} ms/crop  {c['params_M']:.1f} M parameters")

    # figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"R-CNN": "C3", "Fast R-CNN": "C1", "Faster R-CNN": "C0", "RetinaNet": "C4", "YOLO": "C2"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for label, d in det.items():
        axes[0].scatter(d["ms_per_image_median"], d["mAP50"], s=40 + 4 * d["params_M"], color=colors[d["family"]],
                        marker="o" if d["trained_on_kaggle"] else "^", alpha=0.85, edgecolor="k")
        dx, dy, ha = {"RetinaNet (fine-tuned)": (-8, 10, "right"), "Faster R-CNN (fine-tuned)": (8, -14, "left"),
                      "YOLOv8n (fine-tuned)": (6, -12, "left")}.get(label, (6, 5, "left"))
        axes[0].annotate(label, (d["ms_per_image_median"], d["mAP50"]), fontsize=7.5, xytext=(dx, dy), ha=ha,
                         textcoords="offset points")
    axes[0].axhline(ceiling["iou0.5"], color="C3", ls=":", lw=1)
    axes[0].text(0.02, ceiling["iou0.5"] + 0.008, "selective-search limit for R-CNN / Fast R-CNN",
                 color="C3", fontsize=7.5, ha="left", transform=axes[0].get_yaxis_transform())
    axes[0].set_ylim(0.05, 0.72)
    axes[0].set(xscale="log", xlabel="milliseconds per frame (CPU, log scale)", ylabel="mAP50 (test classroom)",
                title="Accuracy vs speed (o = trained on Kaggle, ^ = COCO only; size = parameters)")
    axes[0].grid(alpha=0.3)
    names = list(det)
    y = np.arange(len(names))
    axes[1].barh(y, [det[n]["count_mae"] for n in names], color=[colors[det[n]["family"]] for n in names])
    axes[1].set_yticks(y, names, fontsize=8), axes[1].invert_yaxis()
    axes[1].set(xlabel="count error (students per frame, lower is better)", title="Counting students")
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    paths = [fig_dir / "benchmark_detectors.png"]
    fig.savefig(paths[0], dpi=110, bbox_inches="tight")
    plt.close(fig)

    result = {"detectors": det, "proposal_recall_ceiling": ceiling, "classifiers": classifiers,
              "val_frames_for_thresholds": len(rec["val"]), "figures": [str(p) for p in paths]}
    out = P("eval_dir") / "benchmark.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    lines += ["", f"saved {out}"] + [f"figure: {p}" for p in paths]
    return lines, result
