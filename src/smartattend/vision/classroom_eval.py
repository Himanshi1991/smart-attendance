"""Requirement 7 evaluation on the Kaggle classroom TEST set (a classroom never used for training or for choices).

    python -m smartattend classroom

  7a people counting / detection : YOLOv8n, YOLOv8s, Faster R-CNN (COCO-pretrained) and YOLOv8n fine-tuned on the
                                   Kaggle training classrooms. mAP50 against the Kaggle boxes, count error with a
                                   score threshold chosen on VALIDATION.
  7b attentive / distracted / inactive per student (on the labelled student boxes, so detection errors do not mix in):
                                   the app's face rule (head turn + alertness ResNet) vs ResNet18 and GoogLeNet
                                   fine-tuned on the Kaggle student crops.
  7c trends over time            : the classroom video frames in order: predicted vs labelled count and attentive share.
  7d 2D / 3D convolution figures : on a real frame of the test classroom.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

STATES = ["attentive", "distracted", "inactive"]


# --------------------------------------------------------------------------- data
def load_records(processed: str | Path, split: str) -> list[dict]:
    """Each record: image path, ground-truth boxes (xyxy, pixels), states, frame order key."""
    processed = Path(processed)
    recs = json.loads((processed / f"{split}.json").read_text())
    out = []
    for r in recs:
        name = Path(r["file"]).name
        m = re.search(r"_mp4-(\d+)_jpg", name)
        out.append({"path": processed / "yolo_person" / "images" / split / name, "video": r["video"],
                    "frame": int(m.group(1)) if m else 0,
                    "boxes": np.array([b["xyxy"] for b in r["boxes"]], float).reshape(-1, 4),
                    "states": [b["state"] for b in r["boxes"]]})
    return sorted(out, key=lambda r: (r["video"], r["frame"]))


# --------------------------------------------------------------------------- 7a detection
def predictions(det, records: list[dict], cache: Path) -> tuple[list, list[float]]:
    """Run the detector once at a low threshold (0.05) and cache boxes/scores, so thresholds can be chosen later."""
    if cache.exists():
        d = json.loads(cache.read_text())
        return [(np.array(b).reshape(-1, 4), np.array(s)) for b, s in d["preds"]], d["ms"]
    preds, ms = [], []
    if records:                                    # warm-up: the first call loads weights and allocates memory
        det.timed(cv2.imread(str(records[0]["path"])), 0.05)
    for r in records:
        (b, s), t = det.timed(cv2.imread(str(r["path"])), 0.05)
        preds.append((np.asarray(b).reshape(-1, 4), np.asarray(s)))
        ms.append(t)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"preds": [(b.tolist(), s.tolist()) for b, s in preds], "ms": ms}))
    return preds, ms


def evaluate_detectors(processed: Path, models_dir: Path, cache_dir: Path, names: list[str], log=print) -> dict:
    from smartattend.eval.metrics import coco_map, count_errors, recall_at
    from smartattend.vision.benchmark import best_count_threshold
    from smartattend.vision.detectors import make_detector

    val, test = load_records(processed, "val"), load_records(processed, "test")
    out = {}
    for name in names:
        det = None
        res = {}
        for split, recs in [("val", val), ("test", test)]:
            cache = cache_dir / f"preds_{name}_{split}.json"
            if not cache.exists() and det is None:
                det = make_detector(name, models_dir)
            res[split] = predictions(det, recs, cache)
        thr = best_count_threshold(res["val"][0], [r["boxes"] for r in val])
        preds, ms = res["test"]
        gts = [r["boxes"] for r in test]
        counts = [int((s >= thr).sum()) for _, s in preds]
        out[name] = {**coco_map(preds, gts), "count_threshold_from_val": thr,
                     **{f"count_{k.lower()}": v for k, v in count_errors(counts, [len(g) for g in gts]).items()},
                     "recall50_at_threshold": recall_at(preds, gts, thr), "ms_per_image_median": float(np.median(ms)),
                     "pred_counts": counts}
        log(f"[classroom] {name}: mAP50={out[name]['mAP50']:.3f} count MAE={out[name]['count_mae']:.2f} "
            f"(thr {thr} from val), {np.median(ms):.0f} ms/img")
    out["true_counts"] = [len(r["boxes"]) for r in test]
    return out


# --------------------------------------------------------------------------- 7b states: the app's face rule
def face_rule_states(processed: Path, models_dir: Path, cfg: dict) -> dict:
    """Apply the app's per-face rule to every labelled student: find faces in the full frame, give each face to the
    student box containing its centre, then engagement_state(...). Students with no detectable face -> 'no face'."""
    from smartattend.attendance.face import FaceEngine, ensure_models
    from smartattend.pipeline import engagement_state, face_crop, head_turn
    from smartattend.vision.engagement import EngagementClassifier

    eng = FaceEngine(*ensure_models(models_dir), score_threshold=cfg["face"]["detector_score_threshold"])
    clf = EngagementClassifier(models_dir / "engagement_resnet18.pt")
    es = cfg.get("engagement_states", {})
    y_true, y_pred = [], []
    for r in load_records(processed, "test"):
        img = cv2.imread(str(r["path"]))
        faces = eng.analyze(img)
        probs = clf.predict_proba([face_crop(img, f.box) for f in faces]) if faces else np.zeros((0, 3))
        pred = ["no face"] * len(r["boxes"])
        best = {}
        for fi, f in enumerate(faces):
            cx, cy = f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2
            inside = [i for i, b in enumerate(r["boxes"]) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
            if inside:
                i = min(inside, key=lambda i: (r["boxes"][i][2] - r["boxes"][i][0]) * (r["boxes"][i][3] - r["boxes"][i][1]))
                if i not in best or f.raw[-1] > faces[best[i]].raw[-1]:
                    best[i] = fi
        for i, fi in best.items():
            f = faces[fi]
            pred[i] = engagement_state(probs[fi], head_turn(f.raw), es.get("distracted_head_turn", 0.30),
                                       int(f.box[2]), es.get("min_face_px", 30))
        y_true += r["states"]
        y_pred += pred
    return {"y_true": y_true, "y_pred": y_pred}


# --------------------------------------------------------------------------- 7b states: fine-tuned CNNs
def _crop_arrays(processed: Path, split: str, size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for label, state in enumerate(STATES):
        for f in sorted((processed / "crops" / split / state).glob("*.jpg")):
            X.append(cv2.resize(cv2.imread(str(f))[:, :, ::-1], (size, size)))
            y.append(label)
    return np.stack(X), np.array(y)


def _net(arch: str):
    import torch
    import torchvision

    if arch == "resnet18":
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        trainable = ["layer4", "fc"]
    else:
        net = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)
        net.aux_logits, net.aux1, net.aux2 = False, None, None
        trainable = ["inception5a", "inception5b", "fc"]
    net.fc = torch.nn.Linear(net.fc.in_features, 3)
    for n, p in net.named_parameters():
        p.requires_grad = any(n.startswith(t) for t in trainable)
    return net


def _to_tensor(X: np.ndarray):
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0 - mean) / std


def _augment(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = X.copy()
    flip = rng.random(len(X)) < 0.5
    X[flip] = X[flip, :, ::-1]
    gain = rng.uniform(0.75, 1.25, size=(len(X), 1, 1, 1))
    return np.clip(X.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def predict_cnn(net, X: np.ndarray, batch: int = 64) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(torch.softmax(net(_to_tensor(X[s:s + batch])), 1).numpy())
    return np.concatenate(out)


def finetune_cnn(arch: str, data: dict, epochs: int = 6, lr: float = 1e-3, batch: int = 32, seed: int = 0,
                 log=print) -> tuple[object, dict]:
    """Transfer learning: ImageNet weights, last block + new 3-class head trained; class-weighted loss
    (inactive has only 70 training crops); the epoch with the best VALIDATION macro-F1 is kept."""
    import torch

    from smartattend.eval.metrics import macro_f1

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    (Xtr, ytr), (Xva, yva) = data["train"], data["val"]
    net = _net(arch)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {"train_loss": [], "val_f1": [], "epoch_s": []}
    best = (-1.0, None)
    for ep in range(epochs):
        t0 = time.perf_counter()
        net.train()
        order = rng.permutation(len(Xtr))
        losses = []
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            loss = loss_fn(net(_to_tensor(_augment(Xtr[b], rng))), torch.from_numpy(ytr[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        f1 = macro_f1(yva, predict_cnn(net, Xva).argmax(1), 3)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_f1"].append(f1)
        hist["epoch_s"].append(time.perf_counter() - t0)
        log(f"[classroom] {arch} epoch {ep + 1}/{epochs} loss {hist['train_loss'][-1]:.3f} val macro-F1 {f1:.3f} "
            f"({hist['epoch_s'][-1]:.0f} s)")
        if f1 > best[0]:
            best = (f1, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    hist["best_epoch"] = int(np.argmax(hist["val_f1"])) + 1
    return net, hist


def bootstrap_ci(y_true, y_pred, n: int = 1000, seed: int = 0) -> list[float]:
    from smartattend.eval.metrics import macro_f1

    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = [macro_f1(y_true[i], y_pred[i], 3) for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# --------------------------------------------------------------------------- run everything
def run(cfg: dict, epochs: int = 6, log=print) -> tuple[list[str], dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from smartattend.eval.metrics import classification_report

    P = lambda k: Path(cfg["paths"][k])
    processed, models_dir = P("data_processed") / "kaggle_classroom", P("models_dir")
    exp_dir, fig_dir = P("experiments_dir") / "classroom", P("figures_dir") / "vision"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    result, lines = {}, ["Requirement 7 on the Kaggle classroom TEST set (taimoor-zaeem: 85 frames, 1,159 labelled students)", ""]

    # 7a
    names = ["yolov8n", "yolov8s", "fasterrcnn_resnet50_fpn"]
    if (models_dir / "yolo" / "yolov8n-kaggle.pt").exists():
        names.append("yolov8n-kaggle")
    det = evaluate_detectors(processed, models_dir, exp_dir, names, log)
    result["detection"] = det
    true_counts = np.array(det["true_counts"])
    lines += ["7a) Counting students (threshold for counting chosen on the VALIDATION classroom)",
              f"   students per frame: mean {true_counts.mean():.1f}, range {true_counts.min()}-{true_counts.max()}",
              f"   {'detector':26s} {'mAP50':>6s} {'mAP50-95':>9s} {'count MAE':>10s} {'bias':>6s} {'ms/img':>7s}",
              "   (ms/img = median after one warm-up image, laptop on battery at 1.2 GHz: worst case)"]
    for n in names:
        d = det[n]
        lines.append(f"   {n:26s} {d['mAP50']:>6.3f} {d['mAP50_95']:>9.3f} {d['count_mae']:>10.2f} {d['count_bias']:>+6.2f} "
                     f"{d['ms_per_image_median']:>7.0f}")

    # 7b
    from smartattend.eval.metrics import macro_f1

    t0 = time.perf_counter()
    rule = face_rule_states(processed, models_dir, cfg)
    yt, yp = np.array(rule["y_true"]), np.array(rule["y_pred"])
    judged = np.isin(yp, STATES)
    idx = {s: i for i, s in enumerate(STATES)}
    rep_rule = classification_report([idx[s] for s in yt[judged]], [idx[s] for s in yp[judged]], 3)
    # counting 'no face' / 'unclear' as attentive (the app's default display) gives a full-coverage score
    yp_full = [idx.get(s, 0) for s in yp]
    rep_rule_full = classification_report([idx[s] for s in yt], yp_full, 3)
    cover = {s: int((yp == s).sum()) for s in [*STATES, "unclear", "no face"]}
    result["states_face_rule"] = {"coverage": cover, "judged_fraction": float(judged.mean()),
                                  "judged": rep_rule, "all_students_unjudged_as_attentive": rep_rule_full}
    log(f"[classroom] face rule done in {time.perf_counter() - t0:.0f}s")

    data = {s: _crop_arrays(processed, s) for s in ["train", "val", "test"]}
    yte = data["test"][1]
    cnn = {}
    for arch in ["resnet18", "googlenet"]:
        wpath = models_dir / f"classroom_states_{arch}.pt"
        hist_path = exp_dir / f"states_{arch}_history.json"
        if wpath.exists() and hist_path.exists():
            net = _net(arch)
            net.load_state_dict(torch.load(wpath, map_location="cpu"))
            hist = json.loads(hist_path.read_text())
        else:
            net, hist = finetune_cnn(arch, data, epochs=epochs, log=log)
            torch.save(net.state_dict(), wpath)
            hist_path.write_text(json.dumps(hist))
        t0 = time.perf_counter()
        P_te = predict_cnn(net, data["test"][0])
        ms = (time.perf_counter() - t0) / len(yte) * 1000
        rep = classification_report(yte, P_te.argmax(1), 3)
        cnn[arch] = {"test": rep, "macro_f1_95CI": bootstrap_ci(yte, P_te.argmax(1)), "val_f1_best": max(hist["val_f1"]),
                     "best_epoch": hist["best_epoch"], "ms_per_crop": ms, "history": hist,
                     "params_M": sum(p.numel() for p in net.parameters()) / 1e6,
                     "trainable_params_M": sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6}
    result["states_cnn"] = cnn
    majority = classification_report(yte, np.zeros_like(yte), 3)
    lines += ["", "7b) Attentive / distracted / inactive for each labelled student (test classroom, 1,159 students)",
              f"   always 'attentive' baseline: macro-F1 {majority['macro_f1']:.3f}",
              f"   app's face rule: a face was found and judged for {judged.mean():.0%} of students "
              f"(no face {cover['no face']}, too small {cover['unclear']})",
              f"      on judged students: accuracy {rep_rule['accuracy']:.3f}  macro-F1 {rep_rule['macro_f1']:.3f};  "
              f"all students (unjudged = attentive): macro-F1 {rep_rule_full['macro_f1']:.3f}"]
    for arch, c in cnn.items():
        lines.append(f"   {arch:9s} fine-tuned on Kaggle crops: val {c['val_f1_best']:.3f} (epoch {c['best_epoch']})  TEST accuracy "
                     f"{c['test']['accuracy']:.3f}  macro-F1 {c['test']['macro_f1']:.3f} "
                     f"[95% CI {c['macro_f1_95CI'][0]:.3f}-{c['macro_f1_95CI'][1]:.3f}]  "
                     f"F1 per class {' / '.join(f'{v:.2f}' for v in c['test']['f1'])}  {c['ms_per_crop']:.0f} ms/crop")

    # 7c trends: per frame, true vs predicted count and attentive share (best detector + best CNN on detected boxes)
    best_det = min(names, key=lambda n: det[n]["count_mae"])
    best_cnn = max(cnn, key=lambda a: cnn[a]["val_f1_best"])          # chosen on validation
    net = _net(best_cnn)
    net.load_state_dict(torch.load(models_dir / f"classroom_states_{best_cnn}.pt", map_location="cpu"))
    test = load_records(processed, "test")
    preds, _ = predictions(None, test, exp_dir / f"preds_{best_det}_test.json")
    thr = det[best_det]["count_threshold_from_val"]
    timeline = []
    for r, (b, s) in zip(test, preds):
        img = cv2.imread(str(r["path"]))[:, :, ::-1]
        keep = b[s >= thr]
        crops = [cv2.resize(img[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)], (160, 160))
                 for x0, y0, x1, y1 in keep if x1 - x0 >= 4 and y1 - y0 >= 4]
        st = predict_cnn(net, np.stack(crops)).argmax(1) if crops else np.zeros(0, int)
        timeline.append({"video": r["video"], "frame": r["frame"], "true_count": len(r["boxes"]), "pred_count": len(keep),
                         "true_attentive": float(np.mean([x == "attentive" for x in r["states"]])) if r["states"] else 0.0,
                         "pred_attentive": float(np.mean(st == 0)) if len(st) else 0.0})
    ta = np.array([t["true_attentive"] for t in timeline])
    pa = np.array([t["pred_attentive"] for t in timeline])
    corr = float(np.corrcoef(ta, pa)[0, 1]) if ta.std() > 0 and pa.std() > 0 else float("nan")
    result["trend"] = {"detector": best_det, "state_model": best_cnn, "frames": timeline,
                       "attentive_share_mae": float(np.mean(np.abs(ta - pa))), "attentive_share_corr": corr,
                       "true_mean_attentive": float(ta.mean()), "pred_mean_attentive": float(pa.mean())}
    lines += ["", f"7c) Trend over the test video, end to end ({best_det} + {best_cnn}, both chosen on validation)",
              f"   attentive share per frame: labelled mean {ta.mean():.2f}, predicted mean {pa.mean():.2f}, "
              f"mean abs. error {np.mean(np.abs(ta - pa)):.2f}, correlation {corr:.2f}"]

    # 7d convolution figures on a real frame of the test classroom
    from smartattend.vision.conv_viz import make_figures

    frame = max(test, key=lambda r: len(r["boxes"]))
    conv_paths = make_figures(frame["path"], fig_dir)

    # figures
    paths = list(conv_paths)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    x = np.arange(len(timeline))
    axes[0].plot(x, [t["true_count"] for t in timeline], "k-", label="labelled")
    for n in names:
        axes[0].plot(x, det[n]["pred_counts"], lw=1, label=f"{n} (MAE {det[n]['count_mae']:.1f})")
    axes[0].set(title="Students counted per frame (test classroom)", xlabel="frame (video order)")
    axes[0].legend(fontsize=7)
    axes[1].plot(x, ta, "k-", label="labelled")
    axes[1].plot(x, pa, "C2-", label=f"predicted ({best_det} + {best_cnn})")
    axes[1].set(title="Share of attentive students over time", xlabel="frame (video order)", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    c = cnn[best_cnn]["test"]["confusion"]
    axes[2].imshow(c, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, c[i][j], ha="center", va="center", color="white" if c[i][j] > np.max(c) / 2 else "black")
    axes[2].set_xticks(range(3), STATES), axes[2].set_yticks(range(3), STATES)
    axes[2].set(title=f"{best_cnn}: test confusion (true rows)", xlabel="predicted")
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    paths.append(fig_dir / "classroom_test_eval.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    # annotated example frame
    img = cv2.imread(str(frame["path"]))
    fr_preds, _ = predictions(None, test, exp_dir / f"preds_{best_det}_test.json")
    b, s = fr_preds[test.index(frame)]
    keep = b[s >= thr]
    rgb = img[:, :, ::-1]
    crops = [cv2.resize(rgb[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)], (160, 160)) for x0, y0, x1, y1 in keep]
    st = predict_cnn(net, np.stack(crops)).argmax(1) if crops else []
    colors = [(0, 200, 0), (0, 165, 255), (0, 0, 230)]
    for (x0, y0, x1, y1), k in zip(keep.astype(int), st):
        cv2.rectangle(img, (x0, y0), (x1, y1), colors[k], 3)
        cv2.putText(img, STATES[k], (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[k], 2)
    cv2.putText(img, f"students {len(keep)} (labelled {len(frame['boxes'])})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 4)
    paths.append(fig_dir / "classroom_test_annotated.jpg")
    cv2.imwrite(str(paths[-1]), img)

    result["figures"] = [str(p) for p in paths]
    eval_dir = P("eval_dir")
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "classroom_eval.json").write_text(json.dumps(result, indent=2, default=float))
    lines += ["", f"saved {eval_dir / 'classroom_eval.json'}"] + [f"figure: {p}" for p in paths]
    return lines, result
