"""Requirement 10: integration and evaluation of the classroom pipeline on the Kaggle TEST classroom.

    python -m smartattend evaluate

Pipeline under test: fine-tuned YOLOv8n (students) -> ResNet18 (attentive / distracted / inactive), both chosen on
validation in Requirements 7-8. Measured here, on the 85 test frames (1,159 labelled students):
  robustness      the same frames in low light, motion blur, sensor noise, heavy JPEG, half resolution
  fairness        who is served worse: by student size in the image (back rows) and by the student's true state
  error analysis  missed students, false boxes (duplicates vs background), state confusions, a gallery of worst frames
  latency         per frame on CPU (median, p95), for the classroom pipeline and for the full app, plus peak memory
and one scorecard that also pulls in mAP, count error, macro-F1, attendance and fairness from the earlier stages.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

STATES = ["attentive", "distracted", "inactive"]


def _jpeg(im, q=15):
    return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _noise(im, sigma=25, seed=0):
    return np.clip(im.astype(np.float32) + np.random.default_rng(seed).normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


def _half(im):
    h, w = im.shape[:2]
    return cv2.resize(cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)


CORRUPTIONS = {
    "clean": lambda im: im,
    "low light (30% brightness)": lambda im: cv2.convertScaleAbs(im, alpha=0.3, beta=0),
    "motion blur (15 px)": lambda im: cv2.filter2D(im, -1, np.ones((1, 15), np.float32) / 15),
    "sensor noise (sigma 25)": _noise,
    "heavy JPEG (quality 15)": _jpeg,
    "half resolution": _half,
}


def gt_crops(img_bgr: np.ndarray, boxes: np.ndarray, size: int = 160):
    """Student crops exactly as Requirement 2 made them (plain box crop), with the index of each kept box."""
    H, W = img_bgr.shape[:2]
    crops, keep = [], []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        c = img_bgr[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))]
        if c.size and min(c.shape[:2]) >= 16:
            crops.append(cv2.resize(c[:, :, ::-1], (size, size)))
            keep.append(i)
    return crops, keep


def match(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5):
    """Greedy one-to-one matching (predictions are already sorted by score).
    Returns matched gt index per prediction (-1 = false positive) and the best IoU of every prediction."""
    from smartattend.vision.rcnn import box_iou

    if not len(pred) or not len(gt):
        return np.full(len(pred), -1), np.zeros(len(pred))
    iou = box_iou(pred, gt)
    used, out = set(), np.full(len(pred), -1)
    for i in range(len(pred)):
        for j in np.argsort(-iou[i]):
            if iou[i, j] < iou_thr:
                break
            if j not in used:
                used.add(j)
                out[i] = j
                break
    return out, iou.max(1)


def size_groups(areas: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(areas, edges)            # 0 = smallest third (back rows), 2 = largest third (front rows)


def run(cfg: dict, log=print) -> tuple[list[str], dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import psutil
    import torch

    from smartattend.eval.metrics import classification_report, coco_map, count_errors
    from smartattend.vision import classroom_eval as ce
    from smartattend.vision.detectors import make_detector

    P = lambda k: Path(cfg["paths"][k])
    processed, models_dir, eval_dir = P("data_processed") / "kaggle_classroom", P("models_dir"), P("eval_dir")
    fig_dir = P("figures_dir") / "eval"
    fig_dir.mkdir(parents=True, exist_ok=True)
    recs = ce.load_records(processed, "test")
    thr = json.loads((eval_dir / "classroom_eval.json").read_text())["detection"]["yolov8n-kaggle"]["count_threshold_from_val"]
    det = make_detector("yolov8n-kaggle", models_dir)
    net = ce._net("resnet18")
    net.load_state_dict(torch.load(models_dir / "classroom_states_resnet18.pt", map_location="cpu"))
    idx = {s: i for i, s in enumerate(STATES)}
    gts = [r["boxes"] for r in recs]
    det(cv2.imread(str(recs[0]["path"])), 0.05)                           # warm-up
    ce.predict_cnn(net, np.zeros((2, 160, 160, 3), np.uint8))

    # ---- robustness (and, on clean frames, everything needed for fairness / error analysis / latency)
    robust, clean = {}, {}
    for name, fn in CORRUPTIONS.items():
        preds, y_true, y_pred, t_det, t_state, crop_areas = [], [], [], [], [], []
        for r in recs:
            img = fn(cv2.imread(str(r["path"])))
            t0 = time.perf_counter()
            b, s = det(img, 0.05)
            t_det.append((time.perf_counter() - t0) * 1000)
            order = np.argsort(-s)
            preds.append((b[order], s[order]))
            crops, keep = gt_crops(img, r["boxes"])
            if crops:
                y_pred += ce.predict_cnn(net, np.stack(crops)).argmax(1).tolist()
                y_true += [idx[r["states"][k]] for k in keep]
                crop_areas += [float((r["boxes"][k][2] - r["boxes"][k][0]) * (r["boxes"][k][3] - r["boxes"][k][1])) for k in keep]
            if name == "clean":                                           # app-style timing: classify the DETECTED students
                kept = b[s >= thr]
                dcrops, _ = gt_crops(img, kept)
                t0 = time.perf_counter()
                if dcrops:
                    ce.predict_cnn(net, np.stack(dcrops))
                t_state.append((time.perf_counter() - t0) * 1000)
        counts = [int((s >= thr).sum()) for _, s in preds]
        rep = classification_report(y_true, y_pred, 3)
        robust[name] = {"mAP50": coco_map(preds, gts)["mAP50"],
                        "count_mae": count_errors(counts, [len(g) for g in gts])["MAE"],
                        "state_macro_f1": rep["macro_f1"], "state_accuracy": rep["accuracy"]}
        log(f"[evaluate] {name}: mAP50 {robust[name]['mAP50']:.3f} count MAE {robust[name]['count_mae']:.2f} "
            f"state macro-F1 {rep['macro_f1']:.3f}")
        if name == "clean":
            clean = {"preds": preds, "y_true": np.array(y_true), "y_pred": np.array(y_pred), "t_det": t_det,
                     "t_state": t_state, "counts": counts, "crop_areas": crop_areas}

    # ---- fairness and error analysis on clean frames
    areas_all = np.concatenate([(g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) for g in gts])
    edges = np.quantile(areas_all, [1 / 3, 2 / 3])
    size_names = ["small (back rows)", "medium", "large (front rows)"]
    found, gsize, gstate = [], [], []
    fp_kind = {"duplicate of a found student": 0, "poorly placed box on a student": 0, "background": 0}
    frame_err = []
    for r, (b, s) in zip(recs, clean["preds"]):
        kept = b[s >= thr]
        m, best = match(kept, r["boxes"])
        hit = np.zeros(len(r["boxes"]), bool)
        hit[m[m >= 0]] = True
        found += hit.tolist()
        a = (r["boxes"][:, 2] - r["boxes"][:, 0]) * (r["boxes"][:, 3] - r["boxes"][:, 1])
        gsize += size_groups(a, edges).tolist()
        gstate += [idx[x] for x in r["states"]]
        for i in np.flatnonzero(m < 0):
            if best[i] >= 0.5:
                fp_kind["duplicate of a found student"] += 1
            elif best[i] >= 0.1:
                fp_kind["poorly placed box on a student"] += 1
            else:
                fp_kind["background"] += 1
        frame_err.append(int((~hit).sum() + (m < 0).sum()))
    found, gsize, gstate = np.array(found), np.array(gsize), np.array(gstate)
    det_by_size = {size_names[k]: float(found[gsize == k].mean()) for k in range(3)}
    det_by_state = {STATES[k]: float(found[gstate == k].mean()) for k in range(3)}
    # state classifier by size: recompute sizes for the students that produced crops
    cs = size_groups(np.array(clean["crop_areas"]), edges)
    yt, yp = clean["y_true"], clean["y_pred"]
    state_by_size = {size_names[k]: {"accuracy": float((yt[cs == k] == yp[cs == k]).mean()),
                                     "macro_f1": classification_report(yt[cs == k], yp[cs == k], 3)["macro_f1"],
                                     "n": int((cs == k).sum())} for k in range(3)}
    gap = lambda d: float(max(d.values()) - min(d.values()))
    fairness = {"detection_recall_by_size": det_by_size, "detection_recall_gap_size": gap(det_by_size),
                "detection_recall_by_true_state": det_by_state, "detection_recall_gap_state": gap(det_by_state),
                "state_by_size": state_by_size,
                "state_accuracy_gap_size": gap({k: v["accuracy"] for k, v in state_by_size.items()}),
                "size_edges_px2": edges.tolist()}
    cm = np.array(classification_report(yt, yp, 3)["confusion"])
    errors = {"students": int(len(found)), "missed": int((~found).sum()), "false_boxes": fp_kind,
              "missed_by_size": {size_names[k]: int((~found[gsize == k]).sum()) for k in range(3)},
              "missed_by_state": {STATES[k]: int((~found[gstate == k]).sum()) for k in range(3)},
              "state_confusion": cm.tolist(),
              "largest_state_confusions": sorted(
                  [{"true": STATES[i], "predicted": STATES[j], "n": int(cm[i, j]), "share_of_true": float(cm[i, j] / cm[i].sum())}
                   for i in range(3) for j in range(3) if i != j], key=lambda d: -d["n"])[:3]}

    # ---- latency
    tot = np.array(clean["t_det"]) + np.array(clean["t_state"])
    q = lambda v: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
    latency = {"classroom_pipeline": {"detect": q(clean["t_det"]), "states": q(clean["t_state"]), "total": q(tot),
                                      "frames_per_second_median": float(1000 / np.median(tot))}}
    del det, net
    from smartattend.pipeline import ClassroomAnalyzer

    for dname in [cfg["product"]["detector"], "yolov8n-kaggle"]:
        an = ClassroomAnalyzer(cfg, roster=None, detector=dname)
        an.analyze_frame(cv2.imread(str(recs[0]["path"])))
        tt, parts, seats, peak = [], {k: [] for k in ["detect", "faces", "engagement", "seating"]}, [], 0.0
        for r in recs[::3]:
            img = cv2.imread(str(r["path"]))
            res = an.analyze_frame(img)
            tt.append(res["timing_ms"]["total"])
            for k in parts:
                parts[k].append(res["timing_ms"][k])
            # chairs / benches the detector reports (outside the timing): does this detector still see furniture?
            seats.append(len(an.detector.detect_with_seats(img, an.det_thr)[2]) if hasattr(an.detector, "detect_with_seats") else 0)
            peak = max(peak, psutil.Process().memory_info().rss / 1e6)
        latency[f"full_app_{dname}"] = {"total": q(tt), **{k: q(v) for k, v in parts.items()}, "peak_rss_mb": peak,
                                        "mean_seats_found": float(np.mean(seats)), "frames": len(tt)}
        del an
    battery = None
    try:
        b = psutil.sensors_battery()
        battery = None if b is None else {"plugged_in": bool(b.power_plugged), "percent": b.percent}
    except Exception:
        pass
    latency["machine"] = {"cpu_mhz_now": getattr(psutil.cpu_freq(), "current", None), "battery": battery}

    # ---- scorecard across all stages
    def load(rel):
        p = Path(rel)
        return json.loads(p.read_text()) if p.exists() else None

    card = []
    add = lambda area, metric, value, where: card.append({"area": area, "metric": metric, "value": value, "data": where})
    fr = load(eval_dir / "face_recognition.json")
    if fr:
        add("Attendance", "recall of enrolled students", fr["sessions"]["attendance_recall"], "LFW simulated sessions")
        add("Attendance", "false-present rate", fr["sessions"]["false_present_rate"], "LFW simulated sessions")
        add("Attendance", "stranger false-accept rate", fr["verification"]["FAR"], "LFW pairs")
        add("Fairness", "attendance recall gap (gender / ethnicity)",
            max(fr["sessions"]["recall_gap_gender"]["gap"], fr["sessions"]["recall_gap_ethnicity"]["gap"]), "LFW attributes")
    db = load(P("experiments_dir") / "detection" / "detection_benchmark.json")
    if db:
        d0 = db[cfg["product"]["detector"]]
        add("Detection", f"mAP50 ({cfg['product']['detector']}, varied photos)", d0["mAP50"], "COCO-classroom test")
        add("Counting", "count error, fully labelled photos (people)", d0.get("count_MAE_no_crowd"), "COCO-classroom test")
    add("Detection", "mAP50 (fine-tuned YOLOv8n)", robust["clean"]["mAP50"], "Kaggle test classroom")
    add("Counting", "count error (students per frame)", robust["clean"]["count_mae"], "Kaggle test classroom")
    add("Engagement", "state macro-F1 (ResNet18, 3 states)", robust["clean"]["state_macro_f1"], "Kaggle test classroom")
    ecv = load(P("experiments_dir") / "engagement" / "engagement_cv.json")
    if ecv:
        add("Engagement", "alertness macro-F1 (5-fold CV, one person)", max(ecv["resnet18"]["mean"], ecv["googlenet"]["mean"]),
            "drowsiness frames")
    sq = load(eval_dir / "sequence_models.json")
    if sq:
        add("Trend", "correlation of predicted vs labelled attentive share (moving average)",
            next(v["corr"] for k, v in sq["temporal"].items() if k.startswith("moving average")), "Kaggle test classroom")
    add("Fairness", "detection recall gap, small vs large students", fairness["detection_recall_gap_size"], "Kaggle test classroom")
    add("Fairness", "state accuracy gap, small vs large students", fairness["state_accuracy_gap_size"], "Kaggle test classroom")
    worst = min((k for k in robust if k != "clean"), key=lambda k: robust[k]["mAP50"])
    add("Robustness", f"mAP50 in the worst condition ({worst})", robust[worst]["mAP50"], "Kaggle test classroom")
    add("Robustness", "mAP50 in low light", robust["low light (30% brightness)"]["mAP50"], "Kaggle test classroom")
    add("Latency", "classroom pipeline per frame, p95 (ms)", latency["classroom_pipeline"]["total"]["p95"], "this laptop, CPU")
    add("Latency", f"full app per frame, p95 (ms, {cfg['product']['detector']})",
        latency[f"full_app_{cfg['product']['detector']}"]["total"]["p95"], "this laptop, CPU")
    add("Memory", "peak memory, full app (MB)", latency[f"full_app_{cfg['product']['detector']}"]["peak_rss_mb"], "this laptop")

    result = {"pipeline": {"detector": "yolov8n-kaggle", "count_threshold": thr, "state_model": "resnet18 (fine-tuned)"},
              "robustness": robust, "fairness": fairness, "errors": errors, "latency": latency, "scorecard": card}

    # ---- figures
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    names = list(robust)
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [robust[n]["mAP50"] for n in names], 0.4, label="detection mAP50")
    axes[0].bar(x + 0.2, [robust[n]["state_macro_f1"] for n in names], 0.4, label="state macro-F1")
    axes[0].set_xticks(x, [n.split(" (")[0] for n in names], rotation=20, ha="right"), axes[0].set_ylim(0, 0.8)
    axes[0].set_title("Robustness: the same test frames, degraded"), axes[0].legend(fontsize=8)
    ax2 = axes[0].twinx()
    ax2.plot(x, [robust[n]["count_mae"] for n in names], "ko--", ms=4, label="count error")
    ax2.set_ylabel("count error (students)"), ax2.legend(fontsize=8, loc="upper left")
    w = 0.38
    axes[1].bar(np.arange(3) - w / 2, list(det_by_size.values()), w, label="found by the detector (recall)")
    axes[1].bar(np.arange(3) + w / 2, [v["accuracy"] for v in state_by_size.values()], w, label="state classified correctly")
    axes[1].set_xticks(np.arange(3), size_names), axes[1].set_ylim(0, 1)
    axes[1].set_title("Who is served worse? By student size in the image"), axes[1].legend(fontsize=8)
    kinds = ["missed students", *fp_kind]
    vals = [errors["missed"], *fp_kind.values()]
    axes[2].barh(kinds, vals, color=["C3", "C1", "C1", "C1"])
    axes[2].invert_yaxis(), axes[2].set_title(f"Detection errors on {errors['students']} students (85 frames)")
    for i, v in enumerate(vals):
        axes[2].text(v, i, f" {v}", va="center")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    paths.append(fig_dir / "final_robustness_fairness_errors.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    worst_frames = np.argsort(frame_err)[::-1][:3]
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.2))
    for ax, i in zip(axes, worst_frames):
        r, (b, s) = recs[i], clean["preds"][i]
        img = cv2.imread(str(r["path"]))
        kept = b[s >= thr]
        m, _ = match(kept, r["boxes"])
        hit = np.zeros(len(r["boxes"]), bool)
        hit[m[m >= 0]] = True
        for (x0, y0, x1, y1), ok in zip(r["boxes"].astype(int), hit):
            if not ok:
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 5)                 # missed student
        for (x0, y0, x1, y1), j in zip(kept.astype(int), m):
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0) if j >= 0 else (0, 220, 255), 3)
        ax.imshow(img[:, :, ::-1]), ax.axis("off")
        ax.set_title(f"frame {r['frame']}: {int((~hit).sum())} missed (red), {int((m < 0).sum())} false boxes (yellow), "
                     f"{int((m >= 0).sum())} correct (green)", fontsize=9)
    fig.tight_layout()
    paths.append(fig_dir / "final_error_gallery.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)
    result["figures"] = [str(p) for p in paths]
    (eval_dir / "final_evaluation.json").write_text(json.dumps(result, indent=2, default=float))

    # ---- text
    lines = ["Requirement 10: final evaluation on the Kaggle TEST classroom (85 frames, 1,159 students)",
             f"pipeline: fine-tuned YOLOv8n (count threshold {thr} from validation) -> ResNet18 state classifier", "",
             "Robustness (same frames, degraded)",
             f"   {'condition':30s} {'mAP50':>6s} {'count MAE':>10s} {'state macro-F1':>15s}"]
    lines += [f"   {n:30s} {v['mAP50']:>6.3f} {v['count_mae']:>10.2f} {v['state_macro_f1']:>15.3f}" for n, v in robust.items()]
    lines += ["", "Fairness: who is served worse?",
              "   students found by the detector, by size: " + ", ".join(f"{k} {v:.2f}" for k, v in det_by_size.items())
              + f"   (gap {fairness['detection_recall_gap_size']:.2f})",
              "   students found, by their true state: " + ", ".join(f"{k} {v:.2f}" for k, v in det_by_state.items())
              + f"   (gap {fairness['detection_recall_gap_state']:.2f})",
              "   state classified correctly, by size: " + ", ".join(f"{k} {v['accuracy']:.2f}" for k, v in state_by_size.items())
              + f"   (gap {fairness['state_accuracy_gap_size']:.2f})",
              "", "Error analysis",
              f"   missed students: {errors['missed']} of {errors['students']} ({errors['missed'] / errors['students']:.0%}); by size: "
              + ", ".join(f"{k} {v}" for k, v in errors["missed_by_size"].items()),
              "   false boxes: " + ", ".join(f"{k} {v}" for k, v in fp_kind.items()),
              "   largest state confusions: " + "; ".join(
                  f"{d['true']} -> {d['predicted']} {d['n']} ({d['share_of_true']:.0%} of {d['true']})" for d in errors["largest_state_confusions"]),
              "", f"Latency on this laptop (CPU{'' if not battery else ', plugged in' if battery['plugged_in'] else ', ON BATTERY: worst case'})"]
    cp = latency["classroom_pipeline"]
    lines.append(f"   classroom pipeline: detect {cp['detect']['median']:.0f} ms + states {cp['states']['median']:.0f} ms = "
                 f"{cp['total']['median']:.0f} ms median, {cp['total']['p95']:.0f} ms p95 ({cp['frames_per_second_median']:.1f} frames/s)")
    for k, v in latency.items():
        if k.startswith("full_app_"):
            lines.append(f"   full app ({k[9:]}): {v['total']['median']:.0f} ms median, {v['total']['p95']:.0f} ms p95 "
                         f"(detect {v['detect']['median']:.0f}, faces {v['faces']['median']:.0f}, engagement {v['engagement']['median']:.0f}); "
                         f"peak memory {v['peak_rss_mb']:.0f} MB; chairs/benches detected per frame {v['mean_seats_found']:.1f}")
    lines += ["", "Scorecard", f"   {'area':11s} {'metric':72s} {'value':>9s}  data"]
    for c in card:
        v = c["value"]
        lines.append(f"   {c['area']:11s} {c['metric']:72s} {('-' if v is None else f'{v:.3f}' if abs(v) < 10 else f'{v:.0f}'):>9s}  {c['data']}")
    lines += ["", f"saved {eval_dir / 'final_evaluation.json'}"] + [f"figure: {p}" for p in paths]
    return lines, result
