"""Detection benchmark on the classroom test split: accuracy, counting, size-fairness, latency, robustness."""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

from smartattend.data.coco_classroom import gt_boxes_xyxy, load_split
from smartattend.eval.metrics import coco_map, count_errors, detection_map, fairness_gap, recall_at
from smartattend.vision.detectors import DETECTORS, make_detector

SIZE_BUCKETS = {  # COCO area buckets; in a classroom, 'small' ~ back rows
    "small(<32²)": lambda b: (b[2] - b[0]) * (b[3] - b[1]) < 32 ** 2,
    "medium": lambda b: 32 ** 2 <= (b[2] - b[0]) * (b[3] - b[1]) < 96 ** 2,
    "large(>96²)": lambda b: (b[2] - b[0]) * (b[3] - b[1]) >= 96 ** 2,
}
CORRUPTIONS = {
    "low_light": lambda im: cv2.convertScaleAbs(im, alpha=0.3, beta=0),
    "motion_blur": lambda im: cv2.filter2D(im, -1, np.ones((1, 15), np.float32) / 15),
}


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return float("nan")


def run_detector(det, records, img_dir: Path, transform=None):
    preds, lat = [], []
    for r in records:
        img = cv2.imread(str(img_dir / r["file_name"]))
        if transform:
            img = transform(img)
        (b, s), ms = det.timed(img, 0.05)
        preds.append((b, s))
        lat.append(ms)
    return preds, np.array(lat)


def best_count_threshold(preds, gts) -> float:
    """Pick the score threshold that minimises count MAE on the VALIDATION split."""
    grid = np.round(np.arange(0.1, 0.9, 0.05), 2)
    mae = [np.mean([abs(int((s >= t).sum()) - len(g)) for (_, s), g in zip(preds, gts)]) for t in grid]
    return float(grid[int(np.argmin(mae))])


def benchmark(dataset_dir: str | Path, models_dir: str | Path, out_dir: str | Path, logger: logging.Logger,
              names: list[str] | None = None) -> dict:
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val, test = load_split(dataset_dir, "val"), load_split(dataset_dir, "test")
    gv, gt = [gt_boxes_xyxy(r) for r in val], [gt_boxes_xyxy(r) for r in test]
    names = names or [n for n in DETECTORS if n != "yolov8n-classroom" or (Path(models_dir) / "yolo" / "yolov8n-classroom.pt").exists()]
    results = {}
    cache = out_dir / "predictions"
    cache.mkdir(exist_ok=True)
    for name in names:
        gc.collect()
        mem0 = _rss_mb()
        det = make_detector(name, models_dir)
        mem_model = _rss_mb() - mem0
        det(cv2.imread(str(dataset_dir / "images" / "test" / test[0]["file_name"])))  # warm-up
        pv, _ = run_detector(det, val, dataset_dir / "images" / "val")
        pt, lat = run_detector(det, test, dataset_dir / "images" / "test")
        thr = best_count_threshold(pv, gv)
        m = coco_map(pt, gt)
        counts = [int((s >= thr).sum()) for _, s in pt]
        size_recall = {k: recall_at(pt, gt, thr, size_filter=f) for k, f in SIZE_BUCKETS.items()}
        rob = {}
        sub = slice(0, 40)  # robustness on a fixed 40-image subset to bound CPU time
        for cname, fn in CORRUPTIONS.items():
            pc, _ = run_detector(det, test[sub], dataset_dir / "images" / "test", fn)
            rob[cname] = detection_map(pc, gt[sub])[0.5]
        rob["clean"] = detection_map(pt[sub], gt[sub])[0.5]
        results[name] = {
            "description": DETECTORS[name], **m,
            "count_threshold(val)": thr, **{f"count_{k}": v for k, v in count_errors(counts, [len(g) for g in gt]).items()},
            "recall@thr": recall_at(pt, gt, thr),
            "recall_by_size": size_recall, "size_recall_gap": fairness_gap(size_recall)["gap"],
            "latency_ms_mean": float(lat.mean()), "latency_ms_p95": float(np.percentile(lat, 95)),
            "model_memory_mb": mem_model, "robustness_mAP50": rob,
        }
        np.save(cache / f"{name}.npy", np.array([np.hstack([b, s[:, None]]) for b, s in pt], dtype=object),
                allow_pickle=True)
        logger.info(f"[bench] {name:24s} mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50_95']:.3f} "
                    f"countMAE={results[name]['count_MAE']:.2f} lat={lat.mean():.0f}ms")
        del det
    (out_dir / "detection_benchmark.json").write_text(json.dumps(results, indent=2))
    _plots(results, out_dir)
    return results


def add_crowd_free_counts(dataset_dir: str | Path, out_dir: str | Path) -> dict:
    """COCO annotates at most ~13 people individually and marks the rest of a dense group as one
    'crowd' region, so on those images the GT count is incomplete and over-counting is not an error.
    Recompute count errors on images WITHOUT crowd regions (from cached predictions)."""
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    test = load_split(dataset_dir, "test")
    keep = [i for i, r in enumerate(test) if not r["has_crowd_region"]]
    path = out_dir / "detection_benchmark.json"
    results = json.loads(path.read_text())
    for name, r in results.items():
        preds = np.load(out_dir / "predictions" / f"{name}.npy", allow_pickle=True)
        thr = r["count_threshold(val)"]
        counts = [int((preds[i][:, 4] >= thr).sum()) for i in keep]
        ce = count_errors(counts, [len(test[i]["persons"]) for i in keep])
        r.update({f"count_{k}_no_crowd": v for k, v in ce.items()})
        r["images_no_crowd"] = len(keep)
    path.write_text(json.dumps(results, indent=2))
    return results


def failure_gallery(dataset_dir: Path, out_dir: Path, name: str, thr: float, k: int = 6) -> Path:
    """Images with the largest count errors: green = ground truth, red = predictions above threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = load_split(dataset_dir, "test")
    preds = np.load(out_dir / "predictions" / f"{name}.npy", allow_pickle=True)
    errs = [abs(int((p[:, 4] >= thr).sum()) - len(r["persons"])) for p, r in zip(preds, test)]
    worst = np.argsort(errs)[::-1][:k]
    fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(18, 8))
    for ax, i in zip(axes.ravel(), worst):
        img = cv2.imread(str(dataset_dir / "images" / "test" / test[i]["file_name"]))
        for b in gt_boxes_xyxy(test[i]).astype(int):
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 200, 0), 2)
        for p in preds[i]:
            if p[4] >= thr:
                b = p[:4].astype(int)
                cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 1)
        ax.imshow(img[:, :, ::-1])
        ax.set_title(f"GT={len(test[i]['persons'])} pred={int((preds[i][:, 4] >= thr).sum())}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{name}: largest counting errors (green = ground truth, red = predicted)")
    path = out_dir / f"failures_{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _plots(results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for i, n in enumerate(names):
        r = results[n]
        axes[0].scatter(r["latency_ms_mean"], r["mAP50"], s=80, label=n, color=f"C{i}")
        axes[0].annotate(n, (r["latency_ms_mean"], r["mAP50"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set_xscale("log"), axes[0].set_xlabel("CPU latency per image (ms, log)"), axes[0].set_ylabel("mAP@0.5")
    axes[0].set_title("accuracy vs speed"), axes[0].grid(alpha=0.3)
    x = np.arange(len(SIZE_BUCKETS))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        axes[1].bar(x + i * w, list(results[n]["recall_by_size"].values()), w, label=n)
    axes[1].set_xticks(x + 0.4 - w / 2, list(SIZE_BUCKETS)), axes[1].set_ylabel("recall"), axes[1].set_ylim(0, 1)
    axes[1].set_title("recall by person size (small ~ back rows)"), axes[1].legend(fontsize=7)
    kinds = ["clean", *CORRUPTIONS]
    x = np.arange(len(kinds))
    for i, n in enumerate(names):
        axes[2].bar(x + i * w, [results[n]["robustness_mAP50"][k] for k in kinds], w, label=n)
    axes[2].set_xticks(x + 0.4 - w / 2, kinds), axes[2].set_ylabel("mAP@0.5 (40-image subset)")
    axes[2].set_title("robustness"), axes[2].set_ylim(0, 1)
    fig.savefig(out_dir / "detection_benchmark.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
