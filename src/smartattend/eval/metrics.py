"""Evaluation metrics implemented from first principles (and unit-tested).

* detection: IoU, COCO-style AP with 101-point interpolation, mAP@0.5 and mAP@[.5:.95]
* counting : MAE / MAPE of person counts
* classification: confusion matrix, per-class precision/recall/F1, macro-F1
* fairness : max gap of a metric across groups
* segmentation: IoU of binary masks
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- detection
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match_detections(pred_boxes, pred_scores, gt_boxes, iou_thr: float):
    """Greedy matching in descending score order. Returns (tp flags, scores) per prediction."""
    order = np.argsort(-np.asarray(pred_scores))
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)[order]
    scores = np.asarray(pred_scores, float)[order]
    tp = np.zeros(len(scores), bool)
    if len(gt_boxes) and len(pred_boxes):
        ious = box_iou(pred_boxes, gt_boxes)
        used = np.zeros(len(gt_boxes), bool)
        for i in range(len(pred_boxes)):
            cand = np.where(~used & (ious[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                used[j] = True
                tp[i] = True
    return tp, scores


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO-style AP: area under the monotone precision envelope sampled at 101 recall points."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(recall, r, side="left")
        ap += precision[idx] if idx < len(precision) else 0.0
    return ap / 101


def detection_map(preds: list[tuple[np.ndarray, np.ndarray]], gts: list[np.ndarray],
                  iou_thresholds=(0.5,)) -> dict[float, float]:
    """preds[i] = (boxes xyxy, scores) for image i; gts[i] = gt boxes xyxy. Single class."""
    out = {}
    n_gt = int(sum(len(g) for g in gts))
    for thr in iou_thresholds:
        all_tp, all_sc = [], []
        for (pb, ps), gb in zip(preds, gts):
            tp, sc = match_detections(pb, ps, gb, thr)
            all_tp.append(tp)
            all_sc.append(sc)
        out[thr] = average_precision(np.concatenate(all_tp) if all_tp else np.zeros(0, bool),
                                     np.concatenate(all_sc) if all_sc else np.zeros(0), n_gt)
    return out


def coco_map(preds, gts) -> dict[str, float]:
    thr = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
    aps = detection_map(preds, gts, thr)
    return {"mAP50": aps[0.5], "mAP75": aps[0.75], "mAP50_95": float(np.mean(list(aps.values())))}


def recall_at(preds, gts, score_thr: float, iou_thr: float = 0.5, size_filter=None) -> float:
    """Recall of GT boxes (optionally only those passing size_filter(box)) at a score threshold."""
    hit = total = 0
    for (pb, ps), gb in zip(preds, gts):
        keep = np.asarray(ps) >= score_thr
        pb = np.asarray(pb).reshape(-1, 4)[keep]
        gb = np.asarray(gb).reshape(-1, 4)
        sel = np.array([size_filter(b) for b in gb], bool) if size_filter else np.ones(len(gb), bool)
        if not sel.any():
            continue
        total += sel.sum()
        if len(pb):
            hit += int((box_iou(gb[sel], pb).max(axis=1) >= iou_thr).sum())
    return hit / total if total else float("nan")


# --------------------------------------------------------------------------- counting
def count_errors(pred_counts, true_counts) -> dict[str, float]:
    p, t = np.asarray(pred_counts, float), np.asarray(true_counts, float)
    return {"MAE": float(np.mean(np.abs(p - t))),
            "RMSE": float(np.sqrt(np.mean((p - t) ** 2))),
            "MAPE": float(np.mean(np.abs(p - t) / np.maximum(t, 1))),
            "bias": float(np.mean(p - t))}


# --------------------------------------------------------------------------- classification
def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_report(y_true, y_pred, n_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"accuracy": float(tp.sum() / max(cm.sum(), 1)), "macro_f1": float(f1.mean()),
            "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(), "confusion": cm.tolist()}


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    return classification_report(y_true, y_pred, n_classes)["macro_f1"]


# --------------------------------------------------------------------------- fairness
def fairness_gap(values_by_group: dict[str, float]) -> dict:
    vals = {k: v for k, v in values_by_group.items() if v == v}  # drop NaN
    if not vals:
        return {"gap": float("nan"), "best": None, "worst": None, "by_group": values_by_group}
    best, worst = max(vals, key=vals.get), min(vals, key=vals.get)
    return {"gap": float(vals[best] - vals[worst]), "best": best, "worst": worst, "by_group": values_by_group}


# --------------------------------------------------------------------------- segmentation
def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else float("nan")
