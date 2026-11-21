"""Seating analysis: semantic + instance segmentation of people and chairs, and a row/seat map.

* Semantic segmentation : LR-ASPP MobileNetV3 (torchvision, VOC classes) -> person / chair pixel masks
* Instance segmentation : YOLOv8-n-seg -> one mask per person / chair
* Seat map              : people are grouped into rows by the vertical position of their box bottoms
                          (1-D gap clustering scaled by box height), then ordered left->right; chairs that
                          no person overlaps are reported as empty seats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VOC_PERSON, VOC_CHAIR = 15, 9
COCO_PERSON, COCO_CHAIR, COCO_COUCH, COCO_BENCH = 0, 56, 57, 13


class SemanticSegmenter:
    def __init__(self):
        import torch
        import torchvision.models.segmentation as S

        self.torch = torch
        self.model = S.lraspp_mobilenet_v3_large(weights=S.LRASPP_MobileNet_V3_Large_Weights.DEFAULT).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, img_bgr: np.ndarray, max_side: int = 520) -> np.ndarray:
        """Return per-pixel VOC class ids at the input resolution."""
        h, w = img_bgr.shape[:2]
        s = max_side / max(h, w)
        small = cv2.resize(img_bgr, (int(w * s), int(h * s))) if s < 1 else img_bgr
        x = (small[:, :, ::-1].astype(np.float32) / 255 - self.mean) / self.std
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(x.transpose(2, 0, 1).copy())[None])["out"][0]
        lab = out.argmax(0).numpy().astype(np.uint8)
        return cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)


@dataclass
class Instance:
    cls: int
    box: np.ndarray
    score: float
    mask: np.ndarray | None = None


class InstanceSegmenter:
    def __init__(self, weights: str | Path):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def __call__(self, img_bgr: np.ndarray, conf: float = 0.25) -> list[Instance]:
        r = self.model.predict(img_bgr, conf=conf, classes=[COCO_PERSON, COCO_CHAIR, COCO_COUCH, COCO_BENCH],
                               verbose=False, device="cpu", retina_masks=True)[0]
        out = []
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None
        for i, (b, c, s) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())):
            out.append(Instance(int(c), b, float(s), masks[i].astype(bool) if masks is not None else None))
        return out


@dataclass
class Seat:
    seat_id: str
    row: int
    col: int
    box: np.ndarray
    occupied: bool
    person_index: int | None = None


@dataclass
class SeatMap:
    seats: list[Seat] = field(default_factory=list)
    n_rows: int = 0

    @property
    def occupied(self) -> int:
        return sum(s.occupied for s in self.seats)

    @property
    def empty(self) -> int:
        return sum(not s.occupied for s in self.seats)


def cluster_rows(boxes: np.ndarray, gap_factor: float = 0.45) -> np.ndarray:
    """Assign each box a row index (0 = front = lowest in the image) by splitting the sorted box-bottom
    y-coordinates wherever the gap exceeds gap_factor x the local median box height."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    bottoms = boxes[:, 3]
    heights = boxes[:, 3] - boxes[:, 1]
    order = np.argsort(-bottoms)  # front (bottom of image) first
    rows = np.zeros(len(boxes), int)
    r = 0
    for prev, cur in zip(order[:-1], order[1:]):
        if bottoms[prev] - bottoms[cur] > gap_factor * np.median(heights[[prev, cur]]):
            r += 1
        rows[cur] = r
    return rows


def build_seat_map(person_boxes: np.ndarray, seat_boxes: np.ndarray | None = None, overlap: float = 0.15) -> SeatMap:
    """Occupied seats = people (row/col from clustering); empty seats = chairs not overlapped by any person."""
    person_boxes = np.asarray(person_boxes, float).reshape(-1, 4)
    seat_boxes = np.zeros((0, 4)) if seat_boxes is None else np.asarray(seat_boxes, float).reshape(-1, 4)
    empty = []
    for cb in seat_boxes:
        ix = np.clip(np.minimum(cb[2], person_boxes[:, 2]) - np.maximum(cb[0], person_boxes[:, 0]), 0, None)
        iy = np.clip(np.minimum(cb[3], person_boxes[:, 3]) - np.maximum(cb[1], person_boxes[:, 1]), 0, None)
        frac = ix * iy / max((cb[2] - cb[0]) * (cb[3] - cb[1]), 1e-6)
        if not (frac > overlap).any():
            empty.append(cb)
    all_boxes = np.vstack([person_boxes, np.array(empty).reshape(-1, 4)])
    rows = cluster_rows(all_boxes)
    sm = SeatMap(n_rows=int(rows.max() + 1) if len(rows) else 0)
    for r in range(sm.n_rows):
        idx = np.where(rows == r)[0]
        idx = idx[np.argsort(all_boxes[idx, 0])]
        for c, i in enumerate(idx):
            occ = i < len(person_boxes)
            sm.seats.append(Seat(f"R{r + 1}C{c + 1}", r, c, all_boxes[i], occ, int(i) if occ else None))
    return sm


def draw_seat_map(img: np.ndarray, sm: SeatMap, labels: dict[int, str] | None = None) -> np.ndarray:
    out = img.copy()
    palette = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (171, 71, 188), (0, 172, 193)]
    for s in sm.seats:
        b = s.box.astype(int)
        color = palette[s.row % len(palette)] if s.occupied else (160, 160, 160)
        cv2.rectangle(out, tuple(b[:2]), tuple(b[2:]), color, 2 if s.occupied else 1)
        text = s.seat_id + ("" if s.occupied else " empty")
        if labels and s.person_index in labels:
            text += f" {labels[s.person_index]}"
        cv2.putText(out, text, (b[0], max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def overlay_mask(img: np.ndarray, mask: np.ndarray, color=(0, 140, 255), alpha=0.45) -> np.ndarray:
    out = img.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out
