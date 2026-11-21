"""Person detectors behind one interface: YOLO (Ultralytics) and torchvision two-stage / one-stage models.

    det = make_detector("fasterrcnn_resnet50_fpn")
    boxes_xyxy, scores = det(image_bgr)          # persons only
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

COCO_PERSON_TV = 1      # torchvision COCO label id for person
DETECTORS = {
    "yolov8n": "YOLOv8-n (one-stage, CSP backbone)",
    "yolov8s": "YOLOv8-s (one-stage, CSP backbone)",
    "yolov8n-classroom": "YOLOv8-n fine-tuned on classroom split",
    "fasterrcnn_resnet50_fpn": "Faster R-CNN, ResNet-50 FPN (two-stage)",
    "retinanet_resnet50_fpn": "RetinaNet, ResNet-50 FPN (one-stage, focal loss)",
}


class Detector:
    name: str

    def __call__(self, image_bgr: np.ndarray, score_thr: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def timed(self, image_bgr, score_thr=0.05):
        t0 = time.perf_counter()
        out = self(image_bgr, score_thr)
        return out, (time.perf_counter() - t0) * 1000


class YOLODetector(Detector):
    def __init__(self, weights: str | Path, name: str, imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.name, self.imgsz = name, imgsz
        names = self.model.names
        self.person_id = next((k for k, v in names.items() if v == "person"), 0)

    def __call__(self, image_bgr, score_thr=0.05):
        r = self.model.predict(image_bgr, imgsz=self.imgsz, conf=score_thr, classes=[self.person_id],
                               verbose=False, device="cpu")[0]
        return r.boxes.xyxy.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy().astype(np.float64)

    SEAT_NAMES = ("chair", "couch", "bench")

    def detect_with_seats(self, image_bgr, person_thr: float, seat_thr: float = 0.25):
        """People and seats (chair / couch / bench) from ONE forward pass."""
        seat_ids = [k for k, v in self.model.names.items() if v in self.SEAT_NAMES]
        r = self.model.predict(image_bgr, imgsz=self.imgsz, conf=min(person_thr, seat_thr),
                               classes=[self.person_id, *seat_ids], verbose=False, device="cpu")[0]
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = r.boxes.conf.cpu().numpy().astype(np.float64)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        person = (cls == self.person_id) & (conf >= person_thr)
        seat = np.isin(cls, seat_ids) & (conf >= seat_thr)
        return xyxy[person], conf[person], xyxy[seat]


class TorchvisionDetector(Detector):
    def __init__(self, name: str, max_size: int = 800):
        import torch
        import torchvision.models.detection as D

        self.torch = torch
        builders = {
            "fasterrcnn_resnet50_fpn": (D.fasterrcnn_resnet50_fpn, D.FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
            "retinanet_resnet50_fpn": (D.retinanet_resnet50_fpn, D.RetinaNet_ResNet50_FPN_Weights.DEFAULT),
        }
        fn, weights = builders[name]
        # low internal score threshold so the full precision-recall curve is available for mAP
        kwargs = {"box_score_thresh": 0.01} if name.startswith("faster") else {"score_thresh": 0.01}
        self.model = fn(weights=weights, min_size=min(800, max_size), max_size=max_size, **kwargs).eval()
        self.name = name

    def __call__(self, image_bgr, score_thr=0.05):
        torch = self.torch
        x = torch.from_numpy(image_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            out = self.model([x])[0]
        keep = (out["labels"] == COCO_PERSON_TV) & (out["scores"] >= score_thr)
        return out["boxes"][keep].numpy().astype(np.float64), out["scores"][keep].numpy().astype(np.float64)


def make_detector(name: str, models_dir: str | Path = "models") -> Detector:
    models_dir = Path(models_dir)
    if name.startswith("yolov8"):
        if name == "yolov8n-classroom":
            return YOLODetector(models_dir / "yolo" / "yolov8n-classroom.pt", name)
        return YOLODetector(models_dir / "yolo" / f"{name}.pt", name)
    return TorchvisionDetector(name)
