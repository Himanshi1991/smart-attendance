"""Face detection (YuNet) and face embedding (SFace) using OpenCV's ONNX models - CPU friendly."""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
DETECTOR_FILE, RECOGNIZER_FILE = list(MODEL_URLS)


def ensure_models(model_dir: str | Path, logger: logging.Logger | None = None) -> tuple[Path, Path]:
    """Download the two ONNX models from the OpenCV model zoo if they are not present."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = model_dir / name
        if not path.exists() or path.stat().st_size < 10_000:
            if logger:
                logger.info(f"downloading {name}")
            urllib.request.urlretrieve(url, path)
    return model_dir / DETECTOR_FILE, model_dir / RECOGNIZER_FILE


@dataclass
class Face:
    box: tuple[int, int, int, int]      # x, y, w, h
    score: float
    embedding: np.ndarray               # L2-normalised 128-d vector
    raw: np.ndarray                     # YuNet row (box + 5 landmarks + score)


class FaceEngine:
    def __init__(self, detector_path: str | Path, recognizer_path: str | Path,
                 score_threshold: float = 0.8, nms_threshold: float = 0.3):
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320),
                                                  score_threshold, nms_threshold, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return np.empty((0, 15), dtype=np.float32) if faces is None else faces

    def embed(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(image, face_row)
        feat = self.recognizer.feature(aligned).astype(np.float64).ravel()
        return feat / (np.linalg.norm(feat) + 1e-12)

    def analyze(self, image: np.ndarray) -> list[Face]:
        faces = []
        for row in self.detect(image):
            x, y, w, h = row[:4].astype(int)
            faces.append(Face((int(x), int(y), int(w), int(h)), float(row[14]), self.embed(image, row), row))
        return faces


def draw_faces(image: np.ndarray, faces: list[Face], labels: list[str | None],
               scores: list[float]) -> np.ndarray:
    out = image.copy()
    for face, label, score in zip(faces, labels, scores):
        x, y, w, h = face.box
        color = (0, 200, 0) if label else (0, 0, 230)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {score:.2f}" if label else f"unknown {score:.2f}"
        cv2.putText(out, text, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
