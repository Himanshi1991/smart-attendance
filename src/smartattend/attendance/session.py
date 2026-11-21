"""Aggregate per-frame recognitions into a session attendance register."""
from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from smartattend.attendance.roster import IMAGE_EXTS, Roster, imread_unicode

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_frames(source: str | Path, every_n: int = 10) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR image) from an image, a folder of images, a video file or a webcam index."""
    src = str(source)
    if src.isdigit() or Path(src).suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % every_n == 0:
                    yield idx, frame
                idx += 1
        finally:
            cap.release()
        return
    path = Path(src)
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS) if path.is_dir() else [path]
    for i, f in enumerate(files):
        img = imread_unicode(f)
        if img is not None:
            yield i, img


class AttendanceSession:
    """present : recognised in >= min_frames AND >= min_fraction of processed frames
    review  : recognised in >= min_frames but below min_fraction (late arrival, or an
              occasional false match, e.g. a look-alike stranger) -> teacher decides
    absent  : otherwise"""

    def __init__(self, roster: Roster, min_frames: int = 2, min_fraction: float = 0.5):
        self.roster = roster
        self.min_frames = min_frames
        self.min_fraction = min_fraction
        self.frames_seen = {sid: 0 for sid in roster.student_ids}
        self.first_seen: dict[str, int] = {}
        self.best_score = {sid: 0.0 for sid in roster.student_ids}
        self.n_frames = 0
        self.unknown_per_frame: list[int] = []
        self.faces_per_frame: list[int] = []

    def update(self, frame_idx: int, labels: list[str | None], scores: list[float]) -> None:
        self.n_frames += 1
        self.faces_per_frame.append(len(labels))
        self.unknown_per_frame.append(sum(lbl is None for lbl in labels))
        for lbl, sc in zip(labels, scores):
            if lbl is None:
                continue
            self.frames_seen[lbl] += 1
            self.first_seen.setdefault(lbl, frame_idx)
            self.best_score[lbl] = max(self.best_score[lbl], sc)

    def status(self, sid: str) -> str:
        seen = self.frames_seen[sid]
        if seen < self.min_frames:
            return "absent"
        return "present" if seen >= self.min_fraction * self.n_frames else "review"

    def is_present(self, sid: str) -> bool:
        return self.status(sid) == "present"

    def report(self) -> list[dict]:
        return [{
            "student_id": sid,
            "status": self.status(sid),
            "frames_seen": self.frames_seen[sid],
            "frames_processed": self.n_frames,
            "first_seen_frame": self.first_seen.get(sid),      # None (not "") when never seen: a column mixing numbers and text cannot be shown as a table
            "best_similarity": round(self.best_score[sid], 4),
        } for sid in self.roster.student_ids]

    def summary(self) -> dict:
        st = [self.status(s) for s in self.roster.student_ids]
        return {
            "enrolled": len(self.roster),
            "present": st.count("present"),
            "review": st.count("review"),
            "absent": st.count("absent"),
            "frames_processed": self.n_frames,
            "max_faces_in_frame": max(self.faces_per_frame, default=0),
            "mean_unknown_faces_per_frame": float(np.mean(self.unknown_per_frame)) if self.unknown_per_frame else 0.0,
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        generated = datetime.now().isoformat(timespec="seconds")
        rows = [{**r, "generated_at": generated} for r in self.report()]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["student_id"])
            w.writeheader()
            w.writerows(rows)
        return path
