"""Live (webcam) attendance and camera enrolment.

A student is marked present only after being recognised in `confirm_hits` of the last `window`
processed frames, so a single look-alike false match never marks anyone present.
"""
from __future__ import annotations

import csv
import io
import re
import shutil
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from smartattend.attendance.roster import IMAGE_EXTS, Roster


class LiveAttendance:
    def __init__(self, roster: Roster, confirm_hits: int = 3, window: int = 5):
        self.roster = roster
        self.confirm_hits, self.window = confirm_hits, window
        self.recent = {sid: deque(maxlen=window) for sid in roster.student_ids}
        self.marked: dict[str, dict] = {}   # sid -> {"time", "similarity"}
        self.frames = 0

    def update(self, labels: list[str | None], scores: list[float], when: datetime | None = None) -> list[str]:
        """Feed one frame's recognitions; return the students newly marked present in this frame."""
        when = when or datetime.now()
        self.frames += 1
        seen = {l: s for l, s in zip(labels, scores) if l is not None}
        newly = []
        for sid, hist in self.recent.items():
            hist.append(seen.get(sid))
            hits = [s for s in hist if s is not None]
            if sid not in self.marked and len(hits) >= self.confirm_hits:
                self.marked[sid] = {"time": when.strftime("%H:%M:%S"), "similarity": round(float(max(hits)), 3)}
                newly.append(sid)
        return newly

    def register(self) -> list[dict]:
        return [{"student": sid,
                 "status": "present" if sid in self.marked else "absent",
                 "time_marked": self.marked.get(sid, {}).get("time", ""),
                 "similarity": self.marked.get(sid, {}).get("similarity")}      # None (not "") for absent students: keeps the column numeric
                for sid in self.roster.student_ids]

    def save(self, out_dir: str | Path, date: datetime | None = None) -> Path:
        date = date or datetime.now()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"attendance_{date:%Y-%m-%d}.csv"
        rows = [{**r, "date": f"{date:%Y-%m-%d}"} for r in self.register()]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["student"])
            w.writeheader()
            w.writerows(rows)
        return path


def safe_student_id(text: str) -> str:
    """Folder-safe student id: letters, digits, '-', '_' (spaces become '_')."""
    s = re.sub(r"[^\w\-]+", "_", text.strip(), flags=re.UNICODE).strip("_")
    return s[:60]


def check_enrolment_photo(engine, img: np.ndarray, min_face_px: int = 80) -> tuple[bool, str]:
    faces = engine.analyze(img)
    if not faces:
        return False, "No face found. Face the camera in good light."
    if len(faces) > 1:
        return False, f"{len(faces)} faces found. Only the student should be in the photo."
    w = faces[0].box[2]
    if w < min_face_px:
        return False, f"Face too small ({w}px). Move closer to the camera."
    return True, f"OK, face {w}px, confidence {faces[0].score:.2f}"


def save_student_photos(roster_dir: str | Path, student_id: str, images: list[np.ndarray]) -> Path:
    d = Path(roster_dir) / student_id
    d.mkdir(parents=True, exist_ok=True)
    start = len([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    for i, img in enumerate(images, start=start + 1):
        ok, enc = cv2.imencode(".jpg", img)
        enc.tofile(str(d / f"{i}.jpg"))          # tofile handles non-ASCII Windows paths
    return d


def roster_backup(roster_dir: str | Path) -> bytes:
    """Zip of every enrolled student's photos (<student_id>/<n>.jpg): lets the online app, whose disk is wiped
    on every restart, get its students back with `restore_roster_backup`."""
    buf = io.BytesIO()
    root = Path(roster_dir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if root.exists():
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                for img in sorted(d.iterdir()):
                    if img.suffix.lower() in IMAGE_EXTS:
                        z.writestr(f"{d.name}/{img.name}", img.read_bytes())
    return buf.getvalue()


def restore_roster_backup(roster_dir: str | Path, data: bytes, max_photos: int = 20) -> list[str]:
    """Restore students from a `roster_backup` zip; returns the restored student ids.

    Only `<student_id>/<photo>` entries with a valid id and a decodable image are used (the zip is user input:
    nothing is written from its paths directly). A restored student replaces a student with the same id.
    """
    photos: dict[str, list[np.ndarray]] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if info.is_dir() or len(parts) != 2 or Path(parts[1]).suffix.lower() not in IMAGE_EXTS:
                continue
            sid = parts[0]
            if not sid or safe_student_id(sid) != sid or len(photos.get(sid, [])) >= max_photos:
                continue
            if info.file_size > 20_000_000:
                continue
            img = cv2.imdecode(np.frombuffer(z.read(info), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                photos.setdefault(sid, []).append(img)
    for sid, imgs in photos.items():
        shutil.rmtree(Path(roster_dir) / sid, ignore_errors=True)
        save_student_photos(roster_dir, sid, imgs)
    return sorted(photos)


def rebuild_roster(roster_dir: str | Path, roster_file: str | Path, engine, logger=None) -> Roster:
    roster = Roster.enroll_from_dir(roster_dir, engine, logger)
    roster.save(roster_file)
    return roster


def draw_live(frame: np.ndarray, faces, labels, scores, marked: set[str]) -> np.ndarray:
    out = frame.copy()
    for f, lbl, sc in zip(faces, labels, scores):
        x, y, w, h = f.box
        if lbl is None:
            color, text = (0, 0, 230), f"unknown {sc:.2f}"
        elif lbl in marked:
            color, text = (0, 190, 0), f"{lbl}  PRESENT"
        else:
            color, text = (0, 200, 255), f"{lbl}  confirming..."
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(out, (x, max(0, y - 22)), (x + max(w, 9 * len(text)), y), color, -1)
        cv2.putText(out, text, (x + 3, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
