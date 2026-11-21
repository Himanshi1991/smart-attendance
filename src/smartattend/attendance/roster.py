"""Class roster: one mean face embedding per enrolled student, and one-to-one face matching."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """cv2.imread fails on non-ASCII Windows paths; decode from bytes instead."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


class Roster:
    def __init__(self, student_ids: list[str], centroids: np.ndarray, n_images: list[int] | None = None):
        self.student_ids = list(student_ids)
        self.centroids = np.asarray(centroids, dtype=np.float64).reshape(len(self.student_ids), -1)
        self.n_images = n_images or [0] * len(self.student_ids)

    def __len__(self) -> int:
        return len(self.student_ids)

    @classmethod
    def enroll_from_dir(cls, root: str | Path, engine, logger: logging.Logger | None = None) -> "Roster":
        """Expect root/<student_id>/*.jpg, one clearly visible face per photo (3-5 photos recommended)."""
        ids, centroids, counts = [], [], []
        for student_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            embs = []
            for img_path in sorted(student_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = imread_unicode(img_path)
                faces = engine.analyze(img) if img is not None else []
                if not faces:
                    if logger:
                        logger.warning(f"no face found in {img_path}, skipped")
                    continue
                # use the largest face in an enrolment photo
                embs.append(max(faces, key=lambda f: f.box[2] * f.box[3]).embedding)
            if not embs:
                if logger:
                    logger.warning(f"student {student_dir.name}: no usable photos, NOT enrolled")
                continue
            c = np.mean(embs, axis=0)
            ids.append(student_dir.name)
            centroids.append(c / np.linalg.norm(c))
            counts.append(len(embs))
            if logger:
                logger.info(f"enrolled {student_dir.name} from {len(embs)} photo(s)")
        return cls(ids, np.array(centroids), counts)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, student_ids=np.array(self.student_ids), centroids=self.centroids,
                 n_images=np.array(self.n_images))

    @classmethod
    def load(cls, path: str | Path) -> "Roster":
        d = np.load(path, allow_pickle=False)
        return cls(d["student_ids"].tolist(), d["centroids"], d["n_images"].tolist())

    def match(self, embeddings: np.ndarray, threshold: float) -> tuple[list[str | None], list[float]]:
        """Assign each face to at most one student and each student to at most one face.

        Greedy on descending cosine similarity: a student cannot be 'present twice' in a frame,
        and a face below ``threshold`` stays unknown.
        """
        n_faces = len(embeddings)
        labels: list[str | None] = [None] * n_faces
        if n_faces == 0 or len(self) == 0:
            return labels, [0.0] * n_faces
        sims = np.asarray(embeddings) @ self.centroids.T          # cosine (both L2-normalised)
        scores = sims.max(axis=1).tolist()
        used_faces, used_students = set(), set()
        for flat in np.argsort(-sims, axis=None):
            f, s = np.unravel_index(flat, sims.shape)
            if sims[f, s] < threshold:
                break
            if f in used_faces or s in used_students:
                continue
            labels[f] = self.student_ids[s]
            scores[f] = float(sims[f, s])
            used_faces.add(f)
            used_students.add(s)
        return labels, scores
