"""Integrated product: one call per frame does detection, face recognition, engagement, seating and
class-level aggregation; `process_source` runs a whole session and writes the reports."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from smartattend.data.drowsiness import ENGAGEMENT_CLASSES


def head_turn(face_row: np.ndarray) -> float:
    """Yaw proxy from YuNet landmarks: |nose_x - eye_midpoint_x| / inter-ocular distance (0 = frontal)."""
    re, le, nose = face_row[4:6], face_row[6:8], face_row[8:10]
    iod = np.linalg.norm(le - re) + 1e-6
    return float(abs(nose[0] - (re[0] + le[0]) / 2) / iod)


STATES = ["attentive", "distracted", "inactive"]
STATE_COLORS = {"attentive": (0, 200, 0), "distracted": (0, 165, 255), "inactive": (0, 0, 230),
                "unclear": (170, 170, 170)}


def engagement_state(probs: np.ndarray, turn: float, turn_threshold: float = 0.30,
                     face_px: int | None = None, min_face_px: int = 30) -> str:
    """unclear    : face narrower than min_face_px - too few pixels to judge (on the classroom test set,
                 faces < 30 px were called inactive 41 % of the time vs 0-10 % for larger faces)
    inactive   : the alertness classifier says 'disengaged' (eyes closing, drowsy, yawning)
    distracted : alert, but the head is turned away (|nose offset| >= ~0.3 x eye distance, roughly 30 deg)
    attentive  : otherwise
    Looking down is deliberately NOT counted as distracted: in a classroom it is often note-taking."""
    if face_px is not None and face_px < min_face_px:
        return "unclear"
    if int(np.argmax(probs)) == ENGAGEMENT_CLASSES.index("disengaged"):
        return "inactive"
    return "distracted" if turn >= turn_threshold else "attentive"


def face_crop(img: np.ndarray, box, margin: float = 0.35) -> np.ndarray:
    x, y, w, h = box
    m = int(margin * max(w, h))
    x0, y0 = max(0, x - m), max(0, y - m)
    crop = img[y0:min(img.shape[0], y + h + m), x0:min(img.shape[1], x + w + m)]
    return crop if crop.size and min(crop.shape[:2]) >= 4 else np.zeros((32, 32, 3), np.uint8)


class ClassroomAnalyzer:
    def __init__(self, cfg: dict, roster=None, detector: str | None = None, logger: logging.Logger | None = None):
        import torch

        from smartattend.attendance.face import FaceEngine, ensure_models
        from smartattend.sequence.transformer_agg import TransformerAggregator
        from smartattend.vision.detectors import make_detector
        from smartattend.vision.engagement import EngagementClassifier

        self.cfg, self.roster, self.logger = cfg, roster, logger
        models = Path(cfg["paths"]["models_dir"])
        self.detector = make_detector(detector or cfg["product"]["detector"], models)
        self.det_thr = cfg["product"]["person_threshold"]
        self.faces = FaceEngine(*ensure_models(models), score_threshold=cfg["face"]["detector_score_threshold"])
        self.engagement = EngagementClassifier(models / "engagement_resnet18.pt")
        agg_path = Path(cfg["paths"]["experiments_dir"]) / "sequence" / "transformer_aggregator.pt"
        self.aggregator = None
        if agg_path.exists():
            self.aggregator = TransformerAggregator()
            self.aggregator.load_state_dict(torch.load(agg_path, map_location="cpu"))
            self.aggregator.eval()
        self.torch = torch

    def analyze_frame(self, img: np.ndarray) -> dict:
        from smartattend.vision.seating import build_seat_map

        t = {}
        t0 = time.perf_counter()
        if hasattr(self.detector, "detect_with_seats"):
            boxes, scores, seat_boxes = self.detector.detect_with_seats(img, self.det_thr)
        else:  # two-stage detectors: people only, empty seats cannot be counted
            boxes, scores = self.detector(img, self.det_thr)
            seat_boxes = None
        t["detect"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        faces = self.faces.analyze(img)
        labels, sims = ([None] * len(faces), [0.0] * len(faces))
        if self.roster is not None and faces:
            labels, sims = self.roster.match(np.stack([f.embedding for f in faces]), self.cfg["face"]["match_threshold"])
        t["faces"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        probs = self.engagement.predict_proba([face_crop(img, f.box) for f in faces]) if faces else np.zeros((0, 3))
        t["engagement"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        seat_map = build_seat_map(boxes, seat_boxes)
        # attach each face to the person box that contains its centre (upper part of the body)
        face_person = []
        for f in faces:
            cx, cy = f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2
            inside = [i for i, b in enumerate(boxes) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
            face_person.append(min(inside, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])) if inside else None)
        seat_of = {s.person_index: s.seat_id for s in seat_map.seats if s.occupied}
        t["seating"] = (time.perf_counter() - t0) * 1000

        es = self.cfg.get("engagement_states", {})
        turn_thr, min_px = es.get("distracted_head_turn", 0.30), es.get("min_face_px", 30)
        students = []
        for i, f in enumerate(faces):
            turn = head_turn(f.raw)
            students.append({"student_id": labels[i], "similarity": round(float(sims[i]), 3),
                             "seat": seat_of.get(face_person[i]), "face_box": list(map(int, f.box)),
                             "face_size_px": int(f.box[2]), "head_turn": round(turn, 3),
                             "state": engagement_state(probs[i], turn, turn_thr, int(f.box[2]), min_px),
                             "engagement": ENGAGEMENT_CLASSES[int(probs[i].argmax())],
                             "probs": [round(float(p), 3) for p in probs[i]]})
        class_eng = self._aggregate(students)
        state_counts = {s: sum(st["state"] == s for st in students) for s in [*STATES, "unclear"]}
        t["total"] = sum(t.values())
        return {"persons": int(len(boxes)), "person_boxes": boxes.round(1).tolist(), "person_scores": scores.round(3).tolist(),
                "faces": len(faces), "recognized": sum(l is not None for l in labels), "students": students,
                "rows": seat_map.n_rows, "seats_occupied": seat_map.occupied,
                "seats_empty": seat_map.empty if seat_boxes is not None else None,
                "states": state_counts, "class_engagement": class_eng, "timing_ms": t, "_seat_map": seat_map,
                "_labels": labels}

    def _aggregate(self, students: list[dict]) -> dict:
        if not students:
            return {"engaged_fraction_mean": None, "engaged_fraction_transformer": None}
        P = np.array([s["probs"] for s in students])
        out = {"engaged_fraction_mean": round(float(np.mean(P.argmax(1) == 0)), 3)}
        if self.aggregator is not None:
            size = np.clip(np.array([s["face_size_px"] for s in students]) / 120, 0, 1)
            tok = np.column_stack([P, size, np.ones(len(students)) * 0.9,
                                   [s["head_turn"] for s in students]]).astype(np.float32)[:30]
            with self.torch.no_grad():
                v = self.aggregator(self.torch.tensor(tok)[None], self.torch.zeros(1, len(tok), dtype=self.torch.bool))
            out["engaged_fraction_transformer"] = round(float(v), 3)
        return out

    def draw(self, img: np.ndarray, res: dict) -> np.ndarray:
        from smartattend.vision.seating import draw_seat_map

        out = draw_seat_map(img, res["_seat_map"])
        for s in res["students"]:
            x, y, w, h = s["face_box"]
            c = STATE_COLORS[s["state"]]
            cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
            name = s["student_id"] or "unknown"
            cv2.putText(out, f"{name} | {s['state']}", (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        st = res["states"]
        txt = (f"people {res['persons']}  seats {res['seats_occupied']} occupied"
               + (f" / {res['seats_empty']} empty" if res["seats_empty"] is not None else "")
               + f"  | attentive {st['attentive']}  distracted {st['distracted']}  inactive {st['inactive']}"
               + (f"  unclear {st['unclear']}" if st["unclear"] else ""))
        cv2.rectangle(out, (0, 0), (out.shape[1], 24), (30, 30, 30), -1)
        cv2.putText(out, txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def process_source(cfg: dict, source: str, out_dir: str | Path, roster=None, every_n: int = 10,
                   save_video: bool = True, logger: logging.Logger | None = None) -> dict:
    """Run a whole session (image / folder / video / webcam) and write attendance + engagement reports."""
    from smartattend.attendance.session import AttendanceSession, iter_frames

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    an = ClassroomAnalyzer(cfg, roster, logger=logger)
    session = AttendanceSession(roster, cfg["attendance"]["min_frames_present"],
                                cfg["attendance"].get("min_fraction_present", 0.5)) if roster is not None else None
    timeline, writer, frames = [], None, 0
    for idx, frame in iter_frames(source, every_n):
        res = an.analyze_frame(frame)
        frames += 1
        if session is not None:
            session.update(idx, res["_labels"], [s["similarity"] for s in res["students"]])
        judged = sum(res["states"][s] for s in STATES)
        timeline.append({"frame": idx, "persons": res["persons"], "faces": res["faces"], "recognized": res["recognized"],
                         "seats_occupied": res["seats_occupied"], "seats_empty": res["seats_empty"],
                         "faces_unclear": res["states"]["unclear"],
                         **{f"{s}_fraction": round(res["states"][s] / judged, 3) if judged else None for s in STATES},
                         **res["class_engagement"], "latency_ms": round(res["timing_ms"]["total"], 1)})
        if save_video:
            vis = an.draw(frame, res)
            if writer is None:
                size = (vis.shape[1] // 2 * 2, vis.shape[0] // 2 * 2)
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, size)
            writer.write(cv2.resize(vis, size))  # folders may mix image sizes; video needs one size
        if logger:
            logger.info(f"frame {idx}: persons={res['persons']} faces={res['faces']} recognised={res['recognized']} "
                        f"class_eng={res['class_engagement']} {res['timing_ms']['total']:.0f}ms")
    if writer is not None:
        writer.release()
    with open(out_dir / "engagement_timeline.csv", "w", newline="") as f:
        if timeline:
            w = csv.DictWriter(f, fieldnames=list(timeline[0]))
            w.writeheader()
            w.writerows(timeline)
    summary = {"frames_processed": frames,
               "mean_persons": float(np.mean([t["persons"] for t in timeline])) if timeline else 0,
               "mean_latency_ms": float(np.mean([t["latency_ms"] for t in timeline])) if timeline else 0}
    eng = [t["engaged_fraction_transformer"] for t in timeline if t.get("engaged_fraction_transformer") is not None]
    summary["mean_class_engagement"] = float(np.mean(eng)) if eng else None
    for s in STATES:
        v = [t[f"{s}_fraction"] for t in timeline if t[f"{s}_fraction"] is not None]
        summary[f"mean_{s}_fraction"] = float(np.mean(v)) if v else None
    empty = [t["seats_empty"] for t in timeline if t["seats_empty"] is not None]
    summary["mean_seats_empty"] = float(np.mean(empty)) if empty else None
    if session is not None:
        session.to_csv(out_dir / "attendance.csv")
        summary["attendance"] = session.summary()
    (out_dir / "session_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
