"""Build a reproducible demo: a 12-student roster from LFW, a synthetic 'classroom' video made of
DIFFERENT photos of those people (2 absent, 1 non-enrolled stranger), then run the full product.

    python scripts/make_demo.py
Outputs: data/demo/roster/, data/demo/class_demo.mp4, reports/demo/{annotated.mp4, attendance.csv, ...}
LFW contains news photos of public figures (research use); the demo is for demonstration only.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from smartattend.utils import get_logger, load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = load_config()
    rng = set_seed(cfg["seed"])
    log = get_logger("demo", cfg["paths"]["logs_dir"])
    demo = ROOT / "data" / "demo"
    shutil.rmtree(demo, ignore_errors=True)
    (demo / "roster").mkdir(parents=True)

    df = pd.read_parquet(ROOT / "data/raw/lfw/lfw.parquet", columns=["identity", "image_num", "image"])
    counts = df.groupby("identity").size()
    ids = sorted(counts[(counts >= 6) & (counts <= 40)].index)
    chosen = list(rng.choice(ids, 13, replace=False))
    students, stranger = chosen[:12], chosen[12]
    absent = set(students[-2:])
    dec = lambda b: cv2.imdecode(np.frombuffer(b["bytes"], np.uint8), cv2.IMREAD_COLOR)

    probes = {}
    for sid in chosen:
        rows = df[df["identity"] == sid].sort_values("image_num")
        imgs = [dec(b) for b in rows["image"]]
        if sid in students:
            (demo / "roster" / sid).mkdir()
            for i, im in enumerate(imgs[:3]):
                cv2.imwrite(str(demo / "roster" / sid / f"{i + 1}.jpg"), im)
        probes[sid] = imgs[3:] if sid in students else imgs

    # classroom canvas: 3 rows x 4 desks; back rows smaller (perspective)
    W, H = 1280, 720
    seats = [(r, c) for r in range(3) for c in range(4)]
    present = [s for s in students if s not in absent] + [stranger]
    writer = cv2.VideoWriter(str(demo / "class_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (W, H))
    for f in range(20):
        canvas = np.full((H, W, 3), (205, 214, 222), np.uint8)
        cv2.rectangle(canvas, (0, 0), (W, 70), (60, 80, 60), -1)  # blackboard
        cv2.putText(canvas, "DEMO CLASSROOM (synthetic composite of LFW photos)", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        light = 0.85 + 0.15 * np.sin(f / 3)
        for (r, c), sid in zip(seats, present):
            size = [140, 175, 215][r]                           # back row (r=0) smaller (perspective)
            img = probes[sid][(f // 4) % len(probes[sid])]
            img = cv2.convertScaleAbs(cv2.resize(img, (size, size)), alpha=light)
            x = int(90 + c * 290 + (3 - r) * 12 + rng.integers(-3, 4))
            y = [85, 255, 465][r]
            canvas[y:y + size, x:x + size] = img
            cv2.rectangle(canvas, (x - 10, y + size), (x + size + 10, y + size + 18), (90, 60, 40), -1)  # desk
        writer.write(canvas)
    writer.release()
    truth = {"students": students, "absent": sorted(absent), "stranger": stranger}
    (demo / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    log.info(f"demo built: {truth}")

    from smartattend.attendance.face import FaceEngine, ensure_models
    from smartattend.attendance.roster import Roster
    from smartattend.pipeline import process_source

    eng = FaceEngine(*ensure_models(cfg["paths"]["models_dir"]))
    roster = Roster.enroll_from_dir(demo / "roster", eng, log)
    roster.save(demo / "roster.npz")
    out = ROOT / "reports" / "demo"
    summary = process_source(cfg, str(demo / "class_demo.mp4"), out, roster, every_n=1, logger=log)
    reg = pd.read_csv(out / "attendance.csv")
    marked = set(reg[reg["status"] == "present"]["student_id"])
    correct = sum((s in marked) == (s not in absent) for s in students)
    summary["demo_check"] = {"correct_register_entries": correct, "of": len(students),
                             "flagged_for_review": sorted(reg[reg["status"] == "review"]["student_id"]),
                             "stranger_marked_as_someone": bool(len(marked - set(students)))}
    (out / "session_summary.json").write_text(json.dumps(summary, indent=2))
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ok, still = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out / "demo_frame.png"), still)
    log.info(f"demo summary: {summary}")


if __name__ == "__main__":
    main()
