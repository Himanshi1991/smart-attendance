"""Requirement 1: stakeholder needs -> measurable targets -> automated acceptance tests.

Each requirement names WHO needs it, WHAT is measured, the TARGET, and WHERE the measured value comes
from (a result JSON written by an experiment stage). `python -m smartattend acceptance` checks them all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Requirement:
    rid: str
    stakeholder: str
    need: str
    metric: str
    target: float
    direction: str                       # ">=" (higher is better) or "<=" (lower is better)
    unit: str
    source: str                          # result file, relative to the project root
    extract: Callable[[dict, dict], float]   # (result json, config) -> measured value
    produced_by: str                     # command that creates the result file

    def check(self, value: float) -> bool:
        return value >= self.target if self.direction == ">=" else value <= self.target


def _product(det: dict, cfg: dict) -> dict:
    return det[cfg["product"]["detector"]]


REQUIREMENTS = [
    Requirement("R1", "Teacher", "Correct register without roll-call", "attendance recall", 0.95, ">=", "",
                "reports/eval/face_recognition.json", lambda r, c: r["sessions"]["attendance_recall"],
                "python -m smartattend stage face"),
    Requirement("R2", "Teacher", "Absent students are not marked present", "false-present rate", 0.02, "<=", "",
                "reports/eval/face_recognition.json", lambda r, c: r["sessions"]["false_present_rate"],
                "python -m smartattend stage face"),
    Requirement("R3", "Admin", "No proxy attendance by strangers", "stranger false-accept rate (FAR)", 0.01, "<=", "",
                "reports/eval/face_recognition.json", lambda r, c: r["verification"]["FAR"],
                "python -m smartattend stage face"),
    Requirement("R4", "Students", "Equal treatment across groups", "max attendance-recall gap (gender, ethnicity)",
                0.05, "<=", "", "reports/eval/face_recognition.json",
                lambda r, c: max(r["sessions"]["recall_gap_gender"]["gap"], r["sessions"]["recall_gap_ethnicity"]["gap"]),
                "python -m smartattend stage face"),
    Requirement("R5", "Teacher", "Find every person in the room", "person detection mAP@0.5", 0.80, ">=", "",
                "reports/experiments/detection/detection_benchmark.json", lambda r, c: _product(r, c)["mAP50"],
                "python -m smartattend stage detection"),
    Requirement("R6", "Teacher", "Trustworthy head count", "count error (MAE, fully annotated images)", 1.0, "<=", "people",
                "reports/experiments/detection/detection_benchmark.json", lambda r, c: _product(r, c)["count_MAE_no_crowd"],
                "python -m smartattend stage detection"),
    Requirement("R7", "Teacher", "Know when the class loses focus", "engagement macro-F1 (5-fold CV)", 0.65, ">=", "",
                "reports/experiments/engagement/engagement_cv.json",
                lambda r, c: max(r["resnet18"]["mean"], r["googlenet"]["mean"]),
                "python -m smartattend stage engagement_cv"),
    Requirement("R8", "IT", "Runs on a classroom PC without GPU", "CPU latency per frame, p95", 500.0, "<=", "ms",
                "reports/eval/pipeline_latency.json", lambda r, c: r["total"]["p95"],
                "python -m smartattend stage pipeline_latency"),
    Requirement("R9", "IT", "Fits on an ordinary PC", "peak memory", 2000.0, "<=", "MB",
                "reports/eval/pipeline_latency.json", lambda r, c: r["peak_rss_mb"],
                "python -m smartattend stage pipeline_latency"),
    # R10-R13 were added at integration (Requirement 10), once real classroom video with per-student labels was
    # available (Kaggle test classroom). Their targets were written down before the final evaluation was run.
    Requirement("R10", "Teacher", "Head count in a real class video", "count error, classroom video (MAE)", 2.0, "<=",
                "students", "reports/eval/final_evaluation.json", lambda r, c: r["robustness"]["clean"]["count_mae"],
                "python -m smartattend evaluate"),
    Requirement("R11", "Teacher", "Attentive / distracted / inactive in a NEW classroom",
                "state macro-F1, classroom video", 0.60, ">=", "", "reports/eval/final_evaluation.json",
                lambda r, c: r["robustness"]["clean"]["state_macro_f1"], "python -m smartattend evaluate"),
    Requirement("R12", "Students", "Back-row students are not overlooked", "detection recall gap, small vs large", 0.15,
                "<=", "", "reports/eval/final_evaluation.json", lambda r, c: r["fairness"]["detection_recall_gap_size"],
                "python -m smartattend evaluate"),
    Requirement("R13", "IT", "Keeps working in a dim room", "mAP50 lost in low light", 0.05, "<=", "",
                "reports/eval/final_evaluation.json",
                lambda r, c: r["robustness"]["clean"]["mAP50"] - r["robustness"]["low light (30% brightness)"]["mAP50"],
                "python -m smartattend evaluate"),
]

DEPLOYMENT_CONSTRAINTS = [
    "CPU-only inference on a classroom PC (GPU only for offline training, e.g. Google Colab)",
    "Offline: models are local files, no image leaves the premises",
    "Only consenting, enrolled students are identified; embeddings stored, not photos",
    "A teacher confirms every register; engagement is reported per class, never per student",
]


def evaluate(root: str | Path, cfg: dict) -> list[dict]:
    root = Path(root)
    rows = []
    for req in REQUIREMENTS:
        path = root / req.source
        row = {"id": req.rid, "stakeholder": req.stakeholder, "metric": req.metric,
               "target": f"{req.direction} {req.target:g}{(' ' + req.unit) if req.unit else ''}"}
        if not path.exists():
            rows.append({**row, "measured": None, "status": "NOT RUN", "how": req.produced_by})
            continue
        try:
            value = float(req.extract(json.loads(path.read_text()), cfg))
        except (KeyError, TypeError, ValueError) as e:
            rows.append({**row, "measured": None, "status": "ERROR", "how": f"{req.produced_by} ({e!r})"})
            continue
        rows.append({**row, "measured": value, "status": "PASS" if req.check(value) else "FAIL", "how": req.produced_by})
    return rows


def format_table(rows: list[dict]) -> str:
    def fmt(v):
        if v is None:
            return "-"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.0f}"

    head = f"{'ID':<4}{'Stakeholder':<13}{'Metric':<46}{'Target':<15}{'Measured':<10}Status"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['id']:<4}{r['stakeholder']:<13}{r['metric']:<46}{r['target']:<15}{fmt(r['measured']):<10}{r['status']}")
    n = {s: sum(r["status"] == s for r in rows) for s in ["PASS", "FAIL", "NOT RUN", "ERROR"]}
    lines.append("-" * len(head))
    lines.append(f"{n['PASS']} pass, {n['FAIL']} fail, {n['NOT RUN']} not run, {n['ERROR']} error")
    for r in rows:
        if r["status"] in ("NOT RUN", "ERROR"):
            lines.append(f"  {r['id']}: run `{r['how']}`")
    return "\n".join(lines)
