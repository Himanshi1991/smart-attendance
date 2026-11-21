"""Requirement 10: corruption functions, crop extraction, box matching, size groups, and the classroom acceptance rows."""
import json

import numpy as np
import pytest

from smartattend.eval import final_eval as fe
from smartattend.requirements import REQUIREMENTS, evaluate

CFG = {"product": {"detector": "yolov8s"}}


@pytest.mark.parametrize("name", list(fe.CORRUPTIONS))
def test_corruptions_keep_shape_and_type(name):
    img = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    out = fe.CORRUPTIONS[name](img)
    assert out.shape == img.shape and out.dtype == np.uint8
    if name == "clean":
        assert out is img
    if name.startswith("low light"):
        assert out.mean() < 0.4 * img.mean()


def test_gt_crops_skip_tiny_boxes_and_report_indices():
    img = np.zeros((200, 300, 3), np.uint8)
    boxes = np.array([[10, 10, 110, 150], [50, 50, 58, 120], [-20, 100, 60, 190]], float)   # 2nd is only 8 px wide
    crops, keep = fe.gt_crops(img, boxes)
    assert keep == [0, 2] and all(c.shape == (160, 160, 3) for c in crops)


def test_match_is_one_to_one_and_reports_best_iou():
    gt = np.array([[0, 0, 100, 100], [200, 0, 300, 100]], float)
    pred = np.array([[2, 2, 98, 98],          # correct
                     [5, 5, 105, 105],        # duplicate of the same student
                     [230, 0, 330, 100],      # on student 2 but IoU 0.54 -> correct
                     [260, 60, 360, 160],     # loosely on student 2 (IoU ~0.09)
                     [600, 600, 700, 700]], float)   # background
    m, best = fe.match(pred, gt)
    assert m.tolist() == [0, -1, 1, -1, -1]
    assert best[1] > 0.5 and 0 < best[3] < 0.5 and best[4] == 0


def test_match_handles_empty_inputs():
    m, best = fe.match(np.zeros((0, 4)), np.array([[0, 0, 10, 10]], float))
    assert len(m) == 0
    m, best = fe.match(np.array([[0, 0, 10, 10]], float), np.zeros((0, 4)))
    assert m.tolist() == [-1]


def test_size_groups_split_into_thirds():
    areas = np.arange(1, 91, dtype=float)
    edges = np.quantile(areas, [1 / 3, 2 / 3])
    g = fe.size_groups(areas, edges)
    assert sorted(np.bincount(g).tolist()) == [30, 30, 30] and g[0] == 0 and g[-1] == 2


def test_classroom_requirements_pass_and_fail_by_direction(tmp_path):
    ids = [r.rid for r in REQUIREMENTS]
    assert {"R10", "R11", "R12", "R13"} <= set(ids) and len(ids) == len(set(ids))
    result = {"robustness": {"clean": {"count_mae": 1.5, "state_macro_f1": 0.47, "mAP50": 0.61},
                             "low light (30% brightness)": {"mAP50": 0.60}},
              "fairness": {"detection_recall_gap_size": 0.35}}
    p = tmp_path / "reports/eval/final_evaluation.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(result))
    rows = {r["id"]: r for r in evaluate(tmp_path, CFG)}
    assert rows["R10"]["status"] == "PASS"                      # 1.5 students <= 2
    assert rows["R11"]["status"] == "FAIL"                      # 0.47 < 0.60
    assert rows["R12"]["status"] == "FAIL"                      # gap 0.35 > 0.15
    assert rows["R13"]["status"] == "PASS" and rows["R13"]["measured"] == pytest.approx(0.01)
