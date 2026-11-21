import numpy as np
import pytest

from smartattend.eval import metrics as M


def test_box_iou_basic():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(M.box_iou(a, b)[0], [1.0, 50 / 150, 0.0])


def test_perfect_detections_give_ap_one():
    gts = [np.array([[0, 0, 10, 10], [20, 20, 40, 40]])]
    preds = [(gts[0].astype(float), np.array([0.9, 0.8]))]
    assert M.detection_map(preds, gts)[0.5] == pytest.approx(1.0)


def test_false_positive_ranked_first_lowers_ap():
    gts = [np.array([[0, 0, 10, 10]])]
    preds = [(np.array([[50, 50, 60, 60], [0, 0, 10, 10]], float), np.array([0.9, 0.5]))]
    assert M.detection_map(preds, gts)[0.5] == pytest.approx(0.5, abs=0.01)


def test_map_matches_pycocotools():
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(0)
    images, anns, dets, preds, gts = [], [], [], [], []
    aid = 1
    for img_id in range(1, 21):
        n = rng.integers(1, 6)
        xy = rng.uniform(0, 400, size=(n, 2))
        wh = rng.uniform(30, 120, size=(n, 2))
        g = np.hstack([xy, xy + wh])
        gts.append(g)
        images.append({"id": img_id, "width": 640, "height": 640})
        for b in g:
            anns.append({"id": aid, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": float((b[2] - b[0]) * (b[3] - b[1]))})
            aid += 1
        # predictions: jittered GT, some misses, some false positives
        keep = rng.random(n) > 0.2
        p = g[keep] + rng.normal(scale=8, size=(keep.sum(), 4))
        fp = np.hstack([rng.uniform(0, 400, (2, 2)), rng.uniform(0, 400, (2, 2)) + 60])
        fp[:, 2:] = np.maximum(fp[:, 2:], fp[:, :2] + 20)
        p = np.vstack([p, fp])
        s = rng.random(len(p))
        preds.append((p, s))
        for b, sc in zip(p, s):
            dets.append({"image_id": img_id, "category_id": 1, "score": float(sc),
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    coco.createIndex()
    ev = COCOeval(coco, coco.loadRes(dets), "bbox")
    ev.params.areaRng = [[0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    ev.params.maxDets = [100]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(), ev.accumulate()
    ref50 = ev.eval["precision"][0, :, 0, 0, 0].mean()
    ref = ev.eval["precision"][:, :, 0, 0, 0].mean()
    ours = M.coco_map(preds, gts)
    assert ours["mAP50"] == pytest.approx(ref50, abs=0.01)
    assert ours["mAP50_95"] == pytest.approx(ref, abs=0.01)


def test_count_errors():
    e = M.count_errors([10, 12, 9], [10, 10, 10])
    assert e["MAE"] == pytest.approx(1.0) and e["bias"] == pytest.approx(1 / 3)


def test_macro_f1_and_confusion():
    y = [0, 0, 1, 1, 2, 2]
    p = [0, 0, 1, 2, 2, 2]
    r = M.classification_report(y, p, 3)
    assert r["confusion"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    assert r["macro_f1"] == pytest.approx(np.mean([1.0, 2 / 3, 0.8]))


def test_fairness_gap_and_mask_iou():
    g = M.fairness_gap({"a": 0.9, "b": 0.8, "c": float("nan")})
    assert g["gap"] == pytest.approx(0.1) and g["worst"] == "b"
    m1, m2 = np.zeros((4, 4)), np.zeros((4, 4))
    m1[:2], m2[:, :2] = 1, 1
    assert M.mask_iou(m1, m2) == pytest.approx(4 / 12)
