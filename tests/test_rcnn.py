"""Requirement 8: building blocks of R-CNN / Fast R-CNN."""
import numpy as np
import pytest

from smartattend.vision import rcnn


def test_box_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(rcnn.box_iou(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-9)


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(0)
    p = np.c_[rng.uniform(0, 100, (20, 2)), rng.uniform(150, 300, (20, 2))]
    g = p + rng.normal(scale=10, size=p.shape)
    np.testing.assert_allclose(rcnn.decode(p, rcnn.encode(p, g)), g, atol=1e-6)


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = rcnn.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.3)
    assert sorted(keep.tolist()) == [0, 2]


def test_sample_regions_labels_and_targets():
    gt = np.array([[100, 100, 200, 220]], float)
    props = np.array([[102, 98, 205, 215], [300, 300, 400, 400], [0, 0, 50, 50], [110, 120, 190, 210]], float)
    regions, labels, targets = rcnn.sample_regions(props, gt, np.random.default_rng(0), n=8, pos_frac=0.5,
                                                   pos_iou=0.6, neg_hi=0.3)
    assert labels[:labels.sum()].all() and not labels[labels.sum():].any()          # positives first
    pos_iou = rcnn.box_iou(regions[labels == 1], gt).max(1)
    assert (pos_iou > 0.5).all()
    neg_iou = rcnn.box_iou(regions[labels == 0], gt).max(1)
    assert (neg_iou < 0.3).all()
    np.testing.assert_allclose(rcnn.decode(regions[labels == 1], targets[labels == 1]),
                               np.repeat(gt, (labels == 1).sum(), 0), atol=1e-6)


def test_warp_crops_shape_and_normalisation():
    img = np.full((100, 200, 3), 128, np.uint8)
    out = rcnn.warp_crops(img, np.array([[10, 10, 60, 90], [0, 0, 200, 100]], float), size=32)
    assert out.shape == (2, 32, 32, 3)
    np.testing.assert_allclose(out[0, 0, 0], (128 / 255 - rcnn.MEAN) / rcnn.STD, atol=1e-5)


def test_roi_align_reads_the_right_place():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    fmap = torch.zeros(1, 1, 10, 10)
    fmap[0, 0, 2:4, 5:7] = 1.0                       # a bright 2x2 block at feature cells (y 2-3, x 5-6)
    # image-space box over that block at stride 16: x 80-112, y 32-64
    roi = torch.tensor([[0, 80.0, 32.0, 112.0, 64.0]])
    inside = torchvision.ops.roi_align(fmap, roi, (2, 2), spatial_scale=1 / 16, sampling_ratio=2, aligned=True)
    roi_far = torch.tensor([[0, 0.0, 120.0, 32.0, 152.0]])
    outside = torchvision.ops.roi_align(fmap, roi_far, (2, 2), spatial_scale=1 / 16, sampling_ratio=2, aligned=True)
    assert inside.mean() > 0.7 and outside.abs().max() == 0       # bilinear sampling blends the block edges (0.77)


def test_frame_without_students_still_gives_background_regions():
    """Regression: with neg_lo=0.1 a frame with no students produced an EMPTY batch -> NaN loss -> ruined weights."""
    props = np.array([[0, 0, 50, 50], [10, 10, 80, 90], [100, 100, 200, 200]], float)
    regions, labels, targets = rcnn.sample_regions(props, np.zeros((0, 4)), np.random.default_rng(0), n=8, pos_frac=0.25,
                                                   pos_iou=0.5, neg_hi=0.5, neg_lo=0.1)
    assert len(regions) == 3 and not labels.any() and np.isfinite(targets).all()


def test_zero_size_label_is_ignored():
    gt = np.array([[10, 10, 10, 40], [100, 100, 200, 220]], float)          # first box has zero width
    props = np.array([[102, 98, 205, 215], [300, 300, 400, 400]], float)
    _, labels, targets = rcnn.sample_regions(props, gt, np.random.default_rng(0), n=8, pos_frac=0.5, pos_iou=0.6, neg_hi=0.3)
    assert labels.sum() >= 1 and np.isfinite(targets).all()


def test_fast_rcnn_trains_without_nan_on_frame_with_no_students(tmp_path):
    import cv2

    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    recs, props = [], {}
    for i, boxes in enumerate([np.array([[40, 40, 120, 140]], float), np.zeros((0, 4))]):
        path = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(path), rng.integers(0, 255, (200, 260, 3), dtype=np.uint8))
        recs.append({"path": path, "boxes": boxes})
        props[path.name] = [[35, 45, 125, 135], [150, 20, 250, 120], [0, 100, 90, 199], [60, 60, 130, 150]]
    m = rcnn.FastRCNN(short_side=128)
    hist = m.fit(recs, props, epochs=2, per_image=8, log=lambda s: None)
    assert np.isfinite(hist["loss"]).all()
    b, s = m.detect(cv2.imread(str(recs[0]["path"])), props[recs[0]["path"].name])
    assert np.isfinite(b).all() and np.isfinite(s).all()
