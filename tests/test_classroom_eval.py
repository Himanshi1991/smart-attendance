"""Requirement 7: record loading, prediction caching, augmentation, bootstrap CI, and the transfer-learning nets."""
import json

import numpy as np
import pytest

from smartattend.vision import classroom_eval as ce


def test_load_records_orders_frames_and_reads_boxes(tmp_path):
    recs = [{"file": "dataset/images/v-1_mp4-0010_jpg.rf.a.jpg", "video": "v-1", "boxes": [
                {"xyxy": [0, 0, 10, 10], "state": "attentive"}, {"xyxy": [5, 5, 20, 20], "state": "inactive"}]},
            {"file": "dataset/images/v-1_mp4-0002_jpg.rf.b.jpg", "video": "v-1", "boxes": []}]
    (tmp_path / "test.json").write_text(json.dumps(recs))
    out = ce.load_records(tmp_path, "test")
    assert [r["frame"] for r in out] == [2, 10]
    assert out[0]["boxes"].shape == (0, 4) and out[1]["boxes"].shape == (2, 4)
    assert out[1]["states"] == ["attentive", "inactive"]
    assert out[1]["path"].parent == tmp_path / "yolo_person" / "images" / "test"


def test_predictions_are_cached_and_reused(tmp_path):
    import cv2

    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((20, 20, 3), np.uint8))

    class Fake:
        calls = 0

        def timed(self, image, thr):
            Fake.calls += 1
            return (np.array([[1, 2, 3, 4]], float), np.array([0.9])), 5.0

    recs = [{"path": img}]
    cache = tmp_path / "c.json"
    p1, ms = ce.predictions(Fake(), recs, cache)
    p2, _ = ce.predictions(None, recs, cache)            # second call must not need a detector
    assert Fake.calls == 2 and ms == [5.0]              # 1 warm-up + 1 timed image; warm-up is not recorded
    np.testing.assert_allclose(p1[0][0], p2[0][0])


def test_augment_keeps_dtype_and_shape():
    X = np.random.default_rng(0).integers(0, 256, size=(6, 16, 16, 3), dtype=np.uint8)
    A = ce._augment(X, np.random.default_rng(1))
    assert A.shape == X.shape and A.dtype == np.uint8


def test_bootstrap_ci_brackets_the_score():
    from smartattend.eval.metrics import macro_f1

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, 3, 300))
    lo, hi = ce.bootstrap_ci(y, p, n=300)
    assert lo < macro_f1(y, p, 3) < hi


@pytest.mark.parametrize("arch", ["resnet18", "googlenet"])
def test_transfer_nets_freeze_early_layers_and_output_three_states(arch):
    torch = pytest.importorskip("torch")
    net = ce._net(arch)
    trainable = [n for n, p in net.named_parameters() if p.requires_grad]
    assert trainable and all(n.split(".")[0] in {"layer4", "fc", "inception5a", "inception5b"} for n in trainable)
    out = ce.predict_cnn(net, np.zeros((2, 64, 64, 3), np.uint8))
    assert out.shape == (2, 3) and np.allclose(out.sum(1), 1, atol=1e-5)
