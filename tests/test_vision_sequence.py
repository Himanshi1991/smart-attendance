"""Unit tests for convolution maths, seating logic, leakage-safe splits and the sequence models."""
import cv2
import numpy as np
import pandas as pd
import pytest

from smartattend.vision.conv_viz import conv2d_naive, conv3d_naive
from smartattend.vision.seating import build_seat_map, cluster_rows


def test_conv2d_naive_matches_opencv():
    rng = np.random.default_rng(0)
    img, k = rng.random((12, 15)), rng.normal(size=(3, 3))
    ours = conv2d_naive(img, k)
    ref = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_CONSTANT)[1:-1, 1:-1]  # filter2D = correlation
    np.testing.assert_allclose(ours, ref, atol=1e-10)
    assert conv2d_naive(img, k, stride=2, pad=1).shape == (6, 8)


def test_conv3d_temporal_difference_detects_motion_only():
    vol = np.zeros((4, 6, 6))
    vol[:, 1:3, 1:3] = 1.0          # static square
    for t in range(4):
        vol[t, 4, t + 1] = 1.0       # moving pixel
    k = np.zeros((2, 1, 1))
    k[0], k[1] = -1, 1
    out = conv3d_naive(vol, k)
    assert np.all(out[:, 1:3, 1:3] == 0)
    assert np.abs(out[:, 4]).sum() > 0


def test_cluster_rows_and_seat_map():
    # two rows of three people: front row bottom ~400, back row bottom ~250
    front = [[x, 250, x + 60, 400] for x in (0, 100, 200)]
    back = [[x + 20, 150, x + 60, 250] for x in (0, 100, 200)]
    boxes = np.array(front + back, float)
    rows = cluster_rows(boxes)
    assert set(rows[:3]) == {0} and set(rows[3:]) == {1}
    chairs = np.array([[400, 300, 460, 400], [10, 300, 50, 390]], float)  # one empty, one under a person
    sm = build_seat_map(boxes, chairs)
    assert sm.occupied == 6 and sm.empty == 1 and sm.n_rows == 2
    assert [s.seat_id for s in sm.seats if s.row == 0][:3] == ["R1C1", "R1C2", "R1C3"]


def test_time_block_split_has_no_overlap_and_buffers():
    from smartattend.data.drowsiness import time_block_split

    t = np.arange(0, 60, 0.05)
    df = pd.DataFrame({"t": t, "session": 0, "engagement": (t // 7 % 3).astype(int)})
    sp = time_block_split(df, block_s=5, buffer_s=0.5)
    assert set(sp["split"]) <= {"train", "val", "test"}
    assert sp.groupby("block")["split"].nunique().max() == 1      # a block is never split
    # frames of different splits are never closer than the 2 x buffer gap
    s = sp.sort_values("t")
    change = s["split"].ne(s["split"].shift()) & s["split"].shift().notna()
    gaps = s["t"].diff()[change]
    assert (gaps >= 0.95).all()


def test_flex_lstm_shapes_and_step_gradient():
    torch = pytest.importorskip("torch")
    from smartattend.sequence.lstm_trend import TrendLSTM

    x = torch.randn(4, 7, 3)
    for act in ["tanh", "sigmoid", "step"]:
        m = TrendLSTM(3, 8, act)
        y = m(x)
        assert y.shape == (4, 1)
        y.sum().backward()
        assert m.rnn.x2g.weight.grad is not None and torch.isfinite(m.rnn.x2g.weight.grad).all()


def test_transformer_aggregator_is_permutation_invariant():
    torch = pytest.importorskip("torch")
    from smartattend.sequence.transformer_agg import DeepSets, TransformerAggregator

    torch.manual_seed(0)
    x = torch.rand(2, 9, 6)
    pad = torch.zeros(2, 9, dtype=torch.bool)
    pad[1, 6:] = True
    perm = torch.randperm(6)
    for m in [TransformerAggregator().eval(), DeepSets().eval()]:
        a = m(x, pad)
        xp = x.clone()
        xp[1, :6] = x[1, perm]
        b = m(xp, pad)
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        # padded tokens must not influence the output
        xq = x.clone()
        xq[1, 6:] = 99.0
        torch.testing.assert_close(m(xq, pad)[1], a[1], atol=1e-5, rtol=1e-5)


def test_engagement_state_rule():
    from smartattend.pipeline import engagement_state

    alert, drowsy = np.array([0.8, 0.15, 0.05]), np.array([0.1, 0.2, 0.7])
    assert engagement_state(alert, 0.05) == "attentive"
    assert engagement_state(alert, 0.45) == "distracted"
    assert engagement_state(drowsy, 0.05) == "inactive"
    assert engagement_state(drowsy, 0.45) == "inactive"   # drowsiness takes priority over head turn
    assert engagement_state(drowsy, 0.05, face_px=20) == "unclear"   # too small to judge
    assert engagement_state(alert, 0.05, face_px=80) == "attentive"


def test_head_turn_frontal_vs_turned():
    from smartattend.pipeline import head_turn

    frontal = np.zeros(15)
    frontal[4:10] = [40, 50, 60, 50, 50, 60]   # right eye, left eye, nose centred
    turned = frontal.copy()
    turned[8] = 58
    assert head_turn(frontal) < 0.05 < head_turn(turned)
