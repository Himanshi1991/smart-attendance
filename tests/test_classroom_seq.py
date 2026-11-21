"""Requirement 9: windowing, set padding, temporal Transformer attention, gradient flow, scoring (synthetic, fast)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smartattend.sequence import classroom_seq as cs  # noqa: E402


def _frames(video, n, share=0.5, students=3):
    out = []
    for t in range(n):
        S = [[0.7, 0.2, 0.1, 0.1 * k, 0.5, 0.1, 0.2] for k in range(students)]
        out.append({"video": video, "frame": t, "path": "", "boxes": [[0, 0, 10, 10]] * students, "students": S,
                    "true_share": share, "true_count": students})
    return out


def test_frame_vector_values():
    f = {"students": [[0.8, 0.1, 0.1, 0, 0, 0, 0], [0.2, 0.7, 0.1, 0, 0, 0, 0]]}
    v = cs.frame_vector(f)
    np.testing.assert_allclose(v[:6], [0.5, 0.4, 0.1, 0.5, 0.5, 2 / 20], atol=1e-6)
    assert cs.frame_vector({"students": []}).tolist() == [0.0] * 7      # a frame with no detections


def test_windows_stay_inside_one_video_and_pad_the_start():
    frames = _frames("a", 10, 0.2) + _frames("b", 5, 0.9)
    frames[3]["students"] = [[0.0, 1.0, 0.0, 0, 0, 0, 0]]               # make frame a/3 recognisable
    X, y, meta = cs.windows(frames)
    assert X.shape == (15, cs.L, 7) and len(meta) == 15
    np.testing.assert_allclose(X[0], np.repeat(X[0][-1:], cs.L, 0))      # first window = first frame repeated
    b0 = [m["video"] for m in meta].index("b")
    np.testing.assert_allclose(X[b0], np.repeat(X[b0][-1:], cs.L, 0))    # video b never sees video a's frames
    assert y[b0] == pytest.approx(0.9)
    Xa, _, _ = cs.windows(frames, ["a"])
    assert len(Xa) == 10


def test_set_arrays_padding_mask():
    frames = _frames("a", 2, students=3) + [{"video": "a", "frame": 2, "path": "", "boxes": [], "students": [],
                                             "true_share": 0.0, "true_count": 0}]
    X, pad, y, fs = cs.set_arrays(frames)
    assert X.shape == (3, cs.MAX_STUDENTS, 7)
    assert (~pad[0]).sum() == 3 and (~pad[2]).sum() == 1                # empty frame keeps one zero "student"


def test_temporal_transformer_attention_is_a_distribution():
    torch.manual_seed(0)
    m = cs.TemporalTransformer().eval()
    y, w = m(torch.randn(4, cs.L, 7), return_attn=True)
    assert y.shape == (4, 1) and w.shape == (4, cs.L)
    np.testing.assert_allclose(w.sum(1).detach().numpy(), 1.0, atol=1e-5)


def test_gradient_decays_backwards_in_time_and_sigmoid_decays_fastest():
    g = {a: cs.gradient_through_time(a, T=25, trials=8) for a in ["tanh", "sigmoid", "step"]}
    for a in g:
        assert g[a][0] < g[a][-1]                                        # the oldest step gets the smallest gradient
    assert g["sigmoid"][0] / g["sigmoid"][-1] < g["tanh"][0] / g["tanh"][-1]


def test_scores_clip_and_handle_constant_predictions():
    y = np.array([0.1, 0.5, 0.9])
    assert cs.scores(y, y)["mae"] == 0 and cs.scores(y, y)["corr"] == pytest.approx(1.0)
    assert cs.scores(y, np.array([2.0, 2.0, 2.0]))["corr"] == 0.0        # constant prediction: no correlation, no NaN
    assert cs.scores(y, np.array([2.0, 2.0, 2.0]))["mae"] == pytest.approx(0.5)


def test_lstm_can_learn_a_simple_temporal_rule():
    """Target = mean of feature 3 over the window: needs memory of earlier frames."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(300, cs.L, 7)).astype(np.float32)
    y = X[:, :, 3].mean(1)
    m, _ = cs.fit_regressor(cs.TrendLSTM(7, 16, "tanh"), X[:200], y[:200], X[200:250], y[200:250], epochs=60, seed=0)
    mae = np.mean(np.abs(cs.predict(m, X[250:]) - y[250:]))
    assert mae < 0.6 * np.mean(np.abs(y[250:] - y[:200].mean()))         # clearly better than predicting the average
