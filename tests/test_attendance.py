"""Roster matching and session aggregation, tested with synthetic embeddings (no model download)."""
import numpy as np

from smartattend.attendance.roster import Roster
from smartattend.attendance.session import AttendanceSession


def _unit(v):
    return v / np.linalg.norm(v)


def _roster(n=4, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    return Roster([f"S{i:02d}" for i in range(n)], np.array([_unit(rng.normal(size=dim)) for _ in range(n)]))


def _noisy(c, scale, rng):
    return _unit(c + scale * rng.normal(size=c.shape))


def test_match_identifies_enrolled_students():
    r, rng = _roster(), np.random.default_rng(1)
    faces = np.array([_noisy(r.centroids[2], 0.05, rng), _noisy(r.centroids[0], 0.05, rng)])
    labels, scores = r.match(faces, threshold=0.363)
    assert labels == ["S02", "S00"]
    assert min(scores) > 0.8


def test_stranger_is_unknown():
    r = _roster()
    stranger = _unit(np.random.default_rng(99).normal(size=128))
    labels, _ = r.match(stranger[None], threshold=0.363)
    assert labels == [None]


def test_one_student_cannot_match_two_faces():
    r, rng = _roster(), np.random.default_rng(2)
    faces = np.array([_noisy(r.centroids[1], 0.05, rng), _noisy(r.centroids[1], 0.05, rng)])
    labels, _ = r.match(faces, threshold=0.3)
    assert labels.count("S01") == 1 and labels.count(None) == 1


def test_save_load_roundtrip(tmp_path):
    r = _roster()
    r.save(tmp_path / "roster.npz")
    r2 = Roster.load(tmp_path / "roster.npz")
    assert r2.student_ids == r.student_ids
    np.testing.assert_allclose(r2.centroids, r.centroids)


def test_occasional_match_goes_to_review_not_present():
    # a look-alike stranger matched to S01 in 3 of 10 frames must not mark S01 present
    s = AttendanceSession(_roster(), min_frames=2, min_fraction=0.5)
    for f in range(10):
        s.update(f, ["S00"] + (["S01"] if f < 3 else []), [0.9] + ([0.45] if f < 3 else []))
    assert s.status("S00") == "present"
    assert s.status("S01") == "review"
    assert s.status("S02") == "absent"
    assert s.summary()["review"] == 1


def test_session_requires_min_frames(tmp_path):
    s = AttendanceSession(_roster(), min_frames=2)
    s.update(0, ["S00", "S01", None], [0.8, 0.7, 0.1])
    s.update(10, ["S00"], [0.9])
    rows = {r["student_id"]: r for r in s.report()}
    assert rows["S00"]["status"] == "present" and rows["S00"]["first_seen_frame"] == 0
    assert rows["S01"]["status"] == "absent"  # seen once only
    assert s.summary()["present"] == 1 and s.summary()["max_faces_in_frame"] == 3
    assert s.to_csv(tmp_path / "a.csv").exists()


def test_register_table_can_be_shown_when_some_students_were_never_seen():
    """Regression: 'first_seen_frame' mixed ints and '' -> the Streamlit table crashed (pyarrow ArrowTypeError)."""
    import pandas as pd
    import pyarrow as pa

    roster = Roster(["seen", "never"], np.eye(2))
    s = AttendanceSession(roster, min_frames=1)
    s.update(0, ["seen"], [0.9])
    rows = {r["student_id"]: r for r in s.report()}
    assert rows["seen"]["first_seen_frame"] == 0 and rows["never"]["first_seen_frame"] is None
    pa.Table.from_pandas(pd.DataFrame(s.report()))          # must not raise
