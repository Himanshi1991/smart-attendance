from datetime import datetime

import numpy as np

from smartattend.attendance.live import LiveAttendance, safe_student_id
from smartattend.attendance.roster import Roster


def _roster():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(3, 128))
    return Roster(["alice", "bob", "carol"], c / np.linalg.norm(c, axis=1, keepdims=True))


def test_marked_only_after_confirmation():
    live = LiveAttendance(_roster(), confirm_hits=3, window=5)
    t = datetime(2026, 10, 31, 9, 0, 0)
    assert live.update(["alice"], [0.8], t) == []
    assert live.update(["alice"], [0.8], t) == []
    assert live.update(["alice"], [0.9], t) == ["alice"]     # third hit -> present
    assert live.update(["alice"], [0.9], t) == []            # only reported once
    reg = {r["student"]: r for r in live.register()}
    assert reg["alice"]["status"] == "present" and reg["alice"]["time_marked"] == "09:00:00"
    assert reg["alice"]["similarity"] == 0.9
    assert reg["bob"]["status"] == "absent"


def test_sporadic_lookalike_never_confirmed():
    live = LiveAttendance(_roster(), confirm_hits=3, window=5)
    pattern = ["bob", None, None, None, "bob", None, None, None, "bob", None]  # 1 hit per 4 frames
    for p in pattern:
        live.update([p] if p else [], [0.45] if p else [])
    assert "bob" not in live.marked


def test_save_and_safe_id(tmp_path):
    live = LiveAttendance(_roster())
    for _ in range(3):
        live.update(["carol"], [0.7])
    path = live.save(tmp_path, datetime(2026, 10, 31))
    text = path.read_text(encoding="utf-8")
    assert path.name == "attendance_2026-10-31.csv" and "carol,present" in text
    assert safe_student_id(" 22CS001 Anushka Chauhan ") == "22CS001_Anushka_Chauhan"
    assert safe_student_id("a/b\\c:d") == "a_b_c_d"


def test_live_register_table_can_be_shown_with_present_and_absent_students(tmp_path):
    """Regression: 'similarity' mixed floats and '' -> the live register table crashed (pyarrow ArrowTypeError)."""
    import pandas as pd
    import pyarrow as pa

    live = LiveAttendance(Roster(["here", "away"], np.eye(2)), confirm_hits=3, window=5)
    for _ in range(3):
        live.update(["here"], [0.8])
    rows = {r["student"]: r for r in live.register()}
    assert rows["here"]["status"] == "present" and rows["away"]["similarity"] is None
    pa.Table.from_pandas(pd.DataFrame(live.register()))                     # must not raise
    text = live.save(tmp_path).read_text(encoding="utf-8")
    assert "away,absent,," in text                                          # empty cells in the CSV, as before
