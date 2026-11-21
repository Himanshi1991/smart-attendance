import io
import zipfile

import cv2
import numpy as np

from smartattend.data import kaggle_classroom as K


def test_every_action_maps_to_a_state():
    assert set(K.ACTION_TO_STATE) == set(K.ACTIONS)
    assert set(K.ACTION_TO_STATE.values()) == set(K.STATES)


def test_video_name_parsing():
    assert K.video_of("dataset/images/owais-class-1_mp4-0005_jpg.rf.abc.jpg") == "owais-class-1"
    assert K.video_of("1-second-scene_mp4-0000_jpg.rf.X.jpg") == "1-second-scene"


def test_split_is_by_group_and_every_split_exists():
    assert set(K.GROUP_SPLIT.values()) == {"train", "val", "test"}
    assert set(K.VIDEO_GROUP.values()) == set(K.GROUP_SPLIT)       # every group has a split
    # all videos of one group land in one split (no student appears in two splits)
    for g in set(K.VIDEO_GROUP.values()):
        assert len({K.GROUP_SPLIT[K.VIDEO_GROUP[v]] for v in K.VIDEO_GROUP if K.VIDEO_GROUP[v] == g}) == 1


def _fake_zip(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, cls in [("owais-class-1_mp4-0000_jpg.rf.a", 5), ("taimoor-zaeem-1_mp4-0001_jpg.rf.b", 3)]:
            img = np.full((100, 200, 3), 128, np.uint8)
            z.writestr(f"dataset/images/{name}.jpg", cv2.imencode(".jpg", img)[1].tobytes())
            z.writestr(f"dataset/labels/{name}.txt", f"{cls} 0.5 0.5 0.2 0.4\n")
    p = tmp_path / "archive.zip"
    p.write_bytes(buf.getvalue())
    return p


def test_load_records_converts_yolo_to_pixels(tmp_path):
    recs = {r["video"]: r for r in K.load_records(_fake_zip(tmp_path))}
    b = recs["owais-class-1"]["boxes"][0]
    assert b["action"] == "turn_head" and b["state"] == "distracted"
    assert b["xyxy"] == [80.0, 30.0, 120.0, 70.0]                  # cx .5*200, w .2*200, cy .5*100, h .4*100
    assert recs["taimoor-zaeem-1"]["boxes"][0]["state"] == "inactive"
    assert recs["taimoor-zaeem-1"]["group"] == "taimoor-zaeem"


def test_folder_and_zip_give_same_records(tmp_path):
    zp = _fake_zip(tmp_path)
    folder = tmp_path / "extracted"
    zipfile.ZipFile(zp).extractall(folder)
    assert K.load_records(zp) == K.load_records(folder)
