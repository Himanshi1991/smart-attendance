"""Kaggle 'Classroom Student Engagement Dataset' (mustafaxgm, CC BY-NC 4.0).

481 frames from 11 real classroom videos; every student has a box with one of 8 action labels.
We map actions to the three engagement states the product reports:

    look_forward, read, write, handrise, stand -> attentive
    turn_head, using_device                   -> distracted
    sleep                                     -> inactive

Leakage-safe split: frames are grouped by RECORDING GROUP (videos of the same students in the same room),
not by frame or even by video. A random frame split would put near-identical consecutive frames, and the
same students, in train and test.

    python -m smartattend stage kaggle_data
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

KAGGLE_REF = "mustafaxgm/classroom-student-engagement-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_REF}"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REF}"
ACTIONS = ["handrise", "look_forward", "read", "sleep", "stand", "turn_head", "using_device", "write"]
STATES = ["attentive", "distracted", "inactive"]
ACTION_TO_STATE = {"look_forward": "attentive", "read": "attentive", "write": "attentive", "handrise": "attentive",
                   "stand": "attentive", "turn_head": "distracted", "using_device": "distracted", "sleep": "inactive"}

# videos showing the same students in the same room (checked visually) -> one group
VIDEO_GROUP = {
    "owais-class-1": "owais", "owais-class-2": "owais", "owais-class-3": "owais",
    "taimoor-class-2": "taimoor-class", "taimoor-class-3": "taimoor-class", "taimoor-class-4": "taimoor-class",
    "taimoor-class-5": "taimoor-class",
    "taimoor-zaeem-1": "taimoor-zaeem",
    "mustafa-seminar-stand": "seminar", "ammad-seminar-stand": "seminar",
    "1-second-scene": "misc",
}
GROUP_SPLIT = {"owais": "train", "seminar": "train", "misc": "train", "taimoor-class": "val", "taimoor-zaeem": "test"}


def video_of(filename: str) -> str:
    return re.split(r"_mp4-", Path(filename).name)[0]


def download(raw_dir: str | Path, logger=None) -> Path:
    from smartattend.data.download import _fetch

    zpath = Path(raw_dir) / "kaggle_classroom" / "archive.zip"
    if logger:
        logger.info(f"downloading {KAGGLE_URL}")
    return _fetch(DOWNLOAD_URL, zpath, 100_000_000)


class _Source:
    """Read the dataset from the Kaggle zip OR from an extracted folder (e.g. kagglehub's download in Colab).
    Names are always zip-style: 'dataset/images/....jpg'."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.is_dir():
            root = self.path if (self.path / "dataset").is_dir() else self.path.parent
            self.root = root
            self.names = [f.relative_to(root).as_posix() for f in (root / "dataset").rglob("*") if f.is_file()]
            self.zip = None
        else:
            self.zip = zipfile.ZipFile(self.path)
            self.names = self.zip.namelist()

    def read(self, name: str) -> bytes:
        return self.zip.read(name) if self.zip else (self.root / name).read_bytes()


def load_records(source: str | Path) -> list[dict]:
    """One record per image: video, group, size, and boxes (pixel xyxy) with action and state."""
    z = _Source(source)
    names = set(z.names)
    records = []
    for n in sorted(names):
        if not (n.startswith("dataset/images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))):
            continue
        lab = "dataset/labels/" + Path(n).stem + ".txt"
        img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        boxes = []
        if lab in names:
            for line in z.read(lab).decode().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
                boxes.append({"action": ACTIONS[c], "state": ACTION_TO_STATE[ACTIONS[c]],
                              "xyxy": [round(x0, 1), round(y0, 1), round(x0 + w * W, 1), round(y0 + h * H, 1)]})
        v = video_of(n)
        records.append({"file": n, "video": v, "group": VIDEO_GROUP.get(v, v), "width": W, "height": H, "boxes": boxes})
    return records


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).ravel()), 2)


def find_duplicates(source: str | Path, records: list[dict], max_hamming: int = 3) -> dict:
    """Exact and near-duplicate images. Within a group they are harmless (same split); ACROSS groups they
    would leak, so those are removed."""
    z = _Source(source)
    hashes = [dhash(cv2.imdecode(np.frombuffer(z.read(r["file"]), np.uint8), cv2.IMREAD_COLOR)) for r in records]
    within, across = 0, []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= max_hamming:
                if records[i]["group"] == records[j]["group"]:
                    within += 1
                else:
                    across.append((records[i]["file"], records[j]["file"]))
    return {"near_duplicate_pairs_within_group": within, "near_duplicate_pairs_across_groups": across}


def prepare(raw_dir: str | Path, out_dir: str | Path, logger=None, crop_size: int = 160,
            source: str | Path | None = None) -> dict:
    """source: the Kaggle zip or an extracted folder (default: data/raw/kaggle_classroom/archive.zip,
    downloaded if missing)."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    zpath = Path(source) if source else raw_dir / "kaggle_classroom" / "archive.zip"
    if not zpath.exists():
        download(raw_dir, logger)
    records = load_records(zpath)
    dups = find_duplicates(zpath, records)
    drop = {b for _, b in dups["near_duplicate_pairs_across_groups"]}
    records = [r for r in records if r["file"] not in drop]
    for r in records:
        r["split"] = GROUP_SPLIT[r["group"]]

    z = _Source(zpath)
    for split in ["train", "val", "test"]:
        for sub in ["yolo_person/images", "yolo_person/labels", "yolo_states/images", "yolo_states/labels"]:
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)
        for s in STATES:
            (out_dir / "crops" / split / s).mkdir(parents=True, exist_ok=True)

    crop_count = Counter()
    for r in records:
        img = cv2.imdecode(np.frombuffer(z.read(r["file"]), np.uint8), cv2.IMREAD_COLOR)
        stem = Path(r["file"]).stem
        W, H = r["width"], r["height"]
        person_lines, state_lines = [], []
        for k, b in enumerate(r["boxes"]):
            x0, y0, x1, y1 = b["xyxy"]
            yolo = f"{(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
            person_lines.append(f"0 {yolo}")
            state_lines.append(f"{STATES.index(b['state'])} {yolo}")
            crop = img[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))]
            if crop.size and min(crop.shape[:2]) >= 16:
                h, w = crop.shape[:2]
                s = crop_size / max(h, w)
                crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                cv2.imencode(".jpg", crop)[1].tofile(str(out_dir / "crops" / r["split"] / b["state"] / f"{stem}_{k}.jpg"))
                crop_count[(r["split"], b["state"])] += 1
        for sub, lines in [("yolo_person", person_lines), ("yolo_states", state_lines)]:
            cv2.imencode(".jpg", img)[1].tofile(str(out_dir / sub / "images" / r["split"] / f"{stem}.jpg"))
            (out_dir / sub / "labels" / r["split"] / f"{stem}.txt").write_text("\n".join(lines))

    for sub, names in [("yolo_person", ["person"]), ("yolo_states", STATES)]:
        (out_dir / sub / "data.yaml").write_text(
            f"path: {(out_dir / sub).resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    for split in ["train", "val", "test"]:
        (out_dir / f"{split}.json").write_text(json.dumps([r for r in records if r["split"] == split]))

    stats = {}
    for split in ["train", "val", "test"]:
        rs = [r for r in records if r["split"] == split]
        stats[split] = {"images": len(rs), "videos": sorted({r["video"] for r in rs}),
                        "groups": sorted({r["group"] for r in rs}),
                        "people": sum(len(r["boxes"]) for r in rs),
                        "states": dict(Counter(b["state"] for r in rs for b in r["boxes"])),
                        "actions": dict(Counter(b["action"] for r in rs for b in r["boxes"])),
                        "crops": {s: crop_count[(split, s)] for s in STATES}}
    report = {"source": KAGGLE_URL, "licence": "CC BY-NC 4.0 (Kaggle release; the bundled Roboflow readme says "
              "'Private', the Kaggle licence governs the published copy)", "action_to_state": ACTION_TO_STATE,
              "video_groups": VIDEO_GROUP, "group_split": GROUP_SPLIT,
              "duplicates": {"near_duplicate_pairs_within_group": dups["near_duplicate_pairs_within_group"],
                             "removed_across_groups": len(drop)},
              "splits": stats}
    (out_dir / "provenance.json").write_text(json.dumps(report, indent=2))
    if logger:
        logger.info(f"[kaggle] {json.dumps({k: {kk: v[kk] for kk in ('images', 'people', 'states')} for k, v in stats.items()})}")
    return report


def make_figures(processed_dir: str | Path, out_dir: str | Path, per_state: int = 8) -> list[Path]:
    """(1) label distribution per split, (2) random example crops per state - a visual label check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = json.loads((processed_dir / "provenance.json").read_text())
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    splits = ["train", "val", "test"]
    colors = {"attentive": "#2e9e44", "distracted": "#f59e0b", "inactive": "#d43c3c"}
    bottom = np.zeros(3)
    for s in STATES:
        v = np.array([rep["splits"][sp]["states"].get(s, 0) for sp in splits], float)
        frac = v / np.array([rep["splits"][sp]["people"] for sp in splits])
        axes[0].bar(splits, frac, bottom=bottom, color=colors[s], label=s)
        for i, (f, b) in enumerate(zip(frac, bottom)):
            if f > 0.04:
                axes[0].text(i, b + f / 2, f"{f:.0%}", ha="center", va="center", color="white", fontsize=9)
        bottom += frac
    axes[0].set_title("state mix per split (test = unseen classroom)"), axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    acts = Counter()
    for sp in splits:
        acts.update(rep["splits"][sp]["actions"])
    names = [a for a, _ in acts.most_common()]
    axes[1].barh(names, [acts[a] for a in names], color=[colors[ACTION_TO_STATE[a]] for a in names])
    axes[1].invert_yaxis(), axes[1].set_title("original action labels (colour = mapped state)")
    for i, a in enumerate(names):
        axes[1].text(acts[a] + 20, i, str(acts[a]), va="center", fontsize=9)
    paths.append(out_dir / "kaggle_label_distribution.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, per_state, figsize=(per_state * 1.6, 6))
    for r, s in enumerate(STATES):
        files = sorted((processed_dir / "crops" / "train" / s).glob("*.jpg"))
        for c, f in enumerate(rng.choice(files, min(per_state, len(files)), replace=False)):
            axes[r, c].imshow(cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1])
        for c in range(per_state):
            axes[r, c].axis("off")
        axes[r, 0].set_title(s, loc="left", fontsize=11, color=colors[s])
    fig.suptitle("random training crops per state (label sanity check)")
    paths.append(out_dir / "kaggle_example_crops.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)
    return paths


def load_split(processed_dir: str | Path, split: str) -> list[dict]:
    return json.loads((Path(processed_dir) / f"{split}.json").read_text())
