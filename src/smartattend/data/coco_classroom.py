"""Curate a classroom-like person-detection dataset from COCO val2017.

* test / val  : classroom-like images: a human-written caption mentions a classroom-like scene
                (students, lecture, audience, ...) with >= 2 people, OR a seated indoor group
                (>= 3 people, chairs and desks/laptops/books/screens).
* train       : other multi-person COCO images (>= 4 people) used to fine-tune YOLO.
Near-duplicate images are removed with a perceptual hash before splitting, every image keeps
its Flickr licence and URL (provenance), and splits are disjoint by image.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PERSON = 1
CLASSROOM_WORDS = r"\b(class ?rooms?|students?|lectures?|lecture hall|teachers?|school|audience|seminar|" \
                  r"conference|auditorium|presentation|pupils?|desks?|meeting|classmates?|lesson)\b"
STUDY_OBJECTS = {"laptop", "book", "dining table", "tv", "keyboard", "mouse"}
SEED = 0


def _load(ann_dir: Path):
    inst = json.loads((ann_dir / "instances_val2017.json").read_text())
    caps = json.loads((ann_dir / "captions_val2017.json").read_text())
    return inst, caps


def select_images(ann_dir: str | Path, n_train: int = 450, val_frac: float = 0.3) -> dict:
    ann_dir = Path(ann_dir)
    inst, caps = _load(ann_dir)
    licenses = {l["id"]: l for l in inst["licenses"]}
    captions = defaultdict(list)
    for c in caps["annotations"]:
        captions[c["image_id"]].append(c["caption"].strip())
    cat_names = {c["id"]: c["name"] for c in inst["categories"]}
    objects = defaultdict(set)
    persons = defaultdict(list)
    crowd = set()
    for a in inst["annotations"]:
        objects[a["image_id"]].add(cat_names[a["category_id"]])
        if a["category_id"] != PERSON:
            continue
        if a["iscrowd"]:
            crowd.add(a["image_id"])
        else:
            persons[a["image_id"]].append({"bbox": a["bbox"], "area": a["area"], "segmentation": a["segmentation"]})

    pattern = re.compile(CLASSROOM_WORDS, re.I)
    classroom, pool = [], []
    for im in inst["images"]:
        n = len(persons.get(im["id"], []))
        caps_ = captions[im["id"]]
        hits = sum(bool(pattern.search(c)) for c in caps_)
        rec = {"id": im["id"], "file_name": im["file_name"], "width": im["width"], "height": im["height"],
               "license": licenses[im["license"]]["name"], "license_url": licenses[im["license"]]["url"],
               "flickr_url": im.get("flickr_url", ""), "captions": caps_, "has_crowd_region": im["id"] in crowd,
               "persons": persons.get(im["id"], [])}
        seated_group = (n >= 3 and "chair" in objects[im["id"]] and objects[im["id"]] & STUDY_OBJECTS)
        if (hits >= 1 and n >= 2) or seated_group:
            rec["selected_by"] = "caption" if hits >= 1 else "seated-indoor-group"
            classroom.append(rec)
        elif n >= 4:
            pool.append(rec)

    rng = np.random.default_rng(SEED)
    rng.shuffle(classroom)
    rng.shuffle(pool)
    n_val = int(len(classroom) * val_frac)
    return {"train": pool[:n_train], "val": classroom[:n_val], "test": classroom[n_val:]}


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def deduplicate(selection: dict, img_dir: Path, max_hamming: int = 4) -> tuple[dict, list]:
    """Drop near-duplicates (dHash Hamming distance <= max_hamming). Test images win over val over train."""
    kept, removed, hashes = {k: [] for k in selection}, [], []
    for split in ["test", "val", "train"]:
        for rec in selection[split]:
            img = cv2.imread(str(img_dir / rec["file_name"]))
            if img is None:
                removed.append((split, rec["file_name"], "unreadable"))
                continue
            h = dhash(img)
            if any(bin(h ^ o).count("1") <= max_hamming for o in hashes):
                removed.append((split, rec["file_name"], "near-duplicate"))
                continue
            hashes.append(h)
            kept[split].append(rec)
    return kept, removed


def export(selection: dict, img_dir: Path, out_dir: Path) -> dict:
    """Write YOLO-format dataset (person = class 0) + a JSON ground truth with masks for evaluation."""
    out_dir = Path(out_dir)
    stats = {}
    for split, recs in selection.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for rec in recs:
            src = img_dir / rec["file_name"]
            dst = out_dir / "images" / split / rec["file_name"]
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            W, H = rec["width"], rec["height"]
            lines = []
            for p in rec["persons"]:
                x, y, w, h = p["bbox"]
                lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            (out_dir / "labels" / split / rec["file_name"].replace(".jpg", ".txt")).write_text("\n".join(lines))
        (out_dir / f"{split}.json").write_text(json.dumps(recs))
        stats[split] = {"images": len(recs), "persons": sum(len(r["persons"]) for r in recs)}
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n")
    return stats


def prepare(raw_dir: str | Path, processed_dir: str | Path, logger) -> dict:
    """select -> download images -> deduplicate -> export. Writes a provenance/statistics JSON."""
    from smartattend.data.download import download_coco_annotations, download_coco_images

    raw_dir = Path(raw_dir)
    ann = download_coco_annotations(raw_dir, logger)
    sel = select_images(ann)
    img_dir = download_coco_images(raw_dir, sorted({r["file_name"] for v in sel.values() for r in v}), logger)
    sel, removed = deduplicate(sel, img_dir)
    stats = export(sel, img_dir, processed_dir)
    report = {"stats": stats, "removed": removed,
              "selected_by": {s: dict(Counter(r.get("selected_by", "multi-person") for r in v)) for s, v in sel.items()},
              "licenses": dict(Counter(r["license"] for v in sel.values() for r in v))}
    (Path(processed_dir) / "provenance.json").write_text(json.dumps(report, indent=2))
    logger.info(f"COCO classroom dataset: {stats}; removed {len(removed)}")
    return report


def load_split(processed_dir: str | Path, split: str) -> list[dict]:
    return json.loads((Path(processed_dir) / f"{split}.json").read_text())


def gt_boxes_xyxy(rec: dict) -> np.ndarray:
    b = np.array([p["bbox"] for p in rec["persons"]], dtype=np.float64).reshape(-1, 4)
    b[:, 2:] += b[:, :2]
    return b


def gt_person_mask(rec: dict) -> np.ndarray:
    """Union of polygon person masks (semantic segmentation ground truth)."""
    mask = np.zeros((rec["height"], rec["width"]), np.uint8)
    for p in rec["persons"]:
        seg = p["segmentation"]
        if isinstance(seg, list):
            for poly in seg:
                pts = np.array(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
    return mask
