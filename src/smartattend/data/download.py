"""Reproducible download of every public dataset used by the project (see docs/DATASET_CARD.md).

    python -m smartattend data download

| name        | source                                                    | licence                         |
|-------------|-----------------------------------------------------------|---------------------------------|
| coco        | COCO 2017 val annotations + selected images                | annotations CC BY 4.0; images: per-image Flickr licence recorded |
| lfw         | Labeled Faces in the Wild (HF mirror marcelohaps/lfw)      | research use; attributes from Columbia (Kumar et al. 2009) |
| drowsiness  | HF notgoodkeeper/cnn-based-drowsiness-detection-data       | MIT                             |
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{}"
LFW_PARQUET_URL = "https://huggingface.co/api/datasets/marcelohaps/lfw/parquet/default/train/0.parquet"
LFW_ATTR_URL = "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"
DROWSY_BASE = "https://huggingface.co/datasets/notgoodkeeper/cnn-based-drowsiness-detection-data/resolve/main/"


def _fetch(url: str, dest: Path, min_size: int = 1, retries: int = 8) -> Path:
    """Download with resume (HTTP Range) and retries - large files over flaky connections."""
    if dest.exists() and dest.stat().st_size >= min_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "smartattend/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(tmp, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            break
        except (OSError, urllib.error.URLError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    tmp.replace(dest)
    return dest


def _fetch_many(pairs: list[tuple[str, Path]], workers: int = 16) -> int:
    with ThreadPoolExecutor(workers) as ex:
        return len(list(ex.map(lambda p: _fetch(*p), pairs)))


def download_coco_annotations(raw: Path, logger: logging.Logger) -> Path:
    out = raw / "coco" / "annotations"
    need = ["instances_val2017.json", "captions_val2017.json"]
    if all((out / n).exists() for n in need):
        return out
    zpath = _fetch(COCO_ANN_URL, raw / "coco" / "annotations_trainval2017.zip", 200_000_000)
    logger.info("extracting COCO val2017 annotations")
    with zipfile.ZipFile(zpath) as z:
        for n in need:
            (out / n).parent.mkdir(parents=True, exist_ok=True)
            (out / n).write_bytes(z.read(f"annotations/{n}"))
    zpath.unlink()  # train annotations are not used (too large for 8 GB RAM)
    return out


def download_coco_images(raw: Path, file_names: list[str], logger: logging.Logger) -> Path:
    img_dir = raw / "coco" / "images"
    logger.info(f"downloading {len(file_names)} COCO val2017 images")
    _fetch_many([(COCO_IMG_URL.format(n), img_dir / n) for n in file_names])
    return img_dir


def download_lfw(raw: Path, logger: logging.Logger) -> Path:
    d = raw / "lfw"
    logger.info("downloading LFW (parquet) + attributes")
    _fetch(LFW_PARQUET_URL, d / "lfw.parquet", 100_000_000)
    _fetch(LFW_ATTR_URL, d / "lfw_attributes.txt", 1_000_000)
    return d


def download_drowsiness(raw: Path, logger: logging.Logger) -> Path:
    import csv

    d = raw / "drowsiness"
    for split in ["train", "val", "test"]:
        _fetch(DROWSY_BASE + f"{split}.csv", d / f"{split}.csv")
    images = []
    for split in ["train", "val", "test"]:
        with open(d / f"{split}.csv", encoding="utf-8") as f:
            images += [row["image"] for row in csv.DictReader(f)]
    logger.info(f"downloading {len(images)} drowsiness face images")
    _fetch_many([(DROWSY_BASE + p, d / p) for p in images])
    return d


def download_all(raw: str | Path, logger: logging.Logger) -> None:
    from smartattend.data.coco_classroom import select_images

    raw = Path(raw)
    ann = download_coco_annotations(raw, logger)
    selection = select_images(ann)
    (raw / "coco" / "selection.json").write_text(json.dumps(selection, indent=1))
    download_coco_images(raw, sorted({im["file_name"] for split in selection.values() for im in split}), logger)
    download_lfw(raw, logger)
    download_drowsiness(raw, logger)
    logger.info("all datasets downloaded")
