"""Precompute selective-search region proposals for the Kaggle classroom frames (R-CNN and Fast R-CNN input).

Selective search lives in OpenCV's *contrib* build (cv2.ximgproc). To avoid replacing the project's OpenCV, this
script can load an unpacked contrib wheel from a separate folder, and runs on its own (run it with python -I):

    python -I scripts/selective_search.py --cv2-path models/tools/opencv_contrib
    (on Colab, opencv-contrib is usually already installed: omit --cv2-path)

Output: data/processed/kaggle_classroom/proposals_{train,val,test}.json  {image file name: [[x0,y0,x1,y1], ...]}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def proposals(cv2, img, width: int = 500, top: int = 1000, min_px: int = 20):
    """Selective search ('fast' mode) on a downscaled copy; boxes returned in original pixels, filtered and capped."""
    scale = width / img.shape[1]
    small = cv2.resize(img, (width, int(round(img.shape[0] * scale))))
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(small)
    ss.switchToSelectiveSearchFast()
    out, seen = [], set()
    for x, y, w, h in ss.process():
        X0, Y0, X1, Y1 = x / scale, y / scale, (x + w) / scale, (y + h) / scale
        bw, bh = X1 - X0, Y1 - Y0
        if bw < min_px or bh < min_px or not 0.25 <= bw / bh <= 4:
            continue
        key = (round(X0, -1), round(Y0, -1), round(X1, -1), round(Y1, -1))
        if key in seen:
            continue
        seen.add(key)
        out.append([round(X0, 1), round(Y0, 1), round(X1, 1), round(Y1, 1)])
        if len(out) >= top:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv2-path", help="folder containing an unpacked opencv-contrib 'cv2' package")
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed" / "kaggle_classroom"))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images per split (for timing)")
    args = ap.parse_args()
    if args.cv2_path:
        sys.path.insert(0, str(Path(args.cv2_path).resolve()))
    import cv2

    cv2.setNumThreads(4)
    if not hasattr(cv2, "ximgproc"):
        sys.exit("this cv2 has no ximgproc: install opencv-contrib-python or pass --cv2-path")
    processed = Path(args.processed)
    for split in args.splits.split(","):
        out_path = processed / f"proposals_{split}.json"
        done = json.loads(out_path.read_text()) if out_path.exists() and not args.limit else {}
        files = sorted((processed / "yolo_person" / "images" / split).glob("*.jpg"))
        if args.limit:
            files = files[:args.limit]
        t0 = time.perf_counter()
        for i, f in enumerate(files):
            if f.name not in done:
                done[f.name] = proposals(cv2, cv2.imread(str(f)))
            if (i + 1) % 25 == 0:
                print(f"{split}: {i + 1}/{len(files)}  {(time.perf_counter() - t0) / (i + 1):.2f} s/image", flush=True)
                if not args.limit:
                    out_path.write_text(json.dumps(done))
        if not args.limit:
            out_path.write_text(json.dumps(done))
        n = [len(v) for k, v in done.items() if k in {f.name for f in files}]
        print(f"{split}: {len(files)} images, {sum(n) / max(len(n), 1):.0f} proposals/image, "
              f"{(time.perf_counter() - t0) / max(len(files), 1):.2f} s/image" + ("" if args.limit else f" -> {out_path}"))


if __name__ == "__main__":
    main()
