"""Fine-tune YOLOv8-n on the curated classroom person dataset (CPU).

The data.yaml keeps all 80 COCO class names so the pretrained detection head is reused intact
(person is COCO class 0, matching our labels); re-initialising a 1-class head on 450 images
would throw away most of what the model knows.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]


def write_data_yaml(dataset_dir: Path) -> Path:
    path = dataset_dir / "data80.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO80))
    path.write_text(f"path: {dataset_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                    f"test: images/test\nnames:\n{names}\n")
    return path


def finetune(dataset_dir: str | Path, models_dir: str | Path, out_dir: str | Path, epochs: int = 6,
             imgsz: int = 640, freeze: int = 10, seed: int = 0, run_name: str = "yolov8n_classroom",
             target_name: str = "yolov8n-classroom.pt", batch: int = 8) -> dict:
    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"          # GPU on Colab, CPU on the laptop

    dataset_dir, models_dir, out_dir = Path(dataset_dir), Path(models_dir), Path(out_dir)
    data = write_data_yaml(dataset_dir)
    model = YOLO(str(models_dir / "yolo" / "yolov8n.pt"))
    res = model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=2,
                      freeze=freeze, lr0=0.002, lrf=0.1, warmup_epochs=1, cos_lr=True, seed=seed,
                      deterministic=True, project=str(out_dir.resolve()), name=run_name, exist_ok=True,
                      plots=True, verbose=False, mosaic=0.5, close_mosaic=3)
    best = Path(res.save_dir) / "weights" / "best.pt"
    target = models_dir / "yolo" / target_name
    shutil.copy(best, target)
    summary = {"weights": str(target), "epochs": epochs, "imgsz": imgsz, "freeze_layers": freeze, "seed": seed,
               "metrics_val": {k: float(v) for k, v in res.results_dict.items()}}
    (out_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
