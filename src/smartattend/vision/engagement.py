"""Engagement (alertness) classifier on face crops: ResNet-18 / GoogLeNet transfer learning.

Two training regimes are compared (both CPU-friendly):
  * linear probe : frozen ImageNet backbone, features cached once, linear head trained
  * fine-tune    : ResNet-18 layer4 + head trained with augmentation
Class imbalance is handled with inverse-frequency class weights; model selection uses val macro-F1.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M

from smartattend.data.drowsiness import ENGAGEMENT_CLASSES
from smartattend.eval.metrics import classification_report

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_face(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the training domain: grayscale + CLAHE, replicated to 3 channels, ImageNet-normalised CHW."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = _CLAHE.apply(cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA))
    x = np.repeat(g[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip, small rotation/scale, brightness/contrast jitter, blur - mimics classroom crops."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-12, 12), rng.uniform(0.9, 1.1))
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-30, 30))
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def backbone(name: str) -> tuple[nn.Module, int]:
    if name == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "googlenet":
        m = M.googlenet(weights=M.GoogLeNet_Weights.DEFAULT)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(name)
    return m.eval(), dim


def _load_gray(path: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)


@torch.inference_mode()
def extract_features(model: nn.Module, paths: list[str], batch: int = 32) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), batch):
        x = torch.from_numpy(np.stack([preprocess_face(_load_gray(p)) for p in paths[i:i + batch]]))
        feats.append(model(x).numpy())
    return np.concatenate(feats)


def class_weights(y: np.ndarray, n: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=n).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1) / n
    return torch.tensor(w)


def train_linear_probe(Xtr, ytr, Xva, yva, epochs: int = 200, lr: float = 1e-3, wd: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    head = nn.Linear(Xtr.shape[1], 3)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr))
    xt, yt = torch.tensor((Xtr - mu) / sd), torch.tensor(ytr)
    xv = torch.tensor((Xva - mu) / sd)
    best, best_state = -1, None
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(xt), yt).backward()
        opt.step()
        with torch.no_grad():
            f1 = classification_report(yva, head(xv).argmax(1).numpy(), 3)["macro_f1"]
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, mu, sd


class EngagementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, dim = backbone("resnet18")
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 3))

    def forward(self, x):
        return self.head(self.backbone(x))

    def embed(self, x):
        return self.backbone(x)


def finetune_resnet18(train_df, val_df, epochs: int = 4, lr: float = 3e-4, batch: int = 32, seed: int = 0,
                      logger: logging.Logger | None = None) -> tuple[EngagementNet, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = EngagementNet()
    for name, p in net.backbone.named_parameters():
        p.requires_grad = name.startswith("layer4")
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    ytr = train_df["engagement"].to_numpy()
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr))
    imgs = [_load_gray(p) for p in train_df["path"]]
    xv = torch.from_numpy(np.stack([preprocess_face(_load_gray(p)) for p in val_df["path"]]))
    yv = val_df["engagement"].to_numpy()
    history, best, best_state = [], -1, None
    for ep in range(epochs):
        net.train()
        net.backbone.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)  # keep BN stats
        order = rng.permutation(len(imgs))
        t0, tot = time.time(), 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.from_numpy(np.stack([preprocess_face(augment(imgs[j], rng)) for j in idx]))
            loss = lossf(net(x), torch.from_numpy(ytr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        net.eval()
        with torch.inference_mode():
            pv = torch.cat([net(xv[i:i + 64]) for i in range(0, len(xv), 64)]).argmax(1).numpy()
        rep = classification_report(yv, pv, 3)
        history.append({"epoch": ep + 1, "train_loss": tot / len(order), "val_macro_f1": rep["macro_f1"],
                        "val_acc": rep["accuracy"], "seconds": time.time() - t0})
        if logger:
            logger.info(f"[finetune] epoch {ep + 1} loss={history[-1]['train_loss']:.4f} "
                        f"val_macroF1={rep['macro_f1']:.3f} ({history[-1]['seconds']:.0f}s)")
        if rep["macro_f1"] > best:
            best, best_state = rep["macro_f1"], {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net.eval(), history


class EngagementClassifier:
    """Inference wrapper used by the product pipeline."""

    def __init__(self, weights: str | Path):
        self.net = EngagementNet()
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()

    @torch.inference_mode()
    def predict_proba(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 3), np.float32)
        x = torch.from_numpy(np.stack([preprocess_face(c) for c in crops]))
        return torch.softmax(self.net(x), 1).numpy()

    @torch.inference_mode()
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack([preprocess_face(c) for c in crops]))
        return self.net.embed(x).numpy()


def save_report(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


__all__ = ["ENGAGEMENT_CLASSES", "EngagementClassifier", "preprocess_face", "finetune_resnet18",
           "train_linear_probe", "extract_features", "backbone"]
