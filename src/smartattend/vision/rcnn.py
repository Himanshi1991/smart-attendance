"""R-CNN and Fast R-CNN, written from their papers on top of torchvision backbones (Requirement 8).

R-CNN (Girshick et al., 2014)       proposals -> crop + warp EACH region -> CNN features per region
                                    -> linear classifier (person / background) + linear box regressor -> NMS
Fast R-CNN (Girshick, 2015)         ONE CNN pass over the whole image -> RoIAlign cuts each proposal out of the shared
                                    feature map -> small fully connected head trained with a joint (multi-task) loss:
                                    classification + box regression -> NMS
Faster R-CNN (Ren et al., 2015)     replaces selective search with a learned Region Proposal Network (torchvision)

Simplifications (CPU budget), stated in the guide: ImageNet backbones are frozen; R-CNN uses logistic regression
instead of SVMs (same linear decision rule); Fast R-CNN uses ResNet18 up to layer3 (stride 16) with RoIAlign.
Proposals: OpenCV selective search, precomputed by scripts/selective_search.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------- box helpers
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def encode(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Box-regression targets of the R-CNN paper: centre shift in box widths, log size ratio."""
    pw, ph = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
    gw, gh = g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
    return np.stack([((g[:, 0] + g[:, 2]) - (p[:, 0] + p[:, 2])) / 2 / pw,
                     ((g[:, 1] + g[:, 3]) - (p[:, 1] + p[:, 3])) / 2 / ph,
                     np.log(gw / pw), np.log(gh / ph)], 1)


def decode(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    pw, ph = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
    cx, cy = (p[:, 0] + p[:, 2]) / 2 + d[:, 0] * pw, (p[:, 1] + p[:, 3]) / 2 + d[:, 1] * ph
    w, h = pw * np.exp(np.clip(d[:, 2], -2, 2)), ph * np.exp(np.clip(d[:, 3], -2, 2))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> np.ndarray:
    order, keep = np.argsort(-scores), []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][box_iou(boxes[i:i + 1], boxes[order[1:]])[0] < thr]
    return np.array(keep, int)


def sample_regions(props: np.ndarray, gt: np.ndarray, rng: np.random.Generator, n: int, pos_frac: float,
                   pos_iou: float, neg_hi: float, neg_lo: float = 0.0, jitter_gt: bool = True):
    """Training regions: ground-truth boxes (+ jittered copies) and proposals with IoU >= pos_iou are positives;
    proposals with neg_lo <= IoU < neg_hi are background."""
    gt = np.asarray(gt, float).reshape(-1, 4)
    gt = gt[(gt[:, 2] - gt[:, 0] > 2) & (gt[:, 3] - gt[:, 1] > 2)]      # a zero-size label would give log(0) targets
    if not len(gt):
        neg_lo = 0.0        # a frame with no students: every proposal is background (otherwise the batch is empty)
    iou = box_iou(props, gt).max(1) if len(gt) else np.zeros(len(props))
    pos = [gt]
    if jitter_gt and len(gt):
        wh = np.repeat(gt[:, 2:] - gt[:, :2], 2, axis=1)
        pos.append(gt + rng.normal(scale=0.08, size=gt.shape) * wh)
    pos.append(props[iou >= pos_iou])
    pos = np.concatenate(pos) if len(gt) else np.zeros((0, 4))
    pos = pos[rng.permutation(len(pos))[:int(n * pos_frac)]]
    neg = props[(iou < neg_hi) & (iou >= neg_lo)]
    neg = neg[rng.permutation(len(neg))[:n - len(pos)]]
    regions = np.concatenate([pos, neg]) if len(neg) else pos
    labels = np.r_[np.ones(len(pos)), np.zeros(len(regions) - len(pos))].astype(int)
    targets = np.zeros((len(regions), 4))
    if len(gt) and len(pos):
        match = box_iou(pos, gt).argmax(1)
        targets[:len(pos)] = encode(pos, gt[match])
    return regions, labels, targets


# --------------------------------------------------------------------------- backbones
def backbone(arch: str):
    """Frozen ImageNet CNN that maps a batch of warped crops to one feature vector each."""
    import torch
    import torchvision

    if arch == "resnet18":
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        net = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)
        net.aux_logits, net.aux1, net.aux2 = False, None, None
    net.fc = torch.nn.Identity()
    return net.eval()


def warp_crops(img_rgb: np.ndarray, boxes: np.ndarray, size: int, context: float = 0.1) -> np.ndarray:
    """R-CNN's 'warp': each region (plus a little context) is stretched to size x size, whatever its shape."""
    H, W = img_rgb.shape[:2]
    out = np.empty((len(boxes), size, size, 3), np.float32)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        mx, my = context * (x1 - x0), context * (y1 - y0)
        a, b = int(max(0, x0 - mx)), int(max(0, y0 - my))
        c, d = int(min(W, max(a + 2, x1 + mx))), int(min(H, max(b + 2, y1 + my)))
        out[i] = cv2.resize(img_rgb[b:d, a:c], (size, size))
    return (out / 255.0 - MEAN) / STD


def crop_features(net, img_rgb: np.ndarray, boxes: np.ndarray, size: int, batch: int = 128) -> np.ndarray:
    import torch

    feats = []
    with torch.inference_mode():
        for s in range(0, len(boxes), batch):
            x = torch.from_numpy(warp_crops(img_rgb, boxes[s:s + batch], size).transpose(0, 3, 1, 2).copy())
            feats.append(net(x).numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 512))


# --------------------------------------------------------------------------- R-CNN
class RCNN:
    def __init__(self, arch: str = "resnet18", size: int = 112):
        self.arch, self.size = arch, size
        self.net = backbone(arch)

    def train_features(self, records: list[dict], proposals: dict, per_image: int = 64, seed: int = 0, log=print):
        """The slow part of R-CNN training: one CNN pass per sampled region. Returns raw features, labels, box targets."""
        rng = np.random.default_rng(seed)
        F, Y, T = [], [], []
        t0 = time.perf_counter()
        for i, r in enumerate(records):
            img = cv2.imread(str(r["path"]))[:, :, ::-1]
            regions, labels, targets = sample_regions(np.array(proposals[r["path"].name]).reshape(-1, 4), r["boxes"], rng,
                                                      per_image, pos_frac=0.25, pos_iou=0.6, neg_hi=0.3)
            F.append(crop_features(self.net, img, regions, self.size))
            Y.append(labels)
            T.append(targets)
            if (i + 1) % 50 == 0:
                log(f"[rcnn-{self.arch}] features {i + 1}/{len(records)} ({time.perf_counter() - t0:.0f} s)")
        return np.concatenate(F), np.concatenate(Y), np.concatenate(T)

    def fit_heads(self, F: np.ndarray, Y: np.ndarray, T: np.ndarray, C: float = 0.05, alpha: float = 1000.0) -> "RCNN":
        """The fast part: a linear person/background classifier and a linear box regressor on the cached features.
        C (classifier strength) and alpha (regressor penalty) are hyperparameters, chosen on validation frames."""
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        self.C, self.alpha = C, alpha
        self.scaler = StandardScaler().fit(F)
        Fs = self.scaler.transform(F)
        self.clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Fs, Y)
        self.reg = Ridge(alpha=alpha).fit(Fs[Y == 1], T[Y == 1])        # box regression on positives only
        return self

    def fit(self, records: list[dict], proposals: dict, per_image: int = 64, seed: int = 0, log=print) -> dict:
        t0 = time.perf_counter()
        F, Y, T = self.train_features(records, proposals, per_image, seed, log)
        self.fit_heads(F, Y, T)
        return {"train_regions": int(len(Y)), "positives": int(Y.sum()), "feature_seconds": time.perf_counter() - t0}

    def features(self, img_bgr: np.ndarray, props: np.ndarray) -> np.ndarray:
        return crop_features(self.net, img_bgr[:, :, ::-1], np.asarray(props, float).reshape(-1, 4), self.size)

    def detect_from_features(self, props: np.ndarray, F: np.ndarray, nms_thr: float = 0.3):
        props = np.asarray(props, float).reshape(-1, 4)
        Fs = self.scaler.transform(F)
        scores = self.clf.predict_proba(Fs)[:, 1]
        boxes = decode(props, self.reg.predict(Fs))
        keep = scores >= 0.05
        boxes, scores = boxes[keep], scores[keep]
        k = nms(boxes, scores, nms_thr)
        return boxes[k], scores[k]

    def detect(self, img_bgr: np.ndarray, props: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.detect_from_features(props, self.features(img_bgr, props))


# --------------------------------------------------------------------------- Fast R-CNN
class FastRCNN:
    """ResNet18 trunk up to layer3 (stride 16, 256 channels), run once per image; RoIAlign 7x7; fc 1024-1024;
    two outputs (person score, box deltas) trained together with one loss."""

    def __init__(self, short_side: int = 720):
        import torch
        import torchvision

        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
        self.trunk = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3).eval()
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.head = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(256 * 7 * 7, 1024), torch.nn.ReLU(),
                                        torch.nn.Dropout(0.3), torch.nn.Linear(1024, 1024), torch.nn.ReLU())
        self.cls = torch.nn.Linear(1024, 2)
        self.box = torch.nn.Linear(1024, 4)
        self.short_side, self.torch = short_side, torch

    def _features(self, img_bgr: np.ndarray):
        torch = self.torch
        s = self.short_side / min(img_bgr.shape[:2])
        img = cv2.resize(img_bgr, None, fx=s, fy=s)[:, :, ::-1].astype(np.float32) / 255.0
        x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())[None]
        with torch.inference_mode():
            return self.trunk(x).clone(), s

    def _roi(self, fmap, boxes: np.ndarray, scale: float):
        import torchvision

        rois = self.torch.from_numpy(np.c_[np.zeros(len(boxes)), boxes * scale].astype(np.float32))
        return torchvision.ops.roi_align(fmap, rois, output_size=(7, 7), spatial_scale=1 / 16, sampling_ratio=2,
                                         aligned=True)

    def parameters(self):
        return [*self.head.parameters(), *self.cls.parameters(), *self.box.parameters()]

    def fit(self, records: list[dict], proposals: dict, epochs: int = 6, per_image: int = 64, lr: float = 1e-3,
            seed: int = 0, log=print) -> dict:
        torch = self.torch
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        fmaps = [self._features(cv2.imread(str(r["path"]))) for r in records]   # trunk is frozen: compute once
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        hist = []
        for ep in range(epochs):
            self.head.train()
            tot, t0 = [], time.perf_counter()
            for i in rng.permutation(len(records)):
                r, (fmap, s) = records[i], fmaps[i]
                regions, labels, targets = sample_regions(np.array(proposals[r["path"].name]).reshape(-1, 4), r["boxes"],
                                                          rng, per_image, pos_frac=0.25, pos_iou=0.5, neg_hi=0.5, neg_lo=0.1)
                if not len(regions):                               # an empty batch has a NaN loss and ruins the weights
                    continue
                h = self.head(self._roi(fmap, regions, s))
                y = torch.from_numpy(labels)
                loss = torch.nn.functional.cross_entropy(self.cls(h), y)
                if labels.any():                                   # box loss only where there is a person
                    pos = y == 1
                    loss = loss + torch.nn.functional.smooth_l1_loss(
                        self.box(h)[pos], torch.from_numpy(targets[labels == 1].astype(np.float32)) / 0.2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot.append(loss.item())
            hist.append(float(np.mean(tot)))
            log(f"[fast-rcnn] epoch {ep + 1}/{epochs} loss {hist[-1]:.3f} ({time.perf_counter() - t0:.0f} s)")
        return {"loss": hist}

    def detect(self, img_bgr: np.ndarray, props: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        props = np.asarray(props, float).reshape(-1, 4)
        fmap, s = self._features(img_bgr)
        self.head.eval()
        with torch.inference_mode():
            h = self.head(self._roi(fmap, props, s))
            scores = torch.softmax(self.cls(h), 1)[:, 1].numpy()
            deltas = (self.box(h) * 0.2).numpy()
        boxes = decode(props, deltas)
        keep = scores >= 0.05
        boxes, scores = boxes[keep], scores[keep]
        k = nms(boxes, scores, 0.3)
        return boxes[k], scores[k]

    def state_dict(self):
        return {"head": self.head.state_dict(), "cls": self.cls.state_dict(), "box": self.box.state_dict()}

    def load_state_dict(self, d):
        self.head.load_state_dict(d["head"]), self.cls.load_state_dict(d["cls"]), self.box.load_state_dict(d["box"])


# --------------------------------------------------------------------------- fine-tuning torchvision detectors
def finetune_torchvision(name: str, records: list[dict], epochs: int = 3, lr: float = 5e-3, max_size: int = 960,
                         seed: int = 0, log=print, ckpt: Path | None = None):
    """Faster R-CNN / RetinaNet: keep the COCO head (person = label 1), freeze the ResNet50 body, train FPN + RPN +
    heads on the training classrooms. Same idea as the YOLO fine-tune: adapt, do not relearn."""
    import torch
    import torchvision.models.detection as D

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if name.startswith("faster"):
        model = D.fasterrcnn_resnet50_fpn(weights=D.FasterRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=0,
                                          min_size=540, max_size=max_size)
    else:
        model = D.retinanet_resnet50_fpn(weights=D.RetinaNet_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=0,
                                         min_size=540, max_size=max_size)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(records))
    hist, start = [], 0
    if ckpt is not None and Path(ckpt).exists():                    # resume after an interruption (per-epoch checkpoint)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        hist, start = state["hist"], state["epoch"]
        for _ in range(start * len(records)):
            sched.step()
        for _ in range(start):
            rng.permutation(len(records))
        log(f"[finetune-{name}] resuming after epoch {start}")
    for ep in range(start, epochs):
        model.train()
        losses, t0 = [], time.perf_counter()
        for n, i in enumerate(rng.permutation(len(records))):
            r = records[i]
            img = cv2.imread(str(r["path"]))
            boxes = r["boxes"].copy()
            if rng.random() < 0.5:                                         # horizontal flip
                img = img[:, ::-1]
                boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img.shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img.shape[0])
            boxes = boxes[(boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)]
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.ones(len(boxes), dtype=torch.int64)}
            loss = sum(model([x], [target]).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            if (n + 1) % 50 == 0:
                log(f"[finetune-{name}] epoch {ep + 1} {n + 1}/{len(records)} loss {np.mean(losses[-50:]):.3f} "
                    f"({time.perf_counter() - t0:.0f} s)")
        hist.append(float(np.mean(losses)))
        if ckpt is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "epoch": ep + 1}, ckpt)
    model.eval()
    return model, hist


class TorchvisionFinetuned:
    """Wraps a fine-tuned torchvision detector so it looks like the project's Detector (boxes, scores for person)."""

    def __init__(self, model, name: str):
        import torch

        self.model, self.name, self.torch = model, name, torch
        if hasattr(model, "roi_heads"):
            model.roi_heads.score_thresh = 0.01
        else:
            model.score_thresh = 0.01

    def __call__(self, image_bgr, score_thr=0.05):
        torch = self.torch
        x = torch.from_numpy(image_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            out = self.model([x])[0]
        keep = (out["labels"] == 1) & (out["scores"] >= score_thr)
        return out["boxes"][keep].numpy().astype(np.float64), out["scores"][keep].numpy().astype(np.float64)

    def timed(self, image_bgr, score_thr=0.05):
        t0 = time.perf_counter()
        r = self(image_bgr, score_thr)
        return r, (time.perf_counter() - t0) * 1000


def load_proposals(processed: Path, split: str) -> dict:
    return json.loads((Path(processed) / f"proposals_{split}.json").read_text())
