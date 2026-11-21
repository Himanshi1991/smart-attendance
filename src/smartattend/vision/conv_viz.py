"""Visualising convolution: a hand-computed 2-D convolution, learned ResNet filters / feature maps, and a
3-D (spatio-temporal) convolution over a short frame stack that responds to motion."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def conv2d_naive(img: np.ndarray, k: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Explicit sliding-window cross-correlation (what deep-learning 'convolution' layers compute)."""
    img = np.pad(img.astype(np.float64), pad)
    kh, kw = k.shape
    H = (img.shape[0] - kh) // stride + 1
    W = (img.shape[1] - kw) // stride + 1
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(img[i * stride:i * stride + kh, j * stride:j * stride + kw] * k)
    return out


def conv3d_naive(vol: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Valid 3-D cross-correlation over (T, H, W)."""
    kt, kh, kw = k.shape
    T, H, W = vol.shape[0] - kt + 1, vol.shape[1] - kh + 1, vol.shape[2] - kw + 1
    out = np.zeros((T, H, W))
    for t in range(kt):
        for i in range(kh):
            for j in range(kw):
                out += k[t, i, j] * vol[t:t + T, i:i + H, j:j + W]
    return out


KERNELS = {
    "sobel_x (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
    "sobel_y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
    "box blur 5x5": np.ones((5, 5)) / 25,
}


def make_figures(image_path: str | Path, out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchvision.models as M

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY) / 255.0

    # 1. step-by-step 2D convolution on a tiny patch + full-image results
    fig = plt.figure(figsize=(16, 6.6))
    k = KERNELS["sobel_x (vertical edges)"]
    # take the 7x7 patch with the strongest vertical edge, so the worked numbers are not all zero
    resp = np.abs(cv2.filter2D(gray, -1, k))
    score = cv2.boxFilter(resp, -1, (7, 7), normalize=False)[3:-3, 3:-3]
    r0, c0 = np.unravel_index(np.argmax(score), score.shape)
    patch = np.round(gray[r0:r0 + 7, c0:c0 + 7] * 255)                    # pixel values 0-255
    res = conv2d_naive(patch, k)
    ax = fig.add_subplot(2, 5, 1)
    ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
    for (i, j), v in np.ndenumerate(patch):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="red" if i < 3 and j < 3 else "tab:blue")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="red", lw=2))
    ax.set_title(f"7x7 input patch (pixels {r0}-{r0 + 6}, {c0}-{c0 + 6})\nred = receptive field of out[0,0]", fontsize=9)
    ax = fig.add_subplot(2, 5, 2)
    ax.imshow(k, cmap="bwr")
    for (i, j), v in np.ndenumerate(k):
        ax.text(j, i, f"{v:g}", ha="center", va="center")
    ax.set_title("3x3 kernel (Sobel-x)", fontsize=9)
    ax = fig.add_subplot(2, 5, 3)
    lim = np.abs(res).max() or 1
    ax.imshow(res, cmap="bwr", vmin=-lim, vmax=lim)
    for (i, j), v in np.ndenumerate(res):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"5x5 output (no padding, stride 1)\nout[0,0] = sum(red 3x3 * kernel) = {res[0, 0]:.0f}", fontsize=9)
    ax = fig.add_subplot(2, 5, 4)
    ax.imshow(gray, cmap="gray"), ax.set_title("input image", fontsize=9)
    ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), 7, 7, fill=False, ec="red", lw=1.5))
    s2 = conv2d_naive(gray[::2, ::2], k, stride=2, pad=1)
    ax = fig.add_subplot(2, 5, 5)
    ax.imshow(np.abs(s2), cmap="magma"), ax.set_title(f"stride 2, pad 1 -> {s2.shape}", fontsize=9)
    for n, (name, kk) in enumerate(KERNELS.items()):
        ax = fig.add_subplot(2, 5, 6 + n)
        ax.imshow(np.abs(cv2.filter2D(gray, -1, kk)), cmap="magma"), ax.set_title(name, fontsize=9)
    for a in fig.axes:
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("2-D convolution: a kernel slides over the image computing inner products")
    paths.append(out_dir / "conv2d_explained.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 2. learned ResNet-18 conv1 filters and feature maps at increasing depth
    net = M.resnet18(weights=M.ResNet18_Weights.DEFAULT).eval()
    w = net.conv1.weight.detach().numpy()
    x = cv2.resize(img, (224, 224))[:, :, ::-1].astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()[None]
    feats = {}
    with torch.inference_mode():
        h = net.maxpool(net.relu(net.bn1(net.conv1(t))))
        feats["conv1 (112x112)"] = net.relu(net.bn1(net.conv1(t)))
        h = net.layer1(h); feats["layer1 (56x56)"] = h
        h = net.layer2(h); feats["layer2 (28x28)"] = h
        h = net.layer3(h); feats["layer3 (14x14)"] = h
        h = net.layer4(h); feats["layer4 (7x7)"] = h
    fig, axes = plt.subplots(6, 8, figsize=(14, 11))
    for i, ax in enumerate(axes[:2].ravel()):
        f = w[i].transpose(1, 2, 0)
        ax.imshow((f - f.min()) / (f.max() - f.min()))
        ax.set_title(f"filter {i}", fontsize=7)
    for r, (name, f) in enumerate(feats.items(), start=2):
        if r >= 6:
            break
        top = f[0].mean((1, 2)).argsort(descending=True)[:8]
        for c, ch in enumerate(top):
            axes[r, c].imshow(f[0, ch].numpy(), cmap="viridis")
            axes[r, c].set_title(f"{name} ch{int(ch)}", fontsize=6)
    for a in axes.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("ResNet-18: learned 7x7 conv1 filters (rows 1-2) and most active feature maps by depth")
    paths.append(out_dir / "resnet_filters_featuremaps.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 3. 3-D convolution: temporal-difference kernel over a synthetic 'student raises hand' clip
    T, H, W = 8, 64, 64
    vol = np.zeros((T, H, W))
    for t in range(T):
        vol[t, 30:60, 26:38] = 0.8                      # torso (static)
        vol[t, 18:30, 27:37] = 0.9                      # head (static)
        y = 40 - 4 * t                                  # arm rising over time
        vol[t, max(y, 2):max(y, 2) + 14, 40:45] = 1.0
    k3 = np.zeros((2, 3, 3))
    k3[0], k3[1] = -1 / 9, 1 / 9                        # frame difference of 3x3 averages
    resp = conv3d_naive(vol, k3)
    fig, axes = plt.subplots(2, 7, figsize=(16, 5))
    for t in range(7):
        axes[0, t].imshow(vol[t], cmap="gray"), axes[0, t].set_title(f"frame {t}", fontsize=8)
        axes[1, t].imshow(resp[t], cmap="bwr", vmin=-1, vmax=1), axes[1, t].set_title(f"3D conv out t={t}", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("3-D convolution (kernel 2x3x3 = temporal difference): static body cancels, moving arm lights up")
    paths.append(out_dir / "conv3d_motion.png")
    fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close(fig)
    return paths
