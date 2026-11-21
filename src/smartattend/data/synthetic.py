"""Synthetic per-student engagement features.

PLACEHOLDER DATA: this stands in for the features the vision pipeline will extract
(gaze, head pose, posture, phone use, ...) so the NumPy fundamentals can be developed and
tested before real labelled data exists. It must never be reported as real-world performance.
"""
from __future__ import annotations

import numpy as np

FEATURES = ["gaze_on_board", "head_yaw_abs", "head_pitch", "eye_openness",
            "posture_upright", "hand_activity", "phone_visible", "talking_to_neighbor"]
CLASSES = ["disengaged", "neutral", "engaged"]
GROUPS = ["front_rows", "back_rows"]


def make_engagement_features(n: int = 1200, seed: int = 0, noise: float = 0.6):
    """Return X (n, 8), y (n,) in {0,1,2}, group (n,) in {0,1}.

    A latent engagement score drives the features; back-row students get noisier features
    (smaller faces, more occlusion), which gives the fairness analysis something real to find.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n)
    latent = rng.normal(size=n)
    phone = (rng.random(n) < 1 / (1 + np.exp(2.0 * latent + 1.0))).astype(float)
    talking = (rng.random(n) < 0.15 + 0.15 * (latent < -0.5)).astype(float)
    sigma = noise * np.where(group == 1, 1.6, 1.0)

    def feat(a, b=0.0):
        return a * latent + b + sigma * rng.normal(size=n)

    X = np.column_stack([
        feat(1.0),                                   # gaze on board
        np.abs(feat(-0.6, 0.2)),                     # |head yaw| (nonlinear)
        feat(0.4) - 0.8 * phone,                     # head pitch (looking down at phone)
        feat(0.5, 0.5),                              # eye openness
        feat(0.7),                                   # upright posture
        np.tanh(feat(0.8)) + 0.3 * rng.normal(size=n),  # hand activity (note-taking)
        phone,
        talking,
    ])
    effective = latent - 0.9 * phone - 0.6 * talking
    y = np.digitize(effective, np.quantile(effective, [1 / 3, 2 / 3]))
    return X, y.astype(int), group


def leakage_safe_split(n: int, val: float = 0.15, test: float = 0.15, seed: int = 0):
    """Random index split. For real classroom data, split by session/video instead (see dataset card)."""
    order = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(n * test), int(n * val)
    return order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test]


class Standardizer:
    """Fit mean/std on TRAIN ONLY, then apply to val/test - fitting on all data would leak."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std
