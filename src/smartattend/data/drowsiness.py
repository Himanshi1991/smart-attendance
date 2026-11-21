"""Engagement (alertness) data from the MIT-licensed drowsiness face dataset.

The upstream train/val/test CSVs split consecutive webcam frames at random, so near-identical
neighbouring frames end up in different splits (leakage). We pool all rows, sort by capture time,
cut the timeline into blocks, assign whole blocks to splits and drop a buffer of frames at every
block boundary.

Label mapping (upstream 0-4 drowsiness class -> engagement):
    0 alert            -> engaged
    1-2 mild/moderate  -> neutral
    3-4 drowsy/severe  -> disengaged
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ENGAGEMENT_CLASSES = ["engaged", "neutral", "disengaged"]
LABEL_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
FEATURES = ["ear", "mar", "head_roll", "head_pitch", "head_combined"]
_TS = re.compile(r"(\d{8})_(\d{6})_(\d{3})")


def _timestamp(path: str) -> float:
    d, t, ms = _TS.search(path).groups()
    return datetime.strptime(d + t, "%Y%m%d%H%M%S").timestamp() + int(ms) / 1000


def load_frames(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    df = pd.concat([pd.read_csv(raw_dir / f"{s}.csv") for s in ["train", "val", "test"]], ignore_index=True)
    df = df.drop_duplicates("image").copy()
    df["t"] = df["image"].map(_timestamp)
    df = df.sort_values("t").reset_index(drop=True)
    df["path"] = df["image"].map(lambda p: str(raw_dir / p))
    df["engagement"] = df["drowsiness_class"].map(LABEL_MAP).astype(int)
    # recording sessions: a gap of > 10 s starts a new session
    df["session"] = (df["t"].diff().fillna(0) > 10).cumsum()
    return df


def time_block_split(df: pd.DataFrame, block_s: float = 5.0, buffer_s: float = 0.5,
                     fractions=(0.7, 0.15, 0.15), seed: int = 0) -> pd.DataFrame:
    """Assign contiguous time blocks to train/val/test; frames within buffer_s of a block edge are dropped.

    Blocks are stratified: sorted by their majority label, then dealt out so every split sees every class.
    """
    df = df.copy()
    t0 = df.groupby("session")["t"].transform("min")
    df["block"] = df["session"].astype(str) + "_" + ((df["t"] - t0) // block_s).astype(int).astype(str)
    edges = df.groupby("block")["t"].agg(["min", "max"])
    df = df[(df["t"] - df["block"].map(edges["min"]) >= buffer_s) &
            (df["block"].map(edges["max"]) - df["t"] >= buffer_s)].copy()

    # stratified allocation: within each majority-label class, shuffle its blocks and deal them
    # 70/15/15, guaranteeing every split gets at least one block of every class (if >= 3 blocks)
    rng = np.random.default_rng(seed)
    blocks = df.groupby("block")["engagement"].agg(lambda s: s.mode().iloc[0])
    assignment: dict[str, str] = {}
    for cls, grp in blocks.groupby(blocks):
        names = list(grp.index)
        rng.shuffle(names)
        n = len(names)
        n_val = max(1, round(fractions[1] * n)) if n >= 3 else (1 if n == 2 else 0)
        n_test = max(1, round(fractions[2] * n)) if n >= 3 else 0
        for i, b in enumerate(names):
            assignment[b] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df["split"] = df["block"].map(assignment)
    return df


def load_split_frames(raw_dir: str | Path, seed: int = 0) -> pd.DataFrame:
    return time_block_split(load_frames(raw_dir), seed=seed)
