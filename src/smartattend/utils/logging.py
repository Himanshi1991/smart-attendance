"""Console + file logging shared by every module."""
from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "smartattend", log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(console)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
    logger.propagate = False
    return logger
