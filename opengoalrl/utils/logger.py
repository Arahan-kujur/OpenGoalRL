"""Logging utilities for training and evaluation."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def get_logger(
    name: str = "opengoalrl",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a configured logger with console and optional file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"run_{stamp}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_episode(
    logger: logging.Logger,
    episode: int,
    reward: float,
    steps: int,
    goals: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log a single episode summary."""
    msg = f"Episode {episode:>5d} | reward={reward:+8.2f} | steps={steps:>4d} | goals={goals}"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


def save_config_snapshot(config: dict[str, Any], save_dir: str | Path) -> Path:
    """Persist the full config dict as JSON for reproducibility."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_dir / f"config_{stamp}.json"
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)
    return path
