"""SB3 callback that logs per-rollout training metrics to a CSV file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """Write a row to *csv_path* after every rollout collection.

    Columns: ``timestep``, ``mean_reward``, ``mean_ep_length``,
    ``episodes``, ``goals``, ``shots``.

    Goals and shots are extracted from ``info`` dicts surfaced by the
    environment wrappers when available.
    """

    def __init__(self, csv_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_goals: list[int] = []
        self._header_written = False

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", "mean_reward", "mean_ep_length",
                "episodes", "total_goals",
            ])
        self._header_written = True

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            maybe_ep = info.get("episode")
            if maybe_ep is not None:
                self._ep_rewards.append(maybe_ep["r"])
                self._ep_lengths.append(maybe_ep["l"])
                self._ep_goals.append(info.get("episode_goals", 0))
        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_rewards:
            return
        row = [
            self.num_timesteps,
            float(np.mean(self._ep_rewards)),
            float(np.mean(self._ep_lengths)),
            len(self._ep_rewards),
            sum(self._ep_goals),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._ep_rewards.clear()
        self._ep_lengths.clear()
        self._ep_goals.clear()
