"""SB3 callback that logs per-rollout training metrics to a CSV file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from opengoalrl.metrics.tactical import TacticalMetricsTracker


class MetricsCallback(BaseCallback):
    """Write rollout metrics to *csv_path* after every collection.

    Columns: ``timestep``, ``mean_reward``, ``mean_ep_length``,
    ``episodes``, ``total_goals``, and optional tactical aggregates.
    """

    TACTICAL_FIELDS = [
        "mean_distance_advanced",
        "mean_box_entries",
        "mean_possession_losses",
        "mean_approx_xg",
    ]

    def __init__(
        self,
        csv_path: str | Path,
        verbose: int = 0,
        include_tactical: bool = False,
    ) -> None:
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.include_tactical = include_tactical
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_goals: list[int] = []
        self._tactical_trackers: list[TacticalMetricsTracker] = []
        self._active_tracker: TacticalMetricsTracker | None = None
        self._header_written = False

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "timestep", "mean_reward", "mean_ep_length",
            "episodes", "total_goals",
        ]
        if self.include_tactical:
            header.extend(self.TACTICAL_FIELDS)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        self._header_written = True

    def _on_step(self) -> bool:
        if self.include_tactical and self._active_tracker is None:
            self._active_tracker = TacticalMetricsTracker()

        for info in self.locals.get("infos", []):
            maybe_ep = info.get("episode")
            if maybe_ep is not None:
                self._ep_rewards.append(maybe_ep["r"])
                self._ep_lengths.append(maybe_ep["l"])
                self._ep_goals.append(info.get("episode_goals", 0))
                if self._active_tracker is not None:
                    self._tactical_trackers.append(self._active_tracker)
                    self._active_tracker = TacticalMetricsTracker()
            elif self._active_tracker is not None:
                obs = self.locals.get("new_obs")
                actions = self.locals.get("actions")
                if obs is not None and actions is not None:
                    action = int(actions[0]) if hasattr(actions, "__len__") else int(actions)
                    obs_arr = obs[0] if getattr(obs, "ndim", 1) > 1 else obs
                    self._active_tracker.step(obs_arr, action, info)

        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_rewards:
            return
        row: list[Any] = [
            self.num_timesteps,
            float(np.mean(self._ep_rewards)),
            float(np.mean(self._ep_lengths)),
            len(self._ep_rewards),
            sum(self._ep_goals),
        ]
        if self.include_tactical and self._tactical_trackers:
            finals = [t.finalize() for t in self._tactical_trackers]
            row.extend([
                float(np.mean([m.distance_advanced for m in finals])),
                float(np.mean([m.box_entries for m in finals])),
                float(np.mean([m.possession_losses for m in finals])),
                float(np.mean([m.approx_xg for m in finals])),
            ])
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._ep_rewards.clear()
        self._ep_lengths.clear()
        self._ep_goals.clear()
        self._tactical_trackers.clear()
        self._active_tracker = TacticalMetricsTracker() if self.include_tactical else None
