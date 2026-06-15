"""Tactical metrics computed from simple115v2 observations and info."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_BALL_X_IDX = 88
_BALL_Y_IDX = 89
_BOX_X_MIN = 0.8
_BOX_Y_ABS_MAX = 0.20
_GOAL_X = 1.0
_GOAL_Y = 0.0
_SHOT_ACTION = 12
_PASS_ACTIONS = {9, 10, 11}
_DRIBBLE_ACTIONS = {5, 6, 7, 8}


@dataclass
class EpisodeTacticalMetrics:
    goals: int = 0
    shots: int = 0
    distance_advanced: float = 0.0
    box_entries: int = 0
    possession_losses: int = 0
    max_possession_steps: int = 0
    approx_xg: float = 0.0
    pressure_proxy: float = 0.0
    shot_timing_step: int | None = None
    action_histogram: dict[int, int] = field(default_factory=dict)


class TacticalMetricsTracker:
    """Accumulate tactical metrics over a single episode."""

    def __init__(self) -> None:
        self._start_ball_x: float | None = None
        self._prev_ball_x: float | None = None
        self._in_box = False
        self._box_entries = 0
        self._possession_steps = 0
        self._max_possession = 0
        self._possession_losses = 0
        self._goals = 0
        self._shots = 0
        self._shot_timing: int | None = None
        self._step = 0
        self._pressure_sum = 0.0
        self._pressure_count = 0
        self._actions: Counter[int] = Counter()

    def step(self, obs: np.ndarray, action: int, info: dict[str, Any]) -> None:
        self._step += 1
        ball_x = float(obs[_BALL_X_IDX])
        ball_y = float(obs[_BALL_Y_IDX])

        if self._start_ball_x is None:
            self._start_ball_x = ball_x
        self._prev_ball_x = ball_x

        in_box = ball_x >= _BOX_X_MIN and abs(ball_y) <= _BOX_Y_ABS_MAX
        if in_box and not self._in_box:
            self._box_entries += 1
        self._in_box = in_box

        if info.get("score_reward", 0.0) > 0:
            self._goals += 1

        if action == _SHOT_ACTION:
            self._shots += 1
            if self._shot_timing is None:
                self._shot_timing = self._step

        if info.get("possession_lost"):
            self._possession_losses += 1
            self._max_possession = max(self._max_possession, self._possession_steps)
            self._possession_steps = 0
        else:
            self._possession_steps += 1

        self._pressure_sum += _pressure_proxy(ball_x, ball_y)
        self._pressure_count += 1
        self._actions[int(action)] += 1

    def finalize(self) -> EpisodeTacticalMetrics:
        self._max_possession = max(self._max_possession, self._possession_steps)
        start_x = self._start_ball_x or 0.0
        end_x = self._prev_ball_x or start_x
        distance_advanced = max(0.0, end_x - start_x)
        pressure = (
            self._pressure_sum / self._pressure_count
            if self._pressure_count else 0.0
        )
        ball_y = 0.0
        approx_xg = _approx_xg(end_x, ball_y, self._shots > 0)
        return EpisodeTacticalMetrics(
            goals=self._goals,
            shots=self._shots,
            distance_advanced=distance_advanced,
            box_entries=self._box_entries,
            possession_losses=self._possession_losses,
            max_possession_steps=self._max_possession,
            approx_xg=approx_xg,
            pressure_proxy=pressure,
            shot_timing_step=self._shot_timing,
            action_histogram=dict(self._actions),
        )


def _pressure_proxy(ball_x: float, ball_y: float) -> float:
    """Higher when ball is central and not deep in attacking third."""
    centrality = 1.0 - min(1.0, abs(ball_y) / 0.42)
    depth_penalty = max(0.0, 1.0 - ball_x)
    return float(centrality * depth_penalty)


def _approx_xg(ball_x: float, ball_y: float, took_shot: bool) -> float:
    if not took_shot:
        return 0.0
    dist = np.sqrt((ball_x - _GOAL_X) ** 2 + (ball_y - _GOAL_Y) ** 2)
    return float(max(0.0, min(0.95, 1.0 - dist)))


def aggregate_tactical(metrics: list[EpisodeTacticalMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "mean_distance_advanced": float(np.mean([m.distance_advanced for m in metrics])),
        "mean_box_entries": float(np.mean([m.box_entries for m in metrics])),
        "mean_possession_losses": float(np.mean([m.possession_losses for m in metrics])),
        "mean_approx_xg": float(np.mean([m.approx_xg for m in metrics])),
        "mean_pressure_proxy": float(np.mean([m.pressure_proxy for m in metrics])),
        "shot_rate": 100.0 * sum(m.shots > 0 for m in metrics) / n,
        "scoring_rate": 100.0 * sum(m.goals > 0 for m in metrics) / n,
    }
