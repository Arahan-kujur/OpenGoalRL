"""Continuous reward based on ball proximity to the opponent goal."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengoalrl.rewards.base_reward import RewardComponent

# Opponent goal centre in GRF's coordinate system (simple115v2).
_GOAL_X = 1.0
_GOAL_Y = 0.0

_BALL_X_IDX = 88
_BALL_Y_IDX = 89

# Maximum possible distance on the pitch (corner to corner ≈ 2.2).
_MAX_DIST = 2.2


class DistanceToGoalReward(RewardComponent):
    """Reward that increases as the ball gets closer to the opponent goal.

    Normalised to [0, 1] so it blends well with other reward components.
    """

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        bx = float(next_obs[_BALL_X_IDX])
        by = float(next_obs[_BALL_Y_IDX])
        dist = np.sqrt((bx - _GOAL_X) ** 2 + (by - _GOAL_Y) ** 2)
        return float(1.0 - dist / _MAX_DIST)
