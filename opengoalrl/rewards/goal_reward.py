"""Reward for scoring (or conceding) a goal."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengoalrl.rewards.base_reward import RewardComponent


class GoalReward(RewardComponent):
    """Large positive reward on goal scored, negative on goal conceded.

    Relies on ``info["score_reward"]`` provided by the GRF environment which
    is +1 when the controlled team scores and -1 when they concede.
    """

    def __init__(self, weight: float = 1.0, concede_penalty: float = -1.0) -> None:
        super().__init__(weight)
        self.concede_penalty = concede_penalty

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        score = info.get("score_reward", 0.0)
        if score > 0:
            return 1.0
        if score < 0:
            return self.concede_penalty
        return 0.0
