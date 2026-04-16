"""Reward for taking a shot action."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengoalrl.rewards.base_reward import RewardComponent

# GRF discrete action 12 = "Shot"
_SHOT_ACTION = 12


class ShotReward(RewardComponent):
    """Returns +1 when the agent takes a shot."""

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        return 1.0 if action == _SHOT_ACTION else 0.0
