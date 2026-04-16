"""Wrapper that restricts or remaps the action space."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np


class ActionWrapper(gym.ActionWrapper):
    """Restrict the discrete action space to a scenario-relevant subset.

    If no ``allowed_actions`` are provided the wrapper acts as a pass-through
    (all actions remain available).

    Parameters
    ----------
    env:
        The inner environment.
    allowed_actions:
        List of GRF action indices the agent is allowed to use.  The agent
        selects an index into this list and the wrapper maps it back to the
        original GRF action.
    """

    def __init__(
        self,
        env: gym.Env,
        allowed_actions: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(env)
        if allowed_actions is not None:
            self._action_map = list(allowed_actions)
            self.action_space = gym.spaces.Discrete(len(self._action_map))
        else:
            self._action_map = None

    def action(self, action: int) -> int:
        if self._action_map is not None:
            return self._action_map[action]
        return action
