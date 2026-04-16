"""Base class for composable reward components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class RewardComponent(ABC):
    """A single, composable reward signal.

    Subclasses implement ``compute`` to return a scalar reward given the
    transition tuple.  Instances carry a *weight* so they can be linearly
    combined by the :class:`~opengoalrl.wrappers.reward_wrapper.RewardWrapper`.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @abstractmethod
    def compute(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        """Return a scalar reward for the given transition."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"
