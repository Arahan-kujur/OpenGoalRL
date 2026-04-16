"""Wrapper that normalises and optionally flattens observations."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):
    """Normalise observations to [0, 1] and optionally flatten them.

    Parameters
    ----------
    env:
        The inner environment.
    normalize:
        If ``True`` (default), apply min-max normalisation using running
        estimates.
    flatten:
        If ``True``, flatten the observation to a 1-D array (default
        ``False``; simple115v2 is already flat).
    clip_range:
        Clip raw observations to ``[-clip_range, clip_range]`` before
        normalising.  Prevents outlier values from dominating the range.
    """

    def __init__(
        self,
        env: gym.Env,
        normalize: bool = True,
        flatten: bool = False,
        clip_range: float = 3.0,
    ) -> None:
        super().__init__(env)
        self._normalize = normalize
        self._flatten = flatten
        self._clip_range = clip_range

        base_shape = self.observation_space.shape
        if flatten:
            flat_dim = int(np.prod(base_shape))
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(flat_dim,), dtype=np.float32,
            )
        elif normalize:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=base_shape, dtype=np.float32,
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self._flatten:
            obs = obs.flatten()
        if self._normalize:
            obs = np.clip(obs, -self._clip_range, self._clip_range)
            obs = (obs + self._clip_range) / (2.0 * self._clip_range)
        return obs
