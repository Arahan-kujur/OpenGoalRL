"""Thin wrapper around stable-baselines3 PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class PPOAgent:
    """Configurable PPO agent backed by stable-baselines3.

    Parameters
    ----------
    env:
        A Gymnasium-compatible environment (typically after the full wrapper
        stack has been applied).
    config:
        Dict of PPO hyper-parameters.  Recognised keys mirror SB3's
        constructor: ``learning_rate``, ``n_steps``, ``batch_size``,
        ``n_epochs``, ``gamma``, ``clip_range``, ``seed``.
    """

    def __init__(self, env: gym.Env, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            clip_range=config.get("clip_range", 0.2),
            seed=config.get("seed"),
            verbose=1,
        )

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
    ) -> None:
        """Run PPO training for ``total_timesteps``."""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(
        self, obs: np.ndarray, deterministic: bool = True,
    ) -> tuple[int, Any]:
        """Return action and internal state for a single observation."""
        action, state = self.model.predict(obs, deterministic=deterministic)
        return int(action), state

    def save(self, path: str | Path) -> None:
        """Persist the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env: gym.Env) -> "PPOAgent":
        """Load a saved model and attach it to *env*."""
        agent = cls.__new__(cls)
        agent.model = PPO.load(str(path), env=env)
        return agent
