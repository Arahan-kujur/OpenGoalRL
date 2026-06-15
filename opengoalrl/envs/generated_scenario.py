"""Environment backed by a structured :class:`ScenarioSpec`."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from opengoalrl.envs.base_env import BaseScenarioEnv
from opengoalrl.scenarios.generator import generate_scenario
from opengoalrl.scenarios.spec import OPENGOAL_SCENARIO_BY_GRF, ScenarioSpec

_BALL_X_IDX = 88
_CLEAR_X_THRESHOLD = 0.8


class GeneratedScenarioEnv(BaseScenarioEnv):
    """GRF env selected from a :class:`ScenarioSpec` with metadata in ``info``."""

    def __init__(
        self,
        spec: ScenarioSpec | dict[str, Any],
        max_steps: int | None = None,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.spec = generate_scenario(spec) if isinstance(spec, dict) else spec
        steps = max_steps if max_steps is not None else self.spec.max_steps
        grf_name = self.spec.grf_scenario or "academy_empty_goal_close"
        super().__init__(
            scenario_name=grf_name,
            max_steps=steps,
            render_mode=render_mode,
            **kwargs,
        )
        self.scenario_name = OPENGOAL_SCENARIO_BY_GRF.get(
            grf_name, self.spec.name,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["scenario_spec"] = self.spec.to_dict()
        info["scenario"] = self.spec.name
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        info["scenario_spec"] = self.spec.to_dict()
        info["scenario"] = self.spec.name
        if self._ball_retreated(obs):
            info["possession_lost"] = True
        return obs, reward, terminated, truncated, info

    def _is_ball_cleared(self, obs: np.ndarray) -> bool:
        if self.spec.objective == "score":
            return float(obs[_BALL_X_IDX]) < _CLEAR_X_THRESHOLD
        return False

    @staticmethod
    def _ball_retreated(obs: np.ndarray) -> bool:
        return float(obs[_BALL_X_IDX]) < -0.2
