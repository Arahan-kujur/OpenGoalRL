"""Shared environment construction for training and evaluation scripts."""

from __future__ import annotations

import importlib
import random
from typing import Any

import numpy as np
import torch

from opengoalrl.utils.config_loader import build_reward_components
from opengoalrl.wrappers.action_wrapper import ActionWrapper
from opengoalrl.wrappers.observation_wrapper import ObservationWrapper
from opengoalrl.wrappers.reward_wrapper import RewardWrapper
from opengoalrl.wrappers.scenario_wrapper import ScenarioWrapper

EnvClass = type
EnvRegistryEntry = str | EnvClass

ENV_REGISTRY: dict[str, EnvRegistryEntry] = {
    "corner_kick": "opengoalrl.envs.corner_kick:CornerKickEnv",
    "penalty": "opengoalrl.envs.penalty:PenaltyEnv",
    "empty_goal_close": "opengoalrl.envs.empty_goal_close:EmptyGoalCloseEnv",
    "empty_goal": "opengoalrl.envs.empty_goal:EmptyGoalEnv",
    "run_to_score": "opengoalrl.envs.run_to_score:RunToScoreEnv",
    "pass_and_shoot": "opengoalrl.envs.pass_and_shoot:PassAndShootEnv",
    "three_vs_one": "opengoalrl.envs.three_vs_one:ThreeVsOneEnv",
}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs used by OpenGoalRL scripts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_base_env(env_cfg: dict[str, Any]):
    """Instantiate base env from ``scenario`` or ``scenario_spec``."""
    if "scenario_spec" in env_cfg:
        from opengoalrl.envs.generated_scenario import GeneratedScenarioEnv

        spec = env_cfg["scenario_spec"]
        scenario_name = (
            spec.get("name", "generated") if isinstance(spec, dict) else spec.name
        )
        return GeneratedScenarioEnv(
            spec=spec,
            max_steps=env_cfg.get("max_steps", 400),
            render_mode="human" if env_cfg.get("render", False) else None,
        ), scenario_name

    scenario = env_cfg["scenario"]
    env_cls = _resolve_env_class(scenario)
    render_mode = "human" if env_cfg.get("render", False) else None
    return env_cls(
        max_steps=env_cfg.get("max_steps", 400),
        render_mode=render_mode,
    ), scenario


def build_env(config: dict[str, Any]):
    """Instantiate a single env + full wrapper stack from *config*.

    Wrapper order is intentionally fixed:
    BaseEnv -> ScenarioWrapper -> RewardWrapper -> ObservationWrapper -> ActionWrapper.
    """
    env_cfg = config["environment"]
    env, scenario = _resolve_base_env(env_cfg)

    env = ScenarioWrapper(env, scenario_name=scenario)

    reward_components = build_reward_components(config)
    if reward_components:
        env = RewardWrapper(env, components=reward_components)

    env = ObservationWrapper(env)
    env = ActionWrapper(env)
    return env


def build_vec_env(config: dict[str, Any]):
    """Build a vectorised environment from *config*."""
    n_envs = config.get("training", {}).get("n_envs", 1)
    if n_envs <= 1:
        return build_env(config)

    def _make_env(rank: int):
        def _init():
            return build_env(config)
        return _init

    from stable_baselines3.common.vec_env import SubprocVecEnv

    return SubprocVecEnv([_make_env(i) for i in range(n_envs)])


def _resolve_env_class(scenario: str) -> EnvClass:
    entry = ENV_REGISTRY.get(scenario)
    if entry is None:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available: {list(ENV_REGISTRY)}"
        )
    if isinstance(entry, str):
        module_name, class_name = entry.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    return entry
